//! This module abstracts how to compute values out of nodes.

use std::{cell::RefCell, collections::BTreeMap};

use crate::core_syntax::{ComputValue, Expr, ExprBuilder, Ident, Node, Operator, VariableNameId};

impl AsRef<Ident> for Ident {
    fn as_ref(&self) -> &Ident {
        return self;
    }
}

impl<'a, F, OP1, OP2> AsRef<Ident> for Expr<'a, F, OP1, OP2>
where
    F: ComputValue,
    OP1: Operator,
    OP2: Operator,
{
    fn as_ref(&self) -> &Ident {
        &self.ident
    }
}

/// "Freezes" the expression tree by taking ownership of the expression builder.
/// All the methods use `&self` instead of `&mut self` because the underlying code heavily relies on `RefCell` anyway.
pub struct ComputGraph<'a, F, OP1, OP2>
where
    F: ComputValue,
    OP1: Operator,
    OP2: Operator,
{
    /// Variable values that are saved and restored upon reset.
    saved_variables: RefCell<BTreeMap<Ident, F>>,
    /// Parameters. The parameters are updated during training based on adjoins and learning rate.
    params: RefCell<BTreeMap<Ident, F>>,
    data: RefCell<BTreeMap<Ident, NodeData<F>>>,
    eb: ExprBuilder<F, OP1, OP2>,
    calculator: &'a dyn Calculator<OP1, OP2, F>,
}

/// [Data] holds all the data related to particular [Node].
struct NodeData<F> {
    /// Primal (result of "forward").
    primal: Option<F>,
    // Accumulated adjoin (result of "backward").
    adjoin: Option<F>,
}

impl<F> Default for NodeData<F> {
    fn default() -> Self {
        Self {
            primal: None,
            adjoin: None,
        }
    }
}

impl<'a, F, OP1, OP2> ComputGraph<'a, F, OP1, OP2>
where
    F: ComputValue,
    OP1: Operator,
    OP2: Operator,
{
    /// Take ownership of the expression builder, because it "freezes" the expression. The expression, as represented
    /// by the internal data structures of the expression builder, cannot change from now on.
    pub fn new<F2: ComputValue>(
        eb: ExprBuilder<F2, OP1, OP2>,
        calculator: &'a dyn Calculator<OP1, OP2, F2>,
    ) -> ComputGraph<'a, F2, OP1, OP2> {
        let mut data: BTreeMap<Ident, NodeData<F2>> = BTreeMap::new();
        for (ident, _) in eb.id_to_node.borrow().iter() {
            data.insert(ident.clone(), NodeData::default());
        }
        ComputGraph {
            saved_variables: RefCell::new(BTreeMap::new()),
            params: RefCell::new(BTreeMap::new()),
            data: RefCell::new(data),
            eb,
            calculator,
        }
    }

    /// Set variable once, panic if the variable was already set.
    // TODO: Consider changing value: F to Into<F>
    pub fn set_variable(&mut self, ident: &dyn AsRef<Ident>, value: F) {
        let ident = ident.as_ref();
        if let Some(old) = self.reset_variable(ident, value) {
            panic!(
                "Value for {:?} {} already set to {}",
                self.get_name(ident),
                ident.as_ref(),
                old
            );
        }
    }

    pub fn set_parameter(&mut self, ident: &dyn AsRef<Ident>, value: F) {
        let ident = ident.as_ref();
        self.assert_ident_is_variable(ident);
        self.save_variable(ident, value.clone());
        let old = self
            .params
            .borrow_mut()
            .insert(ident.clone(), value.clone());
        if let Some(old) = old {
            panic!(
                "Value for {:?} {} already set to {}",
                self.get_name(ident),
                ident.as_ref(),
                old
            )
        }
    }

    /// Set variable (primal) to some value. Do not fail if the variable is already set. Useful when
    /// running `.backward()` in a loop for different input variables.
    pub fn reset_variable(&mut self, ident: &dyn AsRef<Ident>, value: F) -> Option<F> {
        let ident = ident.as_ref();
        self.assert_ident_is_variable(ident);
        self.save_variable(ident, value.clone());

        let mut data = self.data.borrow_mut();
        if let Some(p) = data.get_mut(&ident) {
            p.primal.replace(value)
        } else {
            data.insert(
                ident.clone(),
                NodeData {
                    primal: Some(value),
                    ..Default::default()
                },
            );
            None
        }
    }

    pub fn get_node(&self, ident: &Ident) -> Node<F, OP1, OP2> {
        let id_to_node = self.eb.id_to_node.borrow();
        let node = id_to_node
            .get(ident)
            .expect(format!("No node for ident {}", ident).as_str());
        node.clone()
    }

    pub fn get_name(&self, ident: &Ident) -> Option<String> {
        let name_id = VariableNameId::from(*ident);
        self.eb.get_name(&name_id)
    }

    /// Remove primals, keep the variables values so the user only needs to
    /// call [reset_variable][ComputGraph::reset_variable] on some variables, and not all of them.
    pub fn reset_state_for_next_input(&mut self) {
        {
            let mut data = self.data.borrow_mut();
            for (_, node) in data.iter_mut() {
                node.primal = None
            }
        }
        self.refill_primals_that_were_explicitly_set();
    }

    /// Reset the internal state (primals, adjoins). Do not clean parameters.
    pub fn reset_state_for_next_epoch(&mut self) {
        {
            let mut data = self.data.borrow_mut();
            for (_, node) in data.iter_mut() {
                *node = NodeData::default();
            }
        }
        self.saved_variables = RefCell::new(BTreeMap::new());
        self.restore_parameters()
    }

    fn restore_parameters(&mut self) {
        let params_vec: Vec<(Ident, F)>;
        {
            let params = self.params.borrow();
            params_vec = params
                .iter()
                .map(|(ident, value)| (ident.clone(), value.clone()))
                .collect();
        }
        for (ident, value) in params_vec {
            self.set_variable(&ident, value);
        }
    }

    pub fn update_params_lr(&mut self, learning_rate: f32) {
        let mut params = self.params.borrow_mut();
        let param_idents: Vec<Ident> = params.keys().map(|v| v.clone()).collect();
        for ident in param_idents {
            let p = params.get_mut(&ident).unwrap();
            let ad = self.adjoin(&ident);
            // -1.0 because Add and Mul is implemented but Sub not necessarily.
            *p = p.clone() + ad * -1.0 * learning_rate;
        }
    }

    fn save_variable(&mut self, ident: &Ident, value: F) {
        let mut saved_variables = self.saved_variables.borrow_mut();
        saved_variables.insert(ident.clone(), value);
    }

    fn refill_primals_that_were_explicitly_set(&mut self) {
        let saved_variables_vec: Vec<(Ident, F)>;
        {
            let saved_variables = self.saved_variables.borrow();
            saved_variables_vec = saved_variables
                .iter()
                .map(|(ident, value)| (ident.clone(), value.clone()))
                .collect();
        }
        for (ident, value) in saved_variables_vec {
            self.reset_variable(&ident, value);
        }
    }

    /// Forward pass, calculate primals.
    /// This operation is MUTABLE, i.e. it mutates the internal cache of the calculated values.
    pub fn forward(&self, ident: &dyn AsRef<Ident>) -> F {
        let ident = ident.as_ref();
        {
            let data = self.data.borrow();
            let node_data = data
                .get(&ident)
                .expect("Bug: node data is missing in forward()!");
            if let Some(ref primal) = node_data.primal {
                return primal.clone();
            }
        }

        let primal = self.calculator.forward(self, ident);

        {
            let mut data = self.data.borrow_mut();
            let node_data = data
                .get_mut(&ident)
                .expect("Bug: node data is missing in forward()!");
            if let Some(old) = node_data.primal.replace(primal.clone()) {
                panic!("The value for {} already set to {}", ident, old);
            }
        }
        primal
    }

    /// Implement reverse mode of automatic gradient. The procedure is as follows:
    /// 1. Each Node has child nodes, e.g. for node Y that is X1+X2, the X nodes are children.
    ///    For a Y node, consider the contribution of Y to each of its children X.
    /// 2. Calculate the contribution of Y to X as X_. X_ is a partial adjoin that is accumulated
    ///    among all the inputs to X. X_ = Y_ * dY/dX.
    /// 3. Re-run the procedure with each X node.
    /// The function just calculates the values but does not return adjoin, since the adjoin values the user is interested in
    /// (leaf X nodes) is for other nodes that the backward pass is run for (Y).
    pub fn backward(&self, ident: &dyn AsRef<Ident>) {
        let ident = ident.as_ref();
        let adjoin = F::default_adjoin(self.forward(ident));
        self.calculator.backward(self, ident, &adjoin);
    }

    fn assert_ident_is_variable(&self, ident: &Ident) {
        let id_to_node = self.eb.id_to_node.borrow();
        let node = id_to_node.get(ident).expect("No such ident");
        if let Node::Variable(_) = node {
        } else {
            panic!("Not a variable {}: {:?}", ident, node)
        }
    }

    pub fn get_variable_name(&self, name_id: &VariableNameId) -> String {
        self.eb.get_name(name_id).expect("no name for such ident")
    }

    /// Call `add_adjoin` to update adjoin for a node with partial adjoin.
    pub fn add_adjoin(&self, ident: &Ident, adjoin: &F) {
        // TODO try with mut self?
        let mut data = self.data.borrow_mut();
        let node_data = data.get_mut(ident).expect("Bug! Node data missing");
        let updated_adjoin = node_data
            .adjoin
            .as_ref()
            .map_or(adjoin.clone(), |old| old.clone() + adjoin.clone());
        node_data.adjoin.replace(updated_adjoin);
    }

    pub fn primal(&self, ident: &Ident) -> F {
        let data = self.data.borrow();
        let node_data = data.get(ident).expect("Bug! Node data missing");
        if let Some(ref primal) = node_data.primal {
            primal.clone()
            // Cloning full matrices (e.g. for nalgebra) would be impractically inefficient. Use Box or Rc.
        } else {
            panic!("Primal missing for {}", &ident)
        }
    }

    pub fn adjoin(&self, ident: &Ident) -> F {
        let data = self.data.borrow();
        let node_data = data.get(ident).expect("Bug: node data missing!");
        node_data
            .adjoin
            .as_ref()
            .expect(
                format!(
                    "Adjoin missing for {}, maybe you didn't run backward?",
                    ident
                )
                .as_str(),
            )
            .clone()
    }
}

/// Take node and return a calculated value.
pub trait Calculator<OP1, OP2, F>
where
    F: ComputValue,
    OP1: Operator,
    OP2: Operator,
{
    /// Take node and return a computed value for it. The passed computational graph allows querying for the
    /// already computed values. The querying for the values can run actual computation, that can be later
    /// cached in the graph.
    ///
    /// Usually, `calculate_primal` should not be called for `Node::Variable` since all the variables should
    /// be set beforehand with `set_variable`. It's ok to `panic` on Node::Variable.
    fn forward(&self, cg: &ComputGraph<F, OP1, OP2>, ident: &Ident) -> F;

    /// The implementation of the backward pass is responsible for updating partial adjoins though ComputGraph.
    fn backward(&self, cg: &ComputGraph<F, OP1, OP2>, ident: &Ident, adjoin: &F);
}

#[cfg(test)]
mod tests {
    use super::ComputGraph;
    use crate::{
        core_syntax::ExprBuilder,
        float::{
            calculator::FloatCalculator,
            syntax::{FloatOperAry1, FloatOperAry2},
        },
    };

    #[test]
    fn compute_primal() {
        let eb = new_eb();
        let x1 = eb.new_variable("x1");
        let x2 = eb.new_variable("x2");
        let x3 = x1 + x2;
        let x4 = x1 + x2;
        let e = (x1 + x2) * x3 + x4;

        let x1 = x1.ident();
        let x2 = x2.ident();
        let e = e.ident();
        let mut cg = ComputGraph::<f32, FloatOperAry1, FloatOperAry2>::new(eb, &FloatCalculator);
        cg.set_variable(&x1, 3.0);
        cg.set_variable(&x2, 5.0);
        let p = cg.forward(&e);
        assert_eq!(p, (3.0 + 5.0) * (3.0 + 5.0) + (3.0 + 5.0));
    }

    #[test]
    fn compute_simple_tangent() {
        let eb = new_eb();
        let x1 = eb.new_variable("x1");
        let x2 = eb.new_variable("x2");
        let y = x1 * x2;

        let x1 = x1.ident();
        let x2 = x2.ident();
        let y = y.ident();
        let mut cg = ComputGraph::<f32, FloatOperAry1, FloatOperAry2>::new(eb, &FloatCalculator);
        cg.set_variable(&x1, 3.0);
        cg.set_variable(&x2, -4.0);
        cg.forward(&y);
        cg.backward(&y);
        assert_eq!(cg.adjoin(&x1), -4.0); // not sure if this is ok
    }

    fn new_eb() -> ExprBuilder<f32, FloatOperAry1, FloatOperAry2> {
        ExprBuilder::<f32, FloatOperAry1, FloatOperAry2>::new()
    }
}
