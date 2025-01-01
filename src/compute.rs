//! This module abstracts how to compute values out of nodes.

use crate::core_syntax::{ComputValue, Expr, ExprBuilder, Ident, Node, Operator, VariableNameId};
use std::{cell::RefCell, collections::BTreeMap};

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
    ast: RefCell<BTreeMap<Ident, Node2<F, OP1, OP2>>>,
    calculator: &'a dyn Calculator<OP1, OP2, F>,
}

//remove /// Holds all the numeric data related to a node in computation graph.
//remove /// The node types are:
//remove /// - Variable - explicitly set by the user, an input variable. The variables are reset often.
//remove /// - Parameter - parameter of the model, that is updated during back-propagation. The parameters
//remove ///   are initialized once (to random values) and then updated at the end of each training epoch.
//remove #[derive(Clone, Debug)]
//remove enum NodeData<F> {
//remove     /// The node holds a variable, like input variable. The variable is reset for every input.
//remove     Variable {
//remove         /// Primal (result of "forward").
//remove         primal: Option<F>,
//remove         /// Accumulated adjoin (result of "backward"). The u32 is the count of how many "add adjoin" was run on this adjoin.
//remove         /// This count is needed to divide adjoin when applying learn rate.
//remove         adjoin: Option<(F, u32)>,
//remove     },
//remove     /// The node holds model parameters. The node parameters are not reset, they are modified on each epoch.
//remove     Parameter { primal: F, adjoin: Option<(F, u32)> },
//remove     /// The node type was not yet set. Once set, the node type cannot be changed.
//remove     Unset,
//remove }

#[derive(Debug, Clone)]
enum Node2<F, OP1, OP2>
where
    F: ComputValue,
    OP1: Operator,
    OP2: Operator,
{
    Const(F),
    Variable {
        name: String,
        tmp_name_id: VariableNameId, // TODO remove, compat only.
        tensors: Tensors<F>,
    },
    Parameter {
        name: Option<String>,
        tensors: Tensors<F>,
    },
    Ary1 {
        oper: OP1,
        arg1: Ident,
        tensors: Tensors<F>,
    },
    Ary2 {
        oper: OP2,
        arg1: Ident,
        arg2: Ident,
        tensors: Tensors<F>,
    },
}

#[derive(Debug, Clone)]
struct Tensors<F>
where
    F: ComputValue,
{
    primal: Option<F>,
    adjoin: Option<(F, u32)>,
}

impl<F> Default for Tensors<F>
where
    F: ComputValue,
{
    fn default() -> Self {
        Self {
            primal: Default::default(),
            adjoin: Default::default(),
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
        let mut ast: BTreeMap<Ident, Node2<F2, OP1, OP2>> = BTreeMap::new();
        // Translate the expression tree constructed by the user to the one used internally with state-per-node.
        for (ident, expr_node) in eb.id_to_node.borrow().iter() {
            let tensors = Tensors::default();
            let new_node: Node2<F2, OP1, OP2> = match expr_node {
                Node::Const(value) => Node2::Const(value.clone()),
                Node::Variable(name_id) => Node2::Variable {
                    name: eb
                        .get_name(name_id)
                        .expect("variable should have name but did not!"),
                    tmp_name_id: name_id.clone(),
                    tensors,
                },
                Node::Ary1(oper, arg1) => Node2::Ary1 {
                    oper: *oper,
                    arg1: *arg1,
                    tensors,
                },
                Node::Ary2(oper, arg1, arg2) => Node2::Ary2 {
                    oper: *oper,
                    arg1: *arg1,
                    arg2: *arg2,
                    tensors,
                },
            };
            if let Some(_) = ast.insert(*ident, new_node) {
                panic!("Bug. ast node already set.")
            }
        }

        ComputGraph {
            ast: RefCell::new(ast),
            calculator,
        }
    }

    /// Set variable once, panic if the variable was already set.
    // TODO: Consider changing value: F to Into<F>
    pub fn set_variable(&mut self, ident: &dyn AsRef<Ident>, value: F) {
        let ident = ident.as_ref();
        if let Some(old) = self.reset_primal_of_variable(ident, value) {
            panic!(
                "Value for {:?} {} already set to {}",
                self.get_name(ident),
                ident.as_ref(),
                old
            );
        }
    }

    /// Set parameter values. Fail if the value is already set.
    pub fn set_parameter(&mut self, ident: &dyn AsRef<Ident>, value: F) {
        let ident = ident.as_ref().clone();
        let mut ast = self.ast.borrow_mut();
        if let Node2::Parameter { name, tensors } = ast.get_mut(&ident).expect("") {
            if let Some(_) = tensors.primal.replace(value) {
                panic!("Parameter {:?} already has primal set!", name);
            };
            tensors.adjoin.take();
        } else {
            panic!("Node is not Parameter!")
        }
    }

    /// Set variable (primal) to some value. Do not fail if the variable is already set. Useful when
    /// running `.backward()` in a loop for different input variables.
    /// Return old variable primal.
    pub fn reset_primal_of_variable(&mut self, ident: &dyn AsRef<Ident>, value: F) -> Option<F> {
        let ident = ident.as_ref();
        let mut ast = self.ast.borrow_mut();
        let node = ast.get_mut(&ident).unwrap();
        if let Node2::Variable { tensors, .. } = node {
            tensors.primal.replace(value)
        } else {
            panic!("Node is not a Variable!")
        }
    }

    pub fn get_node(&self, ident: &Ident) -> Node<F, OP1, OP2> {
        //let id_to_node = self.eb.id_to_node.borrow();
        //let node = id_to_node
        //    .get(ident)
        //    .expect(format!("No node for ident {}", ident).as_str());
        //node.clone()
        let ast = self.ast.borrow();
        let node = ast
            .get(ident)
            .expect(format!("No node for ident {}", ident).as_str());
        // Temporary compatibility code
        match node {
            Node2::Const(value) => Node::Const(value.clone()),
            Node2::Variable { tmp_name_id, .. } => Node::Variable(*tmp_name_id),
            Node2::Parameter { .. } => panic!(),
            Node2::Ary1 { oper, arg1, .. } => Node::Ary1(*oper, *arg1),
            Node2::Ary2 {
                oper, arg1, arg2, ..
            } => Node::Ary2(*oper, *arg1, *arg2),
        }
    }

    pub fn get_name(&self, ident: &Ident) -> Option<String> {
        if let Node2::Variable { name, .. } = self.get_node2(ident) {
            Some(name)
        } else {
            None
        }
    }

    fn get_node2(&self, ident: &Ident) -> Node2<F, OP1, OP2> {
        let ast = self.ast.borrow();
        ast.get(ident)
            .expect(format!("No node for ident {}", ident).as_str())
            .clone()
    }

    /// Reset primals for variables. Keep adjoins, and primals for Parameters.
    pub fn reset_state_for_next_input(&mut self) {
        {
            let mut ast = self.ast.borrow_mut();
            for (_, node) in ast.iter_mut() {
                match node {
                    Node2::Const(_) => (),
                    Node2::Variable { tensors, .. } => {
                        tensors.primal.take();
                    }
                    Node2::Parameter { .. } => (),
                    Node2::Ary1 { tensors, .. } => {
                        tensors.primal.take();
                    }
                    Node2::Ary2 { tensors, .. } => {
                        tensors.primal.take();
                    }
                }
            }
        }
    }

    /// Reset the internal state (variable primals, adjoins). Do not clean parameters.
    pub fn reset_state_for_next_epoch(&mut self) {
        {
            let mut ast = self.ast.borrow_mut();
            for (_ident, node) in ast.iter_mut() {
                match node {
                    Node2::Const(_) => (),
                    Node2::Variable { tensors, .. } => {
                        *tensors = Tensors::default();
                    }
                    Node2::Parameter { tensors, .. } => {
                        tensors.adjoin.take();
                    }
                    Node2::Ary1 { tensors, .. } => {
                        *tensors = Tensors::default();
                    }
                    Node2::Ary2 { tensors, .. } => {
                        *tensors = Tensors::default();
                    }
                }
            }
        }
    }

    pub fn update_params_lr(&mut self, learning_rate: f32) {
        let mut data = self.data.borrow_mut();
        let param_idents: Vec<Ident> = data
            .iter()
            .filter_map(|(ident, node_data)| {
                if let NodeData::Parameter { .. } = node_data {
                    Some(ident.clone())
                } else {
                    None
                }
            })
            .collect();
        for ident in param_idents {
            let node_data = data.get_mut(&ident).unwrap();
            let (primal, (adjoin, adjoin_update_cnt)) =
                if let NodeData::Parameter { primal, adjoin } = node_data {
                    if let Some(adjoin) = adjoin.as_ref() {
                        (primal, adjoin)
                    } else {
                        panic!("Adjoin missing for parameter!")
                    }
                } else {
                    panic!("Expected Parameter!")
                };
            // -1.0 because Add and Mul is implemented but Sub not necessarily.
            let new_primal = primal.clone()
                + adjoin.clone() * -1.0 * (learning_rate / *adjoin_update_cnt as f32);
            *primal = new_primal;
        }
    }

    /// Forward pass, calculate primals.
    /// This operation is MUTABLE, i.e. it mutates the internal cache of the calculated values.
    pub fn forward(&self, ident: &dyn AsRef<Ident>) -> F {
        let ident = ident.as_ref();
        {
            let mut data = self.data.borrow_mut();
            let node_data = data
                .get(&ident)
                .expect("Bug: node data is missing in forward()!");

            let new_node_data = match node_data {
                NodeData::Variable { primal, .. } => match primal {
                    Some(primal) => return primal.clone(),
                    None => None,
                },

                NodeData::Parameter { primal, .. } => return primal.clone(),
                // If you run .forward() on a node, and the node is Unset, then assume the node is a Variable (e.g. some final "y" in y=ax+b).
                NodeData::Unset => Some(NodeData::Variable {
                    primal: Option::<F>::None,
                    adjoin: None,
                }),
            };
            if let Some(new_node_data) = new_node_data {
                data.insert(ident.clone(), new_node_data).unwrap();
            }
        }

        let calculated_primal = self.calculator.forward(self, ident);

        {
            let mut data = self.data.borrow_mut();
            let node_data = data
                .get_mut(&ident)
                .expect("Bug: node data is missing in forward()!");
            let new_data: NodeData<F> = match node_data {
                NodeData::Variable { primal, adjoin } => match primal {
                    Some(old) => panic!("The value for {} already set to {}", ident, old),
                    None => NodeData::Variable { primal: Some(calculated_primal.clone()), adjoin: adjoin.clone() }
                    ,
                },
                NodeData::Parameter {..} => panic!("The node is Parameter but expected Variable!"),
                NodeData::Unset => panic!("The node is Unset during forward, but it should be already a variable or parameter!"),
            };
            *node_data = new_data;
        }
        calculated_primal
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

    /// Call `add_adjoin` to update adjoin for a node with partial adjoin.
    pub fn add_adjoin(&self, ident: &Ident, adjoin: &F) {
        // TODO try with mut self?
        let mut data = self.data.borrow_mut();
        let node_data = data.get_mut(ident).expect("Bug! Node data missing");
        let maybe_old_adjoin : &mut Option<(F, u32)> = match node_data {
            NodeData::Variable { adjoin , ..} => adjoin,
            NodeData::Parameter {  adjoin , ..} => adjoin,
            NodeData::Unset => panic!("The node is Unset during add_adjoin, but it should be already a variable or parameter!"),
        };
        let updated_adjoin: (F, u32) = if let Some((old_adjoin, old_cnt)) = maybe_old_adjoin {
            (old_adjoin.clone() + adjoin.clone(), *old_cnt + 1)
        } else {
            (adjoin.clone(), 1)
        };
        *maybe_old_adjoin = Some(updated_adjoin)
    }

    pub fn primal(&self, ident: &Ident) -> F {
        let data = self.data.borrow();
        let node_data = data.get(ident).expect("Bug! Node data missing");
        match node_data {
            NodeData::Variable { primal, .. } => if let Some(primal) = primal {
                primal.clone()
            } else {
                panic!("Primal missing for {}", &ident)
            },
            NodeData::Parameter { primal, .. } => primal.clone(),
            NodeData::Unset => panic!("The node is Unset during .primal(), but it should be already a variable or parameter!"),
        }
    }

    pub fn adjoin(&self, ident: &Ident) -> F {
        let data = self.data.borrow();
        let node_data = data.get(ident).expect("Bug: node data missing!");
        let maybe_adjoin = match node_data {
            NodeData::Variable {  adjoin , ..} => adjoin,
            NodeData::Parameter {  adjoin , ..} => adjoin,
            NodeData::Unset => panic!("The node is Unset during .adjoin(), but it should be already a variable or parameter!"),
        };
        match maybe_adjoin {
            Some((adjoin, _cnt)) => adjoin.clone(),
            None => panic!(
                "Adjoin missing for {}, maybe you didn't run backward?",
                ident
            ),
        }
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
