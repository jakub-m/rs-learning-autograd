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
    data: RefCell<BTreeMap<Ident, NodeData<F>>>,
    eb: ExprBuilder<F, OP1, OP2>,
    calculator: &'a dyn Calculator<OP1, OP2, F>,
}

/// Holds all the numeric data related to a node in computation graph.
/// The node types are:
/// - Variable - explicitly set by the user, an input variable. The variables are reset often.
/// - Parameter - parameter of the model, that is updated during back-propagation. The parameters
///   are initialized once (to random values) and then updated at the end of each training epoch.
#[derive(Clone, Debug)]
enum NodeData<F> {
    /// The node holds a variable, like input variable. The variable is reset for every input.
    Variable {
        /// Primal (result of "forward").
        primal: Option<F>,
        // Accumulated adjoin (result of "backward").
        adjoin: Option<F>,
    },
    /// The node holds model parameters. The node parameters are not reset, they are modified on each epoch.
    Parameter { primal: F, adjoin: Option<F> },
    /// The node type was not yet set. Once set, the node type cannot be changed.
    Unset,
}

//impl<F> Default for NodeData<F> {
//    fn default() -> Self {
//        todo!(); // remove, not needed
//        NodeData::Unset
//    }
//}

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
            data.insert(ident.clone(), NodeData::Unset);
        }
        ComputGraph {
            saved_variables: RefCell::new(BTreeMap::new()),
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

    /// Fail if parameter already set.
    pub fn set_parameter(&mut self, ident: &dyn AsRef<Ident>, value: F) {
        let ident = ident.as_ref().clone();
        if let Some(old) = self.reset_parameter(&ident, value) {
            if let NodeData::Unset = old {
            } else {
                panic!(
                    "Parameter {:?} {} already set to {:?}",
                    self.get_name(&ident),
                    &ident,
                    &old
                );
            }
        }
    }

    /// Set Parameter but don't fail if the node was already set. For parameters, it's rather a low-level functionality
    /// and `set_parameter` should be used instead.
    fn reset_parameter(&mut self, ident: &dyn AsRef<Ident>, value: F) -> Option<NodeData<F>> {
        let ident = ident.as_ref().clone();
        self.assert_ident_is_variable(&ident);
        self.save_variable(&ident, value.clone());

        let mut data = self.data.borrow_mut();
        data.insert(
            ident,
            NodeData::Parameter {
                primal: value,
                adjoin: None,
            },
        )
    }

    /// Set variable (primal) to some value. Do not fail if the variable is already set. Useful when
    /// running `.backward()` in a loop for different input variables.
    /// Return old variable value.
    pub fn reset_variable(&mut self, ident: &dyn AsRef<Ident>, value: F) -> Option<F> {
        let ident = ident.as_ref();
        self.assert_ident_is_variable(ident);
        self.save_variable(ident, value.clone());

        let mut data = self.data.borrow_mut();
        if let Some(node_data) = data.get_mut(&ident) {
            let (old_primal, new_data) = match node_data {
                NodeData::Variable { primal, adjoin } => (
                    primal.clone(),
                    NodeData::Variable {
                        primal: Some(value),
                        adjoin: adjoin.clone(),
                    },
                ),
                NodeData::Parameter {
                    primal: _,
                    adjoin: _,
                } => panic!("Node already is a Parameter!"),
                NodeData::Unset => (
                    None,
                    NodeData::Variable {
                        primal: Some(value),
                        adjoin: None,
                    },
                ),
            };
            *node_data = new_data;
            old_primal
        } else {
            data.insert(
                ident.clone(),
                NodeData::Variable {
                    primal: Some(value),
                    adjoin: None,
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

    /// Reset primals for variables. Keep adjoins, and primals for Parameters.
    pub fn reset_state_for_next_input(&mut self) {
        {
            let mut data = self.data.borrow_mut();
            for (_, node) in data.iter_mut() {
                match node {
                    NodeData::Variable { primal: _, adjoin } => {
                        let new_node = NodeData::Variable {
                            primal: None,
                            adjoin: adjoin.clone(),
                        };
                        *node = new_node;
                    }
                    NodeData::Parameter {
                        primal: _,
                        adjoin: _,
                    } => (),
                    NodeData::Unset => (),
                }
            }
        }
        //self.refill_primals_that_were_explicitly_set();
    }

    /// Reset the internal state (variable primals, adjoins). Do not clean parameters.
    pub fn reset_state_for_next_epoch(&mut self) {
        {
            let mut data = self.data.borrow_mut();
            for (ident, node) in data.iter_mut() {
                let new_node = match node {
                    NodeData::Variable {
                        primal: _,
                        adjoin: _,
                    } => NodeData::Variable {
                        primal: None,
                        adjoin: None,
                    },
                    // It could be marginally faster to modify node in-place but more error prone if there are new fields in NodeData.
                    NodeData::Parameter { primal, adjoin: _ } => NodeData::Parameter {
                        primal: primal.clone(),
                        adjoin: None,
                    },
                    NodeData::Unset => NodeData::Unset,
                };
                eprintln!(
                    "reset_state_for_next_epoch {:?} {:?} {:?} ==> {:?}",
                    ident,
                    self.get_name(ident),
                    node,
                    new_node
                );
                *node = new_node;
            }
        }
        self.saved_variables = RefCell::new(BTreeMap::new());
        //self.restore_parameters()
    }

    //fn restore_parameters(&mut self) {
    //    let params_vec: Vec<(Ident, F)>;
    //    {
    //        let params = self.params.borrow();
    //        params_vec = params
    //            .iter()
    //            .map(|(ident, value)| (ident.clone(), value.clone()))
    //            .collect();
    //    }
    //    for (ident, value) in params_vec {
    //        self.set_variable(&ident, value);
    //    }
    //}

    pub fn update_params_lr(&mut self, learning_rate: f32) {
        let mut data = self.data.borrow_mut();
        let param_idents: Vec<Ident> = data
            .iter()
            .filter_map(|(ident, node_data)| {
                if let NodeData::Parameter {
                    primal: _,
                    adjoin: _,
                } = node_data
                {
                    Some(ident.clone())
                } else {
                    None
                }
            })
            .collect();
        for ident in param_idents {
            let node_data = data.get_mut(&ident).unwrap();
            let (primal, adjoin) = if let NodeData::Parameter { primal, adjoin } = node_data {
                if let Some(adjoin) = adjoin.as_ref() {
                    (primal, adjoin)
                } else {
                    panic!("Adjoin missing for parameter!")
                }
            } else {
                panic!("Expected Parameter!")
            };
            // -1.0 because Add and Mul is implemented but Sub not necessarily.
            let new_primal = primal.clone() + adjoin.clone() * -1.0 * learning_rate;
            *primal = new_primal;
        }
    }

    fn save_variable(&mut self, ident: &Ident, value: F) {
        let mut saved_variables = self.saved_variables.borrow_mut();
        saved_variables.insert(ident.clone(), value);
    }

    //fn refill_primals_that_were_explicitly_set(&mut self) {
    //    let saved_variables_vec: Vec<(Ident, F)>;
    //    {
    //        let saved_variables = self.saved_variables.borrow();
    //        saved_variables_vec = saved_variables
    //            .iter()
    //            .map(|(ident, value)| (ident.clone(), value.clone()))
    //            .collect();
    //    }
    //    for (ident, value) in saved_variables_vec {
    //        self.reset_variable(&ident, value);
    //    }
    //}

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
                NodeData::Variable { primal, adjoin: _ } => match primal {
                    Some(primal) => return primal.clone(),
                    None => None,
                },

                NodeData::Parameter { primal, adjoin: _ } => return primal.clone(),
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

        //eprintln!(
        //    "forward {} {:?} before compute_primal",
        //    ident,
        //    self.get_name(ident),
        //);
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
                NodeData::Parameter { primal: _, adjoin: _ } => panic!("The node is Parameter but expected Variable!"),
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
        let maybe_old_adjoin : &mut Option<F> = match node_data {
            NodeData::Variable { primal: _, adjoin } => adjoin,
            NodeData::Parameter { primal: _, adjoin } => adjoin,
            NodeData::Unset => panic!("The node is Unset during add_adjoin, but it should be already a variable or parameter!"),
        };
        let updated_adjoin: F = if let Some(old) = maybe_old_adjoin {
            old.clone() + adjoin.clone()
        } else {
            adjoin.clone()
        };
        *maybe_old_adjoin = Some(updated_adjoin)
    }

    pub fn primal(&self, ident: &Ident) -> F {
        let data = self.data.borrow();
        let node_data = data.get(ident).expect("Bug! Node data missing");
        match node_data {
            NodeData::Variable { primal, adjoin: _ } => if let Some(primal) = primal {
                primal.clone()
            } else {
                panic!("Primal missing for {}", &ident)
            },
            NodeData::Parameter { primal, adjoin: _ } => primal.clone(),
            NodeData::Unset => panic!("The node is Unset during .primal(), but it should be already a variable or parameter!"),
        }
    }

    pub fn adjoin(&self, ident: &Ident) -> F {
        let data = self.data.borrow();
        let node_data = data.get(ident).expect("Bug: node data missing!");
        let maybe_adjoin = match node_data {
            NodeData::Variable { primal: _, adjoin } => adjoin,
            NodeData::Parameter { primal: _, adjoin } => adjoin,
            NodeData::Unset => panic!("The node is Unset during .adjoin(), but it should be already a variable or parameter!"),
        };
        match maybe_adjoin {
            Some(adjoin) => adjoin.clone(),
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
