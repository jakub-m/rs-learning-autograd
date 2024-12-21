//! This module abstracts how to compute values out of nodes.

use core::fmt;
use std::{cell::RefCell, collections::BTreeMap, ops};

use crate::core_syntax::{Expr, ExprBuilder, Ident, Node, Operator, VariableNameId};

impl AsRef<Ident> for Ident {
    fn as_ref(&self) -> &Ident {
        return self;
    }
}

impl<'a, F, OP1, OP2> AsRef<Ident> for Expr<'a, F, OP1, OP2>
where
    OP1: Operator,
    OP2: Operator,
{
    fn as_ref(&self) -> &Ident {
        &self.ident
    }
}

/// A type of the computed value (like, f32). Add is needed to update adjoins.
pub trait ComputValue:
    Clone + fmt::Display + fmt::Debug + DefaultAdjoin + ops::Add<Output = Self>
{
}

pub trait DefaultAdjoin {
    fn default_adjoin() -> Self;
}

/// "Freezes" the expression tree by taking ownership of the expression builder.
pub struct ComputGraph<'a, F, OP1, OP2>
where
    F: ComputValue,
    OP1: Operator,
    OP2: Operator,
{
    /// Primals are computed during a forward pass.
    primals: RefCell<BTreeMap<Ident, F>>,
    /// Adjoins are updated during a backward pass.
    adjoins: RefCell<BTreeMap<Ident, F>>,
    eb: ExprBuilder<F, OP1, OP2>,
    calculator: &'a dyn Calculator<OP1, OP2, F>,
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
        ComputGraph {
            primals: RefCell::new(BTreeMap::new()),
            adjoins: RefCell::new(BTreeMap::new()),
            eb,
            calculator,
        }
    }

    pub fn set_variable(&mut self, ident: &dyn AsRef<Ident>, value: F) {
        let ident = ident.as_ref();
        self.assert_ident_is_variable(ident);
        if let Some(old) = self
            .primals
            .borrow_mut()
            .insert(ident.clone(), value.clone())
        {
            panic!("Value for {} already set to {}", ident, old);
        }
    }

    pub fn get_node(&self, ident: &Ident) -> Node<F, OP1, OP2> {
        let id_to_node = self.eb.id_to_node.borrow();
        let node = id_to_node
            .get(ident)
            .expect(format!("No node for ident {}", ident).as_str());
        node.clone()
    }

    pub fn reset(&mut self) {
        self.primals = RefCell::new(BTreeMap::new());
        self.adjoins = RefCell::new(BTreeMap::new());
    }

    /// Forward pass, calculate primals.
    /// This operation is MUTABLE, i.e. it mutates the internal cache of the calculated values.
    pub fn forward(&self, ident: &dyn AsRef<Ident>) -> F {
        let ident = ident.as_ref();
        if let Some(value) = self.primals.borrow().get(&ident) {
            return value.clone();
        }
        let primal = self.calculator.forward(self, ident);
        if let Some(old) = self
            .primals
            .borrow_mut()
            .insert(ident.clone(), primal.clone())
        {
            panic!("The value for {} already set to {}", ident, old);
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
        self.calculator.backward(self, ident, F::default_adjoin());
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
        let id_to_name = self.eb.id_to_name.borrow();
        id_to_name
            .get(&name_id)
            .expect("no name for such ident")
            .to_owned()
    }

    /// Call `add_adjoin` to update adjoin for a node with partial adjoin.
    pub fn add_adjoin(&self, ident: &Ident, adjoin: F) {
        let mut adjoins = self.adjoins.borrow_mut();
        let updated = adjoins
            .get(ident)
            .map_or(adjoin.clone(), |old| old.clone() + adjoin.clone());
        adjoins.insert(ident.clone(), updated);
    }

    pub fn primal(&self, ident: &Ident) -> F {
        let primals = self.primals.borrow();
        primals
            .get(ident)
            .expect(format!("Primal missing for {}", ident).as_str())
            .clone()
    }

    pub fn adjoin(&self, ident: &Ident) -> F {
        let adjoins = self.adjoins.borrow();
        adjoins
            .get(ident)
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
    fn backward(&self, cg: &ComputGraph<F, OP1, OP2>, ident: &Ident, adjoin: F);
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
