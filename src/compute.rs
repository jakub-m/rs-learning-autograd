//! This module abstracts how to compute values out of nodes.

use core::fmt;
use std::{cell::RefCell, collections::BTreeMap};

use crate::core_syntax::{Expr, ExprBuilder, Ident, Node, Operator, VariableNameId};

impl AsRef<Ident> for Ident {
    fn as_ref(&self) -> &Ident {
        return self;
    }
}

impl<'a, OP2> AsRef<Ident> for Expr<'a, OP2>
where
    OP2: Operator,
{
    fn as_ref(&self) -> &Ident {
        &self.ident
    }
}

/// A type of the computed value (like, f32).
pub trait ComputValue: Clone + fmt::Display + fmt::Debug {}

/// "Freezes" the expression tree by taking ownership of the expression builder.
pub struct ComputGraph<'a, F, OP2>
where
    F: ComputValue,
    OP2: Operator,
{
    primals: RefCell<BTreeMap<Ident, F>>,
    eb: ExprBuilder<OP2>,
    calculator: &'a dyn Calculator<OP2, F>,
}

impl<'a, F, OP2> ComputGraph<'a, F, OP2>
where
    F: ComputValue,
    OP2: Operator,
{
    /// Take ownership of the expression builder, because it "freezes" the expression. The expression, as represented
    /// by the internal data structures of the expression builder, cannot change from now on.
    pub fn new<F2: ComputValue>(
        eb: ExprBuilder<OP2>,
        calculator: &'a dyn Calculator<OP2, F2>,
    ) -> ComputGraph<'a, F2, OP2> {
        ComputGraph {
            primals: RefCell::new(BTreeMap::new()),
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

    /// Forward pass, calculate primals.
    /// This operation is MUTABLE, i.e. it mutates the internal cache of the calculated values.
    pub fn forward(&self, ident: &dyn AsRef<Ident>) -> F {
        let ident = ident.as_ref();
        if let Some(value) = self.primals.borrow().get(&ident) {
            return value.clone();
        }
        let id_to_node = self.eb.id_to_node.borrow();
        let node = id_to_node
            .get(&ident)
            .expect(format!("Node missing for {}", ident).as_str());
        let primal = self.calculator.forward(self, node);
        if let Some(old) = self
            .primals
            .borrow_mut()
            .insert(ident.clone(), primal.clone())
        {
            panic!("The value for {} already set to {}", ident, old);
        }
        primal
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
}

/// Take node and return a calculated value.
pub trait Calculator<OP2, F>
where
    F: ComputValue,
    OP2: Operator,
{
    /// Take node and return a computed value for it. The passed computational graph allows querying for the
    /// already computed values. The querying for the values can run actual computation, that can be later
    /// cached in the graph.
    ///
    /// Usually, `calculate_primal` should not be called for `Node::Variable` since all the variables should
    /// be set beforehand with `set_variable`. It's ok to `panic` on Node::Variable.
    fn forward(&self, cg: &ComputGraph<F, OP2>, node: &Node<OP2>) -> F;
}

#[cfg(test)]
mod tests {
    use super::{Calculator, ComputGraph, ComputValue};
    use crate::{
        core_syntax::{ExprBuilder, Node},
        float_syntax::FloatOper,
    };

    impl ComputValue for f32 {}

    struct FloatCalculator;

    impl Calculator<FloatOper, f32> for FloatCalculator {
        fn forward(&self, cg: &ComputGraph<f32, FloatOper>, node: &Node<FloatOper>) -> f32 {
            match node {
                Node::Variable(name_id) => {
                    panic!(
                        "Variable not set {} {}",
                        cg.get_variable_name(name_id),
                        name_id
                    )
                }
                Node::Ary2(op, ident1, ident2) => match op {
                    FloatOper::Add => {
                        let a = cg.forward(ident1);
                        let b = cg.forward(ident2);
                        a + b
                    }
                    FloatOper::Mul => {
                        let a = cg.forward(ident1);
                        let b = cg.forward(ident2);
                        a * b
                    }
                }
                .into(),
            }
        }
    }

    #[test]
    fn compute_primal() {
        let eb = ExprBuilder::new();
        let x1 = eb.new_variable("x1");
        let x2 = eb.new_variable("x2");
        let x3 = x1 + x2;
        let x4 = x1 + x2;
        let e = (x1 + x2) * x3 + x4;

        let x1 = x1.ident();
        let x2 = x2.ident();
        let e = e.ident();
        let mut cg = ComputGraph::<f32, FloatOper>::new(eb, &FloatCalculator);
        cg.set_variable(&x1, 3.0);
        cg.set_variable(&x2, 5.0);
        let p = cg.forward(&e);
        assert_eq!(p, (3.0 + 5.0) * (3.0 + 5.0) + (3.0 + 5.0));
    }

    #[test]
    fn compute_simple_tangent() {
        let eb = ExprBuilder::new();
        let x1 = eb.new_variable("x1");
        let x2 = eb.new_variable("x2");
        let y = x1 * x2;

        let x1 = x1.ident();
        let x2 = x2.ident();
        let e = y.ident();
        let mut cg = ComputGraph::<f32, FloatOper>::new(eb, &FloatCalculator);
        cg.set_variable(&x1, 3.0);
        cg.set_variable(&x2, -4.0);
        let p = cg.forward(&e);
        assert_eq!(p, (3.0 * -4.0));
    }
}
