//! This module abstracts how to compute values out of nodes.

use core::fmt;
use std::{cell::RefCell, collections::BTreeMap};

use crate::core_syntax::{Expr, ExprBuilder, Ident, Node, Operator};

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

/// A composite of primal and a relevant tangent.
#[derive(Clone, Debug)]
pub struct Composite<F>
where
    F: ComputValue,
{
    pub primal: F,
    pub tangent: F,
}

impl<F> Composite<F>
where
    F: ComputValue,
{
    fn new(primal: F, tangent: F) -> Composite<F> {
        Composite {
            primal: primal.clone(),
            tangent: tangent.clone(),
        }
    }
}

impl<F> From<(F, F)> for Composite<F>
where
    F: ComputValue,
{
    fn from(value: (F, F)) -> Self {
        Composite {
            primal: value.0,
            tangent: value.1,
        }
    }
}

impl<F> fmt::Display for Composite<F>
where
    F: ComputValue,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "({}, {})", self.primal, self.tangent)
    }
}

/// "Freezes" the expression tree by taking ownership of the expression builder.
pub struct ComputGraph<'a, F, OP2>
where
    F: ComputValue,
    OP2: Operator,
{
    composites: RefCell<BTreeMap<Ident, Composite<F>>>,
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
    ) -> ComputGraph<F2, OP2> {
        ComputGraph {
            composites: RefCell::new(BTreeMap::new()),
            eb,
            calculator,
        }
    }

    pub fn set_variable(&mut self, ident: &dyn AsRef<Ident>, value: Composite<F>) {
        let ident = ident.as_ref();
        self.assert_ident_is_variable(ident);
        if let Some(old) = self
            .composites
            .borrow_mut()
            .insert(ident.clone(), value.clone())
        {
            panic!(
                "Value for {} already set to {}",
                self.get_variable_name(ident),
                old
            );
        }
    }

    /// This operation is MUTABLE, i.e. it mutates the internal cache of the calculated values.
    pub fn calculate(&self, ident: &dyn AsRef<Ident>) -> Composite<F> {
        let ident = ident.as_ref();
        if let Some(value) = self.composites.borrow().get(&ident) {
            return value.clone();
        }
        let id_to_node = self.eb.id_to_node.borrow();
        let node = id_to_node
            .get(&ident)
            .expect(format!("Node missing for {}", ident).as_str());
        let composite = self.calculator.calculate(self, node);
        if let Some(old) = self
            .composites
            .borrow_mut()
            .insert(ident.clone(), composite.clone())
        {
            panic!("The value for {} already set to {}", ident, old);
        }
        composite
    }

    fn assert_ident_is_variable(&self, ident: &Ident) {
        let id_to_node = self.eb.id_to_node.borrow();
        let node = id_to_node.get(ident).expect("No such ident");
        if let Node::Variable(_) = node {
        } else {
            panic!("Not a variable {}: {:?}", ident, node)
        }
    }

    pub fn get_variable_name(&self, ident: &Ident) -> String {
        let id_to_name = self.eb.id_to_name.borrow();
        id_to_name
            .get(ident)
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
    fn calculate(&self, cg: &ComputGraph<F, OP2>, node: &Node<OP2>) -> Composite<F>;
}

#[cfg(test)]
mod tests {
    use super::{Calculator, Composite, ComputGraph, ComputValue};
    use crate::{
        core_syntax::{ExprBuilder, Node},
        float_syntax::FloatOper,
    };

    impl ComputValue for f32 {}

    struct FloatCalculator;

    impl Calculator<FloatOper, f32> for FloatCalculator {
        fn calculate(
            &self,
            cg: &ComputGraph<f32, FloatOper>,
            node: &Node<FloatOper>,
        ) -> Composite<f32> {
            match node {
                Node::Variable(ident) => {
                    panic!("Variable not set {} {}", cg.get_variable_name(ident), ident)
                }
                Node::Ary2(op, ident1, ident2) => match op {
                    FloatOper::Add => {
                        let a = cg.calculate(ident1);
                        let b = cg.calculate(ident2);
                        (a.primal + b.primal, a.tangent + b.tangent)
                    }
                    FloatOper::Mul => {
                        let a = cg.calculate(ident1);
                        let b = cg.calculate(ident2);
                        (
                            a.primal * b.primal,
                            a.primal * b.tangent + a.tangent * b.primal,
                        )
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
        cg.set_variable(&x1, (3.0, 0.0).into());
        cg.set_variable(&x2, (5.0, 0.0).into());
        let p = cg.calculate(&e);
        assert_eq!(p.primal, (3.0 + 5.0) * (3.0 + 5.0) + (3.0 + 5.0));
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
        cg.set_variable(&x1, (3.0, 0.0).into());
        cg.set_variable(&x2, (-4.0, 1.0).into());
        let p = cg.calculate(&e);
        assert_eq!(p.primal, (3.0 * -4.0));
        assert_eq!(p.tangent, 3.0);
    }
}
