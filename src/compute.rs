use core::fmt;
use std::collections::BTreeMap;

use crate::core_syntax::{Expr, ExprBuilder, Ident, Node, Operator};

pub trait GetIdent {
    /// Get the identifier.
    fn ident(&self) -> Ident;
}

impl GetIdent for Ident {
    fn ident(&self) -> Ident {
        self.clone()
    }
}

impl<'a, OP2> GetIdent for Expr<'a, OP2>
where
    OP2: Operator,
{
    fn ident(&self) -> Ident {
        self.ident.clone()
    }
}
pub trait ComputeValue: Clone + fmt::Display {}

/// ComputeGraph "freezes" the expression tree by taking ownership of the expression builder.
pub struct ComputeGraph<F, OP2>
where
    F: ComputeValue,
    OP2: Operator,
{
    primals: BTreeMap<Ident, F>,
    eb: ExprBuilder<OP2>,
}

impl<F, OP2> ComputeGraph<F, OP2>
where
    F: ComputeValue,
    OP2: Operator,
{
    // TODO implement .into()
    pub fn from_expr_builder<F2: ComputeValue>(eb: ExprBuilder<OP2>) -> ComputeGraph<F2, OP2> {
        ComputeGraph {
            primals: BTreeMap::new(),
            eb,
        }
    }

    pub fn set_value(&mut self, ident: &dyn GetIdent, value: F) {
        let ident = ident.ident();
        if let Some(old) = self.primals.insert(ident, value.clone()) {
            panic!("Value for {} already set to {}", ident, old);
        }
    }

    pub fn compute_primal_for(&mut self, ident: &dyn GetIdent) -> F {
        let ident = ident.ident();
        if let Some(value) = self.primals.get(&ident) {
            return value.clone();
        }
        let id_to_node = self.eb.id_to_node.borrow();
        let node = id_to_node
            .get(&ident)
            .expect(format!("Node missing for {}", ident).as_str());
        let primal = match node {
            Node::Variable(node_ident) => {
                // panic, since the primal should have been already returned earlier.
                panic!("Variable {} value not set", self.get_variable_name(&ident))
            }
            Node::Ary2(op, ident1, ident2) => {
                let primal1 = self.compute_primal_for(ident1);
                let primal2 = self.compute_primal_for(ident2);
                let primal = op.calc_primal(&primal1, &primal2);
                self.primals.insert(ident, primal);
                primal
            }
        };
        primal
    }

    fn get_variable_name(&self, ident: &Ident) -> String {
        let id_to_name = self.eb.id_to_name.borrow();
        id_to_name
            .get(ident)
            .expect("no name for such ident")
            .to_owned()
    }
}

#[cfg(test)]
mod tests {
    use super::{ComputeGraph, ComputeValue};
    use crate::core_syntax::ExprBuilder;

    impl ComputeValue for f32 {}

    #[test]
    fn compute() {
        let eb = ExprBuilder::new();
        let x1 = eb.new_variable("x1");
        let x2 = eb.new_variable("x2");
        let x3 = x1 + x2;
        let x4 = x1 + x2;
        let e = (x1 + x2) * x3 + x4;

        // TODO add step to freeze the graph so eb can be discarded.
        let x1 = x1.ident();
        let x2 = x2.ident();
        let e = e.ident();
        let mut cg = ComputeGraph::<f32, _>::from_expr_builder::<f32>(eb);
        cg.set_value(&x1, 3.0);
        cg.set_value(&x2, 5.0);
        assert!(false, "{}", e);
        let p = cg.compute_primal_for(&e);
        assert_eq!(p, (3.0 + 5.0) * (3.0 + 5.0) + (3.0 + 5.0))
    }
}
