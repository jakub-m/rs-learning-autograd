//! An exemplary implementation of core syntax for float type.
use std::fmt;
use std::ops;

use crate::core_syntax::Operator;
use crate::core_syntax::{Expr, Node};

// Bespoke set of Ary2 operations
#[derive(Copy, Clone, Debug)]
enum FloatOper {
    Add,
    Mul,
}

/// TODO Just a marker trait. How to remove necessity of explicit `impl Operator for FloatOper`?
impl Operator for FloatOper {}

impl fmt::Display for FloatOper {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            FloatOper::Add => "+",
            FloatOper::Mul => "*",
        };
        write!(f, "{}", s)
    }
}

impl<'a> ops::Add for Expr<'a, FloatOper> {
    type Output = Expr<'a, FloatOper>;

    fn add(self, rhs: Self) -> Self::Output {
        let node = Node::Ary2(FloatOper::Add, self.ident, rhs.ident);
        self.register_and_continue_expr(node)
    }
}
impl<'a> ops::Mul for Expr<'a, FloatOper> {
    type Output = Expr<'a, FloatOper>;

    fn mul(self, rhs: Self) -> Self::Output {
        let node = Node::Ary2(FloatOper::Mul, self.ident, rhs.ident);
        self.register_and_continue_expr(node)
    }
}

#[cfg(test)]
mod tests {
    use crate::core_syntax::ExprBuilder;

    #[test]
    fn syntax() {
        let eb = ExprBuilder::new();
        let x1 = eb.new_variable("x1");
        let x2 = eb.new_variable("x2");
        let x3 = x1 + x2;
        let x4 = x1 + x2;
        let z = x1 + x2 * x3 + x4;
        assert_eq!("(+ (+ x1 (* x2 (+ x1 x2))) (+ x1 x2))", format!("{}", z));
    }
}
