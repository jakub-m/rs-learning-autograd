use std::fmt;
use std::ops;

use crate::core_syntax::Operator;
use crate::core_syntax::{Expr, Node};

#[derive(Copy, Clone, Debug)]
pub enum FloatOperAry1 {}

// Bespoke set of Ary2 operations
#[derive(Copy, Clone, Debug)]
pub enum FloatOperAry2 {
    Add,
    Mul,
}

/// TODO Just a marker trait. How to remove necessity of explicit `impl Operator for FloatOper`?
impl Operator for FloatOperAry1 {}

impl fmt::Display for FloatOperAry1 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        todo!()
    }
}

impl Operator for FloatOperAry2 {}

impl fmt::Display for FloatOperAry2 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            FloatOperAry2::Add => "+",
            FloatOperAry2::Mul => "*",
        };
        write!(f, "{}", s)
    }
}

impl<'a> ops::Add for Expr<'a, FloatOperAry1, FloatOperAry2> {
    type Output = Expr<'a, FloatOperAry1, FloatOperAry2>;

    fn add(self, rhs: Self) -> Self::Output {
        let node = Node::Ary2(FloatOperAry2::Add, self.ident, rhs.ident);
        self.register_and_continue_expr(node)
    }
}
impl<'a> ops::Mul for Expr<'a, FloatOperAry1, FloatOperAry2> {
    type Output = Expr<'a, FloatOperAry1, FloatOperAry2>;

    fn mul(self, rhs: Self) -> Self::Output {
        let node = Node::Ary2(FloatOperAry2::Mul, self.ident, rhs.ident);
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
