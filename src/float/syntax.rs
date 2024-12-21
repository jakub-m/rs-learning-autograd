use std::fmt;
use std::ops;

use crate::core_syntax::Operator;
use crate::core_syntax::{Expr, Node};

#[derive(Copy, Clone, Debug)]
pub enum FloatOperAry1 {
    Cos,
    Sin,
    Ln,
}

// Bespoke set of Ary2 operations
#[derive(Copy, Clone, Debug)]
pub enum FloatOperAry2 {
    Add,
    Sub,
    Mul,
}

/// TODO Just a marker trait. How to remove necessity of explicit `impl Operator for FloatOper`?
impl Operator for FloatOperAry1 {}

impl fmt::Display for FloatOperAry1 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            FloatOperAry1::Cos => "cos",
            FloatOperAry1::Sin => "sin",
            FloatOperAry1::Ln => "ln",
        };
        write!(f, "{}", s)
    }
}

impl Operator for FloatOperAry2 {}

impl fmt::Display for FloatOperAry2 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            FloatOperAry2::Add => "+",
            FloatOperAry2::Mul => "*",
            FloatOperAry2::Sub => "-",
        };
        write!(f, "{}", s)
    }
}

type ExprFloat<'a> = Expr<'a, FloatOperAry1, FloatOperAry2>;

impl<'a> ops::Add for ExprFloat<'a> {
    type Output = ExprFloat<'a>;

    fn add(self, rhs: Self) -> Self::Output {
        let node = Node::Ary2(FloatOperAry2::Add, self.ident, rhs.ident);
        self.register_and_continue_expr(node)
    }
}

impl<'a> ops::Sub for ExprFloat<'a> {
    type Output = ExprFloat<'a>;

    fn sub(self, rhs: Self) -> Self::Output {
        let node = Node::Ary2(FloatOperAry2::Sub, self.ident, rhs.ident);
        self.register_and_continue_expr(node)
    }
}

impl<'a> ops::Mul for ExprFloat<'a> {
    type Output = ExprFloat<'a>;

    fn mul(self, rhs: Self) -> Self::Output {
        let node = Node::Ary2(FloatOperAry2::Mul, self.ident, rhs.ident);
        self.register_and_continue_expr(node)
    }
}

impl<'a> ExprFloat<'a> {
    pub fn cos(&self) -> ExprFloat<'a> {
        let node = Node::Ary1(FloatOperAry1::Cos, self.ident);
        self.register_and_continue_expr(node)
    }

    pub fn sin(&self) -> ExprFloat<'a> {
        let node = Node::Ary1(FloatOperAry1::Sin, self.ident);
        self.register_and_continue_expr(node)
    }

    pub fn ln(&self) -> ExprFloat<'a> {
        let node = Node::Ary1(FloatOperAry1::Ln, self.ident);
        self.register_and_continue_expr(node)
    }
}

#[cfg(test)]
mod tests {
    use super::{FloatOperAry1, FloatOperAry2};
    use crate::core_syntax::ExprBuilder;

    #[test]
    fn syntax() {
        let eb = new_eb();
        let x1 = eb.new_variable("x1");
        let x2 = eb.new_variable("x2");
        let x3 = x1 + x2;
        let x4 = x1 + x2;
        let z = x1 + x2 * x3 + x4;
        assert_eq!("((x1 + (x2 * (x1 + x2))) + (x1 + x2))", format!("{}", z));
    }

    #[test]
    fn cos() {
        let eb = new_eb();
        let x = eb.new_variable("x");
        let y = x.cos().sin();
        assert_eq!("sin(cos(x))", format!("{}", y));
    }

    fn new_eb() -> ExprBuilder<FloatOperAry1, FloatOperAry2> {
        ExprBuilder::new()
    }
}
