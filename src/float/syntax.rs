//! Implement concrete autograd operations for f32 type.

use std::fmt;
use std::ops;

use crate::core_syntax::{ComputValue, DefaultAdjoin, Expr, ExprBuilder, Node, Operator};

#[derive(Copy, Clone, Debug)]
pub enum FloatOperAry1 {
    Cos,
    Sin,
    Ln,
    /// Power to constant integer value.
    PowI(i32),
    Relu,
}

// Bespoke set of Ary2 operations
#[derive(Copy, Clone, Debug)]
pub enum FloatOperAry2 {
    Add,
    Sub,
    Mul,
    /// Power to other expression.
    Pow,
}

impl ComputValue for f32 {}
impl DefaultAdjoin for f32 {
    fn default_adjoin() -> Self {
        1.0
    }
}
impl Operator for FloatOperAry1 {}

impl fmt::Display for FloatOperAry1 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s: String = match self {
            FloatOperAry1::Cos => "cos".to_owned(),
            FloatOperAry1::Sin => "sin".to_owned(),
            FloatOperAry1::Ln => "ln".to_owned(),
            FloatOperAry1::PowI(p) => format!("pow{}", p),
            FloatOperAry1::Relu => "relu".to_owned(),
        };
        write!(f, "{}", s)
    }
}

impl Operator for FloatOperAry2 {}

impl fmt::Display for FloatOperAry2 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            FloatOperAry2::Add => " + ",
            FloatOperAry2::Mul => " * ",
            FloatOperAry2::Sub => " - ",
            FloatOperAry2::Pow => "^",
        };
        write!(f, "{}", s)
    }
}

type ExprFloat<'a> = Expr<'a, f32, FloatOperAry1, FloatOperAry2>;

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

    /// a^p where p is another expression.
    pub fn pow(&self, p: Self) -> ExprFloat<'a> {
        let node = Node::Ary2(FloatOperAry2::Pow, self.ident, p.ident);
        self.register_and_continue_expr(node)
    }

    /// a^b where b is an integer.
    pub fn powi(&self, b: i32) -> ExprFloat<'a> {
        let node = Node::Ary1(FloatOperAry1::PowI(b), self.ident);
        self.register_and_continue_expr(node)
    }

    pub fn relu(&self) -> ExprFloat<'a> {
        let node = Node::Ary1(FloatOperAry1::Relu, self.ident);
        self.register_and_continue_expr(node)
    }
}

pub trait AsConst {
    /// Produce a constant out of a float.
    fn as_const(self, eb: &ExprBuilder<f32, FloatOperAry1, FloatOperAry2>) -> ExprFloat;
}

impl AsConst for f32 {
    fn as_const(self, eb: &ExprBuilder<f32, FloatOperAry1, FloatOperAry2>) -> ExprFloat {
        let node = Node::Const(self);
        eb.register_node_get_expr(node)
    }
}

#[cfg(test)]
mod tests {
    use super::{AsConst, FloatOperAry1, FloatOperAry2};
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

    #[test]
    fn const_value() {
        let eb = new_eb();
        let x = eb.new_variable("x");
        let y = x + 2.0.as_const(&eb);
        assert_eq!("(x + 2)", format!("{}", y));
    }

    #[test]
    fn powi() {
        let eb = new_eb();
        let x = eb.new_variable("x");
        let y = x.powi(2);
        assert_eq!("pow2(x)", format!("{}", y));
    }

    #[ignore]
    #[test]
    fn l2_norm() {
        let eb = new_eb();
        let _x1 = eb.new_variable("x1");
        let _x2 = eb.new_variable("x2");
        let _c1 = 1.0.as_const(&eb);
        let _c2 = 2.0.as_const(&eb);
    }

    fn new_eb() -> ExprBuilder<f32, FloatOperAry1, FloatOperAry2> {
        ExprBuilder::new()
    }
}
