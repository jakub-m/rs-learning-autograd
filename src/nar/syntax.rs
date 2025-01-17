use crate::core_syntax::{ComputValue, DefaultAdjoin, Expr, ExprNode, Operator};
use ndarray as nd;

use std::fmt;
use std::ops;
use std::rc::Rc;

#[derive(Debug, Clone, Copy)]
pub enum NaOperAry1 {
    /// ReLU
    Relu,
    /// Power to integer.
    PowI(i32),
    /// Add all the elements of the matrix and return a single value.
    Sum,
}

impl Operator for NaOperAry1 {}

#[derive(Debug, Clone, Copy)]
pub enum NaOperAry2 {
    Add,
    Sub,
    // Element-wise multiplication.
    MulComp,
    Conv2d,
}

impl Operator for NaOperAry2 {}

impl fmt::Display for NaOperAry1 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            NaOperAry1::Relu => "relu".to_owned(),
            NaOperAry1::PowI(p) => format!("pow{}", p),
            NaOperAry1::Sum => "sum".to_owned(),
        };
        write!(f, "{}", s)
    }
}

impl fmt::Display for NaOperAry2 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            NaOperAry2::Add => " + ",
            NaOperAry2::Sub => " - ",
            NaOperAry2::MulComp => " .* ",
            NaOperAry2::Conv2d => "conv2d",
        };
        write!(f, "{}", s)
    }
}

/// For all the practical purposes, V of value `v` can be treat as M with all the fields set to `v`.
#[derive(Debug, Clone, PartialEq)]
pub enum MatrixF32 {
    /// Matrix. Use [Rc] so `.clone` does not clone whole matrix, but only a reference to the matrix.
    M(Rc<nd::ArrayD<f32>>),
    /// Single value. Useful to have syntax like "matrix*2 + 3.0".
    V(f32),
}

impl MatrixF32 {
    pub fn new_m(m: nd::ArrayD<f32>) -> MatrixF32 {
        MatrixF32::M(Rc::new(m))
    }

    /// Borrow the underlying matrix.
    pub fn m(&self) -> Option<&nd::ArrayD<f32>> {
        match self {
            MatrixF32::M(m) => Some(m),
            MatrixF32::V(_) => None,
        }
    }

    pub fn v(&self) -> Option<f32> {
        match self {
            MatrixF32::M(_) => None,
            MatrixF32::V(v) => Some(*v),
        }
    }
}

impl From<nd::ArrayBase<nd::OwnedRepr<f32>, nd::Dim<nd::IxDynImpl>>> for MatrixF32 {
    fn from(value: nd::ArrayBase<nd::OwnedRepr<f32>, nd::Dim<nd::IxDynImpl>>) -> Self {
        MatrixF32::new_m(value)
    }
}

impl From<f32> for MatrixF32 {
    fn from(value: f32) -> Self {
        MatrixF32::V(value)
    }
}

impl ComputValue for MatrixF32 {}

impl ops::Add for MatrixF32 {
    type Output = Self;

    fn add(self, other: Self) -> Self::Output {
        //DMatrixF32::new(self.0.as_ref() + rhs.0.as_ref())
        match (self, other) {
            (MatrixF32::M(m1), MatrixF32::M(m2)) => {
                MatrixF32::M(Rc::new(m1.as_ref() + m2.as_ref()))
            }
            (MatrixF32::M(m1), MatrixF32::V(v2)) => MatrixF32::M(Rc::new(m1.as_ref() + v2)),
            (MatrixF32::V(v1), MatrixF32::M(m2)) => MatrixF32::M(Rc::new(m2.as_ref() + v1)),
            (MatrixF32::V(v1), MatrixF32::V(v2)) => MatrixF32::V(v1 + v2),
        }
    }
}

impl ops::Mul<f32> for MatrixF32 {
    type Output = Self;

    fn mul(self, rhs: f32) -> Self::Output {
        match self {
            MatrixF32::M(m) => MatrixF32::new_m(m.mapv(|v| v * rhs)),
            MatrixF32::V(v) => MatrixF32::V(v * rhs),
        }
    }
}

impl fmt::Display for MatrixF32 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            MatrixF32::M(m) => format!("{}", m),
            MatrixF32::V(v) => format!("{}", v),
        };
        write!(f, "{}", s)
    }
}

impl DefaultAdjoin for MatrixF32 {
    fn default_adjoin(value: Self) -> Self {
        let a = 1.0;
        match value {
            MatrixF32::M(m) => {
                let m = nd::ArrayD::from_elem(m.shape(), a);
                MatrixF32::M(Rc::new(m))
            }
            MatrixF32::V(_) => MatrixF32::V(a),
        }
    }
}

type ExprMatrix<'a> = Expr<'a, MatrixF32, NaOperAry1, NaOperAry2>;

impl<'a> ExprMatrix<'a> {
    pub fn relu(&self) -> ExprMatrix<'a> {
        let node = ExprNode::Ary1(NaOperAry1::Relu, self.ident);
        self.register_and_continue_expr(node)
    }

    pub fn powi(&self, p: i32) -> ExprMatrix<'a> {
        let node = ExprNode::Ary1(NaOperAry1::PowI(p), self.ident);
        self.register_and_continue_expr(node)
    }

    pub fn sum(&self) -> ExprMatrix<'a> {
        let node = ExprNode::Ary1(NaOperAry1::Sum, self.ident);
        self.register_and_continue_expr(node)
    }

    pub fn conv2d(&self, kernel: ExprMatrix<'a>) -> ExprMatrix<'a> {
        let node = ExprNode::Ary2(NaOperAry2::Conv2d, self.ident, kernel.ident);
        self.register_and_continue_expr(node)
    }
}

impl<'a> ops::Add for ExprMatrix<'a> {
    type Output = ExprMatrix<'a>;

    fn add(self, rhs: Self) -> Self::Output {
        let node = ExprNode::Ary2(NaOperAry2::Add, self.ident, rhs.ident);
        self.register_and_continue_expr(node)
    }
}

impl<'a> ops::Sub for ExprMatrix<'a> {
    type Output = ExprMatrix<'a>;

    fn sub(self, rhs: Self) -> Self::Output {
        let node = ExprNode::Ary2(NaOperAry2::Sub, self.ident, rhs.ident);
        self.register_and_continue_expr(node)
    }
}

/// **Element-wise** multiplication. It's element-wise and not a product since it seems to be more common,
/// and easier to use in an expression.
impl<'a> ops::Mul for ExprMatrix<'a> {
    type Output = ExprMatrix<'a>;

    fn mul(self, rhs: Self) -> Self::Output {
        let node = ExprNode::Ary2(NaOperAry2::MulComp, self.ident, rhs.ident);
        self.register_and_continue_expr(node)
    }
}

#[cfg(test)]
mod tests {
    use super::{MatrixF32, NaOperAry1, NaOperAry2};
    use crate::core_syntax::ExprBuilder;

    #[test]
    fn syntax_add() {
        let eb = new_eb();
        let a = eb.new_variable("a");
        let b = eb.new_variable("b");
        let c = a + b;
        let a = a + a;
        assert_eq!("(a + b)", format!("{}", c));
    }

    #[test]
    fn syntax_copy() {
        let eb = new_eb();
        let a = eb.new_variable("a");
        let c = a + a;
        assert_eq!("(a + a)", format!("{}", c));
    }

    #[test]
    fn syntax_component_mul() {
        let eb = new_eb();
        let a = eb.new_variable("a");
        let b = eb.new_variable("b");
        let c = a * b;
        assert_eq!("(a .* b)", format!("{}", c));
    }

    #[test]
    fn syntax_relu() {
        let eb = new_eb();
        let a = eb.new_variable("a");
        let b = a.relu();
        assert_eq!("relu(a)", format!("{}", b));
    }

    fn new_eb() -> ExprBuilder<MatrixF32, NaOperAry1, NaOperAry2> {
        ExprBuilder::new()
    }
}
