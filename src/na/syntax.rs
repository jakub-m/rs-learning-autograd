use crate::core_syntax::{ComputValue, DefaultAdjoin, Expr, Node, Operator};
use nalgebra as na;
use nalgebra::VecStorage;
use std::fmt;
use std::ops;
use std::rc::Rc;

#[derive(Debug, Clone, Copy)]
pub enum NaOperAry1 {
    //ReLU
    Relu,
}

impl Operator for NaOperAry1 {}

#[derive(Debug, Clone, Copy)]
pub enum NaOperAry2 {
    Add,
    // Element-wise multiplication.
    MulComp,
}

impl Operator for NaOperAry2 {}

impl fmt::Display for NaOperAry1 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            NaOperAry1::Relu => "relu",
        };
        write!(f, "{}", s)
    }
}

impl fmt::Display for NaOperAry2 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            NaOperAry2::Add => " + ",
            NaOperAry2::MulComp => " .* ",
        };
        write!(f, "{}", s)
    }
}

/// Use [Rc] so `.clone` does not clone whole matrix, but only a reference to the matrix.
#[derive(Debug, Clone)]
pub struct DMatrixF32(Rc<na::DMatrix<f32>>);

impl DMatrixF32 {
    pub fn new(m: na::DMatrix<f32>) -> DMatrixF32 {
        DMatrixF32(Rc::new(m))
    }

    /// Borrow the underlying matrix.
    pub fn m(&self) -> &na::DMatrix<f32> {
        &self.0
    }
}

impl From<na::Matrix<f32, na::Dyn, na::Dyn, VecStorage<f32, na::Dyn, na::Dyn>>> for DMatrixF32 {
    fn from(value: na::Matrix<f32, na::Dyn, na::Dyn, VecStorage<f32, na::Dyn, na::Dyn>>) -> Self {
        DMatrixF32(Rc::new(value))
    }
}

impl ComputValue for DMatrixF32 {}

impl ops::Add for DMatrixF32 {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        DMatrixF32::new(self.0.as_ref() + rhs.0.as_ref())
    }
}

impl fmt::Display for DMatrixF32 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl DefaultAdjoin for DMatrixF32 {
    fn default_adjoin(value: Self) -> Self {
        let m = value.m();
        let m = na::DMatrix::from_element(m.nrows(), m.ncols(), 1.0);
        DMatrixF32::new(m)
    }
}

type ExprDMatrix<'a> = Expr<'a, DMatrixF32, NaOperAry1, NaOperAry2>;

impl<'a> ExprDMatrix<'a> {
    pub fn relu(&self) -> ExprDMatrix<'a> {
        let node = Node::Ary1(NaOperAry1::Relu, self.ident());
        self.register_and_continue_expr(node)
    }
}

impl<'a> ops::Add for ExprDMatrix<'a> {
    type Output = ExprDMatrix<'a>;

    fn add(self, rhs: Self) -> Self::Output {
        let node = Node::Ary2(NaOperAry2::Add, self.ident(), rhs.ident());
        self.register_and_continue_expr(node)
    }
}

/// **Element-wise** multiplication. It's element-wise and not a product since it seems to be more common,
/// and easier to use in an expression.
impl<'a> ops::Mul for ExprDMatrix<'a> {
    type Output = ExprDMatrix<'a>;

    fn mul(self, rhs: Self) -> Self::Output {
        let node = Node::Ary2(NaOperAry2::MulComp, self.ident(), rhs.ident());
        self.register_and_continue_expr(node)
    }
}

#[cfg(test)]
mod tests {
    use super::{DMatrixF32, NaOperAry1, NaOperAry2};
    use crate::core_syntax::ExprBuilder;

    #[test]
    fn syntax_add() {
        let eb = new_eb();
        let a = eb.new_variable("a");
        let b = eb.new_variable("b");
        let c = a + b;
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

    fn new_eb() -> ExprBuilder<DMatrixF32, NaOperAry1, NaOperAry2> {
        ExprBuilder::new()
    }
}
