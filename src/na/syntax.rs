use crate::core_syntax::{ComputValue, DefaultAdjoin, Expr, Node, Operator};
use nalgebra as na;
use nalgebra::{DMatrix, Dyn, VecStorage};
use std::fmt;
use std::ops;

#[derive(Debug, Clone, Copy)]
enum NaOperAry1 {
    //ReLU
    Relu,
}

impl Operator for NaOperAry1 {}

#[derive(Debug, Clone, Copy)]
enum NaOperAry2 {
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

impl ComputValue for DMatrix<f32> {}

impl DefaultAdjoin for DMatrix<f32> {
    fn default_adjoin() -> Self {
        todo!() // Is this even possible with DMatrix of unknown size?
    }
}

type ExprDMatrix<'a> = Expr<'a, DMatrix<f32>, NaOperAry1, NaOperAry2>;

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

/// **Element-wise** multiplication.
impl<'a> ops::Mul for ExprDMatrix<'a> {
    type Output = ExprDMatrix<'a>;

    fn mul(self, rhs: Self) -> Self::Output {
        let node = Node::Ary2(NaOperAry2::MulComp, self.ident(), rhs.ident());
        self.register_and_continue_expr(node)
    }
}

#[cfg(test)]
mod tests {
    use super::{NaOperAry1, NaOperAry2};
    use crate::core_syntax::ExprBuilder;
    use nalgebra as na;
    use nalgebra::DMatrix;

    #[test]
    fn syntax_add() {
        let eb = new_eb();
        let a = eb.new_variable("a");
        let b = eb.new_variable("b");
        let c = a + b;
        assert_eq!("(a + b)", format!("{}", c));
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

    fn new_eb() -> ExprBuilder<DMatrix<f32>, NaOperAry1, NaOperAry2> {
        ExprBuilder::new()
    }
}
