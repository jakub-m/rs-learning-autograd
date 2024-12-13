use std::hash::Hash;
use std::ops;
use std::rc::Rc;

#[derive(PartialEq, Eq, Hash, Clone)]
pub enum Expr {
    Variable(usize),
    Add(Rc<Expr>, Rc<Expr>),
    Mul(Rc<Expr>, Rc<Expr>),
}

#[derive(Clone)]
pub struct ExprRef(Rc<Expr>);

pub struct Variable(usize);

impl Variable {
    pub fn new(ident: usize) -> Variable {
        Variable(ident)
    }

    pub fn as_expr(self) -> ExprRef {
        ExprRef(Rc::new(Expr::Variable(self.0)))
    }
}

impl From<Variable> for Expr {
    fn from(variable: Variable) -> Self {
        Expr::Variable(variable.0)
    }
}

impl<'a> ops::Add<&'a ExprRef> for &'a ExprRef {
    type Output = ExprRef;

    fn add(self, rhs: &'a ExprRef) -> Self::Output {
        ExprRef(Rc::new(Expr::Add(Rc::clone(&self.0), Rc::clone(&rhs.0))))
    }
}

impl<'a> ops::Mul<&'a ExprRef> for &'a ExprRef {
    type Output = ExprRef;

    fn mul(self, rhs: &'a ExprRef) -> Self::Output {
        ExprRef(Rc::new(Expr::Mul(Rc::clone(&self.0), Rc::clone(&rhs.0))))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn syntax() {
        let x1 = &Variable::new(1).as_expr();
        let x2 = &Variable::new(2).as_expr();
        let zzz = &(x1 + x2) * x1;
        // I want to avoid .clone(), and just have Copy-able ergonomic expression like (x1 + x2) * x1.
    }
}
