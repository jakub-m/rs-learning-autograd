use nalgebra as na;
use std::ops;
use std::{cell::RefCell, collections::HashSet, marker::PhantomData};
pub trait ComputValue: Clone {}

pub trait Operator: Clone + Copy {}

#[derive(PartialEq, Eq, PartialOrd, Ord, Clone, Copy, Debug, Hash)]
pub struct Ident(usize);

#[derive(Debug)]
pub struct ExprBuilder<F, OP>
where
    F: ComputValue,
    OP: Operator,
{
    name_set: RefCell<HashSet<String>>,
    f: PhantomData<F>,
    op: PhantomData<OP>,
}

#[derive(Clone, Copy, Debug)]
pub struct Expr<'a, F, OP>
where
    F: ComputValue,
    OP: Operator,
{
    ident: Ident,
    eb: &'a ExprBuilder<F, OP>,
}

impl<'a, F, OP> ops::Add for Expr<'a, F, OP>
where
    F: ComputValue,
    OP: Operator,
{
    type Output = Expr<'a, F, OP>;

    fn add(self, rhs: Self) -> Self::Output {
        todo!()
    }
}

#[derive(Clone)]
struct ConcreteF(na::DMatrix<f32>);

#[derive(Clone, Copy)]
enum ConcreteOp {
    Add,
}

impl ComputValue for ConcreteF {}

impl Operator for ConcreteOp {}

#[cfg(test)]
mod tests {
    use std::{cell::RefCell, collections::HashSet, marker::PhantomData};

    use super::{ConcreteF, ConcreteOp, Expr, ExprBuilder, Ident};

    #[test]
    fn test_copy() {
        let eb = new_eb();
        let a;
        {
            a = Expr {
                ident: Ident(1),
                eb: &eb,
            };
        }
        let b = a + a;
    }

    fn new_eb() -> ExprBuilder<ConcreteF, ConcreteOp> {
        ExprBuilder {
            name_set: RefCell::new(HashSet::new()),
            f: PhantomData,
            op: PhantomData,
        }
    }
}
