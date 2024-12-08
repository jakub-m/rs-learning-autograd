use std::ops;
type F = f32;

pub trait Term {}

/// An input variable.
struct Variable;

impl Term for Variable {}

pub struct SubTerm<'a>(&'a dyn Term, &'a dyn Term);

impl<'a> ops::Sub<&'a dyn Term> for &'a dyn Term {
    type Output = SubTerm<'a>;

    fn sub(self, rhs: &'a dyn Term) -> Self::Output {
        SubTerm(self, rhs)
    }
}

impl<'a> Term for SubTerm<'a> {}
pub struct AddTerm<'a>(&'a dyn Term, &'a dyn Term);

impl<'a> ops::Add<&'a dyn Term> for &'a dyn Term {
    type Output = AddTerm<'a>;

    fn add(self, rhs: &'a dyn Term) -> Self::Output {
        AddTerm(self, rhs)
    }
}

impl<'a> Term for AddTerm<'a> {}

pub struct MulTerm<'a>(&'a dyn Term, &'a dyn Term);

impl<'a> ops::Mul<&'a dyn Term> for &'a dyn Term {
    type Output = MulTerm<'a>;

    fn mul(self, rhs: &'a dyn Term) -> Self::Output {
        MulTerm(self, rhs)
    }
}

impl<'a> Term for MulTerm<'a> {}

pub struct DivTerm<'a>(&'a dyn Term, &'a dyn Term);

impl<'a> ops::Div<&'a dyn Term> for &'a dyn Term {
    type Output = DivTerm<'a>;

    fn div(self, rhs: &'a dyn Term) -> Self::Output {
        DivTerm(self, rhs)
    }
}

impl<'a> Term for DivTerm<'a> {}

pub fn sin<'a>(term: &'a dyn Term) -> SinTerm<'a> {
    SinTerm(term)
}

pub struct SinTerm<'a>(&'a dyn Term);

impl<'a> Term for SinTerm<'a> {}

pub fn exp<'a>(term: &'a dyn Term) -> ExpTerm {
    ExpTerm(term)
}
pub struct ExpTerm<'a>(&'a dyn Term);

impl<'a> Term for ExpTerm<'a> {}

#[cfg(test)]
mod tests {
    use super::*;

    //#[test]
    //fn simple1() {
    //    let x = Variable(3.0);
    //    let a = Param(1.0);
    //    let b = Param(0.0);
    //    let y = a * x + b;
    //    assert_eq!(y.value(), 3)
    //}
    // https://youtu.be/wG_nF1awSSY?si=kd5Ny4055k3mXc8r&t=366

    // f(x1, x2) = [sin(x1/x2) + x1/x2 - exp(x2)] x [x1/x2-exp(x2)]
    //   vm1 v0         --v1-    -v1-                --v1-
    //             ----v2----           --v3----           --v3---
    //                           -------v4------     -----v4------
    //               --------------v5----------
    //             --------------------------------v6--------------

    #[test]
    fn yt1() {
        let x1 = Variable;
        let vm1: &dyn Term = &x1;
        let x2 = Variable;
        let v0: &dyn Term = &x2;
        let v1: &dyn Term = &(vm1 / v0);
        let v2: &dyn Term = &sin(v1);
        let v3: &dyn Term = &exp(v0);
        let v4: &dyn Term = &(v1 - v3);
        let v5: &dyn Term = &(v2 + v4);
        let v6: &dyn Term = &(v5 * v4);
    }
}
