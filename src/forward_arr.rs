//! Use forward_mode to calculate gradient with array as input type

use std::ops::{Add, Div, Mul, Sub};

use crate::forward_mode::{Algebraic, Expr, Variable};

type F = f32;
#[derive(Clone, Copy)]
struct Arr<const SIZE: usize>([F; SIZE]);

impl<const SIZE: usize> Algebraic for Arr<SIZE> {
    fn cos(self) -> Self {
        let mut out = self;
        for i in 0..SIZE {
            out.0[i] = self.0[i].cos();
        }
        out
    }

    fn sin(self) -> Self {
        let mut out = self;
        for i in 0..SIZE {
            out.0[i] = self.0[i].sin();
        }
        out
    }

    fn exp(self) -> Self {
        let mut out = self;
        for i in 0..SIZE {
            out.0[i] = self.0[i].exp();
        }
        out
    }
}

impl<const SIZE: usize> Add for Arr<SIZE> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        let mut out = self;
        for i in 0..SIZE {
            out.0[i] = self.0[i] + rhs.0[i];
        }
        out
    }
}
impl<const SIZE: usize> Mul for Arr<SIZE> {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        let mut out = self;
        for i in 0..SIZE {
            out.0[i] = self.0[i] * rhs.0[i];
        }
        out
    }
}
impl<const SIZE: usize> Sub for Arr<SIZE> {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        let mut out = self;
        for i in 0..SIZE {
            out.0[i] = self.0[i] - rhs.0[i];
        }
        out
    }
}
impl<const SIZE: usize> Div for Arr<SIZE> {
    type Output = Self;

    fn div(self, rhs: Self) -> Self::Output {
        let mut out = self;
        for i in 0..SIZE {
            out.0[i] = self.0[i] / rhs.0[i];
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use crate::forward_mode::State;

    use super::*;
    #[test]
    fn linreg() {
        let x = Variable::new("x").as_expr();
        let y_true = Variable::new("y_true").as_expr();
        let w = Variable::new("w").as_expr();
        let b = Variable::new("b").as_expr();

        let s = State::<Arr<4>>::new();
        //s.set_expr_value(expr, value, dot); // what initial dot ???
        let y_pred = &(&w * &x) + &b; // todo do not use references everywhere
        let loss = &(&y_pred - &y_true) * &(&y_pred - &y_true); // todo impl pow
                                                                // todo impl mean
    }
}
