use std::ops;

use super::syntax::{MatrixF32, NaOperAry1, NaOperAry2};
use crate::{
    compute::{Calculator, ComputGraph},
    core_syntax::{Ident, Node},
};
use nalgebra as na;

pub struct MatrixCalculator;
type NaMatrixDynF32 = na::Matrix<f32, na::Dyn, na::Dyn, na::VecStorage<f32, na::Dyn, na::Dyn>>;

impl Calculator<NaOperAry1, NaOperAry2, MatrixF32> for MatrixCalculator {
    fn forward(
        &self,
        cg: &ComputGraph<MatrixF32, NaOperAry1, NaOperAry2>,
        ident: &Ident,
    ) -> MatrixF32 {
        let node = cg.get_node(ident);
        match node {
            Node::Const(value) => value,
            Node::Variable(name_id) => panic!("Variable {} should have been set!", name_id),
            Node::Ary1(op, a) => match op {
                NaOperAry1::Relu => {
                    let primal = cg.forward(&a);
                    match primal {
                        MatrixF32::M(m) => MatrixF32::new_m(m.as_ref().clone().relu()),
                        MatrixF32::V(v) => MatrixF32::V(v.relu()),
                    }
                }
                NaOperAry1::PowI(exp) => {
                    let primal = cg.forward(&a);
                    primal.powi(exp)
                }
            },
            Node::Ary2(op, a, b) => match op {
                NaOperAry2::Add => {
                    let a = cg.forward(&a);
                    let b = cg.forward(&b);
                    match (a, b) {
                        (MatrixF32::M(m1), MatrixF32::M(m2)) => {
                            MatrixF32::new_m(m1.as_ref() + m2.as_ref())
                        }
                        (MatrixF32::M(m), MatrixF32::V(v)) => {
                            MatrixF32::new_m(m.as_ref().clone().add_scalar(v))
                        }
                        (MatrixF32::V(v), MatrixF32::M(m)) => {
                            MatrixF32::new_m(m.as_ref().clone().add_scalar(v))
                        }
                        (MatrixF32::V(v1), MatrixF32::V(v2)) => MatrixF32::V(v1 + v2),
                    }
                }
                NaOperAry2::Sub => {
                    let a = cg.forward(&a);
                    let b = cg.forward(&b);
                    match (a, b) {
                        (MatrixF32::M(m1), MatrixF32::M(m2)) => {
                            MatrixF32::new_m(m1.as_ref() - m2.as_ref())
                        }
                        (MatrixF32::M(m), MatrixF32::V(v)) => {
                            MatrixF32::new_m(m.as_ref().clone().add_scalar(-v))
                        }
                        (MatrixF32::V(v), MatrixF32::M(m)) => {
                            MatrixF32::new_m(m.as_ref().clone().add_scalar(-v))
                        }
                        (MatrixF32::V(v1), MatrixF32::V(v2)) => MatrixF32::V(v1 - v2),
                    }
                }
                NaOperAry2::MulComp => {
                    let a = cg.forward(&a);
                    let b = cg.forward(&b);
                    &a * &b
                }
            },
        }
    }

    fn backward(
        &self,
        cg: &ComputGraph<MatrixF32, NaOperAry1, NaOperAry2>,
        ident: &Ident,
        adjoin: &MatrixF32,
    ) {
        cg.add_adjoin(ident, adjoin);
        let node = cg.get_node(ident);
        match node {
            Node::Const(_) => (),
            Node::Variable(_) => (),
            Node::Ary1(op, v1) => match op {
                NaOperAry1::Relu => {
                    let primal = cg.primal(&v1);
                    let b = primal.backward_relu();
                    self.backward(cg, &v1, &(adjoin * &b))
                }
                NaOperAry1::PowI(p) => {
                    let a = cg.primal(&v1);
                    let a = a.backward_powi(p);
                    let new_adjoin = &a * adjoin;
                    self.backward(cg, &v1, &new_adjoin);
                }
            },
            Node::Ary2(op, v1, v2) => match op {
                NaOperAry2::Add => {
                    self.backward(cg, &v1, adjoin);
                    self.backward(cg, &v2, adjoin);
                }
                NaOperAry2::Sub => {
                    self.backward(cg, &v1, adjoin);
                    self.backward(cg, &v2, &(adjoin * &MatrixF32::V(-1.0)));
                }
                NaOperAry2::MulComp => {
                    let v1_p = cg.primal(&v1);
                    let v2_p = cg.primal(&v2);
                    self.backward(cg, &v1, &(adjoin * &v2_p));
                    self.backward(cg, &v2, &(adjoin * &v1_p));
                }
            },
        }
    }
}

trait Relu {
    fn relu(&self) -> Self;
    fn backward_relu(&self) -> Self;
}

impl Relu for MatrixF32 {
    fn relu(&self) -> Self {
        match self {
            MatrixF32::M(m) => MatrixF32::new_m(m.as_ref().relu()),
            MatrixF32::V(v) => MatrixF32::V(v.relu()),
        }
    }

    fn backward_relu(&self) -> Self {
        match self {
            MatrixF32::M(m) => MatrixF32::new_m(m.as_ref().backward_relu()),
            MatrixF32::V(v) => MatrixF32::V(v.backward_relu()),
        }
    }
}

impl Relu for f32 {
    fn relu(&self) -> Self {
        if *self <= 0.0 {
            0.0
        } else {
            *self
        }
    }

    fn backward_relu(&self) -> Self {
        if *self <= 0.0 {
            0.0
        } else {
            1.0
        }
    }
}

impl Relu for NaMatrixDynF32 {
    fn relu(&self) -> Self {
        let mut m = self.clone();
        for e in m.iter_mut() {
            *e = e.relu()
        }
        m
    }

    fn backward_relu(&self) -> Self {
        let mut m = self.clone();
        for e in m.iter_mut() {
            *e = e.backward_relu();
        }
        m
    }
}

trait PowI {
    fn powi(&self, p: i32) -> Self;
    fn backward_powi(&self, p: i32) -> Self;
}

impl PowI for MatrixF32 {
    fn powi(&self, p: i32) -> Self {
        match self {
            MatrixF32::M(m) => MatrixF32::new_m(PowI::powi(m.as_ref(), p)),
            MatrixF32::V(v) => MatrixF32::V(PowI::powi(v, p)),
        }
    }

    fn backward_powi(&self, p: i32) -> Self {
        match self {
            MatrixF32::M(m) => MatrixF32::new_m(PowI::backward_powi(m.as_ref(), p)),
            MatrixF32::V(v) => MatrixF32::V(PowI::backward_powi(v, p)),
        }
    }
}

impl PowI for f32 {
    fn powi(&self, p: i32) -> Self {
        f32::powi(*self, p)
    }

    fn backward_powi(&self, p: i32) -> Self {
        (p as f32) * PowI::powi(self, p - 1)
    }
}

impl PowI for NaMatrixDynF32 {
    fn powi(&self, p: i32) -> Self {
        let mut m = self.clone();
        m.apply(|v| *v = PowI::powi(v, p));
        m
    }

    fn backward_powi(&self, p: i32) -> Self {
        let mut a = self.clone();
        a.apply(|v| *v = v.backward_powi(p));
        a
    }
}

/// Element-wise multiplication.
impl ops::Mul for &MatrixF32 {
    type Output = MatrixF32;

    fn mul(self, b: Self) -> Self::Output {
        let a = self;
        match (a, b) {
            (MatrixF32::M(m1), MatrixF32::M(m2)) => {
                MatrixF32::new_m(m1.as_ref().component_mul(m2.as_ref()))
            }
            (MatrixF32::M(m), MatrixF32::V(v)) => MatrixF32::new_m(m.as_ref() * (*v)),
            (MatrixF32::V(v), MatrixF32::M(m)) => MatrixF32::new_m(m.as_ref() * (*v)),
            (MatrixF32::V(v1), MatrixF32::V(v2)) => MatrixF32::V(v1 * v2),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::MatrixCalculator;
    use crate::{
        compute::ComputGraph,
        core_syntax::ExprBuilder,
        na2::syntax::{MatrixF32, NaOperAry1, NaOperAry2},
    };
    use nalgebra as na;

    #[test]
    fn forward_add_mul() {
        let eb = new_eb();
        let a = eb.new_variable("a");
        let b = eb.new_variable("b");
        let c = eb.new_variable("c");
        let y = a + b * c;
        assert_eq!("(a + (b .* c))", format!("{}", y));

        let [a, b, c, y] = [a, b, c, y].map(|p| p.ident());
        let mut cb = ComputGraph::<MatrixF32, NaOperAry1, NaOperAry2>::new(eb, &MatrixCalculator);
        cb.set_variable(&a, na::DMatrix::from_element(2, 2, 1.0_f32).into());
        cb.set_variable(&b, na::DMatrix::from_element(2, 2, 2.0_f32).into());
        cb.set_variable(&c, na::DMatrix::from_element(2, 2, 3.0_f32).into());
        let y = cb.forward(&y);
        assert_eq!(y.m(), Some(&na::DMatrix::from_element(2, 2, 7.0)));
    }

    #[test]
    fn forward_relu() {
        let eb = new_eb();
        let a = eb.new_variable("a");
        let y = a.relu();

        let [a, y] = [a, y].map(|p| p.ident());
        let mut cb = ComputGraph::<MatrixF32, NaOperAry1, NaOperAry2>::new(eb, &MatrixCalculator);
        cb.set_variable(
            &a,
            na::DMatrix::from_vec(2, 2, vec![-1.0, -3.0, 0.0, 42.0]).into(),
        );
        let y = cb.forward(&y);
        assert_eq!(
            y.m(),
            Some(&na::DMatrix::from_vec(2, 2, vec![0.0, 0.0, 0.0, 42.0]))
        );
    }

    #[test]
    fn forward_add_sub_mul_mixed() {
        let eb = new_eb();
        let m = eb.new_variable("m");
        let v = eb.new_variable("v");
        let y = (m * (v + v) - v) * m;
        let [m, v, y] = [m, v, y].map(|p| p.ident());
        let mut cb = ComputGraph::<MatrixF32, NaOperAry1, NaOperAry2>::new(eb, &MatrixCalculator);
        cb.set_variable(
            &m,
            na::DMatrix::from_vec(2, 2, vec![3.0, 3.0, 3.0, 3.0]).into(),
        );
        cb.set_variable(&v, 2.0.into());
        let y = cb.forward(&y);
        let expected = (3.0 * (2.0 + 2.0) - 2.0) * 3.0;
        assert_eq!(y.m(), Some(&na::DMatrix::from_element(2, 2, expected)));
    }

    #[test]
    fn backward_add_mul() {
        let eb = new_eb();
        let a = eb.new_variable("a");
        let b = eb.new_variable("b");
        let c = eb.new_variable("c");
        let y = a + b * c;

        let [a, b, c, y] = [a, b, c, y].map(|p| p.ident());
        let mut cb = ComputGraph::<MatrixF32, NaOperAry1, NaOperAry2>::new(eb, &MatrixCalculator);
        cb.set_variable(&a, na::DMatrix::from_element(2, 2, 1.0_f32).into());
        cb.set_variable(&b, na::DMatrix::from_element(2, 2, 2.0_f32).into());
        cb.set_variable(&c, na::DMatrix::from_element(2, 2, 3.0_f32).into());
        cb.forward(&y);
        cb.backward(&y);

        // TODO: check by hand that those adjoins are correct.
        assert_eq!(
            cb.adjoin(&a).m(),
            Some(&na::DMatrix::from_element(2, 2, 1.0))
        );
        assert_eq!(
            cb.adjoin(&b).m(),
            Some(&na::DMatrix::from_element(2, 2, 3.0))
        );
        assert_eq!(
            cb.adjoin(&c).m(),
            Some(&na::DMatrix::from_element(2, 2, 2.0))
        );
    }

    #[test]
    fn backward_relu() {
        let eb = new_eb();
        let a = eb.new_variable("a");
        let y = a.relu();

        let [a, y] = [a, y].map(|p| p.ident());
        let mut cb = ComputGraph::<MatrixF32, NaOperAry1, NaOperAry2>::new(eb, &MatrixCalculator);
        cb.set_variable(
            &a,
            na::DMatrix::from_vec(2, 2, vec![-2.0, 0.0, 0.0, 2.0]).into(),
        );
        cb.forward(&y);
        cb.backward(&y);

        assert_eq!(
            cb.adjoin(&a).m(),
            Some(&na::DMatrix::from_vec(2, 2, vec![0.0, 0.0, 0.0, 1.0]))
        );
    }

    fn new_eb() -> ExprBuilder<MatrixF32, NaOperAry1, NaOperAry2> {
        ExprBuilder::new()
    }
}
