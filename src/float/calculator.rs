use crate::{
    compute::{Calculator, ComputGraph, Node2},
    core_syntax::Ident,
};

use super::syntax::FloatOperAry1;
use super::syntax::FloatOperAry2;

pub struct FloatCalculator;

impl Calculator<FloatOperAry1, FloatOperAry2, f32> for FloatCalculator {
    fn forward(&self, cg: &ComputGraph<f32, FloatOperAry1, FloatOperAry2>, ident: &Ident) -> f32 {
        let node = cg.get_node(ident);
        match node {
            Node2::Const(value) => value,
            // Variable and Parameter should have been already returned by ComputGraph.
            Node2::Variable { name, .. } => panic!("Parameter not set in .forward(): {}", name),
            Node2::Parameter { name, .. } => panic!("Parameter not set in .forward(): {:?}", name),
            Node2::Ary1 {
                oper: op, arg1: a, ..
            } => match op {
                FloatOperAry1::Cos => {
                    let a = cg.forward(&a);
                    a.cos()
                }
                FloatOperAry1::Sin => {
                    let a = cg.forward(&a);
                    a.sin()
                }
                FloatOperAry1::Ln => {
                    let a = cg.forward(&a);
                    a.ln()
                }
                FloatOperAry1::PowI(b) => {
                    let a = cg.forward(&a);
                    a.powi(b)
                }
                FloatOperAry1::Relu => {
                    let a = cg.forward(&a);
                    if a <= 0.0 {
                        0.0
                    } else {
                        a
                    }
                }
            },
            Node2::Ary2 {
                oper: op,
                arg1: a,
                arg2: b,
                ..
            } => match op {
                FloatOperAry2::Add => {
                    let a = cg.forward(&a);
                    let b = cg.forward(&b);
                    a + b
                }
                FloatOperAry2::Sub => {
                    let a = cg.forward(&a);
                    let b = cg.forward(&b);
                    a - b
                }
                FloatOperAry2::Mul => {
                    let a = cg.forward(&a);
                    let b = cg.forward(&b);
                    a * b
                }
                FloatOperAry2::Pow => {
                    let a = cg.forward(&a);
                    let b = cg.forward(&b);
                    a.powf(b)
                }
            }
            .into(),
        }
    }

    fn backward(
        &self,
        cg: &ComputGraph<f32, FloatOperAry1, FloatOperAry2>,
        ident: &Ident,
        adjoin: &f32,
    ) {
        cg.add_adjoin(ident, adjoin);
        let node = cg.get_node(ident);
        match node {
            Node2::Const(_) => (),
            Node2::Variable { .. } => (),
            Node2::Parameter { .. } => (),
            Node2::Ary1 {
                oper: op, arg1: v1, ..
            } => match op {
                FloatOperAry1::Sin => {
                    let v1_p = cg.primal(&v1);
                    let v1_ad = v1_p.cos();
                    self.backward(cg, &v1, &(adjoin * v1_ad));
                }
                FloatOperAry1::Cos => {
                    let v1_p = cg.primal(&v1);
                    let v1_ad = -1.0 * v1_p.sin();
                    self.backward(cg, &v1, &(adjoin * v1_ad));
                }
                FloatOperAry1::Ln => {
                    let v1_p = cg.primal(&v1);
                    let v1_ad = 1.0 / v1_p;
                    self.backward(cg, &v1, &(adjoin * v1_ad));
                }
                FloatOperAry1::PowI(b) => {
                    let a = cg.primal(&v1);
                    self.backward(cg, &v1, &(adjoin * ((b as f32) * a.powi(b - 1))));
                }
                FloatOperAry1::Relu => {
                    let v1_p = cg.primal(&v1);
                    let v1_ad: f32 = if v1_p <= 0.0 { 0.0 } else { 1.0 };
                    self.backward(cg, &v1, &(adjoin * v1_ad));
                }
            },
            Node2::Ary2 {
                oper: op,
                arg1: v1,
                arg2: v2,
                ..
            } => match op {
                FloatOperAry2::Add => {
                    self.backward(cg, &v1, adjoin);
                    self.backward(cg, &v2, adjoin);
                }
                FloatOperAry2::Sub => {
                    self.backward(cg, &v1, adjoin);
                    self.backward(cg, &v2, &(adjoin * -1.0));
                }
                FloatOperAry2::Mul => {
                    let v1_p = cg.primal(&v1);
                    let v2_p = cg.primal(&v2);
                    self.backward(cg, &v1, &(adjoin * v2_p));
                    self.backward(cg, &v2, &(adjoin * v1_p));
                }
                FloatOperAry2::Pow => {
                    // For y=a^b, the derivatives are:
                    // dy/da = b*a^(b-1)
                    // dy/db = (a^b)*ln(a)
                    let a = cg.primal(&v1);
                    let b = cg.primal(&v2);
                    self.backward(cg, &v1, &(adjoin * (b * a.powf(b - 1.0))));
                    self.backward(cg, &v2, &(adjoin * (a.powf(b) * a.ln())));
                }
            },
        }
    }
}
