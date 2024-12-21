use crate::{
    compute::{Calculator, ComputGraph, ComputValue, DefaultAdjoin},
    core_syntax::{Ident, Node},
};

use super::syntax::FloatOperAry1;
use super::syntax::FloatOperAry2;

pub struct FloatCalculator;
impl ComputValue for f32 {}
impl DefaultAdjoin for f32 {
    fn default_adjoin() -> Self {
        1.0
    }
}

impl Calculator<FloatOperAry1, FloatOperAry2, f32> for FloatCalculator {
    fn forward(&self, cg: &ComputGraph<f32, FloatOperAry1, FloatOperAry2>, ident: &Ident) -> f32 {
        let node = cg.get_node(ident);
        match node {
            Node::Variable(name_id) => {
                panic!(
                    "Variable not set {} {}",
                    cg.get_variable_name(&name_id),
                    name_id
                )
            }
            Node::Ary1(op, ident1) => match op {
                FloatOperAry1::Cos => {
                    let a = cg.forward(&ident1);
                    a.cos()
                }
                FloatOperAry1::Sin => {
                    let a = cg.forward(&ident1);
                    a.sin()
                }
                FloatOperAry1::Ln => {
                    let a = cg.forward(&ident1);
                    a.ln()
                }
            },
            Node::Ary2(op, ident1, ident2) => match op {
                FloatOperAry2::Add => {
                    let a = cg.forward(&ident1);
                    let b = cg.forward(&ident2);
                    a + b
                }
                FloatOperAry2::Sub => {
                    let a = cg.forward(&ident1);
                    let b = cg.forward(&ident2);
                    a - b
                }
                FloatOperAry2::Mul => {
                    let a = cg.forward(&ident1);
                    let b = cg.forward(&ident2);
                    a * b
                }
                FloatOperAry2::Pow => {
                    let a = cg.forward(&ident1);
                    let b = cg.forward(&ident2);
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
        adjoin: f32,
    ) {
        cg.add_adjoin(ident, adjoin); // TODO move to common place?
        let node = cg.get_node(ident);
        match node {
            Node::Variable(_) => (),
            Node::Ary1(op, v1) => match op {
                FloatOperAry1::Sin => {
                    let v1_p = cg.primal(&v1);
                    let v1_ad = v1_p.cos();
                    self.backward(cg, &v1, adjoin * v1_ad);
                }
                FloatOperAry1::Cos => {
                    let v1_p = cg.primal(&v1);
                    let v1_ad = -1.0 * v1_p.sin();
                    self.backward(cg, &v1, adjoin * v1_ad);
                }
                FloatOperAry1::Ln => {
                    let v1_p = cg.primal(&v1);
                    let v1_ad = 1.0 / v1_p;
                    self.backward(cg, &v1, adjoin * v1_ad);
                }
            },
            Node::Ary2(op, v1, v2) => match op {
                FloatOperAry2::Add => {
                    self.backward(cg, &v1, adjoin * 1.0);
                    self.backward(cg, &v2, adjoin * 1.0);
                }
                FloatOperAry2::Sub => {
                    self.backward(cg, &v1, adjoin * 1.0);
                    self.backward(cg, &v2, adjoin * -1.0);
                }
                FloatOperAry2::Mul => {
                    let v1_p = cg.primal(&v1);
                    let v2_p = cg.primal(&v2);
                    self.backward(cg, &v1, adjoin * v2_p);
                    self.backward(cg, &v2, adjoin * v1_p);
                }
                FloatOperAry2::Pow => {
                    // For y=a^b, the derivatives are:
                    // dy/da = b*a^(b-1)
                    // dy/db = (a^b)*ln(a)
                    let a = cg.primal(&v1);
                    let b = cg.primal(&v2);
                    self.backward(cg, &v1, adjoin * (b * a.powf(b - 1.0)));
                    self.backward(cg, &v2, adjoin * (a.powf(b) * a.ln()));
                }
            },
        }
    }
}
