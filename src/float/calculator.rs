use crate::{
    compute::{Calculator, ComputGraph, ComputValue, DefaultAdjoin},
    core_syntax::{Ident, Node},
};

use super::syntax::FloatOper;

pub struct FloatCalculator;
impl ComputValue for f32 {}
impl DefaultAdjoin for f32 {
    fn default_adjoin() -> Self {
        1.0
    }
}

impl Calculator<FloatOper, f32> for FloatCalculator {
    fn forward(&self, cg: &ComputGraph<f32, FloatOper>, ident: &Ident) -> f32 {
        let node = cg.get_node(ident);
        match node {
            Node::Variable(name_id) => {
                panic!(
                    "Variable not set {} {}",
                    cg.get_variable_name(&name_id),
                    name_id
                )
            }
            Node::Ary2(op, ident1, ident2) => match op {
                FloatOper::Add => {
                    let a = cg.forward(&ident1);
                    let b = cg.forward(&ident2);
                    a + b
                }
                FloatOper::Mul => {
                    let a = cg.forward(&ident1);
                    let b = cg.forward(&ident2);
                    a * b
                }
            }
            .into(),
        }
    }

    fn backward(&self, cg: &ComputGraph<f32, FloatOper>, ident: &Ident, adjoin: f32) {
        cg.add_adjoin(ident, adjoin); // TODO move to common place?
        let node = cg.get_node(ident);
        match node {
            Node::Variable(_) => (),
            Node::Ary2(op, v1, v2) => match op {
                FloatOper::Add => todo!(),
                FloatOper::Mul => {
                    let v1_p = cg.primal(&v1);
                    let v2_p = cg.primal(&v2);
                    let v1_ad = adjoin * v2_p;
                    let v2_ad = adjoin * v1_p;
                    //cg.add_adjoin(&v1, v1_ad);
                    //cg.add_adjoin(&v2, v2_ad);
                    self.backward(cg, &v1, v1_ad);
                    self.backward(cg, &v2, v2_ad);
                }
            },
        }
    }
}
