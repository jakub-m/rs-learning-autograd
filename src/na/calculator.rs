use super::syntax::{DMatrixF32, NaOperAry1, NaOperAry2};
use crate::{
    compute::{Calculator, ComputGraph},
    core_syntax::Ident,
};

pub struct DMatrixCalculator;

impl Calculator<NaOperAry1, NaOperAry2, DMatrixF32> for DMatrixCalculator {
    fn forward(
        &self,
        cg: &ComputGraph<DMatrixF32, NaOperAry1, NaOperAry2>,
        ident: &Ident,
    ) -> DMatrixF32 {
        todo!()
    }

    fn backward(
        &self,
        cg: &ComputGraph<DMatrixF32, NaOperAry1, NaOperAry2>,
        ident: &Ident,
        adjoin: DMatrixF32,
    ) {
        todo!()
    }
}

#[cfg(test)]
mod tests {
    use super::DMatrixCalculator;
    use crate::{
        compute::ComputGraph,
        core_syntax::ExprBuilder,
        na::syntax::{DMatrixF32, NaOperAry1, NaOperAry2},
    };
    use nalgebra as na;

    #[test]
    fn forward_a_plus_b() {
        let eb = new_eb();
        let a = eb.new_variable("a");
        let b = eb.new_variable("b");
        let c = a + b;

        //let [a, b, c] = [a, b, c].map(|p| p.ident());
        //let mut cb = ComputGraph::<DMatrixF32, NaOperAry1, NaOperAry2>::new(eb, &DMatrixCalculator);
        //cb.set_variable(&a, na::DMatrix::from_element(3, 3, 1.0_f32).into());
        //cb.set_variable(&b, na::DMatrix::from_element(3, 3, 2.0_f32).into());
        todo!()
    }

    fn new_eb() -> ExprBuilder<DMatrixF32, NaOperAry1, NaOperAry2> {
        ExprBuilder::new()
    }
}
