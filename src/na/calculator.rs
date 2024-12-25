use nalgebra::DMatrix;

use super::syntax::{NaOperAry1, NaOperAry2};
use crate::{
    compute::{Calculator, ComputGraph},
    core_syntax::Ident,
};

pub struct DMatrixCalculator;

impl Calculator<NaOperAry1, NaOperAry2, DMatrix<f32>> for DMatrixCalculator {
    fn forward(
        &self,
        cg: &ComputGraph<DMatrix<f32>, NaOperAry1, NaOperAry2>,
        ident: &Ident,
    ) -> DMatrix<f32> {
        todo!()
    }

    fn backward(
        &self,
        cg: &ComputGraph<DMatrix<f32>, NaOperAry1, NaOperAry2>,
        ident: &Ident,
        adjoin: DMatrix<f32>,
    ) {
        todo!()
    }
}

#[cfg(test)]
mod tests {
    use nalgebra::DMatrix;

    use crate::{
        compute::ComputGraph,
        core_syntax::ExprBuilder,
        na::syntax::{NaOperAry1, NaOperAry2},
    };

    use super::DMatrixCalculator;

    #[test]
    fn forward() {
        let eb = new_eb();
        let a = eb.new_variable("a");
        let b = eb.new_variable("b");

        let cb = ComputGraph::<DMatrix<f32>, NaOperAry1, NaOperAry2>::new(eb, &DMatrixCalculator);
    }

    fn new_eb() -> ExprBuilder<DMatrix<f32>, NaOperAry1, NaOperAry2> {
        ExprBuilder::new()
    }
}
