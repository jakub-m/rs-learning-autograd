mod utils;
use nalgebra as na;
use rand::{self, rngs::StdRng};
use rand::{Rng, SeedableRng};
use rs_autograd::{
    compute::ComputGraph,
    core_syntax::ExprBuilder,
    na::{
        calculator::DMatrixCalculator,
        syntax::{DMatrixF32, NaOperAry1, NaOperAry2},
    },
};
use utils::{assert_functions_similar, FloatRange, Opts};

#[ignore]  // continue with this test
#[test]
fn test_na_gradient_descent_sin() {
    let target_poly = |x: f32| x.sin();
    let input_range = FloatRange::new(-3.1, 3.2, 0.1);

    let eb = new_eb();
    let x = eb.new_variable("x");
    let p0 = eb.new_variable("p0");
    let p1 = eb.new_variable("p1");
    let y = (x - p0).relu() * p1;
    let t = eb.new_variable("t");
    let loss = (y - t).powi(2);

    let [x, p0, p1, y, t, loss] = [x, p0, p1, y, t, loss].map(|p| p.ident());
    let mut cg = ComputGraph::<DMatrixF32, _, _>::new(eb, &DMatrixCalculator);

    let mut rng = StdRng::seed_from_u64(42);
    let n_params: usize = 10;
    let mut p0_values =
        na::DMatrix::from_iterator(n_params, 1, (0..n_params).map(|_| rng.gen_range(-3.0..3.0)));
    let mut p1_values =
        na::DMatrix::from_iterator(n_params, 1, (0..n_params).map(|_| rng.gen_range(-3.0..3.0)));

    let learning_rate: f32 = 0.01;
    let n_epochs = 2;
    for i in 0..n_epochs {
        cg.reset();
        cg.set_variable(&p0, p0_values.clone().into());
        cg.set_variable(&p1, p1_values.clone().into());

        let mut x_count = 0;
        let mut total_loss = 0.0;
        for x_inp in input_range.into_iter() {
            x_count += 1;
            cg.reset_primals_keep_variables();
            // TODO Represent vector of x as single x, not a vector.
            cg.reset_variable(
                &x,
                DMatrixF32::new(na::DMatrix::from_element(n_params, 1, x_inp)),
            );
            cg.reset_variable(
                &t,
                DMatrixF32::new(na::DMatrix::from_element(n_params, 1, target_poly(x_inp))),
            );
            total_loss += cg.forward(&loss).m().mean();
            cg.backward(&loss);
        }
        total_loss /= x_count as f32;
        let p0_adjoins = cg.adjoin(&p0);
        let p1_adjoins = cg.adjoin(&p1);
        p0_values = p0_values - (p0_adjoins.m() * learning_rate);
        p1_values = p1_values - (p1_adjoins.m() * learning_rate);
        println!("total_loss {}", total_loss);
    }

    let mut f2 = |x_inp: f32| {
        cg.reset_primals_keep_variables();
        cg.reset_variable(
            &x,
            DMatrixF32::new(na::DMatrix::from_element(n_params, 1, x_inp)),
        );
        cg.forward(&y).m().sum()
    };
    assert_functions_similar(
        target_poly,
        &mut f2,
        &[
            Opts::TestName("test_na_gradient_descent_sin"),
            Opts::InputRange(input_range),
        ],
    );
}

fn new_eb() -> ExprBuilder<DMatrixF32, NaOperAry1, NaOperAry2> {
    ExprBuilder::new()
}
