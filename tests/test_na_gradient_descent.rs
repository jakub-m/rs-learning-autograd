mod utils;
use ndarray as nd;
use rand::{self, rngs::StdRng};
use rand::{Rng, SeedableRng};
use rs_autograd::{
    compute::ComputGraph,
    core_syntax::ExprBuilder,
    nar::{
        calculator::MatrixCalculator,
        syntax::{MatrixF32, NaOperAry1, NaOperAry2},
    },
};
use utils::{assert_functions_similar, FloatRange, Opts};

/// This test with 20x2 parameters, learning rate 0.0001 and epochs 10000 takes ~20 seconds.
#[test]
fn test_na_gradient_descent_sin() {
    let target_poly = |x: f32| x.sin();
    let input_range = FloatRange::new(-3.1, 3.2, 0.1);

    let eb = new_eb();
    let x = eb.new_variable("x");
    let p0 = eb.new_variable("p0");
    let p1 = eb.new_variable("p1");
    let y = ((x - p0).relu() * p1).sum();
    let t = eb.new_variable("t");
    let loss = (y - t).powi(2);

    let [x, p0, p1, y, t, loss] = [x, p0, p1, y, t, loss].map(|p| p.ident());
    let mut cg = ComputGraph::<MatrixF32, _, _>::new(eb, &MatrixCalculator);

    let n_params: usize = 30;
    let learning_rate: f32 = 0.0001;
    let n_epochs = 1000;

    let mut rng = StdRng::seed_from_u64(42);
    cg.set_parameter(
        &p0,
        nd::ArrayD::from_shape_fn(sh((n_params, 1)), |_| rng.gen_range(-3.0..3.0)).into(),
    );
    cg.set_parameter(
        &p1,
        nd::ArrayD::from_shape_fn(sh((n_params, 1)), |_| rng.gen_range(-3.0..3.0)).into(),
    );

    for _ in 0..n_epochs {
        cg.reset();
        let mut x_count = 0;
        let mut total_loss = 0.0;
        for x_inp in input_range.into_iter() {
            x_count += 1;
            cg.reset_primals_keep_variables();
            cg.reset_variable(&x, MatrixF32::V(x_inp));
            cg.reset_variable(&t, MatrixF32::V(target_poly(x_inp)));
            total_loss += cg.forward(&loss).v().unwrap();
            cg.backward(&loss);
        }
        total_loss /= x_count as f32;
        cg.update_params_lr(learning_rate);
        println!("total_loss {}", total_loss);
    }

    let mut f2 = |x_inp: f32| {
        cg.reset_primals_keep_variables();
        cg.reset_variable(&x, MatrixF32::V(x_inp));
        cg.forward(&y).v().unwrap()
    };
    assert_functions_similar(
        target_poly,
        &mut f2,
        &[
            Opts::TestName(
                format!(
                    "test_na_gradient_descent_sin__n_params_{}_lr_{}_epochs_{}",
                    n_params * 2,
                    learning_rate,
                    n_epochs
                )
                .as_str(),
            ),
            Opts::InputRange(input_range),
            Opts::MaxRms(0.1),
        ],
    );
}

fn new_eb() -> ExprBuilder<MatrixF32, NaOperAry1, NaOperAry2> {
    ExprBuilder::new()
}

fn sh((a, b): (usize, usize)) -> nd::IxDyn {
    nd::IxDyn(&[a, b])
}
