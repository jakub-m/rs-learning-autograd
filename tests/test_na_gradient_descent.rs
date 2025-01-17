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

    let n_params: usize = 30;
    let mut rng = StdRng::seed_from_u64(42);
    let mut get_init_param = || {
        let arr = nd::ArrayD::from_shape_fn(sh((n_params, 1)), |_| rng.gen_range(-3.0..3.0));
        MatrixF32::new_m(arr)
    };

    let eb = new_eb();
    let x = eb.new_variable("x");
    let p0 = eb.new_named_parameter("p0", get_init_param());
    let p1 = eb.new_named_parameter("p1", get_init_param());
    let y = ((x - p0).relu() * p1).sum();
    let t = eb.new_variable("t");
    let loss = (y - t).powi(2);

    let [x, y, t, loss] = [x, y, t, loss].map(|p| p.ident);
    let mut cg = ComputGraph::<MatrixF32, _, _>::new(eb, &MatrixCalculator);

    let learning_rate: f32 = 0.01;
    let n_epochs = 1000;

    for _ in 0..n_epochs {
        cg.reset_state_for_next_epoch();
        let mut x_count = 0;
        let mut total_loss = 0.0;
        for x_inp in input_range.into_iter() {
            x_count += 1;
            cg.reset_state_for_next_input();
            cg.reset_primal_of_variable(&x, MatrixF32::V(x_inp));
            cg.reset_primal_of_variable(&t, MatrixF32::V(target_poly(x_inp)));
            total_loss += cg.forward(&loss).v().unwrap();
            cg.backward(&loss);
        }
        total_loss /= x_count as f32;
        cg.update_params_lr(learning_rate);
        println!("total_loss {}", total_loss);
    }

    let mut f2 = |x_inp: f32| {
        cg.reset_state_for_next_input();
        cg.reset_primal_of_variable(&x, MatrixF32::V(x_inp));
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
            Opts::MaxRms(0.11),
        ],
    );
}

fn new_eb() -> ExprBuilder<MatrixF32, NaOperAry1, NaOperAry2> {
    ExprBuilder::new()
}

fn sh((a, b): (usize, usize)) -> nd::IxDyn {
    nd::IxDyn(&[a, b])
}
