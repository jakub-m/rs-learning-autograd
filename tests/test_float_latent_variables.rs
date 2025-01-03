mod table;
mod utils;

use rand::{self, rngs::StdRng};
use rand::{Rng, SeedableRng};
use rs_autograd::{
    compute::ComputGraph,
    core_syntax::{Expr, ExprBuilder, Ident},
    float::{
        calculator::FloatCalculator,
        syntax::{AsConst, FloatOperAry1, FloatOperAry2},
    },
};
use utils::{assert_functions_similar, FloatRange, Opts};

#[test]
fn test_latent_lin_reg() {
    let target_poly = |x: f32| x * 3.0 - 5.0;

    let eb = new_eb();

    let x = eb.new_variable("x");
    let y = x.linreg();
    let t = eb.new_variable("t");
    let loss = (y - t).powi(2);
    let [x, y, t, loss] = [x, y, t, loss].map(|p| p.ident);
    let mut cg = ComputGraph::<f32, _, _>::new(eb, &FloatCalculator);

    let n_epochs = 50;
    let learn_rate = 0.1;

    let input_range = FloatRange::new(0.0, 10.0, 1.0);
    let mut rng = StdRng::seed_from_u64(42);
    let x_vec: Vec<f32> = input_range
        .into_iter()
        .map(|x| target_poly(x) + rng.gen_range(-0.2..0.2))
        .collect();

    for i in 0..n_epochs {
        cg.reset_state_for_next_epoch();
        print!("epoch {}", i);
        let mut tot_loss = 0_f32;
        let mut n = 0.0_f32;
        for x_inp in &x_vec {
            n += 1.0;
            cg.reset_state_for_next_input();
            cg.reset_primal_of_variable(&x, *x_inp);
            cg.reset_primal_of_variable(&t, target_poly(*x_inp));
            tot_loss += cg.forward(&loss);
            cg.backward(&loss);
        }

        tot_loss = tot_loss / n;
        print!("\ttot_loss {}", tot_loss);
        println!("");
        cg.update_params_lr(learn_rate);
    }

    //let mut df = |x_inp: f32| {
    //    cg.reset_state_for_next_input();
    //    cg.reset_primal_of_variable(&x, x_inp);
    //    cg.forward(&y)
    //};
    //assert_functions_similar(
    //    target_poly,
    //    &mut df,
    //    &[
    //        Opts::InputRange(input_range),
    //        Opts::TestName("test_fit_simple_relu.csv"),
    //    ],
    //);
}

fn new_eb() -> ExprBuilder<f32, FloatOperAry1, FloatOperAry2> {
    ExprBuilder::new()
}
