mod table;
mod utils;

use rs_autograd::{
    compute::ComputGraph,
    core_syntax::ExprBuilder,
    float::{
        calculator::FloatCalculator,
        syntax::{FloatOperAry1, FloatOperAry2},
    },
};
use utils::{assert_functions_similar, FloatRange, Opts};

#[test]
fn test_latent_lin_reg() {
    let target_poly = |x: f32| x * -3.0 + 6.0;

    let eb = new_eb();

    let x = eb.new_variable("x");
    let y = x.linreg();

    assert_eq!("((_1 * x) + _2)", format!("{}", y));
    let t = eb.new_variable("t");
    let loss = (y - t).powi(2);

    let [x, t, y, loss] = [x, t, y, loss].map(|p| p.ident);
    let mut cg = ComputGraph::<f32, _, _>::new(eb, &FloatCalculator);

    let n_epochs = 100;
    let learn_rate = 0.01;

    let input_range = FloatRange::new(0.0, 10.0, 1.0);

    for i in 0..n_epochs {
        cg.reset_state_for_next_epoch();
        print!("epoch {}", i);
        let mut tot_loss = 0_f32;
        let mut n = 0;
        for val_x in input_range.into_iter() {
            n += 1;
            cg.reset_state_for_next_input();
            cg.reset_primal_of_variable(&x, val_x);
            cg.reset_primal_of_variable(&t, target_poly(val_x));
            tot_loss += cg.forward(&loss);
            cg.backward(&loss);
        }

        tot_loss = tot_loss / n as f32;
        print!("\ttot_loss {}", tot_loss);
        println!("");
        cg.update_params_lr(learn_rate);
    }

    let mut df = |x_inp: f32| {
        cg.reset_state_for_next_input();
        cg.reset_primal_of_variable(&x, x_inp);
        cg.forward(&y)
    };
    assert_functions_similar(
        target_poly,
        &mut df,
        &[
            Opts::MaxRms(2.0),
            Opts::InputRange(input_range),
            Opts::TestName("test_latent_lin_reg.csv"),
        ],
    );
}

fn new_eb() -> ExprBuilder<f32, FloatOperAry1, FloatOperAry2> {
    ExprBuilder::new()
}
