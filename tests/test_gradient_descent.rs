use std::ops;

use rs_autograd::{
    compute::ComputGraph,
    core_syntax::ExprBuilder,
    float::{
        calculator::FloatCalculator,
        syntax::{AsConst, FloatOperAry1, FloatOperAry2},
    },
};

mod utils;
use utils::{assert_functions_similar, iter_float, Opts};

/// Fit a polynomial using gradient descent.
#[test]
fn test_gradient_descent_polynomial() {
    // The known polynomial.
    let poly = |x: f32| (x + 3.0) * (x - 2.0).powf(2.0) * (x + 1.0).powf(3.0);

    // Define the model.
    let eb = new_eb();
    let x = eb.new_variable("x");
    let b1 = eb.new_variable("b1");
    let b2 = eb.new_variable("b2");
    let b3 = eb.new_variable("b3");
    // "y" is the "model" that we will fit to the polynomial.
    let y = (x - b1) * (x - b2).pow(2.0.as_const(&eb)) * (x - b3).pow(3.0.as_const(&eb));
    // t is the target value
    let t = eb.new_variable("t");
    let loss = (y - t).powi(2);

    assert_eq!(
        "(((x - b1) * ((x - b2)^2)) * ((x - b3)^3))",
        format!("{}", y)
    );

    let [x, t, y, loss] = [x, t, y, loss].map(|expr| expr.ident());
    let b_params = [b1, b2, b3].map(|expr| expr.ident());
    let mut cg = ComputGraph::<f32, _, _>::new(eb, &FloatCalculator);

    // Set initial parameter values (some "random" values).
    let mut param_values = [0.2, -0.1, 0.05];
    //let mut param_values = [-3.0, +2.0, -1.0]; // target
    let n_epochs = 1000;
    let learn_rate = 0.000001;
    for _ in 0..n_epochs {
        println!("params {:?}", param_values);
        // Reset state of primals and adjoins.
        cg.reset();
        // Set equation ("model") parameters.
        for i in 0..param_values.len() {
            cg.reset_variable(&b_params[i], param_values[i]);
        }
        // Set input value (x) and target y for that input.
        let mut n_steps = 0_usize;
        let mut tot_loss = 0_f32;
        for x_inp in iter_float(-3.0_f32, 3.0, 0.1) {
            n_steps += 1;
            cg.reset_variable(&x, x_inp);
            cg.reset_variable(&t, poly(x_inp));
            // Run forward and backward pass.
            tot_loss += cg.forward(&loss);
            cg.backward(&loss);
        }
        // Take adjoins per parameter and apply the gradient to input parameters.
        let adjoins = b_params.map(|b| cg.adjoin(&b));
        println!("adjoins{:?}", adjoins);
        println!("tot_loss {}", tot_loss / (n_steps as f32));
        for i in 0..param_values.len() {
            let d = adjoins[i] * learn_rate / (n_steps as f32);
            param_values[i] = param_values[i] - d;
        }
    }

    let mut df = |x_inp: f32| {
        cg.reset_variable(&x, x_inp);
        cg.forward(&y)
    };
    assert_functions_similar(
        poly,
        &mut df,
        &[
            Opts::Start(-3.0),
            Opts::End(3.0),
            Opts::Step(0.1),
            Opts::TestName("test_gradient_descent_polynomial"),
        ],
    );
}

fn new_eb() -> ExprBuilder<f32, FloatOperAry1, FloatOperAry2> {
    ExprBuilder::new()
}
