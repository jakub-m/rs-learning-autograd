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
use utils::{assert_functions_similar, iter_float, FloatRange, Opts};

#[test]
fn test_gradient_descent_simple() {
    //    // The known polynomial.
    //    let poly = |x: f32| 3.14 * x;
    //    let start = -1.0;
    //    let end = 1.0;
    //    let step = 0.1;
    //
    //    // Define the model.
    //    let eb = new_eb();
    //    let x = eb.new_variable("x");
    //    let a = eb.new_variable("a");
    //    // "y" is the "model" that we will fit to the polynomial.
    //    let y = a * x;
    //    // t is the target value
    //    let t = eb.new_variable("t");
    //    let loss = (y - t).powi(2);
    //
    //    assert_eq!(
    //        "(((x - b1) * ((x - b2)^2)) * ((x - b3)^3))",
    //        format!("{}", y)
    //    );
    //
    //    let [x, a, t, y, loss] = [x, a, t, y, loss].map(|expr| expr.ident());
    //    let mut cg = ComputGraph::<f32, _, _>::new(eb, &FloatCalculator);
    //
    //    // Set initial parameter values (some "random" values).
    //    let mut param_value_a = 0.1;
    //    let n_epochs = 100;
    //    let learn_rate = 0.01;
    //    for i in 0..n_epochs {
    //        print!("epoch {}", i);
    //        print!("\tparams {:?}", param_value_a);
    //        // Reset state of primals and adjoins.
    //        cg.reset();
    //        // Set equation ("model") parameters.
    //        cg.reset_variable(&a, param_value_a);
    //        // Set input value (x) and target y for that input.
    //        let mut n_steps = 0_usize;
    //        let mut tot_loss = 0_f32;
    //        for x_inp in iter_float(start, end, step) {
    //            n_steps += 1;
    //            cg.reset_variable(&x, x_inp);
    //            // The target is the ideal t = ax;
    //            cg.reset_variable(&t, poly(x_inp));
    //            // Run forward and backward pass.
    //            tot_loss += cg.forward(&loss);
    //            cg.backward(&loss);
    //        }
    //        // Take adjoins per parameter and apply the gradient to input parameters.
    //        let b_adjoins = b_params.map(|b| cg.adjoin(&b));
    //        print!("\tadjoins{:?}", b_adjoins);
    //        print!("\ttot_loss {}", tot_loss / (n_steps as f32));
    //        for i in 0..param_values.len() {
    //            let d = b_adjoins[i] * learn_rate / (n_steps as f32);
    //            param_values[i] = param_values[i] - d;
    //        }
    //        println!("");
    //    }
    //
    //    println!("final params {:?}", param_values);
    //    let mut df = |x_inp: f32| {
    //        cg.reset();
    //        for i in 0..param_values.len() {
    //            cg.set_variable(&b_params[i], param_values[i]);
    //        }
    //        cg.set_variable(&x, x_inp);
    //        cg.forward(&y)
    //    };
    //    assert_functions_similar(
    //        poly,
    //        &mut df,
    //        &[
    //            Opts::Start(start),
    //            Opts::End(end),
    //            Opts::Step(step),
    //            Opts::TestName("test_gradient_descent_polynomial"),
    //        ],
    //    );
}

/// Fit a polynomial using gradient descent.
#[ignore]
#[test]
fn test_gradient_descent_polynomial() {
    // The known polynomial.
    let poly = |x: f32| (x - 2.0) * (x + 1.0).powf(2.0) * (x - 1.0).powf(3.0);
    let input_range = FloatRange::new(-0.1, 2.0, 0.1);

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
    let mut param_values = [0.0, 0.0, 0.0];
    // let mut param_values = [2.0, -1.0, 1.0]; // target
    let n_epochs = 100;
    let learn_rate = 0.01;
    for i in 0..n_epochs {
        print!("epoch {}", i);
        print!("\tparams {:?}", param_values);
        // Reset state of primals and adjoins.
        cg.reset();
        // Set equation ("model") parameters.
        for i in 0..param_values.len() {
            cg.reset_variable(&b_params[i], param_values[i]);
        }
        // Set input value (x) and target y for that input.
        let mut n_steps = 0_usize;
        let mut tot_loss = 0_f32;
        for x_inp in input_range.into_iter() {
            n_steps += 1;
            cg.reset_variable(&x, x_inp);
            cg.reset_variable(&t, poly(x_inp));
            // Run forward and backward pass.
            tot_loss += cg.forward(&loss);
            cg.backward(&loss);
        }

        // Take adjoins per parameter and apply the gradient to input parameters.
        let b_adjoins = b_params.map(|b| cg.adjoin(&b));
        print!("\tadjoins{:?}", b_adjoins);
        print!("\ttot_loss {}", tot_loss / (n_steps as f32));
        for i in 0..param_values.len() {
            let d = b_adjoins[i] * learn_rate / (n_steps as f32);
            param_values[i] = param_values[i] - d;
        }
        println!("");
    }

    println!("final params {:?}", param_values);
    let mut df = |x_inp: f32| {
        cg.reset();
        for i in 0..param_values.len() {
            cg.set_variable(&b_params[i], param_values[i]);
        }
        cg.set_variable(&x, x_inp);
        cg.forward(&y)
    };
    //assert_functions_similar(
    //    poly,
    //    &mut df,
    //    &[
    //        Opts::Start(start),
    //        Opts::End(end),
    //        Opts::Step(step),
    //        Opts::TestName("test_gradient_descent_polynomial"),
    //    ],
    //);
}

fn new_eb() -> ExprBuilder<f32, FloatOperAry1, FloatOperAry2> {
    ExprBuilder::new()
}
