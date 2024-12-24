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
use table::Table;
use utils::{assert_functions_similar, FloatRange, Opts};

#[test]
fn test_gradient_descent_simple() {
    // The known polynomial.
    let poly = |x: f32| 3.14 * x;
    let input_range = FloatRange::new(-1.0_f32, 1.0, 0.1);

    // Define the model.
    let eb = new_eb();
    let x = eb.new_variable("x");
    let a = eb.new_variable("a");
    // "y" is the "model" that we will fit to the polynomial.
    let y = a * x;
    // t is the target value
    let t = eb.new_variable("t");
    let loss = (y - t).powi(2);

    assert_eq!("(a * x)", format!("{}", y));

    let [x, a, t, y, loss] = [x, a, t, y, loss].map(|expr| expr.ident());
    let mut cg = ComputGraph::<f32, _, _>::new(eb, &FloatCalculator);

    // Set initial parameter values (some "random" values).
    let mut param_value_a = 0.1;
    let n_epochs = 1000;
    let learn_rate = 0.1;
    for i in 0..n_epochs {
        print!("epoch {}", i);
        print!("\tparams {:?}", param_value_a);
        // Reset state of primals and adjoins.
        cg.reset();
        // Set equation ("model") parameters.
        cg.reset_variable(&a, param_value_a);

        // Set input value (x) and target y for that input.
        let mut tot_loss = 0_f32;
        let mut n = 0.0_f32;
        for x_inp in input_range.into_iter() {
            n += 1.0;
            cg.reset_variable(&x, x_inp);
            // The target is the ideal t = ax;
            cg.reset_variable(&t, poly(x_inp));
            // Run forward and backward pass.
            tot_loss += cg.forward(&loss);
            cg.backward(&loss);
        }

        // Take adjoins per parameter and apply the gradient to input parameters.
        let adjoin_a = cg.adjoin(&a) / n;
        tot_loss = tot_loss / n;
        print!("\tadjoins {}", adjoin_a);
        print!("\ttot_loss {}", tot_loss);

        param_value_a = param_value_a - (learn_rate * adjoin_a);
        println!("");
    } // end of epoch

    let mut df = |x_inp: f32| {
        cg.reset();
        cg.set_variable(&a, param_value_a);
        cg.set_variable(&x, x_inp);
        cg.forward(&y)
    };
    assert_functions_similar(
        poly,
        &mut df,
        &[
            Opts::InputRange(input_range),
            Opts::TestName("test_gradient_descent_simple"),
        ],
    );
}

/// Fit a polynomial using gradient descent.
#[test]
fn test_gradient_descent_polynomial() {
    // The known polynomial.
    let target_poly = |x: f32| (x - 2.0) * (x + 1.0).powf(2.0) * (x - 1.0).powf(3.0);
    let input_range = FloatRange::new(-1.1, 2.2, 0.1);

    let mut table = Table::<f32>::new();
    table.extend_col("x", input_range.into_iter());
    table.extend_col("target", input_range.into_iter().map(target_poly));
    table.assert_columns_have_same_lengths();

    // Define the model.
    let eb = new_eb();
    let x = eb.new_variable("x");
    let b1 = eb.new_variable("b1");
    let b2 = eb.new_variable("b2");
    let b3 = eb.new_variable("b3");
    // "y" is the "model" that we will fit to the polynomial.
    let y = (x - b1) * (x - b2).powi(2) * (x - b3).powi(3);
    // t is the target value
    let t = eb.new_variable("t");
    let loss = (y - t).powi(2);

    assert_eq!(
        "(((x - b1) * pow2((x - b2))) * pow3((x - b3)))",
        format!("{}", y)
    );

    let [x, t, y, loss] = [x, t, y, loss].map(|expr| expr.ident());
    let b_param_ids = [b1, b2, b3].map(|expr| expr.ident());
    let mut cg = ComputGraph::<f32, _, _>::new(eb, &FloatCalculator);

    // Set initial parameter values (some "random" values).
    //let mut b_param_values = [2.0, -1.0, 1.0]; // target
    let mut b_param_values = [0.0, -1.0, 1.0];
    //let mut b_param_values = [0.01_f32, 0.02, 0.03];
    let n_epochs = 100;
    let learn_rate = 0.01;
    for i in 0..n_epochs {
        print!("i {}", i);
        print!("\tb {:?}", b_param_values);
        cg.reset();
        for i in 0..b_param_values.len() {
            cg.reset_variable(&b_param_ids[i], b_param_values[i]);
        }

        let mut n_steps = 0_usize;
        let mut tot_loss = 0_f32;
        for x_inp in input_range.into_iter() {
            n_steps += 1;
            cg.reset_variable(&x, x_inp);
            cg.reset_variable(&t, target_poly(x_inp));
            tot_loss += cg.forward(&loss);
            cg.backward(&loss);
        }

        let b_adjoins = b_param_ids.map(|b| cg.adjoin(&b) / (n_steps as f32));
        tot_loss = tot_loss / (n_steps as f32);
        print!("\tadjoins{:?}", b_adjoins);
        print!("\ttot_loss {}", tot_loss);
        println!("");
        for i in 0..b_param_values.len() {
            b_param_values[i] = b_param_values[i] - b_adjoins[i] * learn_rate;
        }
    }

    let mut df = |x_inp: f32| {
        cg.reset();
        for i in 0..b_param_values.len() {
            cg.set_variable(&b_param_ids[i], b_param_values[i]);
        }
        cg.set_variable(&x, x_inp);
        cg.forward(&y)
    };
    table.extend_col("y_final", input_range.into_iter().map(|x| df(x)));
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
    println!("param_values: {:?}", b_param_values);
    table
        .to_csv("test_gradient_descent_polynomial.csv")
        .unwrap();

    assert!(false);
}

fn new_eb() -> ExprBuilder<f32, FloatOperAry1, FloatOperAry2> {
    ExprBuilder::new()
}
