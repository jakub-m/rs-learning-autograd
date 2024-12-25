mod table;
mod utils;

use rs_autograd::{
    compute::ComputGraph,
    core_syntax::{Expr, ExprBuilder, Ident},
    float::{
        calculator::FloatCalculator,
        syntax::{AsConst, FloatOperAry1, FloatOperAry2},
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
        cg.reset_keep_variables();
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
        cg.reset_keep_variables();
        cg.reset_variable(&x, x_inp);
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

#[test]
fn test_fit_simple_relu() {
    let target_poly = |x: f32| {
        if x < 3.0 {
            0.0
        } else {
            x * 2.0
        }
    };
    panic!();
}

/// Fit a polynomial using gradient descent.
#[ignore]
#[test]
fn test_gradient_descent_polynomial() {
    // The known polynomial.
    //let target_poly = |x: f32| -1.0 * ((x - 2.0) * (x + 1.0).powf(2.0) * (x - 1.0).powf(3.0));
    let target_poly = |x: f32| x.sin();
    let input_range = FloatRange::new(-3.1, 3.2, 0.1);

    let mut table_xy = Table::<f32>::new();
    table_xy.extend_col("x", input_range.into_iter());
    table_xy.extend_col("target", input_range.into_iter().map(target_poly));
    table_xy.assert_columns_have_same_lengths();

    let mut table_loss = Table::<f32>::new();

    // Define the model.
    let eb = new_eb();
    let x = eb.new_variable("x");
    let b_params: Vec<Expr<'_, f32, FloatOperAry1, FloatOperAry2>> = (1..7)
        .map(|i| eb.new_variable(format!("b{}", i).as_str()))
        .collect();

    // "y" is the "model" that we will fit to the polynomial.
    let mut y = (1.0_f32).as_const(&eb);
    for i in 0..b_params.len() {
        let b = b_params[i];
        y = y * (x - b).powi((i as i32) + 1);
    }
    let t = eb.new_variable("t");
    let loss = (y - t).powi(2);

    //assert_eq!(
    //    "(((x - b1) * pow2((x - b2))) * pow3((x - b3)))",
    //    format!("{}", y)
    //);

    let [x, t, y, loss] = [x, t, y, loss].map(|expr| expr.ident());
    let b_params: Vec<Ident> = b_params.iter().map(|expr| expr.ident()).collect();
    let mut cg = ComputGraph::<f32, _, _>::new(eb, &FloatCalculator);

    let mut b_param_values: Vec<f32> = (0..b_params.len()).map(|i| 0.01_f32 * (i as f32)).collect();
    let n_epochs = 1;
    let learn_rate = 0.0000001;
    //for i in 0..n_epochs {
    //    print!("i {}", i);
    //    print!("\tb {:?}", b_param_values);
    //    cg.reset();
    //    for i in 0..b_param_values.len() {
    //        cg.reset_variable(&b_params[i], b_param_values[i]);
    //    }

    //    let mut n_steps = 0_usize;
    //    let mut tot_loss = 0_f32;
    //    for x_inp in input_range.into_iter() {
    //        n_steps += 1;
    //        cg.reset_variable(&x, x_inp);
    //        cg.reset_variable(&t, target_poly(x_inp));
    //        tot_loss += cg.forward(&loss);
    //        cg.backward(&loss);
    //    }

    //    let b_adjoins: Vec<f32> = b_params
    //        .iter()
    //        .map(|b| cg.adjoin(&b) / (n_steps as f32))
    //        .collect();

    //    tot_loss = tot_loss / (n_steps as f32);
    //    table_loss.append_col("loss", tot_loss);
    //    table_loss.append_col("i", i as f32);
    //    print!("\tadjoins{:?}", b_adjoins);
    //    print!("\ttot_loss {}", tot_loss);
    //    println!("");
    //    for i in 0..b_param_values.len() {
    //        b_param_values[i] = b_param_values[i] - b_adjoins[i] * learn_rate;
    //    }
    //}

    let mut df = |x_inp: f32| {
        cg.reset_keep_variables();
        for i in 0..b_param_values.len() {
            cg.set_variable(&b_params.get(i).unwrap(), b_param_values[i]);
        }
        cg.set_variable(&x, x_inp);
        cg.forward(&y)
    };
    table_xy.extend_col("y_final", input_range.into_iter().map(|x| df(x)));
    println!("param_values: {:?}", b_param_values);
    table_xy
        .to_csv("test_gradient_descent_polynomial.csv")
        .unwrap();
    //table_loss
    //    .to_csv("test_gradient_descent_polynomial_loss.csv")
    //    .unwrap();

    assert!(false);
}

fn new_eb() -> ExprBuilder<f32, FloatOperAry1, FloatOperAry2> {
    ExprBuilder::new()
}
