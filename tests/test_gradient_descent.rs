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
        cg.reset_primals_keep_variables();
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
        cg.reset_primals_keep_variables();
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
        let x = x - 2.0;
        3.0 * (if x > 0.0 { x } else { 0.0 })
    };

    let eb = new_eb();

    let x = eb.new_variable("x");
    let params = [eb.new_variable("a"), eb.new_variable("b")];
    let y = (x - params[0]).relu() * params[1];
    let t = eb.new_variable("t");
    let loss = (y - t).powi(2);

    let input_range = FloatRange::new(-2.0, 6.0, 0.1);
    let [x, y, t, loss] = [x, y, t, loss].map(|p| p.ident());
    let params = params.map(|p| p.ident());
    let mut cg = ComputGraph::<f32, _, _>::new(eb, &FloatCalculator);
    let mut param_values = params.map(|_| 0.0_f32);

    for i in 0..param_values.len() {
        param_values[i] = 0.01 * (i as f32);
    }

    let n_epochs = 50;
    let learn_rate = 0.1;

    for i in 0..n_epochs {
        cg.reset();
        print!("epoch {}", i);
        print!("\tparams {:?}", param_values);
        // Reset state of primals and adjoins.
        for i in 0..params.len() {
            cg.reset_variable(&params[i], param_values[i]);
        }

        let mut tot_loss = 0_f32;
        let mut n = 0.0_f32;
        for x_inp in input_range.into_iter() {
            n += 1.0;
            cg.reset_primals_keep_variables();
            cg.reset_variable(&x, x_inp);
            cg.reset_variable(&t, target_poly(x_inp));
            tot_loss += cg.forward(&loss);
            cg.backward(&loss);
        }

        let adjoins = params.map(|p| cg.adjoin(&p) / n);
        tot_loss = tot_loss / n;
        print!("\tadjoins {:?}", adjoins);
        print!("\ttot_loss {}", tot_loss);
        println!("");

        for i in 0..param_values.len() {
            param_values[i] = param_values[i] - (learn_rate * adjoins[i]);
        }
    }

    let mut df = |x_inp: f32| {
        cg.reset_primals_keep_variables();
        cg.reset_variable(&x, x_inp);
        cg.forward(&y)
    };
    assert_functions_similar(
        target_poly,
        &mut df,
        &[
            Opts::InputRange(input_range),
            Opts::TestName("test_fit_simple_relu.csv"),
        ],
    );
}

// #[ignore]
#[test]
fn test_relu_to_sin() {
    let target_poly = |x: f32| x.sin();
    let input_range = FloatRange::new(-3.1, 3.2, 0.1);

    let eb = new_eb();
    let x = eb.new_variable("x");

    let params: Vec<Expr<'_, f32, FloatOperAry1, FloatOperAry2>> = (0..100)
        .map(|i| eb.new_variable(format!("b{}", i).as_str()))
        .collect();

    let mut y = (0.0_f32).as_const(&eb);
    for i in 0..(params.len() / 2) {
        let p0 = params[2 * i];
        let p1 = params[2 * i + 1];
        y = y + (x - p0).relu() * p1;
    }
    let t = eb.new_variable("t");
    let loss = (y - t).powi(2);

    //assert_eq!(
    //    "(((x - b1) * pow2((x - b2))) * pow3((x - b3)))",
    //    format!("{}", y)
    //);

    let [x, t, y, loss] = [x, t, y, loss].map(|expr| expr.ident());
    let params: Vec<Ident> = params.iter().map(|expr| expr.ident()).collect();
    let mut cg = ComputGraph::<f32, _, _>::new(eb, &FloatCalculator);

    let mut rng = StdRng::seed_from_u64(42);
    let mut param_values: Vec<f32> = (0..params.len())
        .map(|_| rng.gen_range(-3.0..3.0))
        .collect();
    // dbg!(&param_values);
    let n_epochs = 10000;
    let learn_rate = 0.0001;
    for i in 0..n_epochs {
        print!("i {}", i);
        //print!("\tparams {:?}", param_values);
        cg.reset();
        for i in 0..param_values.len() {
            cg.reset_variable(&params[i], param_values[i]);
        }

        let mut n_steps = 0_usize;
        let mut tot_loss = 0_f32;
        for x_inp in input_range.into_iter() {
            n_steps += 1;
            cg.reset_primals_keep_variables();
            cg.reset_variable(&x, x_inp);
            cg.reset_variable(&t, target_poly(x_inp));
            tot_loss += cg.forward(&loss);
            cg.backward(&loss);
        }

        let adjoins: Vec<f32> = params
            .iter()
            .map(|p| cg.adjoin(&p) / (n_steps as f32))
            .collect();

        tot_loss = tot_loss / (n_steps as f32);
        for i in 0..param_values.len() {
            param_values[i] = param_values[i] - adjoins[i] * learn_rate;
        }
        //println!("\tadjoins{:?}", adjoins);
        //println!("\tparam_values {:?}", param_values);
        println!("\ttot_loss {}", tot_loss);
        println!("");
    }

    let mut df = |x_inp: f32| {
        cg.reset_primals_keep_variables();
        cg.reset_variable(&x, x_inp);
        cg.forward(&y)
    };
    assert_functions_similar(
        target_poly,
        &mut df,
        &[
            Opts::InputRange(input_range),
            Opts::TestName("test_relu_to_sin.csv"),
        ],
    );
}

fn new_eb() -> ExprBuilder<f32, FloatOperAry1, FloatOperAry2> {
    ExprBuilder::new()
}
