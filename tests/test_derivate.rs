use rs_autograd::{
    compute::ComputGraph,
    core_syntax::ExprBuilder,
    float::{
        calculator::FloatCalculator,
        syntax::{FloatOperAry1, FloatOperAry2},
    },
};

use approx_eq::assert_approx_eq;

mod utils;
use utils::{assert_functions_similar, Opts};

#[test]
fn compare_sin_cos() {
    let mut df = |x: f32| x.cos();
    assert_functions_similar(
        |x| x.sin(),
        &mut df,
        &[
            Opts::TestName("compare_sin"),
            Opts::Step(0.01),
            Opts::End(3.14 / 2.0),
        ],
    );
}

/// Example from https://huggingface.co/blog/andmholm/what-is-automatic-differentiation
#[test]
fn test_hf() {
    let eb = new_eb();
    let x1 = eb.new_variable("x1");
    let x2 = eb.new_variable("x2");
    let vm1 = x1;
    let v0 = x2;
    let v1 = vm1 * v0;
    let v2 = vm1.ln();
    let v3 = v0 + v1;
    let v4 = v3 - v2;
    let y = v4;

    // Compute.
    let x1 = x1.ident();
    let x2 = x2.ident();
    let v0 = v0.ident();
    let v1 = v1.ident();
    let v2 = v2.ident();
    let v3 = v3.ident();
    let v4 = v4.ident();
    let y = y.ident();
    let mut cg = ComputGraph::<f32, _, _>::new(eb, &FloatCalculator);
    cg.set_variable(&x1, 3.0);
    cg.set_variable(&x2, -4.0);

    // Forward.
    let eps = 0.01;
    cg.forward(&y);
    assert_approx_eq!(cg.primal(&y) as f64, -17.10, eps);

    // Backward.
    cg.backward(&y);
    assert_approx_eq!(cg.adjoin(&y) as f64, 1.0, eps);
    assert_approx_eq!(cg.adjoin(&v4) as f64, 1.0, eps);
    assert_approx_eq!(cg.adjoin(&v3) as f64, 1.0, eps);
    assert_approx_eq!(cg.adjoin(&v2) as f64, -1.0, eps);
    assert_approx_eq!(cg.adjoin(&v1) as f64, 1.0, eps);
    assert_approx_eq!(cg.adjoin(&x2) as f64, 4.0, eps);
    assert_approx_eq!(cg.adjoin(&x1) as f64, -4.33, eps);
}

#[test]
fn sin_cos() {
    let eb = new_eb();
    let x = eb.new_variable("x");
    // c is constant, but as of writing this code constants are not supported directly with, say, `x * 0.01`.
    let c = eb.new_variable("c");
    let y = (x * c).sin() * x.cos();
    assert_eq!("(sin((x * c)) * cos(x))", format!("{}", y));

    // Compute.
    let constant = 30.0;
    let x = x.ident();
    let c = c.ident();
    let y = y.ident();
    let mut cg = ComputGraph::<f32, _, _>::new(eb, &FloatCalculator);
    let mut df = |x_inp: f32| {
        cg.reset();
        cg.set_variable(&x, x_inp);
        cg.set_variable(&c, constant);
        cg.forward(&y);
        cg.backward(&y);
        cg.adjoin(&x)
    };
    assert_functions_similar(
        |x| (x * constant).sin() * (x.cos()),
        &mut df,
        &[
            Opts::Step(0.001),
            Opts::End(3.14 / 1.0),
            Opts::TestName("sin_cos"),
            Opts::MaxRms(0.02),
        ],
    );
}
fn new_eb() -> ExprBuilder<FloatOperAry1, FloatOperAry2> {
    ExprBuilder::<FloatOperAry1, FloatOperAry2>::new()
}
