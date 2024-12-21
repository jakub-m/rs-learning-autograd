use rs_autograd::{
    compute::ComputGraph,
    core_syntax::ExprBuilder,
    float::{
        calculator::FloatCalculator,
        syntax::{FloatOperAry1, FloatOperAry2},
    },
};

mod utils;
use utils::assert_functions_similar;

#[test]
fn compare_sin_cos() {
    let mut df = |x: f32| x.cos();
    assert_functions_similar(|x| x.sin(), &mut df, 0.01, 3.14 / 2.0, "compare_sin");
}

#[ignore]
#[test]
fn compare_simple_adjoin() {
    let eb = new_eb();
    let x1 = eb.new_variable("x1");
    let x2 = eb.new_variable("x2");
    let y = x1 * x2;

    let x1 = &x1.ident();
    let x2 = &x2.ident();
    let y = &y.ident();
    let mut cg = ComputGraph::<f32, _, _>::new(eb, &FloatCalculator);
    let mut df = |x: f32| {
        cg.reset();
        cg.set_variable(&x1, x);
        cg.set_variable(&x2, 0.3);
        cg.forward(&y);
        cg.backward(&y);
        cg.adjoin(x1)
    };
    assert_functions_similar(
        |x| x.sin(),
        &mut df,
        0.01,
        3.14 / 2.0,
        "compare_simple_adjoin",
    );
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
        0.01,
        3.14 / 2.0,
        "sin_cos",
    );
}

/// Example from https://huggingface.co/blog/andmholm/what-is-automatic-differentiation
#[test]
fn test_hf() {
    let eb = new_eb();
    let x1 = eb.new_variable("x1");
    let x2 = eb.new_variable("x2");
    let vm1 = x1;
    let v0 = x1;
    let v1 = vm1 * v0;
    let v2 = vm1.ln();
    let v3 = v0 + v1;
    let v4 = v3 - v2;
    let y = v4;
}

fn new_eb() -> ExprBuilder<FloatOperAry1, FloatOperAry2> {
    ExprBuilder::<FloatOperAry1, FloatOperAry2>::new()
}
