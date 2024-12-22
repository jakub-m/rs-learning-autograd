use rs_autograd::{
    compute::ComputGraph,
    core_syntax::ExprBuilder,
    float::{
        calculator::FloatCalculator,
        syntax::{AsConst, FloatOperAry1, FloatOperAry2},
    },
};

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

/// Fit a polynomial using gradient descent.
#[test]
fn test_gradient_descent_polynomial() {
    // The known polynomial.

    let poly = |x: f32| (x + 3.0) * (x - 2.0).powf(2.0) * (x + 1.0).powf(3.0);
    let eb = new_eb();
    let x = eb.new_variable("x");
    let b1 = eb.new_variable("b1");
    let b2 = eb.new_variable("b2");
    let b3 = eb.new_variable("b3");
    let y = (x - b1) * (x - b2).pow(2.0.as_const(&eb)) * (x - b3).pow(3.0.as_const(&eb));

    let [x, b1, b2, b3, y] = [x, b1, b2, b3, y].map(|expr| expr.ident());
    let mut cg = ComputGraph::<f32, _, _>::new(eb, &FloatCalculator);
    let mut df = |x_inp: f32| {
        cg.reset();
        cg.set_variable(&x, x_inp);
        cg.set_variable(&b1, -3.0);
        cg.set_variable(&b2, 2.0);
        cg.set_variable(&b3, -1.0);
        cg.forward(&y);
        cg.backward(&y);
        cg.adjoin(&x)
    };

    assert_functions_similar(
        poly,
        &mut df,
        &[
            Opts::Step(0.01),
            Opts::Start(-3.0),
            Opts::End(3.0),
            Opts::TestName("test_gradient_descent_polynomial"),
            Opts::MaxRms(2.0),
        ],
    );
}

fn new_eb() -> ExprBuilder<f32, FloatOperAry1, FloatOperAry2> {
    ExprBuilder::new()
}
