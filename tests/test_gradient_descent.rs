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

#[ignore]
#[test]
fn test_gradient_descent_polynomial() {
    //let eb = new_eb();
    //let x = eb.new_variable("x");
    //let c1 = eb.new_variable("c1");
    //let c2 = eb.new_variable("c2");
    //let c3 = eb.new_variable("c2");

    //let x = x.ident();
    //let c = c.ident();
    //let y = y.ident();
    //let mut cg = ComputGraph::<f32, _, _>::new(eb, &FloatCalculator);
    //let mut df = |x_inp: f32| {
    //    cg.reset();
    //    cg.set_variable(&x, x_inp);
    //    cg.set_variable(&c, 2.0);
    //    cg.forward(&y);
    //    cg.backward(&y);
    //    cg.adjoin(&x);
    //    0.0
    //};

    //assert_functions_similar(
    //    |x| (x + 3.0) * (x - 2.0).powf(2.0) * (x + 1.0).powf(3.0),
    //    &mut df,
    //    &[
    //        Opts::Step(0.01),
    //        Opts::Start(-3.0),
    //        Opts::End(3.0),
    //        Opts::TestName("test_gradient_descent_polynomial"),
    //    ],
    //);
}

fn new_eb() -> ExprBuilder<f32, FloatOperAry1, FloatOperAry2> {
    ExprBuilder::new()
}
