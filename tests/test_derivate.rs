use rs_autograd::{
    compute::ComputGraph,
    core_syntax::ExprBuilder,
    float::{calculator::FloatCalculator, syntax::FloatOper},
};

mod utils;
use utils::assert_functions_similar;

#[test]
fn compare_sin_cos() {
    let mut df = |x: f32| x.cos();
    assert_functions_similar(|x| x.sin(), &mut df, 100, 0.01, "compare_sin");
}

#[test]
fn compare_simple_adjoin() {
    let eb = ExprBuilder::new();
    let x1 = eb.new_variable("x1");
    let x2 = eb.new_variable("x2");
    let y = x1 * x2;

    let x1 = &x1.ident();
    let x2 = &x2.ident();
    let y = &y.ident();
    let mut cg = ComputGraph::<f32, FloatOper>::new(eb, &FloatCalculator);
    let mut df = |x: f32| {
        cg.reset();
        cg.set_variable(&x1, x);
        cg.set_variable(&x2, 0.3);
        cg.forward(&y);
        cg.backward(&y);

        cg.adjoin(x1)
    };
    assert_functions_similar(|x| x.sin(), &mut df, 100, 0.01, "compare_simple_adjoin");
}
