mod utils;
use nalgebra as na;
use rs_autograd::{
    compute::ComputGraph,
    core_syntax::ExprBuilder,
    na2::{
        calculator::MatrixCalculator,
        syntax::{MatrixF32, NaOperAry1, NaOperAry2},
    },
};
use utils::{assert_function_and_derivative_similar, FloatRange, Opts};

#[test]
fn sum_relu_func() {
    let input_range = FloatRange::new(-2.0, 6.0, 0.1);
    let f = |x: f32| {
        if x < 0.0 {
            0.0
        } else if x <= 2.0 {
            x
        } else if x <= 4.0 {
            2.0
        } else {
            10.0 - 2.0 * x
        }
    };

    let eb = new_eb();
    let x = eb.new_variable("x");
    let p0 = eb.new_variable("p0");
    let p1 = eb.new_variable("p1");
    let y = (p1 * (x - p0).relu()).sum();
    //let y = (x - p0).relu().sum();
    let [x, y, p0, p1] = [x, y, p0, p1].map(|p| p.ident());
    let mut cg = new_cb(eb);
    let mut df = |x_inp: f32| {
        cg.reset();
        dbg!(x_inp);
        cg.set_variable(&x, MatrixF32::V(x_inp));

        cg.set_variable(
            &p0,
            MatrixF32::new_m(na::DMatrix::from_vec(3, 1, vec![0.0, 2.0, 4.0])),
        );
        cg.set_variable(
            &p1,
            MatrixF32::new_m(na::DMatrix::from_vec(3, 1, vec![1.0, -1.0, -2.0])),
        );

        cg.forward(&y);
        cg.backward(&y);

        dbg!(cg.adjoin(&y));
        //dbg!(cg.adjoin(&v0));
        //dbg!(cg.adjoin(&v1));
        dbg!(cg.adjoin(&x));

        // TODO explain this sum here
        let ad = cg.adjoin(&x).m().unwrap().sum();
        //let ad = cg.adjoin(&x).v().unwrap();
        dbg!("final adjoin", ad);
        ad
    };

    assert_function_and_derivative_similar(
        f,
        &mut df,
        &[
            Opts::InputRange(input_range),
            Opts::TestName("sum_relu_func"),
        ],
    );

    fn new_eb() -> ExprBuilder<MatrixF32, NaOperAry1, NaOperAry2> {
        ExprBuilder::new()
    }

    fn new_cb<'a>(
        eb: ExprBuilder<MatrixF32, NaOperAry1, NaOperAry2>,
    ) -> ComputGraph<'a, MatrixF32, NaOperAry1, NaOperAry2> {
        ComputGraph::<MatrixF32, NaOperAry1, NaOperAry2>::new(eb, &MatrixCalculator)
    }
}
