mod utils;
use ndarray as nd;
use rand::{self, rngs::StdRng};
use rand::{Rng, SeedableRng};
use rs_autograd::{
    compute::ComputGraph,
    core_syntax::ExprBuilder,
    nar::{
        calculator::MatrixCalculator,
        syntax::{MatrixF32, NaOperAry1, NaOperAry2},
    },
};
use utils::{assert_functions_similar, FloatRange, Opts};

#[ignore]
#[test]
fn test_conv2d_5x5_forward() {
    // let input_range = FloatRange::new(-3.1, 3.2, 0.1);

    // let n_params: usize = 30;
    // let mut rng = StdRng::seed_from_u64(42);
    // let mut get_init_param = || {
    //     let arr = nd::ArrayD::from_shape_fn(sh((n_params, 1)), |_| rng.gen_range(-3.0..3.0));
    //     MatrixF32::new_m(arr)
    // };

    let eb = new_eb();
    let x = eb.new_variable("x");
    let k = eb.new_variable("k");
    let y = x.conv2d(k);
    // let p0 = eb.new_named_parameter("p0", get_init_param());
    // let p1 = eb.new_named_parameter("p1", get_init_param());
    // let y = ((x - p0).relu() * p1).sum();
    // let t = eb.new_variable("t");
    // let loss = (y - t).powi(2);

    let [x, y] = [x, y].map(|p| p.ident);
    let mut cg = ComputGraph::<MatrixF32, _, _>::new(eb, &MatrixCalculator);

    cg.reset_primal_of_variable(&x, MatrixF32::new_m(get_m(5, 4)));
    // [[0, 1, 2, 3],
    //  [4, 5, 6, 7],
    //  [8, 9, 10, 11],
    //  [12, 13, 14, 15],
    //  [16, 17, 18, 19]]

    let m_forward = cg.forward(&y);
    panic!("{}", m_forward)

    //         cg.reset_primal_of_variable(&t, MatrixF32::V(target_poly(x_inp)));
    //         total_loss += cg.forward(&loss).v().unwrap();
    //         cg.backward(&loss);
    //     }

    // }
}

fn new_eb() -> ExprBuilder<MatrixF32, NaOperAry1, NaOperAry2> {
    ExprBuilder::new()
}

fn get_m(nrows: usize, ncols: usize) -> nd::ArrayD<f32> {
    nd::ArrayD::from_shape_fn(sh((nrows, ncols)), |ix| {
        let i = ix[0];
        let j = ix[1];
        (i * ncols + j) as f32
    })
}

fn sh((a, b): (usize, usize)) -> nd::IxDyn {
    nd::IxDyn(&[a, b])
}
