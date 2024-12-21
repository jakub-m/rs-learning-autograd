use std::{fs::File, io::Write};

use rs_autograd::{
    compute::ComputGraph,
    core_syntax::ExprBuilder,
    float::{calculator::FloatCalculator, syntax::FloatOper},
};

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

// TODO move elsewhere
fn assert_functions_similar<F1, F2>(f: F1, df: &mut F2, n: usize, step: f32, test_name: &str)
where
    F1: Fn(f32) -> f32,
    F2: FnMut(f32) -> f32,
{
    let mut y1_values: Vec<f32> = vec![];
    let mut y2_values: Vec<f32> = vec![];
    let mut x_values: Vec<f32> = vec![];
    let mut y2 = f(0.0);
    for x in (0..n).map(|f| (f as f32) * step) {
        x_values.push(x);
        let y1 = f(x);
        y1_values.push(y1);

        // Push first calculate later so it behaves well at the first point.
        y2_values.push(y2);
        let dy2 = df(x);
        y2 = y2 + dy2 * step;
    }

    let mut failure_message: Option<String> = None;
    for i in 0..n {
        let diff = ((y1_values[i] - y2_values[i]) / y1_values[i]).abs();
        let th = 0.01;
        if diff >= th {
            failure_message.replace(format!(
                "Difference between the functions at sample {} was {} >= {}",
                i, diff, th
            ));
            break;
        }
    }
    if let Some(m) = failure_message {
        write_series_to_file(
            format!("{}.csv", test_name).as_str(),
            &x_values,
            &y1_values,
            &y2_values,
        );
        panic!("{}", m);
    }
}

fn write_series_to_file(
    file_name: &str,
    x_values: &Vec<f32>,
    y1_values: &Vec<f32>,
    y2_values: &Vec<f32>,
) {
    let mut f = File::create(file_name).unwrap();
    f.write(format!("i\tx\ty1\ty2\n").as_bytes()).unwrap();
    for i in 0..x_values.len() {
        f.write(
            format!(
                "{}\t{}\t{}\t{}\n",
                i, x_values[i], y1_values[i], y2_values[i]
            )
            .as_bytes(),
        )
        .unwrap();
    }
}
