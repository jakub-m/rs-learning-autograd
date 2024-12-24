use std::{cmp, fmt, fs::File, io::Write, ops};

pub enum Opts<'a> {
    InputRange(FloatRange<f32>),
    TestName(&'a str),
    MaxRms(f32),
}

/// Compare reference function and with a function reconstructed from its derivate.
pub fn assert_function_and_derivative_similar<F1, F2>(f: F1, df: &mut F2, opts: &[Opts])
where
    F1: Fn(f32) -> f32,
    F2: FnMut(f32) -> f32,
{
    let mut test_name = "unknown";
    let mut max_rms = 0.01;
    let mut input_range = FloatRange::<f32>::new(0.0, 1.0, 0.01);
    for opt in opts {
        match opt {
            Opts::TestName(v) => test_name = *v,
            Opts::MaxRms(v) => max_rms = *v,
            Opts::InputRange(v) => input_range = *v,
        };
    }

    let mut y1_values: Vec<f32> = vec![];
    let mut y2_values: Vec<f32> = vec![];
    let mut x_values: Vec<f32> = vec![];
    let mut y2 = f(input_range.start());

    for x in input_range.into_iter() {
        x_values.push(x);
        let y1 = f(x);
        y1_values.push(y1);

        // Push first calculate later so it behaves well at the first point.
        y2_values.push(y2);
        let dy2 = df(x);
        y2 = y2 + dy2 * input_range.step();
    }
    assert!(x_values.len() == y1_values.len() && x_values.len() == y2_values.len());
    let n = x_values.len();
    assert!(n > 0);

    let mut rms = 0.0;
    for i in 0..n {
        let d = y1_values[i] - y2_values[i];
        rms += d * d;
    }
    rms = (rms / (n as f32)).sqrt();

    if rms > max_rms {
        write_series_to_file(
            format!("{}.csv", test_name).as_str(),
            &x_values,
            &y1_values,
            &y2_values,
        );
        panic!("RMS was {} > {}", rms, max_rms);
    }
}

pub fn assert_functions_similar<F1, F2>(f1: F1, f2: &mut F2, opts: &[Opts])
where
    F1: Fn(f32) -> f32,
    F2: FnMut(f32) -> f32,
{
    let mut test_name = "unknown";
    let mut max_rms = 0.01;
    let mut float_range = FloatRange::new(-1.0_f32, 1.0, 0.01);
    for opt in opts {
        match opt {
            Opts::TestName(v) => test_name = *v,
            Opts::MaxRms(v) => max_rms = *v,
            Opts::InputRange(v) => float_range = *v,
        };
    }

    let mut y1_values: Vec<f32> = vec![];
    let mut y2_values: Vec<f32> = vec![];
    let mut x_values: Vec<f32> = vec![];

    for x in float_range.into_iter() {
        x_values.push(x);
        y1_values.push(f1(x));
        y2_values.push(f2(x));
    }

    let n = x_values.len();
    let mut rms = 0.0;
    for i in 0..n {
        let d = y1_values[i] - y2_values[i];
        rms += d * d;
    }
    rms = (rms / (n as f32)).sqrt();
    if rms > max_rms {
        write_series_to_file(
            format!("{}.csv", test_name).as_str(),
            &x_values,
            &y1_values,
            &y2_values,
        );
        panic!("RMS was {} > {}", rms, max_rms);
    }
}

fn write_series_to_file(
    file_name: &str,
    x_values: &Vec<f32>,
    y1_values: &Vec<f32>,
    y2_values: &Vec<f32>,
) {
    println!("Writing to file {}", file_name);
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

pub trait FloatType:
    fmt::Display
    + Copy
    + cmp::PartialOrd
    + ops::Add<Output = Self>
    + ops::Sub<Output = Self>
    + ops::Div<Output = Self>
    + Default
{
}

impl FloatType for f32 {}

#[derive(Clone, Copy)]
pub struct FloatRange<F>
where
    F: FloatType,
{
    start: F,
    end: F,
    step: F,
}

impl<F> FloatRange<F>
where
    F: FloatType,
{
    pub fn new(start: F, end: F, step: F) -> FloatRange<F> {
        let n_steps = (end - start) / step;
        let zero = F::default();
        if n_steps < zero {
            panic!(
                "Negative number of steps for start={} end={} step={}",
                start, end, step
            )
        }
        FloatRange { start, end, step }
    }

    pub fn start(&self) -> F {
        self.start
    }
    pub fn step(&self) -> F {
        self.step
    }
}

impl<F> IntoIterator for &FloatRange<F>
where
    F: FloatType,
{
    type Item = F;
    type IntoIter = FloatIter<F>;

    fn into_iter(self) -> Self::IntoIter {
        FloatIter {
            curr: self.start,
            end: self.end,
            step: self.step,
        }
    }
}

pub struct FloatIter<F>
where
    F: FloatType,
{
    curr: F,
    end: F,
    step: F,
}

impl<F> Iterator for FloatIter<F>
where
    F: FloatType,
{
    type Item = F;

    fn next(&mut self) -> Option<Self::Item> {
        if self.curr >= self.end {
            None
        } else {
            let curr = self.curr;
            self.curr = self.curr + self.step;
            Some(curr)
        }
    }
}
