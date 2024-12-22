use std::{fs::File, io::Write, ops};

pub enum Opts<'a> {
    Step(f32),
    Start(f32),
    End(f32),
    TestName(&'a str),
    MaxRms(f32),
}

/// Compare reference function and with a function reconstructed from its derivate.
pub fn assert_function_and_derivative_similar<F1, F2>(f: F1, df: &mut F2, opts: &[Opts])
where
    F1: Fn(f32) -> f32,
    F2: FnMut(f32) -> f32,
{
    let mut step = 0.01;
    let mut end = 1.0;
    let mut test_name = "unknown";
    let mut max_rms = 0.01;
    let mut start = 0.0;
    for opt in opts {
        match opt {
            Opts::Step(v) => step = *v,
            Opts::Start(v) => start = *v,
            Opts::End(v) => end = *v,
            Opts::TestName(v) => test_name = *v,
            Opts::MaxRms(v) => max_rms = *v,
        };
    }

    let mut y1_values: Vec<f32> = vec![];
    let mut y2_values: Vec<f32> = vec![];
    let mut x_values: Vec<f32> = vec![];
    let mut y2 = f(start);
    assert!(step > 0.0);
    assert!(start <= end);
    let n = ((end - start) / step) as usize;
    assert!(n >= 10);
    for x in (0..n).map(|i| (i as f32) * step + start) {
        x_values.push(x);
        let y1 = f(x);
        y1_values.push(y1);

        // Push first calculate later so it behaves well at the first point.
        y2_values.push(y2);
        let dy2 = df(x);
        y2 = y2 + dy2 * step;
    }

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
    let mut step = 0.01;
    let mut end = 1.0;
    let mut test_name = "unknown";
    let mut max_rms = 0.01;
    let mut start = 0.0;
    for opt in opts {
        match opt {
            Opts::Step(v) => step = *v,
            Opts::Start(v) => start = *v,
            Opts::End(v) => end = *v,
            Opts::TestName(v) => test_name = *v,
            Opts::MaxRms(v) => max_rms = *v,
        };
    }

    let mut y1_values: Vec<f32> = vec![];
    let mut y2_values: Vec<f32> = vec![];
    let mut x_values: Vec<f32> = vec![];

    assert!(step > 0.0);
    assert!(start <= end);
    for x in iter_float(start, end, step) {
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

pub struct FloatIter<F> {
    curr: F,
    end: F,
    step: F,
}

impl<F> Iterator for FloatIter<F>
where
    F: PartialOrd + ops::Add<Output = F> + Copy,
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

pub fn iter_float<F>(start: F, end: F, step: F) -> FloatIter<F>
where
    F: PartialOrd + ops::Add<Output = F> + Copy,
{
    FloatIter {
        curr: start,
        end,
        step,
    }
}
