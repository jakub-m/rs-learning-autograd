use ndarray::s;
use ndarray::{self as nd};
use std::{fmt, ops};

use super::conv_iter::V2;

/// Convolve two 2d matrices into a single 2d matrix. `k` is the kernel matrix.
#[allow(dead_code)]
pub fn conv2d<A>(
    a: &nd::CowArray<A, nd::IxDyn>,
    k: &nd::CowArray<A, nd::IxDyn>,
) -> nd::Array<A, nd::IxDyn>
where
    A: std::ops::Mul<Output = A> + Copy + num_traits::Zero + fmt::Debug,
{
    let ix_iter = iter_conv2d_slices(a.shape(), k.shape()).unwrap();
    let output_shape = &ix_iter.output_shape();
    let iter_conv = ix_iter.map(|a_ix| {
        let a_slice = a.slice(a_ix);
        assert_eq!(
            a_slice.shape(),
            k.shape(),
            "The shapes of the convoluted slice and kernel must match!"
        );
        let m_mul = &a_slice * k;
        m_mul.sum()
    });
    let elements: Vec<A> = iter_conv.collect();
    nd::ArrayD::from_shape_vec(nd::IxDyn(output_shape), elements).unwrap()
}

/// Produce iterator that yields sliding slice indexes that can be used for convolution.
fn iter_conv2d_slices(
    input_shape: &[usize],
    kernel_shape: &[usize],
) -> Result<SliceIteratorIx2, BadShapeError> {
    for i in 0..2 {
        if input_shape[i] < kernel_shape[i] {
            return Err(BadShapeError::from_string(format!(
                "Input {:?} smaller than kernel {:?}",
                input_shape, kernel_shape
            )));
        }
    }

    let d0_range = [0_usize, (input_shape[0] - kernel_shape[0] + 1)];
    let d1_range = [0_usize, (input_shape[1] - kernel_shape[1] + 1)];
    Ok(SliceIteratorIx2 {
        kernel_shape: [kernel_shape[0], kernel_shape[1]],
        d0_range,
        d1_range,
        d0_curr: d0_range[0],
        d1_curr: d1_range[0],
    })
}

/// Given `v = convolute(a, k)`, where `k` is a kernel, calculate `dv/da` and `dv/dk`, returned in that order.
/// # Arguments
/// * `a` - the input matrix.
/// * `k` - the kernel matrix.
/// * `adv` - the adjoin from the upstream (reverse mode).
pub fn conv2d_adjoin<A>(
    a: &nd::Array2<A>,
    k: &nd::Array2<A>,
    adv: &nd::Array2<A>,
) -> (nd::Array2<A>, nd::Array2<A>)
where
    A: ops::Mul<Output = A>,
    A: Clone + Copy,
    A: num_traits::Zero,
    A: PartialEq,
{
    let a_size = a.shape().into_v2d();
    let mut dv_dk: nd::Array2<A> = nd::Array2::zeros(adv.raw_dim());
    let mut dv_da: nd::Array2<A> = nd::Array2::zeros(a.raw_dim());
    for adv_ix in adv.shape().into_v2d().iter() {
        // Iterate over every cell of the adjoin, and calculate what's the contribution of `k` and `a`.
        if adv[adv_ix.as_ix()] == A::zero() {
            continue;
        }
        for k_ix in k.shape().into_v2d().iter() {
            // Iterate over each kernel cell and at the same time calculate `dv/dk` and `dv/da`.
            if let Some(a_ix) = a_size.contains(adv_ix + k_ix) {
                let adv_ix = adv_ix.as_ix();
                let a_ix = a_ix.as_ix();
                let k_ix = k_ix.as_ix();
                dv_dk[adv_ix] = dv_dk[adv_ix] + a[a_ix] * adv[adv_ix];
                dv_da[a_ix] = dv_da[a_ix] + k[k_ix] + adv[adv_ix];
            }
        }
    }
    (dv_da, dv_dk)
}

#[derive(Debug)]
struct BadShapeError(String);

impl BadShapeError {
    fn from_string(message: String) -> BadShapeError {
        BadShapeError(message)
    }
}

trait V2Helper {
    fn into_v2d(&self) -> V2;
}

impl V2Helper for &[usize] {
    /// Convert nd array shape to V2
    fn into_v2d(&self) -> V2 {
        V2(self[0], self[1])
    }
}

trait ShapeHelper {
    fn as_ix(&self) -> [usize; 2];
}

impl ShapeHelper for V2 {
    fn as_ix(&self) -> [usize; 2] {
        [self.0, self.1]
    }
}

pub struct SliceIteratorIx2 {
    kernel_shape: [usize; 2],
    /// Range of the indices (left inclusive, right exclusive) for dimension 0 of the input matrix. The kernel
    /// is slid along those values.
    d0_range: [usize; 2],
    d1_range: [usize; 2],
    d0_curr: usize,
    d1_curr: usize,
}

impl Iterator for SliceIteratorIx2 {
    type Item = nd::SliceInfo<[nd::SliceInfoElem; 2], nd::Ix2, nd::Ix2>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.d1_curr >= self.d1_range[1] {
            self.d1_curr = self.d1_range[0];
            self.d0_curr += 1
        }
        if self.d0_curr >= self.d0_range[1] {
            return None;
        }
        let d0_a = self.d0_curr;
        let d1_a = self.d1_curr;
        let d0_b = d0_a + self.kernel_shape[0];
        let d1_b = d1_a + self.kernel_shape[1];
        self.d1_curr += 1;
        //eprintln!("{}..{},{}..{}", d0_a, d0_b, d1_a, d1_b);
        Some(s![d0_a..d0_b, d1_a..d1_b,])
    }
}

impl SliceIteratorIx2 {
    /// Return shape of the output matrix.
    pub fn output_shape(&self) -> [usize; 2] {
        [
            self.d0_range[1] - self.d0_range[0],
            self.d1_range[1] - self.d1_range[0],
        ]
    }
}

#[cfg(test)]
mod tests {
    use std::ops;

    use super::conv2d;
    use super::iter_conv2d_slices;
    use ndarray::{self as nd, arr2};

    #[test]
    fn test_iter_conv_sliding() {
        let a = new_arr_inc_i32(4, 5);
        let k = new_arr_inc_i32(3, 3);
        let mut actual_slice_corners: Vec<(i32, i32)> = Vec::new();
        for sl in iter_conv2d_slices(a.shape(), k.shape()).unwrap() {
            let a_slice = a.slice(sl);
            actual_slice_corners.push((a_slice[(0, 0)], a_slice[(2, 2)]));
        }
        assert_eq!(
            actual_slice_corners,
            vec![(0, 12), (1, 13), (2, 14), (5, 17), (6, 18), (7, 19)]
        );
    }

    #[test]
    fn test_fail_iter_on_bad_shapes() {
        assert!(
            iter_conv2d_slices(new_arr_inc_i32(3, 2).shape(), new_arr_inc_i32(3, 3).shape())
                .is_err()
        );
        assert!(
            iter_conv2d_slices(new_arr_inc_i32(2, 3).shape(), new_arr_inc_i32(3, 3).shape())
                .is_err()
        );
    }

    #[test]
    fn test_convolve_two_matrices_2d() {
        let a = new_arr_inc_i32(5, 4);
        let k = new_arr_inc_i32(2, 3);
        let a: nd::CowArray<_, nd::IxDyn> = a.into();
        let k: nd::CowArray<_, nd::IxDyn> = k.into();
        let actual = conv2d(&a, &k);
        // a
        // [[0, 1, 2, 3],
        //  [4, 5, 6, 7],
        //  [8, 9, 10, 11],
        //  [12, 13, 14, 15],
        //  [16, 17, 18, 19]]
        // k
        // [[0, 1, 2],
        //  [3, 4, 5]]
        assert_eq!(actual.shape(), [4, 2]);
        let b = [0, 1, 2, 3, 4, 5];
        let expected = arr2(&[
            [dot([0, 1, 2, 4, 5, 6], b), dot([1, 2, 3, 5, 6, 7], b)],
            [dot([4, 5, 6, 8, 9, 10], b), dot([5, 6, 7, 9, 10, 11], b)],
            [
                dot([8, 9, 10, 12, 13, 14], b),
                dot([9, 10, 11, 13, 14, 15], b),
            ],
            [
                dot([12, 13, 14, 16, 17, 18], b),
                dot([13, 14, 15, 17, 18, 19], b),
            ],
        ]);
        eprintln!(
            "a\n{:?}\nk\n{:?}\nactual\n{:?}\nexpected\n{:?}",
            &a, &k, actual, expected
        );
        assert_eq!(actual.into_dimensionality::<nd::Ix2>().unwrap(), expected);
    }

    fn dot<F, const N: usize>(a: [F; N], b: [F; N]) -> F
    where
        F: ops::Add<F, Output = F>,
        F: ops::Mul<F, Output = F>,
        F: Copy,
    {
        (0..N)
            .map(|i| a[i] * b[i])
            .reduce(|acc, e| acc + e)
            .unwrap()
    }

    fn shape2(r: usize, c: usize) -> nd::Ix2 {
        nd::Ix2(r, c)
    }

    fn new_arr_inc_i32(nrows: usize, ncols: usize) -> nd::ArrayD<i32> {
        nd::ArrayD::from_shape_fn(shape(nrows, ncols), |d| (d[0] * ncols + d[1]) as i32)
    }

    fn new_arr_inc_f32(nrows: usize, ncols: usize) -> nd::Array2<f32> {
        nd::Array2::from_shape_fn(shape2(nrows, ncols), |(ir, ic)| (ir * ncols + ic) as f32)
    }

    fn shape(r: usize, c: usize) -> nd::IxDyn {
        nd::IxDyn(&[r, c])
    }
}
