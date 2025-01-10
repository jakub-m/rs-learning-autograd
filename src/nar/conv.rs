use ndarray as nd;
use ndarray::s;
use std::{fmt, ops};

/// Convolve two 2d matrices into a single 2d matrix. `k` is the kernel matrix.
#[allow(dead_code)]
pub fn conv2d<A>(a: &nd::Array<A, nd::Ix2>, k: &nd::Array<A, nd::Ix2>) -> nd::Array<A, nd::Ix2>
where
    A: std::ops::Mul<Output = A> + Copy + num_traits::Zero + fmt::Debug,
{
    let ix_iter = iter_conv2d_slices(a.shape(), k.shape()).unwrap();
    let output_shape = &ix_iter.output_shape();
    let output_shape = (output_shape[0], output_shape[1]);
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
    nd::Array::from_iter(iter_conv)
        .to_shape(output_shape)
        .unwrap()
        .into_owned()
}

/// Given that there is a convolution V = conv2d(A, K) where A is an array and K is a kernel, this function calculates
/// a derivative `dV/dK`, that is, much does K contribute to V.
pub fn conv2d_adjoin<A>(
    a: &nd::Array<A, nd::Ix2>,
    k: &nd::Array<A, nd::Ix2>,
) -> nd::Array<A, nd::Ix2>
where
    A: Copy + fmt::Debug,
    A: num_traits::Zero,
    A: ops::Sub<Output = A> + ops::Add<Output = A>,
{
    sliding_sum(a, k.shape())
}

/// Take the window of size `k_shape` (e.g. kernel), slide it along matrix `a` and sum all the values in the window.
/// The implementation does not literally slide the window and sum the values, but instead does it in two swipes.
fn sliding_sum<A>(a: &nd::Array<A, nd::Ix2>, k_shape: &[usize]) -> nd::Array<A, nd::Ix2>
where
    A: Copy + fmt::Debug,
    A: num_traits::Zero,
    A: ops::Sub<Output = A> + ops::Add<Output = A>,
{
    let mut sums0 = a.clone();
    let a_shape = sums0.shape().to_owned();

    for i_d0 in 0..a_shape[0] {
        // Say d0 are rows and d1 are columns. For each row do the following:
        // First, sum all the values withing the window. The last value will be the sum.
        // Then, slide the window, and update the sum. The right part of the row will contain
        // the sums of the sliding window.

        sums0[[i_d0, k_shape[1] - 1]] = a.slice(s![i_d0, 0..k_shape[1]]).sum();
        // Now slide the window and update the sum with one value that entered and left the window.
        for i_d1 in k_shape[1]..a_shape[1] {
            sums0[[i_d0, i_d1]] =
                sums0[[i_d0, i_d1 - 1]] - a[[i_d0, i_d1 - k_shape[1]]] + a[[i_d0, i_d1]];
        }
    }

    // Now do the same trick but vertically, and only to the columns on the right.
    // Actually you could do it all in a single pass.
    let mut sums1 = sums0.clone();
    for i_d1 in (a_shape[1] - k_shape[1])..a_shape[1] {
        sums1[[k_shape[0] - 1, i_d1]] = sums0.slice(s![0..k_shape[0], i_d1]).sum();
        for i_d0 in k_shape[0]..a_shape[0] {
            sums1[[i_d0, i_d1]] =
                sums1[[i_d0 - 1, i_d1]] - sums0[[i_d0 - k_shape[0], i_d1]] + sums0[[i_d0, i_d1]];
        }
    }

    sums1
        .slice(s![k_shape[0] - 1.., k_shape[1] - 1..])
        .to_owned()
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

#[derive(Debug)]
struct BadShapeError(String);

impl BadShapeError {
    fn from_string(message: String) -> BadShapeError {
        BadShapeError(message)
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

    use crate::nar::conv::conv2d_adjoin;

    use super::conv2d;
    use super::iter_conv2d_slices;
    use super::sliding_sum;
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
        let actual: ndarray::ArrayBase<ndarray::OwnedRepr<i32>, ndarray::Dim<[usize; 2]>> =
            conv2d(&a, &k);
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
        assert_eq!(actual, expected);
    }

    #[test]
    fn test_sliding_sum() {
        let a = new_arr_inc_i32(5, 4);
        // a
        // [[0, 1, 2, 3],
        //  [4, 5, 6, 7],
        //  [8, 9, 10, 11],
        //  [12, 13, 14, 15],
        //  [16, 17, 18, 19]]
        // k
        // [[. . .]
        //  [. . .]]
        // The result should be 4x2
        let actual = sliding_sum(&a, &[2, 3]);
        let expected = arr2(&[
            [(0 + 1 + 2 + 4 + 5 + 6), (1 + 2 + 3 + 5 + 6 + 7)],
            [(4 + 5 + 6 + 8 + 9 + 10), (5 + 6 + 7 + 9 + 10 + 11)],
            [(8 + 9 + 10 + 12 + 13 + 14), (9 + 10 + 11 + 13 + 14 + 15)],
            [(12 + 13 + 14 + 16 + 17 + 18), (13 + 14 + 15 + 17 + 18 + 19)],
        ]);
        assert_eq!(actual, expected);
    }

    #[test]
    fn test_conv2d_adjoin() {
        let a = new_arr_inc_f32(5, 4);
        let k = new_arr_inc_f32(2, 3);
        let adjoin = conv2d_adjoin(&a, &k); // [4, 2]

        // let k_max = k
        //     .iter()
        //     .reduce(|acc, e| if e > acc { e } else { acc })
        //     .unwrap()
        //     .to_owned();
        // let k = (2.0 * &k - k_max) / k_max;

        dbg!(&adjoin);

        let i0 = 0;
        let i1 = 0;

        let epsilon = 0.001;
        let mut k_left = k.clone();
        let mut k_right = k.clone();
        k_left[[i0, i1]] = k_left[[i0, i1]] - epsilon;
        k_right[[i0, i1]] = k_right[[i0, i1]] + epsilon;
        let conv_left = conv2d(&a, &k_left);
        let conv_right = conv2d(&a, &k_right);

        let dy_dk_approx = (&conv_right - &conv_left) / epsilon;
        let dy_dk_approx = dy_dk_approx;
        dbg!(&dy_dk_approx);
        dbg!(&adjoin[[i0, i1]]);

        //assert_eq!(actual, expected);
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

    //fn shape(r: usize, c: usize) -> nd::IxDyn {
    //    nd::IxDyn(&[r, c])
    //}

    fn new_arr_inc_i32(nrows: usize, ncols: usize) -> nd::Array2<i32> {
        nd::Array2::from_shape_fn(shape2(nrows, ncols), |(ir, ic)| (ir * ncols + ic) as i32)
    }

    fn new_arr_inc_f32(nrows: usize, ncols: usize) -> nd::Array2<f32> {
        nd::Array2::from_shape_fn(shape2(nrows, ncols), |(ir, ic)| (ir * ncols + ic) as f32)
    }
}
