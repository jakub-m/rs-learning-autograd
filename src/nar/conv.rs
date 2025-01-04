use ndarray as nd;

//trait Array2Conv {
//    fn get_center(&self) -> nd::Dim<[usize; 2]>;
//}
//
//impl Array2Conv for nd::ArrayBase<nd::OwnedRepr<usize>, nd::Dim<[usize; 2]>> {
//    /// Works for odd dimensions, e.g. for 5x7 it should return (2, 3)
//    fn get_center(&self) -> nd::Dim<[usize; 2]> {
//        let (r, c) = self.dim();
//        nd::Dim([r / 2, c / 2])
//    }
//}

#[cfg(test)]
mod tests {
    use ndarray::s;
    use ndarray::{self as nd};
    #[test]
    fn iter_conv_sliding() {
        let a = new_arr(4, 5);
        let k = new_arr(3, 3);
        let mut actual_slice_corners: Vec<(i32, i32)> = Vec::new();
        for sl in iter_conv2d_slices(a.shape(), k.shape()).unwrap() {
            let a_slice = a.slice(sl);
            actual_slice_corners.push((a_slice[(0, 0)], a_slice[(2, 2)]));
            // eprintln!("a_slice\n{:?}", a_slice);
        }
        assert_eq!(
            actual_slice_corners,
            vec![(0, 12), (1, 13), (2, 14), (5, 17), (6, 18), (7, 19)]
        );
    }

    #[test]
    fn fail_iter_on_bad_shapes() {
        assert!(iter_conv2d_slices(new_arr(3, 2).shape(), new_arr(3, 3).shape()).is_err());
        assert!(iter_conv2d_slices(new_arr(2, 3).shape(), new_arr(3, 3).shape()).is_err());
    }

    #[test]
    fn fail_iter_on_even_kernel_size() {
        for nrows in (2..10).step_by(2) {
            for ncols in (2..10).step_by(2) {
                let a = new_arr(nrows, ncols);
                assert!(
                    iter_conv2d_slices(a.shape(), a.shape()).is_err(),
                    "Expected failure for ({}, {})",
                    nrows,
                    ncols
                );
            }
        }
    }

    fn shape2(r: usize, c: usize) -> nd::Ix2 {
        nd::Ix2(r, c)
    }

    fn shape(r: usize, c: usize) -> nd::IxDyn {
        nd::IxDyn(&[r, c])
    }

    fn new_arr(nrows: usize, ncols: usize) -> nd::Array2<i32> {
        nd::Array2::from_shape_fn(shape2(nrows, ncols), |(ir, ic)| (ir * ncols + ic) as i32)
    }

    /// Produce iterator that yields sliding slice indexes that can be used for convolution.
    fn iter_conv2d_slices(
        input_shape: &[usize],
        kernel_shape: &[usize],
    ) -> Result<SliceIteratorIx2, BadShapeError> {
        for d in kernel_shape {
            if *d % 2 == 0 {
                return Err(BadShapeError::from_string(format!(
                    "Even shape not allowed in kernel: {:?}",
                    kernel_shape
                )));
            }
        }

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

    struct SliceIteratorIx2 {
        kernel_shape: [usize; 2],
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
}
