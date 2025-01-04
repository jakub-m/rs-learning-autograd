use ndarray as nd;

trait Array2Conv {
    fn get_center(&self) -> nd::Dim<[usize; 2]>;
}

impl Array2Conv for nd::ArrayBase<nd::OwnedRepr<usize>, nd::Dim<[usize; 2]>> {
    /// Works for odd dimensions, e.g. for 5x7 it should return (2, 3)
    fn get_center(&self) -> nd::Dim<[usize; 2]> {
        let (r, c) = self.dim();
        nd::Dim([r / 2, c / 2])
    }
}

#[cfg(test)]
mod tests {
    use super::Array2Conv;
    use ndarray::s;
    use ndarray::{self as nd, IntoDimension};
    #[test]
    fn convolve_simple() {
        let a = nd::Array2::from_shape_fn(shape2(4, 5), |(r, c)| r * 5 + c);
        let k: nd::ArrayBase<nd::OwnedRepr<usize>, nd::Dim<[usize; 2]>> =
            nd::Array2::from_shape_fn(shape2(3, 3), |(r, c)| r * 5 + c);

        println!("a\n{:?}", a);
        println!("k\n{:?}", k);
        //println!("{:?}", k.get_center());

        let p = shape2(2, 3); // offset of kernel center w.r.t. A0
        let c = k.get_center();
        let s0 = p - c; // todo check if dimensions fit...
        let s1 = s0 + k.dim().into_dimension();
        let a_slice = a.slice(s![s0[0]..s1[0], s0[1]..s1[1]]);
        println!("slice\n{:?}", a.slice(s![s0[0]..s1[0], s0[1]..s1[1]]));
        println!("mul k * s_slice\n{:?}", &a_slice * &k);
        println!("conv k x s_slice\n{:?}", (&a_slice * &k).sum());
    }

    fn shape2(r: usize, c: usize) -> nd::Ix2 {
        nd::Ix2(r, c)
    }
    fn shape(r: usize, c: usize) -> nd::IxDyn {
        nd::IxDyn(&[r, c])
    }
}
