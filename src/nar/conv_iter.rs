use core::fmt;
use std::ops;

#[derive(Clone, Copy, Debug, PartialEq)]
struct Ix2(usize, usize);

impl Ix2 {
    fn iter(&self) -> IndicesIterator {
        IndicesIterator {
            curr: Ix2(0, 0),
            last: self.clone(),
        }
    }

    fn contains(&self, other: Ix2) -> Option<Ix2> {
        if self.0 >= other.0 && self.1 >= other.1 {
            Some(other)
        } else {
            None
        }
    }

    fn try_sub(&self, other: Ix2) -> Option<Ix2> {
        // self - other
        if (self.0 >= other.0) & (self.1 >= other.1) {
            Some(Ix2(self.0 - other.0, self.1 - other.1))
        } else {
            None
        }
    }
}

#[derive(Debug, PartialEq)]
struct IndicesIterator {
    curr: Ix2,
    last: Ix2,
}

impl Iterator for IndicesIterator {
    type Item = Ix2;

    fn next(&mut self) -> Option<Self::Item> {
        let ret = self.curr;
        if self.curr.0 >= self.last.0 {
            return None;
        }
        self.curr.1 += 1;
        if self.curr.1 >= self.last.1 {
            self.curr.1 = 0;
            self.curr.0 += 1;
        }
        return Some(ret);
    }
}

impl ops::Add for Ix2 {
    type Output = Ix2;

    fn add(self, other: Self) -> Self::Output {
        Ix2(self.0 + other.0, self.1 + other.1)
    }
}

impl ops::Sub for Ix2 {
    type Output = Ix2;

    fn sub(self, other: Self) -> Self::Output {
        self.try_sub(other)
            .expect(format!("cannot sub {:?} - {:?}", self, other).as_str())
    }
}

impl fmt::Display for Ix2 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[{} {}]", self.0, self.1)
    }
}

#[cfg(test)]
mod tests {
    use super::Ix2;
    #[test]
    fn test_iter_indices_1x1() {
        let ix = Ix2(1, 1);
        let mut actual: Vec<Ix2> = Vec::new();
        for i in ix.iter() {
            actual.push(i)
        }
        assert_eq!(actual, vec![Ix2(0, 0),]);
    }

    #[test]
    fn test_iter_indices_3x2() {
        let ix = Ix2(3, 2);
        let mut actual: Vec<Ix2> = Vec::new();
        for i in ix.iter() {
            actual.push(i)
        }
        assert_eq!(
            actual,
            vec![
                Ix2(0, 0),
                Ix2(0, 1),
                Ix2(1, 0),
                Ix2(1, 1),
                Ix2(2, 0),
                Ix2(2, 1),
            ]
        );
    }

    #[test]
    fn test_iter_conv() {
        let a = Ix2(4, 5);
        let k = Ix2(2, 3);
        let v = a - k + Ix2(1, 1); // size of the convoluted matrix.
        for ix_v in v.iter() {
            for ix_k in a.iter() {
                // calculate dv/dk for convolution v=conv(a, k)
                // for dv(ix_v)/dk(ix_k) where ix_v == ix_a,
                // dv/dk = a(ix_v + ix_k)
                if let Some(ix_da) = k.contains(ix_v + ix_k) {
                    // Update adjoin of K with a(ix_da) * upstream adjoin up(ix_dv)
                    // ...
                    eprintln!("dv{}/dk{} = a{}", ix_v, ix_k, ix_da)
                }
                // while here, calculate also dv/da since
                // dv(ix_v)/da(ix_a+ix_k) = k(ix_k)
                {
                    if let Some(ix_da) = a.contains(ix_v + ix_k) {
                        // Update adjoin of A(ix_da) with K(ix_k) * upstream adjoin up(ix_dv)
                        // eprintln!("dv{}/da{} = k{}", ix_v, ix_da, ix_k)
                    }
                }
            }
        }
    }
}
