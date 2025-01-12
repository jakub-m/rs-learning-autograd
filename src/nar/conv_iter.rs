use core::fmt;
use std::ops;

/// A 2 dimensional vector.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct V2(pub usize, pub usize);

impl V2 {
    pub fn iter(&self) -> V2Iterator {
        V2Iterator {
            curr: V2(0, 0),
            last: self.clone(),
        }
    }

    /// Check if `other` is within the box between point (0, 0) and `self.
    pub fn contains(&self, other: V2) -> Option<V2> {
        if self.0 >= other.0 && self.1 >= other.1 {
            Some(other)
        } else {
            None
        }
    }

    fn try_sub(&self, other: V2) -> Option<V2> {
        // self - other
        if (self.0 >= other.0) & (self.1 >= other.1) {
            Some(V2(self.0 - other.0, self.1 - other.1))
        } else {
            None
        }
    }
}

#[derive(Debug, PartialEq)]
pub struct V2Iterator {
    curr: V2,
    last: V2,
}

impl Iterator for V2Iterator {
    type Item = V2;

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

impl ops::Add for V2 {
    type Output = V2;

    fn add(self, other: Self) -> Self::Output {
        V2(self.0 + other.0, self.1 + other.1)
    }
}

impl ops::Sub for V2 {
    type Output = V2;

    fn sub(self, other: Self) -> Self::Output {
        self.try_sub(other)
            .expect(format!("cannot sub {:?} - {:?}", self, other).as_str())
    }
}

impl fmt::Display for V2 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[{} {}]", self.0, self.1)
    }
}

#[cfg(test)]
mod tests {
    use super::V2;
    #[test]
    fn test_iter_indices_1x1() {
        let ix = V2(1, 1);
        let mut actual: Vec<V2> = Vec::new();
        for i in ix.iter() {
            actual.push(i)
        }
        assert_eq!(actual, vec![V2(0, 0),]);
    }

    #[test]
    fn test_iter_indices_3x2() {
        let ix = V2(3, 2);
        let mut actual: Vec<V2> = Vec::new();
        for i in ix.iter() {
            actual.push(i)
        }
        assert_eq!(
            actual,
            vec![V2(0, 0), V2(0, 1), V2(1, 0), V2(1, 1), V2(2, 0), V2(2, 1),]
        );
    }
}
