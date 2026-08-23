use crate::tensor::core::Tensor;
use std::ops::{Add, Mul};

impl<T> Tensor<T> {
    /// Perform a dot product on two tensors of the same shape.
    ///
    /// # Panics
    ///
    /// Panics if the shapes of `self` and `rhs` do not match.
    #[must_use]
    pub fn dot(&self, rhs: &Self) -> Self
    where
        T: Copy + Mul<Output = T> + Add<Output = T> + Default,
    {
        let dot = {
            assert_eq!(self.shape(), rhs.shape());
            self.data()
                .iter()
                .zip(rhs.data().iter())
                .map(|(a, b)| *a * *b)
                .reduce(|a, b| a + b)
                .unwrap()
        };
        Self::from_parts(
            vec![dot],
            vec![],
            vec![T::default(); 1],
            self.requires_grad() || rhs.requires_grad(),
            None,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use test_case::test_case;

    /// Test that the tensor dot product operator works correctly.
    #[test_case(
        [1, 2, 3],
        [4, 5, 6],
        [3],
        [32],
        [];
        "basic"
    )]
    #[test_case(
        [0, 0, 0],
        [1, 2, 3],
        [3],
        [0],
        [];
        "zero vector"
    )]
    #[test_case(
        [1, 1, 1],
        [1, 1, 1],
        [3],
        [3],
        [];
        "ones"
    )]
    #[test_case(
        [2, 3, 4],
        [5, 6, 7],
        [3],
        [56],
        [];
        "small integers"
    )]
    #[test_case(
        [-1, -2, -3],
        [1, 2, 3],
        [3],
        [-14],
        [];
        "negatives"
    )]
    #[test_case(
        [1, -2, 3],
        [-4, 5, -6],
        [3],
        [-32],
        [];
        "mixed signs"
    )]
    #[test_case(
        [10],
        [7],
        [1],
        [70],
        [];
        "single element"
    )]
    #[test_case(
        [1, 2, 3, 4],
        [4, 3, 2, 1],
        [4],
        [20],
        [];
        "reversed"
    )]
    fn test_dot<const N: usize, const S: usize, const ED: usize, const ES: usize>(
        a: [i32; N],
        b: [i32; N],
        shape: [usize; S],
        expected_data: [i32; ED],
        expected_shape: [usize; ES],
    ) {
        let tensor_one = Tensor::new(a, shape, true);
        let tensor_two = Tensor::new(b, shape, true);
        let result = tensor_one.dot(&tensor_two);
        assert_eq!(result.data(), expected_data);
        assert_eq!(result.shape(), expected_shape);
    }
}
