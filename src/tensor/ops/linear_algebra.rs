use crate::core::{TensorState, next_tensor_id};
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
            let lhs_state = self.state.borrow();
            let rhs_state = rhs.state.borrow();
            assert_eq!(lhs_state.shape, rhs_state.shape);
            lhs_state
                .data
                .iter()
                .zip(rhs_state.data.iter())
                .map(|(a, b)| *a * *b)
                .reduce(|a, b| a + b)
                .unwrap()
        };
        Self::from_state(TensorState {
            id: next_tensor_id(),
            data: vec![dot],
            shape: vec![],
            grad: vec![T::default(); 1],
            node: None,
        })
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
        vec![32],
        vec![];
        "basic"
    )]
    #[test_case(
        [0, 0, 0],
        [1, 2, 3],
        [3],
        vec![0],
        vec![];
        "zero vector"
    )]
    #[test_case(
        [1, 1, 1],
        [1, 1, 1],
        [3],
        vec![3],
        vec![];
        "ones"
    )]
    #[test_case(
        [2, 3, 4],
        [5, 6, 7],
        [3],
        vec![56],
        vec![];
        "small integers"
    )]
    #[test_case(
        [-1, -2, -3],
        [1, 2, 3],
        [3],
        vec![-14],
        vec![];
        "negatives"
    )]
    #[test_case(
        [1, -2, 3],
        [-4, 5, -6],
        [3],
        vec![-32],
        vec![];
        "mixed signs"
    )]
    #[test_case(
        [10],
        [7],
        [1],
        vec![70],
        vec![];
        "single element"
    )]
    #[test_case(
        [1, 2, 3, 4],
        [4, 3, 2, 1],
        [4],
        vec![20],
        vec![];
        "reversed"
    )]
    fn test_dot<const N: usize, const S: usize>(
        a: [i32; N],
        b: [i32; N],
        shape: [usize; S],
        expected_data: Vec<i32>,
        expected_shape: Vec<usize>,
    ) {
        let tensor_one = Tensor::new(a, shape);
        let tensor_two = Tensor::new(b, shape);
        let result = tensor_one.dot(&tensor_two);
        assert_eq!(result.state.borrow().data, expected_data);
        assert_eq!(result.state.borrow().shape, expected_shape);
    }
}
