use crate::tensor::autograd::operations::MeanOperation;
use crate::tensor::core::{OperationNode, Tensor, TensorState, next_tensor_id};
use num_traits::Float;
use std::fmt::Debug;
use std::iter::Sum;
use std::ops::AddAssign;

impl<T> Tensor<T>
where
    T: Sum + Copy + Default,
{
    /// Sum the elements of a tensor.
    #[must_use]
    pub fn sum(&self) -> Self {
        Self::from_state(TensorState {
            id: next_tensor_id(),
            data: vec![self.state.borrow().data.iter().copied().sum()],
            shape: vec![],
            grad: vec![T::default(); 1],
            requires_grad: self.requires_grad(),
            node: None,
        })
    }
}

impl<T> Tensor<T>
where
    T: Float + Default,
{
    /// Compute the absolute value of a tensor.
    #[must_use]
    pub fn abs(&self) -> Self {
        let (data, shape) = {
            let state = self.state.borrow();
            (
                state.data.iter().map(|x| x.abs()).collect::<Vec<_>>(),
                state.shape.clone(),
            )
        };
        let num_elements = data.len();
        Self::from_state(TensorState {
            id: next_tensor_id(),
            data,
            shape,
            grad: vec![T::default(); num_elements],
            requires_grad: self.requires_grad(),
            node: None,
        })
    }
}

impl<T> Tensor<T>
where
    T: Float + Sum + Default + AddAssign + Debug,
{
    /// Compute the mean of a tensor.
    ///
    /// # Panics
    ///
    /// Panics if the tensor length cannot be converted to `T`.
    #[must_use]
    pub fn mean(&self) -> Self {
        let (mean_val, size) = {
            let state = self.state.borrow();
            let sum: T = state.data.iter().copied().sum();
            let size = state.data.len();
            let len = T::from(size).expect("Failed to convert length to Float");
            (sum / len, size)
        };
        let requires_grad = self.requires_grad();
        Self::from_state(TensorState {
            id: next_tensor_id(),
            data: vec![mean_val],
            shape: vec![],
            grad: vec![T::default(); 1],
            requires_grad,
            node: if requires_grad {
                Some(OperationNode {
                    parents: vec![self.clone()],
                    operation: Box::new(MeanOperation { size }),
                })
            } else {
                None
            },
        })
    }
}

impl<T> Tensor<T>
where
    T: Copy + PartialOrd + Default,
{
    /// Compute the minimum or maximum value in the iterator and return it as a scalar tensor.
    fn compute_min_max(&self, comparator: impl Fn(T, T) -> bool) -> Self {
        let result = self
            .state
            .borrow()
            .data
            .iter()
            .copied()
            .reduce(|a, b| if comparator(a, b) { a } else { b })
            .expect("Tensor is empty");
        Self::from_state(TensorState {
            id: next_tensor_id(),
            data: vec![result],
            shape: vec![],
            grad: vec![T::default(); 1],
            requires_grad: self.requires_grad(),
            node: None,
        })
    }

    /// Get the minimum value of the tensor.
    #[must_use]
    pub fn min(&self) -> Self {
        self.compute_min_max(|a, b| a <= b)
    }

    /// Get the maximum value of the tensor.
    #[must_use]
    pub fn max(&self) -> Self {
        self.compute_min_max(|a, b| a >= b)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use test_case::test_case;

    /// Test that summing tensor elements works correctly.
    #[test_case([1, 2, 3], [3], [6], []; "positive vector")]
    #[test_case([0, 0, 0], [3], [0], []; "all zeros")]
    #[test_case([-1, -2, -3], [3], [-6], []; "negative vector")]
    #[test_case([-1, 2, -3, 4], [4], [2], []; "mixed signs")]
    #[test_case([42], [1], [42], []; "single element")]
    #[test_case([1, 2, 3, 4], [2, 2], [10], []; "matrix")]
    fn test_sum<const N: usize, const S: usize, const ED: usize, const ES: usize>(
        data: [i32; N],
        shape: [usize; S],
        expected_data: [i32; ED],
        expected_shape: [usize; ES],
    ) {
        let tensor = Tensor::new(data, shape, true);
        let summed = tensor.sum();
        assert_eq!(summed.state.borrow().data, expected_data);
        assert_eq!(summed.state.borrow().shape, expected_shape);
    }

    /// Test that computing the mean of tensors works correctly.
    #[test_case([1.0, 2.0, 3.0], [3], [2.0], []; "simple float mean")]
    #[test_case([1.0, 2.0, 3.0, 4.0], [4], [2.5], []; "fractional float mean")]
    #[test_case([-1.0, 1.0], [2], [0.0], []; "opposite floats")]
    #[test_case([2.5], [1], [2.5], []; "single float")]
    fn test_mean<const N: usize, const S: usize, const ED: usize, const ES: usize>(
        data: [f32; N],
        shape: [usize; S],
        expected_data: [f32; ED],
        expected_shape: [usize; ES],
    ) {
        let tensor = Tensor::new(data, shape, true);
        let mean_tensor = tensor.mean();
        assert_eq!(mean_tensor.state.borrow().data, expected_data);
        assert_eq!(mean_tensor.state.borrow().shape, expected_shape);
    }

    /// Test that getting the min of a tensor works correctly.
    #[test_case([1, 2, 3], [3], [1], []; "ascending")]
    #[test_case([3, 2, 1], [3], [1], []; "descending")]
    #[test_case([-10, -5, -20], [3], [-20], []; "negative values")]
    #[test_case([-1, 0, 1], [3], [-1], []; "mixed signs")]
    #[test_case([42], [1], [42], []; "single element")]
    #[test_case([1, 9, 3, 7], [2, 2], [1], []; "matrix")]
    fn test_min<const N: usize, const S: usize, const ED: usize, const ES: usize>(
        data: [i32; N],
        shape: [usize; S],
        expected_data: [i32; ED],
        expected_shape: [usize; ES],
    ) {
        let tensor = Tensor::new(data, shape, true);
        let min_tensor = tensor.min();
        assert_eq!(min_tensor.state.borrow().data, expected_data);
        assert_eq!(min_tensor.state.borrow().shape, expected_shape);
    }

    /// Test that getting the max of a tensor works correctly.
    #[test_case([1, 2, 3], [3], [3], []; "ascending")]
    #[test_case([3, 2, 1], [3], [3], []; "descending")]
    #[test_case([-10, -5, -20], [3], [-5], []; "negative values")]
    #[test_case([-1, 0, 1], [3], [1], []; "mixed signs")]
    #[test_case([42], [1], [42], []; "single element")]
    #[test_case([1, 9, 3, 7], [2, 2], [9], []; "matrix")]
    fn test_max<const N: usize, const S: usize, const ED: usize, const ES: usize>(
        data: [i32; N],
        shape: [usize; S],
        expected_data: [i32; ED],
        expected_shape: [usize; ES],
    ) {
        let tensor = Tensor::new(data, shape, true);
        let max_tensor = tensor.max();
        assert_eq!(max_tensor.state.borrow().data, expected_data);
        assert_eq!(max_tensor.state.borrow().shape, expected_shape);
    }

    /// Test that summing an empty tensor works correctly.
    #[test]
    fn test_empty_tensor_sum() {
        let tensor = Tensor::<i32>::new([], [0], true);
        assert_eq!(tensor.sum().state.borrow().data, vec![0]);
    }

    /// Test that computing the max of an empty tensor panics.
    #[test]
    #[should_panic(expected = "Tensor is empty")]
    fn test_empty_tensor_max() {
        let tensor = Tensor::<i32>::new([], [0], true);
        let _ = tensor.max();
    }

    /// Test that computing the min of an empty tensor panics.
    #[test]
    #[should_panic(expected = "Tensor is empty")]
    fn test_empty_tensor_min() {
        let tensor = Tensor::<i32>::new([], [0], true);
        let _ = tensor.min();
    }
}
