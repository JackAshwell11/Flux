use crate::autograd::operations::MeanOperation;
use crate::core::{OperationNode, Tensor, TensorState, next_tensor_id};
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
        Self::from_state(TensorState {
            id: next_tensor_id(),
            data: self.state.borrow().data.iter().map(|x| x.abs()).collect(),
            shape: self.state.borrow().shape.clone(),
            grad: vec![T::default(); 1],
            node: None,
        })
    }
}

impl<T> Tensor<T>
where
    T: Float + Sum + Default + AddAssign + Debug + 'static,
{
    /// Compute the mean of a tensor.
    ///
    /// # Panics
    ///
    /// Panics if the tensor length cannot be converted to `T`.
    #[must_use]
    pub fn mean(&self) -> Self {
        let (mean_val, len) = {
            let state = self.state.borrow();
            let sum: T = state.data.iter().copied().sum();
            let len = T::from(state.data.len()).expect("Failed to convert length to Float");
            (sum / len, len)
        };
        Self::from_state(TensorState {
            id: next_tensor_id(),
            data: vec![mean_val],
            shape: vec![],
            grad: vec![T::default(); 1],
            node: Some(OperationNode {
                parents: vec![self.clone()],
                operation: Box::new(MeanOperation {
                    scale: T::one() / len,
                }),
            }),
        })
    }
}

impl<T> Tensor<T>
where
    T: Copy + PartialOrd + Default,
{
    /// Compute the minimum or maximum value between two values in an iterator.
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
    #[test_case([1, 2, 3], [3], vec![6], vec![]; "positive vector")]
    #[test_case([0, 0, 0], [3], vec![0], vec![]; "all zeros")]
    #[test_case([-1, -2, -3], [3], vec![-6], vec![]; "negative vector")]
    #[test_case([-1, 2, -3, 4], [4], vec![2], vec![]; "mixed signs")]
    #[test_case([42], [1], vec![42], vec![]; "single element")]
    #[test_case([1, 2, 3, 4], [2, 2], vec![10], vec![]; "matrix")]
    fn test_sum<const N: usize, const S: usize>(
        data: [i32; N],
        shape: [usize; S],
        expected_data: Vec<i32>,
        expected_shape: Vec<usize>,
    ) {
        let tensor = Tensor::new(data, shape);
        let summed = tensor.sum();
        assert_eq!(summed.state.borrow().data, expected_data);
        assert_eq!(summed.state.borrow().shape, expected_shape);
    }

    /// Test that computing the mean of tensors works correctly.
    #[test_case([1.0, 2.0, 3.0], [3], vec![2.0], vec![]; "simple float mean")]
    #[test_case([1.0, 2.0, 3.0, 4.0], [4], vec![2.5], vec![]; "fractional float mean")]
    #[test_case([-1.0, 1.0], [2], vec![0.0], vec![]; "opposite floats")]
    #[test_case([2.5], [1], vec![2.5], vec![]; "single float")]
    fn test_mean<const N: usize, const S: usize>(
        data: [f64; N],
        shape: [usize; S],
        expected_data: Vec<f64>,
        expected_shape: Vec<usize>,
    ) {
        let tensor = Tensor::new(data, shape);
        let mean_tensor = tensor.mean();
        assert_eq!(mean_tensor.state.borrow().data, expected_data);
        assert_eq!(mean_tensor.state.borrow().shape, expected_shape);
    }

    /// Test that getting the min of a tensor works correctly.
    #[test_case([1, 2, 3], [3], vec![1], vec![]; "ascending")]
    #[test_case([3, 2, 1], [3], vec![1], vec![]; "descending")]
    #[test_case([-10, -5, -20], [3], vec![-20], vec![]; "negative values")]
    #[test_case([-1, 0, 1], [3], vec![-1], vec![]; "mixed signs")]
    #[test_case([42], [1], vec![42], vec![]; "single element")]
    #[test_case([1, 9, 3, 7], [2, 2], vec![1], vec![]; "matrix")]
    fn test_min<const N: usize, const S: usize>(
        data: [i32; N],
        shape: [usize; S],
        expected_data: Vec<i32>,
        expected_shape: Vec<usize>,
    ) {
        let tensor = Tensor::new(data, shape);
        let min_tensor = tensor.min();
        assert_eq!(min_tensor.state.borrow().data, expected_data);
        assert_eq!(min_tensor.state.borrow().shape, expected_shape);
    }

    /// Test that getting the max of a tensor works correctly.
    #[test_case([1, 2, 3], [3], vec![3], vec![]; "ascending")]
    #[test_case([3, 2, 1], [3], vec![3], vec![]; "descending")]
    #[test_case([-10, -5, -20], [3], vec![-5], vec![]; "negative values")]
    #[test_case([-1, 0, 1], [3], vec![1], vec![]; "mixed signs")]
    #[test_case([42], [1], vec![42], vec![]; "single element")]
    #[test_case([1, 9, 3, 7], [2, 2], vec![9], vec![]; "matrix")]
    fn test_max<const N: usize, const S: usize>(
        data: [i32; N],
        shape: [usize; S],
        expected_data: Vec<i32>,
        expected_shape: Vec<usize>,
    ) {
        let tensor = Tensor::new(data, shape);
        let max_tensor = tensor.max();
        assert_eq!(max_tensor.state.borrow().data, expected_data);
        assert_eq!(max_tensor.state.borrow().shape, expected_shape);
    }

    /// Test that summing an empty tensor works correctly.
    #[test]
    fn test_empty_tensor_sum() {
        let tensor = Tensor::<i32>::new([], [0]);
        assert_eq!(tensor.sum().state.borrow().data, vec![0]);
    }

    /// Test that computing the max of an empty tensor panics.
    #[test]
    #[should_panic(expected = "Tensor is empty")]
    fn test_empty_tensor_max() {
        let tensor = Tensor::<i32>::new([], [0]);
        let _ = tensor.max();
    }

    /// Test that computing the min of an empty tensor panics.
    #[test]
    #[should_panic(expected = "Tensor is empty")]
    fn test_empty_tensor_min() {
        let tensor = Tensor::<i32>::new([], [0]);
        let _ = tensor.min();
    }
}
