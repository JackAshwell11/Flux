use crate::tensor::core::Tensor;
use num_traits::Float;
use std::fmt::Debug;
use std::iter::Sum;
use std::ops::AddAssign;

/// Compute the L1 loss between two tensors.
#[must_use]
pub fn l1_loss<T>(input: Tensor<T>, target: Tensor<T>) -> Tensor<T>
where
    T: Float + Sum + Default + AddAssign + Debug,
{
    (input - target).abs().mean()
}

/// Compute the L2 loss between two tensors.
#[must_use]
pub fn l2_loss<T>(input: Tensor<T>, target: Tensor<T>) -> Tensor<T>
where
    T: Float + Sum + Default + AddAssign + Debug,
{
    let diff = input - target;
    let squared_diffs = diff.clone() * diff;
    squared_diffs.mean()
}

#[cfg(test)]
mod tests {
    use super::*;
    use test_case::test_case;

    /// Test that the L1 loss function computes the correct result.
    #[test_case(
        [1.0, 2.0, 3.0],
        [3],
        [3.0, 2.0, 1.0],
        [3],
        1.333_333_3;
        "basic l1 loss"
    )]
    #[test_case(
        [1.0, 1.0, 1.0],
        [3],
        [1.0, 1.0, 1.0],
        [3],
        0.0;
        "zero loss"
    )]
    #[test_case(
        [5.0],
        [1],
        [2.0],
        [1],
        3.0;
        "single element"
    )]
    fn test_l1_loss<const A: usize, const B: usize, const AS: usize, const BS: usize>(
        input: [f64; A],
        input_shape: [usize; AS],
        target: [f64; B],
        target_shape: [usize; BS],
        expected: f64,
    ) {
        let input_tensor = Tensor::new(input, input_shape, true);
        let target_tensor = Tensor::new(target, target_shape, true);
        let result = l1_loss(input_tensor, target_tensor);
        assert!((result.state.borrow().data[0] - expected).abs() < 1e-6);
    }

    /// Test that the L2 loss function computes the correct result.
    #[test_case(
        [1.0, 2.0, 3.0],
        [3],
        [3.0, 2.0, 1.0],
        [3],
        8.0 / 3.0;
        "basic l2 loss"
    )]
    #[test_case(
        [1.0, 1.0, 1.0],
        [3],
        [1.0, 1.0, 1.0],
        [3],
        0.0;
        "zero l2 loss"
    )]
    #[test_case(
        [2.0],
        [1],
        [5.0],
        [1],
        9.0;
        "single element squared error"
    )]
    fn test_l2_loss<const A: usize, const B: usize, const AS: usize, const BS: usize>(
        input: [f64; A],
        input_shape: [usize; AS],
        target: [f64; B],
        target_shape: [usize; BS],
        expected: f64,
    ) {
        let input_tensor = Tensor::new(input, input_shape, true);
        let target_tensor = Tensor::new(target, target_shape, true);
        let result = l2_loss(input_tensor, target_tensor);
        assert!((result.state.borrow().data[0] - expected).abs() < 1e-6);
    }
}
