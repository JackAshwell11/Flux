use crate::Tensor;
use std::iter::Sum;

use num_traits::Float;

/// Compute the L2 loss between two tensors.
pub fn l2_loss<T>(input: &Tensor<T>, target: &Tensor<T>) -> Tensor<T>
where
    T: Float + Sum,
{
    let squared_diffs = (input - target).pow(T::from(2.0).expect("Failed to convert 2.0 into T"));
    squared_diffs.mean()
}

#[cfg(test)]
mod tests {
    use super::*;
    use test_case::test_case;

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
        let input_tensor = Tensor::new(input, input_shape);
        let target_tensor = Tensor::new(target, target_shape);
        let result = l2_loss(&input_tensor, &target_tensor);
        assert!((result.data[0] - expected).abs() < 1e-6);
    }
}
