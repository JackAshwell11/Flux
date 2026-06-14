use crate::Tensor;
use std::iter::Sum;

use num_traits::Float;

/// Compute the L1 loss between two tensors.
#[must_use]
pub fn l1_loss<T>(input: &Tensor<T>, target: &Tensor<T>) -> Tensor<T>
where
    T: Float + Sum,
{
    let absolute_diffs = (input - target).abs();
    absolute_diffs.mean()
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
        1.3333333;
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
        let input_tensor = Tensor::new(input, input_shape);
        let target_tensor = Tensor::new(target, target_shape);
        let result = l1_loss(&input_tensor, &target_tensor);
        assert!((result.data[0] - expected).abs() < 1e-6);
    }
}
