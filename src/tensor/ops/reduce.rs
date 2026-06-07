use crate::tensor::Tensor;
use std::iter::Sum;

impl<T> Tensor<T>
where
    T: Sum + Copy,
{
    /// Sum the elements of a tensor.
    pub fn sum(&self) -> T {
        self.data.iter().copied().sum()
    }
}

impl<T> Tensor<T>
where
    T: Copy + Sum + Into<f64>,
{
    /// Compute the mean of a tensor.
    pub fn mean(&self) -> Option<f64> {
        if self.data.is_empty() {
            return None;
        }
        let sum = self.sum().into();
        Some(sum / self.data.len() as f64)
    }
}

impl<T> Tensor<T>
where
    T: Copy + PartialOrd,
{
    /// Get the minimum value of the tensor.
    pub fn min(&self) -> Option<T> {
        self.data
            .iter()
            .copied()
            .reduce(|a, b| if a <= b { a } else { b })
    }

    /// Get the maximum value of the tensor.
    pub fn max(&self) -> Option<T> {
        self.data
            .iter()
            .copied()
            .reduce(|a, b| if a >= b { a } else { b })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use test_case::test_case;

    /// Test that summing tensor elements works correctly.
    #[test_case([1, 2, 3], [3], 6; "positive vector")]
    #[test_case([0, 0, 0], [3], 0; "all zeros")]
    #[test_case([-1, -2, -3], [3], -6; "negative vector")]
    #[test_case([-1, 2, -3, 4], [4], 2; "mixed signs")]
    #[test_case([42], [1], 42; "single element")]
    #[test_case([1, 2, 3, 4], [2, 2], 10; "matrix")]
    fn test_sum_i32<const N: usize, const S: usize>(
        data: [i32; N],
        shape: [usize; S],
        expected: i32,
    ) {
        let tensor = Tensor::new(data, shape);
        assert_eq!(tensor.sum(), expected);
    }

    /// Test that summing floating-point tensor elements works correctly.
    #[test_case([1.0, 2.0, 3.0], [3], 6.0; "positive floats")]
    #[test_case([-1.5, 2.5, 3.0], [3], 4.0; "mixed floats")]
    #[test_case([0.25, 0.25, 0.5], [3], 1.0; "fractional floats")]
    fn test_sum_f64<const N: usize, const S: usize>(
        data: [f64; N],
        shape: [usize; S],
        expected: f64,
    ) {
        let tensor = Tensor::new(data, shape);
        assert_eq!(tensor.sum(), expected);
    }

    /// Test that computing the mean of integer tensors works correctly.
    #[test_case([1, 2, 3], [3], Some(2.0); "integer mean")]
    #[test_case([1, 2, 3, 4], [4], Some(2.5); "integer mean truncates")]
    #[test_case([10, 20, 30, 40], [2, 2], Some(25.0); "matrix mean")]
    #[test_case([-3, 3], [2], Some(0.0); "mean of opposite values")]
    #[test_case([42], [1], Some(42.0); "single element")]
    fn test_mean_i32<const N: usize, const S: usize>(
        data: [i32; N],
        shape: [usize; S],
        expected: Option<f64>,
    ) {
        let tensor = Tensor::new(data, shape);
        assert_eq!(tensor.mean(), expected);
    }

    /// Test that computing the mean of floating-point tensors works correctly.
    #[test_case([1.0, 2.0, 3.0], [3], Some(2.0); "simple float mean")]
    #[test_case([1.0, 2.0, 3.0, 4.0], [4], Some(2.5); "fractional float mean")]
    #[test_case([-1.0, 1.0], [2], Some(0.0); "opposite floats")]
    #[test_case([2.5], [1], Some(2.5); "single float")]
    fn test_mean_f64<const N: usize, const S: usize>(
        data: [f64; N],
        shape: [usize; S],
        expected: Option<f64>,
    ) {
        let tensor = Tensor::new(data, shape);
        assert_eq!(tensor.mean(), expected);
    }

    /// Test that getting the min of integer tensors works correctly.
    #[test_case([1, 2, 3], [3], Some(1); "ascending")]
    #[test_case([3, 2, 1], [3], Some(1); "descending")]
    #[test_case([-10, -5, -20], [3], Some(-20); "negative values")]
    #[test_case([-1, 0, 1], [3], Some(-1); "mixed signs")]
    #[test_case([42], [1], Some(42); "single element")]
    #[test_case([1, 9, 3, 7], [2, 2], Some(1); "matrix")]
    fn test_min_i32<const N: usize, const S: usize>(
        data: [i32; N],
        shape: [usize; S],
        expected: Option<i32>,
    ) {
        let tensor = Tensor::new(data, shape);
        assert_eq!(tensor.min(), expected);
    }

    /// Test that getting the min of floating-point tensors works correctly.
    #[test_case([1.0, 2.5, 3.0], [3], Some(1.0); "positive floats")]
    #[test_case([-1.5, -0.5, -3.0], [3], Some(-3.0); "negative floats")]
    #[test_case([2.25], [1], Some(2.25); "single float")]
    fn test_min_f64<const N: usize, const S: usize>(
        data: [f64; N],
        shape: [usize; S],
        expected: Option<f64>,
    ) {
        let tensor = Tensor::new(data, shape);
        assert_eq!(tensor.min(), expected);
    }

    /// Test that getting the max of integer tensors works correctly.
    #[test_case([1, 2, 3], [3], Some(3); "ascending")]
    #[test_case([3, 2, 1], [3], Some(3); "descending")]
    #[test_case([-10, -5, -20], [3], Some(-5); "negative values")]
    #[test_case([-1, 0, 1], [3], Some(1); "mixed signs")]
    #[test_case([42], [1], Some(42); "single element")]
    #[test_case([1, 9, 3, 7], [2, 2], Some(9); "matrix")]
    fn test_max_i32<const N: usize, const S: usize>(
        data: [i32; N],
        shape: [usize; S],
        expected: Option<i32>,
    ) {
        let tensor = Tensor::new(data, shape);
        assert_eq!(tensor.max(), expected);
    }

    /// Test that getting the max of floating-point tensors works correctly.
    #[test_case([1.0, 2.5, 3.0], [3], Some(3.0); "positive floats")]
    #[test_case([-1.5, -0.5, -3.0], [3], Some(-0.5); "negative floats")]
    #[test_case([2.25], [1], Some(2.25); "single float")]
    fn test_max_f64<const N: usize, const S: usize>(
        data: [f64; N],
        shape: [usize; S],
        expected: Option<f64>,
    ) {
        let tensor = Tensor::new(data, shape);
        assert_eq!(tensor.max(), expected);
    }

    /// Test that reduction on empty tensors panics.
    #[test]
    fn test_empty_tensor_reductions() {
        let tensor = Tensor::<i32>::new([], [0]);
        assert_eq!(tensor.sum(), 0);
        assert_eq!(tensor.max(), None);
        assert_eq!(tensor.min(), None);
    }
}
