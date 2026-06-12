use std::ops::{Add, Div, Mul, Sub};

use crate::tensor::Tensor;

macro_rules! impl_tensor_binary_op {
    ($trait:ident, $method:ident, $op:tt) => {
        impl<T> $trait for Tensor<T> where T: $trait<Output = T> {
            type Output = Self;

            /// Apply an operation to two tensors of the same shape.
            fn $method(self, rhs: Self) -> Self::Output {
                assert_eq!(self.shape, rhs.shape);
                Self {
                    data: self
                        .data
                        .into_iter()
                        .zip(rhs.data.into_iter())
                        .map(|(a, b)| a $op b)
                        .collect(),
                    shape: self.shape,
                }
            }
        }
    };
}

macro_rules! impl_tensor_binary_op_ref {
    ($trait:ident, $method:ident, $op:tt) => {
        impl<'a, T> $trait<&'a Tensor<T>> for &Tensor<T> where T: Copy + $trait<Output = T>,
        {
            type Output = Tensor<T>;

            /// Apply an operation to two tensors of the same shape.
            fn $method(self, rhs: &'a Tensor<T>) -> Self::Output {
                assert_eq!(self.shape, rhs.shape);
                Tensor {
                    data: self
                        .data
                        .iter()
                        .zip(rhs.data.iter())
                        .map(|(a, b)| *a $op *b)
                        .collect(),
                    shape: self.shape.clone(),
                }
            }
        }
    };
}

impl_tensor_binary_op!(Add, add, +);
impl_tensor_binary_op_ref!(Add, add, +);
impl_tensor_binary_op!(Sub, sub, -);
impl_tensor_binary_op_ref!(Sub, sub, -);
impl_tensor_binary_op!(Mul, mul, *);
impl_tensor_binary_op_ref!(Mul, mul, *);
impl_tensor_binary_op!(Div, div, /);
impl_tensor_binary_op_ref!(Div, div, /);

#[cfg(test)]
mod tests {
    use super::*;
    use test_case::test_case;

    /// Test that the tensor addition operator works correctly.
    #[test_case(
        [1, 2, 3],
        [4, 5, 6],
        [3],
        vec![5, 7, 9];
        "positive integers"
    )]
    #[test_case(
        [0, 0, 0],
        [1, 2, 3],
        [3],
        vec![1, 2, 3];
        "add zero tensor"
    )]
    #[test_case(
        [-1, -2, -3],
        [1, 2, 3],
        [3],
        vec![0, 0, 0];
        "opposite values"
    )]
    fn test_add<const N: usize, const S: usize>(
        a: [i32; N],
        b: [i32; N],
        shape: [usize; S],
        expected: Vec<i32>,
    ) {
        let tensor_one = Tensor::new(a, shape);
        let tensor_two = Tensor::new(b, shape);
        let result = tensor_one + tensor_two;
        assert_eq!(result.data, expected);
        assert_eq!(result.shape, vec![expected.len()]);
    }

    /// Test that the tensor subtraction operator works correctly.
    #[test_case(
        [5, 6, 7],
        [1, 2, 3],
        [3],
        vec![4, 4, 4];
        "positive result"
    )]
    #[test_case(
        [1, 1, 1],
        [1, 1, 1],
        [3],
        vec![0, 0, 0];
        "subtract equal tensors"
    )]
    #[test_case(
        [0, 0, 0],
        [1, 2, 3],
        [3],
        vec![-1, -2, -3];
        "negative result"
    )]
    fn test_sub<const N: usize, const S: usize>(
        a: [i32; N],
        b: [i32; N],
        shape: [usize; S],
        expected: Vec<i32>,
    ) {
        let tensor_one = Tensor::new(a, shape);
        let tensor_two = Tensor::new(b, shape);
        let result = tensor_one - tensor_two;
        assert_eq!(result.data, expected);
        assert_eq!(result.shape, vec![expected.len()]);
    }

    /// Test that the tensor multiplication operator works correctly.
    #[test_case(
        [1, 2, 3],
        [4, 5, 6],
        [3],
        vec![4, 10, 18];
        "positive integers"
    )]
    #[test_case(
        [0, 1, 2],
        [10, 10, 10],
        [3],
        vec![0, 10, 20];
        "multiply by constant"
    )]
    #[test_case(
        [-1, -2, -3],
        [1, -2, 3],
        [3],
        vec![-1, 4, -9];
        "mixed signs"
    )]
    fn test_mul<const N: usize, const S: usize>(
        a: [i32; N],
        b: [i32; N],
        shape: [usize; S],
        expected: Vec<i32>,
    ) {
        let tensor_one = Tensor::new(a, shape);
        let tensor_two = Tensor::new(b, shape);
        let result = tensor_one * tensor_two;
        assert_eq!(result.data, expected);
        assert_eq!(result.shape, vec![expected.len()]);
    }

    /// Test that the tensor division operator works correctly (integer division).
    #[test_case(
        [8, 9, 10],
        [2, 3, 5],
        [3],
        vec![4, 3, 2];
        "even division"
    )]
    #[test_case(
        [10, 20, 30],
        [2, 5, 10],
        [3],
        vec![5, 4, 3];
        "different divisors"
    )]
    #[test_case(
        [3, 7, 9],
        [1, 2, 3],
        [3],
        vec![3, 3, 3];
        "integer truncation"
    )]
    fn test_div<const N: usize, const S: usize>(
        a: [i32; N],
        b: [i32; N],
        shape: [usize; S],
        expected: Vec<i32>,
    ) {
        let tensor_one = Tensor::new(a, shape);
        let tensor_two = Tensor::new(b, shape);
        let result = tensor_one / tensor_two;
        assert_eq!(result.data, expected);
        assert_eq!(result.shape, vec![expected.len()]);
    }
}
