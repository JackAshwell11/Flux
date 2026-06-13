use std::ops::{Add, Div, Mul, Sub};

use crate::tensor::Tensor;

macro_rules! impl_tensor_binary_op {
    ($trait:ident, $method:ident, $op:tt) => {
        impl<T> $trait for Tensor<T> where T: Copy + $trait<Output = T> {
            type Output = Self;

            /// Apply an operation to two tensors.
            fn $method(self, rhs: Self) -> Self::Output {
                let out_size = self.size().max(rhs.size());
                let mut out = Vec::with_capacity(out_size);
                for i in 0..out_size {
                    let a = if self.size() == 1 { self.data[0] } else { self.data[i] };
                    let b = if rhs.size() == 1 { rhs.data[0] } else { rhs.data[i] };
                    out.push(a $op b);
                }
                Tensor {
                    data: out,
                    shape: vec![out_size],
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

            /// Apply an operation to two tensors.
            fn $method(self, rhs: &'a Tensor<T>) -> Self::Output {
                let out_size = self.size().max(rhs.size());
                let mut out = Vec::with_capacity(out_size);
                for i in 0..out_size {
                    let a = if self.size() == 1 { self.data[0] } else { self.data[i] };
                    let b = if rhs.size() == 1 { rhs.data[0] } else { rhs.data[i] };
                    out.push(a $op b);
                }
                Tensor {
                    data: out,
                    shape: vec![out_size],
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
        [3],
        [4, 5, 6],
        [3],
        vec![5, 7, 9];
        "positive integers"
    )]
    #[test_case(
        [0, 0, 0],
        [3],
        [1, 2, 3],
        [3],
        vec![1, 2, 3];
        "add zero tensor"
    )]
    #[test_case(
        [-1, -2, -3],
        [3],
        [1, 2, 3],
        [3],
        vec![0, 0, 0];
        "opposite values"
    )]
    #[test_case(
        [1],
        [1],
        [5, 10, 15],
        [3],
        vec![6, 11, 16];
        "add scalar to tensor"
    )]
    #[test_case(
        [10],
        [1],
        [5, 5, 5],
        [3],
        vec![15, 15, 15];
        "add same scalar to all tensor elements"
    )]
    #[test_case(
        [1],
        [1],
        [0, -1, -2],
        [3],
        vec![1, 0, -1];
        "add scalar to negative values"
    )]
    fn test_add<const A: usize, const B: usize, const AS: usize, const BS: usize>(
        a: [i32; A],
        a_shape: [usize; AS],
        b: [i32; B],
        b_shape: [usize; BS],
        expected: Vec<i32>,
    ) {
        let tensor_one = Tensor::new(a, a_shape);
        let tensor_two = Tensor::new(b, b_shape);
        let result = tensor_one + tensor_two;
        assert_eq!(result.data, expected);
        assert_eq!(result.shape, vec![expected.len()]);
    }

    /// Test that the tensor subtraction operator works correctly.
    #[test_case(
        [5, 6, 7],
        [3],
        [1, 2, 3],
        [3],
        vec![4, 4, 4];
        "positive result"
    )]
    #[test_case(
        [1, 1, 1],
        [3],
        [1, 1, 1],
        [3],
        vec![0, 0, 0];
        "subtract equal tensors"
    )]
    #[test_case(
        [0, 0, 0],
        [3],
        [1, 2, 3],
        [3],
        vec![-1, -2, -3];
        "negative result"
    )]
    #[test_case(
        [10],
        [1],
        [5, 10, 15],
        [3],
        vec![5, 0, -5];
        "scalar subtraction from tensor"
    )]
    #[test_case(
        [5, 10, 15],
        [3],
        [5],
        [1],
        vec![0, 5, 10];
        "tensor subtraction from scalar"
    )]
    fn test_sub<const A: usize, const B: usize, const AS: usize, const BS: usize>(
        a: [i32; A],
        a_shape: [usize; AS],
        b: [i32; B],
        b_shape: [usize; BS],
        expected: Vec<i32>,
    ) {
        let tensor_one = Tensor::new(a, a_shape);
        let tensor_two = Tensor::new(b, b_shape);
        let result = tensor_one - tensor_two;
        assert_eq!(result.data, expected);
        assert_eq!(result.shape, vec![expected.len()]);
    }

    /// Test that the tensor multiplication operator works correctly.
    #[test_case(
        [1, 2, 3],
        [3],
        [4, 5, 6],
        [3],
        vec![4, 10, 18];
        "positive integers"
    )]
    #[test_case(
        [0, 1, 2],
        [3],
        [10, 10, 10],
        [3],
        vec![0, 10, 20];
        "multiply by constant"
    )]
    #[test_case(
        [-1, -2, -3],
        [3],
        [1, -2, 3],
        [3],
        vec![-1, 4, -9];
        "mixed signs"
    )]
    #[test_case(
        [2],
        [1],
        [1, 2, 3],
        [3],
        vec![2, 4, 6];
        "multiply scalar by tensor"
    )]
    #[test_case(
        [1, 2, 3],
        [3],
        [2],
        [1],
        vec![2, 4, 6];
        "multiply tensor by scalar"
    )]
    fn test_mul<const A: usize, const B: usize, const AS: usize, const BS: usize>(
        a: [i32; A],
        a_shape: [usize; AS],
        b: [i32; B],
        b_shape: [usize; BS],
        expected: Vec<i32>,
    ) {
        let tensor_one = Tensor::new(a, a_shape);
        let tensor_two = Tensor::new(b, b_shape);
        let result = tensor_one * tensor_two;
        assert_eq!(result.data, expected);
        assert_eq!(result.shape, vec![expected.len()]);
    }

    /// Test that the tensor division operator works correctly (integer division).
    #[test_case(
        [8, 9, 10],
        [3],
        [2, 3, 5],
        [3],
        vec![4, 3, 2];
        "even division"
    )]
    #[test_case(
        [10, 20, 30],
        [3],
        [2, 5, 10],
        [3],
        vec![5, 4, 3];
        "different divisors"
    )]
    #[test_case(
        [3, 7, 9],
        [3],
        [1, 2, 3],
        [3],
        vec![3, 3, 3];
        "integer truncation"
    )]
    #[test_case(
        [30, 60, 90],
        [3],
        [3],
        [1],
        vec![10, 20, 30];
        "tensor divided by scalar"
    )]
    #[test_case(
        [100],
        [1],
        [10, 20, 25],
        [3],
        vec![10, 5, 4];
        "scalar divided by each tensor element"
    )]
    fn test_div<const A: usize, const B: usize, const AS: usize, const BS: usize>(
        a: [i32; A],
        a_shape: [usize; AS],
        b: [i32; B],
        b_shape: [usize; BS],
        expected: Vec<i32>,
    ) {
        let tensor_one = Tensor::new(a, a_shape);
        let tensor_two = Tensor::new(b, b_shape);
        let result = tensor_one / tensor_two;
        assert_eq!(result.data, expected);
        assert_eq!(result.shape, vec![expected.len()]);
    }
}
