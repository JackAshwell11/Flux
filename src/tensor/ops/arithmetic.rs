use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Sub, SubAssign};

use crate::tensor::Tensor;

macro_rules! impl_tensor_op {
    ($arith_trait:ident, $assign_trait:ident, $arith_method:ident, $assign_method:ident, $op:tt) => {
        impl<'a, T> $arith_trait<&'a Tensor<T>> for &Tensor<T> where T: Copy + $arith_trait<Output = T>
        {
            type Output = Tensor<T>;

            /// Apply an operation to two referenced tensors.
            fn $arith_method(self, rhs: &'a Tensor<T>) -> Self::Output {
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

        impl<'a, T> $arith_trait<&'a Tensor<T>> for Tensor<T> where T: Copy + $arith_trait<Output = T>,
        {
            type Output = Tensor<T>;

            /// Apply an operation to two tensors.
            fn $arith_method(self, rhs: &'a Tensor<T>) -> Self::Output {
                (&self).$arith_method(rhs)
            }
        }

        impl<T> $arith_trait<T> for &Tensor<T> where T: Copy + $arith_trait<Output = T>
        {
            type Output = Tensor<T>;

            /// Apply an operation to a referenced tensor and a scalar.
            fn $arith_method(self, scalar: T) -> Self::Output {
                Tensor {
                    data: self.data.iter().map(|&x| x $op scalar).collect(),
                    shape: self.shape.clone(),
                }
            }
        }

        impl<'a, T> $assign_trait<&'a Tensor<T>> for Tensor<T> where T: Copy + $arith_trait<Output = T>
        {
            /// Apply an assignment operation to two referenced tensors.
            fn $assign_method(&mut self, rhs: &'a Tensor<T>) {
                for i in 0..self.data.len() {
                    let b = if rhs.data.len() == 1 { rhs.data[0] } else { rhs.data[i] };
                    self.data[i] = self.data[i] $op b;
                }
            }
        }

        impl<T> $assign_trait<Tensor<T>> for Tensor<T> where T: Copy + $arith_trait<Output = T>,
        {
            /// Apply an assignment operation to a referenced tensor and a tensor.
            fn $assign_method(&mut self, rhs: Tensor<T>) {
                self.$assign_method(&rhs);
            }
        }

        impl<T> $assign_trait<T> for Tensor<T> where T: Copy + $arith_trait<Output = T>
        {
            /// Apply an assignment operation to a referenced tensor and a scalar.
            fn $assign_method(&mut self, rhs: T) {
                for x in &mut self.data {
                    *x = (*x) $op rhs;
                }
            }
        }
    };
}

impl_tensor_op!(Add, AddAssign, add, add_assign, +);
impl_tensor_op!(Sub, SubAssign, sub, sub_assign, -);
impl_tensor_op!(Mul, MulAssign, mul, mul_assign, *);
impl_tensor_op!(Div, DivAssign, div, div_assign, /);

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
        let result = &tensor_one + &tensor_two;
        assert_eq!(result.data, expected);
        assert_eq!(result.shape, vec![expected.len()]);
    }

    /// Test that the tensor addition assignment operator works correctly.
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
        vec![6];
        "add scalar to tensor"
    )]
    fn test_add_assign<const A: usize, const B: usize, const AS: usize, const BS: usize>(
        a: [i32; A],
        a_shape: [usize; AS],
        b: [i32; B],
        b_shape: [usize; BS],
        expected: Vec<i32>,
    ) {
        let mut tensor_one = Tensor::new(a, a_shape);
        let tensor_two = Tensor::new(b, b_shape);
        tensor_one += &tensor_two;
        assert_eq!(tensor_one.data, expected);
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
        let result = &tensor_one - &tensor_two;
        assert_eq!(result.data, expected);
        assert_eq!(result.shape, vec![expected.len()]);
    }

    /// Test that the tensor subtraction assignment operator works correctly.
    #[test_case(
        [5, 6, 7],
        [3],
        [4, 5, 6],
        [3],
        vec![1, 1, 1];
        "positive integers"
    )]
    #[test_case(
        [0, 0, 0],
        [3],
        [1, 2, 3],
        [3],
        vec![-1, -2, -3];
        "subtract from zero tensor"
    )]
    #[test_case(
        [-1, -2, -3],
        [3],
        [1, 2, 3],
        [3],
        vec![-2, -4, -6];
        "negative values"
    )]
    #[test_case(
        [10],
        [1],
        [3, 4, 5],
        [3],
        vec![7];
        "scalar subtraction from tensor"
    )]
    fn test_sub_assign<const A: usize, const B: usize, const AS: usize, const BS: usize>(
        a: [i32; A],
        a_shape: [usize; AS],
        b: [i32; B],
        b_shape: [usize; BS],
        expected: Vec<i32>,
    ) {
        let mut tensor_one = Tensor::new(a, a_shape);
        let tensor_two = Tensor::new(b, b_shape);
        tensor_one -= &tensor_two;
        assert_eq!(tensor_one.data, expected);
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
        let result = &tensor_one * &tensor_two;
        assert_eq!(result.data, expected);
        assert_eq!(result.shape, vec![expected.len()]);
    }

    /// Test that the tensor multiplication assignment operator works correctly.
    #[test_case(
        [2, 3, 4],
        [3],
        [2, 2, 2],
        [3],
        vec![4, 6, 8];
        "positive integers"
    )]
    #[test_case(
        [1, 2, 0],
        [3],
        [2, 2, 2],
        [3],
        vec![2, 4, 0];
        "multiplication with zero"
    )]
    #[test_case(
        [-1, -2, -3],
        [3],
        [1, 2, 3],
        [3],
        vec![-1, -4, -9];
        "negative values"
    )]
    #[test_case(
        [10],
        [1],
        [3, 4, 5],
        [3],
        vec![30];
        "scalar multiplication"
    )]
    fn test_mul_assign<const A: usize, const B: usize, const AS: usize, const BS: usize>(
        a: [i32; A],
        a_shape: [usize; AS],
        b: [i32; B],
        b_shape: [usize; BS],
        expected: Vec<i32>,
    ) {
        let mut tensor_one = Tensor::new(a, a_shape);
        let tensor_two = Tensor::new(b, b_shape);
        tensor_one *= &tensor_two;
        assert_eq!(tensor_one.data, expected);
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
        let result = &tensor_one / &tensor_two;
        assert_eq!(result.data, expected);
        assert_eq!(result.shape, vec![expected.len()]);
    }

    /// Test that the tensor division assignment operator works correctly.
    #[test_case(
        [10, 20, 30],
        [3],
        [2, 2, 2],
        [3],
        vec![5, 10, 15];
        "positive integers"
    )]
    #[test_case(
        [0, 0, 0],
        [3],
        [2, 2, 2],
        [3],
        vec![0, 0, 0];
        "division with zero numerator"
    )]
    #[test_case(
        [-10, -20, -30],
        [3],
        [2, 2, 2],
        [3],
        vec![-5, -10, -15];
        "negative values"
    )]
    #[test_case(
        [50],
        [1],
        [5, 10, 25],
        [3],
        vec![10];
        "scalar division"
    )]
    fn test_div_assign<const A: usize, const B: usize, const AS: usize, const BS: usize>(
        a: [i32; A],
        a_shape: [usize; AS],
        b: [i32; B],
        b_shape: [usize; BS],
        expected: Vec<i32>,
    ) {
        let mut tensor_one = Tensor::new(a, a_shape);
        let tensor_two = Tensor::new(b, b_shape);
        tensor_one /= &tensor_two;
        assert_eq!(tensor_one.data, expected);
    }
}
