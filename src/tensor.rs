use std::ops::{Add, Div, Mul, Sub};

/// Represents a tensor with a given shape and data type.
#[derive(Debug)]
pub struct Tensor<T> {
    data: Vec<T>,
    shape: Vec<usize>,
}

impl<T> Tensor<T> {
    /// Create a tensor from raw data.
    pub fn new(data: impl Into<Vec<T>>, shape: impl Into<Vec<usize>>) -> Self {
        let data = data.into();
        let shape = shape.into();
        assert_eq!(data.len(), shape.iter().product::<usize>());
        Self { data, shape }
    }

    /// Perform a dot product on two tensors of the same shape.
    pub fn dot(&self, rhs: &Self) -> T
    where
        T: Copy + Mul<Output = T> + Add<Output = T>,
    {
        assert_eq!(self.shape, rhs.shape);
        self.data
            .iter()
            .zip(rhs.data.iter())
            .map(|(a, b)| *a * *b)
            .reduce(|a, b| a + b)
            .unwrap()
    }
}

impl<T> Tensor<T>
where
    T: Default + Clone,
{
    /// Create a tensor with a given shape and all zeros.
    pub fn zeros(shape: impl Into<Vec<usize>>) -> Self {
        let shape = shape.into();
        Self {
            data: vec![T::default(); shape.iter().product()],
            shape,
        }
    }
}

impl<T> Tensor<T>
where
    T: From<u8> + Clone,
{
    /// Create a tensor with a given shape and all ones.
    pub fn ones(shape: impl Into<Vec<usize>>) -> Self {
        let shape = shape.into();
        Self {
            data: vec![T::from(1); shape.iter().product()],
            shape,
        }
    }
}

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

impl_tensor_binary_op!(Add, add, +);
impl_tensor_binary_op!(Sub, sub, -);
impl_tensor_binary_op!(Mul, mul, *);
impl_tensor_binary_op!(Div, div, /);

#[cfg(test)]
mod tests {
    use super::*;
    use test_case::test_case;

    /// Test that the tensor constructor infers the shape from the data.
    #[test_case([], [0], vec![], vec![0]; "empty tensor")]
    #[test_case([1], [1], vec![1], vec![1]; "single element")]
    #[test_case([1, 2, 3, 4], [4], vec![1, 2, 3, 4], vec![4]; "one-dimensional tensor")]
    #[test_case([1, 2, 3, 4], [2, 2], vec![1, 2, 3, 4], vec![2, 2]; "two-dimensional tensor")]
    #[test_case([10, 20], [2], vec![10, 20], vec![2]; "two elements")]
    fn test_new_infers_shape<const N: usize, const S: usize>(
        data: [usize; N],
        shape: [usize; S],
        expected_data: Vec<usize>,
        expected_shape: Vec<usize>,
    ) {
        let tensor = Tensor::new(data, shape);
        assert_eq!(tensor.data, expected_data);
        assert_eq!(tensor.shape, expected_shape);
    }

    /// Test that the tensor constructor can be used with a custom shape to give data with all zero values.
    #[test_case([], vec![0], vec![]; "scalar zero")]
    #[test_case([2], vec![0; 2], vec![2]; "vector zeros")]
    #[test_case([2, 3], vec![0; 6], vec![2, 3]; "matrix zeros")]
    #[test_case([1, 1], vec![0; 1], vec![1, 1]; "single matrix zero")]
    fn test_zeros<const N: usize>(
        shape: [usize; N],
        expected_data: Vec<i32>,
        expected_shape: Vec<usize>,
    ) {
        let tensor: Tensor<i32> = Tensor::zeros(shape);
        assert_eq!(tensor.data, expected_data);
        assert_eq!(tensor.shape, expected_shape);
    }

    /// Test that the tensor constructor can be used with a custom shape to give data with all one values.
    #[test_case([], vec![1], vec![]; "scalar one")]
    #[test_case([1], vec![1], vec![1]; "single one")]
    #[test_case([2, 2], vec![1; 4], vec![2, 2]; "matrix ones")]
    #[test_case([1, 3], vec![1; 3], vec![1, 3]; "row vector ones")]
    fn test_ones<const N: usize>(
        shape: [usize; N],
        expected_data: Vec<i32>,
        expected_shape: Vec<usize>,
    ) {
        let tensor: Tensor<i32> = Tensor::ones(shape);
        assert_eq!(tensor.data, expected_data);
        assert_eq!(tensor.shape, expected_shape);
    }

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

    /// Test that the tensor dot product operator works correctly.
    #[test_case(
        [1, 2, 3],
        [4, 5, 6],
        [3],
        32;
        "basic"
    )]
    #[test_case(
        [0, 0, 0],
        [1, 2, 3],
        [3],
        0;
        "zero vector"
    )]
    #[test_case(
        [1, 1, 1],
        [1, 1, 1],
        [3],
        3;
        "ones"
    )]
    #[test_case(
        [2, 3, 4],
        [5, 6, 7],
        [3],
        56;
        "small integers"
    )]
    #[test_case(
        [-1, -2, -3],
        [1, 2, 3],
        [3],
        -14;
        "negatives"
    )]
    #[test_case(
        [1, -2, 3],
        [-4, 5, -6],
        [3],
        -32;
        "mixed signs"
    )]
    #[test_case(
        [10],
        [7],
        [1],
        70;
        "single element"
    )]
    #[test_case(
        [1, 2, 3, 4],
        [4, 3, 2, 1],
        [4],
        20;
        "reversed"
    )]
    fn test_dot<const N: usize, const S: usize>(
        a: [i32; N],
        b: [i32; N],
        shape: [usize; S],
        expected: i32,
    ) {
        let tensor_one = Tensor::new(a, shape);
        let tensor_two = Tensor::new(b, shape);
        let result = tensor_one.dot(&tensor_two);
        assert_eq!(result, expected);
    }
}
