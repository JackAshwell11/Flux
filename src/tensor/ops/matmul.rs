use std::ops::{Add, AddAssign, Mul};

use crate::tensor::Tensor;

impl<T> Tensor<T>
where
    T: Default + Copy + Mul<Output = T> + Add<Output = T> + AddAssign,
{
    /// Perform a matrix multiplication on two tensors of the same shape.
    pub fn matmul(self, rhs: Self) -> Self {
        match (self.rank(), rhs.rank()) {
            (0, 0) => {
                // Both tensors are scalars, so do normal multiplication
                Tensor {
                    data: vec![self.data[0] * rhs.data[0]],
                    shape: vec![],
                }
            }
            (0, _) => {
                // Self is a scalar, so multiply by the right-hand side
                Tensor {
                    data: rhs.data.iter().map(|x| self.data[0] * *x).collect(),
                    shape: rhs.shape,
                }
            }
            (_, 0) => {
                // Rhs is a scalar, so multiply by the left-hand side
                Tensor {
                    data: self.data.iter().map(|x| *x * rhs.data[0]).collect(),
                    shape: self.shape,
                }
            }
            (1, 1) => {
                // Both tensors are vectors, so perform vector dot product
                Tensor {
                    data: vec![self.dot(&rhs)],
                    shape: vec![],
                }
            }
            (1, 2) => {
                // Self is a vector, so multiply by the right-hand side
                assert_eq!(self.shape[0], rhs.shape[0]);
                let right_columns = rhs.shape[1];
                let shared_dimension = self.shape[0];
                let mut result = Tensor::zeros(vec![right_columns]);
                for col in 0..right_columns {
                    for i in 0..shared_dimension {
                        result.data[col] += self.data[i] * rhs.data[i * right_columns + col];
                    }
                }
                result
            }
            (2, 1) => {
                // Rhs is a vector, so multiply by the left-hand side
                assert_eq!(self.shape[1], rhs.shape[0]);
                let left_rows = self.shape[0];
                let shared_dimension = self.shape[1];
                let mut result = Tensor::zeros(vec![left_rows]);
                for row in 0..left_rows {
                    for col in 0..shared_dimension {
                        result.data[row] += self.data[row * shared_dimension + col] * rhs.data[col];
                    }
                }
                result
            }
            (2, 2) => {
                // Both tensors are matrices, so perform matrix multiplication
                assert_eq!(self.shape[1], rhs.shape[0]);
                let left_rows = self.shape[0];
                let shared_dimension = self.shape[1];
                let right_columns = rhs.shape[1];
                let mut result = Tensor::zeros(vec![left_rows, right_columns]);
                for i in 0..left_rows {
                    for j in 0..right_columns {
                        for k in 0..shared_dimension {
                            result.data[i * right_columns + j] += self.data
                                [i * shared_dimension + k]
                                * rhs.data[k * right_columns + j];
                        }
                    }
                }
                result
            }
            _ => unimplemented!(
                "Matrix multiplication for tensors of rank {} and {} is not implemented",
                self.rank(),
                rhs.rank()
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use test_case::test_case;

    /// Test that the tensor matrix multiplication operator works correctly.
    #[test_case(
        [1],
        [],
        [2],
        [],
        vec![2],
        vec![];
        "0D scalar * scalar"
    )]
    #[test_case(
        [1],
        [],
        [1, 2, 3],
        [3],
        vec![1, 2, 3],
        vec![3];
        "0D scalar * 1D vector"
    )]
    #[test_case(
        [1, 2, 3],
        [3],
        [1],
        [],
        vec![1, 2, 3],
        vec![3];
        "1D vector * 0D scalar"
    )]
    #[test_case(
        [1, 2, 3],
        [3],
        [4, 5, 6],
        [3],
        vec![32],
        vec![];
        "1D vector * 1D vector"
    )]
    #[test_case(
        [1, 2, 3],
        [3],
        [4, 5, 6, 7, 8, 9],
        [3, 2],
        vec![40, 46],
        vec![2];
        "1D vector * 2D matrix"
    )]
    #[test_case(
        [1, 2, 3, 4, 5, 6],
        [2, 3],
        [7, 8, 9],
        [3],
        vec![50, 122],
        vec![2];
        "2D matrix * 1D vector"
    )]
    #[test_case(
        [1, 2, 3, 4],
        [2, 2],
        [5, 6, 7, 8],
        [2, 2],
        vec![19, 22, 43, 50],
        vec![2, 2];
        "square 2x2"
    )]
    #[test_case(
        [1, 2, 3, 4],
        [2, 2],
        [1, 0, 0, 1],
        [2, 2],
        vec![1, 2, 3, 4],
        vec![2, 2];
        "identity matrix"
    )]
    #[test_case(
        [1, 2, 3, 4, 5, 6],
        [2, 3],
        [7, 8, 9, 10, 11, 12],
        [3, 2],
        vec![58, 64, 139, 154],
        vec![2, 2];
        "rectangular 2x3 * 3x2"
    )]
    #[test_case(
        [1, 2, 3],
        [1, 3],
        [4, 5, 6],
        [3, 1],
        vec![32],
        vec![1, 1];
        "row vector * column vector"
    )]
    #[test_case(
        [1, 2, 3, 4],
        [4, 1],
        [5, 6, 7, 8],
        [1, 4],
        vec![5, 6, 7, 8, 10, 12, 14, 16, 15, 18, 21, 24, 20, 24, 28, 32],
        vec![4, 4];
        "column vector * row vector"
    )]
    #[test_case(
        [-1, 2, -3, 4],
        [2, 2],
        [5, -6, 7, -8],
        [2, 2],
        vec![9, -10, 13, -14],
        vec![2, 2];
        "mixed signs"
    )]
    fn test_matmul<const A: usize, const B: usize, const AS: usize, const BS: usize>(
        a: [i32; A],
        a_shape: [usize; AS],
        b: [i32; B],
        b_shape: [usize; BS],
        expected_data: Vec<i32>,
        expected_shape: Vec<usize>,
    ) {
        let tensor_one = Tensor::new(a, a_shape);
        let tensor_two = Tensor::new(b, b_shape);
        let result = tensor_one.matmul(tensor_two);
        assert_eq!(result.data, expected_data);
        assert_eq!(result.shape, expected_shape);
    }

    /// Test that matrix multiplication panics for unsupported dimensionality or incompatible shapes.
    #[test_case(
        [1, 2],
        [2],
        [3, 4, 5],
        [3];
        "incompatible 1D vectors"
    )]
    #[test_case(
        [1, 2, 3],
        [1, 3],
        [4],
        [1, 1];
        "incompatible 2D shapes"
    )]
    #[test_case(
        [1, 2, 3, 4],
        [1, 2, 2],
        [5, 6, 7, 8],
        [1, 2, 2];
        "3D x 3D not implemented"
    )]
    #[test_case(
        [1, 2, 3, 4],
        [1, 2, 2],
        [5, 6],
        [2];
        "3D x 1D not implemented"
    )]
    #[test_case(
        [1, 2],
        [2],
        [3, 4, 5, 6],
        [1, 2, 2];
        "1D x 3D not implemented"
    )]
    #[should_panic]
    fn test_matmul_invalid_shape<
        const A: usize,
        const B: usize,
        const AS: usize,
        const BS: usize,
    >(
        a: [i32; A],
        a_shape: [usize; AS],
        b: [i32; B],
        b_shape: [usize; BS],
    ) {
        let tensor_one = Tensor::new(a, a_shape);
        let tensor_two = Tensor::new(b, b_shape);
        tensor_one.matmul(tensor_two);
    }
}
