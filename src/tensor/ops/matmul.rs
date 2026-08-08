use crate::tensor::core::{Tensor, TensorState, next_tensor_id};
use std::ops::{Add, AddAssign, Mul};

impl<T> Tensor<T>
where
    T: Default + Copy + Mul<Output = T> + Add<Output = T> + AddAssign,
{
    /// Perform a matrix multiplication on two tensors of the same shape.
    ///
    /// # Panics
    ///
    /// Panics if the shapes of `self` and `rhs` are incompatible for matrix multiplication.
    #[must_use]
    #[allow(clippy::too_many_lines)]
    pub fn matmul(self, rhs: &Self) -> Self {
        match (self.rank(), rhs.rank()) {
            (0, 0) => {
                // Both tensors are scalars, so do normal multiplication
                let value = {
                    let lhs_state = self.state.borrow();
                    let rhs_state = rhs.state.borrow();
                    lhs_state.data[0] * rhs_state.data[0]
                };
                Self::from_state(TensorState {
                    id: next_tensor_id(),
                    data: vec![value],
                    shape: vec![],
                    grad: vec![T::default(); 1],
                    node: None,
                })
            }
            (0, _) => {
                // Self is a scalar, so multiply by the right-hand side
                let (data, shape) = {
                    let lhs_state = self.state.borrow();
                    let rhs_state = rhs.state.borrow();
                    (
                        rhs_state
                            .data
                            .iter()
                            .map(|x| lhs_state.data[0] * *x)
                            .collect::<Vec<T>>(),
                        rhs_state.shape.clone(),
                    )
                };
                let size = data.len();
                Self::from_state(TensorState {
                    id: next_tensor_id(),
                    data,
                    shape,
                    grad: vec![T::default(); size],
                    node: None,
                })
            }
            (_, 0) => {
                // Rhs is a scalar, so multiply by the left-hand side
                let (data, shape) = {
                    let lhs_state = self.state.borrow();
                    let rhs_state = rhs.state.borrow();
                    (
                        lhs_state
                            .data
                            .iter()
                            .map(|x| *x * rhs_state.data[0])
                            .collect::<Vec<T>>(),
                        lhs_state.shape.clone(),
                    )
                };
                let size = data.len();
                Self::from_state(TensorState {
                    id: next_tensor_id(),
                    data,
                    shape,
                    grad: vec![T::default(); size],
                    node: None,
                })
            }
            (1, 1) => {
                // Both tensors are vectors, so perform vector dot product
                self.dot(rhs)
            }
            (1, 2) => {
                // Self is a vector, so multiply by the right-hand side
                let (data, shape) = {
                    let lhs_state = self.state.borrow();
                    let rhs_state = rhs.state.borrow();
                    assert_eq!(lhs_state.shape[0], rhs_state.shape[0]);
                    let right_columns = rhs_state.shape[1];
                    let shared_dimension = lhs_state.shape[0];
                    let mut result = vec![T::default(); right_columns];
                    for (col, val) in result.iter_mut().enumerate() {
                        for i in 0..shared_dimension {
                            *val += lhs_state.data[i] * rhs_state.data[i * right_columns + col];
                        }
                    }
                    (result, vec![right_columns])
                };
                let size = data.len();
                Self::from_state(TensorState {
                    id: next_tensor_id(),
                    data,
                    shape,
                    grad: vec![T::default(); size],
                    node: None,
                })
            }
            (2, 1) => {
                // Rhs is a vector, so multiply by the left-hand side
                let (data, shape) = {
                    let lhs_state = self.state.borrow();
                    let rhs_state = rhs.state.borrow();
                    assert_eq!(lhs_state.shape[1], rhs_state.shape[0]);
                    let left_rows = lhs_state.shape[0];
                    let shared_dimension = lhs_state.shape[1];
                    let mut result = vec![T::default(); left_rows];
                    for (row, val) in result.iter_mut().enumerate() {
                        for col in 0..shared_dimension {
                            *val +=
                                lhs_state.data[row * shared_dimension + col] * rhs_state.data[col];
                        }
                    }
                    (result, vec![left_rows])
                };
                let size = data.len();
                Self::from_state(TensorState {
                    id: next_tensor_id(),
                    data,
                    shape,
                    grad: vec![T::default(); size],
                    node: None,
                })
            }
            (2, 2) => {
                // Both tensors are matrices, so perform matrix multiplication
                let (data, shape) = {
                    let lhs_state = self.state.borrow();
                    let rhs_state = rhs.state.borrow();
                    assert_eq!(lhs_state.shape[1], rhs_state.shape[0]);
                    let left_rows = lhs_state.shape[0];
                    let shared_dimension = lhs_state.shape[1];
                    let right_columns = rhs_state.shape[1];
                    let mut result = vec![T::default(); left_rows * right_columns];
                    for i in 0..left_rows {
                        for j in 0..right_columns {
                            for k in 0..shared_dimension {
                                result[i * right_columns + j] += lhs_state.data
                                    [i * shared_dimension + k]
                                    * rhs_state.data[k * right_columns + j];
                            }
                        }
                    }
                    (result, vec![left_rows, right_columns])
                };
                let size = data.len();
                Self::from_state(TensorState {
                    id: next_tensor_id(),
                    data,
                    shape,
                    grad: vec![T::default(); size],
                    node: None,
                })
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
        [2],
        [];
        "0D scalar * scalar"
    )]
    #[test_case(
        [1],
        [],
        [1, 2, 3],
        [3],
        [1, 2, 3],
        [3];
        "0D scalar * 1D vector"
    )]
    #[test_case(
        [1, 2, 3],
        [3],
        [1],
        [],
        [1, 2, 3],
        [3];
        "1D vector * 0D scalar"
    )]
    #[test_case(
        [1, 2, 3],
        [3],
        [4, 5, 6],
        [3],
        [32],
        [];
        "1D vector * 1D vector"
    )]
    #[test_case(
        [1, 2, 3],
        [3],
        [4, 5, 6, 7, 8, 9],
        [3, 2],
        [40, 46],
        [2];
        "1D vector * 2D matrix"
    )]
    #[test_case(
        [1, 2, 3, 4, 5, 6],
        [2, 3],
        [7, 8, 9],
        [3],
        [50, 122],
        [2];
        "2D matrix * 1D vector"
    )]
    #[test_case(
        [1, 2, 3, 4],
        [2, 2],
        [5, 6, 7, 8],
        [2, 2],
        [19, 22, 43, 50],
        [2, 2];
        "square 2x2"
    )]
    #[test_case(
        [1, 2, 3, 4],
        [2, 2],
        [1, 0, 0, 1],
        [2, 2],
        [1, 2, 3, 4],
        [2, 2];
        "identity matrix"
    )]
    #[test_case(
        [1, 2, 3, 4, 5, 6],
        [2, 3],
        [7, 8, 9, 10, 11, 12],
        [3, 2],
        [58, 64, 139, 154],
        [2, 2];
        "rectangular 2x3 * 3x2"
    )]
    #[test_case(
        [1, 2, 3],
        [1, 3],
        [4, 5, 6],
        [3, 1],
        [32],
        [1, 1];
        "row vector * column vector"
    )]
    #[test_case(
        [1, 2, 3, 4],
        [4, 1],
        [5, 6, 7, 8],
        [1, 4],
        [5, 6, 7, 8, 10, 12, 14, 16, 15, 18, 21, 24, 20, 24, 28, 32],
        [4, 4];
        "column vector * row vector"
    )]
    #[test_case(
        [-1, 2, -3, 4],
        [2, 2],
        [5, -6, 7, -8],
        [2, 2],
        [9, -10, 13, -14],
        [2, 2];
        "mixed signs"
    )]
    fn test_matmul<
        const A: usize,
        const B: usize,
        const AS: usize,
        const BS: usize,
        const ED: usize,
        const ES: usize,
    >(
        a: [i32; A],
        a_shape: [usize; AS],
        b: [i32; B],
        b_shape: [usize; BS],
        expected_data: [i32; ED],
        expected_shape: [usize; ES],
    ) {
        let tensor_one = Tensor::new(a, a_shape);
        let tensor_two = Tensor::new(b, b_shape);
        let result = tensor_one.matmul(&tensor_two);
        assert_eq!(result.state.borrow().data, expected_data);
        assert_eq!(result.state.borrow().shape, expected_shape);
    }

    /// Test that matrix multiplication panics for unsupported dimensionality or incompatible shapes.
    // #[test_case(
    //     [1, 2],
    //     [2],
    //     [3, 4, 5],
    //     [3];
    //     "incompatible 1D vectors"
    // )]
    // #[test_case(
    //     [1, 2, 3],
    //     [1, 3],
    //     [4],
    //     [1, 1];
    //     "incompatible 2D shapes"
    // )]
    #[test_case(
        [1, 2, 3, 4],
        [1, 2, 2],
        [5, 6, 7, 8],
        [1, 2, 2] => panics "not implemented: Matrix multiplication for tensors of rank 3 and 3 is not implemented";
        "3D x 3D not implemented"
    )]
    #[test_case(
        [1, 2, 3, 4],
        [1, 2, 2],
        [5, 6],
        [2] => panics "not implemented: Matrix multiplication for tensors of rank 3 and 1 is not implemented";
        "3D x 1D not implemented"
    )]
    #[test_case(
        [1, 2],
        [2],
        [3, 4, 5, 6],
        [1, 2, 2] => panics "not implemented: Matrix multiplication for tensors of rank 1 and 3 is not implemented";
        "1D x 3D not implemented"
    )]
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
        let _ = tensor_one.matmul(&tensor_two);
    }
}
