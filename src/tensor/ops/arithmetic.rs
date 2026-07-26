use crate::autograd::operations::{
    AddOperation, DivOperation, MulOperation, NegateOperation, SubOperation,
};
use crate::core::{Operation, OperationNode, Tensor, TensorState, next_tensor_id};
use crate::tensor::broadcast::broadcast_forward;
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

/// Compute the resultant elementwise operation with two tensors.
fn compute_elementwise_tensor<T, F, O>(
    lhs: Tensor<T>,
    rhs: Tensor<T>,
    f: F,
    operation: O,
) -> Tensor<T>
where
    T: Copy + Default,
    F: Fn(T, T) -> T,
    O: Operation<T> + 'static,
{
    let size = lhs.size().max(rhs.size());
    let data = {
        let lhs_state = lhs.state.borrow();
        let rhs_state = rhs.state.borrow();
        let lhs_data = broadcast_forward(&lhs_state.data, size);
        let rhs_data = broadcast_forward(&rhs_state.data, size);
        lhs_data
            .iter()
            .zip(rhs_data.iter())
            .map(|(&x, &y)| f(x, y))
            .collect()
    };
    Tensor::from_state(TensorState {
        id: next_tensor_id(),
        data,
        shape: vec![size],
        grad: vec![T::default(); size],
        node: Some(OperationNode {
            parents: vec![lhs, rhs],
            operation: Box::new(operation),
        }),
    })
}

/// Compute the resultant elementwise operation with a tensor and a scalar.
fn compute_scalar_tensor<T, F, O>(lhs: Tensor<T>, rhs: T, f: F, operation: O) -> Tensor<T>
where
    T: Copy + Default,
    F: Fn(T, T) -> T,
    O: Operation<T> + 'static,
{
    let (data, shape) = {
        let lhs_state = lhs.state.borrow();
        (
            lhs_state.data.iter().map(|&x| f(x, rhs)).collect(),
            lhs_state.shape.clone(),
        )
    };
    Tensor::from_state(TensorState {
        id: next_tensor_id(),
        data,
        shape,
        grad: vec![T::default(); lhs.size()],
        node: Some(OperationNode {
            parents: vec![lhs],
            operation: Box::new(operation),
        }),
    })
}

/// Compute the resultant elementwise operation with a single tensor.
fn compute_unary_tensor<T, F, O>(tensor: Tensor<T>, f: F, operation: O) -> Tensor<T>
where
    T: Copy + Default,
    F: Fn(T) -> T,
    O: Operation<T> + 'static,
{
    let (data, shape) = {
        let state = tensor.state.borrow();
        (
            state.data.iter().map(|&x| f(x)).collect(),
            state.shape.clone(),
        )
    };
    Tensor::from_state(TensorState {
        id: next_tensor_id(),
        data,
        shape,
        grad: vec![T::default(); tensor.size()],
        node: Some(OperationNode {
            parents: vec![tensor],
            operation: Box::new(operation),
        }),
    })
}

/// Apply the resultant elementwise operation between a tensor and another tensor updating the
/// left-hand side tensor.
fn apply_elementwise_assign<T, F>(lhs: &mut Tensor<T>, rhs: &Tensor<T>, f: F)
where
    T: Copy,
    F: Fn(T, T) -> T,
{
    let mut lhs_state = lhs.state.borrow_mut();
    let rhs_state = rhs.state.borrow();
    let rhs_data = broadcast_forward(&rhs_state.data, lhs_state.data.len());
    lhs_state
        .data
        .iter_mut()
        .zip(rhs_data.iter())
        .for_each(|(lhs_value, rhs_value)| {
            *lhs_value = f(*lhs_value, *rhs_value);
        });
}

/// Apply the resultant elementwise operation between a tensor and a scalar updating the left-hand
/// side tensor.
fn apply_scalar_assign<T, F>(lhs: &mut Tensor<T>, rhs: T, f: F)
where
    T: Copy,
    F: Fn(T, T) -> T,
{
    let mut lhs_state = lhs.state.borrow_mut();
    for x in &mut lhs_state.data {
        *x = f(*x, rhs);
    }
}

impl<T> Add for Tensor<T>
where
    T: Copy + Add<Output = T> + Default + AddAssign,
{
    type Output = Tensor<T>;

    /// Add two referenced tensors.
    fn add(self, rhs: Self) -> Self::Output {
        compute_elementwise_tensor(self, rhs, |a, b| a + b, AddOperation)
    }
}

impl<T> Add<T> for Tensor<T>
where
    T: Copy + Add<Output = T> + Default + AddAssign,
{
    type Output = Tensor<T>;

    /// Add a referenced tensor and a scalar.
    fn add(self, scalar: T) -> Self::Output {
        compute_scalar_tensor(self, scalar, |a, b| a + b, AddOperation)
    }
}

impl<T> AddAssign<Tensor<T>> for Tensor<T>
where
    T: Copy + Add<Output = T>,
{
    /// Add a tensor to all tensor elements.
    fn add_assign(&mut self, rhs: Tensor<T>) {
        apply_elementwise_assign(self, &rhs, |a, b| a + b);
    }
}

impl<T> AddAssign<T> for Tensor<T>
where
    T: Copy + Add<Output = T>,
{
    /// Add a scalar to all tensor elements.
    fn add_assign(&mut self, rhs: T) {
        apply_scalar_assign(self, rhs, |a, b| a + b);
    }
}

impl<T> Sub for Tensor<T>
where
    T: Copy + Sub<Output = T> + Default + AddAssign + Neg<Output = T>,
{
    type Output = Tensor<T>;

    /// Subtract two referenced tensors.
    fn sub(self, rhs: Self) -> Self::Output {
        compute_elementwise_tensor(self, rhs, |a, b| a - b, SubOperation)
    }
}

impl<T> Sub<T> for Tensor<T>
where
    T: Copy + Sub<Output = T> + Default + AddAssign + Neg<Output = T>,
{
    type Output = Tensor<T>;

    /// Subtract a scalar from a referenced tensor.
    fn sub(self, scalar: T) -> Self::Output {
        compute_scalar_tensor(self, scalar, |a, b| a - b, SubOperation)
    }
}

impl<T> SubAssign<Tensor<T>> for Tensor<T>
where
    T: Copy + Sub<Output = T>,
{
    /// Subtract a tensor from all tensor elements.
    fn sub_assign(&mut self, rhs: Tensor<T>) {
        apply_elementwise_assign(self, &rhs, |a, b| a - b);
    }
}

impl<T> SubAssign<T> for Tensor<T>
where
    T: Copy + Sub<Output = T>,
{
    /// Subtract a tensor from all tensor elements.
    fn sub_assign(&mut self, rhs: T) {
        apply_scalar_assign(self, rhs, |a, b| a - b);
    }
}

impl<T> Mul for Tensor<T>
where
    T: Copy + Mul<Output = T> + Default + AddAssign,
{
    type Output = Tensor<T>;

    /// Multiply two tensors.
    fn mul(self, rhs: Self) -> Self::Output {
        compute_elementwise_tensor(self, rhs, |a, b| a * b, MulOperation)
    }
}

impl<T> Mul<T> for Tensor<T>
where
    T: Copy + Mul<Output = T> + Default + AddAssign,
{
    type Output = Tensor<T>;

    /// Multiply a tensor by a scalar.
    fn mul(self, scalar: T) -> Self::Output {
        compute_scalar_tensor(self, scalar, |a, b| a * b, MulOperation)
    }
}

impl<T> MulAssign<Tensor<T>> for Tensor<T>
where
    T: Copy + Mul<Output = T>,
{
    /// Multiply all tensor elements by a tensor.
    fn mul_assign(&mut self, rhs: Tensor<T>) {
        apply_elementwise_assign(self, &rhs, |a, b| a * b);
    }
}

impl<T> MulAssign<T> for Tensor<T>
where
    T: Copy + Mul<Output = T>,
{
    /// Multiply all tensor elements by a scalar.
    fn mul_assign(&mut self, rhs: T) {
        apply_scalar_assign(self, rhs, |a, b| a * b);
    }
}

impl<T> Div for Tensor<T>
where
    T: Copy + Div<Output = T> + Default + AddAssign + Mul<Output = T> + Neg<Output = T>,
{
    type Output = Tensor<T>;

    /// Divide two tensors.
    fn div(self, rhs: Self) -> Self::Output {
        compute_elementwise_tensor(self, rhs, |a, b| a / b, DivOperation)
    }
}

impl<T> Div<T> for Tensor<T>
where
    T: Copy + Div<Output = T> + Default + AddAssign + Mul<Output = T> + Neg<Output = T>,
{
    type Output = Tensor<T>;

    /// Divide a tensor by a scalar.
    fn div(self, scalar: T) -> Self::Output {
        compute_scalar_tensor(self, scalar, |a, b| a / b, DivOperation)
    }
}

impl<T> DivAssign<Tensor<T>> for Tensor<T>
where
    T: Copy + Div<Output = T>,
{
    /// Divide all tensor elements by a tensor.
    fn div_assign(&mut self, rhs: Tensor<T>) {
        apply_elementwise_assign(self, &rhs, |a, b| a / b);
    }
}

impl<T> DivAssign<T> for Tensor<T>
where
    T: Copy + Div<Output = T>,
{
    /// Divide all tensor elements by a scalar.
    fn div_assign(&mut self, rhs: T) {
        apply_scalar_assign(self, rhs, |a, b| a / b);
    }
}

impl<T> Neg for Tensor<T>
where
    T: Neg<Output = T> + Copy + Default + AddAssign,
{
    type Output = Tensor<T>;

    /// Negate all tensor elements.
    fn neg(self) -> Self::Output {
        compute_unary_tensor(self, |a| -a, NegateOperation)
    }
}

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
        assert_eq!(result.state.borrow().data, expected);
        assert_eq!(result.state.borrow().shape, vec![expected.len()]);
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
        tensor_one += tensor_two;
        assert_eq!(tensor_one.state.borrow().data, expected);
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
        assert_eq!(result.state.borrow().data, expected);
        assert_eq!(result.state.borrow().shape, vec![expected.len()]);
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
        tensor_one -= tensor_two;
        assert_eq!(tensor_one.state.borrow().data, expected);
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
        assert_eq!(result.state.borrow().data, expected);
        assert_eq!(result.state.borrow().shape, vec![expected.len()]);
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
        tensor_one *= tensor_two;
        assert_eq!(tensor_one.state.borrow().data, expected);
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
        assert_eq!(result.state.borrow().data, expected);
        assert_eq!(result.state.borrow().shape, vec![expected.len()]);
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
        tensor_one /= tensor_two;
        assert_eq!(tensor_one.state.borrow().data, expected);
    }
}
