use crate::core::{Operation, Tensor};
use num_traits::{NumCast, One};
use std::fmt::Debug;
use std::ops::{AddAssign, Div, Mul, Neg};

/// The add operation for the autograd engine.
#[derive(Debug)]
pub(crate) struct AddOperation;

/// The subtraction operation for the autograd engine.
#[derive(Debug)]
pub(crate) struct SubOperation;

/// The multiplication operation for the autograd engine.
#[derive(Debug)]
pub(crate) struct MulOperation;

/// The division operation for the autograd engine.
#[derive(Debug)]
pub(crate) struct DivOperation;

/// The mean operation for the autograd engine.
#[derive(Debug)]
pub(crate) struct MeanOperation {
    // The number of elements in the input tensor used to compute the mean.
    pub(crate) size: usize,
}

/// The negate operation for the autograd engine.
#[derive(Debug)]
pub(crate) struct NegateOperation;

impl<T> Operation<T> for AddOperation
where
    T: Copy + AddAssign + Default,
{
    /// Propagates the incoming gradient through the addition operation to both inputs.
    fn backward(&self, grad_output: &Tensor<T>, _parents: &[Tensor<T>]) -> Vec<Tensor<T>> {
        // d(a+b)/da = 1
        // d(a+b)/db = 1
        vec![grad_output.clone(), grad_output.clone()]
    }
}

impl<T> Operation<T> for SubOperation
where
    T: Copy + AddAssign + Neg<Output = T> + Default,
{
    /// Propagates the incoming gradient through the subtraction operation to both inputs.
    fn backward(&self, grad_output: &Tensor<T>, _parents: &[Tensor<T>]) -> Vec<Tensor<T>> {
        // d(a-b)/da = 1
        // d(a-b)/db = -1
        vec![grad_output.clone(), -grad_output.clone()]
    }
}

impl<T> Operation<T> for MulOperation
where
    T: Copy + AddAssign + Default + Mul<Output = T>,
{
    /// Propagates the incoming gradient through the multiplication operation to both inputs.
    fn backward(&self, grad_output: &Tensor<T>, parents: &[Tensor<T>]) -> Vec<Tensor<T>> {
        // d(a*b)/da = b
        // d(a*b)/db = a
        let lhs = &parents[0];
        let rhs = &parents[1];
        vec![
            grad_output.clone() * rhs.clone(),
            grad_output.clone() * lhs.clone(),
        ]
    }
}

impl<T> Operation<T> for DivOperation
where
    T: Copy + AddAssign + Default + Mul<Output = T> + Div<Output = T> + Neg<Output = T>,
{
    /// Propagates the incoming gradient through the division operation to both inputs.
    fn backward(&self, grad_output: &Tensor<T>, parents: &[Tensor<T>]) -> Vec<Tensor<T>> {
        // d(a/b)/da = 1/b
        // d(a/b)/db = -a/b²
        let lhs = &parents[0];
        let rhs = &parents[1];
        vec![
            grad_output.clone() / rhs.clone(),
            grad_output.clone() * (-lhs.clone() / (rhs.clone() * rhs.clone())),
        ]
    }
}

impl<T> Operation<T> for MeanOperation
where
    T: Copy + AddAssign + Debug + Default + Mul<Output = T> + One + Div<Output = T> + NumCast,
{
    /// Propagates the incoming gradient through the mean operation to its input.
    fn backward(&self, grad_output: &Tensor<T>, _parents: &[Tensor<T>]) -> Vec<Tensor<T>> {
        // d(mean(a))/da_i = 1/n
        let size = T::from(self.size).expect("Failed to convert size");
        vec![grad_output.clone() * (T::one() / size)]
    }
}

impl<T> Operation<T> for NegateOperation
where
    T: Copy + AddAssign + Default + Neg<Output = T>,
{
    /// Propagates the incoming gradient through the negation operation to its input.
    fn backward(&self, grad_output: &Tensor<T>, _parents: &[Tensor<T>]) -> Vec<Tensor<T>> {
        // d(-a)/da = -1
        vec![-grad_output.clone()]
    }
}
