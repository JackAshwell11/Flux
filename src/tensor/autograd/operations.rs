use crate::core::{Operation, Tensor};
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
pub(crate) struct MeanOperation<T> {
    /// The reciprocal of the number of elements used to compute the mean.
    pub(crate) scale: T,
}

/// The negate operation for the autograd engine.
#[derive(Debug)]
pub(crate) struct NegateOperation;

impl<T> Operation<T> for AddOperation
where
    T: Copy + AddAssign + Default,
{
    /// Propagates the incoming gradient through the addition operation to both inputs.
    fn backward(&self, incoming_grad: &Tensor<T>, parents: &[Tensor<T>]) {
        // d(a+b)/da = 1
        // d(a+b)/db = 1
        let lhs = &parents[0];
        let rhs = &parents[1];
        let grad = incoming_grad.state.borrow().data.clone();
        lhs.set_grad(&grad);
        rhs.set_grad(&grad);
    }
}

impl<T> Operation<T> for SubOperation
where
    T: Copy + AddAssign + Neg<Output = T> + Default,
{
    /// Propagates the incoming gradient through the subtraction operation to both inputs.
    fn backward(&self, incoming_grad: &Tensor<T>, parents: &[Tensor<T>]) {
        // d(a-b)/da = 1
        // d(a-b)/db = -1
        let lhs = &parents[0];
        let rhs = &parents[1];
        let lhs_grad = incoming_grad.state.borrow().data.clone();
        let rhs_grad = lhs_grad.iter().map(|x| -*x).collect::<Vec<_>>();
        lhs.set_grad(&lhs_grad);
        rhs.set_grad(&rhs_grad);
    }
}

impl<T> Operation<T> for MulOperation
where
    T: Copy + AddAssign + Default + Mul<Output = T>,
{
    /// Propagates the incoming gradient through the multiplication operation to both inputs.
    fn backward(&self, incoming_grad: &Tensor<T>, parents: &[Tensor<T>]) {
        // d(a*b)/da = b
        // d(a*b)/db = a
        let lhs = &parents[0];
        let rhs = &parents[1];
        let lhs_grad = {
            let grad_state = incoming_grad.state.borrow();
            let rhs_state = rhs.state.borrow();
            grad_state
                .data
                .iter()
                .zip(rhs_state.data.iter())
                .map(|(grad, rhs)| *grad * *rhs)
                .collect::<Vec<_>>()
        };
        let rhs_grad = {
            let grad_state = incoming_grad.state.borrow();
            let lhs_state = lhs.state.borrow();
            grad_state
                .data
                .iter()
                .zip(lhs_state.data.iter())
                .map(|(grad, lhs)| *grad * *lhs)
                .collect::<Vec<_>>()
        };
        lhs.set_grad(&lhs_grad);
        rhs.set_grad(&rhs_grad);
    }
}

impl<T> Operation<T> for DivOperation
where
    T: Copy + AddAssign + Default + Mul<Output = T> + Div<Output = T> + Neg<Output = T>,
{
    /// Propagates the incoming gradient through the division operation to both inputs.
    fn backward(&self, incoming_grad: &Tensor<T>, parents: &[Tensor<T>]) {
        // d(a/b)/da = 1/b
        // d(a/b)/db = -a/b²
        let lhs = &parents[0];
        let rhs = &parents[1];
        let lhs_grad = {
            let grad_state = incoming_grad.state.borrow();
            let rhs_state = rhs.state.borrow();
            grad_state
                .data
                .iter()
                .zip(rhs_state.data.iter())
                .map(|(grad, rhs)| *grad / *rhs)
                .collect::<Vec<_>>()
        };
        let rhs_grad = {
            let grad_state = incoming_grad.state.borrow();
            let lhs_state = lhs.state.borrow();
            let rhs_state = rhs.state.borrow();
            grad_state
                .data
                .iter()
                .zip(lhs_state.data.iter().zip(rhs_state.data.iter()))
                .map(|(grad, (lhs, rhs))| *grad * -*lhs / (*rhs * *rhs))
                .collect::<Vec<_>>()
        };
        lhs.set_grad(&lhs_grad);
        rhs.set_grad(&rhs_grad);
    }
}

impl<T> Operation<T> for MeanOperation<T>
where
    T: Copy + AddAssign + Debug + Default + Mul<Output = T>,
{
    /// Propagates the incoming gradient through the mean operation to its input.
    fn backward(&self, incoming_grad: &Tensor<T>, parents: &[Tensor<T>]) {
        // d(mean(a))/da_i = 1/n
        let lhs = &parents[0];
        let grad = {
            let state = incoming_grad.state.borrow();
            state
                .data
                .iter()
                .map(|x| *x * self.scale)
                .collect::<Vec<_>>()
        };
        lhs.set_grad(&grad);
    }
}

impl<T> Operation<T> for NegateOperation
where
    T: Copy + AddAssign + Default + Neg<Output = T>,
{
    /// Propagates the incoming gradient through the negation operation to its input.
    fn backward(&self, incoming_grad: &Tensor<T>, parents: &[Tensor<T>]) {
        // d(-a)/da = -1
        let lhs = &parents[0];
        let grad = {
            let state = incoming_grad.state.borrow();
            state.data.iter().map(|x| -*x).collect::<Vec<_>>()
        };
        lhs.set_grad(&grad);
    }
}
