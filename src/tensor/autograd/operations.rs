use crate::tensor::core::{Operation, Tensor};
use num_traits::{NumCast, One, Zero};
use std::fmt::Debug;
use std::ops::{AddAssign, Div, Mul, MulAssign, Neg};

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

/// The dot product operation for the autograd engine.
#[derive(Debug)]
pub(crate) struct DotOperation;

/// The summation operation for the autograd engine.
#[derive(Debug)]
pub(crate) struct SumOperation;

/// The absolute value operation for the autograd engine.
#[derive(Debug)]
pub(crate) struct AbsOperation;

/// The minimum operation for the autograd engine.
#[derive(Debug)]
pub(crate) struct MinOperation<T> {
    // The indexes of the elements that are considered to be the minimum.
    pub(crate) mask: Vec<T>,
}

/// The maximum operation for the autograd engine.
#[derive(Debug)]
pub(crate) struct MaxOperation<T> {
    // The indexes of the elements that are considered to be the maximum.
    pub(crate) mask: Vec<T>,
}

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

impl<T> Operation<T> for DotOperation
where
    T: Copy + AddAssign + Default + Mul<Output = T>,
{
    /// Propagates the incoming gradient through the dot product operation to both inputs.
    fn backward(&self, grad_output: &Tensor<T>, parents: &[Tensor<T>]) -> Vec<Tensor<T>> {
        // d(a·b)/da = b
        // d(a·b)/db = a
        let lhs = &parents[0];
        let rhs = &parents[1];
        vec![
            grad_output.clone() * rhs.clone(),
            grad_output.clone() * lhs.clone(),
        ]
    }
}

impl<T> Operation<T> for SumOperation
where
    T: Copy + One + Mul<Output = T>,
{
    /// Propagates the incoming gradient through the summation operation to its input.
    fn backward(&self, grad_output: &Tensor<T>, _parents: &[Tensor<T>]) -> Vec<Tensor<T>> {
        // d(sum(a))/da_i = 1
        vec![grad_output.clone()]
    }
}

impl<T> Operation<T> for AbsOperation
where
    T: Copy + AddAssign + Default + PartialOrd + Zero + One + Neg<Output = T>,
{
    /// Propagates the incoming gradient through the absolute value operation to its input.
    fn backward(&self, grad_output: &Tensor<T>, parents: &[Tensor<T>]) -> Vec<Tensor<T>> {
        // d(|a|)/da = sign(a)
        let gradient = parents[0].map(|x| {
            if x > T::zero() {
                T::one()
            } else if x < T::zero() {
                -T::one()
            } else {
                T::zero()
            }
        });
        vec![grad_output.clone() * gradient]
    }
}

/// Computes a new gradient tensor by multiplying the incoming gradient tensor with a mask.
fn mask_gradient<T>(mask: &[T], grad_output: &Tensor<T>) -> Tensor<T>
where
    T: Copy + MulAssign,
{
    let grad = grad_output.clone();
    for (value, &mask) in grad.data_mut().iter_mut().zip(mask.iter()) {
        *value *= mask;
    }
    grad
}

impl<T> Operation<T> for MinOperation<T>
where
    T: Copy + MulAssign + Debug,
{
    /// Propagates the incoming gradient through the minimum operation to its input.
    fn backward(&self, grad_output: &Tensor<T>, _parents: &[Tensor<T>]) -> Vec<Tensor<T>> {
        // d(min(a))/da_i = grad_output_i if a_i is a minimum, otherwise 0
        vec![mask_gradient(&self.mask, grad_output)]
    }
}

impl<T> Operation<T> for MaxOperation<T>
where
    T: Copy + MulAssign + Debug,
{
    /// Propagates the incoming gradient through the maximum operation to its input.
    fn backward(&self, grad_output: &Tensor<T>, _parents: &[Tensor<T>]) -> Vec<Tensor<T>> {
        // d(max(a))/da_i = grad_output_i if a_i is a maximum, otherwise 0
        vec![mask_gradient(&self.mask, grad_output)]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::core::Tensor;
    use test_case::test_case;

    /// Test that the addition operation computes correct gradients.
    #[test_case([2.0], [[2.0], [2.0]]; "scalar")]
    #[test_case([2.0, 3.0], [[2.0, 3.0], [2.0, 3.0]]; "vector")]
    fn test_add_backward<const N: usize>(grad_output: [f32; N], expected: [[f32; N]; 2]) {
        let grad_output = Tensor::new(grad_output, [N], true);
        let gradients = AddOperation.backward(&grad_output, &[]);
        assert_eq!(gradients[0].data(), expected[0]);
        assert_eq!(gradients[1].data(), expected[1]);
    }

    /// Test that the subtraction operation computes correct gradients.
    #[test_case([2.0], [[2.0], [-2.0]]; "scalar")]
    #[test_case([2.0, 3.0], [[2.0, 3.0], [-2.0, -3.0]]; "vector")]
    fn test_sub_backward<const N: usize>(grad_output: [f32; N], expected: [[f32; N]; 2]) {
        let grad_output = Tensor::new(grad_output, [N], true);
        let gradients = SubOperation.backward(&grad_output, &[]);
        assert_eq!(gradients[0].data(), expected[0]);
        assert_eq!(gradients[1].data(), expected[1]);
    }

    /// Test that the multiplication operation computes correct gradients.
    #[test_case(
        [2.0],
        [3.0],
        [1.0],
        [[3.0], [2.0]];
        "scalar"
    )]
    #[test_case(
        [2.0, 4.0],
        [3.0, 5.0],
        [1.0, 1.0],
        [[3.0, 5.0], [2.0, 4.0]];
        "vector"
    )]
    fn test_mul_backward<const N: usize>(
        lhs: [f32; N],
        rhs: [f32; N],
        grad_output: [f32; N],
        expected: [[f32; N]; 2],
    ) {
        let lhs = Tensor::new(lhs, [N], true);
        let rhs = Tensor::new(rhs, [N], true);
        let grad_output = Tensor::new(grad_output, [N], true);
        let gradients = MulOperation.backward(&grad_output, &[lhs, rhs]);
        assert_eq!(gradients[0].data(), expected[0]);
        assert_eq!(gradients[1].data(), expected[1]);
    }

    /// Test that the division operation computes correct gradients.
    #[test_case(
        [6.0],
        [2.0],
        [1.0],
        [[0.5], [-1.5]];
        "scalar"
    )]
    #[test_case(
        [8.0, 9.0],
        [2.0, 3.0],
        [1.0, 1.0],
        [[0.5, 0.333_333_34], [-2.0, -1.0]];
        "vector"
    )]
    fn test_div_backward<const N: usize>(
        lhs: [f32; N],
        rhs: [f32; N],
        grad_output: [f32; N],
        expected: [[f32; N]; 2],
    ) {
        let lhs = Tensor::new(lhs, [N], true);
        let rhs = Tensor::new(rhs, [N], true);
        let grad_output = Tensor::new(grad_output, [N], true);
        let gradients = DivOperation.backward(&grad_output, &[lhs, rhs]);
        assert_eq!(gradients[0].data(), expected[0]);
        assert_eq!(gradients[1].data(), expected[1]);
    }

    /// Test that the mean operation computes correct gradients.
    #[test_case(
        2,
        [1.0, 1.0],
        [0.5, 0.5];
        "two elements"
    )]
    #[test_case(
        4,
        [1.0, 1.0, 1.0, 1.0],
        [0.25, 0.25, 0.25, 0.25];
        "four elements"
    )]
    fn test_mean_backward<const N: usize>(size: usize, grad_output: [f32; N], expected: [f32; N]) {
        let grad_output = Tensor::new(grad_output, [N], true);
        let gradients = MeanOperation { size }.backward(&grad_output, &[]);
        assert_eq!(gradients[0].data(), expected);
    }

    /// Test that the negation operation computes correct gradients.
    #[test_case([2.0], [-2.0]; "scalar")]
    #[test_case([2.0, -3.0], [-2.0, 3.0]; "vector")]
    fn test_neg_backward<const N: usize>(grad_output: [f32; N], expected: [f32; N]) {
        let grad_output = Tensor::new(grad_output, [N], true);
        let gradients = NegateOperation.backward(&grad_output, &[]);
        assert_eq!(gradients[0].data(), expected);
    }

    /// Test that the dot product operation computes correct gradients.
    #[test_case(
        [2.0],
        [3.0],
        [1.0],
        [[3.0], [2.0]];
        "scalar"
    )]
    #[test_case(
        [2.0, 4.0],
        [3.0, 5.0],
        [1.0, 1.0],
        [[3.0, 5.0], [2.0, 4.0]];
        "vector"
    )]
    fn test_dot_backward<const N: usize>(
        lhs: [f32; N],
        rhs: [f32; N],
        grad_output: [f32; N],
        expected: [[f32; N]; 2],
    ) {
        let lhs = Tensor::new(lhs, [N], true);
        let rhs = Tensor::new(rhs, [N], true);
        let grad_output = Tensor::new(grad_output, [N], true);
        let gradients = DotOperation.backward(&grad_output, &[lhs, rhs]);
        assert_eq!(gradients[0].data(), expected[0]);
        assert_eq!(gradients[1].data(), expected[1]);
    }

    /// Test that the sum operation computes correct gradients.
    #[test_case([1.0], [1.0]; "scalar")]
    #[test_case([1.0, 1.0, 1.0], [1.0, 1.0, 1.0]; "vector")]
    fn test_sum_backward<const N: usize>(grad_output: [f32; N], expected: [f32; N]) {
        let grad_output = Tensor::new(grad_output, [N], true);
        let gradients = SumOperation.backward(&grad_output, &[]);
        assert_eq!(gradients[0].data(), expected);
    }

    /// Test that the absolute value operation computes correct gradients.
    #[test_case(
        [2.0],
        [3.0],
        [3.0];
        "positive scalar"
    )]
    #[test_case(
        [-2.0],
        [3.0],
        [-3.0];
        "negative scalar"
    )]
    #[test_case(
        [0.0],
        [3.0],
        [0.0];
        "zero scalar"
    )]
    #[test_case(
        [-2.0, 0.0, 4.0],
        [3.0, 3.0, 3.0],
        [-3.0, 0.0, 3.0];
        "mixed vector"
    )]
    fn test_abs_backward<const N: usize>(
        input: [f32; N],
        grad_output: [f32; N],
        expected: [f32; N],
    ) {
        let input = Tensor::new(input, [N], true);
        let grad_output = Tensor::new(grad_output, [N], true);
        let gradients = AbsOperation.backward(&grad_output, &[input]);
        assert_eq!(gradients[0].data(), expected);
    }

    /// Test that the minimum operation computes correct gradients.
    #[test_case(
        vec![1.0],
        [2.0],
        [2.0];
        "selected scalar"
    )]
    #[test_case(
        vec![0.0],
        [2.0],
        [0.0];
        "unselected scalar"
    )]
    #[test_case(
        vec![1.0, 0.0, 1.0],
        [2.0, 3.0, 4.0],
        [2.0, 0.0, 4.0];
        "multiple selected values"
    )]
    fn test_min_backward<const N: usize>(
        mask: Vec<f32>,
        grad_output: [f32; N],
        expected: [f32; N],
    ) {
        let grad_output = Tensor::new(grad_output, [N], true);
        let gradients = MinOperation { mask }.backward(&grad_output, &[]);
        assert_eq!(gradients[0].data(), expected);
    }

    /// Test that the maximum operation computes correct gradients.
    #[test_case(
        vec![1.0],
        [2.0],
        [2.0];
        "selected scalar"
    )]
    #[test_case(
        vec![0.0],
        [2.0],
        [0.0];
        "unselected scalar"
    )]
    #[test_case(
        vec![0.0, 1.0, 1.0],
        [2.0, 3.0, 4.0],
        [0.0, 3.0, 4.0];
        "multiple selected values"
    )]
    fn test_max_backward<const N: usize>(
        mask: Vec<f32>,
        grad_output: [f32; N],
        expected: [f32; N],
    ) {
        let grad_output = Tensor::new(grad_output, [N], true);
        let gradients = MaxOperation { mask }.backward(&grad_output, &[]);
        assert_eq!(gradients[0].data(), expected);
    }
}
