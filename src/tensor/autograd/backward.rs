use crate::tensor::autograd::graph::topological_sort;
use crate::tensor::broadcast::broadcast_backward;
use crate::tensor::core::Tensor;
use num_traits::One;
use std::fmt::Debug;
use std::ops::AddAssign;

impl<T> Tensor<T>
where
    T: Copy + AddAssign + Default,
{
    /// Accumulates incoming gradients into this tensor's gradient.
    pub fn accumulate_grad(&self, incoming: &[T]) {
        let mut grad = self.grad_mut();
        let incoming = broadcast_backward(incoming, grad.len());
        for (existing, incoming) in grad.iter_mut().zip(incoming.iter()) {
            *existing += *incoming;
        }
    }
}

impl<T> Tensor<T>
where
    T: Clone,
{
    /// Fills the gradient values with the provided value.
    pub fn fill_grad(&self, value: T) {
        self.grad_mut().fill(value);
    }
}

impl<T> Tensor<T>
where
    T: Clone + One + Default + Debug + Copy + AddAssign,
{
    /// Performs reverse-mode automatic differentiation from this tensor.
    pub fn backward(&self) {
        // Skip if the tensor does not require gradients
        if !self.requires_grad() {
            return;
        }

        // Walk the computation graph to topologically sort it to determine order of backpropagation
        let topological_order = topological_sort(&self.clone());

        // Seed the output tensor for the backpropagation
        self.fill_grad(T::one());

        // Backpropagate the gradients in reverse topological order
        for tensor in topological_order.iter().rev() {
            // Get the gradient tensor for this tensor
            let output_grad = Self::new(tensor.grad(), tensor.shape(), false);

            // Apply the backward operation to this tensor's node
            if let Some(node) = tensor.node().as_ref() {
                node.parents
                    .iter()
                    .zip(node.operation.backward(&output_grad, &node.parents))
                    .for_each(|(parent, grad)| {
                        if parent.requires_grad() {
                            parent.accumulate_grad(&grad.data());
                        }
                    });
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use test_case::test_case;

    /// Test that gradients are accumulated correctly.
    #[test_case(
        [1.0, 2.0, 3.0],
        [0.0, 0.0, 0.0],
        [1.0, 2.0, 3.0];
        "empty gradient"
    )]
    #[test_case(
        [1.0, 2.0, 3.0],
        [1.0, 1.0, 1.0],
        [2.0, 3.0, 4.0];
        "existing gradient"
    )]
    fn test_accumulate_grad<const N: usize>(
        incoming: [f32; N],
        initial: [f32; N],
        expected: [f32; N],
    ) {
        let tensor = Tensor::new([0.0; N], [N], true);
        *tensor.grad_mut() = initial.to_vec();
        tensor.accumulate_grad(&incoming);
        assert_eq!(tensor.grad(), expected);
    }

    /// Test that gradients can be filled with a value.
    #[test_case(
        [0.0, 0.0, 0.0],
        5.0,
        [5.0, 5.0, 5.0];
        "fill gradient"
    )]
    #[test_case(
        [0.0],
        -2.0,
        [-2.0];
        "fill scalar gradient"
    )]
    fn test_fill_grad<const N: usize>(initial: [f32; N], value: f32, expected: [f32; N]) {
        let tensor = Tensor::new([1.0; N], [N], true);
        *tensor.grad_mut() = initial.to_vec();
        tensor.fill_grad(value);
        assert_eq!(tensor.grad(), expected);
    }

    /// Test reverse-mode differentiation through a simple graph.
    #[test_case(
        [1.0, 2.0],
        [3.0, 4.0],
        [3.0, 8.0],
        [3.0, 4.0],
        [1.0, 2.0];
        "multiplication graph"
    )]
    #[test_case(
        [2.0, -1.0],
        [-3.0, 4.0],
        [-6.0, -4.0],
        [-3.0, 4.0],
        [2.0, -1.0];
        "negative multiplication graph"
    )]
    fn test_backward<const N: usize>(
        lhs: [f32; N],
        rhs: [f32; N],
        expected_output: [f32; N],
        expected_grad_lhs: [f32; N],
        expected_grad_rhs: [f32; N],
    ) {
        let tensor_a = Tensor::new(lhs, [N], true);
        let tensor_b = Tensor::new(rhs, [N], true);
        let tensor_c = tensor_a.clone() * tensor_b.clone();
        assert_eq!(tensor_c.data(), expected_output);
        tensor_c.backward();
        assert_eq!(tensor_a.grad(), expected_grad_lhs);
        assert_eq!(tensor_b.grad(), expected_grad_rhs);
    }

    /// Test that gradients accumulate through shared graph branches.
    #[test_case(
        [2.0],
        [5.0];
        "scalar square plus identity"
    )]
    #[test_case(
        [-3.0],
        [-5.0];
        "negative scalar dependency"
    )]
    #[test_case(
        [2.0, 3.0],
        [5.0, 7.0];
        "vector square plus identity"
    )]
    fn test_backward_shared_dependency<const N: usize>(value: [f32; N], expected_grad: [f32; N]) {
        // d(x*x+x)/dx = 2x+1
        let tensor = Tensor::new(value, [N], true);
        let square = tensor.clone() * tensor.clone();
        let output = square + tensor.clone();
        output.backward();
        assert_eq!(tensor.grad(), expected_grad);
    }

    /// Test that gradients accumulate through multiple backward calls.
    #[test_case(
        [2.0],
        [8.0];
        "scalar accumulation"
    )]
    #[test_case(
        [2.0, 3.0],
        [8.0, 12.0];
        "vector accumulation"
    )]
    fn test_backward_accumulates_gradients<const N: usize>(
        value: [f32; N],
        expected_grad: [f32; N],
    ) {
        // d(x*x)/dx = 2x
        // Each backward call contributes another gradient
        let tensor = Tensor::new(value, [N], true);
        let output = tensor.clone() * tensor.clone();
        output.backward();
        output.backward();
        assert_eq!(tensor.grad(), expected_grad);
    }

    /// Test that gradients propagate through chained operations.
    #[test_case(
        [2.0],
        [3.0],
        [7.0],
        [2.0];
        "scalar addition then multiplication"
    )]
    #[test_case(
        [-1.0],
        [4.0],
        [2.0],
        [-1.0];
        "negative scalar values"
    )]
    #[test_case(
        [2.0, 3.0],
        [3.0, 4.0],
        [7.0, 10.0],
        [2.0, 3.0];
        "vector addition then multiplication"
    )]
    fn test_backward_chain<const N: usize>(
        lhs: [f32; N],
        rhs: [f32; N],
        expected_grad_lhs: [f32; N],
        expected_grad_rhs: [f32; N],
    ) {
        let tensor_a = Tensor::new(lhs, [N], true);
        let tensor_b = Tensor::new(rhs, [N], true);
        let tensor_c = tensor_a.clone() + tensor_b.clone();
        let tensor_d = tensor_c * tensor_a.clone();
        tensor_d.backward();
        assert_eq!(tensor_a.grad(), expected_grad_lhs);
        assert_eq!(tensor_b.grad(), expected_grad_rhs);
    }

    /// Test that `zero_grad()` clears gradients after backward.
    #[test_case(
        [2.0],
        [3.0];
        "scalar zero after backward"
    )]
    #[test_case(
        [2.0, 3.0],
        [4.0, 5.0];
        "vector zero after backward"
    )]
    fn test_zero_grad_after_backward<const N: usize>(lhs: [f32; N], rhs: [f32; N]) {
        let tensor_a = Tensor::new(lhs, [N], true);
        let tensor_b = Tensor::new(rhs, [N], true);
        let output = tensor_a.clone() * tensor_b.clone();
        output.backward();
        tensor_a.fill_grad(0.0);
        tensor_b.fill_grad(0.0);
        assert_eq!(tensor_a.grad(), vec![0.0; N]);
        assert_eq!(tensor_b.grad(), vec![0.0; N]);
    }

    /// Test gradient propagation through division.
    #[test_case(
        [6.0],
        [2.0],
        [0.5];
        "scalar division"
    )]
    #[test_case(
        [6.0, 8.0],
        [2.0, 4.0],
        [0.5, 0.25];
        "vector division"
    )]
    fn test_backward_division<const N: usize>(
        lhs: [f32; N],
        rhs: [f32; N],
        expected_grad: [f32; N],
    ) {
        let tensor_a = Tensor::new(lhs, [N], true);
        let tensor_b = Tensor::new(rhs, [N], true);
        let output = tensor_a.clone() / tensor_b;
        output.backward();
        assert_eq!(tensor_a.grad(), expected_grad);
    }

    /// Test gradient propagation through negation.
    #[test_case(
        [2.0],
        [-1.0];
        "scalar negation"
    )]
    #[test_case(
        [2.0, -3.0],
        [-1.0, -1.0];
        "vector negation"
    )]
    fn test_backward_negation<const N: usize>(value: [f32; N], expected_grad: [f32; N]) {
        let tensor = Tensor::new(value, [N], true);
        let output = -tensor.clone();
        output.backward();
        assert_eq!(tensor.grad(), expected_grad);
    }

    /// Test that backward only accumulates gradients for tensors that require gradients.
    #[test_case(
        [2.0],
        [3.0],
        false,
        false,
        [0.0],
        [0.0];
        "scalar no tensors require grad"
    )]
    #[test_case(
        [2.0],
        [3.0],
        true,
        false,
        [3.0],
        [0.0];
        "scalar only lhs requires grad"
    )]
    #[test_case(
        [2.0],
        [3.0],
        false,
        true,
        [0.0],
        [2.0];
        "scalar only rhs requires grad"
    )]
    #[test_case(
        [2.0, 3.0],
        [4.0, 5.0],
        true,
        false,
        [4.0, 5.0],
        [0.0, 0.0];
        "vector only lhs requires grad"
    )]
    fn test_backward_respects_requires_grad<const N: usize>(
        lhs: [f32; N],
        rhs: [f32; N],
        lhs_requires_grad: bool,
        rhs_requires_grad: bool,
        expected_grad_lhs: [f32; N],
        expected_grad_rhs: [f32; N],
    ) {
        let tensor_one = Tensor::new(lhs, [N], lhs_requires_grad);
        let tensor_two = Tensor::new(rhs, [N], rhs_requires_grad);
        let output = tensor_one.clone() * tensor_two.clone();
        output.backward();
        assert_eq!(tensor_one.grad(), expected_grad_lhs);
        assert_eq!(tensor_two.grad(), expected_grad_rhs);
    }

    /// Test that backward seeds gradients on leaf tensors.
    #[test_case(
        [2.0],
        [1.0];
        "scalar leaf"
    )]
    #[test_case(
        [2.0, 3.0],
        [1.0, 1.0];
        "vector leaf"
    )]
    fn test_backward_leaf_tensor<const N: usize>(value: [f32; N], expected_grad: [f32; N]) {
        let tensor = Tensor::new(value, [N], true);
        tensor.backward();
        assert_eq!(tensor.grad(), expected_grad);
    }
}
