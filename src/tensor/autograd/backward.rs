use crate::core::{Tensor, TensorState, next_tensor_id};
use crate::tensor::autograd::graph::topological_sort;
use num_traits::One;
use std::fmt::Debug;
use std::ops::AddAssign;

impl<T> Tensor<T>
where
    T: Copy + AddAssign,
{
    /// Accumulates incoming gradients into this tensor's gradient.
    pub fn set_grad(&self, incoming: &[T]) {
        let mut state = self.state.borrow_mut();
        for (existing, incoming) in state.grad.iter_mut().zip(incoming.iter()) {
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
        self.state.borrow_mut().grad.fill(value);
    }
}

impl<T> Tensor<T>
where
    T: Clone + Default,
{
    /// Resets the gradient values to their default state.
    pub fn zero_grad(&self) {
        self.fill_grad(T::default());
    }
}

impl<T> Tensor<T>
where
    T: Clone + One + Default + Debug + Copy + AddAssign,
{
    /// Performs reverse-mode automatic differentiation from this tensor.
    pub fn backward(&self) {
        // Walk the computation graph to topologically sort it to determine order of backpropagation
        let topological_order = topological_sort(self.clone());

        // Seed the output tensor for the backpropagation
        self.fill_grad(T::one());

        // Backpropagate the gradients in reverse topological order
        for tensor in topological_order.iter().rev() {
            // Get the gradient tensor for this tensor
            let tensor_state = tensor.state.borrow();
            let output_grad = Tensor::from_state(TensorState {
                id: next_tensor_id(),
                data: tensor_state.grad.clone(),
                shape: tensor_state.shape.clone(),
                grad: tensor_state.grad.clone(),
                node: None,
            });

            // Apply the backward operation to this tensor's node
            if let Some(node) = tensor_state.node.as_ref() {
                node.operation.backward(&output_grad, &node.parents);
            }
        }
    }
}
