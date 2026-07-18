use crate::core::{Operation, Tensor, TensorState, next_tensor_id};
use num_traits::One;
use std::collections::HashSet;
use std::fmt::Debug;
use std::ops::AddAssign;

/// Recursively walk a tensor computation graph and apply both a pre-order and post-order function
/// on each node in the graph.
fn walk_graph<T, Pre, Post>(
    tensor: Tensor<T>,
    visited: &mut HashSet<usize>,
    depth: usize,
    pre_order: &mut Pre,
    post_order: &mut Post,
) where
    Pre: FnMut(&Tensor<T>, usize),
    Post: FnMut(&Tensor<T>),
{
    // Check if this tensor has already been visited or not
    if !visited.insert(tensor.state.borrow().id) {
        return;
    }

    // Get the tensor and apply the pre-order graph operation
    pre_order(&tensor, depth);

    // Visit the node's parents if it exists
    let parents = {
        let tensor_state = tensor.state.borrow();
        match &tensor_state.node {
            Some(node) => node.parents.clone(),
            None => Vec::new(),
        }
    };
    for parent in parents {
        walk_graph(parent, visited, depth + 1, pre_order, post_order);
    }

    // Apply the post-order graph operation
    post_order(&tensor);
}

impl<T> Tensor<T>
where
    T: Debug + Clone,
{
    /// Print out a visual representation of the autograd graph
    pub fn print_graph(&self)
    where
        T: Clone,
    {
        let mut visited = HashSet::new();
        walk_graph(
            self.clone(),
            &mut visited,
            0,
            &mut |tensor, depth| {
                // Determine the indentation level of this node
                let prefix = if depth == 0 {
                    String::new()
                } else {
                    "│   ".repeat(depth - 1) + "├── "
                };

                // Print out this tensor's data
                let tensor_state = tensor.state.borrow();
                match &tensor_state.node {
                    Some(node) => {
                        println!("{prefix}Node {} {:?}", tensor_state.id, node.operation);
                    }
                    None => {
                        println!("{prefix}Leaf {:?}", tensor_state.data);
                    }
                }
            },
            &mut |_| {},
        );
    }
}

impl<T> Tensor<T>
where
    T: Copy + AddAssign,
{
    /// Accumulates incoming gradients into this tensor's gradient.
    pub fn set_grad(&self, incoming: &Tensor<T>) {
        let mut state = self.state.borrow_mut();
        let incoming_state = incoming.state.borrow();
        for (existing, incoming) in state.grad.iter_mut().zip(incoming_state.grad.iter()) {
            *existing += *incoming;
        }
    }
}

impl<T> Tensor<T>
where
    T: Clone + Default,
{
    /// Resets the gradient values to their default state.
    pub fn zero_grad(&self) {
        self.state.borrow_mut().grad.fill(T::default());
    }
}

impl<T> Tensor<T>
where
    T: Clone + One + Default + Debug + Copy + AddAssign,
{
    pub fn backward(&self) {
        // Walk the computation graph to topologically sort it to determine order of backpropagation
        let mut visited = HashSet::new();
        let mut topological_order = Vec::new();
        walk_graph(
            self.clone(),
            &mut visited,
            0,
            &mut |_, _| {},
            &mut |tensor| {
                topological_order.push(tensor.clone());
            },
        );

        // Seed the output tensor for the backpropagation
        self.state.borrow_mut().grad.fill(T::one());

        // Backpropagate the gradients in reverse topological order
        for tensor in topological_order.iter().rev() {
            // Extract everything we need while borrowing is active
            let (node, output_grad) = {
                let tensor_state = tensor.state.borrow();
                let output_grad = Tensor::from_state(TensorState {
                    id: next_tensor_id(),
                    data: tensor_state.grad.clone(),
                    shape: tensor_state.shape.clone(),
                    grad: vec![T::default(); tensor_state.grad.len()],
                    node: None,
                });
                (tensor_state.node.clone(), output_grad)
            };

            // If this tensor has no parents, we're done
            let Some(node) = node else {
                continue;
            };

            // Propagate the gradient to the parents of this tensor
            match node.operation {
                Operation::Add => {
                    // d(a + b) / da = 1
                    // d(a + b) / db = 1
                    let lhs = &node.parents[0];
                    let rhs = &node.parents[1];
                    lhs.set_grad(&output_grad);
                    rhs.set_grad(&output_grad);
                }
                Operation::Sub => {}
                Operation::Mul => {}
                Operation::Div => {}
                Operation::Mean => {}
            }
        }
    }
}
