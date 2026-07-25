use crate::core::Tensor;
use std::collections::HashSet;
use std::fmt::Debug;

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
    pub(crate) fn print_graph(&self)
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

/// Get a topologically sorted list of the tensor's dependencies.
pub(crate) fn topological_sort<T>(tensor: Tensor<T>) -> Vec<Tensor<T>> {
    let mut visited = HashSet::new();
    let mut order = Vec::new();
    walk_graph(tensor, &mut visited, 0, &mut |_, _| {}, &mut |tensor| {
        order.push(tensor.clone());
    });
    order
}
