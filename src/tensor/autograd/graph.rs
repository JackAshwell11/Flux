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
    #[allow(dead_code)]
    pub(crate) fn print_graph(&self) -> String
    where
        T: Clone,
    {
        let mut output = String::new();
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
                let new_str = match &tensor_state.node {
                    Some(node) => {
                        format!("{prefix}Node {} {:?}", tensor_state.id, node.operation)
                    }
                    None => {
                        format!("{prefix}Leaf {:?}", tensor_state.data)
                    }
                };
                output.push_str(&new_str);
            },
            &mut |_| {},
        );
        output
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::autograd::operations::AddOperation;
    use crate::core::{OperationNode, Tensor};
    use std::collections::HashSet;
    use test_case::test_case;

    /// A helper to construct a computation graph from node definitions.
    fn build_test_graph(nodes: Vec<(Tensor<f32>, Vec<usize>)>) -> Vec<Tensor<f32>> {
        let mut tensors: Vec<Tensor<f32>> = Vec::with_capacity(nodes.len());
        for (tensor, parents_indices) in nodes {
            if !parents_indices.is_empty() {
                tensor.state.borrow_mut().node = Some(OperationNode {
                    parents: parents_indices
                        .iter()
                        .map(|&i| tensors[i].clone())
                        .collect(),
                    operation: Box::new(AddOperation),
                });
            }
            tensors.push(tensor);
        }
        tensors
    }

    /// Test that walking the graph is performed correctly.
    #[test_case(
        vec![(Tensor::new([1.0], [1]), vec![])],
        vec![0],
        vec![0];
        "single tensor graph"
    )]
    #[test_case(
        vec![
            (Tensor::new([1.0], [1]), vec![]),
            (Tensor::new([2.0], [1]), vec![]),
            (Tensor::new([3.0], [1]), vec![0, 1]),
        ],
        vec![2, 0, 1],
        vec![0, 1, 2];
        "three node DAG"
    )]
    #[test_case(
        vec![
            (Tensor::new([1.0], [1]), vec![]),
            (Tensor::new([2.0], [1]), vec![]),
            (Tensor::new([3.0], [1]), vec![0, 1]),
            (Tensor::new([4.0], [1]), vec![2, 1]),
        ],
        vec![3, 2, 0, 1],
        vec![0, 1, 2, 3];
        "four node complex DAG"
    )]
    fn test_walk_graph(
        nodes: Vec<(Tensor<f32>, Vec<usize>)>,
        expected_pre_indices: Vec<usize>,
        expected_post_indices: Vec<usize>,
    ) {
        let tensors = build_test_graph(nodes);
        let mut pre_order = vec![];
        let mut post_order = vec![];
        walk_graph(
            tensors.last().unwrap().clone(),
            &mut HashSet::new(),
            0,
            &mut |tensor, _| pre_order.push(tensor.clone()),
            &mut |tensor| post_order.push(tensor.clone()),
        );
        assert_eq!(
            pre_order,
            expected_pre_indices
                .iter()
                .map(|&i| tensors[i].clone())
                .collect::<Vec<_>>()
        );
        assert_eq!(
            post_order,
            expected_post_indices
                .iter()
                .map(|&i| tensors[i].clone())
                .collect::<Vec<_>>()
        );
    }

    /// Test that printing a graph works correctly.
    #[test]
    fn test_print_graph() {
        let tensor_a: Tensor<f32> = Tensor::new([1.0, 2.0], [2]);
        let tensor_b: Tensor<f32> = Tensor::new([3.0, 4.0], [2]);
        let tensor_c = tensor_a.clone() * tensor_b.clone();
        let expected_output = format!(
            "Node {} MulOperation├── Leaf [1.0, 2.0]├── Leaf [3.0, 4.0]",
            tensor_c.state.borrow().id
        );
        assert_eq!(tensor_c.print_graph(), expected_output);
    }

    /// Test that a tensor graph can be sorted topologically correctly.
    #[test_case(
        vec![(Tensor::new([1.0], [1]), vec![])],
        vec![0];
        "single tensor graph"
    )]
    #[test_case(
        vec![
            (Tensor::new([1.0], [1]), vec![]),
            (Tensor::new([2.0], [1]), vec![]),
            (Tensor::new([3.0], [1]), vec![0, 1]),
        ],
        vec![0, 1, 2];
        "three node DAG"
    )]
    #[test_case(
        vec![
            (Tensor::new([1.0], [1]), vec![]),
            (Tensor::new([2.0], [1]), vec![]),
            (Tensor::new([3.0], [1]), vec![0, 1]),
            (Tensor::new([4.0], [1]), vec![2, 1]),
        ],
        vec![0, 1, 2, 3];
        "four node complex DAG"
    )]
    #[test_case(
        vec![
            (Tensor::new([1.0], [1]), vec![]),
            (Tensor::new([2.0], [1]), vec![]),
            (Tensor::new([3.0], [1]), vec![0, 1]),
            (Tensor::new([4.0], [1]), vec![2]),
            (Tensor::new([5.0], [1]), vec![2, 3]),
        ],
        vec![0, 1, 2, 3, 4];
        "complex multi-branch DAG"
    )]
    #[test_case(
        vec![
            (Tensor::new([1.0], [1]), vec![]),
            (Tensor::new([2.0], [1]), vec![0]),
            (Tensor::new([3.0], [1]), vec![0]),
            (Tensor::new([4.0], [1]), vec![1, 2]),
            (Tensor::new([5.0], [1]), vec![3, 0]),
        ],
        vec![0, 1, 2, 3, 4];
        "diamond dependency DAG with shared leaf"
    )]
    fn test_topological_sort(
        nodes: Vec<(Tensor<f32>, Vec<usize>)>,
        expected_sorted_indices: Vec<usize>,
    ) {
        let tensors = build_test_graph(nodes);
        let sorted_tensors = topological_sort(tensors.last().unwrap().clone());
        assert_eq!(
            sorted_tensors,
            expected_sorted_indices
                .iter()
                .map(|&i| tensors[i].clone())
                .collect::<Vec<_>>()
        );
    }
}
