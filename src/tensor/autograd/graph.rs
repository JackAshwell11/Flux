use crate::tensor::core::Tensor;
use std::collections::HashSet;
use std::fmt::Debug;

/// Recursively walk a tensor computation graph and apply both a pre-order and post-order function
/// on each node in the graph.
fn walk_graph<T, Pre, Post>(
    tensor: &Tensor<T>,
    visited: &mut HashSet<usize>,
    depth: usize,
    pre_order: &mut Pre,
    post_order: &mut Post,
) where
    Pre: FnMut(&Tensor<T>, usize),
    Post: FnMut(&Tensor<T>),
{
    // Check if this tensor has already been visited or not
    if !visited.insert(tensor.id()) {
        return;
    }

    // Get the tensor and apply the pre-order graph operation
    pre_order(tensor, depth);

    // Visit the node's parents if it exists
    let parents = {
        tensor
            .node()
            .as_ref()
            .map(|node| node.parents.clone())
            .unwrap_or_default()
    };
    for parent in parents {
        walk_graph(&parent, visited, depth + 1, pre_order, post_order);
    }

    // Apply the post-order graph operation
    post_order(tensor);
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
            &self.clone(),
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
                let new_str = tensor.node().as_ref().map_or_else(
                    || format!("{prefix}Leaf {:?}", tensor.data()),
                    |node| format!("{prefix}Node {} {:?}", tensor.id(), node.operation),
                );
                output.push_str(&new_str);
            },
            &mut |_| {},
        );
        output
    }
}

/// Get a topologically sorted list of the tensor's dependencies.
pub(crate) fn topological_sort<T>(tensor: &Tensor<T>) -> Vec<Tensor<T>> {
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
    use crate::tensor::core::Tensor;
    use std::collections::HashSet;
    use test_case::test_case;

    /// The type of a function that builds a computation graph.
    type GraphBuilder = fn() -> Vec<Tensor<f32>>;

    /// A helper to construct a computation graph with a single tensor.
    fn single_tensor_graph() -> Vec<Tensor<f32>> {
        vec![Tensor::new([1.0], [1], true)]
    }

    /// A helper to construct a computation graph with three tensors.
    fn three_node_graph() -> Vec<Tensor<f32>> {
        let tensor_a = Tensor::new([1.0, 2.0], [2], true);
        let tensor_b = Tensor::new([3.0, 4.0], [2], true);
        let tensor_c = tensor_a.clone() * tensor_b.clone();
        vec![tensor_a, tensor_b, tensor_c]
    }

    /// A helper to construct a computation graph with four tensors.
    fn four_node_complex_graph() -> Vec<Tensor<f32>> {
        let tensor_a = Tensor::new([1.0], [1], true);
        let tensor_b = Tensor::new([2.0], [1], true);
        let tensor_c = tensor_a.clone() + tensor_b.clone();
        let tensor_d = tensor_c.clone() + tensor_b.clone();
        vec![tensor_a, tensor_b, tensor_c, tensor_d]
    }

    /// A helper to construct a computation graph with multiple branches.
    fn multi_branch_graph() -> Vec<Tensor<f32>> {
        let tensor_a = Tensor::new([1.0], [1], true);
        let tensor_b = Tensor::new([2.0], [1], true);
        let tensor_c = tensor_a.clone() + tensor_b.clone();
        let tensor_d = Tensor::new([4.0], [1], true);
        let tensor_e = tensor_c.clone() + tensor_d.clone();
        let tensor_f = tensor_c.clone() + tensor_e.clone();
        vec![tensor_a, tensor_b, tensor_c, tensor_d, tensor_e, tensor_f]
    }

    /// A helper to construct a computation graph with a diamond dependency.
    fn diamond_dependency_graph() -> Vec<Tensor<f32>> {
        let tensor_a = Tensor::new([1.0], [1], true);
        let tensor_b = Tensor::new([2.0], [1], true);
        let tensor_c = tensor_a.clone() + tensor_b.clone();
        let tensor_d = tensor_a.clone() + tensor_c.clone();
        let tensor_e = tensor_c.clone() + tensor_d.clone();
        vec![tensor_a, tensor_b, tensor_c, tensor_d, tensor_e]
    }

    /// Test that walking the graph is performed correctly.
    #[test_case(super::single_tensor_graph, [0], [0]; "single tensor graph")]
    #[test_case(super::three_node_graph, [2, 0, 1], [0, 1, 2]; "three node DAG")]
    #[test_case(super::four_node_complex_graph, [3, 2, 0, 1], [0, 1, 2, 3]; "four node complex DAG")]
    fn test_walk_graph<const E: usize>(
        build_graph: GraphBuilder,
        expected_pre_indices: [usize; E],
        expected_post_indices: [usize; E],
    ) {
        let tensors = build_graph();
        let mut pre_order = vec![];
        let mut post_order = vec![];
        walk_graph(
            &tensors.last().unwrap().clone(),
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
        let tensor_a: Tensor<f32> = Tensor::new([1.0, 2.0], [2], true);
        let tensor_b: Tensor<f32> = Tensor::new([3.0, 4.0], [2], true);
        let tensor_c = tensor_a * tensor_b;
        let expected_output = format!(
            "Node {} MulOperation├── Leaf [1.0, 2.0]├── Leaf [3.0, 4.0]",
            tensor_c.id()
        );
        assert_eq!(tensor_c.print_graph(), expected_output);
    }

    /// Test that a tensor graph can be sorted topologically correctly.
    #[test_case(super::single_tensor_graph, [0]; "single tensor graph")]
    #[test_case(super::three_node_graph, [0, 1, 2]; "three node DAG")]
    #[test_case(super::four_node_complex_graph, [0, 1, 2, 3]; "four node complex DAG")]
    #[test_case(super::multi_branch_graph, [0, 1, 2, 3, 4, 5]; "complex multi-branch DAG")]
    #[test_case(super::diamond_dependency_graph, [0, 1, 2, 3, 4]; "diamond dependency DAG")]
    fn test_topological_sort<const E: usize>(
        build_graph: GraphBuilder,
        expected_sorted_indices: [usize; E],
    ) {
        let tensors = build_graph();
        let sorted_tensors = topological_sort(&tensors.last().unwrap().clone());
        assert_eq!(
            sorted_tensors,
            expected_sorted_indices
                .iter()
                .map(|&i| tensors[i].clone())
                .collect::<Vec<_>>()
        );
    }
}
