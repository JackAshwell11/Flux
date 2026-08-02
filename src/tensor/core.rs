use std::cell::RefCell;
use std::fmt::Debug;
use std::rc::Rc;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Represents a tensor in Flux.
#[derive(Debug)]
pub struct Tensor<T> {
    // The pointer to the tensor state
    pub state: Rc<RefCell<TensorState<T>>>,
}

/// Stores the mutable data and autograd information for a tensor.
#[derive(Debug)]
pub struct TensorState<T> {
    // The ID of this tensor in Flux
    pub(crate) id: usize,

    // The core data for the tensor
    pub data: Vec<T>,
    pub(crate) shape: Vec<usize>,

    // The gradient of this tensor
    pub grad: Vec<T>,

    // The autograd link for this tensor
    pub(crate) node: Option<OperationNode<T>>,
}

/// Represents a node in the computation graph.
#[derive(Debug)]
pub(crate) struct OperationNode<T> {
    // The parents of this tensor in the computation graph
    pub(crate) parents: Vec<Tensor<T>>,

    // The operation that produced this tensor
    pub(crate) operation: Box<dyn Operation<T>>,
}

/// Represents an operation on tensors.
pub(crate) trait Operation<T>: Debug {
    /// Propagates the incoming gradient through the operation to its inputs.
    fn backward(&self, grad_output: &Tensor<T>, parents: &[Tensor<T>]) -> Vec<Tensor<T>>;
}

impl<T> Clone for Tensor<T> {
    /// Allows tensors to be cheaply cloned by cloning their shared state pointer.
    fn clone(&self) -> Self {
        Self {
            state: Rc::clone(&self.state),
        }
    }
}

impl<T> PartialEq for Tensor<T> {
    /// Allows tensors to be compared by comparing their shared state pointers.
    fn eq(&self, other: &Self) -> bool {
        Rc::ptr_eq(&self.state, &other.state)
    }
}

/// Generate a unique ID for a tensor.
pub(crate) fn next_tensor_id() -> usize {
    static TENSOR_COUNTER: AtomicUsize = AtomicUsize::new(0);
    TENSOR_COUNTER.fetch_add(1, Ordering::Relaxed)
}

impl<T> Tensor<T> {
    /// Get the rank of the tensor.
    #[must_use]
    pub fn rank(&self) -> usize {
        self.state.borrow().shape.len()
    }

    /// Get the size of the tensor.
    #[must_use]
    pub fn size(&self) -> usize {
        self.state.borrow().data.len()
    }

    /// Get the data at a given index.
    #[must_use]
    pub fn get(&self, index: usize) -> T
    where
        T: Copy,
    {
        self.state.borrow().data[index]
    }
}

impl<T> Tensor<T>
where
    T: Default + Clone,
{
    /// Create a tensor from a TensorState struct.
    pub(crate) fn from_state(state: TensorState<T>) -> Self {
        Self {
            state: Rc::new(RefCell::new(state)),
        }
    }

    /// Create a tensor from raw data.
    ///
    /// # Panics
    ///
    /// Panics if the number of elements in `data` does not match the product of `shape`.
    pub fn new(data: impl Into<Vec<T>>, shape: impl Into<Vec<usize>>) -> Self {
        let data = data.into();
        let shape = shape.into();
        let num_elements = shape.iter().product::<usize>();
        assert_eq!(data.len(), num_elements);
        Self::from_state(TensorState {
            id: next_tensor_id(),
            data,
            shape,
            grad: vec![T::default(); num_elements],
            node: None,
        })
    }

    /// Create a tensor with a given shape and all zeros.
    pub fn zeros(shape: impl Into<Vec<usize>>) -> Self {
        let shape = shape.into();
        let num_elements = shape.iter().product::<usize>();
        Self::from_state(TensorState {
            id: next_tensor_id(),
            data: vec![T::default(); num_elements],
            shape,
            grad: vec![T::default(); num_elements],
            node: None,
        })
    }
}

impl<T> Tensor<T>
where
    T: From<f32> + Clone + Default,
{
    /// Create a tensor with a given shape and all ones.
    pub fn ones(shape: impl Into<Vec<usize>>) -> Self {
        let shape = shape.into();
        let num_elements = shape.iter().product::<usize>();
        Self::from_state(TensorState {
            id: next_tensor_id(),
            data: vec![T::from(1.0); num_elements],
            shape,
            grad: vec![T::default(); num_elements],
            node: None,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use test_case::test_case;

    /// Test that cloning a tensor preserves its state pointer.
    #[test]
    fn test_clone() {
        let tensor: Tensor<f32> = Tensor::zeros([2, 2]);
        let cloned_tensor = tensor.clone();
        assert!(Rc::ptr_eq(&tensor.state, &cloned_tensor.state));
    }

    /// Test that unique tensor IDs are generated sequentially.
    #[test]
    fn test_next_tensor_id() {
        let id_one = next_tensor_id();
        let id_two = next_tensor_id();
        let id_three = next_tensor_id();
        assert_eq!(id_two, id_one + 1);
        assert_eq!(id_three, id_two + 1);
    }

    /// Test that rank returns the number of dimensions in the tensor shape.
    #[test_case([], 0; "scalar rank")]
    #[test_case([1], 1; "one-dimensional rank")]
    #[test_case([2, 3], 2; "two-dimensional rank")]
    #[test_case([2, 3, 4], 3; "three-dimensional rank")]
    fn test_rank<const N: usize>(shape: [usize; N], expected_rank: usize) {
        let tensor: Tensor<f32> = Tensor::zeros(shape);
        assert_eq!(tensor.rank(), expected_rank);
    }

    /// Test that size returns the total number of elements in the tensor data.
    #[test_case([], 1; "scalar size")]
    #[test_case([1], 1; "single element size")]
    #[test_case([4], 4; "vector size")]
    #[test_case([2, 3], 6; "matrix size")]
    #[test_case([2, 3, 4], 24; "three-dimensional size")]
    fn test_size<const N: usize>(shape: [usize; N], expected_size: usize) {
        let tensor: Tensor<i32> = Tensor::zeros(shape);
        assert_eq!(tensor.size(), expected_size);
    }

    /// Test that the get method retrieves data correctly at a given index.
    #[test_case([1.0, 2.0, 3.0, 4.0], [4], 0, 1.0; "first element")]
    #[test_case([1.0, 2.0, 3.0, 4.0], [4], 3, 4.0; "last element")]
    #[test_case([10.0, 20.0, 30.0, 40.0], [4], 2, 30.0; "middle element")]
    #[test_case([-1.0, -2.0, -3.0, -4.0], [4], 1, -2.0; "negative values")]
    fn test_get<const N: usize, const S: usize>(
        data: [f32; N],
        shape: [usize; S],
        index: usize,
        expected_value: f32,
    ) {
        let tensor = Tensor::new(data, shape);
        assert_eq!(tensor.get(index), expected_value);
    }

    /// Test that tensors can be correctly created from a TensorState.
    #[test]
    fn test_from_state() {
        let state = TensorState {
            id: 42,
            data: vec![1.0, 2.0, 3.0],
            shape: vec![3],
            grad: vec![0.0, 0.0, 0.0],
            node: None,
        };
        let tensor: Tensor<f32> = Tensor::from_state(state);
        let tensor_state = tensor.state.borrow();
        assert_eq!(tensor_state.id, 42);
        assert_eq!(tensor_state.data, vec![1.0, 2.0, 3.0]);
        assert_eq!(tensor_state.shape, vec![3]);
        assert_eq!(tensor_state.grad, vec![0.0, 0.0, 0.0]);
        assert!(tensor_state.node.is_none());
    }

    /// Test that the tensor constructor produces a tensor with the correct ID, data, shape,
    /// gradient, and node.
    #[test_case([], [0], vec![], vec![0]; "empty tensor")]
    #[test_case([1.0], [1], vec![1.0], vec![1]; "single element")]
    #[test_case([1.0, 2.0, 3.0, 4.0], [4], vec![1.0, 2.0, 3.0, 4.0], vec![4]; "one-dimensional tensor")]
    #[test_case([1.0, 2.0, 3.0, 4.0], [2, 2], vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]; "two-dimensional tensor")]
    #[test_case([10.0, 20.0], [2], vec![10.0, 20.0], vec![2]; "two elements")]
    fn test_new<const N: usize, const S: usize>(
        data: [f32; N],
        shape: [usize; S],
        expected_data: Vec<f32>,
        expected_shape: Vec<usize>,
    ) {
        let tensor = Tensor::new(data, shape);
        let tensor_state = tensor.state.borrow();
        assert!(tensor_state.id > 0);
        assert_eq!(tensor_state.data, expected_data);
        assert_eq!(tensor_state.shape, expected_shape);
        assert_eq!(tensor_state.grad.len(), tensor_state.data.len());
        assert!(tensor_state.node.is_none());
    }

    /// Test that the tensors created with zeros have the correct ID, data, gradient, and node.
    #[test_case([], vec![0.0], vec![]; "scalar zero")]
    #[test_case([2], vec![0.0; 2], vec![2]; "vector zeros")]
    #[test_case([2, 3], vec![0.0; 6], vec![2, 3]; "matrix zeros")]
    #[test_case([1, 1], vec![0.0; 1], vec![1, 1]; "single matrix zero")]
    fn test_zeros<const N: usize>(
        shape: [usize; N],
        expected_data: Vec<f32>,
        expected_shape: Vec<usize>,
    ) {
        let tensor: Tensor<f32> = Tensor::zeros(shape);
        let tensor_state = tensor.state.borrow();
        assert!(tensor_state.id > 0);
        assert_eq!(tensor_state.data, expected_data);
        assert_eq!(tensor_state.shape, expected_shape);
        assert_eq!(tensor_state.grad.len(), tensor_state.data.len());
        assert!(tensor_state.node.is_none());
    }

    /// Test that tensors created with ones have the correct ID, data, gradient, and node.
    #[test_case([], vec![1.0], vec![]; "scalar one")]
    #[test_case([1], vec![1.0], vec![1]; "single one")]
    #[test_case([2, 2], vec![1.0; 4], vec![2, 2]; "matrix ones")]
    #[test_case([1, 3], vec![1.0; 3], vec![1, 3]; "row vector ones")]
    fn test_ones<const N: usize>(
        shape: [usize; N],
        expected_data: Vec<f32>,
        expected_shape: Vec<usize>,
    ) {
        let tensor: Tensor<f32> = Tensor::ones(shape);
        let tensor_state = tensor.state.borrow();
        assert!(tensor_state.id > 0);
        assert_eq!(tensor_state.data, expected_data);
        assert_eq!(tensor_state.shape, expected_shape);
        assert_eq!(tensor_state.grad.len(), tensor_state.data.len());
        assert!(tensor_state.node.is_none());
    }
}
