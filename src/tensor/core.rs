use std::cell::{Ref, RefCell, RefMut};
use std::fmt::Debug;
use std::rc::Rc;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Represents a tensor in Flux.
#[derive(Debug)]
pub struct Tensor<T> {
    // The pointer to the tensor state
    state: Rc<RefCell<TensorState<T>>>,
}

/// Stores the mutable data and autograd information for a tensor.
#[derive(Debug)]
struct TensorState<T> {
    // The ID of this tensor in Flux
    id: usize,

    // The core data for the tensor
    data: Vec<T>,
    shape: Vec<usize>,

    // The gradient of this tensor
    grad: Vec<T>,
    requires_grad: bool,

    // The autograd link for this tensor
    node: Option<OperationNode<T>>,
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

impl<T> PartialEq for Tensor<T>
where
    T: PartialEq + Clone,
{
    /// Allows tensors to be compared by comparing their shared state pointers.
    fn eq(&self, other: &Self) -> bool {
        (self.shape() == other.shape()) && (self.data() == other.data())
    }
}

/// Generate a unique ID for a tensor.
fn next_tensor_id() -> usize {
    static TENSOR_COUNTER: AtomicUsize = AtomicUsize::new(0);
    TENSOR_COUNTER.fetch_add(1, Ordering::Relaxed)
}

impl<T> Tensor<T> {
    /// Get the ID of the tensor.
    #[must_use]
    pub(crate) fn id(&self) -> usize {
        self.state.borrow().id
    }

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

    /// Get the tensor data.
    #[must_use]
    pub fn data(&self) -> Vec<T>
    where
        T: Clone,
    {
        self.state.borrow().data.clone()
    }

    /// Get a mutable reference to the tensor data.
    #[must_use]
    pub fn data_mut(&self) -> RefMut<'_, Vec<T>> {
        RefMut::map(self.state.borrow_mut(), |state| &mut state.data)
    }

    /// Get the tensor shape.
    #[must_use]
    pub fn shape(&self) -> Vec<usize> {
        self.state.borrow().shape.clone()
    }

    /// Get the tensor gradient.
    #[must_use]
    pub fn grad(&self) -> Vec<T>
    where
        T: Clone,
    {
        self.state.borrow().grad.clone()
    }

    /// Get a mutable reference to the tensor gradient.
    #[must_use]
    pub fn grad_mut(&self) -> RefMut<'_, Vec<T>> {
        RefMut::map(self.state.borrow_mut(), |state| &mut state.grad)
    }

    /// Get whether this tensor's gradient computation is required.
    #[must_use]
    pub fn requires_grad(&self) -> bool {
        self.state.borrow().requires_grad
    }

    /// Get the autograd node associated with this tensor.
    pub(crate) fn node(&self) -> Ref<'_, Option<OperationNode<T>>> {
        Ref::map(self.state.borrow(), |state| &state.node)
    }
}

impl<T> Tensor<T>
where
    T: Default + Clone,
{
    /// Create a tensor from its individual components.
    pub(crate) fn from_parts(
        data: impl Into<Vec<T>>,
        shape: impl Into<Vec<usize>>,
        grad: impl Into<Vec<T>>,
        requires_grad: bool,
        node: Option<OperationNode<T>>,
    ) -> Self {
        let data = data.into();
        let shape = shape.into();
        let grad = grad.into();
        Self {
            state: Rc::new(RefCell::new(TensorState {
                id: next_tensor_id(),
                data,
                shape,
                grad,
                requires_grad,
                node,
            })),
        }
    }

    /// Create a tensor from raw data.
    ///
    /// # Panics
    ///
    /// Panics if the number of elements in `data` does not match the product of `shape`.
    pub fn new(data: impl Into<Vec<T>>, shape: impl Into<Vec<usize>>, requires_grad: bool) -> Self {
        let data = data.into();
        let shape = shape.into();
        let num_elements = shape.iter().product::<usize>();
        assert_eq!(data.len(), num_elements);
        Self::from_parts(
            data,
            shape,
            vec![T::default(); num_elements],
            requires_grad,
            None,
        )
    }

    /// Create a tensor with a given shape and all zeros.
    pub fn zeros(shape: impl Into<Vec<usize>>, requires_grad: bool) -> Self {
        let shape = shape.into();
        let num_elements = shape.iter().product::<usize>();
        Self::from_parts(
            vec![T::default(); num_elements],
            shape,
            vec![T::default(); num_elements],
            requires_grad,
            None,
        )
    }
}

impl<T> Tensor<T>
where
    T: From<f32> + Clone + Default,
{
    /// Create a tensor with a given shape and all ones.
    pub fn ones(shape: impl Into<Vec<usize>>, requires_grad: bool) -> Self {
        let shape = shape.into();
        let num_elements = shape.iter().product::<usize>();
        Self::from_parts(
            vec![T::from(1.0); num_elements],
            shape,
            vec![T::default(); num_elements],
            requires_grad,
            None,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use test_case::test_case;

    /// Test that cloning a tensor preserves its state pointer.
    #[test]
    fn test_clone() {
        let tensor: Tensor<f32> = Tensor::zeros([2, 2], false);
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

    /// Test that id returns a unique ID for each tensor.
    #[test]
    fn test_id() {
        let tensor_one: Tensor<f32> = Tensor::zeros([1], false);
        let tensor_two: Tensor<f32> = Tensor::zeros([1], false);
        assert_ne!(tensor_one.id(), tensor_two.id());
    }

    /// Test that rank returns the number of dimensions in the tensor shape.
    #[test_case([], 0; "scalar rank")]
    #[test_case([1], 1; "one-dimensional rank")]
    #[test_case([2, 3], 2; "two-dimensional rank")]
    #[test_case([2, 3, 4], 3; "three-dimensional rank")]
    fn test_rank<const N: usize>(shape: [usize; N], expected_rank: usize) {
        let tensor: Tensor<f32> = Tensor::zeros(shape, false);
        assert_eq!(tensor.rank(), expected_rank);
    }

    /// Test that size returns the total number of elements in the tensor data.
    #[test_case([], 1; "scalar size")]
    #[test_case([1], 1; "single element size")]
    #[test_case([4], 4; "vector size")]
    #[test_case([2, 3], 6; "matrix size")]
    #[test_case([2, 3, 4], 24; "three-dimensional size")]
    fn test_size<const N: usize>(shape: [usize; N], expected_size: usize) {
        let tensor: Tensor<i32> = Tensor::zeros(shape, false);
        assert_eq!(tensor.size(), expected_size);
    }

    /// Test that data returns the tensor's underlying data.
    #[test_case([1.0], [1.0]; "single element")]
    #[test_case([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]; "vector")]
    #[test_case([1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0]; "matrix")]
    fn test_data<const N: usize, const E: usize>(data: [f32; N], expected: [f32; E]) {
        let tensor = Tensor::new(data, [N], false);
        assert_eq!(tensor.data(), expected);
    }

    /// Test that `data_mut` allows the tensor's data to be modified.
    #[test]
    fn test_data_mut() {
        let tensor: Tensor<f32> = Tensor::zeros([2], false);
        tensor.data_mut()[0] = 1.0;
        tensor.data_mut()[1] = 2.0;
        assert_eq!(tensor.data(), [1.0, 2.0]);
    }

    /// Test that shape returns the tensor's shape.
    #[test_case([1], [1]; "one-dimensional")]
    #[test_case([2, 3], [2, 3]; "two-dimensional")]
    #[test_case([2, 3, 4], [2, 3, 4]; "three-dimensional")]
    fn test_shape<const N: usize, const E: usize>(shape: [usize; N], expected: [usize; E]) {
        let tensor: Tensor<f32> = Tensor::zeros(shape, false);
        assert_eq!(tensor.shape(), expected);
    }

    /// Test that grad returns the tensor's gradient.
    #[test]
    fn test_grad() {
        let tensor: Tensor<f32> = Tensor::zeros([2], false);
        assert_eq!(tensor.grad(), [0.0, 0.0]);
    }

    /// Test that `grad_mut` allows the tensor's gradient to be modified.
    #[test]
    fn test_grad_mut() {
        let tensor: Tensor<f32> = Tensor::zeros([2], false);
        tensor.grad_mut()[0] = 1.0;
        tensor.grad_mut()[1] = 2.0;
        assert_eq!(tensor.grad(), [1.0, 2.0]);
    }

    /// Test that `requires_grad` returns whether gradient computation is required.
    #[test_case(false; "does not require gradients")]
    #[test_case(true; "requires gradients")]
    fn test_requires_grad(requires_grad: bool) {
        let tensor: Tensor<f32> = Tensor::zeros([2], requires_grad);
        assert_eq!(tensor.requires_grad(), requires_grad);
    }

    /// Test that node returns the tensor's autograd node.
    #[test]
    fn test_node() {
        let tensor: Tensor<f32> = Tensor::zeros([2], false);
        assert!(tensor.node().is_none());
    }

    /// Test that tensors can be correctly created from individual parts.
    #[test]
    fn test_from_parts() {
        let tensor: Tensor<f32> = Tensor::from_parts(
            vec![1.0, 2.0, 3.0],
            vec![3],
            vec![0.1, 0.2, 0.3],
            true,
            None,
        );
        assert_eq!(tensor.data(), vec![1.0, 2.0, 3.0]);
        assert_eq!(tensor.shape(), vec![3]);
        assert_eq!(tensor.grad(), vec![0.1, 0.2, 0.3]);
        assert!(tensor.requires_grad());
        assert!(tensor.node().is_none());
    }

    /// Test that the tensor constructor produces a tensor with the correct ID, data, shape,
    /// gradient, and node.
    #[test_case([], [0], [], [0]; "empty tensor")]
    #[test_case([1.0], [1], [1.0], [1]; "single element")]
    #[test_case([1.0, 2.0, 3.0, 4.0], [4], [1.0, 2.0, 3.0, 4.0], [4]; "one-dimensional tensor")]
    #[test_case([1.0, 2.0, 3.0, 4.0], [2, 2], [1.0, 2.0, 3.0, 4.0], [2, 2]; "two-dimensional tensor")]
    #[test_case([10.0, 20.0], [2], [10.0, 20.0], [2]; "two elements")]
    fn test_new<const N: usize, const S: usize, const ED: usize, const ES: usize>(
        data: [f32; N],
        shape: [usize; S],
        expected_data: [f32; ED],
        expected_shape: [usize; ES],
    ) {
        let tensor = Tensor::new(data, shape, false);
        assert_eq!(tensor.data(), expected_data);
        assert_eq!(tensor.shape(), expected_shape);
        assert_eq!(tensor.grad().len(), tensor.data().len());
        assert!(tensor.node().is_none());
    }

    /// Test that the tensors created with zeros have the correct ID, data, gradient, and node.
    #[test_case([], [0.0], []; "scalar zero")]
    #[test_case([2], [0.0; 2], [2]; "vector zeros")]
    #[test_case([2, 3], [0.0; 6], [2, 3]; "matrix zeros")]
    #[test_case([1, 1], [0.0; 1], [1, 1]; "single matrix zero")]
    fn test_zeros<const N: usize, const ED: usize, const ES: usize>(
        shape: [usize; N],
        expected_data: [f32; ED],
        expected_shape: [usize; ES],
    ) {
        let tensor: Tensor<f32> = Tensor::zeros(shape, false);
        assert_eq!(tensor.data(), expected_data);
        assert_eq!(tensor.shape(), expected_shape);
        assert_eq!(tensor.grad().len(), tensor.data().len());
        assert!(tensor.node().is_none());
    }

    /// Test that tensors created with ones have the correct ID, data, gradient, and node.
    #[test_case([], [1.0], []; "scalar one")]
    #[test_case([1], [1.0], [1]; "single one")]
    #[test_case([2, 2], [1.0; 4], [2, 2]; "matrix ones")]
    #[test_case([1, 3], [1.0; 3], [1, 3]; "row vector ones")]
    fn test_ones<const N: usize, const ED: usize, const ES: usize>(
        shape: [usize; N],
        expected_data: [f32; ED],
        expected_shape: [usize; ES],
    ) {
        let tensor: Tensor<f32> = Tensor::ones(shape, false);
        assert_eq!(tensor.data(), expected_data);
        assert_eq!(tensor.shape(), expected_shape);
        assert_eq!(tensor.grad().len(), tensor.data().len());
        assert!(tensor.node().is_none());
    }
}
