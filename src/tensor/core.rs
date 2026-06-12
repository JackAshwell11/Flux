/// Represents a tensor with a given shape and data type.
#[derive(Debug, Clone)]
pub struct Tensor<T> {
    pub(crate) data: Vec<T>,
    pub(crate) shape: Vec<usize>,
}

impl<T> Tensor<T> {
    /// Get the rank of the tensor.
    pub fn rank(&self) -> usize {
        self.shape.len()
    }

    /// Get the size of the tensor.
    pub fn size(&self) -> usize {
        self.data.len()
    }

    /// Create a tensor from raw data.
    pub fn new(data: impl Into<Vec<T>>, shape: impl Into<Vec<usize>>) -> Self {
        let data = data.into();
        let shape = shape.into();
        assert_eq!(data.len(), shape.iter().product::<usize>());
        Self { data, shape }
    }
}

impl<T> Tensor<T>
where
    T: Default + Clone,
{
    /// Create a tensor with a given shape and all zeros.
    pub fn zeros(shape: impl Into<Vec<usize>>) -> Self {
        let shape = shape.into();
        Self {
            data: vec![T::default(); shape.iter().product()],
            shape,
        }
    }
}

impl<T> Tensor<T>
where
    T: From<u8> + Clone,
{
    /// Create a tensor with a given shape and all ones.
    pub fn ones(shape: impl Into<Vec<usize>>) -> Self {
        let shape = shape.into();
        Self {
            data: vec![T::from(1); shape.iter().product()],
            shape,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use test_case::test_case;

    /// Test that rank returns the number of dimensions in the tensor shape.
    #[test_case([], 0; "scalar rank")]
    #[test_case([1], 1; "one-dimensional rank")]
    #[test_case([2, 3], 2; "two-dimensional rank")]
    #[test_case([2, 3, 4], 3; "three-dimensional rank")]
    fn test_rank<const N: usize>(shape: [usize; N], expected_rank: usize) {
        let tensor: Tensor<i32> = Tensor::zeros(shape);
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

    /// Test that the tensor constructor produces a tensor with the correct data and shape.
    #[test_case([], [0], vec![], vec![0]; "empty tensor")]
    #[test_case([1], [1], vec![1], vec![1]; "single element")]
    #[test_case([1, 2, 3, 4], [4], vec![1, 2, 3, 4], vec![4]; "one-dimensional tensor")]
    #[test_case([1, 2, 3, 4], [2, 2], vec![1, 2, 3, 4], vec![2, 2]; "two-dimensional tensor")]
    #[test_case([10, 20], [2], vec![10, 20], vec![2]; "two elements")]
    fn test_new<const N: usize, const S: usize>(
        data: [usize; N],
        shape: [usize; S],
        expected_data: Vec<usize>,
        expected_shape: Vec<usize>,
    ) {
        let tensor = Tensor::new(data, shape);
        assert_eq!(tensor.data, expected_data);
        assert_eq!(tensor.shape, expected_shape);
    }

    /// Test that the tensor constructor can be used with a custom shape to give data with all zero
    /// values.
    #[test_case([], vec![0], vec![]; "scalar zero")]
    #[test_case([2], vec![0; 2], vec![2]; "vector zeros")]
    #[test_case([2, 3], vec![0; 6], vec![2, 3]; "matrix zeros")]
    #[test_case([1, 1], vec![0; 1], vec![1, 1]; "single matrix zero")]
    fn test_zeros<const N: usize>(
        shape: [usize; N],
        expected_data: Vec<i32>,
        expected_shape: Vec<usize>,
    ) {
        let tensor: Tensor<i32> = Tensor::zeros(shape);
        assert_eq!(tensor.data, expected_data);
        assert_eq!(tensor.shape, expected_shape);
    }

    /// Test that the tensor constructor can be used with a custom shape to give data with all one values.
    #[test_case([], vec![1], vec![]; "scalar one")]
    #[test_case([1], vec![1], vec![1]; "single one")]
    #[test_case([2, 2], vec![1; 4], vec![2, 2]; "matrix ones")]
    #[test_case([1, 3], vec![1; 3], vec![1, 3]; "row vector ones")]
    fn test_ones<const N: usize>(
        shape: [usize; N],
        expected_data: Vec<i32>,
        expected_shape: Vec<usize>,
    ) {
        let tensor: Tensor<i32> = Tensor::ones(shape);
        assert_eq!(tensor.data, expected_data);
        assert_eq!(tensor.shape, expected_shape);
    }
}
