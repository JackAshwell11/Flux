use crate::tensor::core::Tensor;

impl<T> Tensor<T>
where
    T: Copy + Default,
{
    /// Perform a map operation on the tensor.
    #[must_use]
    pub fn map<F>(&self, f: F) -> Self
    where
        F: Fn(T) -> T,
    {
        let (data, shape) = {
            (
                self.data().iter().copied().map(f).collect::<Vec<T>>(),
                self.shape(),
            )
        };
        let num_elements = data.len();
        Self::from_parts(
            data,
            shape,
            vec![T::default(); num_elements],
            self.requires_grad(),
            None,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use test_case::test_case;

    /// A function that doubles the input value.
    fn double(x: i32) -> i32 {
        x * 2
    }

    /// A function that increments the input value by 1.
    fn increment(x: i32) -> i32 {
        x + 1
    }

    /// A function that squares the input value.
    fn square(x: i32) -> i32 {
        x * x
    }

    /// A function that maps a character to a different character.
    fn remap_char(x: char) -> char {
        match x {
            'a' => 'x',
            'b' => 'y',
            'c' => 'z',
            value => value,
        }
    }

    /// Test that mapping integer tensors applies the function to every element and preserves shape.
    #[test_case([1, 2, 3, 4], [4], double, [2, 4, 6, 8], [4]; "double one-dimensional tensor")]
    #[test_case([1, 2, 3, 4], [2, 2], double, [2, 4, 6, 8], [2, 2]; "double two-dimensional tensor")]
    #[test_case([-2, -1, 0, 1, 2], [5], double, [-4, -2, 0, 2, 4], [5]; "double mixed signs")]
    #[test_case([0], [1], double, [0], [1]; "double single zero")]
    #[test_case([1, 2, 3], [3], increment, [2, 3, 4], [3]; "increment vector")]
    #[test_case([1, 2, 3, 4, 5, 6], [2, 3], increment, [2, 3, 4, 5, 6, 7], [2, 3]; "increment matrix")]
    #[test_case([-3, -2, -1], [3], increment, [-2, -1, 0], [3]; "increment negative values")]
    #[test_case([1, 2, 3, 4], [4], square, [1, 4, 9, 16], [4]; "square vector")]
    #[test_case([-3, -2, -1, 0, 1, 2, 3], [7], square, [9, 4, 1, 0, 1, 4, 9], [7]; "square mixed signs")]
    #[test_case([2, 3, 4, 5], [2, 2], square, [4, 9, 16, 25], [2, 2]; "square matrix")]
    fn test_map_i32<const N: usize, const S: usize, const ED: usize, const ES: usize>(
        data: [i32; N],
        shape: [usize; S],
        f: fn(i32) -> i32,
        expected_data: [i32; ED],
        expected_shape: [usize; ES],
    ) {
        let tensor = Tensor::new(data, shape, true);
        let mapped = tensor.map(f);
        assert_eq!(mapped.data(), expected_data);
        assert_eq!(mapped.shape(), expected_shape);
    }

    /// Test that mapping a tensor with a passthrough function preserves the same data.
    #[test]
    fn test_map() {
        let tensor = Tensor::new([-2, -1, 0, 1, 2], [5], true);
        let mapped = tensor.map(|x| x);
        assert_eq!(mapped.data(), tensor.data());
        assert_eq!(mapped.shape(), tensor.shape());
    }

    /// Test that mapping an empty tensor returns an empty tensor with the same shape.
    #[test]
    fn test_map_preserves_empty_tensor() {
        let tensor: Tensor<i32> = Tensor::new([], [0], true);
        let mapped = tensor.map(|x| x + 10);
        assert_eq!(mapped.data(), Vec::new());
        assert_eq!(mapped.shape(), vec![0]);
    }

    /// Test that mapping a scalar tensor preserves scalar shape.
    #[test]
    fn test_map_preserves_scalar_shape() {
        let tensor = Tensor::new([7], [], true);
        let mapped = tensor.map(|x| x * 3);
        assert_eq!(mapped.data(), vec![21]);
        assert_eq!(mapped.shape(), Vec::new());
    }

    /// Test that mapping does not mutate the original tensor.
    #[test]
    fn test_map_does_not_modify_original_tensor() {
        let tensor = Tensor::new([1, 2, 3, 4], [2, 2], true);
        let mapped = tensor.map(|x| x * 10);
        assert_eq!(tensor.data(), vec![1, 2, 3, 4]);
        assert_eq!(tensor.shape(), vec![2, 2]);
        assert_eq!(mapped.data(), vec![10, 20, 30, 40]);
        assert_eq!(mapped.shape(), vec![2, 2]);
    }

    /// Test that map closures can capture values from their environment.
    #[test]
    fn test_map_can_capture_environment() {
        let offset = 5;
        let tensor = Tensor::new([1, 2, 3], [3], true);
        let mapped = tensor.map(|x| x + offset);
        assert_eq!(mapped.data(), vec![6, 7, 8]);
        assert_eq!(mapped.shape(), vec![3]);
    }

    /// Test that mapping works for floating-point tensors.
    #[test]
    fn test_map_with_floating_point_values() {
        let tensor = Tensor::new([1.0, 2.5, -3.0], [3], true);
        let mapped = tensor.map(|x| x / 2.0);
        assert_eq!(mapped.data(), vec![0.5, 1.25, -1.5]);
        assert_eq!(mapped.shape(), vec![3]);
    }

    /// Test that mapping works for boolean tensors.
    #[test]
    fn test_map_with_bool_values() {
        let tensor = Tensor::new([true, false, true, false], [2, 2], true);
        let mapped = tensor.map(|x| !x);
        assert_eq!(mapped.data(), vec![false, true, false, true]);
        assert_eq!(mapped.shape(), vec![2, 2]);
    }

    /// Test that mapping works for character tensors.
    #[test]
    fn test_map_with_char_values() {
        let tensor = Tensor::new(['a', 'b', 'c'], [3], true);
        let mapped = tensor.map(remap_char);
        assert_eq!(mapped.data(), vec!['x', 'y', 'z']);
        assert_eq!(mapped.shape(), vec![3]);
    }

    /// Test that mapping preserves higher-rank tensor shapes.
    #[test]
    fn test_map_preserves_higher_rank_shape() {
        let tensor = Tensor::new([1, 2, 3, 4, 5, 6, 7, 8], [2, 2, 2], true);
        let mapped = tensor.map(|x| x - 1);
        assert_eq!(mapped.data(), vec![0, 1, 2, 3, 4, 5, 6, 7]);
        assert_eq!(mapped.shape(), vec![2, 2, 2]);
    }
}
