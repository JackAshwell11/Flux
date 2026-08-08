use std::ops::AddAssign;

/// Broadcast a tensor for a forward elementwise operation.
pub fn broadcast_forward<T>(data: &[T], target_size: usize) -> Vec<T>
where
    T: Copy,
{
    if data.len() == target_size {
        // Tensor already has target size
        data.to_vec()
    } else if data.len() == 1 {
        // Expand scalar to fill target size
        vec![data[0]; target_size]
    } else {
        // Don't currently support broadcasting between different sizes
        panic!(
            "Cannot broadcast tensor of size {} to size {}",
            data.len(),
            target_size
        );
    }
}

/// Reduce a broadcasted gradient back to the original tensor shape.
pub fn broadcast_backward<T>(grad: &[T], parent_size: usize) -> Vec<T>
where
    T: Copy + Default + AddAssign,
{
    if grad.len() == parent_size {
        // Gradient matches the parent's shape
        grad.to_vec()
    } else if parent_size == 1 {
        // Sum gradient contributions back into one value
        let mut result = vec![T::default(); 1];
        for value in grad {
            result[0] += *value;
        }
        result
    } else if grad.len() == 1 {
        // Expand gradient to fill parent size
        vec![grad[0]; parent_size]
    } else {
        // Don't currently support broadcasting between different sizes
        panic!(
            "Cannot reduce gradient of size {} to size {}",
            grad.len(),
            parent_size
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use test_case::test_case;

    /// Test that broadcasting forwards works correctly.
    #[test_case([1.0, 2.0, 3.0], 3, [1.0, 2.0, 3.0]; "matching vector size")]
    #[test_case([10.0], 1, [10.0]; "matching scalar size")]
    #[test_case([-1.0, -2.0, -3.0, -4.0], 4, [-1.0, -2.0, -3.0, -4.0]; "matching negative size")]
    #[test_case([5.0], 3, [5.0, 5.0, 5.0]; "scalar to vector")]
    #[test_case([-2.0], 4, [-2.0, -2.0, -2.0, -2.0]; "negative scalar to vector")]
    fn test_broadcast_forward<const N: usize, const E: usize>(
        data: [f32; N],
        target_size: usize,
        expected: [f32; E],
    ) {
        assert_eq!(broadcast_forward(&data, target_size), expected);
    }

    /// Test that broadcasting forwards panics for unsupported sizes.
    #[test_case([1.0, 2.0], 3 => panics "Cannot broadcast tensor of size 2 to size 3"; "larger target size")]
    #[test_case([1.0, 2.0, 3.0], 2 => panics "Cannot broadcast tensor of size 3 to size 2"; "smaller target size")]
    fn test_broadcast_forward_invalid<const N: usize>(data: [f32; N], target_size: usize) {
        let _ = broadcast_forward(&data, target_size);
    }

    /// Test that broadcasting backwards works correctly,
    #[test_case([1.0, 2.0, 3.0], 3, [1.0, 2.0, 3.0]; "matching vector gradient")]
    #[test_case([10.0], 1, [10.0]; "matching scalar gradient")]
    #[test_case([-1.0, -2.0, -3.0, -4.0], 4, [-1.0, -2.0, -3.0, -4.0]; "matching negative gradient")]
    #[test_case([-5.0], 3, [-5.0, -5.0, -5.0]; "matching negative scalar gradient")]
    #[test_case([1.0, 2.0, 3.0], 1, [6.0]; "sum vector into scalar")]
    #[test_case([-1.0, -2.0, -3.0, -4.0], 1, [-10.0]; "sum negative into scalar")]
    #[test_case([5.0, 5.0], 1, [10.0]; "sum repeated gradients")]
    #[test_case([5.0], 3, [5.0, 5.0, 5.0]; "expand scalar scalar gradients")]
    #[test_case([-2.0], 4, [-2.0, -2.0, -2.0, -2.0]; "expand negative scalar gradients")]
    fn test_broadcast_backward<const N: usize, const E: usize>(
        grad: [f32; N],
        parent_size: usize,
        expected: [f32; E],
    ) {
        assert_eq!(broadcast_backward(&grad, parent_size), expected);
    }

    /// Test that broadcasting backward panic for unsupported sizes.
    #[test_case([1.0, 2.0], 3 => panics "Cannot reduce gradient of size 2 to size 3"; "larger parent size")]
    #[test_case([1.0, 2.0, 3.0], 2 => panics "Cannot reduce gradient of size 3 to size 2"; "smaller parent size")]
    fn test_broadcast_backward_invalid<const N: usize>(grad: [f32; N], parent_size: usize) {
        let _ = broadcast_backward(&grad, parent_size);
    }
}
