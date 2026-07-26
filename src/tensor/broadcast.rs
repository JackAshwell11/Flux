use std::ops::AddAssign;

/// Broadcast a tensor for a forward elementwise operation.
pub(crate) fn broadcast_forward<T>(data: &[T], target_size: usize) -> Vec<T>
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
pub(crate) fn broadcast_backward<T>(grad: &[T], parent_size: usize) -> Vec<T>
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
