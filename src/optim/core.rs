use crate::tensor::core::Tensor;

/// Represents an optimiser for a model.
pub trait Optimiser {
    /// Returns a reference to the parameters of the model.
    fn params(&self) -> &[Tensor<f32>];

    /// Performs a single step of the optimiser.
    fn step(&self);

    /// Sets the gradients of all parameters to zero.
    fn zero_grad(&self) {
        for param in self.params() {
            param.fill_grad(0.0);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::core::Tensor;

    /// A mock optimiser for testing purposes.
    struct DummyOptimiser {
        params: Vec<Tensor<f32>>,
    }

    impl DummyOptimiser {
        /// Creates a new `DummyOptimiser` with the given parameters.
        pub fn new(params: Vec<Tensor<f32>>) -> Self {
            Self { params }
        }
    }

    impl Optimiser for DummyOptimiser {
        /// Returns a reference to the parameters of the optimiser.
        fn params(&self) -> &[Tensor<f32>] {
            &self.params
        }

        /// Performs a single step of the optimiser.
        fn step(&self) {
            for param in &self.params {
                param.fill_grad(1.0);
            }
        }
    }

    /// Test that `zero_grad` correctly resets gradients to zero.
    #[test]
    fn test_zero_grad() {
        let tensor_one = Tensor::new([0.5], [1], true);
        let tensor_two = Tensor::new([1.5], [1], true);
        let optimiser = DummyOptimiser::new(vec![tensor_one.clone(), tensor_two.clone()]);
        tensor_one.fill_grad(-0.5);
        tensor_two.fill_grad(1.0);
        optimiser.zero_grad();
        assert_eq!(tensor_one.grad(), vec![0.0]);
        assert_eq!(tensor_two.grad(), vec![0.0]);
    }
}
