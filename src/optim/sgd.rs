use crate::optim::core::Optimiser;
use crate::tensor::core::Tensor;

/// Represents a Stochastic Gradient Descent optimiser.
pub struct SGD {
    /// The parameters to optimise.
    params: Vec<Tensor<f32>>,

    /// The learning rate.
    lr: f32,
}

impl SGD {
    /// Create an SGD optimiser.
    #[must_use]
    pub const fn new(params: Vec<Tensor<f32>>, lr: f32) -> Self {
        Self { params, lr }
    }
}

impl Optimiser for SGD {
    /// Returns a reference to the parameters of the optimiser.
    fn params(&self) -> &[Tensor<f32>] {
        &self.params
    }

    /// Performs a single step of the optimiser.
    fn step(&self) {
        for param in &self.params {
            let mut state = param.state.borrow_mut();
            for i in 0..state.data.len() {
                state.data[i] = self.lr.mul_add(-state.grad[i], state.data[i]);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::core::Tensor;
    use test_case::test_case;

    /// Test that `SGD` can be initialised correctly.
    #[test]
    fn test_sgd_initialization() {
        let tensor_one = Tensor::new([1.0], [1], true);
        let tensor_two = Tensor::new([2.0], [1], true);
        let optimiser = SGD::new(vec![tensor_one.clone(), tensor_two.clone()], 0.01);
        assert_eq!(optimiser.lr, 0.01);
        assert_eq!(optimiser.params().len(), 2);
        assert!(optimiser.params().contains(&tensor_one));
        assert!(optimiser.params().contains(&tensor_two));
    }

    /// Test that `step` correctly functions with various learning rates and graient values.
    #[test_case(0.1, 0.001, [0.099, 0.198])]
    #[test_case(1.0, 0.01, [0.99, 1.98])]
    #[test_case(2.0, 0.1, [1.9, 3.8])]
    fn test_sgd_step<const N: usize>(initial_value: f32, lr: f32, expected: [f32; N]) {
        let tensor_one = Tensor::new([initial_value], [1], true);
        let tensor_two = Tensor::new([initial_value * 2.0], [1], true);
        let optimiser = SGD::new(vec![tensor_one.clone(), tensor_two.clone()], lr);
        tensor_one.fill_grad(1.0);
        tensor_two.fill_grad(2.0);
        optimiser.step();
        assert_eq!(tensor_one.state.borrow().data, [expected[0]]);
        assert_eq!(tensor_two.state.borrow().data, [expected[1]]);
    }

    /// Test that `step` correctly functions with only one tensor to optimise.
    #[test_case(1.0, 0.001, [0.999, 2.0])]
    #[test_case(10.0, 0.01, [9.99, 20.0])]
    #[test_case(100.0, 0.1, [99.9, 200.0])]
    fn test_sgd_step_single_tensor<const N: usize>(
        initial_value: f32,
        lr: f32,
        expected: [f32; N],
    ) {
        let tensor_one = Tensor::new([initial_value], [1], true);
        let tensor_two = Tensor::new([initial_value * 2.0], [1], true);
        let optimiser = SGD::new(vec![tensor_one.clone()], lr);
        tensor_one.fill_grad(1.0);
        tensor_two.fill_grad(1.0);
        optimiser.step();
        assert_eq!(tensor_one.state.borrow().data, [expected[0]]);
        assert_eq!(tensor_two.state.borrow().data, [expected[1]]);
    }
}
