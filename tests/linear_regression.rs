use flux::loss::l2_loss;
use flux::tensor::core::Tensor;

/// Controls the number of iterations to run the training loop for.
const ITERATIONS: usize = 10000;

/// Controls the learning rate for the gradient descent algorithm.
const LEARNING_RATE: f32 = 0.01;

/// Controls the tolerance for the loss function.
const LOSS_TOLERANCE: f32 = 1e-10;

/// Controls the expected gradient value after training.
const EXPECTED_GRADIENT: f32 = 2.0;

/// Controls the expected y-intercept value after training.
const EXPECTED_INTERCEPT: f32 = 1.0;

/// Controls the tolerance for the test.
const EPSILON: f32 = 1e-4;

/// Performs an in-place stochastic gradient descent (SGD) update:
///     data -= lr * grad
pub fn gradient_descent_step(param: &mut Tensor<f32>) {
    let mut state = param.state.borrow_mut();
    for i in 0..state.data.len() {
        state.data[i] -= LEARNING_RATE * state.grad[i];
    }
}

/// Test that the linear regression algorithm works correctly.
#[test]
fn test_linear_regression() {
    // Initialise the input tensors
    let x_tensor = Tensor::new([1.0, 2.0, 3.0, 4.0], [4]);
    let y_tensor = Tensor::new([3.0, 5.0, 7.0, 9.0], [4]);

    // Initialise the output tensors which will be optimised
    let mut gradient: Tensor<f32> = Tensor::zeros([1]);
    let mut y_intercept: Tensor<f32> = Tensor::zeros([1]);

    // Track the previous loss value
    let mut previous_loss = f32::MAX;

    // Run the gradient descent algorithm
    for _ in 0..ITERATIONS {
        // Calculate the predicted output
        let y_pred = x_tensor.clone() * gradient.clone() + y_intercept.clone();

        // Compute the loss function and check that it's decreasing
        let loss = l2_loss(y_pred.clone(), y_tensor.clone());
        assert!(loss.get(0) <= previous_loss + LOSS_TOLERANCE);
        previous_loss = loss.get(0);

        // Compute the backward pass to calculate the gradients
        loss.backward();

        // Update the weights using the gradient descent algorithm
        gradient_descent_step(&mut gradient);
        gradient_descent_step(&mut y_intercept);

        // Zero the gradients to prepare for the next iteration
        gradient.zero_grad();
        y_intercept.zero_grad();
    }

    // Check that the output is correct
    assert!((gradient.get(0) - EXPECTED_GRADIENT).abs() < EPSILON);
    assert!((y_intercept.get(0) - EXPECTED_INTERCEPT).abs() < EPSILON);
}
