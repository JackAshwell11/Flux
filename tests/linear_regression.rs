use flux::loss::l2_loss;
use flux::optim::core::Optimiser;
use flux::optim::sgd::SGD;
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

/// Test that the linear regression algorithm works correctly.
#[test]
fn test_linear_regression() {
    // Initialise the input tensors
    let x_tensor = Tensor::new([1.0, 2.0, 3.0, 4.0], [4], false);
    let y_tensor = Tensor::new([3.0, 5.0, 7.0, 9.0], [4], false);

    // Initialise the output tensors which will be optimised
    let gradient: Tensor<f32> = Tensor::zeros([1], true);
    let y_intercept: Tensor<f32> = Tensor::zeros([1], true);

    // Create the optimiser for the gradient descent algorithm
    let optimiser = SGD::new(vec![gradient.clone(), y_intercept.clone()], LEARNING_RATE);

    // Track the previous loss value
    let mut previous_loss = f32::MAX;

    // Run the gradient descent algorithm
    for _ in 0..ITERATIONS {
        // Calculate the predicted output
        let y_pred = x_tensor.clone() * gradient.clone() + y_intercept.clone();

        // Compute the loss function and check that it's decreasing
        let loss = l2_loss(y_pred.clone(), y_tensor.clone());
        assert!(loss.data()[0] <= previous_loss + LOSS_TOLERANCE);
        previous_loss = loss.data()[0];

        // Compute the backward pass to calculate the gradients
        loss.backward();

        // Perform a gradient descent step and reset the gradients for the next step
        optimiser.step();
        optimiser.zero_grad();
    }

    // Check that the output is correct
    assert!((gradient.data()[0] - EXPECTED_GRADIENT).abs() < EPSILON);
    assert!((y_intercept.data()[0] - EXPECTED_INTERCEPT).abs() < EPSILON);
}
