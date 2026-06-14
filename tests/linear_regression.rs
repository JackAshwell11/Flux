use flux::loss::l2_loss;
use flux::tensor::Tensor;

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
        let y_pred = &x_tensor * &gradient + &y_intercept;

        // Compute the loss function and check that its decreasing
        let current_loss = l2_loss(&y_pred, &y_tensor).get(0);
        assert!(current_loss <= previous_loss + LOSS_TOLERANCE);
        previous_loss = current_loss;

        // Compute the gradient of the loss function
        let difference = &y_pred - &y_tensor;
        let gradient_derivative = (&x_tensor * &difference).mean();
        let intercept_derivative = difference.mean();

        // Update the weights using the gradient descent algorithm
        gradient -= &gradient_derivative * LEARNING_RATE;
        y_intercept -= &intercept_derivative * LEARNING_RATE;
    }

    // Check that the output is correct
    assert!((gradient.get(0) - EXPECTED_GRADIENT).abs() < EPSILON);
    assert!((y_intercept.get(0) - EXPECTED_INTERCEPT).abs() < EPSILON);
}
