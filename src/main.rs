use flux::loss::l2_loss;
use flux::optim::core::Optimiser;
use flux::optim::sgd::SGD;
use flux::tensor::core::Tensor;

const ITERATIONS: usize = 1000;

const LEARNING_RATE: f32 = 0.1;

fn main() {
    // Initialise the input tensors
    let x_tensor = Tensor::new([1.0, 2.0, 3.0, 4.0], [4], false);
    let y_tensor = Tensor::new([3.0, 5.0, 7.0, 9.0], [4], false);

    // Initialise the output tensors which will be optimised
    let gradient = Tensor::zeros([1], true);
    let y_intercept = Tensor::zeros([1], true);

    // Create the optimiser for the gradient descent algorithm
    let optimiser = SGD::new(vec![gradient.clone(), y_intercept.clone()], LEARNING_RATE);

    // Run the gradient descent algorithm
    for _ in 0..ITERATIONS {
        // Calculate the predicted output
        let y_pred = x_tensor.clone() * gradient.clone() + y_intercept.clone();

        // Compute the loss function and check that it's decreasing
        let loss = l2_loss(y_pred.clone(), y_tensor.clone());

        // Compute the backward pass to calculate the gradients
        loss.backward();

        // Perform a gradient descent step and reset the gradients for the next step
        optimiser.step();
        optimiser.zero_grad();

        println!("grad {:?}", gradient.data());
        println!("y-intercept {:?}", y_intercept.data());
    }
}
