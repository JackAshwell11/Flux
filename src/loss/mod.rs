//! Holds loss functions for machine learning.

/// The Mean Absolute Error (L1) loss function.
pub mod l1_loss;

/// The Mean Squared Error (L2) loss function.
pub mod l2_loss;

/// Re-export loss functions.
pub use l1_loss::l1_loss;
pub use l2_loss::l2_loss;
