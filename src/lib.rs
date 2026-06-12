//! Flux is a Rust-based machine learning and reinforcement learning library.

/// Tensor operations and storage.
pub mod tensor;

/// Loss functions.
pub mod loss;

/// Re-export tensor operations.
pub use tensor::*;

/// Re-export loss functions.
pub use loss::*;
