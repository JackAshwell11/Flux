//! Holds tensor-related functionality.

/// Core tensor functionality.
pub mod core;

/// Tensor operations.
pub mod ops;

/// Re-export core tensor functionality.
pub use core::Tensor;

/// Re-export tensor operations.
pub use ops::*;
