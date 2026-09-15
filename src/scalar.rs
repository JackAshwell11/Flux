use num_traits::{Float, NumAssign, NumCast, Signed};
use std::fmt::Debug;
use std::iter::Sum;

/// Trait shared by every tensor element type.
pub trait FluxNum:
    Copy + Default + Debug + NumAssign + Signed + Sum + PartialOrd + NumCast
{
}
impl<T> FluxNum for T where
    T: Copy + Default + Debug + NumAssign + Signed + Sum + PartialOrd + NumCast
{
}

/// Trait shared for anything requiring floating-point behaviour.
pub trait FluxFloat: FluxNum + Float {}
impl<T> FluxFloat for T where T: FluxNum + Float {}
