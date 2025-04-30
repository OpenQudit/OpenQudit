//! This module implements unitary matrix objects and traits.

mod function;
mod matrix;

pub use function::DifferentiableUnitaryFn;
pub use function::DoublyDifferentiableUnitaryFn;
pub use function::UnitaryFn;
pub use matrix::UnitaryMatrix;
