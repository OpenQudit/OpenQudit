#![warn(missing_docs)]

//! Qudit-Core is the core package in the OpenQudit library.
mod bitwidth;
mod function;
mod perm;
mod radices;
mod radix;
mod scalar;
mod system;
mod shape;
mod quantum;

pub mod accel;
// pub mod matrix;
pub mod memory;
// pub mod unitary;
pub mod array;
// pub mod state;

pub use bitwidth::BitWidthConvertible;
pub use function::HasBounds;
pub use function::HasParams;
pub use function::HasPeriods;
pub use function::ParamIndices;
pub use function::ParamInfo;
pub use perm::calc_index_permutation;
pub use perm::QuditPermutation;
pub use radices::QuditRadices;
pub use radices::ToRadices;
pub use radix::ToRadix;
pub use scalar::ComplexScalar;
pub use scalar::RealScalar;
pub use shape::TensorShape;
pub use system::ClassicalSystem;
pub use system::HybridSystem;
pub use system::QuditSystem;
pub use quantum::UnitaryMatrix;
pub use quantum::Ket;

////////////////////////////////////////////////////////////////////////
/// Complex number types.
////////////////////////////////////////////////////////////////////////
pub use faer::c32;
pub use faer::c64;
pub use faer::cx128 as c128;
////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////
/// Python Register Helpers.
////////////////////////////////////////////////////////////////////////
#[cfg(feature = "python")]
use pyo3::prelude::{Bound, PyModule, PyResult};

/// A trait for objects that can register importables with a PyO3 module.
#[cfg(feature = "python")]
pub struct PyRegistrar {
    /// The registration function
    pub func: fn(parent_module: &Bound<'_, PyModule>) -> PyResult<()>,
}

#[cfg(feature = "python")]
inventory::collect!(PyRegistrar);
////////////////////////////////////////////////////////////////////////
