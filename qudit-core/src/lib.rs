#![warn(missing_docs)]

//! Qudit-Core is the core package in the OpenQudit library.
mod bitwidth;
mod function;
mod perm;
mod radices;
mod radix;
mod scalar;
mod system;

pub mod accel;
pub mod matrix;
pub mod memory;
pub mod unitary;

pub use bitwidth::BitWidthConvertible;
pub use function::HasBounds;
pub use function::HasParams;
pub use function::HasPeriods;
pub use function::ParamIndices;
pub use perm::calc_index_permutation;
pub use perm::QuditPermutation;
pub use radices::QuditRadices;
pub use radices::ToRadices;
pub use radix::ToRadix;
pub use scalar::ComplexScalar;
pub use scalar::RealScalar;
pub use system::ClassicalSystem;
pub use system::HybridSystem;
pub use system::QuditSystem;

////////////////////////////////////////////////////////////////////////
/// Complex number types.
////////////////////////////////////////////////////////////////////////
pub use faer::c32;
pub use faer::c64;
////////////////////////////////////////////////////////////////////////
