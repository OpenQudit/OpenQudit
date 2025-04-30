mod math;
pub use math::*;
mod frpr;
pub mod matrix;
pub mod perm;
pub mod pure;
mod scalar;
pub mod unitary;

pub mod function;

pub use faer_core::complex_native::c32;
pub use faer_core::complex_native::c64;
pub use frpr::fused_reshape_permute_reshape_into;
pub use frpr::fused_reshape_permute_reshape_into_prepare;
pub use frpr::fused_reshape_permuted_reshape_into_impl;
pub use function::BoundedFn;
pub use function::Function;
pub use perm::QuditPermutation;
pub use scalar::ComplexScalar;
pub use scalar::RealScalar;

// pub type Matrix<T: Entity> = Mat<T>;
// pub type Vector<T: Entity> = AVec<T>;
// Favor
pub use faer_core::Mat as Matrix;
pub use faer_core::Row as Vector;
