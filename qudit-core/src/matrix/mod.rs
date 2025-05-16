//! Matrix and Matrix-like types for use in the Openqudit library.

mod matmat;
mod matvec;

/// Re-export of the macro constructors from the faer_core crate.
pub use faer::mat;

pub use faer::Col;
pub use faer::Row;
pub use faer::RowRef;
pub use faer::ColRef;
pub use faer::RowMut;
pub use faer::ColMut;

/// Re-export of the basic matrix types from the faer_core crate.
pub use faer::Mat;
pub use faer::MatMut;
pub use faer::MatRef;

/// Matrix Vector (3d tensor) type.
pub use matvec::MatVec;
pub use matvec::MatVecMut;
pub use matvec::MatVecRef;

/// Matrix Matrix (4d tensor) type.
pub use matmat::SymSqMatMat;
pub use matmat::SymSqMatMatMut;
pub use matmat::SymSqMatMatRef;
