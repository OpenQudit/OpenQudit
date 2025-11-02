//! Implements the tensor struct and associated methods for the Openqudit library.

mod tensor;
mod symsq;

pub use tensor::Tensor;
pub use tensor::TensorRef;
pub use tensor::TensorMut;

pub use symsq::SymSqTensor;
pub use symsq::SymSqTensorRef;
pub use symsq::SymSqTensorMut;

// Helper for bounds checking
fn is_in_bounds<const D: usize>(indices: &[usize; D], dimensions: &[usize; D]) -> bool {
    for i in 0..D {
        if indices[i] >= dimensions[i] {
            return false;
        }
    }
    true
}

// Helper for panic on out of bounds
fn check_bounds<const D: usize>(indices: &[usize; D], dimensions: &[usize; D]) {
    for i in 0..D {
        if indices[i] >= dimensions[i] {
            panic!(
                "Index out of bounds: index {} is {} but dimension {} has size {}",
                i, indices[i], i, dimensions[i]
            );
        }
    }
}

fn calc_continuous_strides<const D: usize>(dims: &[usize; D]) -> [usize; D] {
    if D == 0 {
        [0; D]
    } else if D == 1 {
        [1; D]
    } else {
        let mut stride_acm = 1;
        let mut strides = [0; D];
        for (i, d) in dims.iter().enumerate().rev() {
            strides[i] = stride_acm;
            stride_acm *= d;
        }
        strides
    }
}

fn calc_cache_optimal_strides<const D: usize>(dims: &[usize; D]) -> [usize; D] {
    todo!()
}
