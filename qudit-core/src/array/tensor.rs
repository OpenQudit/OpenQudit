use std::fmt::{self, Debug, Display, Formatter, Write};
use std::ptr::NonNull;


// Helper for flat index calculation
fn calculate_flat_index<const D: usize>(indices: &[usize; D], strides: &[usize; D]) -> usize {
    let mut flat_idx = 0;
    for i in 0..D {
        flat_idx += indices[i] * strides[i];
    }
    flat_idx
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

use crate::memory::{alloc_zeroed_memory, calc_next_stride, Memorable, MemoryBuffer};

pub struct Tensor<C: Memorable, const D: usize> {
    data: MemoryBuffer<C>,
    dimensions: [usize; D],
    strides: [usize; D],
}

impl<C: Memorable, const D: usize> Tensor<C, D> {
    pub fn zeros(dims: &[usize]) -> Self {
        assert_eq!(dims.len(), D);

        // special case for zero dimenions
        if D == 0 {
            return Self {
                data: alloc_zeroed_memory::<C>(1),
                dimensions: [0; D],
                strides: [0; D],
            };
        }

        // special case for one dimension
        if D == 1 {
            return Self {
                data: alloc_zeroed_memory::<C>(dims[0]),
                dimensions: [dims[0]; D],
                strides: [1; D],
            };
        }
        let mut stride_acm = 1;
        let mut strides = [0; D];
        let mut dimensions = [0; D];
        for (i, d) in dims.iter().enumerate().rev() {
            strides[i] = stride_acm;
            dimensions[i] = *d;
            stride_acm *= d;
            // TODO: better stride calculations/spacing?
        }

        let data = alloc_zeroed_memory::<C>(stride_acm);
        Self {
            data,
            dimensions,
            strides,
        }
    }

    /// returns a pointer to the tensor data.
    #[inline(always)]
    pub fn as_ptr(&self) -> *const C {
        self.data.as_ptr()
    }

    /// returns a mutable pointer to the tensor data.
    #[inline(always)]
    pub fn as_mut_ptr(&mut self) -> *mut C {
        self.data.as_mut_ptr()
    }

    /// Returns a reference to the element at the given multi-dimensional index.
    ///
    /// # Panics
    /// Panics if the index is out of bounds.
    pub fn at(&self, indices: [usize; D]) -> &C {
        check_bounds(&indices, &self.dimensions);
        // SAFETY: Bounds are checked by `check_bounds`.
        unsafe { self.at_unchecked(indices) }
    }

    /// Returns a mutable reference to the element at the given multi-dimensional index.
    ///
    /// # Panics
    /// Panics if the index is out of bounds.
    pub fn at_mut(&mut self, indices: [usize; D]) -> &mut C {
        check_bounds(&indices, &self.dimensions);
        // SAFETY: Bounds are checked by `check_bounds`.
        unsafe { self.at_unchecked_mut(indices) }
    }

    /// Returns a reference to the element at the given multi-dimensional index, without bounds checking.
    ///
    /// # Safety
    /// The caller must ensure that `indices` are within the tensor's dimensions.
    #[inline(always)]
    pub unsafe fn at_unchecked(&self, indices: [usize; D]) -> &C {
        let flat_idx = calculate_flat_index(&indices, &self.strides);
        &*self.data.as_ptr().add(flat_idx)
    }

    /// Returns a reference to the element at the given multi-dimensional index, without bounds checking.
    ///
    /// # Safety
    /// The caller must ensure that `indices` are within the tensor's dimensions.
    #[inline(always)]
    pub unsafe fn at_unchecked_mut(&mut self, indices: [usize; D]) -> &mut C {
        let flat_idx = calculate_flat_index(&indices, &self.strides);
        &mut *self.data.as_mut_ptr().add(flat_idx)
    }

    /// Returns the tensor's dimensions.
    pub fn dimensions(&self) -> &[usize; D] {
        &self.dimensions
    }

    /// Returns the tensor's shape.
    pub fn shape(&self) -> &[usize; D] {
        &self.dimensions
    }

    /// Returns the tensor's strides.
    pub fn strides(&self) -> &[usize; D] {
        &self.strides
    }
}


// Helper struct for recursively formatting the tensor data
// to display it as a multi-dimensional array.
struct TensorDataDebugHelper<'a, C: Display> {
    data_ptr: *const C,
    dimensions: &'a [usize],
    strides: &'a [usize],
    current_dim_idx: usize,
    current_flat_offset: usize,
}

impl<'a, C: Display> Debug for TensorDataDebugHelper<'a, C> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let indent = "\t".repeat(self.current_dim_idx);
        // Base case: If we've reached the deepest dimension level,
        // it means we are at an individual element. Print its value directly.
        if self.current_dim_idx == self.dimensions.len() {
            // SAFETY: The `current_flat_offset` is calculated based on the tensor's
            // dimensions and strides. It is assumed to be within the bounds of the
            // allocated data, as guaranteed by the `Tensor` and `TensorRef` structures.
            unsafe {
                write!(f, "{}", &*self.data_ptr.add(self.current_flat_offset))
            }
        } else {
            // Recursive case: We are at an intermediate dimension.
            // Print this dimension as a list of sub-tensors/elements.
            let dim_size = self.dimensions[self.current_dim_idx];
            let dim_stride = self.strides[self.current_dim_idx];

            // let mut list_formatter = f.debug_list();
            if self.current_dim_idx == self.dimensions.len() - 1 {
                write!(f, "{}[", indent)?;
            } else {
                write!(f, "{}[\n", indent)?;
            }
            for i in 0..dim_size {
                let next_offset = self.current_flat_offset + i * dim_stride;

                write!(f, "{:?}", TensorDataDebugHelper {
                    data_ptr: self.data_ptr,
                    dimensions: self.dimensions,
                    strides: self.strides,
                    current_dim_idx: self.current_dim_idx + 1,
                    current_flat_offset: next_offset,
                })?;

                if self.current_dim_idx == self.dimensions.len() - 1 && i != dim_size - 1 {
                    write!(f, ", ")?;
                }

            }
            if self.current_dim_idx == self.dimensions.len() - 1 {
                write!(f, "],\n",)
            } else {
                write!(f, "{}],\n", indent)
            }
            // write!(f, "\n")
            // list_formatter.finish()
        }
    }
}


impl<C: Display + Debug + Memorable, const D: usize> Debug for Tensor<C, D> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_struct("Tensor")
            .field("dimensions", &self.dimensions)
            .field("strides", &self.strides)
            .field("data", &TensorDataDebugHelper {
                data_ptr: self.data.as_ptr(), // Pointer to the start of the data buffer
                dimensions: &self.dimensions,
                strides: &self.strides,
                current_dim_idx: 0, // Start formatting from the first dimension (index 0)
                current_flat_offset: 0, // Start from offset 0 in the flat data buffer
            })
            .finish()
    }
}

impl<'a, C: Display + Debug + Memorable, const D: usize> Debug for TensorRef<'a, C, D> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_struct("TensorRef")
            .field("dimensions", &self.dimensions)
            .field("strides", &self.strides)
            .field("data", &TensorDataDebugHelper {
                data_ptr: self.data.as_ptr(), // `self.data` is already a `*const C` for `TensorRef`
                dimensions: &self.dimensions,
                strides: &self.strides,
                current_dim_idx: 0,
                current_flat_offset: 0,
            })
            .finish()
    }
}

// impl<C: std::fmt::Debug + Memorable> std::fmt::Debug for Tensor<C> {
//     fn fmt(&self, f: &mut std::fmt::formatter<'_>) -> std::fmt::result {
//         f.debug_list()
//             .entries((0..self.nblocks).map(|b| self.block_ref(b)))
//             .finish()
//     }
// }

// impl<C: Memorable> Index<(usize, usize, usize, usize)> for Tensor<C> {
//     type output = c;

//     fn index(&self, idx: (usize, usize, usize, usize)) -> &self::output {
//         self.panic_on_out_of_bounds(idx.0, idx.1, idx.2, idx.3);
//         // safety: the bounds have been checked.
//         &self.data[idx.0 * self.block_stride
//             + idx.1 * self.mat_stride
//             + idx.3 * self.col_stride
//             + idx.2]
//     }
// }

// impl<c: memorable> indexmut<(usize, usize, usize, usize)> for tensor4d<c> {
//     fn index_mut(&mut self, idx: (usize, usize, usize, usize)) -> &mut self::output {
//         self.panic_on_out_of_bounds(idx.0, idx.1, idx.2, idx.3);
//         // safety: the bounds have been checked.
//         &mut self.data[idx.0 * self.block_stride
//             + idx.1 * self.mat_stride
//             + idx.3 * self.col_stride
//             + idx.2]
//     }
// }

pub struct TensorRef<'a, C: Memorable, const D: usize> {
    data: NonNull<C>,
    dimensions: [usize; D],
    strides: [usize; D],
    __marker: std::marker::PhantomData<&'a C>,
}

impl<'a, C: Memorable, const D: usize> TensorRef<'a, C, D> {
    /// Creates a `TensorRef` from pointers to tensor data, dimensions, and strides.
    ///
    /// # Safety
    ///
    /// The caller must ensure:
    /// - The pointers are valid and non-null.
    /// - For each unit, the entire memory region addressed by the tensor is
    ///   within a single allocation.
    /// - The memory is accessible by the pointer.
    /// - No mutable aliasing occurs. No mutable references to the tensor data
    ///  exist when the `MatVecRef` is alive.
    pub unsafe fn from_raw_parts(
        data: *const C,
        dimensions: [usize; D],
        strides: [usize; D],
    ) -> Self {
        // SAFETY: The pointer is never used in an mutable context.
        let ptr = unsafe { NonNull::new_unchecked(data as *mut C) };

        Self {
            data: ptr,
            dimensions,
            strides,
            __marker: std::marker::PhantomData,
        }
    }

    /// returns a pointer to the tensor data.
    #[inline(always)]
    pub fn as_ptr(&self) -> *const C {
        self.data.as_ptr()
    }

    /// Returns a reference to the element at the given multi-dimensional index.
    ///
    /// # Panics
    /// Panics if the index is out of bounds.
    pub fn at(&self, indices: [usize; D]) -> &C {
        check_bounds(&indices, &self.dimensions);
        // SAFETY: Bounds are checked by `check_bounds`.
        unsafe { self.at_unchecked(indices) }
    }

    /// Returns a reference to the element at the given multi-dimensional index, without bounds checking.
    ///
    /// # Safety
    /// The caller must ensure that `indices` are within the tensor's dimensions.
    #[inline(always)]
    pub unsafe fn at_unchecked(&self, indices: [usize; D]) -> &C {
        let flat_idx = calculate_flat_index(&indices, &self.strides);
        &*self.data.as_ptr().add(flat_idx)
    }

    /// Returns the tensor's dimensions.
    pub fn dimensions(&self) -> &[usize; D] {
        &self.dimensions
    }

    /// Returns the tensor's strides.
    pub fn strides(&self) -> &[usize; D] {
        &self.strides
    }
}
