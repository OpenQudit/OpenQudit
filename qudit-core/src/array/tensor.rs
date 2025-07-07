use std::fmt::{self, Debug, Display, Formatter, Write};
use std::ptr::NonNull;
use crate::memory::{alloc_zeroed_memory, calc_next_stride, Memorable, MemoryBuffer};

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

/// A tensor struct that holds data in an aligned memory buffer.
pub struct Tensor<C: Memorable, const D: usize> {
    /// The data buffer containing the tensor elements.
    pub data: MemoryBuffer<C>,
    dimensions: [usize; D],
    strides: [usize; D],
} 

impl<C: Memorable, const D: usize> Tensor<C, D> {
    /// Helper function to calculate strides based on dimensions.
    /// This function is intended for internal use by the constructors.
    ///
    /// For a D-dimensional tensor with dimensions `[d0, d1, ..., dD-1]`,
    /// the strides are calculated such that `strides[i]` is the number of elements
    /// to skip in the flattened data buffer to move one step along dimension `i`.
    /// The last dimension (D-1) always has a stride of 1.
    fn calculate_strides(dimensions: &[usize; D]) -> [usize; D] {
        let mut strides = [0; D];
        if D == 0 {
            // Scalar case (0-dimensional tensor): no dimensions to stride over.
            // The strides array will be empty.
            return strides;
        }

        // The innermost dimension (D-1) has a stride of 1, as moving one step
        // in this dimension means moving one element in the flattened data.
        strides[D - 1] = 1;

        // Iterate backwards from the second-to-last dimension to calculate strides.
        // The stride for dimension `i` is the stride for `i+1` multiplied by the size of dimension `i+1`.
        for i in (0..(D - 1)).rev() {
            strides[i] = strides[i + 1] * dimensions[i + 1];
        }
        strides
    }


    /// Creates a new `Tensor` from a `MemoryBuffer` and its dimensions.
    ///
    /// The `MemoryBuffer` should contain the flattened data of the tensor.
    /// The `dimensions` array defines the shape of the tensor.
    ///
    /// # Panics
    /// Panics if the total number of elements implied by `dimensions`
    /// (which is the product of all dimension sizes) does not match the length
    /// of the provided `data` buffer.
    /// ```
    pub fn new(data: MemoryBuffer<C>, dimensions: [usize; D]) -> Self {
        let total_elements: usize = dimensions.iter().product();
        assert_eq!(data.len(), total_elements,
            "Data buffer length ({}) must match total elements implied by dimensions ({})",
            data.len(), total_elements);

        let strides = Self::calculate_strides(&dimensions);
        Tensor {
            data,
            dimensions,
            strides,
        }
    }

    /// Creates a new `Tensor` from a flat `Vec` and its dimensions.
    ///
    /// This is a convenience constructor that automatically converts the `Vec`
    /// into a `MemoryBuffer` and then calls the `new` constructor.
    ///
    /// # Panics
    /// Panics if the total number of elements implied by `dimensions`
    /// (product of all dimension sizes) does not match the length of the `data_vec`.
    ///
    /// # Examples
    /// ```
    /// let tensor_from_vec = Tensor::from_vec(vec![10, 20, 30, 40], [2, 2]);
    /// assert_eq!(tensor_from_vec.dimensions, [2, 2]);
    /// assert_eq!(tensor_from_vec.strides, [2, 1]);
    /// assert_eq!(tensor_from_vec.data.as_slice(), &[10, 20, 30, 40]);
    /// ```
    pub fn from_slice(slice: &[C], dimensions: [usize; D]) -> Self {
        let data = MemoryBuffer::from_slice(64, slice);
        Self::new(data, dimensions)
    }

    /// Creates a new `Tensor` from a slice of data, explicit dimensions, and strides.
    ///
    /// This constructor allows for creating tensors with custom stride patterns,
    /// which can be useful for representing views or sub-tensors of larger data
    /// structures without copying the underlying data.
    ///
    /// # Panics
    /// Panics if:
    /// - The `dimensions` and `strides` arrays do not have the same number of elements as `D`.
    /// - The total number of elements implied by `dimensions` and `strides` (i.e., the
    ///   maximum flat index + 1) exceeds the length of the `slice`.
    /// - Any stride is zero unless its corresponding dimension is also zero.
    ///
    /// # Arguments
    /// * `slice` - The underlying data slice.
    /// * `dimensions` - An array of `usize` defining the size of each dimension.
    /// * `strides` - An array of `usize` defining the stride for each dimension.
    ///
    /// # Examples
    /// ```
    /// // Create a 2x3 tensor from a slice with custom strides
    /// let data = vec![1, 2, 3, 4, 5, 6];
    /// let tensor = Tensor::from_slice_with_strides(
    ///     &data,
    ///     [2, 3], // 2 rows, 3 columns
    ///     [3, 1], // Stride for rows is 3 elements, for columns is 1 element
    /// );
    /// assert_eq!(tensor.dimensions(), &[2, 3]);
    /// assert_eq!(tensor.strides(), &[3, 1]);
    /// assert_eq!(tensor.at([0, 0]), &1);
    /// assert_eq!(tensor.at([0, 1]), &2);
    /// assert_eq!(tensor.at([1, 0]), &4);
    ///
    /// // Creating a column vector view from a larger matrix's data
    /// let matrix_data = vec![1, 2, 3, 4, 5, 6, 7, 8, 9]; // A 3x3 matrix's data
    /// // View the second column (elements 2, 5, 8) as a 3x1 tensor
    /// let column_view = Tensor::from_slice_with_strides(
    ///     &matrix_data,
    ///     [3, 1], // 3 rows, 1 column
    ///     [3, 1], // Stride to next row is 3, stride to next column is 1 (but only 1 column)
    /// );
    /// assert_eq!(column_view.dimensions(), &[3, 1]);
    /// assert_eq!(column_view.strides(), &[3, 1]);
    /// // Note: This example is slightly misleading as the slice itself doesn't change for the column.
    /// // A more accurate example for strides would involve a sub-view that skips elements.
    /// // This specific case would be more typical for `TensorRef`.
    /// ```
    pub fn from_slice_with_strides(slice: &[C], dimensions: [usize; D], strides: [usize; D]) -> Self {
        // Validate that dimensions and strides match the tensor's dimensionality.
        // This is implicitly handled by the const generic D.

        // Calculate the maximum flat index that can be accessed with the given dimensions and strides.
        // This determines the minimum required length of the underlying slice.
        let mut max_flat_index = 0;
        for i in 0..D {
            if dimensions[i] > 0 {
                // For each dimension, the maximum index reached is (dimension_size - 1).
                // Multiply by the stride to get the offset for that dimension.
                max_flat_index += (dimensions[i] - 1) * strides[i];
            } else {
                // If a dimension is zero, its contribution to the max_flat_index is zero.
                // However, if a dimension is zero, its stride should ideally also be zero or handled carefully.
                // We'll panic if stride is non-zero for a zero dimension, as it's likely an error.
                if strides[i] != 0 {
                    panic!("Stride for dimension {} is non-zero ({}) but dimension size is zero.", i, strides[i]);
                }
            }
        }

        // The required length of the slice is max_flat_index + 1 (because indices are 0-based).
        let required_len = if D == 0 { 0 } else { max_flat_index + 1 };

        // Ensure the slice is large enough to contain all accessed elements.
        assert!(slice.len() >= required_len,
            "Input slice length ({}) is too small for the given dimensions and strides. Minimum required length: {}",
            slice.len(), required_len);

        // Ensure that if a dimension is non-zero, its stride is also non-zero.
        // A zero stride for a non-zero dimension means no progress is made along that dimension,
        // which might be an error or indicate a degenerate tensor.
        for i in 0..D {
            if dimensions[i] > 0 && strides[i] == 0 {
                panic!("Stride for non-zero dimension {} cannot be zero.", i);
            }
        }

        let data = MemoryBuffer::from_slice(64, slice);
        Tensor {
            data,
            dimensions,
            strides,
        }
    }

    /// Creates a new tensor with all elements initialized to zero,
    /// with specified shape.
    /// 
    /// # Arguments
    /// 
    /// * `dims` - A slice of `usize` containing the size of each dimension.
    /// 
    /// # Returns
    /// 
    /// * An new tensor with specified shape, filled with zeros.
    /// 
    /// # Panics
    /// 
    /// * If the length of `dims` is not equal to the number of
    ///     dimensions of the tensor.
    /// 
    /// # Examples
    /// ```
    /// use qudit_core::array::Tensor;
    /// 
    /// let test_tensor = Tensor::<f64, 2>::zeros(&[3, 4]);
    /// 
    /// for i in 0..3 {
    ///     for j in 0..4 {
    ///         assert_eq!(test_tensor.data[(4*i + j) as usize], 0.0);
    ///     }
    /// }
    /// ```
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

    /// Creates a new tensor with all elements initialized to zero,
    /// with specified shape and strides.
    /// 
    /// # Arguments
    /// 
    /// * `dims` - A slice of `usize` containing the size of each dimension
    /// * `strides` - A slice of `usize` containing the stride for each dimension.
    /// 
    /// # Returns
    /// 
    /// * A new tensor with specified shape and strides, filled with zeros.
    /// 
    /// # Panics
    /// 
    /// * If the length of `dims` or `strides` is not equal to the number of
    ///     dimensions of the tensor.
    /// * If the size of any dimension is zero but the corresponding stride is non-zero.
    /// * If the size of any dimension is non-zero but the corresponding stride is zero.
    ///
    /// # Examples
    /// ```
    /// use qudit_core::array::Tensor;
    /// 
    /// let test_tensor = Tensor::<f64, 2>::zeros_with_strides(&[3, 4], &[4, 1]);
    ///
    /// for i in 0..3 {
    ///     for j in 0..4 {
    ///         assert_eq!(test_tensor.at([i, j]), &0.0);
    ///     }
    /// }
    /// ```
    pub fn zeros_with_strides(dims: &[usize], strides: &[usize]) -> Self {
        assert_eq!(dims.len(), D);
        assert_eq!(strides.len(), D);

        // Handle the 0-dimensional tensor case (scalar)
        if D == 0 {
            // A 0-dimensional tensor (scalar) conceptually holds one element, even if its dimensions array is empty.
            // Allocate memory for this single element.
            return Self {
                data: alloc_zeroed_memory::<C>(1),
                dimensions: [0; D], // For D=0, this is an empty array `[]`
                strides: [0; D],    // For D=0, this is an empty array `[]`
            };
        }

        let mut max_flat_index = 0;
        for i in 0..D {
            if dims[i] > 0 {
                // For each dimension, the maximum index reached is (dimension_size - 1).
                // Multiply by the stride to get the offset for that dimension.
                max_flat_index += (dims[i] - 1) * strides[i];
            } else {
                // If a dimension is zero, its contribution to the max_flat_index is zero.
                // If the dimension is zero but its stride is non-zero, it indicates a likely error.
                if strides[i] != 0 {
                    panic!("Stride for dimension {} is non-zero ({}) but dimension size is zero.", i, strides[i]);
                }
            }
        }

        // The required length of the underlying data buffer is max_flat_index + 1,
        // as indices are 0-based.
        let required_len = max_flat_index + 1;

        // Ensure that if a dimension is non-zero, its stride is also non-zero.
        // A zero stride for a non-zero dimension implies no progress along that dimension,
        // which could lead to unexpected behavior or degenerate tensor access.
        for i in 0..D {
            if dims[i] > 0 && strides[i] == 0 {
                panic!("Stride for non-zero dimension {} cannot be zero.", i);
            }
        }

        let data = alloc_zeroed_memory::<C>(required_len);

        Self {
            data,
            dimensions: dims.try_into().unwrap(), // Convert slice to array, safe due to assert_eq!
            strides: strides.try_into().unwrap(), // Convert slice to array, safe due to assert_eq!
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

/// A view struct of a tensor. It holds a reference to the underlying data.
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
