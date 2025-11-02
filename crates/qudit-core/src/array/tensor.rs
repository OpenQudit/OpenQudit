//! Implements the tensor struct and associated methods for the Openqudit library.

use std::fmt::{self, Debug, Display, Formatter};
use std::ptr::NonNull;
use faer::{MatMut, MatRef, RowMut, RowRef};

use crate::memory::{alloc_zeroed_memory, Memorable, MemoryBuffer};
use super::check_bounds;

/// Helper for flat index calculation from multi-dimensional indices.
#[inline(always)]
fn calculate_flat_index<const D: usize>(indices: &[usize; D], strides: &[usize; D]) -> usize {
    let mut flat_idx = 0;
    for i in 0..D {
        flat_idx += indices[i] * strides[i];
    }
    flat_idx
}

/// A tensor struct that holds data in an aligned memory buffer.
pub struct Tensor<C: Memorable, const D: usize> {
    /// The data buffer containing the tensor elements.
    data: MemoryBuffer<C>,
    /// The dimensions of the tensor (size of each axis).
    dims: [usize; D],
    /// The strides for each dimension.
    strides: [usize; D],
} 

impl<C: Memorable, const D: usize> Tensor<C, D> {
    /// Creates a new tensor from a memory buffer with specified dimensions and strides.
    /// 
    /// # Arguments
    /// 
    /// * `data` - The memory buffer containing the tensor data
    /// * `dims` - Array specifying the size of each dimension
    /// * `strides` - Array specifying the stride for each dimension
    /// 
    /// # Panics
    /// 
    /// * If any dimension or stride is zero
    /// * If the data buffer is not large enough for the specified dimensions and strides
    pub fn new(data: MemoryBuffer<C>, dims: [usize; D], strides: [usize; D]) -> Self {
        assert!(dims.iter().all(|&d| d != 0), "Cannot have a zero-length dimension.");
        assert!(strides.iter().all(|&d| d != 0), "Cannot have a zero-length stride.");

        let mut max_element = [0; D];
        for (i, d) in dims.iter().enumerate() {
            max_element[i] = d - 1;
        }
        let max_flat_index = calculate_flat_index(&max_element, &strides);

        assert!(data.len() >= max_flat_index, "Data buffer is not large enough.");

        Self {
            data,
            dims,
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
    /// # use qudit_core::array::Tensor;
    /// 
    /// let test_tensor = Tensor::<f64, 2>::zeros([3, 4]);
    /// 
    /// for i in 0..3 {
    ///     for j in 0..4 {
    ///         assert_eq!(test_tensor.get(&[i, j]), &0.0);
    ///     }
    /// }
    /// ```
    pub fn zeros(dims: [usize; D]) -> Self {
        let strides = super::calc_continuous_strides(&dims);
        let data = alloc_zeroed_memory::<C>(strides[0] * dims[0]);
        Self::new(data, dims, strides)
    }

    /// Returns a reference to the dimensions of the tensor.
    pub fn dims(&self) -> &[usize; D] {
        &self.dims
    }

    /// Returns a reference to the strides of the tensor.
    pub fn strides(&self) -> &[usize; D] {
        &self.strides
    }

    /// Returns the rank (number of dimensions) of the tensor.
    pub fn rank(&self) -> usize {
        D
    }

    /// Returns the total number of elements in the tensor.
    pub fn num_elements(&self) -> usize {
        self.dims.iter().product()
    }

    /// Returns a raw pointer to the tensor's data.
    pub fn as_ptr(&self) -> *const C {
        self.data.as_ptr()
    }

    /// Returns a mutable raw pointer to the tensor's data.
    pub fn as_ptr_mut(&mut self) -> *mut C {
        self.data.as_mut_ptr()
    }

    /// Returns an immutable reference to the tensor.
    pub fn as_ref(&self) -> TensorRef<'_, C, D> {
        unsafe {
            TensorRef::from_raw_parts(self.data.as_ptr(), self.dims, self.strides)
        }
    }

    /// Returns a mutable reference to the tensor.
    pub fn as_mut(&mut self) -> TensorMut<'_, C, D> {
        unsafe {
            TensorMut::from_raw_parts(self.data.as_mut_ptr(), self.dims, self.strides)
        }
    }

    /// Returns a reference to an element at the given indices.
    ///
    /// # Panics
    ///
    /// Panics if the indices are out of bounds.
    pub fn get(&self, indices: &[usize; D]) -> &C {
        check_bounds(indices, &self.dims);
        // Safety: bounds are checked by `check_bounds`
        unsafe { self.get_unchecked(indices) }
    }

    /// Returns a mutable reference to an element at the given indices.
    ///
    /// # Panics
    ///
    /// Panics if the indices are out of bounds.
    pub fn get_mut(&mut self, indices: &[usize; D]) -> &mut C {
        check_bounds(indices, &self.dims);
        // Safety: bounds are checked by `check_bounds`
        unsafe { self.get_mut_unchecked(indices) }
    }

    /// Returns an immutable reference to an element at the given indices, without performing bounds checks.
    ///
    /// # Safety
    ///
    /// Calling this method with out-of-bounds `indices` is undefined behavior.
    #[inline(always)]
    pub unsafe fn get_unchecked(&self, indices: &[usize; D]) -> &C {
        &*self.ptr_at(indices)
    }

    /// Returns a mutable reference to an element at the given indices, without performing bounds checks.
    ///
    /// # Safety
    ///
    /// Calling this method with out-of-bounds `indices` is undefined behavior.
    #[inline(always)]
    pub unsafe fn get_mut_unchecked(&mut self, indices: &[usize; D]) -> &mut C {
        &mut *self.ptr_at_mut(indices)
    }

    /// Returns a raw pointer to an element at the given indices, without performing bounds checks.
    ///
    /// # Safety
    ///
    /// Calling this method with out-of-bounds `indices` is undefined behavior.
    #[inline(always)]
    pub unsafe fn ptr_at(&self, indices: &[usize; D]) -> *const C {
        let flat_idx = calculate_flat_index(indices, &self.strides);
        self.as_ptr().add(flat_idx)
    }

    /// Returns a mutable raw pointer to an element at the given indices, without performing bounds checks.
    ///
    /// # Safety
    ///
    /// Calling this method with out-of-bounds `indices` is undefined behavior.
    #[inline(always)]
    pub unsafe fn ptr_at_mut(&mut self, indices: &[usize; D]) -> *mut C {
        let flat_idx = calculate_flat_index(indices, &self.strides);
        self.as_ptr_mut().add(flat_idx)
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
    /// # use qudit_core::array::Tensor;
    /// let tensor_from_slice = Tensor::from_slice(&vec![10, 20, 30, 40], [2, 2]);
    /// assert_eq!(tensor_from_slice.dims(), &[2, 2]);
    /// assert_eq!(tensor_from_slice.strides(), &[2, 1]);
    /// ```
    pub fn from_slice(slice: &[C], dims: [usize; D]) -> Self {
        let strides = super::calc_continuous_strides(&dims);
        Self::from_slice_with_strides(slice, dims, strides)
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
    /// # use qudit_core::array::Tensor;
    /// // Create a 2x3 tensor from a slice with custom strides
    /// let data = vec![1, 2, 3, 4, 5, 6];
    /// let tensor = Tensor::from_slice_with_strides(
    ///     &data,
    ///     [2, 3], // 2 rows, 3 columns
    ///     [3, 1], // Stride for rows is 3 elements, for columns is 1 element
    /// );
    /// assert_eq!(tensor.dims(), &[2, 3]);
    /// assert_eq!(tensor.strides(), &[3, 1]);
    /// assert_eq!(tensor.get(&[0, 0]), &1);
    /// assert_eq!(tensor.get(&[0, 1]), &2);
    /// assert_eq!(tensor.get(&[1, 0]), &4);
    ///
    /// // Creating a column vector view from a larger matrix's data
    /// let matrix_data = vec![1, 2, 3, 4, 5, 6, 7, 8, 9]; // A 3x3 matrix's data
    /// // View the second column (elements 2, 5, 8) as a 3x1 tensor
    /// let column_view = Tensor::from_slice_with_strides(
    ///     &matrix_data,
    ///     [3, 1], // 3 rows, 1 column
    ///     [3, 1], // Stride to next row is 3, stride to next column is 1 (but only 1 column)
    /// );
    /// assert_eq!(column_view.dims(), &[3, 1]);
    /// assert_eq!(column_view.strides(), &[3, 1]);
    /// // Note: This example is slightly misleading as the slice itself doesn't change for the column.
    /// // A more accurate example for strides would involve a sub-view that skips elements.
    /// // This specific case would be more typical for `TensorRef`.
    /// ```
    pub fn from_slice_with_strides(slice: &[C], dims: [usize; D], strides: [usize; D]) -> Self {
        let data = MemoryBuffer::from_slice(64, slice);
        Self::new(data, dims, strides)
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
    /// # use qudit_core::array::Tensor;
    /// 
    /// let test_tensor = Tensor::<f64, 2>::zeros_with_strides(&[3, 4], &[4, 1]);
    ///
    /// for i in 0..3 {
    ///     for j in 0..4 {
    ///         assert_eq!(test_tensor.get(&[i, j]), &0.0);
    ///     }
    /// }
    /// ```
    pub fn zeros_with_strides(dims: &[usize; D], strides: &[usize; D]) -> Self {
        let data = alloc_zeroed_memory::<C>(strides[0] * dims[0]);
        Self::new(data, dims.clone(), strides.clone())
    }
}

impl<C: Memorable, const D: usize> std::ops::Index<[usize; D]> for Tensor<C, D> {
    type Output = C;

    fn index(&self, indices: [usize; D]) -> &Self::Output {
        self.get(&indices)
    }
}

impl<C: Memorable, const D: usize> std::ops::IndexMut<[usize; D]> for Tensor<C, D> {
    fn index_mut(&mut self, indices: [usize; D]) -> &mut Self::Output {
        self.get_mut(&indices)
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
                writeln!(f, "{}[", indent)?;
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
            .field("dimensions", &self.dims)
            .field("strides", &self.strides)
            .field("data", &TensorDataDebugHelper {
                data_ptr: self.data.as_ptr(), // Pointer to the start of the data buffer
                dimensions: &self.dims,
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
            .field("dimensions", &self.dims)
            .field("strides", &self.strides)
            .field("data", &TensorDataDebugHelper {
                data_ptr: self.data.as_ptr(), // `self.data` is already a `*const C` for `TensorRef`
                dimensions: &self.dims,
                strides: &self.strides,
                current_dim_idx: 0,
                current_flat_offset: 0,
            })
            .finish()
    }
}

// TODO: add iterators for subtensors

/// An immutable view into tensor data.
/// 
/// This struct provides read-only access to tensor data without owning the underlying memory.
/// It holds a reference to the data along with dimension and stride information.
#[derive(Clone, Copy)]
pub struct TensorRef<'a, C: Memorable, const D: usize> {
    /// Non-null pointer to the tensor data.
    data: NonNull<C>,
    /// The dimensions of the tensor (size of each axis).
    dims: [usize; D],
    /// The strides for each dimension.
    strides: [usize; D],
    /// Phantom data to enforce lifetime constraints.
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
        dims: [usize; D],
        strides: [usize; D],
    ) -> Self {
        // SAFETY: The pointer is never used in an mutable context.
        let ptr = unsafe { NonNull::new_unchecked(data as *mut C) };

        Self {
            data: ptr,
            dims,
            strides,
            __marker: std::marker::PhantomData,
        }
    }

    /// Returns a reference to the dimensions of the tensor.
    pub fn dims(&self) -> &[usize; D] {
        &self.dims
    }

    /// Returns a reference to the strides of the tensor.
    pub fn strides(&self) -> &[usize; D] {
        &self.strides
    }

    /// Returns the rank (number of dimensions) of the tensor.
    pub fn rank(&self) -> usize {
        D
    }

    /// Returns the total number of elements in the tensor.
    pub fn num_elements(&self) -> usize {
        self.dims.iter().product()
    }

    /// Returns a raw pointer to the tensor's data.
    pub fn as_ptr(&self) -> *const C {
        self.data.as_ptr()
    }

    /// Returns a reference to an element at the given indices.
    ///
    /// # Panics
    ///
    /// Panics if the indices are out of bounds.
    pub fn get(&self, indices: &[usize; D]) -> &C {
        check_bounds(indices, &self.dims);
        // Safety: bounds are checked by `check_bounds`
        unsafe { self.get_unchecked(indices) }
    }

    /// Returns an immutable reference to an element at the given indices, without performing bounds checks.
    ///
    /// # Safety
    ///
    /// Calling this method with out-of-bounds `indices` is undefined behavior.
    #[inline(always)]
    pub unsafe fn get_unchecked(&self, indices: &[usize; D]) -> &C {
        &*self.ptr_at(indices)
    }

    /// Returns a raw pointer to an element at the given indices, without performing bounds checks.
    ///
    /// # Safety
    ///
    /// Calling this method with out-of-bounds `indices` is undefined behavior.
    #[inline(always)]
    pub unsafe fn ptr_at(&self, indices: &[usize; D]) -> *const C {
        let flat_idx = calculate_flat_index(indices, &self.strides);
        self.as_ptr().add(flat_idx)
    }

    /// Creates an owned `Tensor` by copying the data from this `TensorRef`.
    pub fn to_owned(self) -> Tensor<C, D> {
        let mut max_element = [0; D];
        for (i, d) in self.dims.iter().enumerate() {
            max_element[i] = d - 1;
        }
        let max_flat_index = calculate_flat_index(&max_element, &self.strides);
        // Safety: Memory is nonnull and shared throughout max_flat_index.
        // Slice is copied from and dropped immediately.
        unsafe {
            let slice = std::slice::from_raw_parts(self.data.as_ptr(), max_flat_index + 1);
            Tensor::from_slice_with_strides(slice, self.dims, self.strides)
        }
    }
}

impl<'a, C: Memorable, const D: usize> std::ops::Index<[usize; D]> for TensorRef<'a, C, D> {
    type Output = C;

    fn index(&self, indices: [usize; D]) -> &Self::Output {
        self.get(&indices)
    }
}


/// A mutable view into tensor data.
/// 
/// This struct provides read-write access to tensor data without owning the underlying memory.
/// It holds a mutable reference to the data along with dimension and stride information.
pub struct TensorMut<'a, C: Memorable, const D: usize> {
    /// Non-null pointer to the tensor data.
    data: NonNull<C>,
    /// The dimensions of the tensor (size of each axis).
    dims: [usize; D],
    /// The strides for each dimension.
    strides: [usize; D],
    /// Phantom data to enforce lifetime constraints.
    __marker: std::marker::PhantomData<&'a mut C>,
}

impl<'a, C: Memorable, const D: usize> TensorMut<'a, C, D> {
    /// Creates a new `SymSqTensorMut` from raw parts.
    ///
    /// # Safety
    ///
    /// The caller must ensure that `data` points to a valid memory block of `C` elements,
    /// and that `dims` and `strides` accurately describe the layout of the tensor
    /// within that memory block. The `data` pointer must be valid for the lifetime `'a`
    /// and that it is safe to mutate the data.
    pub unsafe fn from_raw_parts(data: *mut C, dims: [usize; D], strides: [usize; D]) -> Self {
        Self {
            data: NonNull::new_unchecked(data),
            dims,
            strides,
            __marker: std::marker::PhantomData,
        }
    }

    /// Returns a reference to the dimensions of the tensor.
    pub fn dims(&self) -> &[usize; D] {
        &self.dims
    }

    /// Returns a reference to the strides of the tensor.
    pub fn strides(&self) -> &[usize; D] {
        &self.strides
    }

    /// Returns the rank (number of dimensions) of the tensor.
    pub fn rank(&self) -> usize {
        D
    }

    /// Returns the total number of elements in the tensor.
    pub fn num_elements(&self) -> usize {
        self.dims.iter().product()
    }

    /// Returns a mutable raw pointer to the tensor's data.
    pub fn as_ptr(&self) -> *const C {
        self.data.as_ptr() as *const C
    }

    /// Returns a mutable raw pointer to the tensor's data.
    pub fn as_ptr_mut(&mut self) -> *mut C {
        self.data.as_ptr()
    }

    /// Returns a reference to an element at the given indices.
    ///
    /// # Panics
    ///
    /// Panics if the indices are out of bounds.
    pub fn get(&self, indices: &[usize; D]) -> &C {
        check_bounds(indices, &self.dims);
        // Safety: bounds are checked by `check_bounds`
        unsafe { self.get_unchecked(indices) }
    }

    /// Returns a mutable reference to an element at the given indices.
    ///
    /// # Panics
    ///
    /// Panics if the indices are out of bounds.
    pub fn get_mut(&mut self, indices: &[usize; D]) -> &mut C {
        check_bounds(indices, &self.dims);
        // Safety: bounds are checked by `check_bounds`
        unsafe { self.get_mut_unchecked(indices) }
    }

    /// Returns an immutable reference to an element at the given indices, without performing bounds checks.
    ///
    /// # Safety
    ///
    /// Calling this method with out-of-bounds `indices` is undefined behavior.
    #[inline(always)]
    pub unsafe fn get_unchecked(&self, indices: &[usize; D]) -> &C {
        &*self.ptr_at(indices)
    }

    /// Returns a mutable reference to an element at the given indices, without performing bounds checks.
    ///
    /// # Safety
    ///
    /// Calling this method with out-of-bounds `indices` is undefined behavior.
    #[inline(always)]
    pub unsafe fn get_mut_unchecked(&mut self, indices: &[usize; D]) -> &mut C {
        &mut *self.ptr_at_mut(indices)
    }

    /// Returns a raw pointer to an element at the given indices, without performing bounds checks.
    ///
    /// # Safety
    ///
    /// Calling this method with out-of-bounds `indices` is undefined behavior.
    #[inline(always)]
    pub unsafe fn ptr_at(&self, indices: &[usize; D]) -> *const C {
        let flat_idx = calculate_flat_index(indices, &self.strides);
        self.as_ptr().add(flat_idx)
    }

    /// Returns a mutable raw pointer to an element at the given indices, without performing bounds checks.
    ///
    /// # Safety
    ///
    /// Calling this method with out-of-bounds `indices` is undefined behavior.
    #[inline(always)]
    pub unsafe fn ptr_at_mut(&mut self, indices: &[usize; D]) -> *mut C {
        let flat_idx = calculate_flat_index(indices, &self.strides);
        self.as_ptr_mut().add(flat_idx)
    }
}

impl<'a, C: Memorable, const D: usize> std::ops::Index<[usize; D]> for TensorMut<'a, C, D> {
    type Output = C;

    fn index(&self, indices: [usize; D]) -> &Self::Output {
        self.get(&indices)
    }
}

impl<'a, C: Memorable, const D: usize> std::ops::IndexMut<[usize; D]> for TensorMut<'a, C, D> {
    fn index_mut(&mut self, indices: [usize; D]) -> &mut Self::Output {
        self.get_mut(&indices)
    }
}

// TODO add some documentation plus a todo tag on relevant rust issues
impl<C: Memorable> Tensor<C, 4> {
    /// Returns an immutable view of a 3D subtensor at the given index.
    pub fn subtensor_ref(&self, m: usize) -> TensorRef<'_, C, 3> {
        check_bounds(&[m, 0, 0, 0], &self.dims);
        // Safety: bounds have been checked.
        unsafe { self.subtensor_ref_unchecked(m) }
    }

    /// Returns a mutable view of a 3D subtensor at the given index.
    pub fn subtensor_mut(&mut self, m: usize) -> TensorMut<'_, C, 3> {
        check_bounds(&[m, 0, 0, 0], &self.dims);
        // Safety: bounds have been checked.
        unsafe { self.subtensor_mut_unchecked(m) }
    }

    #[inline(always)]
    /// Returns an immutable view of a 3D subtensor at the given index without bounds checking.
    pub unsafe fn subtensor_ref_unchecked(&self, m: usize) -> TensorRef<'_, C, 3> {
        TensorRef::from_raw_parts(self.ptr_at(&[m, 0, 0, 0]), [self.dims[1], self.dims[2], self.dims[3]], [self.strides[1], self.strides[2], self.strides[3]])
    }

    #[inline(always)]
    /// Returns a mutable view of a 3D subtensor at the given index without bounds checking.
    pub unsafe fn subtensor_mut_unchecked(&mut self, m: usize) -> TensorMut<'_, C, 3> {
        TensorMut::from_raw_parts(self.ptr_at_mut(&[m, 0, 0, 0]), [self.dims[1], self.dims[2], self.dims[3]], [self.strides[1], self.strides[2], self.strides[3]])
    }
}

impl<C: Memorable> Tensor<C, 3> {
    /// Returns an immutable matrix view of a 2D subtensor at the given index.
    pub fn subtensor_ref(&self, m: usize) -> MatRef<'_, C> {
        check_bounds(&[m, 0, 0], &self.dims);
        // Safety: bounds have been checked.
        unsafe { self.subtensor_ref_unchecked(m) }
    }

    /// Returns a mutable matrix view of a 2D subtensor at the given index.
    pub fn subtensor_mut(&mut self, m: usize) -> MatMut<'_, C> {
        check_bounds(&[m, 0, 0], &self.dims);
        // Safety: bounds have been checked.
        unsafe { self.subtensor_mut_unchecked(m) }
    }

    #[inline(always)]
    /// Returns an immutable matrix view of a 2D subtensor at the given index without bounds checking.
    pub unsafe fn subtensor_ref_unchecked(&self, m: usize) -> MatRef<'_, C> {
        MatRef::from_raw_parts(self.ptr_at(&[m, 0, 0]), self.dims[1], self.dims[2], self.strides[1] as isize, self.strides[2] as isize)
    }

    #[inline(always)]
    /// Returns a mutable matrix view of a 2D subtensor at the given index without bounds checking.
    pub unsafe fn subtensor_mut_unchecked(&mut self, m: usize) -> MatMut<'_, C> {
        MatMut::from_raw_parts_mut(self.ptr_at_mut(&[m, 0, 0]), self.dims[1], self.dims[2], self.strides[1] as isize, self.strides[2] as isize)
    }
}

impl<C: Memorable> Tensor<C, 2> {
    /// Returns an immutable row view of a 1D subtensor at the given index.
    pub fn subtensor_ref(&self, m: usize) -> RowRef<'_, C> {
        check_bounds(&[m, 0], &self.dims);
        // Safety: bounds have been checked.
        unsafe { self.subtensor_ref_unchecked(m) }
    }

    /// Returns a mutable row view of a 1D subtensor at the given index.
    pub fn subtensor_mut(&mut self, m: usize) -> RowMut<'_, C> {
        check_bounds(&[m, 0], &self.dims);
        // Safety: bounds have been checked.
        unsafe { self.subtensor_mut_unchecked(m) }
    }

    #[inline(always)]
    /// Returns an immutable row view of a 1D subtensor at the given index without bounds checking.
    pub unsafe fn subtensor_ref_unchecked(&self, m: usize) -> RowRef<'_, C> {
        RowRef::from_raw_parts(self.ptr_at(&[m, 0]), self.dims[1], self.strides[1] as isize)
    }

    #[inline(always)]
    /// Returns a mutable row view of a 1D subtensor at the given index without bounds checking.
    pub unsafe fn subtensor_mut_unchecked(&mut self, m: usize) -> RowMut<'_, C> {
        RowMut::from_raw_parts_mut(self.ptr_at_mut(&[m, 0]), self.dims[1], self.strides[1] as isize)
    }
}

impl<C: Memorable> Tensor<C, 1> {
    /// Returns an immutable reference to an element at the given index.
    pub fn subtensor_ref(&self, m: usize) -> &C {
        self.get(&[m])
    }

    /// Returns a mutable reference to an element at the given index.
    pub fn subtensor_mut(&mut self, m: usize) -> &mut C {
        self.get_mut(&[m])
    }

    #[inline(always)]
    /// Returns an immutable reference to an element at the given index without bounds checking.
    pub unsafe fn subtensor_ref_unchecked(&self, m: usize) -> &C {
        self.get_unchecked(&[m])
    }

    #[inline(always)]
    /// Returns a mutable reference to an element at the given index without bounds checking.
    pub unsafe fn subtensor_mut_unchecked(&mut self, m: usize) -> &mut C {
        self.get_mut_unchecked(&[m])
    }
}

impl<'a, C: Memorable> TensorRef<'a, C, 4> {
    /// Returns an immutable view of a 3D subtensor at the given index.
    pub fn subtensor_ref(&self, m: usize) -> TensorRef<'a, C, 3> {
        check_bounds(&[m, 0, 0, 0], &self.dims);
        // Safety: bounds have been checked.
        unsafe { self.subtensor_ref_unchecked(m) }
    }

    #[inline(always)]
    /// Returns an immutable view of a 3D subtensor at the given index without bounds checking.
    pub unsafe fn subtensor_ref_unchecked(&self, m: usize) -> TensorRef<'a, C, 3> {
        TensorRef::from_raw_parts(self.ptr_at(&[m, 0, 0, 0]), [self.dims[1], self.dims[2], self.dims[3]], [self.strides[1], self.strides[2], self.strides[3]])
    }
}

impl<'a, C: Memorable> TensorRef<'a, C, 3> {
    /// Returns an immutable matrix view of a 2D subtensor at the given index.
    pub fn subtensor_ref(&self, m: usize) -> MatRef<'a, C> {
        check_bounds(&[m, 0, 0], &self.dims);
        // Safety: bounds have been checked.
        unsafe { self.subtensor_ref_unchecked(m) }
    }

    #[inline(always)]
    /// Returns an immutable matrix view of a 2D subtensor at the given index without bounds checking.
    pub unsafe fn subtensor_ref_unchecked(&self, m: usize) -> MatRef<'a, C> {
        MatRef::from_raw_parts(self.ptr_at(&[m, 0, 0]), self.dims[1], self.dims[2], self.strides[1] as isize, self.strides[2] as isize)
    }
}

impl<'a, C: Memorable> TensorRef<'a, C, 2> {
    /// Returns an immutable row view of a 1D subtensor at the given index.
    pub fn subtensor_ref(&self, m: usize) -> RowRef<'a, C> {
        check_bounds(&[m, 0], &self.dims);
        // Safety: bounds have been checked.
        unsafe { self.subtensor_ref_unchecked(m) }
    }

    #[inline(always)]
    /// Returns an immutable row view of a 1D subtensor at the given index without bounds checking.
    pub unsafe fn subtensor_ref_unchecked(&self, m: usize) -> RowRef<'a, C> {
        RowRef::from_raw_parts(self.ptr_at(&[m, 0]), self.dims[1], self.strides[1] as isize)
    }
}

impl<'a, C: Memorable> TensorRef<'a, C, 1> {
    /// Returns an immutable reference to an element at the given index.
    pub fn subtensor_ref(&self, m: usize) -> &C {
        self.get(&[m])
    }

    #[inline(always)]
    /// Returns an immutable reference to an element at the given index without bounds checking.
    pub unsafe fn subtensor_ref_unchecked(&self, m: usize) -> &C {
        self.get_unchecked(&[m])
    }
}

impl<'a, C: Memorable> TensorMut<'a, C, 4> {
    /// Returns an immutable view of a 3D subtensor at the given index.
    pub fn subtensor_ref(&self, m: usize) -> TensorRef<'a, C, 3> {
        check_bounds(&[m, 0, 0, 0], &self.dims);
        // Safety: bounds have been checked.
        unsafe { self.subtensor_ref_unchecked(m) }
    }

    /// Returns a mutable view of a 3D subtensor at the given index.
    pub fn subtensor_mut(&mut self, m: usize) -> TensorMut<'a, C, 3> {
        check_bounds(&[m, 0, 0, 0], &self.dims);
        // Safety: bounds have been checked.
        unsafe { self.subtensor_mut_unchecked(m) }
    }

    #[inline(always)]
    /// Returns an immutable view of a 3D subtensor at the given index without bounds checking.
    pub unsafe fn subtensor_ref_unchecked(&self, m: usize) -> TensorRef<'a, C, 3> {
        TensorRef::from_raw_parts(self.ptr_at(&[m, 0, 0, 0]), [self.dims[1], self.dims[2], self.dims[3]], [self.strides[1], self.strides[2], self.strides[3]])
    }

    #[inline(always)]
    /// Returns a mutable view of a 3D subtensor at the given index without bounds checking.
    pub unsafe fn subtensor_mut_unchecked(&mut self, m: usize) -> TensorMut<'a, C, 3> {
        TensorMut::from_raw_parts(self.ptr_at_mut(&[m, 0, 0, 0]), [self.dims[1], self.dims[2], self.dims[3]], [self.strides[1], self.strides[2], self.strides[3]])
    }
}

impl<'a, C: Memorable> TensorMut<'a, C, 3> {
    /// Returns an immutable matrix view of a 2D subtensor at the given index.
    pub fn subtensor_ref(&self, m: usize) -> MatRef<'a, C> {
        check_bounds(&[m, 0, 0], &self.dims);
        // Safety: bounds have been checked.
        unsafe { self.subtensor_ref_unchecked(m) }
    }

    /// Returns a mutable matrix view of a 2D subtensor at the given index.
    pub fn subtensor_mut(&mut self, m: usize) -> MatMut<'a, C> {
        check_bounds(&[m, 0, 0], &self.dims);
        // Safety: bounds have been checked.
        unsafe { self.subtensor_mut_unchecked(m) }
    }

    #[inline(always)]
    /// Returns an immutable matrix view of a 2D subtensor at the given index without bounds checking.
    pub unsafe fn subtensor_ref_unchecked(&self, m: usize) -> MatRef<'a, C> {
        MatRef::from_raw_parts(self.ptr_at(&[m, 0, 0]), self.dims[1], self.dims[2], self.strides[1] as isize, self.strides[2] as isize)
    }

    #[inline(always)]
    /// Returns a mutable matrix view of a 2D subtensor at the given index without bounds checking.
    pub unsafe fn subtensor_mut_unchecked(&mut self, m: usize) -> MatMut<'a, C> {
        MatMut::from_raw_parts_mut(self.ptr_at_mut(&[m, 0, 0]), self.dims[1], self.dims[2], self.strides[1] as isize, self.strides[2] as isize)
    }
}

impl<'a, C: Memorable> TensorMut<'a, C, 2> {
    /// Returns an immutable row view of a 1D subtensor at the given index.
    pub fn subtensor_ref(&self, m: usize) -> RowRef<'a, C> {
        check_bounds(&[m, 0], &self.dims);
        // Safety: bounds have been checked.
        unsafe { self.subtensor_ref_unchecked(m) }
    }

    /// Returns a mutable row view of a 1D subtensor at the given index.
    pub fn subtensor_mut(&mut self, m: usize) -> RowMut<'a, C> {
        check_bounds(&[m, 0], &self.dims);
        // Safety: bounds have been checked.
        unsafe { self.subtensor_mut_unchecked(m) }
    }

    #[inline(always)]
    /// Returns an immutable row view of a 1D subtensor at the given index without bounds checking.
    pub unsafe fn subtensor_ref_unchecked(&self, m: usize) -> RowRef<'a, C> {
        RowRef::from_raw_parts(self.ptr_at(&[m, 0]), self.dims[1], self.strides[1] as isize)
    }

    #[inline(always)]
    /// Returns a mutable row view of a 1D subtensor at the given index without bounds checking.
    pub unsafe fn subtensor_mut_unchecked(&mut self, m: usize) -> RowMut<'a, C> {
        RowMut::from_raw_parts_mut(self.ptr_at_mut(&[m, 0]), self.dims[1], self.strides[1] as isize)
    }
}

impl<'a, C: Memorable> TensorMut<'a, C, 1> {
    /// Returns an immutable reference to an element at the given index.
    pub fn subtensor_ref(&self, m: usize) -> &C {
        self.get(&[m])
    }

    /// Returns a mutable reference to an element at the given index.
    pub fn subtensor_mut(&mut self, m: usize) -> &mut C {
        self.get_mut(&[m])
    }

    #[inline(always)]
    /// Returns an immutable reference to an element at the given index without bounds checking.
    pub unsafe fn subtensor_ref_unchecked(&self, m: usize) -> &C {
        self.get_unchecked(&[m])
    }

    #[inline(always)]
    /// Returns a mutable reference to an element at the given index without bounds checking.
    pub unsafe fn subtensor_mut_unchecked(&mut self, m: usize) -> &mut C {
        self.get_mut_unchecked(&[m])
    }
}
