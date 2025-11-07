use std::ptr::NonNull;
// TODO: update faer imports to crate imports
// TODO: Make helper methods for debug and display that extracts shared functionality from Tensor
// TODO: add basic derives for clone, PartialEq, Debug and Display
// TODO: Use strong typing where it makes sense
// TODO: Add helpful, useful, succinct documentation with examples.
use super::check_bounds;
use faer::{MatMut, MatRef, RowMut, RowRef};

use crate::{
    array::TensorMut,
    array::TensorRef,
    memory::{Memorable, MemoryBuffer, alloc_zeroed_memory},
};

/// Convert SymSqMatMat external indexing to internal indexing.
///
/// See [index_to_coords] for more information.
///
/// When storing the upper triangular part of a matrix (including the
/// diagonal) into a compact vector, you essentially flatten the
/// upper triangular part of the matrix column-wise into a one-dimensional
/// array. Let's say you have an N*N matrix and a compact vector V of
/// length N(N+1)/2 to store the upper triangular part of the matrix.
/// For a matrix coordinate (i,j) in the upper triangular part
/// where i<=j, the corresponding vector index k can be calculated
/// using the formula:
///
/// ```math
///     k = j * (j+1) / 2 + i
/// ```
#[inline(always)]
fn coords_to_index(i: usize, j: usize) -> usize {
    if i <= j {
        j * (j + 1) / 2 + i
    } else {
        i * (i + 1) / 2 + j
    }
}

#[inline(always)]
fn calculate_flat_index<const D: usize>(indices: &[usize; D], strides: &[usize; D]) -> usize {
    let mut flat_idx = coords_to_index(indices[0], indices[1]) * strides[1];
    for i in 2..D {
        flat_idx += indices[i] * strides[i];
    }
    flat_idx
}

// Schwarz's Theorem is satisfied for quantum tensor networks
//
// TODO: when const generics can appear in const expressions, this can be rewritten better
/// A tensor with D dimensions, where the first two dimensions are equal and symmetric.
pub struct SymSqTensor<C: Memorable, const D: usize> {
    data: MemoryBuffer<C>,
    dims: [usize; D],
    strides: [usize; D],
}

impl<C: Memorable, const D: usize> SymSqTensor<C, D> {
    /// Creates a new symmetric square tensor with the given data, dimensions, and strides.
    pub fn new(data: MemoryBuffer<C>, dims: [usize; D], strides: [usize; D]) -> Self {
        assert!(
            D >= 2,
            "Symmetric square tensors must have 2 or more dimensions."
        );
        assert!(
            dims[0] == dims[1],
            "Symmetric square tensors must be square in their two major dimensions."
        );
        assert!(
            strides[0] == strides[1] * dims[1],
            "Symmetric square tensors must be continuous across their two major dimensions."
        );
        assert!(
            dims.iter().all(|&d| d != 0),
            "Cannot have a zero-length dimension."
        );
        assert!(
            strides.iter().all(|&d| d != 0),
            "Cannot have a zero-length stride."
        );

        let mut max_element = [0; D];
        for (i, d) in dims.iter().enumerate() {
            max_element[i] = d - 1;
        }
        let max_flat_index = calculate_flat_index(&max_element, &strides);

        assert!(
            data.len() >= max_flat_index,
            "Data buffer is not large enough."
        );

        Self {
            data,
            dims,
            strides,
        }
    }

    /// Creates a new symmetric square tensor filled with zeros.
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
    pub fn as_ref(&self) -> SymSqTensorRef<'_, C, D> {
        unsafe { SymSqTensorRef::from_raw_parts(self.data.as_ptr(), self.dims, self.strides) }
    }

    /// Returns a mutable reference to the tensor.
    pub fn as_mut(&mut self) -> SymSqTensorMut<'_, C, D> {
        unsafe { SymSqTensorMut::from_raw_parts(self.data.as_mut_ptr(), self.dims, self.strides) }
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
        unsafe { &*self.ptr_at(indices) }
    }

    /// Returns a mutable reference to an element at the given indices, without performing bounds checks.
    ///
    /// # Safety
    ///
    /// Calling this method with out-of-bounds `indices` is undefined behavior.
    #[inline(always)]
    pub unsafe fn get_mut_unchecked(&mut self, indices: &[usize; D]) -> &mut C {
        unsafe { &mut *self.ptr_at_mut(indices) }
    }

    /// Returns a raw pointer to an element at the given indices, without performing bounds checks.
    ///
    /// # Safety
    ///
    /// Calling this method with out-of-bounds `indices` is undefined behavior.
    #[inline(always)]
    pub unsafe fn ptr_at(&self, indices: &[usize; D]) -> *const C {
        unsafe {
            let flat_idx = calculate_flat_index(indices, &self.strides);
            self.as_ptr().add(flat_idx)
        }
    }

    /// Returns a mutable raw pointer to an element at the given indices, without performing bounds checks.
    ///
    /// # Safety
    ///
    /// Calling this method with out-of-bounds `indices` is undefined behavior.
    #[inline(always)]
    pub unsafe fn ptr_at_mut(&mut self, indices: &[usize; D]) -> *mut C {
        unsafe {
            let flat_idx = calculate_flat_index(indices, &self.strides);
            self.as_ptr_mut().add(flat_idx)
        }
    }
}

impl<C: Memorable, const D: usize> std::ops::Index<[usize; D]> for SymSqTensor<C, D> {
    type Output = C;

    fn index(&self, indices: [usize; D]) -> &Self::Output {
        self.get(&indices)
    }
}

impl<C: Memorable, const D: usize> std::ops::IndexMut<[usize; D]> for SymSqTensor<C, D> {
    fn index_mut(&mut self, indices: [usize; D]) -> &mut Self::Output {
        self.get_mut(&indices)
    }
}

#[derive(Clone, Copy)]
/// An immutable reference to a symmetric square tensor.
pub struct SymSqTensorRef<'a, C: Memorable, const D: usize> {
    data: NonNull<C>,
    dims: [usize; D],
    strides: [usize; D],
    __marker: std::marker::PhantomData<&'a C>,
}

impl<'a, C: Memorable, const D: usize> SymSqTensorRef<'a, C, D> {
    /// Creates a new `SymSqTensorRef` from raw parts.
    ///
    /// # Safety
    ///
    /// The caller must ensure that `data` points to a valid memory block of `C` elements,
    /// and that `dims` and `strides` accurately describe the layout of the tensor
    /// within that memory block. The `data` pointer must be valid for the lifetime `'a`.
    pub unsafe fn from_raw_parts(data: *const C, dims: [usize; D], strides: [usize; D]) -> Self {
        unsafe {
            // SAFETY: The pointer is never used in an mutable context.
            let mut_ptr = data as *mut C;

            Self {
                data: NonNull::new_unchecked(mut_ptr),
                dims,
                strides,
                __marker: std::marker::PhantomData,
            }
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
        unsafe { &*self.ptr_at(indices) }
    }

    /// Returns a raw pointer to an element at the given indices, without performing bounds checks.
    ///
    /// # Safety
    ///
    /// Calling this method with out-of-bounds `indices` is undefined behavior.
    #[inline(always)]
    pub unsafe fn ptr_at(&self, indices: &[usize; D]) -> *const C {
        unsafe {
            let flat_idx = calculate_flat_index(indices, &self.strides);
            self.as_ptr().add(flat_idx)
        }
    }
}

impl<'a, C: Memorable, const D: usize> std::ops::Index<[usize; D]> for SymSqTensorRef<'a, C, D> {
    type Output = C;

    fn index(&self, indices: [usize; D]) -> &Self::Output {
        self.get(&indices)
    }
}

/// A mutable reference to a symmetric square tensor.
pub struct SymSqTensorMut<'a, C: Memorable, const D: usize> {
    data: NonNull<C>,
    dims: [usize; D],
    strides: [usize; D],
    __marker: std::marker::PhantomData<&'a mut C>,
}

impl<'a, C: Memorable, const D: usize> SymSqTensorMut<'a, C, D> {
    /// Creates a new `SymSqTensorMut` from raw parts.
    ///
    /// # Safety
    ///
    /// The caller must ensure that `data` points to a valid memory block of `C` elements,
    /// and that `dims` and `strides` accurately describe the layout of the tensor
    /// within that memory block. The `data` pointer must be valid for the lifetime `'a`
    /// and that it is safe to mutate the data.
    pub unsafe fn from_raw_parts(data: *mut C, dims: [usize; D], strides: [usize; D]) -> Self {
        unsafe {
            Self {
                data: NonNull::new_unchecked(data),
                dims,
                strides,
                __marker: std::marker::PhantomData,
            }
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
        unsafe { &*self.ptr_at(indices) }
    }

    /// Returns a mutable reference to an element at the given indices, without performing bounds checks.
    ///
    /// # Safety
    ///
    /// Calling this method with out-of-bounds `indices` is undefined behavior.
    #[inline(always)]
    pub unsafe fn get_mut_unchecked(&mut self, indices: &[usize; D]) -> &mut C {
        unsafe { &mut *self.ptr_at_mut(indices) }
    }

    /// Returns a raw pointer to an element at the given indices, without performing bounds checks.
    ///
    /// # Safety
    ///
    /// Calling this method with out-of-bounds `indices` is undefined behavior.
    #[inline(always)]
    pub unsafe fn ptr_at(&self, indices: &[usize; D]) -> *const C {
        unsafe {
            let flat_idx = calculate_flat_index(indices, &self.strides);
            self.as_ptr().add(flat_idx)
        }
    }

    /// Returns a mutable raw pointer to an element at the given indices, without performing bounds checks.
    ///
    /// # Safety
    ///
    /// Calling this method with out-of-bounds `indices` is undefined behavior.
    #[inline(always)]
    pub unsafe fn ptr_at_mut(&mut self, indices: &[usize; D]) -> *mut C {
        unsafe {
            let flat_idx = calculate_flat_index(indices, &self.strides);
            self.as_ptr_mut().add(flat_idx)
        }
    }
}

impl<'a, C: Memorable, const D: usize> std::ops::Index<[usize; D]> for SymSqTensorMut<'a, C, D> {
    type Output = C;

    fn index(&self, indices: [usize; D]) -> &Self::Output {
        self.get(&indices)
    }
}

impl<'a, C: Memorable, const D: usize> std::ops::IndexMut<[usize; D]> for SymSqTensorMut<'a, C, D> {
    fn index_mut(&mut self, indices: [usize; D]) -> &mut Self::Output {
        self.get_mut(&indices)
    }
}

// TODO add some documentation plus a todo tag on relevant rust issues (const generic expressions)
impl<C: Memorable> SymSqTensor<C, 5> {
    /// Returns an immutable reference to the 3D subtensor at the given matrix indices.
    pub fn subtensor_ref(&self, m1: usize, m2: usize) -> TensorRef<'_, C, 3> {
        check_bounds(&[m1, m2, 0, 0, 0], &self.dims);
        // Safety: bounds have been checked.
        unsafe { self.subtensor_ref_unchecked(m1, m2) }
    }

    /// Returns a mutable reference to the 3D subtensor at the given matrix indices.
    pub fn subtensor_mut(&mut self, m1: usize, m2: usize) -> TensorMut<'_, C, 3> {
        check_bounds(&[m1, m2, 0, 0, 0], &self.dims);
        // Safety: bounds have been checked.
        unsafe { self.subtensor_mut_unchecked(m1, m2) }
    }

    /// Returns an immutable reference to the 3D subtensor at the given matrix indices without bounds checking.
    ///
    /// # Safety
    ///
    /// Caller should ensure that m1 and m2 are in bounds.
    pub unsafe fn subtensor_ref_unchecked(&self, m1: usize, m2: usize) -> TensorRef<'_, C, 3> {
        unsafe {
            TensorRef::from_raw_parts(
                self.ptr_at(&[m1, m2, 0, 0, 0]),
                [self.dims[2], self.dims[3], self.dims[4]],
                [self.strides[2], self.strides[3], self.strides[4]],
            )
        }
    }

    /// Returns a mutable reference to the 3D subtensor at the given matrix indices without bounds checking.
    ///
    /// # Safety
    ///
    /// Caller should ensure that m1 and m2 are in bounds.
    pub unsafe fn subtensor_mut_unchecked(&mut self, m1: usize, m2: usize) -> TensorMut<'_, C, 3> {
        unsafe {
            TensorMut::from_raw_parts(
                self.ptr_at_mut(&[m1, m2, 0, 0, 0]),
                [self.dims[2], self.dims[3], self.dims[4]],
                [self.strides[2], self.strides[3], self.strides[4]],
            )
        }
    }
}

impl<C: Memorable> SymSqTensor<C, 4> {
    /// Returns an immutable matrix reference to the subtensor at the given indices.
    pub fn subtensor_ref(&self, m1: usize, m2: usize) -> MatRef<'_, C> {
        check_bounds(&[m1, m2, 0, 0], &self.dims);
        // Safety: bounds have been checked.
        unsafe { self.subtensor_ref_unchecked(m1, m2) }
    }

    /// Returns a mutable matrix reference to the subtensor at the given indices.
    pub fn subtensor_mut(&mut self, m1: usize, m2: usize) -> MatMut<'_, C> {
        check_bounds(&[m1, m2, 0, 0], &self.dims);
        // Safety: bounds have been checked.
        unsafe { self.subtensor_mut_unchecked(m1, m2) }
    }

    /// Returns an immutable matrix reference to the subtensor at the given indices without bounds checking.
    ///
    /// # Safety
    ///
    /// Caller should ensure that m1 and m2 are in bounds.
    pub unsafe fn subtensor_ref_unchecked(&self, m1: usize, m2: usize) -> MatRef<'_, C> {
        unsafe {
            MatRef::from_raw_parts(
                self.ptr_at(&[m1, m2, 0, 0]),
                self.dims[2],
                self.dims[3],
                self.strides[2] as isize,
                self.strides[3] as isize,
            )
        }
    }

    /// Returns a mutable matrix reference to the subtensor at the given indices without bounds checking.
    ///
    /// # Safety
    ///
    /// Caller should ensure that m1 and m2 are in bounds.
    pub unsafe fn subtensor_mut_unchecked(&mut self, m1: usize, m2: usize) -> MatMut<'_, C> {
        unsafe {
            MatMut::from_raw_parts_mut(
                self.ptr_at_mut(&[m1, m2, 0, 0]),
                self.dims[2],
                self.dims[3],
                self.strides[2] as isize,
                self.strides[3] as isize,
            )
        }
    }
}

impl<C: Memorable> SymSqTensor<C, 3> {
    /// Returns an immutable row reference to the subtensor at the given indices.
    pub fn subtensor_ref(&self, m1: usize, m2: usize) -> RowRef<'_, C> {
        check_bounds(&[m1, m2, 0], &self.dims);
        // Safety: bounds have been checked.
        unsafe { self.subtensor_ref_unchecked(m1, m2) }
    }

    /// Returns a mutable row reference to the subtensor at the given indices.
    pub fn subtensor_mut(&mut self, m1: usize, m2: usize) -> RowMut<'_, C> {
        check_bounds(&[m1, m2, 0], &self.dims);
        // Safety: bounds have been checked.
        unsafe { self.subtensor_mut_unchecked(m1, m2) }
    }

    /// Returns an immutable row reference to the subtensor at the given indices without bounds checking.
    ///
    /// # Safety
    ///
    /// Caller should ensure that m1 and m2 are in bounds.
    pub unsafe fn subtensor_ref_unchecked(&self, m1: usize, m2: usize) -> RowRef<'_, C> {
        unsafe {
            RowRef::from_raw_parts(
                self.ptr_at(&[m1, m2, 0]),
                self.dims[2],
                self.strides[2] as isize,
            )
        }
    }

    /// Returns a mutable row reference to the subtensor at the given indices without bounds checking.
    ///
    /// # Safety
    ///
    /// Caller should ensure that m1 and m2 are in bounds.
    pub unsafe fn subtensor_mut_unchecked(&mut self, m1: usize, m2: usize) -> RowMut<'_, C> {
        unsafe {
            RowMut::from_raw_parts_mut(
                self.ptr_at_mut(&[m1, m2, 0]),
                self.dims[2],
                self.strides[2] as isize,
            )
        }
    }
}

impl<C: Memorable> SymSqTensor<C, 2> {
    /// Returns an immutable reference to the element at the given indices.
    pub fn subtensor_ref(&self, m1: usize, m2: usize) -> &C {
        self.get(&[m1, m2])
    }

    /// Returns a mutable reference to the element at the given indices.
    pub fn subtensor_mut(&mut self, m1: usize, m2: usize) -> &mut C {
        self.get_mut(&[m1, m2])
    }

    /// Returns an immutable reference to the element at the given indices without bounds checking.
    ///
    /// # Safety
    ///
    /// Caller should ensure that m1 and m2 are in bounds.
    pub unsafe fn subtensor_ref_unchecked(&self, m1: usize, m2: usize) -> &C {
        unsafe { self.get_unchecked(&[m1, m2]) }
    }

    /// Returns a mutable reference to the element at the given indices without bounds checking.
    ///
    /// # Safety
    ///
    /// Caller should ensure that m1 and m2 are in bounds.
    pub unsafe fn subtensor_mut_unchecked(&mut self, m1: usize, m2: usize) -> &mut C {
        unsafe { self.get_mut_unchecked(&[m1, m2]) }
    }
}

impl<'a, C: Memorable> SymSqTensorRef<'a, C, 5> {
    /// Returns an immutable reference to the 3D subtensor at the given matrix indices.
    pub fn subtensor_ref(&self, m1: usize, m2: usize) -> TensorRef<'a, C, 3> {
        check_bounds(&[m1, m2, 0, 0, 0], &self.dims);
        // Safety: bounds have been checked.
        unsafe { self.subtensor_ref_unchecked(m1, m2) }
    }

    /// Returns an immutable reference to the 3D subtensor at the given matrix indices without bounds checking.
    ///
    /// # Safety
    ///
    /// Caller should ensure that m1 and m2 are in bounds.
    pub unsafe fn subtensor_ref_unchecked(&self, m1: usize, m2: usize) -> TensorRef<'a, C, 3> {
        unsafe {
            TensorRef::from_raw_parts(
                self.ptr_at(&[m1, m2, 0, 0, 0]),
                [self.dims[2], self.dims[3], self.dims[4]],
                [self.strides[2], self.strides[3], self.strides[4]],
            )
        }
    }
}

impl<'a, C: Memorable> SymSqTensorRef<'a, C, 4> {
    /// Returns an immutable matrix reference to the subtensor at the given indices.
    pub fn subtensor_ref(&self, m1: usize, m2: usize) -> MatRef<'a, C> {
        check_bounds(&[m1, m2, 0, 0], &self.dims);
        // Safety: bounds have been checked.
        unsafe { self.subtensor_ref_unchecked(m1, m2) }
    }

    /// Returns an immutable matrix reference to the subtensor at the given indices without bounds checking.
    ///
    /// # Safety
    ///
    /// Caller should ensure that m1 and m2 are in bounds.
    pub unsafe fn subtensor_ref_unchecked(&self, m1: usize, m2: usize) -> MatRef<'a, C> {
        unsafe {
            MatRef::from_raw_parts(
                self.ptr_at(&[m1, m2, 0, 0]),
                self.dims[2],
                self.dims[3],
                self.strides[2] as isize,
                self.strides[3] as isize,
            )
        }
    }
}

impl<'a, C: Memorable> SymSqTensorRef<'a, C, 3> {
    /// Returns an immutable row reference to the subtensor at the given indices.
    pub fn subtensor_ref(&self, m1: usize, m2: usize) -> RowRef<'a, C> {
        check_bounds(&[m1, m2, 0], &self.dims);
        // Safety: bounds have been checked.
        unsafe { self.subtensor_ref_unchecked(m1, m2) }
    }

    /// Returns an immutable row reference to the subtensor at the given indices without bounds checking.
    ///
    /// # Safety
    ///
    /// Caller should ensure that m1 and m2 are in bounds.
    pub unsafe fn subtensor_ref_unchecked(&self, m1: usize, m2: usize) -> RowRef<'a, C> {
        unsafe {
            RowRef::from_raw_parts(
                self.ptr_at(&[m1, m2, 0]),
                self.dims[2],
                self.strides[2] as isize,
            )
        }
    }
}

impl<'a, C: Memorable> SymSqTensorRef<'a, C, 2> {
    /// Returns an immutable reference to the element at the given indices.
    pub fn subtensor_ref(&self, m1: usize, m2: usize) -> &C {
        self.get(&[m1, m2])
    }

    /// Returns an immutable reference to the element at the given indices without bounds checking.
    ///
    /// # Safety
    ///
    /// Caller should ensure that m1 and m2 are in bounds.
    pub unsafe fn subtensor_ref_unchecked(&self, m1: usize, m2: usize) -> &C {
        unsafe { self.get_unchecked(&[m1, m2]) }
    }
}

impl<'a, C: Memorable> SymSqTensorMut<'a, C, 5> {
    /// Returns an immutable reference to the 3D subtensor at the given matrix indices.
    pub fn subtensor_ref(&self, m1: usize, m2: usize) -> TensorRef<'a, C, 3> {
        check_bounds(&[m1, m2, 0, 0, 0], &self.dims);
        // Safety: bounds have been checked.
        unsafe { self.subtensor_ref_unchecked(m1, m2) }
    }

    /// Returns a mutable reference to the 3D subtensor at the given matrix indices.
    pub fn subtensor_mut(&mut self, m1: usize, m2: usize) -> TensorMut<'a, C, 3> {
        check_bounds(&[m1, m2, 0, 0, 0], &self.dims);
        // Safety: bounds have been checked.
        unsafe { self.subtensor_mut_unchecked(m1, m2) }
    }

    /// Returns an immutable reference to the 3D subtensor at the given matrix indices without bounds checking.
    ///
    /// # Safety
    ///
    /// Caller should ensure that m1 and m2 are in bounds.
    pub unsafe fn subtensor_ref_unchecked(&self, m1: usize, m2: usize) -> TensorRef<'a, C, 3> {
        unsafe {
            TensorRef::from_raw_parts(
                self.ptr_at(&[m1, m2, 0, 0, 0]),
                [self.dims[2], self.dims[3], self.dims[4]],
                [self.strides[2], self.strides[3], self.strides[4]],
            )
        }
    }

    /// Returns a mutable reference to the 3D subtensor at the given matrix indices without bounds checking.
    ///
    /// # Safety
    ///
    /// Caller should ensure that m1 and m2 are in bounds.
    pub unsafe fn subtensor_mut_unchecked(&mut self, m1: usize, m2: usize) -> TensorMut<'a, C, 3> {
        unsafe {
            TensorMut::from_raw_parts(
                self.ptr_at_mut(&[m1, m2, 0, 0, 0]),
                [self.dims[2], self.dims[3], self.dims[4]],
                [self.strides[2], self.strides[3], self.strides[4]],
            )
        }
    }
}

impl<'a, C: Memorable> SymSqTensorMut<'a, C, 4> {
    /// Returns an immutable matrix reference to the subtensor at the given indices.
    pub fn subtensor_ref(&self, m1: usize, m2: usize) -> MatRef<'a, C> {
        check_bounds(&[m1, m2, 0, 0], &self.dims);
        // Safety: bounds have been checked.
        unsafe { self.subtensor_ref_unchecked(m1, m2) }
    }

    /// Returns a mutable matrix reference to the subtensor at the given indices.
    pub fn subtensor_mut(&mut self, m1: usize, m2: usize) -> MatMut<'a, C> {
        check_bounds(&[m1, m2, 0, 0], &self.dims);
        // Safety: bounds have been checked.
        unsafe { self.subtensor_mut_unchecked(m1, m2) }
    }

    /// Returns an immutable matrix reference to the subtensor at the given indices without bounds checking.
    ///
    /// # Safety
    ///
    /// Caller should ensure that m1 and m2 are in bounds.
    pub unsafe fn subtensor_ref_unchecked(&self, m1: usize, m2: usize) -> MatRef<'a, C> {
        unsafe {
            MatRef::from_raw_parts(
                self.ptr_at(&[m1, m2, 0, 0]),
                self.dims[2],
                self.dims[3],
                self.strides[2] as isize,
                self.strides[3] as isize,
            )
        }
    }

    /// Returns a mutable matrix reference to the subtensor at the given indices without bounds checking.
    ///
    /// # Safety
    ///
    /// Caller should ensure that m1 and m2 are in bounds.
    pub unsafe fn subtensor_mut_unchecked(&mut self, m1: usize, m2: usize) -> MatMut<'a, C> {
        unsafe {
            MatMut::from_raw_parts_mut(
                self.ptr_at_mut(&[m1, m2, 0, 0]),
                self.dims[2],
                self.dims[3],
                self.strides[2] as isize,
                self.strides[3] as isize,
            )
        }
    }
}

impl<'a, C: Memorable> SymSqTensorMut<'a, C, 3> {
    /// Returns an immutable row reference to the subtensor at the given indices.
    pub fn subtensor_ref(&self, m1: usize, m2: usize) -> RowRef<'a, C> {
        check_bounds(&[m1, m2, 0], &self.dims);
        // Safety: bounds have been checked.
        unsafe { self.subtensor_ref_unchecked(m1, m2) }
    }

    /// Returns a mutable row reference to the subtensor at the given indices.
    pub fn subtensor_mut(&mut self, m1: usize, m2: usize) -> RowMut<'a, C> {
        check_bounds(&[m1, m2, 0], &self.dims);
        // Safety: bounds have been checked.
        unsafe { self.subtensor_mut_unchecked(m1, m2) }
    }

    /// Returns an immutable row reference to the subtensor at the given indices without bounds checking.
    ///
    /// # Safety
    ///
    /// Caller should ensure that m1 and m2 are in bounds.
    pub unsafe fn subtensor_ref_unchecked(&self, m1: usize, m2: usize) -> RowRef<'a, C> {
        unsafe {
            RowRef::from_raw_parts(
                self.ptr_at(&[m1, m2, 0]),
                self.dims[2],
                self.strides[2] as isize,
            )
        }
    }

    /// Returns a mutable row reference to the subtensor at the given indices without bounds checking.
    ///
    /// # Safety
    ///
    /// Caller should ensure that m1 and m2 are in bounds.
    pub unsafe fn subtensor_mut_unchecked(&mut self, m1: usize, m2: usize) -> RowMut<'a, C> {
        unsafe {
            RowMut::from_raw_parts_mut(
                self.ptr_at_mut(&[m1, m2, 0]),
                self.dims[2],
                self.strides[2] as isize,
            )
        }
    }
}

impl<'a, C: Memorable> SymSqTensorMut<'a, C, 2> {
    /// Returns an immutable reference to the element at the given indices.
    pub fn subtensor_ref(&self, m1: usize, m2: usize) -> &C {
        self.get(&[m1, m2])
    }

    /// Returns a mutable reference to the element at the given indices.
    pub fn subtensor_mut(&mut self, m1: usize, m2: usize) -> &mut C {
        self.get_mut(&[m1, m2])
    }

    /// Returns an immutable reference to the element at the given indices without bounds checking.
    ///
    /// # Safety
    ///
    /// Caller should ensure that m1 and m2 are in bounds.
    pub unsafe fn subtensor_ref_unchecked(&self, m1: usize, m2: usize) -> &C {
        unsafe { self.get_unchecked(&[m1, m2]) }
    }

    /// Returns a mutable reference to the element at the given indices without bounds checking.
    ///
    /// # Safety
    ///
    /// Caller should ensure that m1 and m2 are in bounds.
    pub unsafe fn subtensor_mut_unchecked(&mut self, m1: usize, m2: usize) -> &mut C {
        unsafe { self.get_mut_unchecked(&[m1, m2]) }
    }
}
