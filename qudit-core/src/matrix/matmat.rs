use std::ptr::NonNull;

use faer::{MatMut, MatRef};
use faer_traits::ComplexField;

use crate::memory::{
    alloc_zeroed_memory, calc_col_stride, calc_mat_stride, Memorable, MemoryBuffer,
};

use super::MatVec;

/// Convert SymSqMatMat internal indexing to external indexing major indexing.
///
/// The internal indexing is used to store the upper triangular part of the
/// tensor in compact form as a vector. The external indexing is used to
/// index the two most significant dimensions of the tensor as a 4-dimensional
/// array.
///
/// Since the data is stored in compact form as column-major, we first
/// find j by solving for the smallest N such that N(N+1)/2 <= k. We can
/// then just substitute in the equation to find i in [coords_to_index].
#[inline(always)]
#[allow(dead_code)]
pub fn index_to_coords(index: usize) -> (usize, usize) {
    let j = ((((8 * index + 1) as f64).sqrt().floor() as usize) - 1) / 2;
    let i = index - j * (j + 1) / 2;
    (i, j)
}

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
pub fn coords_to_index(i: usize, j: usize) -> usize {
    if i <= j {
        j * (j + 1) / 2 + i
    } else {
        i * (i + 1) / 2 + j
    }
}

macro_rules! matmat_prop_impl {
    () => {
        /// Returns the number of rows in the tensor.
        #[inline(always)]
        pub fn nrows(&self) -> usize {
            self.nrows
        }

        /// Returns the number of columns in the tensor.
        #[inline(always)]
        pub fn ncols(&self) -> usize {
            self.ncols
        }

        /// Returns the number of matrices in the tensor.
        #[inline(always)]
        pub fn nmats(&self) -> usize {
            self.nmats
        }

        /// Returns the number of elements in the tensor.
        #[inline(always)]
        pub fn len(&self) -> usize {
            self.nmats() * self.nmats() * self.nrows() * self.ncols()
        }

        /// Returns the row stride of the tensor.
        #[inline(always)]
        pub fn row_stride(&self) -> usize {
            1
        }

        /// Returns the column stride of the tensor.
        #[inline(always)]
        pub fn col_stride(&self) -> usize {
            self.col_stride
        }

        /// Returns the matrix stride of the tensor.
        #[inline(always)]
        pub fn mat_stride(&self) -> usize {
            self.mat_stride
        }
    };
}

macro_rules! matmat_bounds_impl {
    () => {
        /// Checks that the indices are within bounds for this tensor.
        #[inline(always)]
        pub fn is_in_bounds(&self, m1: usize, m2: usize, r: usize, c: usize) -> bool {
            m1 < self.nmats && m2 < self.nmats && r < self.nrows && c < self.ncols
        }

        /// Panic if the indices are out-of-bounds for this tensor.
        #[inline(always)]
        #[track_caller]
        pub fn panic_on_out_of_bounds(&self, m1: usize, m2: usize, r: usize, c: usize) {
            if m1 >= self.nmats {
                panic!(
                    "Major matrix index out of bounds: index = {}, which is >= {}.",
                    m1, self.nmats
                );
            }

            if m2 >= self.nmats {
                panic!(
                    "Minor matrix index out of bounds: index = {}, which is >= {}.",
                    m2, self.nmats
                );
            }

            if r >= self.nrows {
                panic!(
                    "Row index out of bounds: index = {}, which is >= {}.",
                    r, self.nrows
                );
            }

            if c >= self.ncols {
                panic!(
                    "Column index out of bounds: index = {}, which is >= {}.",
                    c, self.ncols
                );
            }
        }
    };
}

macro_rules! matmat_read_impl {
    ($ptr_access:ident) => {
        /// Reads an element from the tensor without bounds checking.
        ///
        /// # Arguments
        ///
        /// * `m1` - The major matrix index.
        ///
        /// * `m2` - The minor matrix index.
        ///
        /// * `r` - The row index.
        ///
        /// * `c` - The column index.
        ///
        /// # Safety
        ///
        /// - The caller must ensure that the indices are within bounds.
        ///
        /// # Examples
        ///
        /// ```
        /// use qudit_core::matrix::SymSqMatMat;
        /// let mut mat_mat = SymSqMatMat::<f64>::zeros(3, 2, 2);
        /// unsafe {
        ///    mat_mat.write_unchecked(0, 0, 0, 0, 1.0);
        ///    assert_eq!(mat_mat.read_unchecked(0, 0, 0, 0), 1.0);
        ///    mat_mat.write_unchecked(1, 1, 0, 0, 2.0);
        ///    assert_eq!(mat_mat.read_unchecked(1, 1, 0, 0), 2.0);
        /// }
        /// ```
        ///
        /// # See Also
        ///
        /// - [read] method for a safe version of this method.
        #[inline(always)]
        #[track_caller]
        pub unsafe fn read_unchecked(&self, m1: usize, m2: usize, r: usize, c: usize) -> C {
            *self.$ptr_access(coords_to_index(m1, m2), r, c)
        }

        /// Reads an element from the tensor.
        ///
        /// # Arguments
        ///
        /// * `m1` - The major matrix index.
        ///
        /// * `m2` - The minor matrix index.
        ///
        /// * `r` - The row index.
        ///
        /// * `c` - The column index.
        ///
        /// # Returns
        ///
        /// The element at the specified indices.
        ///
        /// # Panics
        ///
        /// Panics if the indices are out of bounds.
        ///
        /// # Examples
        ///
        /// ```
        /// use qudit_core::matrix::SymSqMatMat;
        /// let mut mat_mat = SymSqMatMat::<f64>::zeros(3, 2, 2);
        /// mat_mat.write(0, 0, 0, 0, 1.0);
        /// assert_eq!(mat_mat.read(0, 0, 0, 0), 1.0);
        /// mat_mat.write(1, 1, 0, 0, 2.0);
        /// assert_eq!(mat_mat.read(1, 1, 0, 0), 2.0);
        /// ```
        ///
        /// # See Also
        ///
        /// - [read_unchecked] method for a version of this method without bounds checking.
        #[inline]
        #[track_caller]
        pub fn read(&self, m1: usize, m2: usize, r: usize, c: usize) -> C {
            self.panic_on_out_of_bounds(m1, m2, r, c);
            // SAFETY: The bounds have been checked.
            unsafe { self.read_unchecked(m1, m2, r, c) }
        }
    };
}

macro_rules! matmat_write_impl {
    ($ptr_access:ident) => {
        /// Writes an element to the tensor without bounds checking.
        ///
        /// See the [read_unchecked] method for more information.
        ///
        /// # Arguments
        ///
        /// * `m1` - The major matrix index.
        ///
        /// * `m2` - The minor matrix index.
        ///
        /// * `r` - The row index.
        ///
        /// * `c` - The column index.
        ///
        /// * `val` - The value to write.
        ///
        /// # Safety
        ///
        /// - The caller must ensure that the indices are within bounds.
        ///
        /// # See Also
        ///
        /// - [write] method for a safe version of this method.
        /// - [read] method for a safe version of the [read_unchecked] method.
        /// - [read_unchecked] method for more information on reading elements.
        #[inline(always)]
        #[track_caller]
        pub unsafe fn write_unchecked(&mut self, m1: usize, m2: usize, r: usize, c: usize, val: C) {
            *self.$ptr_access(coords_to_index(m1, m2), r, c) = val;
        }

        /// Writes an element to the tensor.
        ///
        /// See the [read] method for more information.
        ///
        /// # Arguments
        ///
        /// * `m1` - The matrix index.
        ///
        /// * `m2` - The matrix index.
        ///
        /// * `r` - The row index.
        ///
        /// * `c` - The column index.
        ///
        /// * `val` - The value to write.
        ///
        /// # Panics
        ///
        /// Panics if the indices are out of bounds.
        ///
        /// # See Also
        ///
        /// - [write_unchecked] method for a version of this method without bounds checking.
        /// - [read] method for a safe version of the [read_unchecked] method.
        /// - [read_unchecked] method for more information on reading elements.
        #[inline]
        #[track_caller]
        pub fn write(&mut self, m1: usize, m2: usize, r: usize, c: usize, val: C) {
            self.panic_on_out_of_bounds(m1, m2, r, c);
            // SAFETY: The bounds have been checked.
            unsafe { self.write_unchecked(m1, m2, r, c, val) };
        }
    };
}

/// A symmetric 4-dimensional tensor with a matrix-vector storage layout.
///
/// The tensor is indexed by four indices: the major matrix index, the minor
/// matrix index, the row index, and the column index. The row index is the
/// fastest moving index, followed by the column index, the minor matrix index,
/// and finally the major matrix index. However, the upper triangular half of
/// the tensor is stored in compact form as a vector, so the internal data
/// is accessed by three indices. The [coords_to_index] and [index_to_coords]
/// functions can be used to convert between the two indexing schemes.
///
/// Additionally, the tensor is square in its two major dimensions, so the
/// number of rows of columns of matrices are the same. This object was
/// designed to be used as a Hessian tensor, where the Hessian is symmetric.
///
/// The tensor is stored as a contiguous block of memory rather than a
/// vector of matrices. This allows for better memory performance when
/// iterating over the elements of the tensor.
///
/// # Type Parameters
///
/// - `C`: The type of the elements in the tensor.
///
/// # See Also
///
/// - [Mat] for a 2-dimensional matrix.
/// - [MatVec] for a 3-dimensional tensor with a matrix-vector storage layout.
/// - [SymSqMatMatRef] for a const reference into a SymSqMatMat.
/// - [SymSqMatMatMut] for a mutable reference to a SymSqMatMat.
pub struct SymSqMatMat<C: Memorable> {
    data: MemoryBuffer<C>,
    nrows: usize,
    ncols: usize,
    nmats: usize,
    col_stride: usize,
    mat_stride: usize,
}

impl<C: Memorable> SymSqMatMat<C> {
    /// Creates a new, zeroed matrix-matrix tensor with the specified dimensions.
    ///
    /// # Arguments
    ///
    /// * `nrows` - The number of rows in each matrix.
    ///
    /// * `ncols` - The number of columns in each matrix.
    ///
    /// * `nmats` - The number of matrices in each row/col of the tensor.
    ///
    /// # Returns
    ///
    /// A new, zeroed matrix-matrix tensor with the specified dimensions.
    ///
    /// # Panics
    ///
    /// Panics if the number of elements in the tensor would overflow.
    ///
    /// # Examples
    ///
    /// ```
    /// use qudit_core::matrix::SymSqMatMat;
    /// let mat_mat = SymSqMatMat::<f64>::zeros(3, 2, 2);
    /// assert_eq!(mat_mat.nrows(), 3);
    /// assert_eq!(mat_mat.ncols(), 2);
    /// assert_eq!(mat_mat.nmats(), 2);
    /// ```
    ///
    /// # See Also
    ///
    /// - [read] method for reading elements from the tensor.
    /// - [write] method for writing elements to the tensor.
    pub fn zeros(nrows: usize, ncols: usize, nmats: usize) -> Self {
        let col_stride = calc_col_stride::<C>(nrows, ncols);
        let mat_stride = calc_mat_stride::<C>(nrows, ncols, col_stride);
        let nmats_plus_one = nmats.checked_add(1).expect("Matrix-matrix size overflow.");
        let internal_nmats = nmats
            .checked_mul(nmats_plus_one)
            .expect("Matrix-matrix size overflow.")
            .checked_div(2)
            .expect("Matrix-matrix size underflow.");
        let mat_vec_size = internal_nmats
            .checked_mul(mat_stride)
            .expect("Matrix-vector size overflow.");
        let data = alloc_zeroed_memory::<C>(mat_vec_size);
        Self {
            data,
            nrows,
            ncols,
            nmats,
            col_stride,
            mat_stride,
        }
    }

    /// Returns a pointer to a specific element in the tensor.
    ///
    /// Uses internal indexing.
    ///
    /// # Safety
    ///
    /// - The caller must ensure that the indices are within bounds.
    #[inline(always)]
    #[track_caller]
    unsafe fn unchecked_ptr_at(&self, m: usize, r: usize, c: usize) -> *const C {
        let offset = m
            .wrapping_mul(self.mat_stride)
            .wrapping_add(c.wrapping_mul(self.col_stride))
            .wrapping_add(r);

        self.data.as_ptr().offset(offset as isize)
    }

    /// Returns a mutable pointer to a specific element in the tensor.
    ///
    /// Uses internal indexing.
    ///
    /// # Safety
    ///
    /// - The caller must ensure that the indices are within bounds.
    #[inline(always)]
    #[track_caller]
    unsafe fn unchecked_ptr_mut_at(&mut self, m: usize, r: usize, c: usize) -> *mut C {
        let offset = m
            .wrapping_mul(self.mat_stride)
            .wrapping_add(c.wrapping_mul(self.col_stride))
            .wrapping_add(r);

        self.data.as_mut_ptr().offset(offset as isize)
    }

    /// Returns a pointer to the tensor data.
    #[inline(always)]
    pub fn as_ptr(&self) -> *const C {
        self.data.as_ptr()
    }

    /// Returns a mutable pointer to the tensor data.
    #[inline(always)]
    pub fn as_mut_ptr(&mut self) -> *mut C {
        self.data.as_mut_ptr()
    }

    /// Returns a mutable reference to the tensor.
    #[inline(always)]
    pub fn as_mut(&mut self) -> SymSqMatMatMut<'_, C> {
        unsafe {
            SymSqMatMatMut::from_raw_parts(
                self.as_mut_ptr(),
                self.nrows,
                self.ncols,
                self.nmats,
                self.col_stride,
                self.mat_stride,
            )
        }
    }

    /// Returns a const reference to the tensor.
    #[inline(always)]
    pub fn as_ref(&self) -> SymSqMatMatRef<'_, C> {
        unsafe {
            SymSqMatMatRef::from_raw_parts(
                self.as_ptr(),
                self.nrows,
                self.ncols,
                self.nmats,
                self.col_stride,
                self.mat_stride,
            )
        }
    }

    /// Returns a reference to a specific matrix in the tensor.
    ///
    /// # Arguments
    ///
    /// * `m1` - The major matrix index.
    /// * `m2` - The minor matrix index.
    #[inline(always)]
    pub fn mat_ref(&self, m1: usize, m2: usize) -> MatRef<'_, C> {
        assert!(m1 < self.nmats);
        assert!(m2 < self.nmats);
        unsafe {
            faer::MatRef::from_raw_parts(
                self.unchecked_ptr_at(coords_to_index(m1, m2), 0, 0),
                self.nrows,
                self.ncols,
                1,
                self.col_stride.try_into().unwrap(),
            )
        }
    }

    /// Returns a mutable reference to a specific matrix in the tensor.
    ///
    /// # Arguments
    ///
    /// * `m1` - The major matrix index.
    /// * `m2` - The minor matrix index.
    #[inline(always)]
    pub fn mat_mut(&mut self, m1: usize, m2: usize) -> MatMut<'_, C> {
        assert!(m1 < self.nmats);
        assert!(m2 < self.nmats);
        unsafe {
            faer::MatMut::from_raw_parts_mut(
                self.unchecked_ptr_mut_at(coords_to_index(m1, m2), 0, 0),
                self.nrows,
                self.ncols,
                1,
                self.col_stride.try_into().unwrap(),
            )
        }
    }

    /// Returns a copy of a specific matrix-vector row in this tensor.
    ///
    /// # Arguments
    ///
    /// * `m1` - The major matrix index.
    ///
    /// # Note:
    ///
    /// This method performs a full copy of the row as the row may not
    /// be stored consecutively in memory. As a result, this method is
    /// not efficient and should be used sparingly.
    #[inline(always)]
    pub fn get_row(&self, m1: usize) -> MatVec<C> {
        let mut mat_vec = MatVec::<C>::zeros(self.nrows, self.ncols, self.nmats);
        for m2 in 0..self.nmats {
            for r in 0..self.nrows {
                for c in 0..self.ncols {
                    let val = self.read(m1, m2, r, c);
                    mat_vec.write(m2, r, c, val);
                }
            }
        }
        mat_vec
    }

    matmat_prop_impl!();
    matmat_bounds_impl!();
    matmat_read_impl!(unchecked_ptr_at);
    matmat_write_impl!(unchecked_ptr_mut_at);
}

impl<C: Memorable + ComplexField> std::ops::Mul<C> for SymSqMatMat<C> {
    type Output = SymSqMatMat<C>;

    fn mul(self, rhs: C) -> Self::Output {
        let mut symsqmat = SymSqMatMat::<C>::zeros(self.nrows, self.ncols, self.nmats);
        for m1 in 0..self.nmats {
            for m2 in m1..self.nmats {
                for r in 0..self.nrows {
                    for c in 0..self.ncols {
                        let val = self.read(m1, m2, r, c);
                        symsqmat.write(m1, m2, r, c, val.mul_by_ref(&rhs));
                    }
                }
            }
        }
        symsqmat
    }
}

impl<C: Memorable + ComplexField> std::ops::MulAssign<C> for SymSqMatMat<C> {
    fn mul_assign(&mut self, rhs: C) {
        for m1 in 0..self.nmats {
            for m2 in m1..self.nmats {
                for r in 0..self.nrows {
                    for c in 0..self.ncols {
                        let val = self.read(m1, m2, r, c);
                        self.write(m1, m2, r, c, val.mul_by_ref(&rhs));
                    }
                }
            }
        }
    }
}

impl<C: Memorable + std::fmt::Debug> std::fmt::Debug for SymSqMatMat<C> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut debug_struct = f.debug_struct("SymSqMatMat");
        debug_struct.field("nrows", &self.nrows);
        debug_struct.field("ncols", &self.ncols);
        debug_struct.field("nmats", &self.nmats);
        debug_struct.field("col_stride", &self.col_stride);
        debug_struct.field("mat_stride", &self.mat_stride);
        // TODO: Better Debugging
        debug_struct.finish()
    }
}

impl<C: Memorable + ComplexField> PartialEq for SymSqMatMat<C> {
    fn eq(&self, other: &Self) -> bool {
        if self.nrows != other.nrows || self.ncols != other.ncols || self.nmats != other.nmats {
            return false;
        }

        for m1 in 0..self.nmats {
            for m2 in m1..self.nmats {
                for r in 0..self.nrows {
                    for c in 0..self.ncols {
                        if self.read(m1, m2, r, c) != other.read(m1, m2, r, c) {
                            return false;
                        }
                    }
                }
            }
        }

        true
    }
}

/// A mutable reference to a 4-dimensional symmetrical square tensor.
///
/// See the [SymSqMatMat] type for more information.
pub struct SymSqMatMatMut<'a, C: Memorable> {
    data: NonNull<C>,
    nrows: usize,
    ncols: usize,
    nmats: usize,
    col_stride: usize,
    mat_stride: usize,
    __marker: std::marker::PhantomData<&'a C>,
}

impl<'a, C: Memorable> SymSqMatMatMut<'a, C> {
    /// Creates a `SymSqMatMatMut` from pointers to tensor data, dimensions, and strides.
    ///
    /// # Safety
    ///
    /// The caller must ensure:
    /// - The pointers are valid and non-null.
    /// - For each unit, the entire memory region addressed by the tensor is
    ///   within a single allocation.
    /// - The memory is accessible by the pointer.
    /// - No mutable aliasing occurs. No mutable or const references to the
    ///  tensor data exist when the `SymSqMatMatMut` is alive.
    pub unsafe fn from_raw_parts(
        data: *mut C,
        nrows: usize,
        ncols: usize,
        nmats: usize,
        col_stride: usize,
        mat_stride: usize,
    ) -> Self {
        Self {
            data: NonNull::new_unchecked(data),
            nrows,
            ncols,
            nmats,
            col_stride,
            mat_stride,
            __marker: std::marker::PhantomData,
        }
    }

    /// Returns a mutable pointer to a specific element in the tensor.
    ///
    /// # Safety
    ///
    /// - The caller must ensure that the indices are within bounds.
    #[inline(always)]
    #[track_caller]
    unsafe fn unchecked_ptr_mut_at(&self, m: usize, r: usize, c: usize) -> *mut C {
        let offset = m
            .wrapping_mul(self.mat_stride)
            .wrapping_add(c.wrapping_mul(self.col_stride))
            .wrapping_add(r);

        self.data.as_ptr().offset(offset as isize)
    }

    /// Returns a mut reference to a specific matrix in the tensor.
    #[inline(always)]
    pub fn mat_mut(&self, m1: usize, m2: usize) -> MatMut<C> {
        let m = coords_to_index(m1, m2);
        let offset = m.wrapping_mul(self.mat_stride);
        unsafe {
            let ptr = self.data.as_ptr().offset(offset as isize);
            faer::MatMut::from_raw_parts_mut(
                ptr,
                self.nrows,
                self.ncols,
                1,
                self.col_stride.try_into().unwrap(),
            )
        }
    }

    matmat_prop_impl!();
    matmat_bounds_impl!();
    matmat_read_impl!(unchecked_ptr_mut_at);
    matmat_write_impl!(unchecked_ptr_mut_at);
}

/// A const reference to a 4-dimensional symmetrical square tensor.
///
/// See the [SymSqMatMat] type for more information.
pub struct SymSqMatMatRef<'a, C: Memorable> {
    data: NonNull<C>,
    nrows: usize,
    ncols: usize,
    nmats: usize,
    col_stride: usize,
    mat_stride: usize,
    __marker: std::marker::PhantomData<&'a C>,
}

impl<'a, C: Memorable> SymSqMatMatRef<'a, C> {
    /// Creates a `SymSqMatMatRef` from pointers to tensor data, dimensions, and strides.
    ///
    /// # Safety
    ///
    /// The caller must ensure:
    /// - The pointers are valid and non-null.
    /// - For each unit, the entire memory region addressed by the tensor is
    ///   within a single allocation.
    /// - The memory is accessible by the pointer.
    /// - No mutable aliasing occurs. No mutable references to the
    ///  tensor data exist when the `SymSqMatMatRef` is alive.
    pub unsafe fn from_raw_parts(
        data: *const C,
        nrows: usize,
        ncols: usize,
        nmats: usize,
        col_stride: usize,
        mat_stride: usize,
    ) -> Self {
        // SAFETY: The pointer is never used in an mutable context.
        let mut_ptr = data as *mut C;

        Self {
            data: NonNull::new_unchecked(mut_ptr),
            nrows,
            ncols,
            nmats,
            col_stride,
            mat_stride,
            __marker: std::marker::PhantomData,
        }
    }

    /// Returns a pointer to a specific element in the tensor.
    ///
    /// # Safety
    ///
    /// - The caller must ensure that the indices are within bounds.
    #[inline(always)]
    #[track_caller]
    unsafe fn unchecked_ptr_at(&self, m: usize, r: usize, c: usize) -> *mut C {
        let offset = m
            .wrapping_mul(self.mat_stride)
            .wrapping_add(c.wrapping_mul(self.col_stride))
            .wrapping_add(r);
        self.data.as_ptr().offset(offset as isize)
    }

    /// Returns a const reference to a specific matrix in the tensor.
    #[inline(always)]
    pub fn mat_ref(&self, m1: usize, m2: usize) -> MatRef<C> {
        let m = coords_to_index(m1, m2);
        let offset = m.wrapping_mul(self.mat_stride);
        unsafe {
            let ptr = self.data.as_ptr().offset(offset as isize);
            faer::MatRef::from_raw_parts(
                ptr,
                self.nrows,
                self.ncols,
                1,
                self.col_stride.try_into().unwrap(),
            )
        }
    }

    matmat_prop_impl!();
    matmat_bounds_impl!();
    matmat_read_impl!(unchecked_ptr_at);
}
