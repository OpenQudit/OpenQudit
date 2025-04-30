use std::{
    ops::{Index, IndexMut},
    ptr::NonNull,
};

use faer::{MatMut, MatRef};
use faer_traits::ComplexField;

use crate::memory::{
    alloc_zeroed_memory, calc_col_stride, calc_mat_stride, Memorable, MemoryBuffer,
};

macro_rules! matvec_prop_impl {
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
            self.nmats() * self.nrows() * self.ncols()
        }

        /// Returns whether the tensor is empty.
        #[inline(always)]
        pub fn is_empty(&self) -> bool {
            self.len() == 0
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

macro_rules! matvec_bounds_impl {
    () => {
        /// Checks that the indices are within bounds for this tensor.
        #[inline(always)]
        pub fn is_in_bounds(&self, m: usize, r: usize, c: usize) -> bool {
            m < self.nmats && r < self.nrows && c < self.ncols
        }

        /// Panic if the indices are out-of-bounds for this tensor.
        #[inline(always)]
        #[track_caller]
        pub fn panic_on_out_of_bounds(&self, m: usize, r: usize, c: usize) {
            if m >= self.nmats {
                panic!(
                    "Matrix index out of bounds: index = {}, which is >= {}.",
                    m, self.nmats
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

macro_rules! matvec_read_impl {
    ($ptr_access:ident) => {
        /// Reads an element from the tensor without bounds checking.
        ///
        /// # Arguments
        ///
        /// * `m` - The matrix index.
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
        /// use qudit_core::matrix::MatVec;
        /// let mut mat_vec = MatVec::<f64>::zeros(3, 2, 2);
        /// unsafe {
        ///    mat_vec.write_unchecked(0, 0, 0, 1.0);
        ///    assert_eq!(mat_vec.read_unchecked(0, 0, 0), 1.0);
        ///    mat_vec.write_unchecked(1, 0, 0, 2.0);
        ///    assert_eq!(mat_vec.read_unchecked(1, 0, 0), 2.0);
        ///    mat_vec.write_unchecked(1, 1, 1, 3.0);
        ///    assert_eq!(mat_vec.read_unchecked(1, 1, 1), 3.0);
        /// }
        /// ```
        ///
        /// # See Also
        ///
        /// - [read] method for a safe version of this method.
        #[inline(always)]
        #[track_caller]
        pub unsafe fn read_unchecked(&self, m: usize, r: usize, c: usize) -> C {
            *self.$ptr_access(m, r, c)
        }

        /// Reads an element from the tensor.
        ///
        /// # Arguments
        ///
        /// * `m` - The matrix index.
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
        /// use qudit_core::matrix::MatVec;
        /// let mut mat_vec = MatVec::<f64>::zeros(3, 2, 2);
        /// mat_vec.write(0, 0, 0, 1.0);
        /// assert_eq!(mat_vec.read(0, 0, 0), 1.0);
        /// mat_vec.write(1, 0, 0, 2.0);
        /// assert_eq!(mat_vec.read(1, 0, 0), 2.0);
        /// mat_vec.write(1, 1, 1, 3.0);
        /// assert_eq!(mat_vec.read(1, 1, 1), 3.0);
        /// ```
        ///
        /// # See Also
        ///
        /// - [read_unchecked] method for a version of this method without bounds checking.
        #[inline]
        #[track_caller]
        pub fn read(&self, m: usize, r: usize, c: usize) -> C {
            self.panic_on_out_of_bounds(m, r, c);
            // SAFETY: The bounds have been checked.
            unsafe { self.read_unchecked(m, r, c) }
        }
    };
}

macro_rules! matvec_write_impl {
    ($ptr_access:ident) => {
        /// Writes an element to the tensor without bounds checking.
        ///
        /// See the [read_unchecked] method for more information.
        ///
        /// # Arguments
        ///
        /// * `m` - The matrix index.
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
        pub unsafe fn write_unchecked(&mut self, m: usize, r: usize, c: usize, val: C) {
            *self.$ptr_access(m, r, c) = val;
        }

        /// Writes an element to the tensor.
        ///
        /// See the [read] method for more information.
        ///
        /// # Arguments
        ///
        /// * `m` - The matrix index.
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
        pub fn write(&mut self, m: usize, r: usize, c: usize, val: C) {
            self.panic_on_out_of_bounds(m, r, c);
            // SAFETY: The bounds have been checked.
            unsafe { self.write_unchecked(m, r, c, val) };
        }
    };
}

/// A 3-dimensional tensor with a matrix-vector storage layout.
///
/// The tensor is indexed by three indices: the matrix index, the row index,
/// and the column index. The row index is the fastest moving index, followed
/// by the column index, and finally the matrix index.
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
/// - [Mat] for a standard matrix type.
/// - [SymSqMatMat] for a symmetric square 4-dimensional tensor type.
/// - [MatVecRef] A const reference into a MatVec.
/// - [MatVecMut] A mutable reference to a MatVec.
pub struct MatVec<C: Memorable> {
    data: MemoryBuffer<C>,
    nrows: usize,
    ncols: usize,
    nmats: usize,
    col_stride: usize,
    mat_stride: usize,
}

impl<C: Memorable> MatVec<C> {
    /// Creates a new, zeroed matrix-vector tensor with the specified dimensions.
    ///
    /// # Arguments
    ///
    /// * `nrows` - The number of rows in each matrix.
    ///
    /// * `ncols` - The number of columns in each matrix.
    ///
    /// * `nmats` - The number of matrices in the tensor.
    ///
    /// # Returns
    ///
    /// A new, zeroed matrix-vector tensor with the specified dimensions.
    ///
    /// # Panics
    ///
    /// Panics if the number of elements in the tensor would overflow.
    ///
    /// # Examples
    ///
    /// ```
    /// use qudit_core::matrix::MatVec;
    /// let mat_vec = MatVec::<f64>::zeros(3, 2, 2);
    /// assert_eq!(mat_vec.nrows(), 3);
    /// assert_eq!(mat_vec.ncols(), 2);
    /// assert_eq!(mat_vec.nmats(), 2);
    /// ```
    ///
    /// # See Also
    ///
    /// - [read] method for reading elements from the tensor.
    /// - [write] method for writing elements to the tensor.
    pub fn zeros(nrows: usize, ncols: usize, nmats: usize) -> Self {
        let col_stride = calc_col_stride::<C>(nrows, ncols);
        let mat_stride = calc_mat_stride::<C>(nrows, ncols, col_stride);
        let mat_vec_size = nmats
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
    pub fn as_mut(&mut self) -> MatVecMut<'_, C> {
        unsafe {
            MatVecMut::from_raw_parts(
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
    pub fn as_ref(&self) -> MatVecRef<'_, C> {
        unsafe {
            MatVecRef::from_raw_parts(
                self.as_ptr(),
                self.nrows,
                self.ncols,
                self.nmats,
                self.col_stride,
                self.mat_stride,
            )
        }
    }

    /// Returns a const reference to a specific matrix in the tensor.
    ///
    /// # Arguments
    ///
    /// * `m` - The matrix index.
    #[inline(always)]
    pub fn mat_ref(&self, m: usize) -> MatRef<C> {
        let offset = m.wrapping_mul(self.mat_stride);
        unsafe {
            faer::MatRef::from_raw_parts(
                self.data.as_ptr().offset(offset as isize),
                self.nrows,
                self.ncols,
                1,
                self.col_stride().try_into().unwrap(),
            )
        }
    }

    /// Returns a mutable reference to a specific matrix in the tensor.
    ///
    /// # Arguments
    ///
    /// * `m` - The matrix index.
    pub fn mat_mut(&mut self, m: usize) -> MatMut<C> {
        let offset = m.wrapping_mul(self.mat_stride);
        unsafe {
            faer::MatMut::from_raw_parts_mut(
                self.data.as_mut_ptr().offset(offset as isize),
                self.nrows,
                self.ncols,
                1,
                self.col_stride.try_into().unwrap(),
            )
        }
    }

    matvec_prop_impl!();
    matvec_bounds_impl!();
    matvec_read_impl!(unchecked_ptr_at);
    matvec_write_impl!(unchecked_ptr_mut_at);
}

impl<C: std::fmt::Debug + Memorable> std::fmt::Debug for MatVec<C> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_list()
            .entries((0..self.nmats).map(|m| self.mat_ref(m)))
            .finish()
    }
}

impl<C: ComplexField + Memorable> PartialEq for MatVec<C> {
    fn eq(&self, other: &Self) -> bool {
        if self.nrows != other.nrows || self.ncols != other.ncols || self.nmats != other.nmats {
            return false;
        }

        for m in 0..self.nmats {
            for r in 0..self.nrows {
                for c in 0..self.ncols {
                    if self.read(m, r, c) != other.read(m, r, c) {
                        return false;
                    }
                }
            }
        }

        true
    }
}

impl<C: Memorable> Clone for MatVec<C> {
    fn clone(&self) -> Self {
        let mut mat_vec = MatVec::<C>::zeros(self.nrows, self.ncols, self.nmats);
        for m in 0..self.nmats {
            for r in 0..self.nrows {
                for c in 0..self.ncols {
                    mat_vec.write(m, r, c, self.read(m, r, c));
                }
            }
        }
        mat_vec
    }
}

impl<C: ComplexField + Memorable> std::ops::Mul<C> for MatVec<C> {
    type Output = MatVec<C>;

    fn mul(self, rhs: C) -> Self::Output {
        let mut mat_vec = MatVec::<C>::zeros(self.nrows, self.ncols, self.nmats);
        for m in 0..self.nmats {
            for r in 0..self.nrows {
                for c in 0..self.ncols {
                    mat_vec.write(m, r, c, self.read(m, r, c).mul_by_ref(&rhs));
                }
            }
        }
        mat_vec
    }
}

impl<C: ComplexField + Memorable> std::ops::MulAssign<C> for MatVec<C> {
    fn mul_assign(&mut self, rhs: C) {
        for m in 0..self.nmats {
            for r in 0..self.nrows {
                for c in 0..self.ncols {
                    self.write(m, r, c, self.read(m, r, c).mul_by_ref(&rhs));
                }
            }
        }
    }
}

impl<C: ComplexField + Memorable> std::ops::Sub for MatVec<C> {
    type Output = MatVec<C>;

    fn sub(self, rhs: Self) -> Self::Output {
        let mut mat_vec = MatVec::<C>::zeros(self.nrows, self.ncols, self.nmats);
        for m in 0..self.nmats {
            for r in 0..self.nrows {
                for c in 0..self.ncols {
                    mat_vec.write(m, r, c, self.read(m, r, c).sub_by_ref(&rhs.read(m, r, c)));
                }
            }
        }
        mat_vec
    }
}

use super::Mat;

impl<C: Memorable> Index<(usize, usize, usize)> for MatVec<C> {
    type Output = C;

    fn index(&self, idx: (usize, usize, usize)) -> &Self::Output {
        self.panic_on_out_of_bounds(idx.0, idx.1, idx.2);
        // SAFETY: The bounds have been checked.
        &self.data[idx.0 * self.mat_stride + idx.2 * self.col_stride + idx.1]
    }
}

impl<C: Memorable> IndexMut<(usize, usize, usize)> for MatVec<C> {
    fn index_mut(&mut self, idx: (usize, usize, usize)) -> &mut Self::Output {
        self.panic_on_out_of_bounds(idx.0, idx.1, idx.2);
        // SAFETY: The bounds have been checked.
        &mut self.data[idx.0 * self.mat_stride + idx.2 * self.col_stride + idx.1]
    }
}

impl<C: Memorable> FromIterator<Mat<C>> for MatVec<C> {
    fn from_iter<I: IntoIterator<Item = Mat<C>>>(iter: I) -> Self {
        let mats: Vec<Mat<C>> = iter.into_iter().collect();
        let nmats = mats.len();
        let nrows = mats[0].nrows();
        let ncols = mats[0].ncols();
        let mut mat_vec = MatVec::<C>::zeros(nrows, ncols, nmats);
        for m in 0..nmats {
            for r in 0..nrows {
                for c in 0..ncols {
                    mat_vec[(m, r, c)] = mats[m][(r, c)];
                }
            }
        }
        mat_vec
    }
}

/// A mutable reference to a 3-dimensional tensor with a matrix-vector layout.
///
/// See the [MatVec] type for more information.
pub struct MatVecMut<'a, C: Memorable> {
    data: NonNull<C>,
    nrows: usize,
    ncols: usize,
    nmats: usize,
    col_stride: usize,
    mat_stride: usize,
    __marker: std::marker::PhantomData<&'a C>,
}

impl<'a, C: Memorable> MatVecMut<'a, C> {
    /// Creates a `MatVecMut` from pointers to tensor data, dimensions, and strides.
    ///
    /// # Safety
    ///
    /// The caller must ensure:
    /// - The pointers are valid and non-null.
    /// - For each unit, the entire memory region addressed by the tensor is
    ///   within a single allocation.
    /// - The memory is accessible by the pointer.
    /// - No mutable aliasing occurs. No mutable or const references to the
    ///   tensor data exist when the `MatVecMut` is alive.
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

    /// Returns a mutable pointer to the tensor data.
    /// TODO: remove with faer update
    #[inline(always)]
    pub fn as_mut_ptr(self) -> NonNull<C> {
        self.data
    }

    /// Returns a const reference to a specific matrix in the tensor.
    ///
    /// # Arguments
    ///
    /// * `m` - The matrix index.
    #[inline(always)]
    pub fn mat_mut(&mut self, m: usize) -> MatMut<C> {
        let offset = m.wrapping_mul(self.mat_stride);
        unsafe {
            let ptr = self.data.as_ptr().offset(offset as isize);
            MatMut::from_raw_parts_mut(
                ptr,
                self.nrows,
                self.ncols,
                1,
                self.col_stride.try_into().unwrap(),
            )
        }
    }

    matvec_prop_impl!();
    matvec_bounds_impl!();
    matvec_read_impl!(unchecked_ptr_mut_at);
    matvec_write_impl!(unchecked_ptr_mut_at);
}

/// A const reference to a 3-dimensional tensor with a matrix-vector layout.
///
/// See the [MatVec] type for more information.
pub struct MatVecRef<'a, C: Memorable> {
    data: NonNull<C>,
    nrows: usize,
    ncols: usize,
    nmats: usize,
    col_stride: usize,
    mat_stride: usize,
    __marker: std::marker::PhantomData<&'a C>,
}

impl<'a, C: Memorable> MatVecRef<'a, C> {
    /// Creates a `MatVecRef` from pointers to tensor data, dimensions, and strides.
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
        nrows: usize,
        ncols: usize,
        nmats: usize,
        col_stride: usize,
        mat_stride: usize,
    ) -> Self {
        // SAFETY: The pointer is never used in an mutable context.
        let ptr = unsafe { NonNull::new_unchecked(data as *mut C) };

        Self {
            data: ptr,
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
    unsafe fn unchecked_ptr_at(&self, m: usize, r: usize, c: usize) -> *const C {
        let offset = m
            .wrapping_mul(self.mat_stride)
            .wrapping_add(c.wrapping_mul(self.col_stride))
            .wrapping_add(r);
        self.data.as_ptr().offset(offset as isize)
    }

    /// Returns a const reference to a specific matrix in the tensor.
    ///
    /// # Arguments
    ///
    /// * `m` - The matrix index.
    #[inline(always)]
    pub fn mat_ref(&self, m: usize) -> MatRef<C> {
        let offset = m.wrapping_mul(self.mat_stride);
        unsafe {
            let ptr = self.data.as_ptr().offset(offset as isize);
            MatRef::from_raw_parts(
                ptr,
                self.nrows,
                self.ncols,
                1,
                self.col_stride.try_into().unwrap(),
            )
        }
    }

    matvec_prop_impl!();
    matvec_bounds_impl!();
    matvec_read_impl!(unchecked_ptr_at);
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_mat_vec() {
        let mut mat_grad = MatVec::<f64>::zeros(3, 2, 2);
        assert_eq!(mat_grad.nrows, 3);
        assert_eq!(mat_grad.ncols, 2);
        assert_eq!(mat_grad.nmats, 2);

        mat_grad.write(0, 0, 0, 1.0);
        println!("mat_grad: {:?}", mat_grad.read(0, 0, 0));
    }
}
