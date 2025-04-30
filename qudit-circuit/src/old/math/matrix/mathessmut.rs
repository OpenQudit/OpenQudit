use std::ptr::NonNull;

use faer_core::mat;
use faer_core::MatMut;
use faer_core::MatRef;
use faer_entity::Entity;
use faer_entity::GroupFor;
use num_traits::ToPrimitive;

pub type MemoryPtr<C: Entity> = GroupFor<C, NonNull<C::Unit>>;

pub struct MatHessMut<'a, C: Entity> {
    data: MemoryPtr<C>,
    nrows: usize,
    ncols: usize,
    num_params: usize,
    col_stride: usize,
    mat_stride: usize,
    __marker: std::marker::PhantomData<&'a C>,
}

impl<'a, C: Entity> MatHessMut<'a, C> {
    pub unsafe fn from_raw_parts(
        data: GroupFor<C, *mut C::Unit>,
        nrows: usize,
        ncols: usize,
        num_params: usize,
        col_stride: usize,
    ) -> Self {
        let mat_stride = col_stride * ncols;
        Self {
            data: C::faer_map(data, |ptr| NonNull::new_unchecked(ptr)),
            nrows,
            ncols,
            num_params,
            col_stride,
            mat_stride,
            __marker: std::marker::PhantomData,
        }
    }

    #[inline(always)]
    pub fn num_params(&self) -> usize {
        self.num_params
    }

    // Since the Hessian is stored in compact form as column-major,
    // we first find j by solving for the smallest N such that
    // N(N+1)/2 <= k. We can then just undo the equation for k in
    // coords to index.
    pub fn index_to_coords(&self, index: usize) -> (usize, usize) {
        // TODO: check if index is in bounds
        let j = (((8 * index + 1) as f64).sqrt().floor().to_usize().unwrap()
            - 1)
            / 2;
        let i = index - j * (j + 1) / 2;
        (i, j)
    }

    /// When storing the upper triangular part of a matrix (including the
    /// diagonal) into a compact vector, you essentially flatten the
    /// upper triangular part of the matrix column-wise into a one-dimensional
    /// array. Let's say you have an N*N matrix and a compact vector V of
    /// length N(N+1)/2 to store the upper triangular part of the matrix.
    /// For a matrix coordinate (i,j) in the upper triangular part
    /// where i<=j, the corresponding vector index k can be calculated
    /// using the formula:
    /// ```math
    ///     k = j * (j+1) / 2 + i
    /// ```
    pub fn coords_to_index(&self, coords: (usize, usize)) -> usize {
        // TODO: check if coords is in bounds
        let (i, j) = coords;
        if i <= j {
            j * (j + 1) / 2 + i
        } else {
            i * (i + 1) / 2 + j
        }
    }

    pub fn read(&self, p2: usize, p1: usize, r: usize, c: usize) -> C {
        let index = self.coords_to_index((p2, p1));
        let offset = index * self.mat_stride + c * self.col_stride + r;
        C::faer_from_units(C::faer_map(
            C::faer_as_ref(&self.data),
            |ptr| unsafe { *ptr.as_ptr().offset(offset as isize) },
        ))
    }

    pub fn write(&mut self, p2: usize, p1: usize, r: usize, c: usize, val: C) {
        let index = self.coords_to_index((p2, p1));
        let offset = index * self.mat_stride + c * self.col_stride + r;
        C::faer_map(
            C::faer_zip(
                C::faer_into_units(val),
                C::faer_as_mut(&mut self.data),
            ),
            |(uval, ptr)| unsafe {
                *(ptr.as_ptr().offset(offset as isize)) = uval
            },
        );
    }

    pub fn get_matmut(&mut self, p2: usize, p1: usize) -> MatMut<'_, C> {
        let index = self.coords_to_index((p2, p1));
        let offset = index * self.mat_stride;
        unsafe {
            mat::from_raw_parts_mut(
                C::faer_map(C::faer_as_mut(&mut self.data), |ptr| {
                    ptr.as_ptr().offset(offset as isize)
                }),
                self.nrows,
                self.ncols,
                1,
                self.col_stride as isize,
            )
        }
    }

    pub fn get_matref(&self, p2: usize, p1: usize) -> MatRef<'_, C> {
        let index = self.coords_to_index((p2, p1));
        let offset = index * self.mat_stride;
        unsafe {
            mat::from_raw_parts(
                C::faer_map(C::faer_as_ref(&self.data), |ptr| {
                    (ptr.as_ptr() as *const C::Unit).offset(offset as isize)
                }),
                self.nrows,
                self.ncols,
                1,
                self.col_stride as isize,
            )
        }
    }
}
