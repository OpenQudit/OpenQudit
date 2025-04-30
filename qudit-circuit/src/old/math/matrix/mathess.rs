use std::fmt;
use std::fmt::Formatter;
use std::mem::size_of;

use aligned_vec::avec;
use aligned_vec::AVec;
use aligned_vec::CACHELINE_ALIGN;
use bytemuck::Zeroable;
use faer_core::mat;
use faer_core::MatRef;
use faer_entity::Entity;
use faer_entity::GroupFor;
use num_traits::ToPrimitive;

use super::MatHessMut;

pub type Memory<C: Entity> = GroupFor<C, AVec<C::Unit>>;

pub fn alloc_memory<C: Entity>(size: usize) -> Memory<C> {
    C::faer_map(C::UNIT, |()| avec![<C::Unit as Zeroable>::zeroed(); size])
}

pub struct MatHess<C: Entity> {
    data: Memory<C>,
    nrows: usize,
    ncols: usize,
    num_params: usize,
    col_stride: usize,
    mat_stride: usize,
}

impl<C: Entity> PartialEq for MatHess<C> {
    fn eq(&self, other: &Self) -> bool {
        if self.nrows != other.nrows
            || self.ncols != other.ncols
            || self.num_params != other.num_params
            || self.col_stride != other.col_stride
            || self.mat_stride != other.mat_stride
        {
            return false;
        }

        for p1 in 0..self.num_params {
            for p2 in p1..self.num_params {
                for i in 0..self.nrows {
                    for j in 0..self.ncols {
                        if self.read(p1, p2, i, j) != other.read(p1, p2, i, j) {
                            return false;
                        }
                    }
                }
            }
        }

        true
    }
}

impl<C: Entity> std::fmt::Debug for MatHess<C> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "MatHess {{ nrows: {}, ncols: {}, num_params: {}, col_stride: {}, mat_stride: {} }}", self.nrows, self.ncols, self.num_params, self.col_stride, self.mat_stride)
    }
}

impl<C: Entity> MatHess<C> {
    pub fn get_col_stride(nrows: usize, _ncols: usize) -> isize {
        // TODO: Memory and stuff like umm... cache lines... experiments
        let nrows = nrows as isize;
        let remainder =
            nrows * size_of::<C::Unit>() as isize % CACHELINE_ALIGN as isize;
        nrows + remainder
    }

    pub fn num_params(&self) -> usize {
        self.num_params
    }

    pub fn nrows(&self) -> usize {
        self.nrows
    }

    pub fn ncols(&self) -> usize {
        self.ncols
    }

    pub fn zeros(num_params: usize, nrows: usize, ncols: usize) -> Self {
        // TODO: Validate inputs and memory calculations for overflow
        // TODO: check for zero size
        let col_stride = Self::get_col_stride(nrows, ncols) as usize;
        let mat_stride = col_stride * ncols;
        let grad_size = mat_stride * (num_params * (num_params + 1) / 2);
        let grad_mem_unit_size = grad_size * size_of::<C::Unit>();

        let data = alloc_memory::<C>(grad_mem_unit_size);

        Self {
            data,
            nrows,
            ncols,
            col_stride,
            mat_stride,
            num_params,
        }
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
            |(uval, ptr)| {
                ptr[offset] = uval;
            },
        );
    }

    pub fn as_mut(&mut self) -> MatHessMut<'_, C> {
        unsafe {
            MatHessMut::from_raw_parts(
                C::faer_map(C::faer_as_mut(&mut self.data), |ptr| {
                    ptr.as_mut_ptr()
                }),
                self.nrows,
                self.ncols,
                self.num_params,
                self.col_stride,
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
