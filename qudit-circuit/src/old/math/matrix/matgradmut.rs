use std::ptr::NonNull;

use faer_core::mat;
use faer_core::MatMut;
use faer_entity::Entity;
use faer_entity::GroupFor;

pub type MemoryPtr<C: Entity> = GroupFor<C, NonNull<C::Unit>>;

pub struct MatGradMut<'a, C: Entity> {
    data: MemoryPtr<C>,
    nrows: usize,
    ncols: usize,
    num_params: usize,
    col_stride: usize,
    mat_stride: usize,
    __marker: std::marker::PhantomData<&'a C>,
}

impl<'a, C: Entity> MatGradMut<'a, C> {
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

    pub fn read(&self, p: usize, r: usize, c: usize) -> C {
        let offset = p * self.mat_stride + c * self.col_stride + r;
        C::faer_from_units(C::faer_map(
            C::faer_as_ref(&self.data),
            |ptr| unsafe { *ptr.as_ptr().offset(offset as isize) },
        ))
    }

    pub fn write(&mut self, p: usize, r: usize, c: usize, val: C) {
        let offset = p * self.mat_stride + c * self.col_stride + r;
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

    pub fn get_matmut(&mut self, p: usize) -> MatMut<'_, C> {
        let offset = p * self.mat_stride;
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
}
