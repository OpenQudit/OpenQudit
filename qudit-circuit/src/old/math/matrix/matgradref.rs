use std::ptr::NonNull;

use faer_core::mat;
use faer_core::MatRef;
use faer_entity::Entity;
use faer_entity::GroupFor;

// TODO: Change GroupFor to GroupCopyFor
pub type MemoryPtr<C: Entity> = GroupFor<C, NonNull<C::Unit>>;

pub struct MatGradRef<'a, C: Entity> {
    data: MemoryPtr<C>,
    nrows: usize,
    ncols: usize,
    num_params: usize,
    col_stride: usize,
    mat_stride: usize,
    __marker: std::marker::PhantomData<&'a C>,
}

impl<'a, C: Entity> MatGradRef<'a, C> {
    pub unsafe fn from_raw_parts(
        data: GroupFor<C, *const C::Unit>,
        nrows: usize,
        ncols: usize,
        num_params: usize,
        col_stride: usize,
    ) -> Self {
        let mat_stride = col_stride * ncols;
        Self {
            data: C::faer_map(data, |ptr| {
                NonNull::new_unchecked(ptr as *mut C::Unit)
            }),
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

    pub fn read(&self, p: usize, r: usize, c: usize) -> C {
        let offset = p * self.mat_stride + c * self.col_stride + r;
        C::faer_from_units(C::faer_map(
            C::faer_as_ref(&self.data),
            |ptr| unsafe { *ptr.as_ptr().offset(offset as isize) },
        ))
    }

    pub fn get_matref(&self, p: usize) -> MatRef<'_, C> {
        let offset = p * self.mat_stride;
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
