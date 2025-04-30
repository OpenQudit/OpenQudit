use std::mem::size_of;

use aligned_vec::avec;
use aligned_vec::AVec;
use aligned_vec::CACHELINE_ALIGN;
use bytemuck::Zeroable;
use faer_entity::Entity;
use faer_entity::GroupFor;

use crate::math::matrix::MatGradMut;

pub type Memory<C: Entity> = GroupFor<C, AVec<C::Unit>>;

pub fn alloc_memory<C: Entity>(size: usize) -> Memory<C> {
    C::faer_map(C::UNIT, |()| avec![<C::Unit as Zeroable>::zeroed(); size])
}

pub struct MatGrad<C: Entity> {
    data: Memory<C>,
    nrows: usize,
    ncols: usize,
    num_params: usize,
    col_stride: usize,
    mat_stride: usize,
}

impl<C: Entity> MatGrad<C> {
    pub fn get_col_stride(nrows: usize, _ncols: usize) -> isize {
        // TODO: Memory and stuff like umm... cache lines... experiments
        let nrows = nrows as isize;
        let remainder =
            nrows * size_of::<C::Unit>() as isize % CACHELINE_ALIGN as isize;
        nrows + remainder
    }

    pub fn zeros(num_params: usize, nrows: usize, ncols: usize) -> Self {
        // TODO: Validate inputs and memory calculations for overflow
        // TODO: check for zero size
        let col_stride = Self::get_col_stride(nrows, ncols) as usize;
        let mat_stride = col_stride * ncols;
        let grad_size = mat_stride * num_params;
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
            |(uval, ptr)| {
                ptr[offset] = uval;
            },
        );
    }

    pub fn as_mut(&mut self) -> MatGradMut<'_, C> {
        unsafe {
            MatGradMut::from_raw_parts(
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
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_mat_grad() {
        let mut mat_grad = MatGrad::<f64>::zeros(3, 2, 2);
        assert_eq!(mat_grad.nrows, 2);
        assert_eq!(mat_grad.ncols, 2);
        assert_eq!(mat_grad.num_params, 3);

        mat_grad.write(0, 0, 0, 1.0);
        println!("mat_grad: {:?}", mat_grad.read(0, 0, 0));
    }
}
