use faer_core::{
    mat::{from_raw_parts, from_raw_parts_mut},
    MatMut, MatRef,
};

use crate::{
    math::{
        matrix::{MatGradMut, MatGradRef, MatHessMut, MatHessRef},
        ComplexScalar, Function,
    },
    sim::qvm::Memory,
    Gate, QuditSystem,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct MatrixBuffer {
    pub nrows: usize,
    pub ncols: usize,
    pub num_params: usize,
}

impl MatrixBuffer {
    pub fn size(&self) -> usize {
        self.nrows * self.ncols * self.num_params
    }
}

impl From<Gate> for MatrixBuffer {
    fn from(gate: Gate) -> Self {
        Self {
            nrows: gate.get_dimension(),
            ncols: gate.get_dimension(),
            num_params: gate.get_num_params(),
        }
    }
}

impl From<&Gate> for MatrixBuffer {
    fn from(gate: &Gate) -> Self {
        Self {
            nrows: gate.get_dimension(),
            ncols: gate.get_dimension(),
            num_params: gate.get_num_params(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct SizedMatrixBuffer {
    pub offset: usize,
    pub nrows: usize,
    pub ncols: usize,
    pub col_stride: isize,
    pub num_params: usize,
}

impl SizedMatrixBuffer {
    pub fn as_matmut<'a, C: ComplexScalar>(
        &self,
        memory: &mut Memory<C>,
    ) -> MatMut<'a, C> {
        unsafe {
            from_raw_parts_mut(
                C::faer_map(
                    C::faer_as_mut(memory),
                    #[inline(always)]
                    |mem| {
                        mem.as_ptr().offset(self.offset as isize)
                            as *mut C::Unit
                    },
                ),
                self.nrows,
                self.ncols,
                1,
                self.col_stride,
            )
        }
    }

    pub fn as_matref<'a, C: ComplexScalar>(
        &self,
        memory: &Memory<C>,
    ) -> MatRef<'a, C> {
        unsafe {
            from_raw_parts(
                C::faer_map(
                    C::faer_as_ref(memory),
                    #[inline(always)]
                    |mem| {
                        mem.as_ptr().offset(self.offset as isize)
                            as *const C::Unit
                    },
                ),
                self.nrows,
                self.ncols,
                1,
                self.col_stride,
            )
        }
    }

    pub fn as_matgradmut<'a, C: ComplexScalar>(
        &self,
        memory: &mut Memory<C>,
    ) -> MatGradMut<'a, C> {
        let mat_size = self.col_stride * self.ncols as isize;
        unsafe {
            MatGradMut::from_raw_parts(
                C::faer_map(
                    C::faer_as_mut(memory),
                    #[inline(always)]
                    |mem| {
                        mem.as_ptr().offset(self.offset as isize + mat_size)
                            as *mut C::Unit
                    },
                ),
                self.nrows,
                self.ncols,
                self.num_params,
                self.col_stride as usize,
            )
        }
    }

    pub fn as_matgradref<'a, C: ComplexScalar>(
        &self,
        memory: &Memory<C>,
    ) -> MatGradRef<'a, C> {
        let mat_size = self.col_stride * self.ncols as isize;
        unsafe {
            MatGradRef::from_raw_parts(
                C::faer_map(
                    C::faer_as_ref(memory),
                    #[inline(always)]
                    |mem| {
                        mem.as_ptr().offset(self.offset as isize + mat_size)
                            as *const C::Unit
                    },
                ),
                self.nrows,
                self.ncols,
                self.num_params,
                self.col_stride as usize,
            )
        }
    }

    pub fn as_mathessmut<'a, C: ComplexScalar>(
        &self,
        memory: &mut Memory<C>,
    ) -> MatHessMut<'a, C> {
        let mat_size = self.col_stride * self.ncols as isize;
        let grad_size = mat_size * self.num_params as isize;
        unsafe {
            MatHessMut::from_raw_parts(
                C::faer_map(
                    C::faer_as_mut(memory),
                    #[inline(always)]
                    |mem| {
                        mem.as_ptr()
                            .offset(self.offset as isize + mat_size + grad_size)
                            as *mut C::Unit
                    },
                ),
                self.nrows,
                self.ncols,
                self.num_params,
                self.col_stride as usize,
            )
        }
    }

    pub fn as_mathessref<'a, C: ComplexScalar>(
        &self,
        memory: &Memory<C>,
    ) -> MatHessRef<'a, C> {
        let mat_size = self.col_stride * self.ncols as isize;
        let grad_size = mat_size * self.num_params as isize;
        unsafe {
            MatHessRef::from_raw_parts(
                C::faer_map(
                    C::faer_as_ref(memory),
                    #[inline(always)]
                    |mem| {
                        mem.as_ptr()
                            .offset(self.offset as isize + mat_size + grad_size)
                            as *const C::Unit
                    },
                ),
                self.nrows,
                self.ncols,
                self.num_params,
                self.col_stride as usize,
            )
        }
    }
}
