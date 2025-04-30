use qudit_core::matrix::MatMut;
use qudit_core::matrix::MatRef;
use qudit_core::matrix::MatVecMut;
use qudit_core::matrix::MatVecRef;
use qudit_core::matrix::SymSqMatMatMut;
use qudit_core::matrix::SymSqMatMatRef;
use qudit_core::memory::MemoryBuffer;
use qudit_core::ComplexScalar;
use qudit_expr::UnitaryExpression;
use qudit_core::QuditSystem;
use qudit_core::HasParams;

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

impl From<UnitaryExpression> for MatrixBuffer {
    fn from(expr: UnitaryExpression) -> Self {
        Self {
            nrows: expr.dimension(),
            ncols: expr.dimension(),
            num_params: expr.num_params(),
        }
    }
}

impl From<&UnitaryExpression> for MatrixBuffer {
    fn from(expr: &UnitaryExpression) -> Self {
        Self {
            nrows: expr.dimension(),
            ncols: expr.dimension(),
            num_params: expr.num_params(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct SizedMatrixBuffer {
    pub offset: usize,
    pub nrows: usize,
    pub ncols: usize,
    pub col_stride: isize,
    pub mat_stride: isize,
    pub num_params: usize,
}

impl SizedMatrixBuffer {
    pub fn as_matmut<'a, C: ComplexScalar>(
        &self,
        memory: &mut MemoryBuffer<C>,
    ) -> MatMut<'a, C> {
        unsafe {
            faer::MatMut::from_raw_parts_mut(
                memory.as_mut_ptr().offset(self.offset as isize),
                self.nrows,
                self.ncols,
                1,
                self.col_stride.try_into().unwrap(),
            )
        }
    }

    pub fn as_matref<'a, C: ComplexScalar>(
        &self,
        memory: &MemoryBuffer<C>,
    ) -> MatRef<'a, C> {
        unsafe {
            faer::MatRef::from_raw_parts(
                memory.as_ptr().offset(self.offset as isize),
                self.nrows,
                self.ncols,
                1,
                self.col_stride,
            )
        }
    }

    pub fn as_matvecmut<'a, C: ComplexScalar>(
        &self,
        memory: &mut MemoryBuffer<C>,
    ) -> MatVecMut<'a, C> {
        let mat_size = self.col_stride * self.ncols as isize;
        unsafe {
            MatVecMut::from_raw_parts(
                memory.as_mut_ptr().offset(self.offset as isize + mat_size),
                self.nrows,
                self.ncols,
                self.num_params,
                self.col_stride as usize,
                self.mat_stride as usize,
            )
        }
    }

    pub fn as_matvecref<'a, C: ComplexScalar>(
        &self,
        memory: &MemoryBuffer<C>,
    ) -> MatVecRef<'a, C> {
        let mat_size = self.col_stride * self.ncols as isize;
        unsafe {
            MatVecRef::from_raw_parts(
                memory.as_ptr().offset(self.offset as isize + mat_size),
                self.nrows,
                self.ncols,
                self.num_params,
                self.col_stride as usize,
                self.mat_stride as usize,
            )
        }
    }

    pub fn as_symsqmatmut<'a, C: ComplexScalar>(
        &self,
        memory: &mut MemoryBuffer<C>,
    ) -> SymSqMatMatMut<'a, C> {
        let mat_size = self.col_stride * self.ncols as isize;
        let grad_size = mat_size * self.num_params as isize;
        unsafe {
            SymSqMatMatMut::from_raw_parts(
                memory.as_mut_ptr().offset(self.offset as isize + mat_size + grad_size),
                self.nrows,
                self.ncols,
                self.num_params,
                self.col_stride as usize,
                self.mat_stride as usize,
            )
        }
    }

    pub fn as_symsqmatref<'a, C: ComplexScalar>(
        &self,
        memory: &MemoryBuffer<C>,
    ) -> SymSqMatMatRef<'a, C> {
        let mat_size = self.col_stride * self.ncols as isize;
        let grad_size = mat_size * self.num_params as isize;
        unsafe {
            SymSqMatMatRef::from_raw_parts(
                memory.as_ptr().offset(self.offset as isize + mat_size + grad_size),
                self.nrows,
                self.ncols,
                self.num_params,
                self.col_stride as usize,
                self.mat_stride as usize,
            )
        }
    }
}
