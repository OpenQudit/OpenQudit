use faer::{MatMut, MatRef, RowMut, RowRef};
use qudit_core::array::SymSqTensorMut;
use qudit_core::array::SymSqTensorRef;
use qudit_core::array::TensorMut;
use qudit_core::array::TensorRef;
use qudit_core::memory::calc_col_stride;
use qudit_core::memory::calc_mat_stride;
use qudit_core::memory::calc_next_stride;
use qudit_core::{memory::MemoryBuffer, ComplexScalar};
use qudit_expr::{DifferentiationLevel, GenerationShape};
use qudit_expr::{FUNCTION, GRADIENT, HESSIAN};

use crate::bytecode::TensorBuffer;

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct SizedTensorBuffer<C: ComplexScalar> {
    offset: usize,
    shape: GenerationShape,
    ncols: usize,
    nrows: usize,
    nmats: usize,
    nparams: usize,
    col_stride: usize,
    row_stride: usize,
    mat_stride: usize,
    unit_stride: usize,
    _phantom: std::marker::PhantomData<C>,
}

impl<C: ComplexScalar> SizedTensorBuffer<C> {
    #[inline]
    pub fn contiguous(offset: usize, buffer: &TensorBuffer) -> Self {
        let row_stride = 1;
        let col_stride = buffer.nrows();
        let mat_stride = col_stride * buffer.ncols();
        let unit_stride = buffer.shape().num_elements();
        SizedTensorBuffer {
            offset,
            shape: buffer.shape(),
            ncols: buffer.ncols(),
            nrows: buffer.nrows(),
            nmats: buffer.nmats(),
            nparams: buffer.num_params(),
            col_stride,
            row_stride,
            mat_stride,
            unit_stride,
            _phantom: std::marker::PhantomData,
        }
    }

    #[inline]
    pub fn new(offset: usize, buffer: &TensorBuffer) -> Self {
        // let col_stride = 1;
        // let row_stride = buffer.ncols();
        // let mat_stride = buffer.ncols() * buffer.nmats();
        // let param_stride = mat_stride * buffer.nmats();
        let row_stride = 1;
        let col_stride = calc_col_stride::<C>(buffer.nrows(), buffer.ncols());
        let mat_stride = calc_mat_stride::<C>(buffer.nrows(), buffer.ncols(), col_stride);
        let unit_stride = match buffer.shape() {
            GenerationShape::Scalar => 1,
            GenerationShape::Vector(nelems) => calc_next_stride::<C>(nelems),
            GenerationShape::Matrix(_, _) => mat_stride,
            GenerationShape::Tensor3D(_, _, _) => {
                calc_next_stride::<C>(mat_stride * buffer.nmats())
            }
            _ => panic!("Tensor4D should not be constructed explicitly."),
        };
        SizedTensorBuffer {
            offset,
            shape: buffer.shape(),
            ncols: buffer.ncols(),
            nrows: buffer.nrows(),
            nmats: buffer.nmats(),
            nparams: buffer.num_params(),
            col_stride,
            row_stride,
            mat_stride,
            unit_stride,
            _phantom: std::marker::PhantomData,
        }
    }

    #[inline(always)]
    pub fn offset(&self) -> usize {
        self.offset
    }

    #[inline(always)]
    pub fn ncols(&self) -> usize {
        self.ncols
    }

    #[inline(always)]
    pub fn nrows(&self) -> usize {
        self.nrows
    }

    #[inline(always)]
    pub fn nmats(&self) -> usize {
        self.nmats
    }

    #[inline(always)]
    pub fn nparams(&self) -> usize {
        self.nparams
    }

    #[inline(always)]
    pub fn shape(&self) -> GenerationShape {
        self.shape
    }

    #[inline(always)]
    pub fn col_stride(&self) -> usize {
        self.col_stride
    }

    #[inline(always)]
    pub fn row_stride(&self) -> usize {
        self.row_stride
    }

    #[inline(always)]
    pub fn mat_stride(&self) -> usize {
        self.mat_stride
    }

    #[inline(always)]
    pub fn unit_stride(&self) -> usize {
        self.unit_stride
    }

    #[inline(always)]
    pub fn dims(&self) -> Vec<usize> {
        match self.shape() {
            GenerationShape::Scalar => vec![],
            GenerationShape::Vector(nelems) => vec![nelems],
            GenerationShape::Matrix(_, _) => vec![self.nrows, self.ncols],
            GenerationShape::Tensor3D(_, _, _) => vec![self.nmats, self.nrows, self.ncols],
            _ => panic!("Tensor4D should not be constructed explicitly."),
        }
    }

    #[inline(always)]
    pub fn strides(&self) -> Vec<usize> {
        match self.shape() {
            GenerationShape::Scalar => vec![],
            GenerationShape::Vector(_) => vec![1],
            GenerationShape::Matrix(_, _) => vec![self.row_stride, self.col_stride],
            GenerationShape::Tensor3D(_, _, _) => {
                vec![self.mat_stride, self.row_stride, self.col_stride]
            }
            _ => panic!("Tensor4D should not be constructed explicitly."),
        }
    }

    #[inline(always)]
    pub fn grad_strides(&self) -> Vec<usize> {
        match self.shape() {
            GenerationShape::Scalar => vec![self.unit_stride],
            GenerationShape::Vector(_) => vec![self.unit_stride, 1],
            GenerationShape::Matrix(_, _) => {
                vec![self.unit_stride, self.row_stride, self.col_stride]
            }
            GenerationShape::Tensor3D(_, _, _) => vec![
                self.unit_stride,
                self.mat_stride,
                self.row_stride,
                self.col_stride,
            ],
            _ => panic!("Tensor4D should not be constructed explicitly."),
        }
    }

    #[inline(always)]
    pub fn unit_size(&self) -> usize {
        self.shape.num_elements()
    }

    #[inline(always)]
    pub fn unit_memory_size(&self) -> usize {
        self.unit_stride
        // let (max_stride, dim) = self.strides().into_iter()
        //     .zip(self.dims().into_iter())
        //     .fold((1, 1), |a, b| a.max(b));
        // max_stride * dim
    }

    #[inline(always)]
    pub fn grad_size(&self) -> usize {
        self.unit_size() * (self.nparams)
    }

    #[inline(always)]
    pub fn grad_memory_size(&self) -> usize {
        self.unit_memory_size() * (self.nparams)
    }

    #[inline(always)]
    pub fn hess_size(&self) -> usize {
        self.unit_size() * (self.nparams * (self.nparams + 1) / 2)
    }

    #[inline(always)]
    pub fn hess_memory_size(&self) -> usize {
        self.unit_memory_size() * (self.nparams * (self.nparams + 1) / 2)
    }

    #[inline(always)]
    pub fn memory_size(&self, diff_lvl: DifferentiationLevel) -> usize {
        match diff_lvl {
            FUNCTION => self.unit_memory_size(),
            GRADIENT => self.unit_memory_size() + self.grad_memory_size(),
            HESSIAN => self.unit_memory_size() + self.grad_memory_size() + self.hess_memory_size(),
            _ => panic!("Invalid differentiation level."),
        }
    }

    #[inline(always)]
    pub fn size(&self, diff_lvl: DifferentiationLevel) -> usize {
        match diff_lvl {
            FUNCTION => self.unit_size(),
            GRADIENT => self.unit_size() + self.grad_size(),
            HESSIAN => self.unit_size() + self.grad_size() + self.hess_size(),
            _ => panic!("Invalid differentiation level."),
        }
    }

    #[inline(always)]
    pub unsafe fn as_ptr(&self, memory: &MemoryBuffer<C>) -> *const C { unsafe {
        memory.as_ptr().add(self.offset)
    }}

    #[inline(always)]
    pub unsafe fn as_ptr_mut(&self, memory: &mut MemoryBuffer<C>) -> *mut C { unsafe {
        memory.as_mut_ptr().add(self.offset)
    }}

    // ----- Buffers -----

    #[inline(always)]
    pub unsafe fn as_scalar_ref(&self, memory: &MemoryBuffer<C>) -> &C { unsafe {
        &*memory.as_ptr().add(self.offset)
    }}

    #[inline(always)]
    #[allow(clippy::mut_from_ref)] // mut comes from indexing memory which is mutable
    pub unsafe fn as_scalar_mut(&self, memory: &mut MemoryBuffer<C>) -> &mut C { unsafe {
        &mut *memory.as_mut_ptr().add(self.offset)
    }}

    #[inline(always)]
    pub unsafe fn as_vector_ref(&self, memory: &MemoryBuffer<C>) -> RowRef<'_, C> { unsafe {
        RowRef::from_raw_parts(memory.as_ptr().add(self.offset), self.ncols, 1isize)
    }}

    #[inline(always)]
    pub unsafe fn as_vector_mut(&self, memory: &mut MemoryBuffer<C>) -> RowMut<'_, C> { unsafe {
        RowMut::from_raw_parts_mut(memory.as_mut_ptr().add(self.offset), self.ncols, 1isize)
    }}

    #[inline(always)]
    pub unsafe fn as_matrix_ref(&self, memory: &MemoryBuffer<C>) -> MatRef<'_, C> { unsafe {
        MatRef::from_raw_parts(
            memory.as_ptr().add(self.offset),
            self.nrows,
            self.ncols,
            self.row_stride as isize,
            self.col_stride as isize,
        )
    }}

    #[inline(always)]
    pub unsafe fn as_matrix_mut(&self, memory: &mut MemoryBuffer<C>) -> MatMut<'_, C> { unsafe {
        MatMut::from_raw_parts_mut(
            memory.as_mut_ptr().add(self.offset),
            self.nrows,
            self.ncols,
            self.row_stride as isize,
            self.col_stride as isize,
        )
    }}

    #[inline(always)]
    pub unsafe fn as_tensor3d_ref(&self, memory: &MemoryBuffer<C>) -> TensorRef<'_, C, 3> { unsafe {
        TensorRef::from_raw_parts(
            memory.as_ptr().add(self.offset),
            [self.nmats, self.nrows, self.ncols],
            [self.mat_stride, self.row_stride, self.col_stride],
        )
    }}

    #[inline(always)]
    pub unsafe fn as_tensor3d_mut(&self, memory: &mut MemoryBuffer<C>) -> TensorMut<'_, C, 3> { unsafe {
        TensorMut::from_raw_parts(
            memory.as_mut_ptr().add(self.offset),
            [self.nmats, self.nrows, self.ncols],
            [self.mat_stride, self.row_stride, self.col_stride],
        )
    }}

    // ----- Gradient -----

    #[inline(always)]
    pub unsafe fn grad_as_vector_ref(&self, memory: &MemoryBuffer<C>) -> RowRef<'_, C> { unsafe {
        RowRef::from_raw_parts(
            memory.as_ptr().add(self.offset + self.unit_memory_size()),
            self.nparams,
            self.unit_stride as isize,
        )
    }}

    #[inline(always)]
    pub unsafe fn grad_as_vector_mut(&self, memory: &mut MemoryBuffer<C>) -> RowMut<'_, C> { unsafe {
        RowMut::from_raw_parts_mut(
            memory
                .as_mut_ptr()
                .add(self.offset + self.unit_memory_size()),
            self.nparams,
            self.unit_stride as isize,
        )
    }}

    #[inline(always)]
    pub unsafe fn grad_as_matrix_ref(&self, memory: &MemoryBuffer<C>) -> MatRef<'_, C> { unsafe {
        MatRef::from_raw_parts(
            memory.as_ptr().add(self.offset + self.unit_memory_size()),
            self.nparams,
            self.dims()[0], // TODO: remove allocation safely
            self.unit_stride as isize,
            1_isize,
        )
    }}

    #[inline(always)]
    pub unsafe fn grad_as_matrix_mut(&self, memory: &mut MemoryBuffer<C>) -> MatMut<'_, C> { unsafe {
        MatMut::from_raw_parts_mut(
            memory
                .as_mut_ptr()
                .add(self.offset + self.unit_memory_size()),
            self.nparams,
            self.dims()[0], // TODO: remove allocation safely
            self.unit_stride as isize,
            1_isize,
        )
    }}

    #[inline(always)]
    pub unsafe fn grad_as_tensor3d_ref(&self, memory: &MemoryBuffer<C>) -> TensorRef<'_, C, 3> { unsafe {
        TensorRef::from_raw_parts(
            memory.as_ptr().add(self.offset + self.unit_memory_size()),
            [self.nparams, self.nrows, self.ncols],
            [self.unit_stride, self.row_stride, self.col_stride],
        )
    }}

    #[inline(always)]
    pub unsafe fn grad_as_tensor3d_mut(&self, memory: &mut MemoryBuffer<C>) -> TensorMut<'_, C, 3> { unsafe {
        TensorMut::from_raw_parts(
            memory
                .as_mut_ptr()
                .add(self.offset + self.unit_memory_size()),
            [self.nparams, self.nrows, self.ncols],
            [self.unit_stride, self.row_stride, self.col_stride],
        )
    }}

    #[inline(always)]
    pub unsafe fn grad_as_tensor4d_ref(&self, memory: &MemoryBuffer<C>) -> TensorRef<'_, C, 4> { unsafe {
        TensorRef::from_raw_parts(
            memory.as_ptr().add(self.offset + self.unit_memory_size()),
            [self.nparams, self.nmats, self.nrows, self.ncols],
            [
                self.unit_stride,
                self.mat_stride,
                self.row_stride,
                self.col_stride,
            ],
        )
    }}

    #[inline(always)]
    pub unsafe fn grad_as_tensor4d_mut(&self, memory: &mut MemoryBuffer<C>) -> TensorMut<'_, C, 4> { unsafe {
        TensorMut::from_raw_parts(
            memory
                .as_mut_ptr()
                .add(self.offset + self.unit_memory_size()),
            [self.nparams, self.nmats, self.nrows, self.ncols],
            [
                self.unit_stride,
                self.mat_stride,
                self.row_stride,
                self.col_stride,
            ],
        )
    }}

    // ----- Hessian -----

    #[inline(always)]
    pub unsafe fn hess_as_symsq_matrix_ref(
        &self,
        memory: &MemoryBuffer<C>,
    ) -> SymSqTensorRef<'_, C, 2> { unsafe {
        SymSqTensorRef::from_raw_parts(
            memory
                .as_ptr()
                .add(self.offset + self.unit_memory_size() + self.grad_memory_size()),
            [self.nparams, self.nparams],
            [self.nparams * self.unit_stride, self.unit_stride],
        )
    }}

    #[inline(always)]
    pub unsafe fn hess_as_symsq_matrix_mut(
        &self,
        memory: &mut MemoryBuffer<C>,
    ) -> SymSqTensorMut<'_, C, 2> { unsafe {
        SymSqTensorMut::from_raw_parts(
            memory
                .as_mut_ptr()
                .add(self.offset + self.unit_memory_size() + self.grad_memory_size()),
            [self.nparams, self.nparams],
            [self.nparams * self.unit_stride, self.unit_stride],
        )
    }}

    #[inline(always)]
    pub unsafe fn hess_as_symsq_tensor3d_ref(
        &self,
        memory: &MemoryBuffer<C>,
    ) -> SymSqTensorRef<'_, C, 3> { unsafe {
        SymSqTensorRef::from_raw_parts(
            memory
                .as_ptr()
                .add(self.offset + self.unit_memory_size() + self.grad_memory_size()),
            [self.nparams, self.nparams, self.ncols],
            [self.nparams * self.unit_stride, self.unit_stride, 1],
        )
    }}

    #[inline(always)]
    pub unsafe fn hess_as_symsq_tensor3d_mut(
        &self,
        memory: &mut MemoryBuffer<C>,
    ) -> SymSqTensorMut<'_, C, 3> { unsafe {
        SymSqTensorMut::from_raw_parts(
            memory
                .as_mut_ptr()
                .add(self.offset + self.unit_memory_size() + self.grad_memory_size()),
            [self.nparams, self.nparams, self.ncols],
            [self.nparams * self.unit_stride, self.unit_stride, 1],
        )
    }}

    #[inline(always)]
    pub unsafe fn hess_as_symsq_tensor4d_ref(
        &self,
        memory: &MemoryBuffer<C>,
    ) -> SymSqTensorRef<'_, C, 4> { unsafe {
        SymSqTensorRef::from_raw_parts(
            memory
                .as_ptr()
                .add(self.offset + self.unit_memory_size() + self.grad_memory_size()),
            [self.nparams, self.nparams, self.nrows, self.ncols],
            [
                self.nparams * self.unit_stride,
                self.unit_stride,
                self.row_stride,
                self.col_stride,
            ],
        )
    }}

    #[inline(always)]
    pub unsafe fn hess_as_symsq_tensor4d_mut(
        &self,
        memory: &mut MemoryBuffer<C>,
    ) -> SymSqTensorMut<'_, C, 4> { unsafe {
        SymSqTensorMut::from_raw_parts(
            memory
                .as_mut_ptr()
                .add(self.offset + self.unit_memory_size() + self.grad_memory_size()),
            [self.nparams, self.nparams, self.nrows, self.ncols],
            [
                self.nparams * self.unit_stride,
                self.unit_stride,
                self.row_stride,
                self.col_stride,
            ],
        )
    }}

    #[inline(always)]
    pub unsafe fn hess_as_symsq_tensor5d_ref(
        &self,
        memory: &MemoryBuffer<C>,
    ) -> SymSqTensorRef<'_, C, 5> { unsafe {
        SymSqTensorRef::from_raw_parts(
            memory
                .as_ptr()
                .add(self.offset + self.unit_memory_size() + self.grad_memory_size()),
            [
                self.nparams,
                self.nparams,
                self.nmats,
                self.nrows,
                self.ncols,
            ],
            [
                self.nparams * self.unit_stride,
                self.unit_stride,
                self.mat_stride,
                self.row_stride,
                self.col_stride,
            ],
        )
    }}

    #[inline(always)]
    pub unsafe fn hess_as_symsq_tensor5d_mut(
        &self,
        memory: &mut MemoryBuffer<C>,
    ) -> SymSqTensorMut<'_, C, 5> { unsafe {
        SymSqTensorMut::from_raw_parts(
            memory
                .as_mut_ptr()
                .add(self.offset + self.unit_memory_size() + self.grad_memory_size()),
            [
                self.nparams,
                self.nparams,
                self.nmats,
                self.nrows,
                self.ncols,
            ],
            [
                self.nparams * self.unit_stride,
                self.unit_stride,
                self.mat_stride,
                self.row_stride,
                self.col_stride,
            ],
        )
    }}
}
