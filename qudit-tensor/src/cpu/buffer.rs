use faer::{RowMut, RowRef, MatRef, MatMut};
use qudit_core::array::TensorRef;
use qudit_core::array::TensorMut;
use qudit_core::array::SymSqTensorRef;
use qudit_core::array::SymSqTensorMut;
use qudit_core::memory;
use qudit_core::memory::calc_col_stride;
use qudit_core::memory::calc_mat_stride;
use qudit_core::memory::calc_next_stride;
use qudit_core::{memory::MemoryBuffer, ComplexScalar};
use qudit_expr::{FUNCTION, GRADIENT, HESSIAN};
use qudit_expr::{DifferentiationLevel, GenerationShape};

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
    param_stride: usize,
    _phantom: std::marker::PhantomData<C>,
}

impl<C: ComplexScalar> SizedTensorBuffer<C> {

    #[inline]
    pub fn new(offset: usize, buffer: &TensorBuffer) -> Self {
        let col_stride = calc_col_stride::<C>(buffer.nrows(), buffer.ncols());
        let mat_stride = calc_mat_stride::<C>(buffer.nrows(), buffer.ncols(), col_stride);
        let param_stride = calc_next_stride::<C>(mat_stride*buffer.nmats());
        SizedTensorBuffer {
            offset,
            shape: buffer.shape(),
            ncols: buffer.ncols(),
            nrows: buffer.nrows(),
            nmats: buffer.nmats(),
            nparams: buffer.num_params(),
            col_stride,
            row_stride: 1,
            mat_stride,
            param_stride,
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
    pub fn strides(&self) -> Vec<usize> {
        match self.shape() {
            GenerationShape::Scalar => vec![],
            GenerationShape::Vector(_) => vec![self.col_stride], // TODO: Should this be 1??
            GenerationShape::Matrix(_, _) => vec![self.row_stride, self.col_stride],
            GenerationShape::Tensor3D(_, _, _) => vec![self.mat_stride, self.row_stride, self.col_stride],
            _ => panic!("Tensor4D should not be constructed explicitly."),
        }
    }

    #[inline(always)]
    pub fn grad_strides(&self) -> Vec<usize> {
        match self.shape() {
            GenerationShape::Scalar => vec![self.param_stride],// TODO: Should this be 1??
            GenerationShape::Vector(_) => vec![self.param_stride, self.col_stride], // TODO: Should this be 1??
            GenerationShape::Matrix(_, _) => vec![self.param_stride, self.row_stride, self.col_stride],
            GenerationShape::Tensor3D(_, _, _) => vec![self.param_stride, self.mat_stride, self.row_stride, self.col_stride],
            _ => panic!("Tensor4D should not be constructed explicitly."),
        }
    }

    #[inline(always)]
    pub fn unit_size(&self) -> usize {
        self.ncols * self.nrows * self.nmats
    }

    #[inline(always)]
    pub fn unit_memory_size(&self) -> usize {
        let max_stride = self.strides().iter().fold(1, |a, b| a.max(*b));
        match self.shape() {
            GenerationShape::Scalar => 1,
            GenerationShape::Vector(_) => self.ncols * max_stride,
            GenerationShape::Matrix(_, _) => self.nrows * max_stride,
            GenerationShape::Tensor3D(_, _, _) => self.nmats * max_stride,
            _ => panic!("Tensor4D should not be constructed explicitly."),
        }
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
    pub unsafe fn as_ptr(&self, memory: &MemoryBuffer<C>) -> *const C {
        memory.as_ptr().add(self.offset)
    }

    #[inline(always)]
    pub unsafe fn as_ptr_mut(&self, memory: &mut MemoryBuffer<C>) -> *mut C {
        memory.as_mut_ptr().add(self.offset)
    }

    // ----- Buffers -----
   
    #[inline(always)]
    pub unsafe fn as_scalar_ref(&self, memory: &MemoryBuffer<C>) -> &C {
        &*memory.as_ptr().add(self.offset)
    }

    #[inline(always)]
    pub unsafe fn as_scalar_mut(&self, memory: &mut MemoryBuffer<C>) -> &mut C {
        &mut *memory.as_mut_ptr().add(self.offset)
    }

    #[inline(always)]
    pub unsafe fn as_vector_ref(&self, memory: &MemoryBuffer<C>) -> RowRef<C> {
        RowRef::from_raw_parts(
            memory.as_ptr().add(self.offset),
            self.ncols,
            self.col_stride as isize, // TODO: Should this (and similar others) be 1?
        )
    }

    #[inline(always)]
    pub unsafe fn as_vector_mut(&self, memory: &mut MemoryBuffer<C>) -> RowMut<C> {
        RowMut::from_raw_parts_mut(
            memory.as_mut_ptr().add(self.offset),
            self.ncols,
            self.col_stride as isize,
        )
    }

    #[inline(always)]
    pub unsafe fn as_matrix_ref(&self, memory: &MemoryBuffer<C>) -> MatRef<C> {
        MatRef::from_raw_parts(
            memory.as_ptr().add(self.offset),
            self.nrows,
            self.ncols,
            self.row_stride as isize,
            self.col_stride as isize,
        )
    }

    #[inline(always)]
    pub unsafe fn as_matrix_mut(&self, memory: &mut MemoryBuffer<C>) -> MatMut<C> {
        MatMut::from_raw_parts_mut(
            memory.as_mut_ptr().add(self.offset),
            self.nrows,
            self.ncols,
            self.row_stride as isize,
            self.col_stride as isize,
        )
    }

    #[inline(always)]
    pub unsafe fn as_tensor3d_ref(&self, memory: &MemoryBuffer<C>) -> TensorRef<C, 3> {
        TensorRef::from_raw_parts(
            memory.as_ptr().add(self.offset),
            [self.nmats, self.nrows, self.ncols],
            [self.mat_stride, self.row_stride, self.col_stride],
        )
    }

    #[inline(always)]
    pub unsafe fn as_tensor3d_mut(&self, memory: &mut MemoryBuffer<C>) -> TensorMut<C, 3> {
        TensorMut::from_raw_parts(
            memory.as_mut_ptr().add(self.offset),
            [self.nmats, self.nrows, self.ncols],
            [self.mat_stride, self.row_stride, self.col_stride],
        )
    }
 
    // ----- Gradient -----

    #[inline(always)]
    pub unsafe fn grad_as_vector_ref(&self, memory: &MemoryBuffer<C>) -> RowRef<C> {
        RowRef::from_raw_parts(
            memory.as_ptr().add(self.offset + self.unit_size()),
            self.nparams,
            self.param_stride as isize,
        )
    }

    #[inline(always)]
    pub unsafe fn grad_as_vector_mut(&self, memory: &mut MemoryBuffer<C>) -> RowMut<C> {
        RowMut::from_raw_parts_mut(
            memory.as_mut_ptr().add(self.offset + self.unit_size()),
            self.nparams,
            self.param_stride as isize,
        )
    }

    #[inline(always)]
    pub unsafe fn grad_as_matrix_ref(&self, memory: &MemoryBuffer<C>) -> MatRef<C> {
        MatRef::from_raw_parts(
            memory.as_ptr().add(self.offset + self.unit_size()),
            self.nparams,
            self.ncols,
            self.param_stride as isize,
            self.col_stride as isize,
        )
    }

    #[inline(always)]
    pub unsafe fn grad_as_matrix_mut(&self, memory: &mut MemoryBuffer<C>) -> MatMut<C> {
        MatMut::from_raw_parts_mut(
            memory.as_mut_ptr().add(self.offset + self.unit_size()),
            self.nparams,
            self.ncols,
            self.param_stride as isize,
            self.col_stride as isize,
        )
    }

    #[inline(always)]
    pub unsafe fn grad_as_tensor3d_ref(&self, memory: &MemoryBuffer<C>) -> TensorRef<C, 3> {
        TensorRef::from_raw_parts(
            memory.as_ptr().add(self.offset + self.unit_size()),
            [self.nparams, self.nrows, self.ncols],
            [self.param_stride, self.row_stride, self.col_stride],
        )
    }

    #[inline(always)]
    pub unsafe fn grad_as_tensor3d_mut(&self, memory: &mut MemoryBuffer<C>) -> TensorMut<C, 3> {
        TensorMut::from_raw_parts(
            memory.as_mut_ptr().add(self.offset + self.unit_size()),
            [self.nparams, self.nrows, self.ncols],
            [self.param_stride, self.row_stride, self.col_stride],
        )
    }

    #[inline(always)]
    pub unsafe fn grad_as_tensor4d_ref(&self, memory: &MemoryBuffer<C>) -> TensorRef<C, 4> {
        TensorRef::from_raw_parts(
            memory.as_ptr().add(self.offset + self.unit_size()),
            [self.nparams, self.nmats, self.nrows, self.ncols],
            [self.param_stride, self.mat_stride, self.row_stride, self.col_stride],
        )
    }

    #[inline(always)]
    pub unsafe fn grad_as_tensor4d_mut(&self, memory: &mut MemoryBuffer<C>) -> TensorMut<C, 4> {
        TensorMut::from_raw_parts(
            memory.as_mut_ptr().add(self.offset + self.unit_size()),
            [self.nparams, self.nmats, self.nrows, self.ncols],
            [self.param_stride, self.mat_stride, self.row_stride, self.col_stride],
        )
    }

    // ----- Hessian -----

    #[inline(always)]
    pub unsafe fn hess_as_symsq_matrix_ref(&self, memory: &MemoryBuffer<C>) -> SymSqTensorRef<C, 2> { 
        SymSqTensorRef::from_raw_parts(
            memory.as_ptr().add(self.offset + self.unit_size() + self.grad_size()),
            [self.nparams, self.nparams],
            [self.nparams*self.param_stride, self.param_stride],
        )
    }

    #[inline(always)]
    pub unsafe fn hess_as_symsq_matrix_mut(&self, memory: &mut MemoryBuffer<C>) -> SymSqTensorMut<C, 2> {
        SymSqTensorMut::from_raw_parts(
            memory.as_mut_ptr().add(self.offset + self.unit_size() + self.grad_size()),
            [self.nparams, self.nparams],
            [self.nparams*self.param_stride, self.param_stride],
        )
    }

    #[inline(always)]
    pub unsafe fn hess_as_symsq_tensor3d_ref(&self, memory: &MemoryBuffer<C>) -> SymSqTensorRef<C, 3> {
        SymSqTensorRef::from_raw_parts(
            memory.as_ptr().add(self.offset + self.unit_size() + self.grad_size()),
            [self.nparams, self.nparams, self.ncols],
            [self.nparams*self.param_stride, self.param_stride, self.col_stride],
        )
    }

    #[inline(always)]
    pub unsafe fn hess_as_symsq_tensor3d_mut(&self, memory: &mut MemoryBuffer<C>) -> SymSqTensorMut<C, 3> {
        SymSqTensorMut::from_raw_parts(
            memory.as_mut_ptr().add(self.offset + self.unit_size() + self.grad_size()),
            [self.nparams, self.nparams, self.ncols],
            [self.nparams*self.param_stride, self.param_stride, self.col_stride],
        )
    }

    #[inline(always)]
    pub unsafe fn hess_as_symsq_tensor4d_ref(&self, memory: &MemoryBuffer<C>) -> SymSqTensorRef<C, 4> {
        SymSqTensorRef::from_raw_parts(
            memory.as_ptr().add(self.offset + self.unit_size() + self.grad_size()),
            [self.nparams, self.nparams, self.nrows, self.ncols],
            [self.nparams*self.param_stride, self.param_stride, self.row_stride, self.col_stride],
        )
    }

    #[inline(always)]
    pub unsafe fn hess_as_symsq_tensor4d_mut(&self, memory: &mut MemoryBuffer<C>) -> SymSqTensorMut<C, 4> {
        SymSqTensorMut::from_raw_parts(
            memory.as_mut_ptr().add(self.offset + self.unit_size() + self.grad_size()),
            [self.nparams, self.nparams, self.nrows, self.ncols],
            [self.nparams*self.param_stride, self.param_stride, self.row_stride, self.col_stride],
        )
    }

    #[inline(always)]
    pub unsafe fn hess_as_symsq_tensor5d_ref(&self, memory: &MemoryBuffer<C>) -> SymSqTensorRef<C, 5> {
        SymSqTensorRef::from_raw_parts(
            memory.as_ptr().add(self.offset + self.unit_size() + self.grad_size()),
            [self.nparams, self.nparams, self.nmats, self.nrows, self.ncols],
            [self.nparams*self.param_stride, self.param_stride, self.mat_stride, self.row_stride, self.col_stride],
        )
    }

    #[inline(always)]
    pub unsafe fn hess_as_symsq_tensor5d_mut(&self, memory: &mut MemoryBuffer<C>) -> SymSqTensorMut<C, 5> {
        SymSqTensorMut::from_raw_parts(
            memory.as_mut_ptr().add(self.offset + self.unit_size() + self.grad_size()),
            [self.nparams, self.nparams, self.nmats, self.nrows, self.ncols],
            [self.nparams*self.param_stride, self.param_stride, self.mat_stride, self.row_stride, self.col_stride],
        )
    }
}
