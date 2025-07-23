use std::marker::PhantomData;

use faer::RowMut;
use faer::RowRef;
use qudit_core::array::Tensor;
use qudit_core::array::TensorRef;
use qudit_core::matrix::MatMut;
use qudit_core::matrix::MatRef;
use qudit_core::matrix::MatVecMut;
use qudit_core::matrix::MatVecRef;
use qudit_core::matrix::SymSqMatMatMut;
use qudit_core::matrix::SymSqMatMatRef;
use qudit_core::memory::MemoryBuffer;
use qudit_core::ComplexScalar;
use qudit_expr::DifferentiationLevel;
use qudit_core::TensorShape;
use qudit_expr::GenerationShape;
use qudit_expr::UnitaryExpression;
use qudit_core::QuditSystem;
use qudit_core::HasParams;

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct TensorBuffer {
    pub shape: GenerationShape,
    pub num_params: usize,
}

impl TensorBuffer {
    pub fn ncols(&self) -> usize {
        self.shape.ncols()
    }

    pub fn nrows(&self) -> usize {
        self.shape.nrows()
    }

    pub fn nmats(&self) -> usize {
        self.shape.nmats()
    }

    // pub fn specialize<C: ComplexScalar>(&self, offset: usize, diff_lvl: DifferentiationLevel) -> SizedTensorBuffer<C> {
    //     let col_stride = qudit_core::memory::calc_col_stride::<C>(self.nrows(), self.ncols());
    //     let mat_stride = qudit_core::memory::calc_mat_stride::<C>(self.nrows(), self.ncols(), col_stride);
    //     SizedTensorBuffer {
    //         offset,
    //         shape: self.shape.clone(),
    //         row_stride: 1,
    //         col_stride,
    //         mat_stride,
    //         ten_stride: mat_stride, // TODO:
    //         num_params: self.num_params,
    //         _phantom: PhantomData,
    //     }
    // }
}


