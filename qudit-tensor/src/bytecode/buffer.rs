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
    shape: GenerationShape,
    num_params: usize,
}

impl TensorBuffer {
    pub fn new(shape: GenerationShape, num_params: usize) -> Self {
        TensorBuffer {
            shape,
            num_params,
        }
    }
    pub fn ncols(&self) -> usize {
        self.shape.ncols()
    }

    pub fn nrows(&self) -> usize {
        self.shape.nrows()
    }

    pub fn nmats(&self) -> usize {
        self.shape.nmats()
    }

    pub fn num_params(&self) -> usize {
        self.num_params
    }

    pub fn shape(&self) -> GenerationShape {
        self.shape
    }
}

