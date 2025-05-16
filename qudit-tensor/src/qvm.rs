// use aligned_vec::{avec, AVec};
// use bytemuck::Zeroable;
use faer::reborrow::ReborrowMut;
use qudit_expr::DifferentiationLevel;
use qudit_expr::Module;
use qudit_expr::TensorGenerationShape;


use crate::bytecode::SizedTensorBuffer;

use super::bytecode::Bytecode;
use super::bytecode::SpecializedInstruction;
use qudit_core::accel::fused_reshape_permute_reshape_into_impl;
use qudit_core::matrix::MatVecMut;
use qudit_core::matrix::MatVecRef;
use qudit_core::matrix::SymSqMatMatMut;
use qudit_core::matrix::MatMut;
use qudit_core::matrix::MatRef;
use qudit_core::memory::MemoryBuffer;
use qudit_core::memory::alloc_zeroed_memory;
use qudit_core::ComplexScalar;

pub struct QVM<C: ComplexScalar> {
    first_run: bool,
    static_instructions: Vec<SpecializedInstruction<C>>,
    dynamic_instructions: Vec<SpecializedInstruction<C>>,
    #[allow(dead_code)]
    module: Module<C>,
    memory: MemoryBuffer<C>,
    diff_lvl: DifferentiationLevel,
}

pub enum QVMReturnType<'a, C: ComplexScalar> {
    Scalar(C),
    RowVector(qudit_core::matrix::RowRef<'a, C>),
    ColVector(qudit_core::matrix::ColRef<'a, C>),
    Matrix(qudit_core::matrix::MatRef<'a, C>),
    MatVec(qudit_core::matrix::MatVecRef<'a, C>),
}

impl<'a, C: ComplexScalar> QVMReturnType<'a, C> {
    pub fn unpack_scalar(self) -> C {
        match self {
            QVMReturnType::Scalar(s) => s,
            _ => panic!("cannot unpack a non-scalar type as a scalar"),
        }
    }

    pub fn unpack_row_vector(self) -> qudit_core::matrix::RowRef<'a, C> {
        match self {
            QVMReturnType::RowVector(v) => v,
            _ => panic!("cannot unpack a non-row-vector type as a row-vector"),
        }
    }

    pub fn unpack_col_vector(self) -> qudit_core::matrix::ColRef<'a, C> {
        match self {
            QVMReturnType::ColVector(v) => v,
            _ => panic!("cannot unpack a non-col-vector type as a col-vector"),
        }
    }

    pub fn unpack_matrix(self) -> qudit_core::matrix::MatRef<'a, C> {
        match self {
            QVMReturnType::Matrix(m) => m,
            _ => panic!("cannot unpack a non-matrix type as a matrix"),
        }
    }

    pub fn unpack_matvec(self) -> qudit_core::matrix::MatVecRef<'a, C> {
        match self {
            QVMReturnType::MatVec(m) => m,
            _ => panic!("cannot unpack a non-matvec type as a matvec"),
        }
    }
}

pub struct QVMResult<'a, C: ComplexScalar> {
    memory: &'a MemoryBuffer<C>,
    buffer: SizedTensorBuffer,
}

impl<'a, C: ComplexScalar> QVMResult<'a, C> {
    pub fn get_fn_result(&self) -> QVMReturnType<'a, C> {
        match self.buffer.shape() {
            TensorGenerationShape::Scalar => {
                todo!()
            }
            TensorGenerationShape::Vector(len) => {
                todo!()
            }
            TensorGenerationShape::Matrix(rows, cols) => {
                QVMReturnType::Matrix(self.buffer.as_matref(self.memory))
            }
            TensorGenerationShape::Tensor(mats, rows, cols) => {
                println!("mats: {}, rows: {}, cols: {}", mats, rows, cols);
                QVMReturnType::MatVec(self.buffer.as_matvecref_non_gradient(self.memory))
            }
        }
    }

    pub fn get_grad_result(&self) -> QVMReturnType<'a, C> {
        todo!()
    }

    pub fn get_hess_result(&self) -> QVMReturnType<'a, C> {
        todo!()
    }
}


impl<C: ComplexScalar> QVM<C> {
    pub fn new(program: Bytecode, diff_lvl: DifferentiationLevel) -> Self {
        let (sinsts, dinsts, module, mem_size) = program.specialize::<C>(diff_lvl);

        Self {
            first_run: true,
            static_instructions: sinsts,
            dynamic_instructions: dinsts,
            module,
            memory: alloc_zeroed_memory::<C>(mem_size),
            diff_lvl,
        }
    }

    #[inline(always)]
    fn first_run(&mut self) {
        if !self.first_run {
            return;
        }

        // Evaluate static code
        for inst in &self.static_instructions {
            inst.evaluate(&[], &mut self.memory);
        }

        self.first_run = false;
    }

    pub fn evaluate(&mut self, args: &[C::R]) -> QVMResult<'_, C> {
        self.first_run();

        match self.diff_lvl {
            DifferentiationLevel::None => {
                for inst in &self.dynamic_instructions {
                    inst.evaluate(args, &mut self.memory);
                }
            }
            DifferentiationLevel::Gradient => {
                for inst in &self.dynamic_instructions {
                    inst.evaluate_gradient(args, &mut self.memory);
                }
            }
            DifferentiationLevel::Hessian => {
                for inst in &self.dynamic_instructions {
                    inst.evaluate_hessian(args, &mut self.memory);
                }
            }
        }
    
        self.get_output_buffer()
    }

    pub fn get_output_buffer(&self) -> QVMResult<'_, C> { 
        let last_instruction = if self.dynamic_instructions.len() == 0 {
            self.static_instructions.last()
        } else {
            self.dynamic_instructions.last()
        };

        match last_instruction {
            None => panic!("Seriously..."),
            Some(inst) => {
                QVMResult {
                    memory: &self.memory,
                    buffer: inst.get_output_buffer(),
                }
            }
        }
    }
}

// TODO: TEST: No params in entire circuit, constant everything
