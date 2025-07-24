use faer::MatMut;
use qudit_core::{matrix::{MatVecMut, SymSqMatMatMut}, memory::MemoryBuffer, ComplexScalar};
use qudit_expr::{DifferentiationLevel, Module};
use qudit_expr::{FUNCTION, GRADIENT, HESSIAN};

use crate::{bytecode::BytecodeInstruction, cpu::instructions::{ConsecutiveParamSingleWriteStruct, FRPRStruct, KronStruct}};

use super::buffer::SizedTensorBuffer;
use super::instructions::MatmulStruct;
use super::instructions::SplitParamSingleWriteStruct;

pub enum SpecializedInstruction<C: ComplexScalar> {
    FRPR(FRPRStruct<C>),
    KronB(KronStruct<C>),
    KronS(KronStruct<C>),
    MatMulB(MatmulStruct<C>),
    MatMulS(MatmulStruct<C>),
    WriteCS(ConsecutiveParamSingleWriteStruct<C>),
    WriteSS(SplitParamSingleWriteStruct<C>),
}

// TODO: rename specialized to tnvminstruction

impl<C: ComplexScalar> SpecializedInstruction<C> {
    pub fn new(
        inst: &BytecodeInstruction,
        buffers: &Vec<SizedTensorBuffer<C>>,
        module: &Module<C>,
        D: DifferentiationLevel,
    ) -> Self {
        match inst {
            BytecodeInstruction::FRPR(in_index, shape, perm, out_index) => {
                let spec_a = buffers[*in_index].clone();
                let spec_b = buffers[*out_index].clone();
                SpecializedInstruction::FRPR(FRPRStruct::new(
                    spec_a,
                    shape,
                    perm,
                    spec_b,
                    D,
                ))
            },
            BytecodeInstruction::Hadamard(a, b, c, p1, p2) => {
                todo!()
            },
            BytecodeInstruction::Kron(a, b, c, p1, p2) => {
                let spec_a = buffers[*a].clone();
                let spec_b = buffers[*b].clone();
                let spec_c = buffers[*c].clone();
                if spec_c.shape().is_matrix() {
                    SpecializedInstruction::KronS(KronStruct::new(
                        spec_a,
                        spec_b,
                        spec_c,
                        p1.clone(),
                        p2.clone(),
                    ))
                } else {
                    SpecializedInstruction::KronB(KronStruct::new(
                        spec_a,
                        spec_b,
                        spec_c,
                        p1.clone(),
                        p2.clone(),
                    ))
                }
            },
            BytecodeInstruction::Matmul(a, b, c, p1, p2) => {
                let spec_a = buffers[*a].clone();
                let spec_b = buffers[*b].clone();
                let spec_c = buffers[*c].clone();
                if spec_c.shape().is_matrix() {
                    SpecializedInstruction::MatMulS(MatmulStruct::new(
                        spec_a,
                        spec_b,
                        spec_c,
                        p1.clone(),
                        p2.clone(),
                    ))
                } else {
                    SpecializedInstruction::MatMulB(MatmulStruct::new(
                        spec_a,
                        spec_b,
                        spec_c,
                        p1.clone(),
                        p2.clone(),
                    ))
                }
            },
            BytecodeInstruction::ConsecutiveParamWrite(name, param_start_index, buffer_index) => {
                let write_fn = unsafe { module.get_function_raw(&name) };
                SpecializedInstruction::WriteCS(ConsecutiveParamSingleWriteStruct::new(
                    write_fn,
                    *param_start_index,
                    buffers[*buffer_index].clone(),
                ))
            },
            BytecodeInstruction::SplitParamWrite(name, param_indices, index) => {
                let write_fn = unsafe { module.get_function_raw(&name) };
                SpecializedInstruction::WriteSS(SplitParamSingleWriteStruct::new(
                    write_fn,
                    buffers[*index].clone(),
                ))
            },
            BytecodeInstruction::Trace(in_index, dimension_pairs, out_index) => {
                todo!()
            },
        }
    }

    #[inline(always)]
    pub fn get_output_buffer(&self) -> &SizedTensorBuffer<C> {
        match self {
            SpecializedInstruction::FRPR(s) => s.get_output_buffer(),
            SpecializedInstruction::KronB(s) => s.get_output_buffer(),
            SpecializedInstruction::KronS(s) => s.get_output_buffer(),
            SpecializedInstruction::MatMulB(s) => s.get_output_buffer(),
            SpecializedInstruction::MatMulS(s) => s.get_output_buffer(),
            SpecializedInstruction::WriteCS(s) => s.get_output_buffer(),
            SpecializedInstruction::WriteSS(s) => s.get_output_buffer(),
        }
    }

    #[inline(always)]
    pub unsafe fn evaluate<const D: DifferentiationLevel>(&self, params: &[C::R], memory: &mut MemoryBuffer<C>) {
        match self {
            SpecializedInstruction::FRPR(s) => s.evaluate(memory),
            SpecializedInstruction::KronB(s) => s.batched_evaluate::<D>(memory),
            SpecializedInstruction::KronS(s) => s.evaluate::<D>(memory),
            SpecializedInstruction::MatMulB(s) => s.batched_evaluate::<D>(memory),
            SpecializedInstruction::MatMulS(s) => s.evaluate::<D>(memory),
            SpecializedInstruction::WriteCS(s) => s.evaluate(params, memory),
            SpecializedInstruction::WriteSS(s) => s.evaluate(params, memory),
        }
    }
}

