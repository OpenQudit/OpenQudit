use faer::MatMut;
use qudit_core::{matrix::{MatVecMut, SymSqMatMatMut}, memory::MemoryBuffer, ComplexScalar};
use qudit_expr::{DifferentiationLevel, Module};
use qudit_expr::{FUNCTION, GRADIENT, HESSIAN};

use crate::{bytecode::BytecodeInstruction, cpu::instructions::{ConsecutiveParamSingleWriteStruct, FRPRStruct}};

use super::buffer::SizedTensorBuffer;
use super::instructions::MatmulStruct;
use super::instructions::SplitParamSingleWriteStruct;

pub enum SpecializedInstruction<C: ComplexScalar> {
    FRPR(FRPRStruct<C>),
    WriteCS(ConsecutiveParamSingleWriteStruct<C>),
    WriteSS(SplitParamSingleWriteStruct<C>),
    MatMulB(MatmulStruct<C>),
    MatMulS(MatmulStruct<C>),
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
            // BytecodeInstruction::OverlappingMatmul(a, b, c, a_shared_indices, b_shared_indices) => {
            //     todo!()
            //     // let spec_a = buffers[*a].clone();
            //     // let spec_b = buffers[*b].clone();
            //     // let spec_c = buffers[*c].clone();
            //     // SpecializedInstruction::OMatMul(OverlappingMatMulStruct::new(
            //     //     spec_a, spec_b, spec_c, a_shared_indices.clone(), b_shared_indices.clone(),
            //     // ))
            // },
            // BytecodeInstruction::DisjointKron(a, b, c) => {
            //     todo!()
            //     // let spec_a = buffers[*a].clone();
            //     // let spec_b = buffers[*b].clone();
            //     // let spec_c = buffers[*c].clone();
            //     // SpecializedInstruction::DKron(DisjointKronStruct::new(
            //     //     spec_a, spec_b, spec_c,
            //     // ))
            // },
            // BytecodeInstruction::OverlappingKron(a, b, c, a_shared_indices, b_shared_indices) => {
            //     todo!()
            //     // let spec_a = buffers[*a].clone();
            //     // let spec_b = buffers[*b].clone();
            //     // let spec_c = buffers[*c].clone();
            //     // SpecializedInstruction::OKron(OverlappingKronStruct::new(
            //     //     spec_a, spec_b, spec_c, a_shared_indices.clone(), b_shared_indices.clone(),
            //     // ))
            // },
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
            _ => panic!("Unsupported bytecode operation."),
        }
    }

    #[inline(always)]
    pub fn get_output_buffer(&self) -> &SizedTensorBuffer<C> {
        match self {
            SpecializedInstruction::FRPR(s) => s.get_output_buffer(),
            SpecializedInstruction::WriteSS(s) => s.get_output_buffer(),
            SpecializedInstruction::WriteCS(s) => s.get_output_buffer(),
            SpecializedInstruction::MatMulB(s) => s.get_output_buffer(),
            SpecializedInstruction::MatMulS(s) => s.get_output_buffer(),
        }
    }

    #[inline(always)]
    pub unsafe fn evaluate<const D: DifferentiationLevel>(&self, params: &[C::R], memory: &mut MemoryBuffer<C>) {
        match self {
            SpecializedInstruction::FRPR(s) => s.evaluate(memory),
            SpecializedInstruction::WriteSS(s) => s.evaluate(params, memory),
            SpecializedInstruction::WriteCS(s) => s.evaluate(params, memory),
            SpecializedInstruction::MatMulB(s) => s.batched_evaluate::<D>(memory),
            SpecializedInstruction::MatMulS(s) => s.evaluate::<D>(memory),
        }
    }
}

