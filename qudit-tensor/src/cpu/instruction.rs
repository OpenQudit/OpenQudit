use faer::MatMut;
use qudit_core::{matrix::{MatVecMut, SymSqMatMatMut}, memory::MemoryBuffer, ComplexScalar};
use qudit_expr::{DifferentiationLevel, Module};
use qudit_expr::{FUNCTION, GRADIENT, HESSIAN};

use crate::{bytecode::BytecodeInstruction, cpu::instructions::{ConsecutiveParamSingleWriteStruct, FRPRStruct, HadamardStruct, KronStruct, TraceStruct}};

use super::buffer::SizedTensorBuffer;
use super::instructions::MatmulStruct;
use super::instructions::SplitParamSingleWriteStruct;

pub enum TNVMInstruction<C: ComplexScalar> {
    FRPR(FRPRStruct<C>),
    HadamardB(HadamardStruct<C>),
    HadamardS(HadamardStruct<C>),
    KronB(KronStruct<C>),
    KronS(KronStruct<C>),
    MatMulB(MatmulStruct<C>),
    MatMulS(MatmulStruct<C>),
    Trace(TraceStruct<C>),
    WriteCS(ConsecutiveParamSingleWriteStruct<C>),
    WriteSS(SplitParamSingleWriteStruct<C>),
}

impl<C: ComplexScalar> TNVMInstruction<C> {
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
                TNVMInstruction::FRPR(FRPRStruct::new(
                    spec_a,
                    shape,
                    perm,
                    spec_b,
                    D,
                ))
            },
            BytecodeInstruction::Hadamard(a, b, c, p1, p2) => {
                let spec_a = buffers[*a].clone();
                let spec_b = buffers[*b].clone();
                let spec_c = buffers[*c].clone();
                if spec_c.shape().is_matrix() {
                    TNVMInstruction::HadamardS(HadamardStruct::new(
                        spec_a,
                        spec_b,
                        spec_c,
                        p1.clone(),
                        p2.clone(),
                    ))
                } else {
                    TNVMInstruction::HadamardB(HadamardStruct::new(
                        spec_a,
                        spec_b,
                        spec_c,
                        p1.clone(),
                        p2.clone(),
                    ))
                }
            },
            BytecodeInstruction::Kron(a, b, c, p1, p2) => {
                let spec_a = buffers[*a].clone();
                let spec_b = buffers[*b].clone();
                let spec_c = buffers[*c].clone();
                if spec_c.shape().is_matrix() {
                    TNVMInstruction::KronS(KronStruct::new(
                        spec_a,
                        spec_b,
                        spec_c,
                        p1.clone(),
                        p2.clone(),
                    ))
                } else {
                    TNVMInstruction::KronB(KronStruct::new(
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
                    TNVMInstruction::MatMulS(MatmulStruct::new(
                        spec_a,
                        spec_b,
                        spec_c,
                        p1.clone(),
                        p2.clone(),
                    ))
                } else {
                    TNVMInstruction::MatMulB(MatmulStruct::new(
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
                TNVMInstruction::WriteCS(ConsecutiveParamSingleWriteStruct::new(
                    write_fn,
                    *param_start_index,
                    buffers[*buffer_index].clone(),
                ))
            },
            BytecodeInstruction::SplitParamWrite(name, param_indices, index) => {
                let write_fn = unsafe { module.get_function_raw(&name) };
                TNVMInstruction::WriteSS(SplitParamSingleWriteStruct::new(
                    write_fn,
                    buffers[*index].clone(),
                ))
            },
            BytecodeInstruction::Trace(in_index, dimension_pairs, out_index) => {
                let spec_a = buffers[*in_index].clone();
                let spec_b = buffers[*out_index].clone();
                TNVMInstruction::Trace(TraceStruct::new(
                    spec_a,
                    dimension_pairs.clone(),
                    spec_b,
                    D,
                ))
            },
        }
    }

    #[inline(always)]
    pub fn get_output_buffer(&self) -> &SizedTensorBuffer<C> {
        match self {
            TNVMInstruction::FRPR(s) => s.get_output_buffer(),
            TNVMInstruction::HadamardB(s) => s.get_output_buffer(),
            TNVMInstruction::HadamardS(s) => s.get_output_buffer(),
            TNVMInstruction::KronB(s) => s.get_output_buffer(),
            TNVMInstruction::KronS(s) => s.get_output_buffer(),
            TNVMInstruction::MatMulB(s) => s.get_output_buffer(),
            TNVMInstruction::MatMulS(s) => s.get_output_buffer(),
            TNVMInstruction::Trace(s) => s.get_output_buffer(),
            TNVMInstruction::WriteCS(s) => s.get_output_buffer(),
            TNVMInstruction::WriteSS(s) => s.get_output_buffer(),
        }
    }

    #[inline(always)]
    pub unsafe fn evaluate<const D: DifferentiationLevel>(&self, params: &[C::R], memory: &mut MemoryBuffer<C>) {
        match self {
            TNVMInstruction::FRPR(s) => s.evaluate(memory),
            TNVMInstruction::HadamardB(s) => s.batched_evaluate::<D>(memory),
            TNVMInstruction::HadamardS(s) => s.evaluate::<D>(memory),
            TNVMInstruction::KronB(s) => s.batched_evaluate::<D>(memory),
            TNVMInstruction::KronS(s) => s.evaluate::<D>(memory),
            TNVMInstruction::MatMulB(s) => s.batched_evaluate::<D>(memory),
            TNVMInstruction::MatMulS(s) => s.evaluate::<D>(memory),
            TNVMInstruction::Trace(s) => s.evaluate(memory),
            TNVMInstruction::WriteCS(s) => s.evaluate(params, memory),
            TNVMInstruction::WriteSS(s) => s.evaluate(params, memory),
        }
    }
}

