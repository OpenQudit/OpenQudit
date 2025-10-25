use std::sync::{Arc, Mutex};
use std::{cell::RefCell, rc::Rc};

use faer::MatMut;
use qudit_core::{memory::MemoryBuffer, ComplexScalar};
use qudit_expr::{DifferentiationLevel, ExpressionCache, ExpressionId, Module};
use qudit_expr::{FUNCTION, GRADIENT, HESSIAN};

use crate::{bytecode::BytecodeInstruction, cpu::instructions::{WriteStruct, FRPRStruct, HadamardStruct, KronStruct, TraceStruct}};

use super::buffer::SizedTensorBuffer;
use super::instructions::MatmulStruct;

pub enum TNVMInstruction<C: ComplexScalar, const D: DifferentiationLevel> {
    FRPR(FRPRStruct<C, D>),
    HadamardB(HadamardStruct<C>),
    HadamardS(HadamardStruct<C>),
    KronB(KronStruct<C>),
    KronS(KronStruct<C>),
    MatMulB(MatmulStruct<C>),
    MatMulS(MatmulStruct<C>),
    Trace(TraceStruct<C>),
    Write(WriteStruct<C, D>),
}

impl<C: ComplexScalar, const D: DifferentiationLevel> TNVMInstruction<C, D> {
    pub fn new(
        inst: &BytecodeInstruction,
        buffers: &Vec<SizedTensorBuffer<C>>,
        expressions: Arc<Mutex<ExpressionCache>>,
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
            BytecodeInstruction::Write(expr_id, param_info, buffer_index) => {
                let out_buffer = buffers[*buffer_index].clone();
                let write_fns = std::array::from_fn(|i| expressions.lock().unwrap().get_fn::<C::R>(*expr_id, i + 1));
                let output_map = expressions.lock().unwrap().get_output_map::<C::R>(
                    *expr_id,
                    out_buffer.row_stride() as u64,
                    out_buffer.col_stride() as u64,
                    out_buffer.mat_stride() as u64,
                );
                let param_map = param_info.get_param_map();
                let const_map = param_info.get_const_map();

                TNVMInstruction::Write(WriteStruct::new(
                    write_fns,
                    param_map,
                    output_map,
                    const_map,
                    out_buffer,
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
            TNVMInstruction::Write(s) => s.get_output_buffer(),
        }
    }

    #[inline(always)]
    pub unsafe fn evaluate<const E: DifferentiationLevel>(&self, params: &[C::R], memory: &mut MemoryBuffer<C>) {
        match self {
            TNVMInstruction::FRPR(s) => s.evaluate::<E>(memory),
            TNVMInstruction::HadamardB(s) => s.batched_evaluate::<E>(memory),
            TNVMInstruction::HadamardS(s) => s.evaluate::<E>(memory),
            TNVMInstruction::KronB(s) => s.batched_evaluate::<E>(memory),
            TNVMInstruction::KronS(s) => s.evaluate::<E>(memory),
            TNVMInstruction::MatMulB(s) => s.batched_evaluate::<E>(memory),
            TNVMInstruction::MatMulS(s) => s.evaluate::<E>(memory),
            TNVMInstruction::Trace(s) => s.evaluate(memory),
            TNVMInstruction::Write(s) => s.evaluate::<E>(params, memory),
        }
    }
}

