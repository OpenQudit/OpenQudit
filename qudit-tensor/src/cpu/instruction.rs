use faer::MatMut;
use qudit_core::{matrix::{MatVecMut, SymSqMatMatMut}, memory::MemoryBuffer, ComplexScalar};
use qudit_expr::DifferentiationLevel;

use super::buffer::SizedTensorBuffer;
use super::instructions::IndependentSingleMatmulStruct;
use super::instructions::SplitParamSingleWriteStruct;

pub enum SpecializedInstruction<C: ComplexScalar, const D: DifferentiationLevel> {
    WriteSS(SplitParamSingleWriteStruct<C>),
    MatMulDS(IndependentSingleMatmulStruct<C, D>),
}

impl<C: ComplexScalar, const D: DifferentiationLevel> SpecializedInstruction<C, D> {
    #[inline(always)]
    pub fn get_output_buffer(&self) -> &SizedTensorBuffer<C> {
        match self {
            SpecializedInstruction::WriteSS(s) => s.get_output_buffer(),
            SpecializedInstruction::MatMulDS(s) => s.get_output_buffer(),
        }
    }
}

impl<C: ComplexScalar> SpecializedInstruction<C, 0> {
    #[inline(always)]
    pub unsafe fn evaluate(&self, params: &[C::R], memory: &mut MemoryBuffer<C>) {
        match self {
            SpecializedInstruction::WriteSS(s) => s.evaluate(params, memory),
            SpecializedInstruction::MatMulDS(s) => s.evaluate(memory),
        }
    }
}

impl<C: ComplexScalar> SpecializedInstruction<C, 1> {
    #[inline(always)]
    pub unsafe fn evaluate(&self, params: &[C::R], memory: &mut MemoryBuffer<C>) {
        match self {
            SpecializedInstruction::WriteSS(s) => s.evaluate(params, memory),
            SpecializedInstruction::MatMulDS(s) => s.evaluate(memory),
        }
    }
}

impl<C: ComplexScalar> SpecializedInstruction<C, 2> {
    #[inline(always)]
    pub unsafe fn evaluate(&self, params: &[C::R], memory: &mut MemoryBuffer<C>) {
        match self {
            SpecializedInstruction::WriteSS(s) => s.evaluate(params, memory),
            SpecializedInstruction::MatMulDS(s) => s.evaluate(memory),
        }
    }
}
