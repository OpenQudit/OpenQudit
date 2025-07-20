use faer::MatMut;
use qudit_core::{matrix::{MatVecMut, SymSqMatMatMut}, memory::MemoryBuffer, ComplexScalar};

use super::{instructions::{ConsecutiveParamWriteStruct, DisjointKronStruct, DisjointMatmulStruct, FRPRStruct, OverlappingKronStruct, OverlappingMatMulStruct, SplitParamWriteStruct}, SizedTensorBuffer};

pub enum SpecializedInstruction<C: ComplexScalar> {
    CWrite(ConsecutiveParamWriteStruct<C>),
    SWrite(SplitParamWriteStruct<C>),
    DMatMul(DisjointMatmulStruct<C>),
    OMatMul(OverlappingMatMulStruct<C>),
    DKron(DisjointKronStruct),
    OKron(OverlappingKronStruct),
    FRPR(FRPRStruct),
    // Reshape(ReshapeStruct),
}

impl<C: ComplexScalar> SpecializedInstruction<C> {
    pub fn get_output_buffer(&self) -> SizedTensorBuffer {
        match self {
            SpecializedInstruction::CWrite(w) => w.buffer.clone(),
            SpecializedInstruction::SWrite(w) => w.buffer.clone(),
            SpecializedInstruction::DMatMul(m) => m.out.clone(),
            SpecializedInstruction::OMatMul(m) => m.out.clone(),
            SpecializedInstruction::DKron(k) => k.out.clone(),
            SpecializedInstruction::OKron(k) => k.out.clone(),
            SpecializedInstruction::FRPR(f) => f.out.clone(),
            // SpecializedInstruction::Reshape(r) => r.out.clone(),
        }
    }

    #[inline(always)]
    pub fn evaluate (
        &self,
        params: &[C::R],
        memory: &mut MemoryBuffer<C>,
    ) {
        match self {
            SpecializedInstruction::CWrite(w) => {w.evaluate(params, memory)},
            SpecializedInstruction::SWrite(w) => {w.evaluate(params, memory)},
            SpecializedInstruction::DMatMul(m) => m.evaluate(memory),
            SpecializedInstruction::OMatMul(m) => m.evaluate(memory),
            SpecializedInstruction::DKron(k) => k.evaluate::<C>(memory),
            SpecializedInstruction::OKron(k) => k.evaluate::<C>(memory),
            SpecializedInstruction::FRPR(f) => f.evaluate::<C>(memory),
            // SpecializedInstruction::Reshape(r) => r.evaluate::<C>(memory),
        }
    }

    pub fn evaluate_gradient(
        &self,
        params: &[C::R],
        memory: &mut MemoryBuffer<C>,
    ) {
        match self {
            SpecializedInstruction::CWrite(w) => {
                w.evaluate_gradient(params, memory)
            },
            SpecializedInstruction::SWrite(w) => {
                w.evaluate_gradient(params, memory)
            },
            SpecializedInstruction::DMatMul(m) => {
                m.evaluate_gradient(memory)
            },
            SpecializedInstruction::OMatMul(m) => {
                m.evaluate_gradient(memory)
            },
            SpecializedInstruction::DKron(k) => {
                k.evaluate_gradient::<C>(memory)
            },
            SpecializedInstruction::OKron(k) => {
                k.evaluate_gradient::<C>(memory)
            },
            SpecializedInstruction::FRPR(f) => {
                f.evaluate_gradient::<C>(memory)
            },
            // SpecializedInstruction::Reshape(r) => {
            //     r.evaluate_gradient::<C>(memory)
            // },
        }
    }

    pub fn evaluate_hessian (
        &self,
        params: &[C::R],
        memory: &mut MemoryBuffer<C>,
    ) {
        match self {
            SpecializedInstruction::CWrite(w) => {
                w.evaluate_hessian(params, memory)
            },
            SpecializedInstruction::SWrite(w) => {
                w.evaluate_hessian(params, memory)
            },
            SpecializedInstruction::DMatMul(m) => {
                m.evaluate_hessian(memory)
            },
            SpecializedInstruction::OMatMul(m) => {
                m.evaluate_hessian(memory)
            },
            SpecializedInstruction::DKron(k) => {
                k.evaluate_hessian::<C>(memory)
            },
            SpecializedInstruction::OKron(k) => {
                k.evaluate_hessian::<C>(memory)
            },
            SpecializedInstruction::FRPR(f) => {
                f.evaluate_hessian::<C>(memory)
            },
            // SpecializedInstruction::Reshape(r) => {
            //     r.evaluate_hessian::<C>(memory)
            // },
        }
    }
}
