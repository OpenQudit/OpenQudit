use faer::MatMut;
use qudit_core::{matrix::{MatVecMut, SymSqMatMatMut}, memory::MemoryBuffer, ComplexScalar};

use super::instructions::{ConsecutiveParamWriteStruct, SplitParamWriteStruct, DisjointMatmulStruct, OverlappingMatMulStruct, DisjointKronStruct, OverlappingKronStruct, FRPRStruct};

pub enum SpecializedInstruction<C: ComplexScalar> {
    CWrite(ConsecutiveParamWriteStruct<C>),
    SWrite(SplitParamWriteStruct<C>),
    DMatMul(DisjointMatmulStruct<C>),
    OMatMul(OverlappingMatMulStruct<C>),
    DKron(DisjointKronStruct),
    OKron(OverlappingKronStruct),
    FRPR(FRPRStruct),
}

impl<C: ComplexScalar> SpecializedInstruction<C> {
    #[inline(always)]
    pub fn execute_unitary (
        &self,
        params: &[C::R],
        memory: &mut MemoryBuffer<C>,
    ) {
        match self {
            SpecializedInstruction::CWrite(w) => {w.execute_unitary(params, memory)},
            SpecializedInstruction::SWrite(w) => {w.execute_unitary(params, memory)},
            SpecializedInstruction::DMatMul(m) => m.execute_unitary(memory),
            SpecializedInstruction::OMatMul(m) => m.execute_unitary(memory),
            SpecializedInstruction::DKron(k) => k.execute_unitary::<C>(memory),
            SpecializedInstruction::OKron(k) => k.execute_unitary::<C>(memory),
            SpecializedInstruction::FRPR(f) => f.execute_unitary::<C>(memory),
        }
    }

    pub fn execute_unitary_and_gradient(
        &self,
        params: &[C::R],
        memory: &mut MemoryBuffer<C>,
    ) {
        match self {
            SpecializedInstruction::CWrite(w) => {
                w.execute_unitary_and_gradient(params, memory)
            },
            SpecializedInstruction::SWrite(w) => {
                w.execute_unitary_and_gradient(params, memory)
            },
            SpecializedInstruction::DMatMul(m) => {
                m.execute_unitary_and_gradient(memory)
            },
            SpecializedInstruction::OMatMul(m) => {
                m.execute_unitary_and_gradient(memory)
            },
            SpecializedInstruction::DKron(k) => {
                k.execute_unitary_and_gradient::<C>(memory)
            },
            SpecializedInstruction::OKron(k) => {
                k.execute_unitary_and_gradient::<C>(memory)
            },
            SpecializedInstruction::FRPR(f) => {
                f.execute_unitary_and_gradient::<C>(memory)
            },
        }
    }

    pub fn execute_unitary_gradient_and_hessian (
        &self,
        params: &[C::R],
        memory: &mut MemoryBuffer<C>,
    ) {
        match self {
            SpecializedInstruction::CWrite(w) => {
                w.execute_unitary_gradient_and_hessian(params, memory)
            },
            SpecializedInstruction::SWrite(w) => {
                w.execute_unitary_gradient_and_hessian(params, memory)
            },
            SpecializedInstruction::DMatMul(m) => {
                m.execute_unitary_gradient_and_hessian(memory)
            },
            SpecializedInstruction::OMatMul(m) => {
                m.execute_unitary_gradient_and_hessian(memory)
            },
            SpecializedInstruction::DKron(k) => {
                k.execute_unitary_gradient_and_hessian::<C>(memory)
            },
            SpecializedInstruction::OKron(k) => {
                k.execute_unitary_gradient_and_hessian::<C>(memory)
            },
            SpecializedInstruction::FRPR(f) => {
                f.execute_unitary_gradient_and_hessian::<C>(memory)
            },
        }
    }

    // pub fn execute_unitary_into (
    //     &self,
    //     params: &[C::R],
    //     memory: &mut MemoryBuffer<C>,
    //     out: MatMut<C>,
    // ) {
    //     match self {
    //         SpecializedInstruction::Write(w) => {
    //             w.execute_unitary_into(params, memory, out)
    //         },
    //         SpecializedInstruction::Matmul(m) => {
    //             m.execute_unitary_into::<C>(memory, out)
    //         },
    //         SpecializedInstruction::Kron(k) => {
    //             k.execute_unitary_into::<C>(memory, out)
    //         },
    //         SpecializedInstruction::FRPR(f) => {
    //             f.execute_unitary_into::<C>(memory, out)
    //         },
    //     }
    // }

    // pub fn execute_unitary_and_gradient_into (
    //     &self,
    //     params: &[C::R],
    //     memory: &mut MemoryBuffer<C>,
    //     out: MatMut<C>,
    //     grad: MatVecMut<C>,
    // ) {
    //     match self {
    //         SpecializedInstruction::Write(w) => w
    //             .execute_unitary_and_gradient_into(
    //                 params, memory, out, grad,
    //             ),
    //         SpecializedInstruction::Matmul(m) => {
    //             m.execute_unitary_and_gradient_into::<C>(memory, out, grad)
    //         },
    //         SpecializedInstruction::Kron(k) => {
    //             k.execute_unitary_and_gradient_into::<C>(memory, out, grad)
    //         },
    //         SpecializedInstruction::FRPR(f) => {
    //             f.execute_unitary_and_gradient_into::<C>(memory, out, grad)
    //         },
    //     }
    // }

    // pub fn execute_unitary_gradient_and_hessian_into (
    //     &self,
    //     params: &[C::R],
    //     memory: &mut MemoryBuffer<C>,
    //     out: MatMut<C>,
    //     grad: MatVecMut<C>,
    //     hess: SymSqMatMatMut<C>,
    // ) {
    //     match self {
    //         SpecializedInstruction::Write(w) => w
    //             .execute_unitary_gradient_and_hessian_into(
    //                 params, memory, out, grad, hess,
    //             ),
    //         SpecializedInstruction::Matmul(m) => m
    //             .execute_unitary_gradient_and_hessian_into::<C>(
    //                 memory, out, grad, hess,
    //             ),
    //         SpecializedInstruction::Kron(k) => k
    //             .execute_unitary_gradient_and_hessian_into::<C>(
    //                 memory, out, grad, hess,
    //             ),
    //         SpecializedInstruction::FRPR(f) => f
    //             .execute_unitary_gradient_and_hessian_into::<C>(
    //                 memory, out, grad, hess,
    //             ),
    //     }
    // }
}
