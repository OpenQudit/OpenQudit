use faer_core::MatMut;

use crate::{
    math::{
        matrix::{MatGradMut, MatHessMut},
        ComplexScalar,
    },
    sim::qvm::Memory,
};

use super::instructions::{FRPRStruct, KronStruct, MatmulStruct, WriteStruct};

pub enum SpecializedInstruction {
    Write(WriteStruct),
    Matmul(MatmulStruct),
    Kron(KronStruct),
    FRPR(FRPRStruct),
}

impl SpecializedInstruction {
    #[inline(always)]
    pub fn execute_unitary<C: ComplexScalar>(
        &self,
        params: &[C::Re],
        memory: &mut Memory<C>,
    ) {
        match self {
            SpecializedInstruction::Write(w) => {
                w.execute_unitary::<C>(params, memory)
            },
            SpecializedInstruction::Matmul(m) => m.execute_unitary::<C>(memory),
            SpecializedInstruction::Kron(k) => k.execute_unitary::<C>(memory),
            SpecializedInstruction::FRPR(f) => f.execute_unitary::<C>(memory),
        }
    }

    pub fn execute_unitary_and_gradient<C: ComplexScalar>(
        &self,
        params: &[C::Re],
        memory: &mut Memory<C>,
    ) {
        match self {
            SpecializedInstruction::Write(w) => {
                w.execute_unitary_and_gradient::<C>(params, memory)
            },
            SpecializedInstruction::Matmul(m) => {
                m.execute_unitary_and_gradient::<C>(memory)
            },
            SpecializedInstruction::Kron(k) => {
                k.execute_unitary_and_gradient::<C>(memory)
            },
            SpecializedInstruction::FRPR(f) => {
                f.execute_unitary_and_gradient::<C>(memory)
            },
        }
    }

    pub fn execute_unitary_gradient_and_hessian<C: ComplexScalar>(
        &self,
        params: &[C::Re],
        memory: &mut Memory<C>,
    ) {
        match self {
            SpecializedInstruction::Write(w) => {
                w.execute_unitary_gradient_and_hessian::<C>(params, memory)
            },
            SpecializedInstruction::Matmul(m) => {
                m.execute_unitary_gradient_and_hessian::<C>(memory)
            },
            SpecializedInstruction::Kron(k) => {
                k.execute_unitary_gradient_and_hessian::<C>(memory)
            },
            SpecializedInstruction::FRPR(f) => {
                f.execute_unitary_gradient_and_hessian::<C>(memory)
            },
        }
    }

    pub fn execute_unitary_into<C: ComplexScalar>(
        &self,
        params: &[C::Re],
        memory: &mut Memory<C>,
        mut out: MatMut<C>,
    ) {
        match self {
            SpecializedInstruction::Write(w) => {
                w.execute_unitary_into::<C>(params, memory, out)
            },
            SpecializedInstruction::Matmul(m) => {
                m.execute_unitary_into::<C>(memory, out)
            },
            SpecializedInstruction::Kron(k) => {
                k.execute_unitary_into::<C>(memory, out)
            },
            SpecializedInstruction::FRPR(f) => {
                f.execute_unitary_into::<C>(memory, out)
            },
        }
    }

    pub fn execute_unitary_and_gradient_into<C: ComplexScalar>(
        &self,
        params: &[C::Re],
        memory: &mut Memory<C>,
        mut out: MatMut<C>,
        mut grad: MatGradMut<C>,
    ) {
        match self {
            SpecializedInstruction::Write(w) => w
                .execute_unitary_and_gradient_into::<C>(
                    params, memory, out, grad,
                ),
            SpecializedInstruction::Matmul(m) => {
                m.execute_unitary_and_gradient_into::<C>(memory, out, grad)
            },
            SpecializedInstruction::Kron(k) => {
                k.execute_unitary_and_gradient_into::<C>(memory, out, grad)
            },
            SpecializedInstruction::FRPR(f) => {
                f.execute_unitary_and_gradient_into::<C>(memory, out, grad)
            },
        }
    }

    pub fn execute_unitary_gradient_and_hessian_into<C: ComplexScalar>(
        &self,
        params: &[C::Re],
        memory: &mut Memory<C>,
        mut out: MatMut<C>,
        mut grad: MatGradMut<C>,
        mut hess: MatHessMut<C>,
    ) {
        match self {
            SpecializedInstruction::Write(w) => w
                .execute_unitary_gradient_and_hessian_into::<C>(
                    params, memory, out, grad, hess,
                ),
            SpecializedInstruction::Matmul(m) => m
                .execute_unitary_gradient_and_hessian_into::<C>(
                    memory, out, grad, hess,
                ),
            SpecializedInstruction::Kron(k) => k
                .execute_unitary_gradient_and_hessian_into::<C>(
                    memory, out, grad, hess,
                ),
            SpecializedInstruction::FRPR(f) => f
                .execute_unitary_gradient_and_hessian_into::<C>(
                    memory, out, grad, hess,
                ),
        }
    }
}
