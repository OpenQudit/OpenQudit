use faer_core::MatMut;

use crate::{
    math::{
        matrix::{MatGradMut, MatHessMut},
        unitary::{
            DifferentiableUnitaryFn, DoublyDifferentiableUnitaryFn, UnitaryFn,
        },
        ComplexScalar, Function,
    },
    sim::{bytecode::SizedMatrixBuffer, qvm::Memory},
    Gate,
};

pub struct WriteStruct {
    pub gate: Gate,
    pub idx: usize,
    pub buffer: SizedMatrixBuffer,
}

impl WriteStruct {
    pub fn new(gate: Gate, idx: usize, buffer: SizedMatrixBuffer) -> Self {
        Self { gate, idx, buffer }
    }

    #[inline(always)]
    pub fn execute_unitary<C: ComplexScalar>(
        &self,
        params: &[C::Re],
        memory: &mut Memory<C>,
    ) {
        let gate_params =
            &params[self.idx..self.idx + self.gate.get_num_params()];
        let mut matmut = self.buffer.as_matmut::<C>(memory);
        self.gate.write_unitary(gate_params, &mut matmut);
    }

    #[inline(always)]
    pub fn execute_unitary_and_gradient<C: ComplexScalar>(
        &self,
        params: &[C::Re],
        memory: &mut Memory<C>,
    ) {
        let gate_params =
            &params[self.idx..self.idx + self.gate.get_num_params()];
        let mut matmut = self.buffer.as_matmut::<C>(memory);
        let mut matgradmut = self.buffer.as_matgradmut::<C>(memory);
        self.gate.write_unitary_and_gradient(
            gate_params,
            &mut matmut,
            &mut matgradmut,
        );
    }

    #[inline(always)]
    pub fn execute_unitary_gradient_and_hessian<C: ComplexScalar>(
        &self,
        params: &[C::Re],
        memory: &mut Memory<C>,
    ) {
        let gate_params =
            &params[self.idx..self.idx + self.gate.get_num_params()];
        let mut matmut = self.buffer.as_matmut::<C>(memory);
        let mut matgradmut = self.buffer.as_matgradmut::<C>(memory);
        let mut mathessmut = self.buffer.as_mathessmut::<C>(memory);
        self.gate.write_unitary_gradient_and_hessian(
            gate_params,
            &mut matmut,
            &mut matgradmut,
            &mut mathessmut,
        );
    }

    #[inline(always)]
    pub fn execute_unitary_into<C: ComplexScalar>(
        &self,
        params: &[C::Re],
        memory: &mut Memory<C>,
        mut out: MatMut<C>,
    ) {
        let gate_params =
            &params[self.idx..self.idx + self.gate.get_num_params()];
        self.gate.write_unitary(gate_params, &mut out);
    }

    #[inline(always)]
    pub fn execute_unitary_and_gradient_into<C: ComplexScalar>(
        &self,
        params: &[C::Re],
        memory: &mut Memory<C>,
        mut out: MatMut<C>,
        mut matgradmut: MatGradMut<C>,
    ) {
        let gate_params =
            &params[self.idx..self.idx + self.gate.get_num_params()];
        self.gate.write_unitary_and_gradient(
            gate_params,
            &mut out,
            &mut matgradmut,
        );
    }

    #[inline(always)]
    pub fn execute_unitary_gradient_and_hessian_into<C: ComplexScalar>(
        &self,
        params: &[C::Re],
        memory: &mut Memory<C>,
        mut out: MatMut<C>,
        mut matgradmut: MatGradMut<C>,
        mut mathessmut: MatHessMut<C>,
    ) {
        let gate_params =
            &params[self.idx..self.idx + self.gate.get_num_params()];
        self.gate.write_unitary_gradient_and_hessian(
            gate_params,
            &mut out,
            &mut matgradmut,
            &mut mathessmut,
        );
    }
}
