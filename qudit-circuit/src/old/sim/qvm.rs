use aligned_vec::{avec, AVec};
use bytemuck::Zeroable;
use faer_core::MatMut;
use faer_core::MatRef;
use faer_entity::{Entity, GroupFor};

use super::bytecode::Bytecode;
use super::bytecode::SpecializedInstruction;
use crate::math::fused_reshape_permuted_reshape_into_impl;
use crate::math::matrix::MatGradMut;
use crate::math::matrix::MatGradRef;
use crate::math::matrix::MatHessMut;
use crate::math::ComplexScalar;

pub type Memory<C: Entity> = GroupFor<C, AVec<C::Unit>>;

pub fn alloc_memory<C: Entity>(size: usize) -> Memory<C> {
    C::faer_map(C::UNIT, |()| avec![<C::Unit as Zeroable>::zeroed(); size])
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum QVMType {
    Unitary,
    UnitaryAndGradient,
    UnitaryGradientAndHessian,
}

impl QVMType {
    pub fn gradient_capable(&self) -> bool {
        match self {
            QVMType::Unitary => false,
            QVMType::UnitaryAndGradient => true,
            QVMType::UnitaryGradientAndHessian => true,
        }
    }

    pub fn hessian_capable(&self) -> bool {
        match self {
            QVMType::Unitary => false,
            QVMType::UnitaryAndGradient => false,
            QVMType::UnitaryGradientAndHessian => true,
        }
    }
}

pub struct QVM<C: ComplexScalar> {
    first_run: bool,
    static_instructions: Vec<SpecializedInstruction>,
    dynamic_instructions: Vec<SpecializedInstruction>,
    memory: Memory<C>,
    qvm_type: QVMType,
}

impl<C: ComplexScalar> QVM<C> {
    pub fn new(program: Bytecode, ty: QVMType) -> Self {
        let (sinsts, dinsts, mem_size) = program.specialize::<C>(ty);

        Self {
            first_run: true,
            static_instructions: sinsts,
            dynamic_instructions: dinsts,
            memory: alloc_memory::<C>(mem_size),
            qvm_type: ty,
        }
    }

    #[inline(always)]
    fn first_run(&mut self) {
        if !self.first_run {
            return;
        }

        // Warm up necessary unitary buffers to identity
        // TODO: Evaluate if any other buffers need to be warmed up here
        for inst in self.static_instructions.iter() {
            if let SpecializedInstruction::Write(w) = inst {
                let mut matmut = w.buffer.as_matmut(&mut self.memory);
                for i in 0..matmut.nrows() {
                    matmut.write(i, i, C::one());
                }
            }
        }

        for inst in self.dynamic_instructions.iter() {
            if let SpecializedInstruction::Write(w) = inst {
                let mut matmut = w.buffer.as_matmut(&mut self.memory);
                for i in 0..matmut.nrows() {
                    matmut.write(i, i, C::one());
                }
            }
        }

        // Evaluate static code
        for inst in &self.static_instructions {
            inst.execute_unitary::<C>(&[], &mut self.memory);
            // TODO: what happens if all code is static?
        }

        self.first_run = false;
    }

    pub fn get_unitary(&mut self, params: &[C::Re]) -> MatRef<C> {
        self.first_run();

        for inst in &self.dynamic_instructions {
            inst.execute_unitary::<C>(params, &mut self.memory);
        }

        match &self.dynamic_instructions[self.dynamic_instructions.len() - 1] {
            SpecializedInstruction::Write(w) => {
                w.buffer.as_matref(&mut self.memory)
            },
            SpecializedInstruction::Matmul(m) => {
                m.out.as_matref(&mut self.memory)
            },
            SpecializedInstruction::Kron(k) => {
                k.out.as_matref(&mut self.memory)
            },
            SpecializedInstruction::FRPR(f) => {
                f.out.as_matref(&mut self.memory)
            },
        }
    }

    pub fn get_unitary_and_gradient(
        &mut self,
        params: &[C::Re],
    ) -> (MatRef<C>, MatGradRef<C>) {
        if !self.qvm_type.gradient_capable() {
            panic!("QVM is not gradient capable, cannot calculate gradient.");
        }

        self.first_run();

        for inst in &self.dynamic_instructions {
            inst.execute_unitary_and_gradient::<C>(params, &mut self.memory);
        }

        match &self.dynamic_instructions[self.dynamic_instructions.len() - 1] {
            SpecializedInstruction::Write(w) => (
                w.buffer.as_matref(&mut self.memory),
                w.buffer.as_matgradref(&mut self.memory),
            ),
            SpecializedInstruction::Matmul(m) => (
                m.out.as_matref(&mut self.memory),
                m.out.as_matgradref(&mut self.memory),
            ),
            SpecializedInstruction::Kron(k) => (
                k.out.as_matref(&mut self.memory),
                k.out.as_matgradref(&mut self.memory),
            ),
            SpecializedInstruction::FRPR(f) => (
                f.out.as_matref(&mut self.memory),
                f.out.as_matgradref(&mut self.memory),
            ),
        }
    }

    pub fn write_unitary(&mut self, params: &[C::Re], mut out_utry: MatMut<C>) {
        self.first_run();

        for inst in
            &self.dynamic_instructions[..self.dynamic_instructions.len() - 1]
        {
            inst.execute_unitary::<C>(params, &mut self.memory);
        }

        match &self.dynamic_instructions[self.dynamic_instructions.len() - 1] {
            SpecializedInstruction::Write(w) => {
                w.execute_unitary_into::<C>(params, &mut self.memory, out_utry)
            },
            SpecializedInstruction::Matmul(m) => {
                m.execute_unitary_into::<C>(&mut self.memory, out_utry)
            },
            SpecializedInstruction::Kron(k) => {
                k.execute_unitary_into::<C>(&mut self.memory, out_utry)
            },
            SpecializedInstruction::FRPR(f) => {
                let input_matref = f.input.as_matref::<C>(&mut self.memory);
                unsafe {
                    fused_reshape_permuted_reshape_into_impl(
                        input_matref,
                        f.out.as_matmut::<C>(&mut self.memory),
                        &f.ins[..f.len],
                        &f.outs[..f.len],
                        &f.dims[..f.len],
                    );
                }

                // CODE SMELL: Read after write aliasing; no UB yet, but lets get rid of this asap
                let out_matref = f.out.as_matref::<C>(&mut self.memory);

                // TODO: In buffer optimization, track output buffer, ensure it lines up with faer
                // standards to avoid this:
                // Need to manually copy the data over since the col_stride of out_utry may be
                // different than the frpr is designed for... bummer
                for i in 0..out_matref.nrows() {
                    for j in 0..out_matref.ncols() {
                        out_utry.write(i, j, out_matref.read(i, j));
                    }
                }
            },
        }
    }

    pub fn write_unitary_and_gradient(
        &mut self,
        params: &[C::Re],
        mut out_utry: MatMut<C>,
        mut out_grad: MatGradMut<C>,
    ) {
        if !self.qvm_type.gradient_capable() {
            panic!("QVM is not gradient capable, cannot calculate gradient.");
        }

        self.first_run();

        for inst in
            &self.dynamic_instructions[..self.dynamic_instructions.len() - 1]
        {
            inst.execute_unitary_and_gradient::<C>(params, &mut self.memory);
        }

        match &self.dynamic_instructions[self.dynamic_instructions.len() - 1] {
            SpecializedInstruction::Write(w) => w
                .execute_unitary_and_gradient_into::<C>(
                    params,
                    &mut self.memory,
                    out_utry,
                    out_grad,
                ),
            SpecializedInstruction::Matmul(m) => m
                .execute_unitary_and_gradient_into::<C>(
                    &mut self.memory,
                    out_utry,
                    out_grad,
                ),
            SpecializedInstruction::Kron(k) => k
                .execute_unitary_and_gradient_into::<C>(
                    &mut self.memory,
                    out_utry,
                    out_grad,
                ),
            SpecializedInstruction::FRPR(f) => {
                let input_matref = f.input.as_matref::<C>(&mut self.memory);
                let out_matmut = f.out.as_matmut::<C>(&mut self.memory);
                unsafe {
                    fused_reshape_permuted_reshape_into_impl(
                        input_matref,
                        out_matmut,
                        &f.ins[..f.len],
                        &f.outs[..f.len],
                        &f.dims[..f.len],
                    );
                }

                // CODE SMELL: Read after write aliasing; no UB yet, but lets get rid of this asap
                let out_matref = f.out.as_matref::<C>(&mut self.memory);

                // TODO: Seriously, get on this
                // TODO: In buffer optimization, track output buffer, ensure it lines up with faer
                // standards to avoid this:
                // Need to manually copy the data over since the col_stride of out_utry may be
                // different than the frpr is designed for... bummer
                for i in 0..out_matref.nrows() {
                    for j in 0..out_matref.ncols() {
                        out_utry.write(i, j, out_matref.read(i, j));
                    }
                }

                for i in 0..f.input.num_params as isize {
                    let input_gradref =
                        f.input.as_matref::<C>(&mut self.memory);
                    let out_gradmut = f.out.as_matmut::<C>(&mut self.memory);
                    unsafe {
                        fused_reshape_permuted_reshape_into_impl(
                            input_gradref,
                            out_gradmut,
                            &f.ins[..f.len],
                            &f.outs[..f.len],
                            &f.dims[..f.len],
                        );
                    }
                    // CODE SMELL: Read after write aliasing; no UB yet, but lets get rid of this asap
                    let out_gradref = f.out.as_matref::<C>(&mut self.memory);

                    // TODO: In buffer optimization, track output buffer, ensure it lines up with faer
                    // standards to avoid this:
                    // Need to manually copy the data over since the col_stride of out_utry may be
                    // different than the frpr is designed for... bummer
                    for r in 0..out_gradref.nrows() {
                        for c in 0..out_gradref.ncols() {
                            out_grad.write(
                                i as usize,
                                r,
                                c,
                                out_gradref.read(r, c),
                            );
                        }
                    }
                }
            },
        }
    }

    pub fn write_unitary_gradient_and_hessian(
        &mut self,
        params: &[C::Re],
        mut out_utry: MatMut<C>,
        mut out_grad: MatGradMut<C>,
        mut out_hess: MatHessMut<C>,
    ) {
        if !self.qvm_type.hessian_capable() {
            panic!("QVM is not gradient capable, cannot calculate gradient.");
        }

        self.first_run();

        for inst in
            &self.dynamic_instructions[..self.dynamic_instructions.len() - 1]
        {
            inst.execute_unitary_gradient_and_hessian::<C>(
                params,
                &mut self.memory,
            );
        }

        match &self.dynamic_instructions[self.dynamic_instructions.len() - 1] {
            SpecializedInstruction::Write(w) => w
                .execute_unitary_gradient_and_hessian_into::<C>(
                    params,
                    &mut self.memory,
                    out_utry,
                    out_grad,
                    out_hess,
                ),
            SpecializedInstruction::Matmul(m) => m
                .execute_unitary_gradient_and_hessian_into::<C>(
                    &mut self.memory,
                    out_utry,
                    out_grad,
                    out_hess,
                ),
            SpecializedInstruction::Kron(k) => k
                .execute_unitary_gradient_and_hessian_into::<C>(
                    &mut self.memory,
                    out_utry,
                    out_grad,
                    out_hess,
                ),
            SpecializedInstruction::FRPR(f) => {
                f.execute_unitary_gradient_and_hessian::<C>(&mut self.memory);

                // CODE SMELL: Read after write aliasing; no UB yet, but lets get rid of this asap
                let out_matref = f.out.as_matref::<C>(&mut self.memory);

                // TODO: Seriously, get on this
                // TODO: In buffer optimization, track output buffer, ensure it lines up with faer
                // standards to avoid this:
                // Need to manually copy the data over since the col_stride of out_utry may be
                // different than the frpr is designed for... bummer
                for i in 0..out_matref.nrows() {
                    for j in 0..out_matref.ncols() {
                        out_utry.write(i, j, out_matref.read(i, j));
                    }
                }

                for i in 0..f.input.num_params as isize {
                    // CODE SMELL: Read after write aliasing; no UB yet, but lets get rid of this asap
                    let out_gradref = f.out.as_matref::<C>(&mut self.memory);

                    // TODO: In buffer optimization, track output buffer, ensure it lines up with faer
                    // standards to avoid this:
                    // Need to manually copy the data over since the col_stride of out_utry may be
                    // different than the frpr is designed for... bummer
                    for r in 0..out_gradref.nrows() {
                        for c in 0..out_gradref.ncols() {
                            out_grad.write(
                                i as usize,
                                r,
                                c,
                                out_gradref.read(r, c),
                            );
                        }
                    }
                }

                // TODO: URGENT: BAD: WARNING: BUG: FIX: Since I removed the
                // matrix index to as_matref this hack doesn't even work now.
                // Seriouslly fix this.

                for p1 in 0..f.input.num_params as isize {
                    for p2 in p1..f.input.num_params as isize {
                        // CODE SMELL: Read after write aliasing; no UB yet, but lets get rid of this asap
                        let out_hessref =
                            f.out.as_matref::<C>(&mut self.memory);

                        // TODO: In buffer optimization, track output buffer, ensure it lines up with faer
                        // standards to avoid this:
                        // Need to manually copy the data over since the col_stride of out_utry may be
                        // different than the frpr is designed for... bummer
                        for r in 0..out_hessref.nrows() {
                            for c in 0..out_hessref.ncols() {
                                out_hess.write(
                                    p1 as usize,
                                    p2 as usize,
                                    r,
                                    c,
                                    out_hessref.read(r, c),
                                );
                            }
                        }
                    }
                }
            },
        }
    }
}

// TODO: TEST: No params in entire circuit, constant everything
