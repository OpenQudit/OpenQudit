// use aligned_vec::{avec, AVec};
// use bytemuck::Zeroable;
use faer::reborrow::ReborrowMut;
use qudit_expr::DifferentiationLevel;
use qudit_expr::Module;

use super::bytecode::Bytecode;
use super::bytecode::SpecializedInstruction;
use qudit_core::accel::fused_reshape_permute_reshape_into_impl;
use qudit_core::matrix::MatVecMut;
use qudit_core::matrix::MatVecRef;
use qudit_core::matrix::SymSqMatMatMut;
use qudit_core::matrix::MatMut;
use qudit_core::matrix::MatRef;
use qudit_core::memory::MemoryBuffer;
use qudit_core::memory::alloc_zeroed_memory;
use qudit_core::ComplexScalar;

pub struct QVM<C: ComplexScalar> {
    first_run: bool,
    static_instructions: Vec<SpecializedInstruction<C>>,
    dynamic_instructions: Vec<SpecializedInstruction<C>>,
    #[allow(dead_code)]
    module: Module<C>,
    memory: MemoryBuffer<C>,
    diff_lvl: DifferentiationLevel,
}

impl<C: ComplexScalar> QVM<C> {
    pub fn new(program: Bytecode, diff_lvl: DifferentiationLevel) -> Self {
        let (sinsts, dinsts, module, mem_size) = program.specialize::<C>(diff_lvl);

        Self {
            first_run: true,
            static_instructions: sinsts,
            dynamic_instructions: dinsts,
            module,
            memory: alloc_zeroed_memory::<C>(mem_size),
            diff_lvl,
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
                    *matmut.rb_mut().get_mut(i, i) = C::one();
                }
            }
        }

        for inst in self.dynamic_instructions.iter() {
            if let SpecializedInstruction::Write(w) = inst {
                let mut matmut = w.buffer.as_matmut(&mut self.memory);
                for i in 0..matmut.nrows() {
                    *matmut.rb_mut().get_mut(i, i) = C::one();
                }
            }
        }

        // Evaluate static code
        for inst in &self.static_instructions {
            inst.execute_unitary(&[], &mut self.memory);
            // TODO: what happens if all code is static?
        }

        self.first_run = false;
    }

    pub fn get_unitary(&mut self, params: &[C::R]) -> MatRef<C> {
        self.first_run();

        for inst in &self.dynamic_instructions {
            inst.execute_unitary(params, &mut self.memory);
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
        params: &[C::R],
    ) -> (MatRef<C>, MatVecRef<C>) {
        if !self.diff_lvl.gradient_capable() {
            panic!("QVM is not gradient capable, cannot calculate gradient.");
        }

        self.first_run();

        for inst in &self.dynamic_instructions {
            inst.execute_unitary_and_gradient(params, &mut self.memory);
        }

        match &self.dynamic_instructions[self.dynamic_instructions.len() - 1] {
            SpecializedInstruction::Write(w) => (
                w.buffer.as_matref(&mut self.memory),
                w.buffer.as_matvecref(&mut self.memory),
            ),
            SpecializedInstruction::Matmul(m) => (
                m.out.as_matref(&mut self.memory),
                m.out.as_matvecref(&mut self.memory),
            ),
            SpecializedInstruction::Kron(k) => (
                k.out.as_matref(&mut self.memory),
                k.out.as_matvecref(&mut self.memory),
            ),
            SpecializedInstruction::FRPR(f) => (
                f.out.as_matref(&mut self.memory),
                f.out.as_matvecref(&mut self.memory),
            ),
        }
    }

    pub fn write_unitary(&mut self, params: &[C::R], mut out_utry: MatMut<C>) {
        self.first_run();

        for inst in
            &self.dynamic_instructions[..self.dynamic_instructions.len() - 1]
        {
            inst.execute_unitary(params, &mut self.memory);
        }

        match &self.dynamic_instructions[self.dynamic_instructions.len() - 1] {
            SpecializedInstruction::Write(w) => {
                w.execute_unitary_into(params, &mut self.memory, out_utry)
            },
            SpecializedInstruction::Matmul(m) => {
                m.execute_unitary_into(&mut self.memory, out_utry)
            },
            SpecializedInstruction::Kron(k) => {
                k.execute_unitary_into(&mut self.memory, out_utry)
            },
            SpecializedInstruction::FRPR(f) => {
                let input_matref = f.input.as_matref(&mut self.memory);
                unsafe {
                    fused_reshape_permute_reshape_into_impl(
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
                        *out_utry.rb_mut().get_mut(i, j) = out_matref[(i, j)];
                    }
                }
            },
        }
    }

    pub fn write_unitary_and_gradient(
        &mut self,
        params: &[C::R],
        mut out_utry: MatMut<C>,
        mut out_grad: MatVecMut<C>,
    ) {
        if !self.diff_lvl.gradient_capable() {
            panic!("QVM is not gradient capable, cannot calculate gradient.");
        }

        self.first_run();

        for inst in
            &self.dynamic_instructions[..self.dynamic_instructions.len() - 1]
        {
            inst.execute_unitary_and_gradient(params, &mut self.memory);
        }

        match &self.dynamic_instructions[self.dynamic_instructions.len() - 1] {
            SpecializedInstruction::Write(w) => w
                .execute_unitary_and_gradient_into(
                    params,
                    &mut self.memory,
                    out_utry,
                    out_grad,
                ),
            SpecializedInstruction::Matmul(m) => m
                .execute_unitary_and_gradient_into(
                    &mut self.memory,
                    out_utry,
                    out_grad,
                ),
            SpecializedInstruction::Kron(k) => k
                .execute_unitary_and_gradient_into(
                    &mut self.memory,
                    out_utry,
                    out_grad,
                ),
            SpecializedInstruction::FRPR(f) => {
                let input_matref = f.input.as_matref::<C>(&mut self.memory);
                let out_matmut = f.out.as_matmut(&mut self.memory);
                unsafe {
                    fused_reshape_permute_reshape_into_impl(
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
                        *out_utry.rb_mut().get_mut(i, j) = out_matref[(i, j)];
                    }
                }

                for i in 0..f.input.num_params as isize {
                    let input_gradref =
                        f.input.as_matref::<C>(&mut self.memory);
                    let out_gradmut = f.out.as_matmut::<C>(&mut self.memory);
                    unsafe {
                        fused_reshape_permute_reshape_into_impl(
                            input_gradref,
                            out_gradmut,
                            &f.ins[..f.len],
                            &f.outs[..f.len],
                            &f.dims[..f.len],
                        );
                    }
                    // CODE SMELL: Read after write aliasing; no UB yet, but lets get rid of this asap
                    let out_gradref = f.out.as_matref(&mut self.memory);

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
                                out_gradref[(r, c)],
                            );
                        }
                    }
                }
            },
        }
    }

    pub fn write_unitary_gradient_and_hessian(
        &mut self,
        params: &[C::R],
        mut out_utry: MatMut<C>,
        mut out_grad: MatVecMut<C>,
        mut out_hess: SymSqMatMatMut<C>,
    ) {
        if !self.diff_lvl.hessian_capable() {
            panic!("QVM is not gradient capable, cannot calculate gradient.");
        }

        self.first_run();

        for inst in
            &self.dynamic_instructions[..self.dynamic_instructions.len() - 1]
        {
            inst.execute_unitary_gradient_and_hessian(
                params,
                &mut self.memory,
            );
        }

        match &self.dynamic_instructions[self.dynamic_instructions.len() - 1] {
            SpecializedInstruction::Write(w) => w
                .execute_unitary_gradient_and_hessian_into(
                    params,
                    &mut self.memory,
                    out_utry,
                    out_grad,
                    out_hess,
                ),
            SpecializedInstruction::Matmul(m) => m
                .execute_unitary_gradient_and_hessian_into(
                    &mut self.memory,
                    out_utry,
                    out_grad,
                    out_hess,
                ),
            SpecializedInstruction::Kron(k) => k
                .execute_unitary_gradient_and_hessian_into(
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
                        *out_utry.rb_mut().get_mut(i, j) = out_matref[(i, j)];
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
                                out_gradref[(r, c)],
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
                                    out_hessref[(r, c)],
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
