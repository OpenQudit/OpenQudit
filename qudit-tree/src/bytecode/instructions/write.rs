use qudit_core::matrix::MatMut;
use qudit_core::matrix::SymSqMatMatMut;
use qudit_core::matrix::MatVecMut;
use qudit_core::ComplexScalar;
use crate::bytecode::SizedMatrixBuffer;
use qudit_core::memory::MemoryBuffer;
use qudit_expr::UtryFunc;
use qudit_expr::UtryGradFunc;

pub struct WriteStruct<C: ComplexScalar> {
    pub utry_fn: UtryFunc<C>,
    pub utry_grad_fn: Option<UtryGradFunc<C>>,
    pub idx: usize,
    pub buffer: SizedMatrixBuffer,
}

impl<C: ComplexScalar> WriteStruct<C> {
    pub fn new(utry_fn: UtryFunc<C>, utry_grad_fn: Option<UtryGradFunc<C>>, idx: usize, buffer: SizedMatrixBuffer) -> Self {
        Self { utry_fn, utry_grad_fn, idx, buffer }
    }

    #[inline(always)]
    pub fn execute_unitary(
        &self,
        params: &[C::R],
        memory: &mut MemoryBuffer<C>,
    ) {
        let gate_params =
            &params[self.idx..self.idx + self.buffer.num_params];
        let matmut = self.buffer.as_matmut::<C>(memory);
        unsafe {
            let matmutptr = matmut.as_ptr_mut() as *mut C::R;
            (self.utry_fn)(gate_params.as_ptr() as *const C::R, matmutptr);
        }
    }

    #[inline(always)]
    pub fn execute_unitary_and_gradient(
        &self,
        params: &[C::R],
        memory: &mut MemoryBuffer<C>,
    ) {
        let gate_params =
            &params[self.idx..self.idx + self.buffer.num_params];
        let matmut = self.buffer.as_matmut::<C>(memory);
        let matgradmut = self.buffer.as_matvecmut::<C>(memory);
        unsafe {
            let matmutptr = matmut.as_ptr_mut() as *mut C::R;
            let matgradmutptr = matgradmut.as_mut_ptr().as_ptr() as *mut C::R;
            self.utry_grad_fn.unwrap()(gate_params.as_ptr() as *const C::R, matmutptr, matgradmutptr);
        }
    }

    #[inline(always)]
    pub fn execute_unitary_gradient_and_hessian(
        &self,
        _params: &[C::R],
        _memory: &mut MemoryBuffer<C>,
    ) {
        todo!()
        // let gate_params =
        //     &params[self.idx..self.idx + self.gate.get_num_params];
        // let mut matmut = self.buffer.as_matmut::<C>(memory);
        // let mut matgradmut = self.buffer.as_matvecmut::<C>(memory);
        // let mut mathessmut = self.buffer.as_symsqmatmut::<C>(memory);
        // self.gate.write_unitary_gradient_and_hessian(
        //     gate_params,
        //     &mut matmut,
        //     &mut matgradmut,
        //     &mut mathessmut,
        // );
    }

    #[inline(always)]
    pub fn execute_unitary_into(
        &self,
        params: &[C::R],
        _memory: &mut MemoryBuffer<C>,
        out: MatMut<C>,
    ) {
        let gate_params =
            &params[self.idx..self.idx + self.buffer.num_params];
        unsafe {
            let outptr = out.as_ptr_mut() as *mut C::R;
            (self.utry_fn)(gate_params.as_ptr() as *const C::R, outptr);
        }
    }

    #[inline(always)]
    pub fn execute_unitary_and_gradient_into(
        &self,
        params: &[C::R],
        _memory: &mut MemoryBuffer<C>,
        out: MatMut<C>,
        matgradmut: MatVecMut<C>,
    ) {
        let gate_params =
            &params[self.idx..self.idx + self.buffer.num_params];
        unsafe {
            let outptr = out.as_ptr_mut() as *mut C::R;
            let matgradmutptr = matgradmut.as_mut_ptr().as_ptr() as *mut C::R;
            self.utry_grad_fn.unwrap()(gate_params.as_ptr() as *const C::R, outptr, matgradmutptr);
        }
    }

    #[inline(always)]
    pub fn execute_unitary_gradient_and_hessian_into(
        &self,
        _params: &[C::R],
        _memory: &mut MemoryBuffer<C>,
        _out: MatMut<C>,
        _matgradmut: MatVecMut<C>,
        _mathessmut: SymSqMatMatMut<C>,
    ) {
        todo!()
    }
}
