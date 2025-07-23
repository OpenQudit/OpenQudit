use qudit_core::matrix::{MatMut, MatRef};
use qudit_core::array::{TensorRef, TensorMut, SymSqTensorMut, SymSqTensorRef};
use qudit_core::{memory, ComplexScalar};
use super::super::buffer::SizedTensorBuffer;
use qudit_expr::DifferentiationLevel;
use qudit_expr::{FUNCTION, GRADIENT, HESSIAN};
use qudit_core::memory::MemoryBuffer;
use qudit_core::accel::MatMulPlan;

pub struct IndependentBatchMatmulStruct<C: ComplexScalar> {
    pub left: SizedTensorBuffer<C>,
    pub right: SizedTensorBuffer<C>,
    pub out: SizedTensorBuffer<C>,
    pub plan: MatMulPlan<C>,
}

impl<C: ComplexScalar> IndependentBatchMatmulStruct<C> {
    pub fn new(
        left: SizedTensorBuffer<C>,
        right: SizedTensorBuffer<C>,
        out: SizedTensorBuffer<C>,
    ) -> Self {
        let plan = MatMulPlan::new(left.nrows(), right.ncols(), left.ncols());
        Self { left, right, out, plan }
    }

    #[inline(always)]
    unsafe fn calculate_unitary(&self, left: TensorRef<C, 3>, right: TensorRef<C, 3>, mut out: TensorMut<C, 3>) {
        for m in 0..self.left.nmats() {
            let left_mat = left.subtensor_ref_unchecked(m);
            let right_mat = right.subtensor_ref_unchecked(m);
            let out_mat = out.subtensor_mut_unchecked(m);
            self.plan.execute_unchecked(
                left_mat,
                right_mat,
                out_mat,
            );
        }
    }

    #[inline(always)]
    unsafe fn calculate_gradient(
        &self,
        left_utry: TensorRef<C, 3>,
        left_grad: TensorRef<C, 4>,
        right_utry: TensorRef<C, 3>,
        right_grad: TensorRef<C, 4>,
        mut out: TensorMut<C, 4>,
    ) {
        let mut grad_idx = 0;

        for i in 0..self.left.nparams() {
            let left_gradref = left_grad.subtensor_ref_unchecked(i);
            let mut out_gradmut = out.subtensor_mut_unchecked(grad_idx);

            for m in 0..self.left.nmats() {
                let left_mat = left_gradref.subtensor_ref_unchecked(m);
                let right_mat = right_utry.subtensor_ref_unchecked(m);
                let out_mat = out_gradmut.subtensor_mut_unchecked(m);
                self.plan.execute_unchecked(
                    left_mat,
                    right_mat,
                    out_mat,
                );
            }

            grad_idx += 1;
        }

        for i in 0..self.right.nparams() {
            let right_gradref = right_grad.subtensor_ref_unchecked(i);
            let mut out_gradmut = out.subtensor_mut_unchecked(grad_idx);

            for m in 0..self.left.nmats() {
                let left_mat = left_utry.subtensor_ref_unchecked(m);
                let right_mat = right_gradref.subtensor_ref_unchecked(m);
                let out_mat = out_gradmut.subtensor_mut_unchecked(m);
                self.plan.execute_unchecked(
                    left_mat,
                    right_mat,
                    out_mat,
                );
            }

            grad_idx += 1;
        }
    }

    #[inline(always)]
    unsafe fn calculate_hessian(
        &self,
        left_utry: TensorRef<C, 3>,
        left_grad: TensorRef<C, 4>,
        left_hess: SymSqTensorRef<C, 5>,
        right_utry: TensorRef<C, 3>,
        right_grad: TensorRef<C, 4>,
        right_hess: SymSqTensorRef<C, 5>,
        mut out: SymSqTensorMut<C, 5>,
    ) {
        // Upper left block: left_hess * right_utry
        let left_nparams = left_hess.dims()[0];
        for left_hess_row in 0..left_nparams {
            for left_hess_col in left_hess_row..left_nparams {
                let left_hess_ref = left_hess.subtensor_ref_unchecked(left_hess_row, left_hess_col);
                let mut hess_ref = out.subtensor_mut_unchecked(left_hess_row, left_hess_col);

                for m in 0..self.left.nmats() {
                    let left_mat = left_hess_ref.subtensor_ref_unchecked(m);
                    let right_mat = right_utry.subtensor_ref_unchecked(m);
                    let out_mat = hess_ref.subtensor_mut_unchecked(m);
                    self.plan.execute_unchecked(
                        left_mat,
                        right_mat,
                        out_mat,
                    );
                }
            }
        }

        // Lower right block: left_utry * right_hess
        let right_nparams = right_hess.dims()[0];
        for right_hess_row in 0..right_nparams {
            for right_hess_col in right_hess_row..right_nparams {
                let right_hess_ref = right_hess.subtensor_ref_unchecked(right_hess_row, right_hess_col);
                let mut hess_ref = out.subtensor_mut_unchecked(
                    left_nparams + right_hess_row,
                    left_nparams + right_hess_col,
                );

                for m in 0..self.left.nmats() {
                    let left_mat = left_utry.subtensor_ref_unchecked(m);
                    let right_mat = right_hess_ref.subtensor_ref_unchecked(m);
                    let out_mat = hess_ref.subtensor_mut_unchecked(m);
                    self.plan.execute_unchecked(
                        left_mat,
                        right_mat,
                        out_mat,
                    );
                }
            }
        }

        // Upper right block: right_grad * left_grad
        for left_grad_row in 0..left_nparams {
            let left_grad_ref = left_grad.subtensor_ref_unchecked(left_grad_row);
            for right_grad_col in 0..right_nparams {
                let right_grad_ref = right_grad.subtensor_ref_unchecked(right_grad_col);
                let mut hess_ref = out.subtensor_mut_unchecked(
                    left_grad_row,
                    left_nparams + right_grad_col,
                );
                for m in 0..self.left.nmats() {
                    let left_mat = left_grad_ref.subtensor_ref_unchecked(m);
                    let right_mat = right_grad_ref.subtensor_ref_unchecked(m);
                    let out_mat = hess_ref.subtensor_mut_unchecked(m);
                    self.plan.execute_unchecked(
                        left_mat,
                        right_mat,
                        out_mat,
                    );
                }
            }
        }
    }

    #[inline(always)]
    pub fn get_output_buffer(&self) -> &SizedTensorBuffer<C> {
        &self.out
    }

    #[inline(always)]
    pub unsafe fn evaluate<const D: DifferentiationLevel>(&self, memory: &mut MemoryBuffer<C>) {
        let left = self.left.as_tensor3d_ref(memory);
        let right = self.right.as_tensor3d_ref(memory);
        let mut out = self.out.as_tensor3d_mut(memory);
        self.calculate_unitary(left, right, out);

        if D >= GRADIENT {
            let left_grad = self.left.grad_as_tensor4d_ref(memory);
            let right_grad = self.right.grad_as_tensor4d_ref(memory);
            let out_grad = self.out.grad_as_tensor4d_mut(memory);
            self.calculate_gradient(
                left,
                left_grad,
                right,
                right_grad,
                out_grad,
            );

            if D >= HESSIAN {
                let left_hess = self.left.hess_as_symsq_tensor5d_ref(memory);
                let right_hess = self.right.hess_as_symsq_tensor5d_ref(memory);
                let out_hess = self.out.hess_as_symsq_tensor5d_mut(memory);
                self.calculate_hessian(
                    left,
                    left_grad,
                    left_hess,
                    right,
                    right_grad,
                    right_hess,
                    out_hess,
                );
            }
        }
    }
}
