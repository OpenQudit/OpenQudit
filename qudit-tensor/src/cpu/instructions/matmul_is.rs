use std::collections::BTreeMap;

use qudit_core::matrix::{MatMut, MatRef};
use qudit_core::array::{TensorRef, TensorMut, SymSqTensorMut, SymSqTensorRef};
use qudit_core::{memory, ComplexScalar, ParamIndices};
use super::super::buffer::SizedTensorBuffer;
use qudit_expr::DifferentiationLevel;
use qudit_expr::{FUNCTION, GRADIENT, HESSIAN};
use qudit_core::memory::MemoryBuffer;
use qudit_core::accel::MatMulPlan;

pub struct IndependentSingleMatmulStruct<C: ComplexScalar> {
    pub left: SizedTensorBuffer<C>,
    pub right: SizedTensorBuffer<C>,
    pub out: SizedTensorBuffer<C>,
    pub plan: MatMulPlan<C>,
}

impl<C: ComplexScalar> IndependentSingleMatmulStruct<C> {
    pub fn new(
        left: SizedTensorBuffer<C>,
        right: SizedTensorBuffer<C>,
        out: SizedTensorBuffer<C>,
        left_param_map: ParamIndices,
        right_param_map: ParamIndices,
    ) -> Self {
        let mut offset_map = BTreeMap::new();
        for (i, param) in left_param_map.iter().enumerate() {
            offset_map.insert(param, (left.offset() + left.unit_memory_size()*(i+1), right.offset(), false, 0, 0));
        }
        for (i, param) in right_param_map.iter().enumerate() {
            offset_map.entry(param).and_modify(|offs| {offs.2 = true; offs.3 = left.offset(); offs.4 = right.offset() + right.unit_memory_size()*(i+1)}).or_insert((left.offset(), right.offset() + right.unit_memory_size()*(i+1), false, 0, 0));
        }
        let mut vec = offset_map.into_iter().collect::<Vec<_>>();
        vec.sort();
        let grad_offset_map = vec.into_iter().enumerate().map(|(i, (_, (l_off, r_off, prod, l2_off, r2_off)))| (l_off, r_off, out.offset() + out.unit_memory_size()*(i+1), prod, l2_off, r2_off)).collect::<Vec<_>>();

            // for (l_off, r_off, o_off, l2_off, r2_off, prod) in self.offset_map() {
        let plan = MatMulPlan::new(left.nrows(), right.ncols(), left.ncols());
        Self { left, right, out, plan }
    }

    #[inline(always)]
    fn calculate_unitary(&self, left: MatRef<C>, right: MatRef<C>, out: MatMut<C>) {
        self.plan.execute_unchecked(
            left,
            right,
            out,
        );
    }

    #[inline(always)]
    unsafe fn calculate_gradient(
        &self,
        left_utry: MatRef<C>,
        left_grad: TensorRef<C, 3>,
        right_utry: MatRef<C>,
        right_grad: TensorRef<C, 3>,
        mut out: TensorMut<C, 3>,
    ) {
        let mut grad_idx = 0;

        for i in 0..self.left.nparams() {
            let left_gradref = left_grad.subtensor_ref_unchecked(i);
            let out_gradmut = out.subtensor_mut_unchecked(grad_idx);

            self.plan.execute_unchecked(
                left_gradref,
                right_utry,
                out_gradmut,
            );

            grad_idx += 1;
        }

        for i in 0..self.right.nparams() {
            let right_gradref = right_grad.subtensor_ref_unchecked(i);
            let out_gradmut = out.subtensor_mut_unchecked(grad_idx);

            self.plan.execute_unchecked(
                left_utry,
                right_gradref,
                out_gradmut,
            );

            grad_idx += 1;
        }
    }

    #[inline(always)]
    unsafe fn calculate_hessian(
        &self,
        left_utry: MatRef<C>,
        left_grad: TensorRef<C, 3>,
        left_hess: SymSqTensorRef<C, 4>,
        right_utry: MatRef<C>,
        right_grad: TensorRef<C, 3>,
        right_hess: SymSqTensorRef<C, 4>,
        mut out: SymSqTensorMut<C, 4>,
    ) {
        // Upper left block: left_hess * right_utry
        let left_nparams = left_hess.dims()[0];
        for left_hess_row in 0..left_nparams {
            for left_hess_col in left_hess_row..left_nparams {
                let left_hess_ref = left_hess.subtensor_ref_unchecked(left_hess_row, left_hess_col);
                let hess_ref = out.subtensor_mut_unchecked(left_hess_row, left_hess_col);
                self.plan.execute_unchecked(
                    left_hess_ref,
                    right_utry,
                    hess_ref,
                );
            }
        }

        // Lower right block: left_utry * right_hess
        let right_nparams = right_hess.dims()[0];
        for right_hess_row in 0..right_nparams {
            for right_hess_col in right_hess_row..right_nparams {
                let right_hess_ref = right_hess.subtensor_ref_unchecked(right_hess_row, right_hess_col);
                let hess_ref = out.subtensor_mut_unchecked(
                    left_nparams + right_hess_row,
                    left_nparams + right_hess_col,
                );
                self.plan.execute_unchecked(
                    left_utry,
                    right_hess_ref,
                    hess_ref,
                );
            }
        }

        // Upper right block: right_grad * left_grad
        for left_grad_row in 0..left_nparams {
            let left_grad_ref = left_grad.subtensor_ref_unchecked(left_grad_row);
            for right_grad_col in 0..right_nparams {
                let right_grad_ref = right_grad.subtensor_ref_unchecked(right_grad_col);
                let hess_ref = out.subtensor_mut_unchecked(
                    left_grad_row,
                    left_nparams + right_grad_col,
                );
                self.plan.execute_unchecked(
                    left_grad_ref,
                    right_grad_ref,
                    hess_ref,
                );
            }
        }
    }

    #[inline(always)]
    pub fn get_output_buffer(&self) -> &SizedTensorBuffer<C> {
        &self.out
    }

    #[inline(always)]
    pub unsafe fn evaluate<const D: DifferentiationLevel>(&self, memory: &mut MemoryBuffer<C>) {
        let left = self.left.as_matrix_ref(memory);
        let right = self.right.as_matrix_ref(memory);
        let out = self.out.as_matrix_mut(memory);
        self.calculate_unitary(left, right, out);

        if D >= GRADIENT {

            // for (l_off, r_off, o_off, l2_off, r2_off, prod) in self.offset_map() {
            //     let left = memory.as_ptr().add(l_off);
            //     let right = memory.as_ptr().add(r_off);
            //     let out = memory.as_ptr().add(o_off);
            //     self.plan.execute_unchecked(left, right, out);

            //     if prod {    
            //         let left = memory.as_ptr().add(l2_off);
            //         let right = memory.as_ptr().add(r2_off);
            //         self.plan.execute_add_unchecked(left, right, out);
            //     }
            // }

            let left_grad = self.left.grad_as_tensor3d_ref(memory);
            let right_grad = self.right.grad_as_tensor3d_ref(memory);
            let out_grad = self.out.grad_as_tensor3d_mut(memory);
            self.calculate_gradient(
                left,
                left_grad,
                right,
                right_grad,
                out_grad,
            );

            if D >= HESSIAN {
                let left_hess = self.left.hess_as_symsq_tensor4d_ref(memory);
                let right_hess = self.right.hess_as_symsq_tensor4d_ref(memory);
                let out_hess = self.out.hess_as_symsq_tensor4d_mut(memory);
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

