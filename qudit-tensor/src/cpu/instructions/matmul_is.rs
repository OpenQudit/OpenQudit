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
    left: SizedTensorBuffer<C>,
    right: SizedTensorBuffer<C>,
    out: SizedTensorBuffer<C>,
    // left_offset, r_offset, out_off, dependent_variable, l2_off, r2_off
    grad_offset_map: Vec<(usize, usize, usize, bool, usize, usize)>,
    plan: MatMulPlan<C>,
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
            offset_map.entry(param).and_modify(|offs| {offs.2 = true; offs.3 = left.offset(); offs.4 = right.offset() + right.unit_memory_size()*(i+1);}).or_insert((left.offset(), right.offset() + right.unit_memory_size()*(i+1), false, 0, 0));
        }
        let mut vec = offset_map.into_iter().collect::<Vec<_>>();
        vec.sort();
        let grad_offset_map = vec.into_iter().enumerate().map(|(i, (_, (l_off, r_off, prod, l2_off, r2_off)))| (l_off, r_off, out.offset() + out.unit_memory_size()*(i+1), prod, l2_off, r2_off)).collect::<Vec<_>>();

        let mut offset_map = BTreeMap::new();
        for (i, param_i) in left_param_map.iter().enumerate() {
            for (j, param_j) in left_param_map.iter().enumerate() {
                if param_i > param_j {
                    continue
                }
                let k = if i <= j { j * (j + 1) / 2 + i } else { i * (i + 1) / 2 + j };
                offset_map.insert((param_i, param_j), (left.offset() + left.grad_memory_size() + left.unit_memory_size()*(k+1), right.offset(), false, 0, 0, left.offset() + left.unit_memory_size()*(i+1), 0, left.offset() + left.unit_memory_size()*(j+1), 0));
            }
        }

        for (i, param_i) in right_param_map.iter().enumerate() {
            for (j, param_j) in right_param_map.iter().enumerate() {
                if param_i > param_j {
                    continue
                }
                let k = if i <= j { j * (j + 1) / 2 + i } else { i * (i + 1) / 2 + j };
                offset_map.entry((param_i, param_j))
                    .and_modify(|offs| {
                        offs.2 = true;
                        offs.3 = left.offset();
                        offs.4 = right.offset() + right.grad_memory_size() + right.unit_memory_size()*(k+1);
                        offs.6 = right.offset() + left.unit_memory_size()*(j*1);
                        offs.8 = right.offset() + left.unit_memory_size()*(i*1);
                    }).or_insert((left.offset(), right.offset() + left.grad_memory_size() + left.unit_memory_size()*(k+1), false, 0, 0, 0, right.offset() + right.unit_memory_size()*(j+1), 0, right.offset() + right.unit_memory_size()*(i+1)));
            }
        }
        for (i, param_i) in left_param_map.iter().enumerate() {
            for (j, param_j) in right_param_map.iter().enumerate() {
                offset_map.entry((param_i, param_j))
                    // DOUBLE CHECK after sleep:
                    // if offset_map already contians this then param_i in right and param_j in left
                    // since its shared, I don't do anything here and just skip
                    // This spot is for partial of A(x)*B(y) w.r.t (x,y) so A'(x)B'(x)
                    // TODO: also after sleep double check all the indices (left.offset() is used
                    // everywhere where right.offset() should be)
                    .or_insert((left.offset() + left.unit_memory_size()*(i+1), right.offset() + right.unit_memory_size()*(j+1), false, 0, 0, 0, 0, 0, 0));
            }
        }


            // for (l_off, r_off, o_off, l2_off, r2_off, prod) in self.offset_map() {
        let plan = MatMulPlan::new(left.nrows(), right.ncols(), left.ncols());
        Self { left, right, out, grad_offset_map, plan }
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
        let left = memory.as_ptr();
        let right = memory.as_ptr();
        let out = memory.as_ptr();
        self.plan.execute_unchecked(left, right, out);

        if D >= GRADIENT {
            for (l_off, r_off, o_off, prod, l2_off, r2_off) in self.grad_offset_map() {
                let left = memory.as_ptr().add(l_off);
                let right = memory.as_ptr().add(r_off);
                let out = memory.as_ptr().add(o_off);
                self.plan.execute_unchecked(left, right, out);

                if prod {    
                    let left = memory.as_ptr().add(l2_off);
                    let right = memory.as_ptr().add(r2_off);
                    self.plan.execute_add_unchecked(left, right, out);
                }
            }
        }

        if D >= HESSIAN {
            for (l_off, r_off, o_off, prod, l2_off, r2_off, l3_off, r3_off) in self.hess_offset_map() {
                let left = memory.as_ptr().add(l_off);
                let right = memory.as_ptr().add(r_off);
                let out = memory.as_ptr().add(o_off);
                self.plan.execute_unchecked(left, right, out);

                if prod {    
                    let left = memory.as_ptr().add(l2_off);
                    let right = memory.as_ptr().add(r2_off);
                    self.plan.execute_add_unchecked(left, right, out);

                    let left = memory.as_ptr().add(l3_off);
                    let right = memory.as_ptr().add(r3_off);
                    self.plan.execute_add_unchecked(left, right, out, alpha = 2); // Cannot do
                    // alpha = 2, must do two matmuls for xy case (alpha=2 only works in xx case)
                }
            }
        }
    }
}

