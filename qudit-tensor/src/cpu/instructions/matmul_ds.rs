use qudit_core::matrix::{MatMut, MatRef};
use qudit_core::array::{TensorRef, TensorMut, SymSqTensorMut, SymSqTensorRef};
use qudit_core::{memory, ComplexScalar};
use super::super::buffer::SizedTensorBuffer;
use qudit_expr::DifferentiationLevel;
use qudit_expr::{FUNCTION, GRADIENT, HESSIAN};
use qudit_core::memory::MemoryBuffer;
use qudit_core::accel::MatMulPlan;

pub struct DependentSingleMatmulStruct<C: ComplexScalar> {
    pub left: SizedTensorBuffer<C>,
    pub right: SizedTensorBuffer<C>,
    pub out: SizedTensorBuffer<C>,
    pub left_shared_params: Vec<usize>,
    pub right_shared_params: Vec<usize>,
    pub plan: MatMulPlan<C>,
}

impl<C: ComplexScalar> DependentSingleMatmulStruct<C> {
    pub fn new(
        left: SizedTensorBuffer<C>,
        right: SizedTensorBuffer<C>,
        out: SizedTensorBuffer<C>,
        left_shared_params: Vec<usize>,
        right_shared_params: Vec<usize>,
    ) -> Self {
        let plan = MatMulPlan::new(left.nrows(), right.ncols(), left.ncols());
        Self {
            left,
            right,
            out,
            left_shared_params,
            right_shared_params,
            plan,
        }
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
            if self.left_shared_params.contains(&i) {
                continue;
            }

            let left_gradref = left_grad.subtensor_ref_unchecked(i);
            let out_gradmut = out.subtensor_mut_unchecked(grad_idx);

            self.plan.execute_unchecked(
                left_gradref,
                right_utry,
                out_gradmut,
            );

            grad_idx += 1;
        }

        let param_pairs = self.left_shared_params.iter().zip(self.right_shared_params.iter());
        for (left_idx, right_idx) in param_pairs {
            let left_gradref = left_grad.mat_ref(*left_idx);
            let right_gradref = right_grad.mat_ref(*right_idx);
            let out_gradmut = out.mat_mut(grad_idx);

            self.plan.execute_unchecked(
                left_gradref,
                right_utry,
                out_gradmut,
            );

            let out_gradmut = out.mat_mut(grad_idx);

            self.plan.execute_add_unchecked(
                left_utry,
                right_gradref,
                out_gradmut,
            );

            grad_idx += 1;
        }

        for i in 0..self.right.num_params {
            if self.right_shared_params.contains(&i) {
                continue;
            }
            let right_gradref = right_grad.mat_ref(i);
            let out_gradmut = out.mat_mut(grad_idx);

            self.plan.execute_unchecked(
                left_utry,
                right_gradref,
                out_gradmut,
            );

            grad_idx += 1;
        }
    }

    #[inline(always)]
    fn calculate_hessian(
        &self,
        left_utry: MatRef<C>,
        left_grad: MatVecRef<C>,
        left_hess: SymSqMatMatRef<C>,
        right_utry: MatRef<C>,
        right_grad: MatVecRef<C>,
        right_hess: SymSqMatMatRef<C>,
        out: SymSqMatMatMut<C>,
    ) {
        todo!()
    }

    #[inline(always)]
    pub fn evaluate(&self, memory: &mut MemoryBuffer<C>) {
        if self.left.is_tensor3d() {
            let left_matvecref = self.left.as_matvecref_non_gradient::<C>(memory);
            let right_matvecref = self.right.as_matvecref_non_gradient::<C>(memory);
            let mut out_matvecmut = self.out.as_matvecmut_non_gradient::<C>(memory);
            for m in 0..self.left.nmats() {
                let left_matref = left_matvecref.mat_ref(m);
                let right_matref = right_matvecref.mat_ref(m);
                let out_matmut = out_matvecmut.mat_mut(m);
                self.calculate_unitary(left_matref, right_matref, out_matmut);
            }
        } else {
            let left_matref = self.left.as_matref::<C>(memory);
            let right_matref = self.right.as_matref::<C>(memory);
            let out_matmut = self.out.as_matmut::<C>(memory);
            self.calculate_unitary(left_matref, right_matref, out_matmut);
        }
    }

    #[inline(always)]
    pub fn evaluate_gradient(
        &self,
        memory: &mut MemoryBuffer<C>,
    ) {
        if self.left.is_tensor3d() {
            todo!()
        } else {
            let left_matref = self.left.as_matref::<C>(memory);
            let left_matgradref = self.left.as_matvecref::<C>(memory);
            let right_matref = self.right.as_matref::<C>(memory);
            let right_matgradref = self.right.as_matvecref::<C>(memory);
            let out_matmut = self.out.as_matmut::<C>(memory);
            let out_matgradmut = self.out.as_matvecmut::<C>(memory);
            self.calculate_unitary(left_matref, right_matref, out_matmut);
            self.calculate_gradient(
                left_matref,
                left_matgradref,
                right_matref,
                right_matgradref,
                out_matgradmut,
            );
        }
    }

    #[inline(always)]
    pub fn evaluate_hessian(
        &self,
        memory: &mut MemoryBuffer<C>,
    ) {
        todo!()
        // let left_matref = self.left.as_matref::<C>(memory);
        // let left_matgradref = self.left.as_matvecref::<C>(memory);
        // let left_mathessref = self.left.as_symsqmatref::<C>(memory);
        // let right_matref = self.right.as_matref::<C>(memory);
        // let right_matgradref = self.right.as_matvecref::<C>(memory);
        // let right_mathessref = self.right.as_symsqmatref::<C>(memory);
        // let out_matmut = self.out.as_matmut::<C>(memory);
        // let out_matgradmut = self.out.as_matvecmut::<C>(memory);
        // let out_mathessmut = self.out.as_symsqmatmut::<C>(memory);
        // self.calculate_unitary(left_matref, right_matref, out_matmut);
        // self.calculate_gradient(
        //     left_matref,
        //     left_matgradref.clone(),
        //     right_matref,
        //     right_matgradref.clone(),
        //     out_matgradmut,
        // );
        // self.calculate_hessian(
        //     left_matref,
        //     left_matgradref,
        //     left_mathessref,
        //     right_matref,
        //     right_matgradref,
        //     right_mathessref,
        //     out_mathessmut,
        // );
    }
}
