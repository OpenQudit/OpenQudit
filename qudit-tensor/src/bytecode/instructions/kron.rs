use qudit_core::matrix::{MatMut, MatRef};
use qudit_core::matrix::{SymSqMatMatMut, SymSqMatMatRef};
use qudit_core::matrix::{MatVecMut, MatVecRef};
use qudit_core::accel::kron as matrix_kron;
use qudit_core::accel::kron_add as matrix_kron_add;
use qudit_core::ComplexScalar;
use super::super::buffer::SizedTensorBuffer;
use qudit_core::memory::MemoryBuffer;

pub struct DisjointKronStruct {
    pub left: SizedTensorBuffer,
    pub right: SizedTensorBuffer,
    pub out: SizedTensorBuffer,
}

pub struct OverlappingKronStruct {
    pub left: SizedTensorBuffer,
    pub right: SizedTensorBuffer,
    pub out: SizedTensorBuffer,
    pub left_shared_indices: Vec<usize>,
    pub right_shared_indices: Vec<usize>,
}

impl OverlappingKronStruct {
    pub fn new(
        left: SizedTensorBuffer,
        right: SizedTensorBuffer,
        out: SizedTensorBuffer,
        left_shared_indices: Vec<usize>,
        right_shared_indices: Vec<usize>,
    ) -> Self {
        Self { left, right, out, left_shared_indices, right_shared_indices }
    }

    #[inline(always)]
    fn calculate_unitary<C: ComplexScalar>(
        &self,
        left: MatRef<C>,
        right: MatRef<C>,
        out: MatMut<C>,
    ) {
        matrix_kron(left, right, out);
    }

    #[inline(always)]
    fn calculate_gradient<C: ComplexScalar>(
        &self,
        left_utry: MatRef<C>,
        left_grad: MatVecRef<C>,
        right_utry: MatRef<C>,
        right_grad: MatVecRef<C>,
        mut out: MatVecMut<C>,
    ) {
        let mut grad_idx = 0;

        for i in 0..self.left.num_params {
            if self.left_shared_indices.contains(&i) {
                continue;
            }

            let left_gradref = left_grad.mat_ref(i);
            let out_gradmut = out.mat_mut(grad_idx);

            matrix_kron(
                left_gradref,
                right_utry,
                out_gradmut,
            );

            grad_idx += 1;
        }

        let param_pairs = self.left_shared_indices.iter().zip(self.right_shared_indices.iter());
        for (left_idx, right_idx) in param_pairs {
            let left_gradref = left_grad.mat_ref(*left_idx);
            let right_gradref = right_grad.mat_ref(*right_idx);
            let out_gradmut = out.mat_mut(grad_idx);

            matrix_kron(
                left_gradref,
                right_utry,
                out_gradmut,
            );
            let out_gradmut = out.mat_mut(grad_idx);

            matrix_kron_add(
                left_utry,
                right_gradref,
                out_gradmut,
            );

            grad_idx += 1;
        }

        for i in 0..self.right.num_params {
            if self.right_shared_indices.contains(&i) {
                continue;
            }
            let right_gradref = right_grad.mat_ref(i);
            let out_gradmut = out.mat_mut(grad_idx);

            matrix_kron(
                left_utry,
                right_gradref,
                out_gradmut,
            );

            grad_idx += 1;
        }
    }

    #[inline(always)]
    fn calculate_hessian<C: ComplexScalar>(
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
    pub fn evaluate<C: ComplexScalar>(&self, memory: &mut MemoryBuffer<C>) {
        let left_matref = self.left.as_matref::<C>(memory);
        let right_matref = self.right.as_matref::<C>(memory);
        let out_matmut = self.out.as_matmut::<C>(memory);
        self.calculate_unitary(left_matref, right_matref, out_matmut);
    }

    #[inline(always)]
    pub fn evaluate_gradient<C: ComplexScalar>(
        &self,
        memory: &mut MemoryBuffer<C>,
    ) {
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

    #[inline(always)]
    pub fn evaluate_hessian<C: ComplexScalar>(
        &self,
        memory: &mut MemoryBuffer<C>,
    ) {
        let left_matref = self.left.as_matref::<C>(memory);
        let left_matgradref = self.left.as_matvecref::<C>(memory);
        let left_mathessref = self.left.as_symsqmatref::<C>(memory);
        let right_matref = self.right.as_matref::<C>(memory);
        let right_matgradref = self.right.as_matvecref::<C>(memory);
        let right_mathessref = self.right.as_symsqmatref::<C>(memory);
        let out_matmut = self.out.as_matmut::<C>(memory);
        let out_matgradmut = self.out.as_matvecmut::<C>(memory);
        let out_mathessmut = self.out.as_symsqmatmut::<C>(memory);
        self.calculate_unitary(left_matref, right_matref, out_matmut);
        self.calculate_gradient(
            left_matref,
            left_matgradref.clone(),
            right_matref,
            right_matgradref.clone(),
            out_matgradmut,
        );
        self.calculate_hessian(
            left_matref,
            left_matgradref,
            left_mathessref,
            right_matref,
            right_matgradref,
            right_mathessref,
            out_mathessmut,
        );
    }
}

impl DisjointKronStruct {
    pub fn new(
        left: SizedTensorBuffer,
        right: SizedTensorBuffer,
        out: SizedTensorBuffer,
    ) -> Self {
        Self { left, right, out }
    }

    #[inline(always)]
    fn calculate_unitary<C: ComplexScalar>(
        &self,
        left: MatRef<C>,
        right: MatRef<C>,
        out: MatMut<C>,
    ) {
        matrix_kron(left, right, out);
    }

    #[inline(always)]
    fn calculate_gradient<C: ComplexScalar>(
        &self,
        left_utry: MatRef<C>,
        left_grad: MatVecRef<C>,
        right_utry: MatRef<C>,
        right_grad: MatVecRef<C>,
        mut out: MatVecMut<C>,
    ) {
        let mut grad_idx = 0;

        for i in 0..self.left.num_params {
            let left_gradref = left_grad.mat_ref(i);
            let out_gradmut = out.mat_mut(grad_idx);
            matrix_kron(left_gradref, right_utry, out_gradmut);
            grad_idx += 1;
        }

        for i in 0..self.right.num_params {
            let right_gradref = right_grad.mat_ref(i);
            let out_gradmut = out.mat_mut(grad_idx);
            matrix_kron(left_utry, right_gradref, out_gradmut);
            grad_idx += 1;
        }
    }

    #[inline(always)]
    fn calculate_hessian<C: ComplexScalar>(
        &self,
        left_utry: MatRef<C>,
        left_grad: MatVecRef<C>,
        left_hess: SymSqMatMatRef<C>,
        right_utry: MatRef<C>,
        right_grad: MatVecRef<C>,
        right_hess: SymSqMatMatRef<C>,
        out: SymSqMatMatMut<C>,
    ) {
        // Upper left block: right_utry * left_hess
        for left_hess_row in 0..left_hess.nmats() {
            for left_hess_col in left_hess_row..left_hess.nmats() {
                let left_hess_ref =
                    left_hess.mat_ref(left_hess_row, left_hess_col);
                let hess_ref = out.mat_mut(left_hess_row, left_hess_col);
                matrix_kron(left_hess_ref, right_utry, hess_ref);
            }
        }

        // Lower right block: right_hess * left_utry
        for right_hess_row in 0..right_hess.nmats() {
            for right_hess_col in right_hess_row..right_hess.nmats() {
                let right_hess_ref =
                    right_hess.mat_ref(right_hess_row, right_hess_col);
                let hess_ref = out.mat_mut(
                    left_hess.nmats() + right_hess_row,
                    left_hess.nmats() + right_hess_col,
                );
                matrix_kron(left_utry, right_hess_ref, hess_ref);
            }
        }

        // Upper right block: right_grad * left_grad
        for left_grad_row in 0..left_grad.nmats() {
            let left_grad_ref = left_grad.mat_ref(left_grad_row);
            for right_grad_col in 0..right_grad.nmats() {
                let right_grad_ref = right_grad.mat_ref(right_grad_col);
                let hess_ref = out.mat_mut(
                    left_grad_row,
                    left_hess.nmats() + right_grad_col,
                );
                matrix_kron(left_grad_ref, right_grad_ref, hess_ref);
            }
        }
    }

    #[inline(always)]
    pub fn evaluate<C: ComplexScalar>(&self, memory: &mut MemoryBuffer<C>) {
        let left_matref = self.left.as_matref::<C>(memory);
        let right_matref = self.right.as_matref::<C>(memory);
        let out_matmut = self.out.as_matmut::<C>(memory);
        self.calculate_unitary(left_matref, right_matref, out_matmut);
    }

    #[inline(always)]
    pub fn evaluate_gradient<C: ComplexScalar>(
        &self,
        memory: &mut MemoryBuffer<C>,
    ) {
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

    #[inline(always)]
    pub fn evaluate_hessian<C: ComplexScalar>(
        &self,
        memory: &mut MemoryBuffer<C>,
    ) {
        let left_matref = self.left.as_matref::<C>(memory);
        let left_matgradref = self.left.as_matvecref::<C>(memory);
        let left_mathessref = self.left.as_symsqmatref::<C>(memory);
        let right_matref = self.right.as_matref::<C>(memory);
        let right_matgradref = self.right.as_matvecref::<C>(memory);
        let right_mathessref = self.right.as_symsqmatref::<C>(memory);
        let out_matmut = self.out.as_matmut::<C>(memory);
        let out_matgradmut = self.out.as_matvecmut::<C>(memory);
        let out_mathessmut = self.out.as_symsqmatmut::<C>(memory);
        self.calculate_unitary(left_matref, right_matref, out_matmut);
        self.calculate_gradient(
            left_matref,
            left_matgradref,
            right_matref,
            right_matgradref,
            out_matgradmut,
        );
        // TODO: Remove this, after copy is implemented for non mut matref types
        let left_matgradref = self.left.as_matvecref::<C>(memory);
        let right_matgradref = self.right.as_matvecref::<C>(memory);
        self.calculate_hessian(
            left_matref,
            left_matgradref,
            left_mathessref,
            right_matref,
            right_matgradref,
            right_mathessref,
            out_mathessmut,
        );
    }

    // #[inline(always)]
    // pub fn evaluate_into<C: ComplexScalar>(
    //     &self,
    //     memory: &mut MemoryBuffer<C>,
    //     out: MatMut<C>,
    // ) {
    //     let left_matref = self.left.as_matref::<C>(memory);
    //     let right_matref = self.right.as_matref::<C>(memory);
    //     self.calculate_unitary(left_matref, right_matref, out);
    // }

    // #[inline(always)]
    // pub fn evaluate_gradient_into<C: ComplexScalar>(
    //     &self,
    //     memory: &mut MemoryBuffer<C>,
    //     out: MatMut<C>,
    //     out_grad: MatVecMut<C>,
    // ) {
    //     let left_matref = self.left.as_matref::<C>(memory);
    //     let left_matgradref = self.left.as_matvecref::<C>(memory);
    //     let right_matref = self.right.as_matref::<C>(memory);
    //     let right_matgradref = self.right.as_matvecref::<C>(memory);
    //     self.calculate_unitary(left_matref, right_matref, out);
    //     self.calculate_gradient(
    //         left_matref,
    //         left_matgradref,
    //         right_matref,
    //         right_matgradref,
    //         out_grad,
    //     );
    // }

    // #[inline(always)]
    // pub fn evaluate_hessian_into<C: ComplexScalar>(
    //     &self,
    //     memory: &mut MemoryBuffer<C>,
    //     out: MatMut<C>,
    //     out_grad: MatVecMut<C>,
    //     out_hess: SymSqMatMatMut<C>,
    // ) {
    //     let left_matref = self.left.as_matref::<C>(memory);
    //     let left_matgradref = self.left.as_matvecref::<C>(memory);
    //     let left_mathessref = self.left.as_symsqmatref::<C>(memory);
    //     let right_matref = self.right.as_matref::<C>(memory);
    //     let right_matgradref = self.right.as_matvecref::<C>(memory);
    //     let right_mathessref = self.right.as_symsqmatref::<C>(memory);
    //     self.calculate_unitary(left_matref, right_matref, out);
    //     self.calculate_gradient(
    //         left_matref,
    //         left_matgradref,
    //         right_matref,
    //         right_matgradref,
    //         out_grad,
    //     );
    //     // TODO: Remove this, after copy is implemented for non mut matref types
    //     let left_matgradref = self.left.as_matvecref::<C>(memory);
    //     let right_matgradref = self.right.as_matvecref::<C>(memory);
    //     self.calculate_hessian(
    //         left_matref,
    //         left_matgradref,
    //         left_mathessref,
    //         right_matref,
    //         right_matgradref,
    //         right_mathessref,
    //         out_hess,
    //     );
    // }
}
