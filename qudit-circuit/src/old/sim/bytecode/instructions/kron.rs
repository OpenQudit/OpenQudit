use faer_core::{MatMut, MatRef};

use crate::{
    math::{
        matrix::{MatGradMut, MatGradRef, MatHessMut, MatHessRef},
        matrix_kron, ComplexScalar,
    },
    sim::{bytecode::SizedMatrixBuffer, qvm::Memory},
};

pub struct KronStruct {
    pub left: SizedMatrixBuffer,
    pub right: SizedMatrixBuffer,
    pub out: SizedMatrixBuffer,
}

impl KronStruct {
    pub fn new(
        left: SizedMatrixBuffer,
        right: SizedMatrixBuffer,
        out: SizedMatrixBuffer,
    ) -> Self {
        Self { left, right, out }
    }

    #[inline(always)]
    fn calculate_unitary<C: ComplexScalar>(
        &self,
        left: MatRef<C>,
        right: MatRef<C>,
        mut out: MatMut<C>,
    ) {
        matrix_kron(left, right, out);
    }

    #[inline(always)]
    fn calculate_gradient<C: ComplexScalar>(
        &self,
        left_utry: MatRef<C>,
        left_grad: MatGradRef<C>,
        right_utry: MatRef<C>,
        right_grad: MatGradRef<C>,
        mut out: MatGradMut<C>,
    ) {
        let mut grad_idx = 0;

        for i in 0..self.left.num_params {
            let left_gradref = left_grad.get_matref(i);
            let out_gradmut = out.get_matmut(grad_idx);
            matrix_kron(left_gradref, right_utry, out_gradmut);
            grad_idx += 1;
        }

        for i in 0..self.right.num_params {
            let right_gradref = right_grad.get_matref(i);
            let out_gradmut = out.get_matmut(grad_idx);
            matrix_kron(left_utry, right_gradref, out_gradmut);
            grad_idx += 1;
        }
    }

    #[inline(always)]
    fn calculate_hessian<C: ComplexScalar>(
        &self,
        left_utry: MatRef<C>,
        left_grad: MatGradRef<C>,
        left_hess: MatHessRef<C>,
        right_utry: MatRef<C>,
        right_grad: MatGradRef<C>,
        right_hess: MatHessRef<C>,
        mut out: MatHessMut<C>,
    ) {
        // Upper left block: right_utry * left_hess
        for left_hess_row in 0..left_hess.num_params() {
            for left_hess_col in left_hess_row..left_hess.num_params() {
                let left_hess_ref =
                    left_hess.get_matref(left_hess_row, left_hess_col);
                let hess_ref = out.get_matmut(left_hess_row, left_hess_col);
                matrix_kron(left_hess_ref, right_utry, hess_ref);
            }
        }

        // Lower right block: right_hess * left_utry
        for right_hess_row in 0..right_hess.num_params() {
            for right_hess_col in right_hess_row..right_hess.num_params() {
                let right_hess_ref =
                    right_hess.get_matref(right_hess_row, right_hess_col);
                let hess_ref = out.get_matmut(
                    left_hess.num_params() + right_hess_row,
                    left_hess.num_params() + right_hess_col,
                );
                matrix_kron(left_utry, right_hess_ref, hess_ref);
            }
        }

        // Upper right block: right_grad * left_grad
        for left_grad_row in 0..left_grad.num_params() {
            let left_grad_ref = left_grad.get_matref(left_grad_row);
            for right_grad_col in 0..right_grad.num_params() {
                let right_grad_ref = right_grad.get_matref(right_grad_col);
                let hess_ref = out.get_matmut(
                    left_grad_row,
                    left_hess.num_params() + right_grad_col,
                );
                matrix_kron(left_grad_ref, right_grad_ref, hess_ref);
            }
        }
    }

    #[inline(always)]
    pub fn execute_unitary<C: ComplexScalar>(&self, memory: &mut Memory<C>) {
        let left_matref = self.left.as_matref::<C>(memory);
        let right_matref = self.right.as_matref::<C>(memory);
        let out_matmut = self.out.as_matmut::<C>(memory);
        self.calculate_unitary(left_matref, right_matref, out_matmut);
    }

    #[inline(always)]
    pub fn execute_unitary_and_gradient<C: ComplexScalar>(
        &self,
        memory: &mut Memory<C>,
    ) {
        let left_matref = self.left.as_matref::<C>(memory);
        let left_matgradref = self.left.as_matgradref::<C>(memory);
        let right_matref = self.right.as_matref::<C>(memory);
        let right_matgradref = self.right.as_matgradref::<C>(memory);
        let out_matmut = self.out.as_matmut::<C>(memory);
        let out_matgradmut = self.out.as_matgradmut::<C>(memory);
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
    pub fn execute_unitary_gradient_and_hessian<C: ComplexScalar>(
        &self,
        memory: &mut Memory<C>,
    ) {
        let left_matref = self.left.as_matref::<C>(memory);
        let left_matgradref = self.left.as_matgradref::<C>(memory);
        let left_mathessref = self.left.as_mathessref::<C>(memory);
        let right_matref = self.right.as_matref::<C>(memory);
        let right_matgradref = self.right.as_matgradref::<C>(memory);
        let right_mathessref = self.right.as_mathessref::<C>(memory);
        let out_matmut = self.out.as_matmut::<C>(memory);
        let out_matgradmut = self.out.as_matgradmut::<C>(memory);
        let out_mathessmut = self.out.as_mathessmut::<C>(memory);
        self.calculate_unitary(left_matref, right_matref, out_matmut);
        self.calculate_gradient(
            left_matref,
            left_matgradref,
            right_matref,
            right_matgradref,
            out_matgradmut,
        );
        // TODO: Remove this, after copy is implemented for non mut matref types
        let left_matgradref = self.left.as_matgradref::<C>(memory);
        let right_matgradref = self.right.as_matgradref::<C>(memory);
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

    #[inline(always)]
    pub fn execute_unitary_into<C: ComplexScalar>(
        &self,
        memory: &mut Memory<C>,
        mut out: MatMut<C>,
    ) {
        let left_matref = self.left.as_matref::<C>(memory);
        let right_matref = self.right.as_matref::<C>(memory);
        self.calculate_unitary(left_matref, right_matref, out);
    }

    #[inline(always)]
    pub fn execute_unitary_and_gradient_into<C: ComplexScalar>(
        &self,
        memory: &mut Memory<C>,
        mut out: MatMut<C>,
        mut out_grad: MatGradMut<C>,
    ) {
        let left_matref = self.left.as_matref::<C>(memory);
        let left_matgradref = self.left.as_matgradref::<C>(memory);
        let right_matref = self.right.as_matref::<C>(memory);
        let right_matgradref = self.right.as_matgradref::<C>(memory);
        self.calculate_unitary(left_matref, right_matref, out);
        self.calculate_gradient(
            left_matref,
            left_matgradref,
            right_matref,
            right_matgradref,
            out_grad,
        );
    }

    #[inline(always)]
    pub fn execute_unitary_gradient_and_hessian_into<C: ComplexScalar>(
        &self,
        memory: &mut Memory<C>,
        mut out: MatMut<C>,
        mut out_grad: MatGradMut<C>,
        mut out_hess: MatHessMut<C>,
    ) {
        let left_matref = self.left.as_matref::<C>(memory);
        let left_matgradref = self.left.as_matgradref::<C>(memory);
        let left_mathessref = self.left.as_mathessref::<C>(memory);
        let right_matref = self.right.as_matref::<C>(memory);
        let right_matgradref = self.right.as_matgradref::<C>(memory);
        let right_mathessref = self.right.as_mathessref::<C>(memory);
        self.calculate_unitary(left_matref, right_matref, out);
        self.calculate_gradient(
            left_matref,
            left_matgradref,
            right_matref,
            right_matgradref,
            out_grad,
        );
        // TODO: Remove this, after copy is implemented for non mut matref types
        let left_matgradref = self.left.as_matgradref::<C>(memory);
        let right_matgradref = self.right.as_matgradref::<C>(memory);
        self.calculate_hessian(
            left_matref,
            left_matgradref,
            left_mathessref,
            right_matref,
            right_matgradref,
            right_mathessref,
            out_hess,
        );
    }
}
