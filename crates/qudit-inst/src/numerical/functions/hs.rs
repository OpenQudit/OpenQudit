use std::sync::Arc;

use super::super::InstantiationProblem;
use crate::InstantiationTarget;
use crate::numerical::Hessian;
use crate::numerical::Jacobian;
use crate::numerical::ProvidesJacobian;
use crate::numerical::ProvidesResidualFunction;
use crate::numerical::ResidualFunction;
use crate::numerical::function::CostFunction;
use crate::numerical::function::Function;
use crate::numerical::function::Gradient;
use crate::numerical::problem::Problem;
use crate::numerical::problem::ProvidesCostFunction;
use crate::numerical::problem::ProvidesGradient;
use faer::ColMut;
use faer::MatMut;
use faer::Row;
use faer::RowMut;
use faer::reborrow::ReborrowMut;
use qudit_circuit::QuditCircuit;
use qudit_core::BitWidthConvertible;
use qudit_core::ComplexScalar;
use qudit_core::RealScalar;
use qudit_expr::DifferentiationLevel;
use qudit_expr::FUNCTION;
use qudit_expr::GRADIENT;
use qudit_expr::HESSIAN;
use qudit_tensor::PinnedTNVM;
use qudit_tensor::QuditTensor;
use qudit_tensor::TNVM;
use qudit_tensor::compile_network;

#[derive(Clone)]
pub struct HSProblem<R: RealScalar> {
    pub circuit: Arc<QuditCircuit>,
    pub target: Arc<InstantiationTarget<R::C>>,
}

impl<R: RealScalar> HSProblem<R> {
    pub fn new(circuit: Arc<QuditCircuit>, target: Arc<InstantiationTarget<R::C>>) -> Self {
        HSProblem { circuit, target }
    }

    pub fn build_cost<const D: DifferentiationLevel>(&self) -> PinnedTNVM<R::C, D> {
        let mut builder = self.circuit.as_tensor_network_builder();
        // TODO: assert target.num_outputs = builder.num_open_outputs
        // TODO: assert target.num_inputs = builder.num_open_inputs
        // TODO: assert target.num_batch = builder.num_batch || target.num_batch = 1
        match &*self.target {
            InstantiationTarget::UnitaryMatrix(u) => {
                let qudits = builder.open_output_indices();
                builder = builder.prepend_unitary(u.dagger(), qudits);
            }
        }

        let code = compile_network(builder.trace_all_open_wires().build());
        TNVM::<R::C, D>::new(&code, Some(&self.circuit.params().const_map()))
    }

    pub fn build_residual<const D: DifferentiationLevel>(&self) -> PinnedTNVM<R::C, D> {
        // TODO: deduplicate code
        let mut builder = self.circuit.as_tensor_network_builder();
        // TODO: assert target.num_outputs = builder.num_open_outputs
        // TODO: assert target.num_inputs = builder.num_open_inputs
        // TODO: assert target.num_batch = builder.num_batch || target.num_batch = 1
        match &*self.target {
            InstantiationTarget::UnitaryMatrix(u) => {
                let qudits = builder.open_output_indices();
                builder = builder.prepend_unitary(u.dagger(), qudits);
            }
        }

        let code = compile_network(builder.build());
        // dbg!(&code);
        TNVM::<R::C, D>::new(&code, Some(&self.circuit.params().const_map()))
    }
}

impl<R: RealScalar> InstantiationProblem<R> for HSProblem<R> {
    fn from_instantiation(
        circuit: Arc<QuditCircuit>,
        target: Arc<InstantiationTarget<R::C>>,
        data: Arc<crate::DataMap>,
    ) -> Self {
        Self::new(circuit, target)
    }
}

impl<R: RealScalar> Problem for HSProblem<R> {
    fn num_params(&self) -> usize {
        self.circuit.num_params()
    }
}

impl<R: RealScalar> ProvidesCostFunction<R> for HSProblem<R> {
    type CostFunction = HSFunction<R, 0>;

    fn build_cost_function(&self) -> Self::CostFunction {
        HSFunction {
            tnvm: self.build_cost(),
            n: R::from64(4.0),
        } // TODO: N comes from circuit
    }
}

impl<R: RealScalar> ProvidesResidualFunction<R> for HSProblem<R> {
    type ResidualFunction = HSFunction<R, 0>;

    fn build_residual_function(&self) -> Self::ResidualFunction {
        HSFunction {
            tnvm: self.build_residual(),
            n: R::from64(4.0),
        }
    }
}

impl<R: RealScalar> ProvidesJacobian<R> for HSProblem<R> {
    type Jacobian = HSFunction<R, GRADIENT>;

    fn build_jacobian(&self) -> Self::Jacobian {
        HSFunction {
            tnvm: self.build_residual(),
            n: R::from64(4.0),
        }
    }
}

pub struct HSFunction<R: RealScalar, const D: DifferentiationLevel> {
    tnvm: PinnedTNVM<R::C, D>,
    n: R,
}

impl<R: RealScalar, const D: DifferentiationLevel> Function for HSFunction<R, D> {
    fn num_params(&self) -> usize {
        self.tnvm.num_params()
        // self.tnvm.borrow().num_params()
    }
}

use num_complex::ComplexFloat;

impl<R: RealScalar, const D: DifferentiationLevel> CostFunction<R> for HSFunction<R, D> {
    fn cost(&mut self, params: &[R]) -> R {
        let result = self.tnvm.evaluate::<FUNCTION>(params);
        let trace = result.get_fn_result().unpack_scalar(); // This isn't a scalar if kraus
        // dimension // TODO: add unpack vector able to handle scalar
        let inner = trace.abs() / self.n;
        R::from64(1.0) - inner
        // (1 - (trace.clone().abs() / self.N).square()).sqrt()
        // TODO: consider using proper norm (might be faster to use for gradient and hessian due
        // not having to compute abs inside for loop, i.e. trade one sqrt for num_params.
    }
}

// TODO: some safety so D>=1
impl<R: RealScalar, const D: DifferentiationLevel> Gradient<R> for HSFunction<R, D> {
    fn gradient_into(&mut self, params: &[R], grad_out: RowMut<R>) {
        self.cost_and_gradient_into(params, grad_out);
    }

    fn cost_and_gradient_into(&mut self, params: &[R], grad_out: RowMut<R>) -> R {
        let result = self.tnvm.evaluate::<GRADIENT>(params);
        let grad_trace = result.get_grad_result().unpack_vector();
        let trace = result.get_fn_result().unpack_scalar();
        let trace_re = trace.real();
        let trace_im = trace.imag();
        for (out, trace) in grad_out.iter_mut().zip(grad_trace.iter()) {
            let trace_re_d = trace.real();
            let trace_im_d = trace.imag();
            let num = trace_re * trace_re_d + trace_im * trace_im_d;
            let dem = self.n * trace.abs();
            *out = -(num / dem)
        }
        R::from64(1.0) - (trace.abs() / self.n)
    }
}

impl<R: RealScalar, const D: DifferentiationLevel> Hessian<R> for HSFunction<R, D> {
    fn hessian_into(&mut self, params: &[R], hess_out: faer::MatMut<R>) {
        todo!()
    }

    fn cost_gradient_and_hessian_into(
        &mut self,
        params: &[R],
        grad_out: RowMut<R>,
        hess_out: faer::MatMut<R>,
    ) -> R {
        let result = self.tnvm.evaluate::<HESSIAN>(params);
        let hess_trace = result.get_hess_result().unpack_symsq_matrix();
        let grad_trace = result.get_grad_result().unpack_vector();
        let trace = result.get_fn_result().unpack_scalar();

        // for (hess_out_col, (hess_trace_x_col, grad_trace_x)) in hess_out.col_iter_mut().zip(hess_trace.col_iter()

        todo!("Need better iterators")
        // Math = (-1/|trace|^3)*[R^3R_xy + RR_xyI^2 + I^2R_yR_x - RR_yII_x - II_yRR_x + R^2I_yI_x
        // + R^2II_xy + I^3I_xy]
        // R = Re(trace)
        // R_x = Re(grad_trace @ x)
        // Re_xy = Re(hess_trace @ xy)
        // I = Im..

        // R::new(1.0) - (trace.abs() / self.N)
    }
}

impl<R: RealScalar, const D: DifferentiationLevel> ResidualFunction<R> for HSFunction<R, D> {
    fn num_residuals(&self) -> usize {
        let out_shape = self.tnvm.out_shape();
        (out_shape.num_elements() - out_shape.nmats()) * 2
    }

    fn residuals_into(&mut self, params: &[R], mut residuals_out: ColMut<R>) {
        let result = self.tnvm.evaluate::<FUNCTION>(params);
        let kraus_ops = result.get_fn_result2().unpack_tensor3d();

        let mut residual_index = 0;
        for k in 0..kraus_ops.dims()[0] {
            let mat = kraus_ops.subtensor_ref(k);

            for (j, col) in mat.col_iter().enumerate() {
                for (i, elem) in col.iter().enumerate() {
                    if i == j {
                        continue;
                    }
                    unsafe {
                        let out = residuals_out.rb_mut().get_mut_unchecked(residual_index);
                        *out = elem.real();
                        residual_index += 1;
                        let out = residuals_out.rb_mut().get_mut_unchecked(residual_index);
                        *out = elem.imag();
                        residual_index += 1;
                    }
                }
            }

            let diag_iter = 0;
            for diag_iter in 0..(mat.nrows() - 1) {
                unsafe {
                    let d_i = mat.get_unchecked(diag_iter, diag_iter);
                    let d_j = mat.get_unchecked(diag_iter + 1, diag_iter + 1);
                    let out = residuals_out.rb_mut().get_mut_unchecked(residual_index);
                    *out = d_j.real() - d_i.real();
                    residual_index += 1;
                    let out = residuals_out.rb_mut().get_mut_unchecked(residual_index);
                    *out = d_j.imag() - d_i.imag();
                    residual_index += 1;
                }
            }
        }
    }
}

impl<R: RealScalar, const D: DifferentiationLevel> Jacobian<R> for HSFunction<R, D> {
    fn residuals_and_jacobian_into(
        &mut self,
        params: &[R],
        mut residuals_out: ColMut<R>,
        mut jacobian_out: MatMut<R>,
    ) {
        let result = self.tnvm.evaluate::<GRADIENT>(params);
        let kraus_ops = result.get_fn_result2().unpack_tensor3d();

        // for k in 0..kraus_ops.dims()[0] {
        //     let mat = kraus_ops.subtensor_ref(k);
        //     for (j, col) in mat.col_iter().enumerate() {
        //         for (i, elem) in col.iter().enumerate() {
        //             print!("{} ", elem);
        //         }
        //         println!("");
        //     }
        //     println!("");
        // }
        // println!("");

        let mut residual_index = 0;
        for k in 0..kraus_ops.dims()[0] {
            let mat = kraus_ops.subtensor_ref(k);

            for (j, col) in mat.col_iter().enumerate() {
                for (i, elem) in col.iter().enumerate() {
                    if i == j {
                        continue;
                    }
                    unsafe {
                        let out = residuals_out.rb_mut().get_mut_unchecked(residual_index);
                        *out = elem.real();
                        residual_index += 1;
                        let out = residuals_out.rb_mut().get_mut_unchecked(residual_index);
                        *out = elem.imag();
                        residual_index += 1;
                    }
                }
            }

            for diag_iter in 0..(mat.nrows() - 1) {
                unsafe {
                    let d_i = mat.get_unchecked(diag_iter, diag_iter);
                    let d_j = mat.get_unchecked(diag_iter + 1, diag_iter + 1);
                    let out = residuals_out.rb_mut().get_mut_unchecked(residual_index);
                    *out = d_j.real() - d_i.real();
                    residual_index += 1;
                    let out = residuals_out.rb_mut().get_mut_unchecked(residual_index);
                    *out = d_j.imag() - d_i.imag();
                    residual_index += 1;
                }
            }
        }

        let grad_out = result.get_grad_result2().unpack_tensor4d();

        for p in 0..grad_out.dims()[0] {
            let partial_kraus_ops = grad_out.subtensor_ref(p);
            let mut jacobian_col_out = jacobian_out.rb_mut().col_mut(p);
            let mut residual_index = 0;

            for k in 0..partial_kraus_ops.dims()[0] {
                let mat = partial_kraus_ops.subtensor_ref(k);
                // dbg!(&mat);

                for (j, col) in mat.col_iter().enumerate() {
                    for (i, elem) in col.iter().enumerate() {
                        if i == j {
                            continue;
                        }
                        unsafe {
                            let out = jacobian_col_out.rb_mut().get_mut_unchecked(residual_index);
                            *out = elem.real();
                            residual_index += 1;
                            let out = jacobian_col_out.rb_mut().get_mut_unchecked(residual_index);
                            *out = elem.imag();
                            residual_index += 1;
                        }
                    }
                }

                for diag_iter in 0..(mat.nrows() - 1) {
                    unsafe {
                        let d_i = mat.get_unchecked(diag_iter, diag_iter);
                        let d_j = mat.get_unchecked(diag_iter + 1, diag_iter + 1);
                        let out = jacobian_col_out.rb_mut().get_mut_unchecked(residual_index);
                        *out = d_j.real() - d_i.real();
                        residual_index += 1;
                        let out = jacobian_col_out.rb_mut().get_mut_unchecked(residual_index);
                        *out = d_j.imag() - d_i.imag();
                        residual_index += 1;
                    }
                }
            }
        }
    }

    fn jacobian_into(&mut self, params: &[R], mut jacobian_out: MatMut<R>) {
        let result = self.tnvm.evaluate::<GRADIENT>(params);
        let grad_out = result.get_grad_result2().unpack_tensor4d();

        for p in 0..grad_out.dims()[0] {
            let partial_kraus_ops = grad_out.subtensor_ref(p);
            let mut jacobian_col_out = jacobian_out.rb_mut().col_mut(p);
            let mut residual_index = 0;

            for k in 0..partial_kraus_ops.dims()[0] {
                let mat = partial_kraus_ops.subtensor_ref(k);

                for (j, col) in mat.col_iter().enumerate() {
                    for (i, elem) in col.iter().enumerate() {
                        if i == j {
                            continue;
                        }
                        unsafe {
                            let out = jacobian_col_out.rb_mut().get_mut_unchecked(residual_index);
                            *out = elem.real();
                            residual_index += 1;
                            let out = jacobian_col_out.rb_mut().get_mut_unchecked(residual_index);
                            *out = elem.imag();
                            residual_index += 1;
                        }
                    }
                }

                let diag_iter = 0;
                for diag_iter in 0..(mat.nrows() - 1) {
                    unsafe {
                        let d_i = mat.get_unchecked(diag_iter, diag_iter);
                        let d_j = mat.get_unchecked(diag_iter + 1, diag_iter + 1);
                        let out = jacobian_col_out.rb_mut().get_mut_unchecked(residual_index);
                        *out = d_j.real() - d_i.real();
                        residual_index += 1;
                        let out = jacobian_col_out.rb_mut().get_mut_unchecked(residual_index);
                        *out = d_j.imag() - d_i.imag();
                        residual_index += 1;
                    }
                }
            }
        }
    }
}
