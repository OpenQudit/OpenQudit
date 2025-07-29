use qudit_circuit::QuditCircuit;
use qudit_core::matrix::MatMut;
use qudit_core::matrix::Row;
use qudit_core::matrix::RowMut;
use qudit_core::ComplexScalar;
use qudit_core::RealScalar;
use qudit_core::BitWidthConvertible;
use qudit_expr::DifferentiationLevel;
use qudit_tensor::compile_network;
use qudit_tensor::QuditTensor;
use qudit_tensor::TNVM;
use qudit_tensor::PinnedTNVM;
use crate::numerical::Hessian;
use crate::numerical::Jacobian;
use crate::numerical::ResidualFunction;
use crate::InstantiationTarget;
use super::super::InstantiationProblem;
use crate::numerical::function::Function;
use crate::numerical::function::CostFunction;
use crate::numerical::function::Gradient;
use crate::numerical::problem::Problem;
use crate::numerical::problem::ProvidesCostFunction;
use crate::numerical::problem::ProvidesGradient;
use faer::reborrow::ReborrowMut;

pub struct HSProblem<'a, R: RealScalar> {
    pub circuit: &'a QuditCircuit<R::C>,
    pub target: &'a InstantiationTarget<R::C>,
}


impl<'a, R: RealScalar> HSProblem<'a, R> {
    pub fn new(circuit: &'a QuditCircuit<R::C>, target: &'a InstantiationTarget<R::C>) -> Self {
        HSProblem {
            circuit,
            target,
        }
    }

    pub fn build_cost<const D: DifferentiationLevel>(&self) -> PinnedTNVM<R::C, D> { 
        let mut builder = self.circuit.as_tensor_network_builder();
        // TODO: assert target.num_outputs = builder.num_open_outputs
        // TODO: assert target.num_inputs = builder.num_open_inputs
        // TODO: assert target.num_batch = builder.num_batch || target.num_batch = 1
        match self.target {
            InstantiationTarget::UnitaryMatrix(u) => {
                let qudits = builder.open_output_indices();
                builder = builder.prepend_unitary(u.dagger(), qudits);
            },
        }

        let code = compile_network(builder.trace_all_open_wires().build());
        TNVM::<R::C, D>::new(&code)
    }

    pub fn build_residual<const D: DifferentiationLevel>(&self) -> PinnedTNVM<R::C, D> {
        // TODO: deduplicate code
        let mut builder = self.circuit.as_tensor_network_builder();
        // TODO: assert target.num_outputs = builder.num_open_outputs
        // TODO: assert target.num_inputs = builder.num_open_inputs
        // TODO: assert target.num_batch = builder.num_batch || target.num_batch = 1
        match self.target {
            InstantiationTarget::UnitaryMatrix(u) => {
                let qudits = builder.open_output_indices();
                builder = builder.prepend_unitary(u.dagger(), qudits);
            },
        }

        let code = compile_network(builder.build());
        TNVM::<R::C, D>::new(&code)
    }
}

impl<'a, R: RealScalar> InstantiationProblem<'a, R> for HSProblem<'a, R> {
    fn from_instantiation(
        circuit: &'a QuditCircuit<R::C>,
        target: &'a InstantiationTarget<R::C>,
        data: &'a crate::DataMap,
    ) -> Self {
        Self::new(circuit, target)
    }
}

impl<'a, R: RealScalar> Problem for HSProblem<'a, R> {
    fn num_params(&self) -> usize {
        self.circuit.num_params()
    }
}

impl<'a, R: RealScalar> ProvidesCostFunction<R> for HSProblem<'a, R> {
    type CostFunction = HSFunction<R, 0>;

    fn build_cost_function(&self) -> Self::CostFunction {
        HSFunction { tnvm: self.build_cost().into(), N: R::from64(4.0) } // TODO: N comes from circuit
    }
}

// // impl<'a, R: RealScalar> ProvidesGradient<C> for HSProblem<'a, R> {
// //     type Gradient = QuantumCostFunction;
// //     fn build_gradient(&self) -> Self::Gradient {
// //         QuantumCostFunction
// //     }
// // }

pub struct HSFunction<R: RealScalar, const D: DifferentiationLevel> {
    tnvm: PinnedTNVM<R::C, D>,
    N: R,
}

impl<R: RealScalar, const D: DifferentiationLevel> Function for HSFunction<R, D> {
    fn num_params(&self) -> usize {
        todo!()
        // self.tnvm.borrow().num_params()
    }
}

use num_complex::ComplexFloat;

impl<R: RealScalar, const D: DifferentiationLevel> CostFunction<R> for HSFunction<R, D> {
    fn cost(&mut self, params: &[R]) -> R {
        let result = self.tnvm.evaluate(params);
        let trace = result.get_fn_result().unpack_scalar(); // This isn't a scalar if kraus
        // dimension // TODO: add unpack vector able to handle scalar
        let inner = trace.abs() / self.N;
        R::new(1.0) - inner
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
        let result = self.tnvm.evaluate(params);
        let grad_trace = result.get_grad_result().unpack_vector();
        let trace = result.get_fn_result().unpack_scalar();
        let trace_re = trace.real();
        let trace_im = trace.imag();
        for (out, trace) in grad_out.iter_mut().zip(grad_trace.iter()) {
            let trace_re_d = trace.real();
            let trace_im_d = trace.imag();
            let num = trace_re*trace_re_d + trace_im*trace_im_d;
            let dem = self.N * trace.abs();
            *out = -(num / dem)
        }
        R::new(1.0) - (trace.abs() / self.N) 
    }
}

impl<R: RealScalar, const D: DifferentiationLevel> Hessian<R> for HSFunction<R, D> {
    fn hessian_into(&mut self, params: &[R], hess_out: qudit_core::matrix::MatMut<R>) {
        todo!()
    }

    fn cost_gradient_and_hessian_into(&mut self, params: &[R], grad_out: RowMut<R>, hess_out: qudit_core::matrix::MatMut<R>) -> R {        
        let result = self.tnvm.evaluate(params);
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
    fn residuals_into(&mut self, params: &[R], mut residuals_out: RowMut<R>) {
        let result = self.tnvm.evaluate(params);
        // TODO: be able to tell tnvm to make last buffer contiguous so I can just read out a
        // vector...; even better add an evaluate_into for tnvm :)
        let matrix_out = result.get_fn_result().unpack_matrix();// TODO: not always a matrix...
        
        let mut residual_index = 0;
        for (j, col) in matrix_out.col_iter().enumerate() {
            for (i, elem) in col.iter().enumerate() {
                // SAFETY: because I said so (meaning: no safety at all; TODO: do better :)
                unsafe {
                    let out = residuals_out.rb_mut().get_mut_unchecked(residual_index);
                    *out = if i == j { elem.real() - R::new(1.0) } else { elem.real() };
                    residual_index += 1;
                    let out = residuals_out.rb_mut().get_mut_unchecked(residual_index);
                    *out = elem.imag();
                    residual_index += 1;
                }
            }
        }
    }
}

impl<R: RealScalar, const D: DifferentiationLevel> Jacobian<R> for HSFunction<R, D> {
    fn jacobian_into(&mut self, params: &[R], mut jacobian_out: MatMut<R>) {
        let result = self.tnvm.evaluate(params);
        let grad_out = result.get_grad_result().unpack_tensor3d();

        for p in 0..grad_out.dims()[0] {    
            let grad_matrix_out = grad_out.subtensor_ref(p);
            let mut jacobian_col_out = jacobian_out.rb_mut().col_mut(p);
            let mut residual_index = 0;
            for (j, col) in grad_matrix_out.col_iter().enumerate() {
                for (i, elem) in col.iter().enumerate() {
                    // SAFETY: because I said so (meaning: no safety at all; TODO: do better :)
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
        }
    }
}
