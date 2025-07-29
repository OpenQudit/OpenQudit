//! This module defines traits for cost functions, gradients, Hessians, residual functions, and Jacobians.

use qudit_core::{matrix::{Mat, MatMut, MatRef, Row, RowMut, RowRef}, RealScalar};

pub trait Function {
    fn num_params(&self) -> usize;
}

/// A trait for cost functions.
pub trait CostFunction<R: RealScalar>: Function {
    /// Calculates the cost for the given parameters.
    fn cost(&mut self, params: &[R]) -> R;
}

/// A trait for gradients.
pub trait Gradient<R: RealScalar>: CostFunction<R> {
    fn allocate_gradient(&self) -> Row<R> {
        Row::zeros(self.num_params())
    }

    /// Calculates the gradient for the given parameters.
    fn gradient_into(&mut self, params: &[R], grad_out: RowMut<R>);

    /// Calculates the cost and gradient for the given parameters.
    fn cost_and_gradient_into(&mut self, params: &[R], grad_out: RowMut<R>) -> R {
        let cost = self.cost(params);
        self.gradient_into(params, grad_out);
        cost
    }

    // TODO: non-into methods that allocate buffers and return owned values
    // TODO: batch_cost_and_gradient.. and others
}

/// A trait for Hessians.
pub trait Hessian<R: RealScalar>: Gradient<R> {
    fn allocate_hessian(&self) -> Mat<R> {
        Mat::zeros(self.num_params(), self.num_params())
    }

    /// Calculates the Hessian for the given parameters.
    fn hessian_into(&mut self, params: &[R], hess_out: MatMut<R>);

    /// Calculates the cost, gradient, and Hessian for the given parameters.
    fn cost_gradient_and_hessian_into(&mut self, params: &[R], grad_out: RowMut<R>, hess_out: MatMut<R>) -> R {
        let cost = self.cost(params);
        self.gradient_into(params, grad_out);
        self.hessian_into(params, hess_out);
        cost
    }
}

/// A trait for residual functions.
pub trait ResidualFunction<R: RealScalar>: Function {
    fn allocate_residual(&self) -> Row<R> {
        Row::zeros(self.num_params())
    }

    /// Calculates the residuals for the given parameters.
    fn residuals_into(&mut self, params: &[R], residuals_out: RowMut<R>);
}

/// A trait for Jacobians.
pub trait Jacobian<R: RealScalar>: ResidualFunction<R> {
    fn allocate_jacobian(&self) -> Mat<R> {
        Mat::zeros(self.num_params(), self.num_params())
    }

    /// Calculates the Jacobian for the given parameters.
    fn jacobian_into(&mut self, params: &[R], jacobian_out: MatMut<R>);

    /// Calculates the residuals and Jacobian for the given parameters.
    fn residuals_and_jacobian_into(&mut self, params: &[R], residuals_out: RowMut<R>, jacobian_out: MatMut<R>) {
        self.residuals_into(params, residuals_out);
        self.jacobian_into(params, jacobian_out);
    }
}
