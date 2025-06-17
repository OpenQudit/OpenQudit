//! This module defines traits for cost functions, gradients, Hessians, residual functions, and Jacobians.

use qudit_core::ComplexScalar;

pub trait Function<C: ComplexScalar> {
    fn num_params(&self) -> usize;
}

/// A trait for cost functions.
pub trait CostFunction<C: ComplexScalar>: Function<C> {
    /// Calculates the cost for the given parameters.
    fn cost(&self, params: &[C::R]) -> C::R;
}

/// A trait for gradients.
pub trait Gradient<C: ComplexScalar>: CostFunction<C> {
    /// Calculates the gradient for the given parameters.
    fn gradient(&self, params: &[C::R]) -> &[C::R];

    /// Calculates the cost and gradient for the given parameters.
    fn cost_and_gradient(&self, params: &[C::R]) -> (C::R, &[C::R]) {
        (self.cost(params), self.gradient(params))
    }
}

/// A trait for residual functions.
pub trait ResidualFunction<C: ComplexScalar> {
    /// Calculates the residuals for the given parameters.
    fn residuals(&self, params: &[C::R]) -> &[C::R];
}

/// A trait for Jacobians.
pub trait Jacobian<C: ComplexScalar>: ResidualFunction<C> {
    /// Calculates the Jacobian for the given parameters.
    fn jacobian(&self, params: &[C::R]) -> &[C::R];

    /// Calculates the residuals and Jacobian for the given parameters.
    fn cost_and_jacobian(&self, params: &[C::R]) -> (&[C::R], &[C::R]) {
        (self.residuals(params), self.jacobian(params))
    }
}

/// A trait for Hessians.
pub trait Hessian<C: ComplexScalar>: Gradient<C> {
    /// Calculates the Hessian for the given parameters.
    fn hessian(&self, params: &[C::R]) -> &[C::R];

    /// Calculates the cost, gradient, and Hessian for the given parameters.
    fn cost_gradient_and_hessian(&self, params: &[C::R]) -> (C::R, &[C::R], &[C::R]) {
        (self.cost(params), self.gradient(params), self.hessian(params))
    }
}
