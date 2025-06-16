//! This module defines traits for cost functions, gradients, Hessians, residual functions, and Jacobians.

use qudit_core::ComplexScalar;

pub trait Function<C: ComplexScalar> {
    fn num_params(&self) -> usize;
}

pub trait CostFunction<C: ComplexScalar>: Function<C> {
    fn cost(&self, params: &[C::R]) -> C::R;
}

pub trait Gradient<C: ComplexScalar>: CostFunction<C> {
    fn gradient(&self, params: &[C::R]) -> &[C::R];

    fn cost_and_gradient(&self, params: &[C::R]) -> (C::R, &[C::R]) {
        (self.cost(params), self.gradient(params))
    }
}

pub trait ResidualFunction<C: ComplexScalar> {
    fn residuals(&self, params: &[C::R]) -> &[C::R];
}

pub trait Jacobian<C: ComplexScalar>: ResidualFunction<C> {
    fn jacobian(&self, params: &[C::R]) -> &[C::R];

    fn cost_and_jacobian(&self, params: &[C::R]) -> (&[C::R], &[C::R]) {
        (self.residuals(params), self.jacobian(params))
    }
}

pub trait Hessian<C: ComplexScalar>: Gradient<C> {
    fn hessian(&self, params: &[C::R]) -> &[C::R];

    fn cost_gradient_and_hessian(&self, params: &[C::R]) -> (C::R, &[C::R], &[C::R]) {
        (self.cost(params), self.gradient(params), self.hessian(params))
    }
}
