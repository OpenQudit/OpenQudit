//! This module defines the `Problem` trait and associated traits for defining optimization problems.
//!
//! Any object implementing the `Problem` trait and some associated `ProvidesX` traits defines how an
//! `OptimizationStrategy` can build the associated methods necessary for optimization. Certain optimizers
//! and strategies will require specific capabilities and will fail to compile if not provided.
//! This design is for performance reasons. Some problems may be computationally expensive to build as well as evaluate.
//! The separation of a `Problem` from the `Function` it generates allows performance optimizations in these cases.

use crate::numerical::function::CostFunction;
use crate::numerical::function::Function;
use crate::numerical::function::Gradient;
use crate::numerical::function::Hessian;
use crate::numerical::function::Jacobian;
use crate::numerical::function::ResidualFunction;
use qudit_core::ComplexScalar;
use qudit_core::RealScalar;

/// A trait for defining optimization problems.
pub trait Problem {
    /// Returns the number of parameters in the problem.
    fn num_params(&self) -> usize;
}

/// A trait for problems that provide a cost function.
pub trait ProvidesCostFunction<R: RealScalar>: Problem {
    /// The type of the cost function.
    type CostFunction: CostFunction<R>;

    /// Builds the cost function.
    fn build_cost_function(&self) -> Self::CostFunction;
}

/// A trait for problems that provide a gradient.
pub trait ProvidesGradient<R: RealScalar>: Problem {
    /// The type of the cost function.
    type Gradient: Gradient<R>;

    /// Builds the gradient.
    fn build_gradient(&self) -> Self::Gradient;
}

/// A trait for problems that provide a Hessian.
pub trait ProvidesHessian<R: RealScalar>: Problem {
    /// The type of the cost function.
    type Hessian: Hessian<R>;

    /// Builds the Hessian.
    fn build_hessian(&self) -> Self::Hessian;
}

/// A trait for problems that provide a residual function.
pub trait ProvidesResidualFunction<R: RealScalar>: Problem {
    /// The type of the cost function.
    type ResidualFunction: ResidualFunction<R>;

    /// Builds the residual function.
    fn build_residual_function(&self) -> Self::ResidualFunction;
}

/// A trait for problems that provide a Jacobian.
pub trait ProvidesJacobian<R: RealScalar>: Problem {
    /// The type of the cost function.
    type Jacobian: Jacobian<R>;

    /// Builds the Jacobian.
    fn build_jacobian(&self) -> Self::Jacobian;
}
