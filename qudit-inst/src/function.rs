use qudit_core::ComplexScalar;

pub trait Function<C: ComplexScalar> {
    fn num_params(&self) -> usize;
}

/// A trait for functions that can be evaluated to a single cost value.
pub trait CostFunction<C: ComplexScalar>: Function<C> {
    fn cost(&self, params: &[C::R]) -> C::R;
}

/// A trait for functions that can provide a gradient.
///
/// This uses a "supertrait" (`: CostFunction<...>`) to enforce that any type
/// implementing `Gradient` must also implement `CostFunction`.
pub trait Gradient<C: ComplexScalar>: CostFunction<C> {
    /// Computes only the gradient of the function.
    fn gradient(&self, params: &[C::R]) -> &[C::R];

    /// Computes both the cost and the gradient in a single, potentially optimized pass.
    ///
    /// This method provides a default implementation that calls `cost` and `gradient`
    /// separately. Users with an efficient combined computation can override this
    /// method for better performance.
    fn cost_and_gradient(&self, params: &[C::R]) -> (C::R, &[C::R]) {
        // Default "un-optimized" implementation
        (self.cost(params), self.gradient(params))
    }
}

/// A trait for functions that compute a vector of residuals.
pub trait ResidualFunction<C: ComplexScalar> {
    fn residuals(&self, params: &[C::R]) -> &[C::R];
}

/// A trait for functions that can provide the Jacobian of the residuals.
///
/// This is a supertrait of `Residual`, enforcing that any type implementing
/// `Jacobian` must also implement `Residual`.
pub trait Jacobian<C: ComplexScalar>: ResidualFunction<C> {
    /// Computes only the jacobian of the function.
    fn jacobian(&self, params: &[C::R]) -> &[C::R];

    /// Computes both the cost and the jacobian in a single, potentially optimized pass.
    ///
    /// This method provides a default implementation that calls `cost` and `jacobian`
    /// separately. Users with an efficient combined computation can override this
    /// method for better performance.
    fn cost_and_jacobian(&self, params: &[C::R]) -> (&[C::R], &[C::R]) {
        // Default "un-optimized" implementation
        (self.residuals(params), self.jacobian(params))
    }
}


/// A trait for functions that can provide the Hessian matrix of the cost function.
///
/// This is a supertrait of `Gradient`, enforcing that any type implementing
/// `Hessian` must also implement `Gradient`.
pub trait Hessian<C: ComplexScalar>: Gradient<C> {
    /// Computes only the Hessian matrix of the function.
    /// The Hessian is typically returned as a flattened `n x n` matrix in row-major order.
    fn hessian(&self, params: &[C::R]) -> &[C::R];

    /// Computes the cost, gradient, and Hessian in a single, potentially optimized pass.
    ///
    /// This method provides a default implementation that calls `cost`, `gradient`, and `hessian`
    /// separately. Users with an efficient combined computation can override this
    /// method for better performance.
    fn cost_gradient_and_hessian(&self, params: &[C::R]) -> (C::R, &[C::R], &[C::R]) {
        // Default "un-optimized" implementation
        (self.cost(params), self.gradient(params), self.hessian(params))
    }
}
