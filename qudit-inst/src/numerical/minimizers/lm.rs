use qudit_core::ComplexScalar;
use qudit_core::RealScalar;
use faer::prelude::*;
use faer::{Col, Mat, MatRef, MatMut, Scale};
use crate::numerical::MinimizationAlgorithm;
use crate::numerical::MinimizationResult;
use crate::numerical::ProvidesJacobian;
use crate::numerical::Jacobian;
use crate::numerical::ResidualFunction;
use crate::numerical::Function;

/// Levenberg-Marquardt (LM) optimization algorithm.
///
/// This struct implements the Levenberg-Marquardt algorithm for non-linear
/// least squares problems. It combines the Gauss-Newton algorithm with
/// gradient descent.
pub struct LM<R: RealScalar> {
    /// Maximum number of iterations.
    pub max_iterations: usize,
    /// Tolerance for convergence (change in parameters).
    pub tolerance: R,
    /// Initial damping parameter.
    pub initial_lambda: R,
    /// Multiplier for lambda when a step is rejected.
    pub lambda_increase_factor: R,
    /// Divisor for lambda when a step is accepted.
    pub lambda_decrease_factor: R,
}

impl<R: RealScalar> Default for LM<R> {
    fn default() -> Self {
        Self {
            max_iterations: 100,
            tolerance: R::from64(1e-6),
            initial_lambda: R::from64(1e-3),
            lambda_increase_factor: R::from64(10.0),
            lambda_decrease_factor: R::from64(10.0),
        }
    }
}

impl<R, P> MinimizationAlgorithm<R, P> for LM<R>
where
    R: RealScalar,
    P: ProvidesJacobian<R>,
{
    type Func = P::Jacobian;

    fn initialize(&self, problem: &P) -> P::Jacobian 
    {
        problem.build_jacobian()
    }

    fn minimize(&self, objective: &mut P::Jacobian, x0: &[R]) -> MinimizationResult<R> {
        let n_params = objective.num_params();
        let mut x = Col::from_fn(x0.len(), |i| x0[i]);

        let mut lambda = self.initial_lambda;

        for iter in 0..self.max_iterations {
            let (residuals_vec, jacobian_mat) = objective.cost_and_jacobian(x.as_slice());
            
            let r = Col::from_slice(residuals_vec);
            let J = Mat::from_row_major_slice(jacobian_mat, r.len(), n_params);

            // Calculate current cost
            let current_cost = r.dot(&r) * C::R::from_f64(0.5);

            // Build J^T J (approximation of Hessian) and J^T r
            let JtJ = J.transpose() * J;
            let JtR = J.transpose() * r;

            // Add damping to JtJ
            let damped_JtJ = JtJ + Mat::from_fn(n_params, n_params, |i, j| {
                if i == j { JtJ.read(i, i) * lambda } else { C::R::zero() }
            });

            // Solve for delta_x
            let delta_x_result = damped_JtJ.lu().solve(&JtR);

            let delta_x = match delta_x_result {
                Ok(dx) => dx,
                Err(_) => {
                    // Handle singular matrix: increase lambda and retry (or fail)
                    lambda *= self.lambda_increase_factor;
                    continue; // Skip to next iteration
                }
            };

            // Calculate x_new
            let x_new = &x + &delta_x;

            // Evaluate cost at x_new
            let (new_residuals_vec, _) = objective.cost_and_jacobian(x_new.as_slice());
            let new_r = Col::from_slice(new_residuals_vec);
            let new_cost = new_r.dot(&new_r) * C::R::from_f64(0.5);

            if new_cost < current_cost {
                // Step accepted
                x = x_new;
                lambda /= self.lambda_decrease_factor;
            } else {
                // Step rejected
                lambda *= self.lambda_increase_factor;
            }

            // Check for convergence
            if delta_x.norm_l2() < self.tolerance {
                return MinimizationResult {
                    converged: true,
                    iterations: iter + 1,
                    final_cost: current_cost,
                    params: x.as_slice().to_vec(),
                };
            }
        }

        // If max iterations reached without convergence
        MinimizationResult {
            converged: false,
            iterations: self.max_iterations,
            final_cost: objective.cost_and_jacobian(x.as_slice()).0.dot(&objective.cost_and_jacobian(x.as_slice()).0) * C::R::from_f64(0.5), // Final cost
            params: x.as_slice().to_vec(),
        }
    }
}
