use qudit_core::ComplexScalar;
use qudit_core::RealScalar;
use faer::prelude::*;
use faer::{Accum, Par, Col, Mat, MatRef, MatMut, Scale, linalg::matmul::matmul};
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
            tolerance: R::from64(1e-8),
            initial_lambda: R::from64(0.01),
            lambda_increase_factor: R::from64(2.0),
            lambda_decrease_factor: R::from64(3.0),
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

    /// Reference: https://arxiv.org/pdf/1201.5885
    /// Reference: https://scispace.com/pdf/the-levenberg-marquardt-algorithm-implementation-and-theory-u1ziue6l3q.pdf
    fn minimize(&self, objective: &mut P::Jacobian, x0: &[R]) -> MinimizationResult<R> {
        let n_params = objective.num_params();
        let mut x = Col::from_fn(x0.len(), |i| x0[i].clone());

        let mut lambda = self.initial_lambda;
        let mut Rbuf = objective.allocate_residual();
        let mut Jbuf = objective.allocate_jacobian();
        let mut JtJ: Mat<R> = Mat::zeros(objective.num_params(), objective.num_params());
        let mut Jtr: Mat<R> = Mat::zeros(objective.num_params(), 1);

        for iter in 0..self.max_iterations {
            // Safety: x is non null, initialized, and read-only while param_slice is alive.
            unsafe {
                // Calculate new residuals and jacobian
                let param_slice = std::slice::from_raw_parts(x.as_ptr(), x.nrows());
                objective.residuals_and_jacobian_into(param_slice, Rbuf.as_mut(), Jbuf.as_mut());
            }
            
            // Calculate current cost
            let current_cost = Rbuf.squared_norm_l2() * R::new(0.5);

            // Calculate hessian and gradient approximation to form new linear system problem
            matmul(&mut JtJ, Accum::Replace, &Jbuf.transpose(), &Jbuf, R::one(), Par::Seq);
            matmul(&mut Jtr, Accum::Replace, &Jbuf.transpose(), &Rbuf.as_mat(), R::one(), Par::Seq);

            // Damp hessian approximation
            for j in JtJ.diagonal_mut().column_vector_mut().iter_mut() {
                *j += lambda * *j; // TODO: add min and max?
            }

            // Solve for delta_x
            let llt = JtJ.llt(faer::Side::Lower).unwrap(); // TODO: If this fails; retry with
            // larger lambda
            let delta_x = llt.solve(&Jtr); // TODO: remove unnecessary allocations using scratch
            // space: https://faer.veganb.tw/docs/dense-linalg/linsolve/

            // Calculate x_new
            let x_new = &x - &delta_x.col(0); // TODO: remove allocation

            // Evaluate cost at x_new
            // Safety: x_new is non null, initialized, and read-only while param_slice is alive.
            unsafe {
                let param_slice = std::slice::from_raw_parts(x_new.as_ptr(), x_new.nrows());
                objective.residuals_into(param_slice, Rbuf.as_mut());
            }
            let new_cost = Rbuf.squared_norm_l2() * R::new(0.5);

            // println!("Current cost: {}, New cost: {}", current_cost, new_cost);
            if new_cost < current_cost {
                // Step accepted
                x = x_new;
                lambda /= self.lambda_decrease_factor;
            } else {
                // Step rejected
                lambda *= self.lambda_increase_factor;
            }
            // println!("Lambda: {}", lambda);

            // Check for convergence
            if delta_x.norm_l2() < self.tolerance {
                // println!("Converged");
                let params_out = Vec::from_iter(x.iter().copied());
                return MinimizationResult::simple_success(params_out, new_cost)
            }
        }

        // TODO: better result converying the failure
        MinimizationResult {
            params: vec![],
            fun: R::new(1.0),
            status: 1,
            message: None,
        }
    }
}
