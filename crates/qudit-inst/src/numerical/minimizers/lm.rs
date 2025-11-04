#![allow(non_snake_case)] // A lot of math in here, okay to be flexible with var names.

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
use faer::linalg::solvers::Solve;

/// Levenberg-Marquardt (LM) optimization algorithm.
///
/// This struct implements the Levenberg-Marquardt algorithm for non-linear
/// least squares problems. It combines the Gauss-Newton algorithm with
/// gradient descent.
#[derive(Clone)]
pub struct LM<R: RealScalar> {
    /// Maximum number of iterations.
    pub max_iterations: usize,
    /// Tolerance for convergence (change in parameters).
    pub tolerance: R,
    /// Initial damping parameter.
    pub initial_lambda: Option<R>,
    /// Initial lambda scaling factor.
    pub tau: R,
    /// Multiplier for lambda when a step is rejected.
    pub lambda_increase_factor: R,
    /// Divisor for lambda when a step is accepted.
    pub lambda_decrease_factor: R,

    pub minimum_lambda: R,
    /// Maximum allowed lambda value - if exceeded, optimization fails
    pub maximum_lambda: R,
    /// Minimum gradient norm - if gradient becomes smaller, optimization fails
    pub minimum_gradient_norm: R,
    /// Absolute tolerance for cost difference - terminate if cost improvement is too small
    pub diff_tol_a: R,
    /// Relative tolerance for cost difference - terminate if cost improvement is too small relative to current cost
    pub diff_tol_r: R,
}

impl<R: RealScalar> Default for LM<R> {
    fn default() -> Self {
        Self {
            max_iterations: 75,
            tolerance: R::from64(1e-8),
            initial_lambda: None,
            tau: R::from64(1e-3),
            lambda_increase_factor: R::from64(2.0),
            lambda_decrease_factor: R::from64(3.0),
            minimum_lambda: R::from64(10.0)*R::epsilon(),
            maximum_lambda: R::from64(1e12),
            minimum_gradient_norm: R::from64(10.0)*R::epsilon(),
            diff_tol_a: R::from64(1e-10),
            diff_tol_r: R::from64(1e-6),
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
        // dbg!(&x0);
        let mut x = Col::from_fn(x0.len(), |i| x0[i].clone());

        let mut Rbuf = objective.allocate_residual();
        let mut Jbuf = objective.allocate_jacobian();
        let mut JtJ: Mat<R> = Mat::zeros(objective.num_params(), objective.num_params());
        let mut Jtr: Mat<R> = Mat::zeros(objective.num_params(), 1);
        let mut predicted_reduction_buf: Mat<R> = Mat::zeros(1, 1);

        // /////// FINITE ANALYSIS TEST
        // unsafe {
        //     // Calculate new residuals and jacobian
        //     let param_slice = std::slice::from_raw_parts(x.as_ptr(), x.nrows());
        //     objective.residuals_and_jacobian_into(param_slice, Rbuf.as_mut(), Jbuf.as_mut());
        // }

        // // Finite difference test for Jacobian correctness
        // let epsilon = R::from64(1e-6); // Small perturbation
        // let mut Rbuf_perturbed = objective.allocate_residual();

        // println!("\n--- Finite Difference Jacobian Test ---");
        // for j in 0..n_params {
        //     let mut x_perturbed = x.clone(); // Copy x
        //     // Perturb the j-th parameter
        //     x_perturbed[j] += epsilon;

        //     // Calculate residuals at x_perturbed
        //     unsafe {
        //         let param_slice = std::slice::from_raw_parts(x_perturbed.as_ptr(), x_perturbed.nrows());
        //         objective.residuals_into(param_slice, Rbuf_perturbed.as_mut());
        //     }

        //     // Calculate finite difference approximation for the j-th column of the Jacobian
        //     // J_fd_j = (R(x + e_j * epsilon) - R(x)) / epsilon
        //     let fd_jacobian_col_j = (Rbuf_perturbed.as_mat() - Rbuf.as_mat()) * Scale(R::one() / epsilon);

        //     // Get the analytical j-th column of the Jacobian
        //     let analytical_jacobian_col_j = Jbuf.col(j);

        //     // Compare: calculate the L2 norm of the difference
        //     let diff_norm = (fd_jacobian_col_j - analytical_jacobian_col_j.as_mat()).norm_l2();
        //     let analytical_norm = analytical_jacobian_col_j.norm_l2();

        //     println!(
        //         "Parameter {}: Diff Norm = {}, Analytical Norm = {}",
        //         j, diff_norm, analytical_norm
        //     );

        //     // Optional: Add a threshold for warning
        //     if analytical_norm > R::from64(1e-10) && diff_norm / analytical_norm > R::from64(1e-4) {
        //         // A relative error of 0.01% might be acceptable, adjust as needed
        //         println!(
        //             "  WARNING: Large relative discrepancy. Relative error: {}",
        //             diff_norm / analytical_norm
        //         );
        //     } else if analytical_norm <= R::from64(1e-10) && diff_norm > R::from64(1e-8) {
        //         // If analytical norm is very small, check absolute difference
        //         println!(
        //             "  WARNING: Large absolute discrepancy (analytical norm near zero). Absolute error: {}",
        //             diff_norm
        //         );
        //     }
        // }
        // println!("--- End Finite Difference Jacobian Test ---\n");
        // ///////

        // Safety: x is non null, initialized, and read-only while param_slice is alive.
        unsafe {
            // Calculate new residuals and jacobian
            let param_slice = std::slice::from_raw_parts(x.as_ptr(), x.nrows());
            objective.residuals_and_jacobian_into(param_slice, Rbuf.as_mut(), Jbuf.as_mut());
            // dbg!(&Jbuf);
        }

        // Calculate current cost
        let mut current_cost = Rbuf.squared_norm_l2() * R::from64(0.5);

        // Check for invalid initial cost
        if current_cost.is_nan() || current_cost.is_infinite() {
            return MinimizationResult {
                params: vec![],
                fun: current_cost,
                status: 2,
                message: Some("Initial cost function is invalid (NaN or infinity)".to_string()),
            };
        }

        // Calculate hessian and gradient approximation to form new linear system problem
        matmul(&mut JtJ, Accum::Replace, &Jbuf.transpose(), &Jbuf, R::one(), Par::Seq);
        matmul(&mut Jtr, Accum::Replace, &Jbuf.transpose(), &Rbuf.as_mat(), R::one(), Par::Seq);
   
        // Check gradient norm
        let gradient_norm = Jtr.norm_l2();
        if gradient_norm < self.minimum_gradient_norm {
            return MinimizationResult {
                params: Vec::from_iter(x.iter().copied()),
                fun: current_cost,
                status: 6,
                message: Some("Gradient norm too small - optimization stagnated".to_string()),
            };
        }
   
        // Start lambda scaled to problem if not specified
        let mut lambda = match self.initial_lambda {
            None => { 
                let mut iter = JtJ.diagonal().column_vector().iter();
                let mut l = iter.next().expect("Empty Jacobian");
                for cand_l in iter {
                    if cand_l > l {
                        l = cand_l;
                    }
                }
                *l
            }
            Some(l) => l,
        };

        for iter in 0..self.max_iterations {

            // Check if lambda has grown too large (ill-conditioned problem)
            if lambda > self.maximum_lambda {
                return MinimizationResult {
                    params: Vec::from_iter(x.iter().copied()),
                    fun: current_cost,
                    status: 3,
                    message: Some("Lambda exceeded maximum threshold - problem is ill-conditioned".to_string()),
                };
            }

            // Damp hessian approximation
            for j in JtJ.diagonal_mut().column_vector_mut().iter_mut() {
                *j += lambda * j.clamp(R::epsilon()*R::from64(10.0), R::from64(10.0));
            }

            // Check JTJ for NaN or infinity values
            let mut has_invalid_jtj = false;
            for col in JtJ.col_iter() {
                for elem in col.iter() {
                    if elem.is_nan() || elem.is_infinite() || elem.is_subnormal() {
                        has_invalid_jtj = true;
                    }
                }
            }
            if has_invalid_jtj {
                // println!("Step rejected due to NaN or infinity in JtJ matrix.");
                // Recalculate JtJ to reset it to the undamped version before increasing lambda
                matmul(&mut JtJ, Accum::Replace, &Jbuf.transpose(), &Jbuf, R::one(), Par::Seq);
                lambda *= self.lambda_increase_factor;
                // println!("New lambda: {}", lambda);
                continue;
            }

            // // check condition of JTJ
            // let s = JtJ.svd().unwrap();
            // let max_s = s.S().column_vector().iter().next().unwrap();
            // let min_s = s.S().column_vector().iter().last().unwrap();
            // if RealScalar::abs(*min_s) < R::from64(1e-12) { // Avoid division by a near-zero number
            //     dbg!("Infinity");
            // } else {
            //     dbg!(max_s.to64() / min_s.to64());
            // }


            // Solve for delta_x
            let delta_x = match JtJ.llt(faer::Side::Lower) {
                Ok(llt) => { llt.solve(&Jtr) },
                Err(e) => {
                    // Step rejected
                    // println!("Step rejected due to invalid llt: {}", e);
                    // matmul(&mut JtJ, Accum::Replace, &Jbuf.transpose(), &Jbuf, R::one(), Par::Seq);
                    // lambda *= self.lambda_increase_factor;
                    // println!("New lambda: {}", lambda);
                    // continue;
                    JtJ.lblt(faer::Side::Lower).solve(&Jtr)
                }
            };
            // TODO: If this fails; fail step
            // let delta_x = llt.solve(&Jtr); // TODO: remove unnecessary allocations using scratch
            // space: https://faer.veganb.tw/docs/dense-linalg/linsolve/

            // Calculate x_new
            let x_new = &x - &delta_x.col(0); // TODO: remove allocation

            // Evaluate cost at x_new
            // Safety: x_new is non null, initialized, and read-only while param_slice is alive.
            unsafe {
                let param_slice = std::slice::from_raw_parts(x_new.as_ptr(), x_new.nrows());
                objective.residuals_into(param_slice, Rbuf.as_mut());
            }
            let new_cost = Rbuf.squared_norm_l2() * R::from64(0.5);

            // Check for invalid cost
            if new_cost.is_nan() || new_cost.is_infinite() {
                return MinimizationResult {
                    params: Vec::from_iter(x.iter().copied()),
                    fun: current_cost,
                    status: 4,
                    message: Some("Cost function became invalid (NaN or infinity)".to_string()),
                };
            }

            // Calculate gain ratio for proper LM updates
            let actual_reduction = current_cost - new_cost;
            
            // Simplified predicted reduction: step_norm^2 * (lambda + gradient_norm)
            let step_norm = delta_x.norm_l2();
            let gradient_norm = Jtr.norm_l2();
            let predicted_reduction = step_norm * step_norm * (lambda + gradient_norm);
            
            let gain_ratio = if predicted_reduction < R::from64(1e-14) || predicted_reduction.is_nan() || predicted_reduction.is_infinite() {
                R::from64(-1.0) // Treat as bad step
            } else {
                actual_reduction / predicted_reduction
            };

            // Accept step if gain ratio is positive
            if gain_ratio > R::zero() {
                // Step accepted
                x = x_new;
                if new_cost.is_close_with_tolerance(current_cost, self.diff_tol_r, self.diff_tol_a) {
                // if RealScalar::abs(new_cost - current_cost) <= self.diff_tol_a + self.diff_tol_r * RealScalar::abs(current_cost) {
                    return MinimizationResult {
                        params: Vec::from_iter(x.iter().copied()),
                        fun: new_cost,
                        status: 7,
                        message: Some("Terminated because lack of sufficient cost improvement.".to_string()),
                    }
                }
                current_cost = new_cost;
                
                // Update lambda based on gain ratio quality
                if gain_ratio > R::from64(0.75) {
                    // Very good step, decrease damping aggressively
                    lambda /= self.lambda_decrease_factor;
                } else if gain_ratio > R::from64(0.25) {
                    // Moderate step, decrease damping moderately
                    lambda /= R::from64(2.0);
                }
                // For gain_ratio <= 0.25, keep lambda unchanged
                
                lambda = if lambda < self.minimum_lambda { self.minimum_lambda } else { lambda };

                // Check for invalid parameters
                for param in x.iter() {
                    if param.is_nan() || param.is_infinite() {
                        return MinimizationResult {
                            params: Vec::from_iter(x.iter().copied()),
                            fun: current_cost,
                            status: 5,
                            message: Some("Parameters became invalid (NaN or infinity)".to_string()),
                        };
                    }
                }

                // Safety: x is non null, initialized, and read-only while param_slice is alive.
                unsafe {
                    // Calculate new residuals and jacobian
                    let param_slice = std::slice::from_raw_parts(x.as_ptr(), x.nrows());
                    objective.residuals_and_jacobian_into(param_slice, Rbuf.as_mut(), Jbuf.as_mut());
                    // dbg!(&Jbuf);
                }

                // Calculate hessian and gradient approximation to form new linear system problem
                matmul(&mut JtJ, Accum::Replace, &Jbuf.transpose(), &Jbuf, R::one(), Par::Seq);
                matmul(&mut Jtr, Accum::Replace, &Jbuf.transpose(), &Rbuf.as_mat(), R::one(), Par::Seq);

                // Check gradient norm after step acceptance
                let gradient_norm = Jtr.norm_l2();
                if gradient_norm < self.minimum_gradient_norm {
                    return MinimizationResult {
                        params: Vec::from_iter(x.iter().copied()),
                        fun: current_cost,
                        status: 6,
                        message: Some("Gradient norm too small - optimization stagnated".to_string()),
                    };
                }
            } else {
                // Step rejected - poor gain ratio
                matmul(&mut JtJ, Accum::Replace, &Jbuf.transpose(), &Jbuf, R::one(), Par::Seq);
                
                // Update lambda based on how bad the step was
                if gain_ratio < R::from64(-0.5) {
                    // Very bad step, increase damping aggressively
                    lambda *= self.lambda_increase_factor * R::from64(2.0);
                } else {
                    // Moderately bad step, increase damping normally
                    lambda *= self.lambda_increase_factor;
                }
            }

            // Check for convergence 
            if delta_x.norm_l2() < self.tolerance {
                let params_out = Vec::from_iter(x.iter().copied());
                return MinimizationResult::simple_success(params_out, current_cost)
            }
        }

        // TODO: better result converying the failure
        MinimizationResult {
            params: vec![],
            fun: R::from64(1.0),
            status: 1,
            message: None,
        }
    }
}
