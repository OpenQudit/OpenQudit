//! Defines traits for functions that map real parameters to unitary matrices.

use crate::matrix::Mat;
use crate::matrix::MatMut;
use crate::matrix::MatVec;
use crate::matrix::MatVecMut;
use crate::matrix::SymSqMatMat;
use crate::matrix::SymSqMatMatMut;

use super::UnitaryMatrix;
use crate::ComplexScalar;
use crate::HasParams;
use crate::QuditSystem;

/// A trait representing a function that maps a vector of real parameters to a unitary matrix.
///
/// This trait extends [`QuditSystem`], which requires that every unitary function
/// be associated with a sized qudit system.
///
/// Type parameter `C` is constrained to implement `ComplexScalar`, which ensures that the
/// elements of the unitary matrix are complex numbers.
pub trait UnitaryFn<C: ComplexScalar>: QuditSystem + HasParams {
    /// Generates a unitary matrix from a slice of real parameters.
    ///
    /// This method constructs an identity matrix of appropriate dimensions for
    /// the quantum system and then delegates to `write_unitary` to fill in
    /// the actual unitary matrix elements.
    ///
    /// # Arguments
    ///
    /// * `params` - A slice of real numbers representing the parameters of the unitary function.
    ///
    /// # Returns
    ///
    /// A `UnitaryMatrix` object that encapsulates the generated unitary matrix along with the
    /// radices (dimensions) of the qudits in the quantum system.
    #[inline]
    fn get_unitary(&self, params: impl AsRef<[C::R]>) -> UnitaryMatrix<C> {
        let mut utry = Mat::identity(self.dimension(), self.dimension());
        self.write_unitary(params.as_ref(), utry.as_mut());
        UnitaryMatrix::new(self.radices(), utry)
    }

    /// Writes the elements of a unitary matrix into a mutable buffer based on the provided parameters.
    ///
    /// Callers must ensure this function is called with either an identity
    /// matrix or a zero matrix (for gradient computation) as the initial state of the buffer.
    /// It is also expected that the buffer may contain the result of a previous invocation.
    ///
    /// # Arguments
    ///
    /// * `params` - A slice of real numbers representing the parameters of the unitary function.
    /// * `utry` - A mutable reference to a matrix buffer where the unitary matrix will be written.
    fn write_unitary(&self, params: &[C::R], utry: MatMut<C>);
}

/// A differentiable unitary function can have its gradient computed.
pub trait DifferentiableUnitaryFn<C: ComplexScalar>: UnitaryFn<C> {
    /// Generates the gradient of this unitary function with respect to the parameters.
    #[inline]
    fn get_gradient(&self, params: &[C::R]) -> MatVec<C> {
        self.get_unitary_and_gradient(params).1
    }

    /// Generates the unitary matrix and gradient of this unitary function with respect to the parameters.
    #[inline]
    fn get_unitary_and_gradient(&self, params: &[C::R]) -> (UnitaryMatrix<C>, MatVec<C>) {
        let mut utry = Mat::identity(self.dimension(), self.dimension());
        let mut grad = MatVec::zeros(self.dimension(), self.dimension(), self.num_params());
        self.write_unitary_and_gradient(params, utry.as_mut(), grad.as_mut());
        (UnitaryMatrix::new(self.radices(), utry), grad)
    }

    /// Write the unitary matrix and gradient values into mutable buffers based on the provided parameters.
    ///
    /// Callers must ensure this function is called with an identity matrix for the unitary matrix and zero matrices for the gradient.
    /// It is also expected that the buffers may contain the result of a previous invocation.
    ///
    /// # Arguments
    ///
    /// * `params` - A slice of real numbers representing the input values for the function.
    /// * `out_utry` - A mutable reference to a matrix buffer where the unitary matrix will be written.
    /// * `out_grad` - A mutable reference to a matrix buffer where the gradient will be written.
    fn write_unitary_and_gradient(
        &self,
        params: &[C::R],
        out_utry: MatMut<C>,
        out_grad: MatVecMut<C>,
    );
}

/// A doubly differentiable unitary function can have its Hessian computed.
pub trait DoublyDifferentiableUnitaryFn<C: ComplexScalar>: DifferentiableUnitaryFn<C> {
    /// Generates the Hessian of this unitary function with respect to the parameters.
    ///
    /// The Hessian is a symmetric square matrix of second-order partial derivatives.
    #[inline]
    fn get_hessian(&self, params: &[C::R]) -> SymSqMatMat<C> {
        self.get_unitary_gradient_and_hessian(params).2
    }

    /// Generates the unitary matrix and Hessian at a specific point in parameter space.
    #[inline]
    fn get_unitary_and_hessian(&self, params: &[C::R]) -> (UnitaryMatrix<C>, SymSqMatMat<C>) {
        let (u, _, h) = self.get_unitary_gradient_and_hessian(params);
        (u, h)
    }

    /// Generates the gradient and Hessian of this unitary function with respect to the parameters.
    #[inline]
    fn get_gradient_and_hessian(&self, params: &[C::R]) -> (MatVec<C>, SymSqMatMat<C>) {
        let (_, g, h) = self.get_unitary_gradient_and_hessian(params);
        (g, h)
    }

    /// Generates the unitary matrix, gradient, and Hessian of this unitary function with respect to the parameters.
    #[inline]
    fn get_unitary_gradient_and_hessian(
        &self,
        params: &[C::R],
    ) -> (UnitaryMatrix<C>, MatVec<C>, SymSqMatMat<C>) {
        let mut utry = Mat::identity(self.dimension(), self.dimension());
        let mut grad = MatVec::zeros(self.dimension(), self.dimension(), self.num_params());
        let mut hess = SymSqMatMat::zeros(self.dimension(), self.dimension(), self.num_params());
        self.write_unitary_gradient_and_hessian(
            params,
            utry.as_mut(),
            grad.as_mut(),
            hess.as_mut(),
        );
        (UnitaryMatrix::new(self.radices(), utry), grad, hess)
    }

    /// Write the unitary matrix, gradient, and Hessian values into mutable buffers based on the provided parameters.
    ///
    /// Callers must ensure this function is called with an identity matrix for the unitary matrix, and zero matrices for the gradient and the Hessian.
    ///
    /// # Arguments
    ///
    /// * `params` - A slice of real numbers representing the input values for the function.
    /// * `out_utry` - A mutable reference to a matrix buffer where the unitary matrix will be written.
    /// * `out_grad` - A mutable reference to a matrix buffer where the gradient will be written.
    /// * `out_hess` - A mutable reference to a matrix buffer where the Hessian will be written.
    fn write_unitary_gradient_and_hessian(
        &self,
        params: &[C::R],
        out_utry: MatMut<C>,
        out_grad: MatVecMut<C>,
        out_hess: SymSqMatMatMut<C>,
    );
}

// TODO: fn get_diagonal_hessian -> only pure diagonal elements (useful for
// some quasi-newton methods) (maybe as a separate trait?)

#[cfg(test)]
pub mod test {
    use super::*;
    use faer::Scale;

    ///////////////////////////////////////////////////////////////////////////
    /// Macros for Test Generation
    ///////////////////////////////////////////////////////////////////////////

    macro_rules! test_unitary_fn_base {
        ($f32_strat:expr, $f64_strat:expr) => {
            proptest! {
                #[test]
                fn test_unitary_32_fn_get_unitary_is_unitary((ufn, params) in $f32_strat) {
                    let unitary32: UnitaryMatrix<crate::math::c32> = ufn.get_unitary(&params);
                    assert!(UnitaryMatrix::<crate::math::c32>::is_unitary(&unitary32));
                }

                #[test]
                fn test_unitary_64_fn_get_unitary_is_unitary((ufn, params) in $f64_strat) {
                    let unitary64: UnitaryMatrix<crate::math::c64> = ufn.get_unitary(&params);
                    assert!(UnitaryMatrix::<crate::math::c64>::is_unitary(&unitary64));
                }
            }
        };
    }

    macro_rules! test_differentiable_unitary_fn_base {
        ($f32_strat:expr, $f64_strat:expr) => {
            use crate::math::unitary::function::test::assert_gradient_combo_fns_equal_separate_fns;
            use crate::math::unitary::function::test::assert_unitary_gradient_function_works;

            proptest! {
                #[test]
                fn test_gradient_32_fn_closely_approximates_unitary((ufn, params) in $f32_strat) {
                    assert_unitary_gradient_function_works::<crate::math::c32, _>(ufn, &params);
                }

                #[test]
                fn test_gradient_64_fn_closely_approximates_unitary((ufn, params) in $f64_strat) {
                    assert_unitary_gradient_function_works::<crate::math::c64, _>(ufn, &params);
                }

                #[test]
                fn test_gradient_32_combo_fn_equal_separate_fns((ufn, params) in $f32_strat) {
                    assert_gradient_combo_fns_equal_separate_fns::<crate::math::c32, _>(ufn, &params);
                }

                #[test]
                fn test_gradient_64_combo_fn_equal_separate_fns((ufn, params) in $f64_strat) {
                    assert_gradient_combo_fns_equal_separate_fns::<crate::math::c64, _>(ufn, &params);
                }
            }
        };
    }

    macro_rules! test_doubly_differentiable_unitary_fn_base {
        ($f32_strat:expr, $f64_strat:expr) => {
            use crate::math::unitary::function::test::assert_hessian_combo_fns_equal_separate_fns;
            use crate::math::unitary::function::test::assert_unitary_hessian_function_works;

            proptest! {
                #[test]
                fn test_hessian_32_fn_closely_approximates_unitary((ufn, params) in $f32_strat) {
                    assert_unitary_hessian_function_works::<crate::math::c32, _>(ufn, &params);
                }

                #[test]
                fn test_hessian_64_fn_closely_approximates_unitary((ufn, params) in $f64_strat) {
                    assert_unitary_hessian_function_works::<crate::math::c64, _>(ufn, &params);
                }

                #[test]
                fn test_hessian_32_combo_fn_equal_separate_fns((ufn, params) in $f32_strat) {
                    assert_hessian_combo_fns_equal_separate_fns::<crate::math::c32, _>(ufn, &params);
                }

                #[test]
                fn test_hessian_64_combo_fn_equal_separate_fns((ufn, params) in $f64_strat) {
                    assert_hessian_combo_fns_equal_separate_fns::<crate::math::c64, _>(ufn, &params);
                }
            }
        };
    }

    pub(crate) use test_differentiable_unitary_fn_base;
    pub(crate) use test_doubly_differentiable_unitary_fn_base;
    pub(crate) use test_unitary_fn_base;

    macro_rules! build_test_fn_macro {
        ($macro_name:ident, $base_test_name:ident) => {
            macro_rules! $macro_name {
                ($gate_strategy:expr, $param_strategy:expr) => {
                    crate::math::unitary::function::test::$base_test_name!(
                        (
                            $gate_strategy,
                            $param_strategy.prop_map(|v| {
                                v.into_iter().map(|elt| elt as f32).collect::<Vec<f32>>()
                            })
                        ),
                        (
                            $gate_strategy,
                            $param_strategy.prop_map(|v| {
                                v.into_iter().map(|elt| elt as f64).collect::<Vec<f64>>()
                            })
                        )
                    );
                };
                ($unified_strategy:expr) => {
                    crate::math::unitary::function::test::$base_test_name!(
                        $unified_strategy.prop_map(|(g, v)| {
                            (g, v.into_iter().map(|elt| elt as f32).collect::<Vec<f32>>())
                        }),
                        $unified_strategy.prop_map(|(g, v)| {
                            (g, v.into_iter().map(|elt| elt as f64).collect::<Vec<f64>>())
                        })
                    );
                };
            }

            pub(crate) use $macro_name;
        };
    }

    build_test_fn_macro!(test_unitary_fn, test_unitary_fn_base);
    build_test_fn_macro!(
        test_differentiable_unitary_fn,
        test_differentiable_unitary_fn_base
    );
    build_test_fn_macro!(
        test_doubly_differentiable_unitary_fn,
        test_doubly_differentiable_unitary_fn_base
    );

    ///////////////////////////////////////////////////////////////////////////
    /// Unitary Function Automatic Tests
    ///////////////////////////////////////////////////////////////////////////

    pub fn assert_unitary_gradient_function_works<C, F>(f: F, params: &[C::R])
    where
        C: ComplexScalar,
        F: DifferentiableUnitaryFn<C>,
    {
        assert!(check_gradient_function_finite_difference(&f, params));
        assert!(check_gradient_function_approximate_hessian_symmetry(
            &f, params
        ));
    }

    fn check_gradient_function_finite_difference<C, F>(f: &F, params: &[C::R]) -> bool
    where
        C: ComplexScalar,
        F: DifferentiableUnitaryFn<C>,
    {
        let eps = C::GRAD_EPSILON;
        let grad = f.get_gradient(&params) * C::complex(eps * C::real(2.0), 0.0);
        for i in 0..f.num_params() {
            let mut params_plus = params.to_owned();
            params_plus[i] += eps;
            let mut params_minus = params.to_owned();
            params_minus[i] -= eps;
            let plus = f.get_unitary(&params_plus);
            let minus = f.get_unitary(&params_minus);
            let finite_diff = plus - minus;
            let error = finite_diff - grad.mat_ref(i);
            if error.norm_l2() > eps {
                return false;
            }
        }
        true
    }

    /// <https://dl.acm.org/doi/10.1145/356012.356013>
    fn check_gradient_function_approximate_hessian_symmetry<C, F>(f: &F, params: &[C::R]) -> bool
    where
        C: ComplexScalar,
        F: DifferentiableUnitaryFn<C>,
    {
        let eps = C::GRAD_EPSILON;
        let grad = f.get_gradient(&params);
        let mut grads = Vec::new();
        for i in 0..f.num_params() {
            let mut params_plus = params.to_owned();
            params_plus[i] += eps;
            grads.push(f.get_gradient(&params_plus));
        }
        let mut hess_approx = Vec::new();
        for i in 0..f.num_params() {
            let mut hess_row_approx = Vec::new();
            for j in 0..f.num_params() {
                let finite_diff = (grads[j].mat_ref(i).clone() - grad.mat_ref(i).clone())
                    * Scale(C::complex(eps, 0.0).inv());
                hess_row_approx.push(finite_diff);
            }
            hess_approx.push(hess_row_approx);
        }
        for i in 0..f.num_params() {
            for j in (i + 1)..f.num_params() {
                if (hess_approx[i][j].to_owned() - hess_approx[j][i].to_owned())
                    .norm_l2()
                    // .powi(2)
                    > eps
                {
                    return false;
                }
            }
        }
        true
    }

    pub fn assert_gradient_combo_fns_equal_separate_fns<C, F>(f: F, params: &[C::R])
    where
        C: ComplexScalar,
        F: DifferentiableUnitaryFn<C>,
    {
        let utry = f.get_unitary(&params);
        let grad = f.get_gradient(&params);
        let (utry2, grad2) = f.get_unitary_and_gradient(&params);
        assert_eq!(utry, utry2);
        assert_eq!(grad, grad2);
    }

    pub fn assert_unitary_hessian_function_works<C, F>(f: F, params: &[C::R])
    where
        C: ComplexScalar,
        F: DoublyDifferentiableUnitaryFn<C>,
    {
        assert!(check_hessian_function_finite_difference(&f, params));
        assert!(check_hessian_function_approximate_thirdorder_symmetry(
            &f, params
        ));
    }

    fn check_hessian_function_finite_difference<C, F>(f: &F, params: &[C::R]) -> bool
    where
        C: ComplexScalar,
        F: DoublyDifferentiableUnitaryFn<C>,
    {
        let eps = C::GRAD_EPSILON;
        let scalar = C::real(2.0) * eps;
        let hess = f.get_hessian(&params) * C::complex(scalar, 0.0);
        for i in 0..f.num_params() {
            let mut params_plus = params.to_owned();
            params_plus[i] += eps;
            let mut params_minus = params.to_owned();
            params_minus[i] -= eps;
            let plus = f.get_gradient(&params_plus);
            let minus = f.get_gradient(&params_minus);
            for j in 0..plus.len() {
                let finite_diff = plus.mat_ref(j).clone() - minus.mat_ref(j).clone();
                let error = finite_diff - hess.mat_ref(i, j).clone();
                if error.norm_l2() > eps {
                    return false;
                }
            }
        }
        true
    }

    fn check_hessian_function_approximate_thirdorder_symmetry<C, F>(f: &F, params: &[C::R]) -> bool
    where
        C: ComplexScalar,
        F: DoublyDifferentiableUnitaryFn<C>,
    {
        let eps = C::GRAD_EPSILON;
        let hess = f.get_hessian(&params);
        let mut hesss = Vec::new();
        for i in 0..f.num_params() {
            let mut params_plus = params.to_owned();
            params_plus[i] += eps;
            hesss.push(f.get_hessian(&params_plus));
        }
        let mut third_order_approx = Vec::new();
        for i in 0..f.num_params() {
            let mut third_order_major = Vec::new();
            for j in 0..f.num_params() {
                let finite_diff =
                    (hesss[j].get_row(i) - hess.get_row(i)) * C::complex(eps, 0.0).inv();
                third_order_major.push(finite_diff);
            }
            third_order_approx.push(third_order_major);
        }
        for i in 0..f.num_params() {
            for j in (i + 1)..f.num_params() {
                for k in 0..third_order_approx[i][j].nmats() {
                    let m1 = third_order_approx[i][j].mat_ref(k);
                    let m2 = third_order_approx[j][i].mat_ref(k);
                    if (m1 - m2).norm_l2() > eps {
                        return false;
                    }
                }
            }
        }
        true
    }

    pub fn assert_hessian_combo_fns_equal_separate_fns<C, F>(f: F, params: &[C::R])
    where
        C: ComplexScalar,
        F: DoublyDifferentiableUnitaryFn<C>,
    {
        let utry = f.get_unitary(&params);
        let grad = f.get_gradient(&params);
        let hess = f.get_hessian(&params);

        let (utry2, hess2) = f.get_unitary_and_hessian(&params);
        let (grad2, hess3) = f.get_gradient_and_hessian(&params);
        let (utry3, grad3, hess4) = f.get_unitary_gradient_and_hessian(&params);

        assert_eq!(utry, utry2);
        assert_eq!(utry, utry3);

        assert_eq!(grad, grad2);
        assert_eq!(grad, grad3);

        assert_eq!(hess, hess2);
        assert_eq!(hess, hess3);
        assert_eq!(hess, hess4);
    }
}
