use faer_core::Mat;
use faer_core::MatMut;

use super::UnitaryGradient;
use super::UnitaryHessian;
use super::UnitaryMatrix;
use crate::math::matrix::MatGradMut;
use crate::math::matrix::MatHessMut;
use crate::math::ComplexScalar;
use crate::math::Function;
use crate::QuditSystem;

/// A unitary function is maps a vector of real parameters to a unitary matrix.
pub trait UnitaryFn<C: ComplexScalar>: QuditSystem + Function {
    #[inline]
    fn get_unitary(&self, params: &[C::Re]) -> UnitaryMatrix<C> {
        let mut utry =
            Mat::identity(self.get_dimension(), self.get_dimension());
        self.write_unitary(params, &mut utry.as_mut());
        UnitaryMatrix::new(self.get_radices(), utry)
    }

    // TODO: Guarante write functions are called only with identity (or zero for
    // gradient) buffer or with previous invocation
    fn write_unitary(&self, params: &[C::Re], utry: &mut MatMut<C>);
}

/// A differentiable unitary function can have its gradient computed.
pub trait DifferentiableUnitaryFn<C: ComplexScalar>: UnitaryFn<C> {
    #[inline]
    fn get_gradient(&self, params: &[C::Re]) -> UnitaryGradient<C> {
        todo!()
        // let mut grad = UnitaryGradient::zeros(self.get_radices(), params.len());
        // self.write_gradient(params, &mut grad);
        // grad
    }

    #[inline]
    fn get_unitary_and_gradient(
        &self,
        params: &[C::Re],
    ) -> (UnitaryMatrix<C>, UnitaryGradient<C>) {
        todo!()
        // let mut utry = Mat::identity(self.get_dimension(), self.get_dimension());
        // let mut grad = UnitaryGradient::zeros(self.get_radices(), params.len());
        // self.write_unitary_and_gradient(params, &mut utry.as_mut(), &mut grad);
        // (UnitaryMatrix::new(self.get_radices(), utry), grad)
    }

    fn write_gradient(&self, params: &[C::Re], out_grad: &mut MatGradMut<C>);

    #[inline]
    fn write_unitary_and_gradient(
        &self,
        params: &[C::Re],
        out_utry: &mut MatMut<C>,
        out_grad: &mut MatGradMut<C>,
    ) {
        self.write_unitary(params, out_utry);
        self.write_gradient(params, out_grad);
    }
}

/// A doubly differentiable unitary function can have its Hessian computed.
pub trait DoublyDifferentiableUnitaryFn<C: ComplexScalar>:
    DifferentiableUnitaryFn<C>
{
    #[inline]
    fn get_hessian(&self, params: &[C::Re]) -> UnitaryHessian<C> {
        todo!()
        // let mut hess = UnitaryHessian::zeros(self.get_radices(), params.len());
        // self.write_hessian(params, &mut hess);
        // hess
    }

    #[inline]
    fn get_unitary_and_hessian(
        &self,
        params: &[C::Re],
    ) -> (UnitaryMatrix<C>, UnitaryHessian<C>) {
        todo!()
        // let mut utry = Mat::identity(self.get_dimension(), self.get_dimension());
        // let mut hess = UnitaryHessian::zeros(self.get_radices(), params.len());
        // self.write_unitary_and_hessian(params, &mut utry.as_mut(), &mut hess);
        // (UnitaryMatrix::new(self.get_radices(), utry), hess)
    }

    #[inline]
    fn get_gradient_and_hessian(
        &self,
        params: &[C::Re],
    ) -> (UnitaryGradient<C>, UnitaryHessian<C>) {
        todo!()
        // let mut grad = UnitaryGradient::zeros(self.get_radices(), params.len());
        // let mut hess = UnitaryHessian::zeros(self.get_radices(), params.len());
        // self.write_gradient_and_hessian(params, &mut grad, &mut hess);
        // (grad, hess)
    }

    #[inline]
    fn get_unitary_gradient_and_hessian(
        &self,
        params: &[C::Re],
    ) -> (UnitaryMatrix<C>, UnitaryGradient<C>, UnitaryHessian<C>) {
        todo!()
        // let mut utry = Mat::identity(self.get_dimension(), self.get_dimension());
        // let mut grad = UnitaryGradient::zeros(self.get_radices(), params.len());
        // let mut hess = UnitaryHessian::zeros(self.get_radices(), params.len());
        // self.write_unitary_gradient_and_hessian(params, &mut utry.as_mut(), &mut grad, &mut hess);
        // (UnitaryMatrix::new(self.get_radices(), utry), grad, hess)
    }

    fn write_hessian(&self, params: &[C::Re], out_hess: &mut MatHessMut<C>);

    #[inline]
    fn write_unitary_and_hessian(
        &self,
        params: &[C::Re],
        out_utry: &mut MatMut<C>,
        out_hess: &mut MatHessMut<C>,
    ) {
        self.write_unitary(params, out_utry);
        self.write_hessian(params, out_hess);
    }

    #[inline]
    fn write_gradient_and_hessian(
        &self,
        params: &[C::Re],
        out_grad: &mut MatGradMut<C>,
        out_hess: &mut MatHessMut<C>,
    ) {
        self.write_gradient(params, out_grad);
        self.write_hessian(params, out_hess);
    }

    #[inline]
    fn write_unitary_gradient_and_hessian(
        &self,
        params: &[C::Re],
        out_utry: &mut MatMut<C>,
        out_grad: &mut MatGradMut<C>,
        out_hess: &mut MatHessMut<C>,
    ) {
        self.write_unitary(params, out_utry);
        self.write_gradient(params, out_grad);
        self.write_hessian(params, out_hess);
    }
}

// TODO: fn get_diagonal_hessian -> only pure diagonal elements (useful for
// some quasi-newton methods) (maybe as a separate trait?)

#[cfg(test)]
pub mod test {
    use super::*;

    // TODO: Add automated tests for write functions
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

    use faer_core::scale;
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
                                v.into_iter()
                                    .map(|elt| elt as f32)
                                    .collect::<Vec<f32>>()
                            })
                        ),
                        (
                            $gate_strategy,
                            $param_strategy.prop_map(|v| {
                                v.into_iter()
                                    .map(|elt| elt as f64)
                                    .collect::<Vec<f64>>()
                            })
                        )
                    );
                };
                ($unified_strategy:expr) => {
                    crate::math::unitary::function::test::$base_test_name!(
                        $unified_strategy.prop_map(|(g, v)| {
                            (
                                g,
                                v.into_iter()
                                    .map(|elt| elt as f32)
                                    .collect::<Vec<f32>>(),
                            )
                        }),
                        $unified_strategy.prop_map(|(g, v)| {
                            (
                                g,
                                v.into_iter()
                                    .map(|elt| elt as f64)
                                    .collect::<Vec<f64>>(),
                            )
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

    pub fn assert_unitary_gradient_function_works<C, F>(f: F, params: &[C::Re])
    where
        C: ComplexScalar,
        F: DifferentiableUnitaryFn<C>,
    {
        assert!(check_gradient_function_finite_difference(&f, params));
        assert!(check_gradient_function_approximate_hessian_symmetry(
            &f, params
        ));
    }

    fn check_gradient_function_finite_difference<C, F>(
        f: &F,
        params: &[C::Re],
    ) -> bool
    where
        C: ComplexScalar,
        F: DifferentiableUnitaryFn<C>,
    {
        let eps = C::GRAD_EPSILON;
        let grad =
            f.get_gradient(&params) * C::complex(eps * C::real(2.0), 0.0);
        for i in 0..f.get_num_params() {
            let mut params_plus = params.to_owned();
            params_plus[i] += eps;
            let mut params_minus = params.to_owned();
            params_minus[i] -= eps;
            let plus = f.get_unitary(&params_plus);
            let minus = f.get_unitary(&params_minus);
            let finite_diff = plus - minus;
            let error = finite_diff - grad[i].clone();
            if error.norm_l2() > eps {
                return false;
            }
        }
        true
    }

    /// <https://dl.acm.org/doi/10.1145/356012.356013>
    fn check_gradient_function_approximate_hessian_symmetry<C, F>(
        f: &F,
        params: &[C::Re],
    ) -> bool
    where
        C: ComplexScalar,
        F: DifferentiableUnitaryFn<C>,
    {
        let eps = C::GRAD_EPSILON;
        let grad = f.get_gradient(&params);
        let mut grads = Vec::new();
        for i in 0..f.get_num_params() {
            let mut params_plus = params.to_owned();
            params_plus[i] += eps;
            grads.push(f.get_gradient(&params_plus));
        }
        let mut hess_approx = Vec::new();
        for i in 0..f.get_num_params() {
            let mut hess_row_approx = Vec::new();
            for j in 0..f.get_num_params() {
                let finite_diff = (grads[j][i].clone() - grad[i].clone())
                    * scale(C::complex(eps, 0.0).faer_inv());
                hess_row_approx.push(finite_diff);
            }
            hess_approx.push(hess_row_approx);
        }
        for i in 0..f.get_num_params() {
            for j in (i + 1)..f.get_num_params() {
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

    pub fn assert_gradient_combo_fns_equal_separate_fns<C, F>(
        f: F,
        params: &[C::Re],
    ) where
        C: ComplexScalar,
        F: DifferentiableUnitaryFn<C>,
    {
        let utry = f.get_unitary(&params);
        let grad = f.get_gradient(&params);
        let (utry2, grad2) = f.get_unitary_and_gradient(&params);
        assert_eq!(utry, utry2);
        assert_eq!(grad, grad2);
    }

    pub fn assert_unitary_hessian_function_works<C, F>(f: F, params: &[C::Re])
    where
        C: ComplexScalar,
        F: DoublyDifferentiableUnitaryFn<C>,
    {
        assert!(check_hessian_function_finite_difference(&f, params));
        assert!(check_hessian_function_approximate_thirdorder_symmetry(
            &f, params
        ));
        // TODO: assert hessian is symmetric
    }

    fn check_hessian_function_finite_difference<C, F>(
        f: &F,
        params: &[C::Re],
    ) -> bool
    where
        C: ComplexScalar,
        F: DoublyDifferentiableUnitaryFn<C>,
    {
        let eps = C::GRAD_EPSILON;
        let scalar = C::real(2.0) * eps;
        let hess = f.get_hessian(&params) * C::complex(scalar, 0.0);
        for i in 0..f.get_num_params() {
            let mut params_plus = params.to_owned();
            params_plus[i] += eps;
            let mut params_minus = params.to_owned();
            params_minus[i] -= eps;
            let plus = f.get_gradient(&params_plus);
            let minus = f.get_gradient(&params_minus);
            for j in 0..plus.len() {
                let finite_diff = plus[j].clone() - minus[j].clone();
                let error =
                    finite_diff - hess.partials.get_matref(i, j).clone();
                if error.norm_l2() > eps {
                    return false;
                }
            }
        }
        true
    }

    fn check_hessian_function_approximate_thirdorder_symmetry<C, F>(
        f: &F,
        params: &[C::Re],
    ) -> bool
    where
        C: ComplexScalar,
        F: DoublyDifferentiableUnitaryFn<C>,
    {
        let eps = C::GRAD_EPSILON;
        let hess = f.get_hessian(&params);
        let mut hesss = Vec::new();
        for i in 0..f.get_num_params() {
            let mut params_plus = params.to_owned();
            params_plus[i] += eps;
            hesss.push(f.get_hessian(&params_plus));
        }
        let mut third_order_approx = Vec::new();
        for i in 0..f.get_num_params() {
            let mut third_order_major = Vec::new();
            for j in 0..f.get_num_params() {
                let finite_diff = (hesss[j].get_row(i) - hess.get_row(i))
                    * C::complex(eps, 0.0).faer_inv();
                third_order_major.push(finite_diff);
            }
            third_order_approx.push(third_order_major);
        }
        for i in 0..f.get_num_params() {
            for j in (i + 1)..f.get_num_params() {
                for (m1, m2) in third_order_approx[i][j]
                    .iter()
                    .zip(third_order_approx[j][i].iter())
                {
                    if (m1 - m2).norm_l2()
                // .powi(2)
                    > eps
                    // TODO: Python may have got forced indention right, I can get lazy
                    // TODO: Fix this amalgamation of code
                    {
                        return false;
                    }
                }
            }
        }
        true
    }

    pub fn assert_hessian_combo_fns_equal_separate_fns<C, F>(
        f: F,
        params: &[C::Re],
    ) where
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
