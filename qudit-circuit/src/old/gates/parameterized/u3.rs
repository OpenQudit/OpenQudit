use std::ops::Range;

use faer_core::MatMut;
use gate_macros::SimpleSingleQubitGate;
use num_traits::Float;
use num_traits::Zero;

use super::super::NamedGate;
use crate::math::matrix::MatGradMut;
use crate::math::matrix::MatHessMut;
use crate::math::unitary::DifferentiableUnitaryFn;
use crate::math::unitary::DoublyDifferentiableUnitaryFn;
use crate::math::unitary::UnitaryFn;
use crate::math::BoundedFn;
use crate::math::ComplexScalar;
use crate::math::Function;
use crate::radices;
use crate::QuditRadices;
use crate::QuditSystem;

/// The single-qubit U3 gate parameterizes a general single-qubit unitary.
///
/// The U3 gate is given by the following matrix:
///
/// $$
/// \begin{pmatrix}
///    \cos{\frac{\theta_0}{2}} &
///    -\exp({i\theta_2})\sin{\frac{\theta_0}{2}} \\\\
///    \exp({i\theta_1})\sin{\frac{\theta_0}{2}} &
///    \exp({i(\theta_1 + \theta_2)})\cos{\frac{\theta_0}{2}} \\\\
/// \end{pmatrix}
/// $$
///
/// References:
/// - <https://arxiv.org/abs/1707.03429>
/// - <https://qiskit.org/documentation/stubs/qiskit.circuit.library.UGate.html>
#[derive(Hash, PartialEq, Eq, Clone, Debug, SimpleSingleQubitGate)]
pub struct U3Gate;

impl Function for U3Gate {
    fn get_num_params(&self) -> usize {
        3
    }
}

impl BoundedFn for U3Gate {
    fn get_bounds(&self) -> Vec<Range<f64>> {
        let pi = std::f64::consts::PI;
        let tpi = 2.0 * pi;

        vec![-tpi..tpi, -pi..pi, -pi..pi]
    }
}

impl<C: ComplexScalar> UnitaryFn<C> for U3Gate {
    #[inline]
    fn write_unitary(&self, params: &[C::Re], utry: &mut MatMut<C>) {
        let ct = C::complex((params[0] * C::real(0.5)).cos(), C::Re::zero()); // cos(theta/2)
        let st = C::complex((params[0] * C::real(0.5)).sin(), C::Re::zero()); // sin(theta/2)
        let ep = C::cis(params[1]); // exp(i * phi)
        let el = C::cis(params[2]); // exp(i * lambda)

        utry.write(0, 0, ct);
        utry.write(0, 1, el * -st);
        utry.write(1, 0, ep * st);
        utry.write(1, 1, ep * el * ct);
    }
}

impl<C: ComplexScalar> DifferentiableUnitaryFn<C> for U3Gate {
    #[inline]
    fn write_gradient(&self, params: &[C::Re], grad: &mut MatGradMut<C>) {
        let ct = C::complex((params[0] * C::real(0.5)).cos(), C::Re::zero()); // cos(theta/2)
        let st = C::complex((params[0] * C::real(0.5)).sin(), C::Re::zero()); // sin(theta/2)
        let sp = params[1].sin(); // sin(phi)
        let cp = params[1].cos(); // cos(phi)
        let sl = params[2].sin(); // sin(lambda)
        let cl = params[2].cos(); // cos(lambda)
        let ep = C::complex(cp, sp); // exp(i * phi)
        let el = C::complex(cl, sl); // exp(i * lambda)
        let dep = C::complex(-sp, cp); // i * exp(i * phi)
        let del = C::complex(-sl, cl); // i * exp(i * lambda)

        grad.write(0, 0, 0, st * C::real(-0.5));
        grad.write(0, 0, 1, ct * el * C::real(-0.5));
        grad.write(0, 1, 0, ct * ep * C::real(0.5));
        grad.write(0, 1, 1, st * el * ep * C::real(-0.5));

        grad.write(1, 1, 0, st * dep);
        grad.write(1, 1, 1, ct * el * dep);

        grad.write(2, 0, 1, -st * del);
        grad.write(2, 1, 1, ct * ep * del);
    }

    #[inline]
    fn write_unitary_and_gradient(
        &self,
        params: &[C::Re],
        utry: &mut MatMut<C>,
        grad: &mut MatGradMut<C>,
    ) {
        let ct = C::complex((params[0] * C::real(0.5)).cos(), C::Re::zero()); // cos(theta/2)
        let st = C::complex((params[0] * C::real(0.5)).sin(), C::Re::zero()); // sin(theta/2)
        let sp = params[1].sin(); // sin(phi)
        let cp = params[1].cos(); // cos(phi)
        let sl = params[2].sin(); // sin(lambda)
        let cl = params[2].cos(); // cos(lambda)
        let ep = C::complex(cp, sp); // exp(i * phi)
        let el = C::complex(cl, sl); // exp(i * lambda)
        let dep = C::complex(-sp, cp); // i * exp(i * phi)
        let del = C::complex(-sl, cl); // i * exp(i * lambda)

        utry.write(0, 0, ct);
        utry.write(0, 1, el * -st);
        utry.write(1, 0, ep * st);
        utry.write(1, 1, ep * el * ct);

        grad.write(0, 0, 0, st * C::real(-0.5));
        grad.write(0, 0, 1, ct * el * C::real(-0.5));
        grad.write(0, 1, 0, ct * ep * C::real(0.5));
        grad.write(0, 1, 1, st * el * ep * C::real(-0.5));

        grad.write(1, 1, 0, st * dep);
        grad.write(1, 1, 1, ct * el * dep);

        grad.write(2, 0, 1, -st * del);
        grad.write(2, 1, 1, ct * ep * del);
    }
}

impl<C: ComplexScalar> DoublyDifferentiableUnitaryFn<C> for U3Gate {
    #[inline]
    fn write_hessian(&self, params: &[C::Re], hess: &mut MatHessMut<C>) {
        let ct = C::complex((params[0] * C::real(0.5)).cos(), C::Re::zero()); // cos(theta/2)
        let st = C::complex((params[0] * C::real(0.5)).sin(), C::Re::zero()); // sin(theta/2)
        let sp = params[1].sin(); // sin(phi)
        let cp = params[1].cos(); // cos(phi)
        let sl = params[2].sin(); // sin(lambda)
        let cl = params[2].cos(); // cos(lambda)
        let ep = C::complex(cp, sp); // exp(i * phi)
        let el = C::complex(cl, sl); // exp(i * lambda)
        let dep = C::complex(-sp, cp); // i * exp(i * phi)
        let del = C::complex(-sl, cl); // i * exp(i * lambda)

        hess.write(0, 0, 0, 0, ct * C::real(-0.25));
        hess.write(0, 0, 0, 1, st * el * C::real(0.25));
        hess.write(0, 0, 1, 0, st * ep * C::real(-0.25));
        hess.write(0, 0, 1, 1, ct * el * ep * C::real(-0.25));

        hess.write(0, 1, 1, 0, ct * dep * C::real(0.5));
        hess.write(0, 1, 1, 1, st * el * dep * C::real(-0.5));

        hess.write(0, 2, 0, 1, ct * del * C::real(-0.5));
        hess.write(0, 2, 1, 1, st * ep * del * C::real(-0.5));

        hess.write(1, 1, 1, 0, -st * ep);
        hess.write(1, 1, 1, 1, -ct * el * ep);

        hess.write(1, 2, 1, 1, ct * del * dep);

        hess.write(2, 2, 0, 1, st * el);
        hess.write(2, 2, 1, 1, -ct * ep * el);
    }
}

#[cfg(test)]
mod test {
    use proptest::prelude::*;

    use faer_core::mat;

    use super::*;
    use crate::math::c64;
    use crate::math::unitary::UnitaryMatrix;

    #[test]
    fn test_u3_gate_zero_params_equal_identity() {
        let gate = U3Gate;
        let unitary = gate.get_unitary(&[0.0, 0.0, 0.0]);
        let correct: UnitaryMatrix<c64> = UnitaryMatrix::identity(radices![2]);
        unitary.assert_close_to(&correct);
    }

    #[test]
    fn test_u3_gate_pi_pi2_params_equal_x_gate() {
        let gate = U3Gate;
        let pi = std::f64::consts::PI;
        let unitary = gate.get_unitary(&[pi, -pi / 2.0, pi / 2.0]);
        let correct: UnitaryMatrix<c64> = UnitaryMatrix::new(
            radices![2],
            mat![
                [c64::new(0.0, 0.0), c64::new(1.0, 0.0)],
                [c64::new(1.0, 0.0), c64::new(0.0, 0.0)]
            ],
        );
        unitary.assert_close_to(&correct);
    }

    use crate::math::unitary::function::test::test_differentiable_unitary_fn;
    use crate::math::unitary::function::test::test_doubly_differentiable_unitary_fn;
    use crate::math::unitary::function::test::test_unitary_fn;

    test_unitary_fn!(Just(U3Gate), prop::collection::vec(-10.0..10.0, 3));
    test_differentiable_unitary_fn!(
        Just(U3Gate),
        prop::collection::vec(-10.0..10.0, 3)
    );
    test_doubly_differentiable_unitary_fn!(
        Just(U3Gate),
        prop::collection::vec(-10.0..10.0, 3)
    );
}
