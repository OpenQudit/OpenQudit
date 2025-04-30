use std::ops::Range;

use faer_core::MatMut;
use gate_macros::SimpleSingleQuditGate;
use num_traits::Float;

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

/// The single-qudit phase gate.
///
/// The common qubit phase gate is given by the following matrix:
///
/// $$
/// \begin{pmatrix}
///     1 & 0 \\\\
///     0 & \exp({i\theta}) \\\\
/// \end{pmatrix}
/// $$
///
/// The qutrit phase gate has two parameterized relative phases:
///
/// $$
/// \begin{pmatrix}
///     1 & 0 & 0 \\\\
///     0 & \exp({i\theta_0}) & 0 \\\\
///    0 & 0 & \exp({i\theta_1}) \\\\
/// \end{pmatrix}
/// $$
///
/// The d-level phase gate has d-1 parameterized relative phases. This
/// gate is Clifford iff all of the relative phases are powers of roots
/// of unity.
///
/// References:
/// - <https://www.nature.com/articles/s41467-022-34851-z>
/// - <https://arxiv.org/pdf/2204.13681.pdf>
#[derive(Hash, PartialEq, Eq, Clone, Debug, SimpleSingleQuditGate)]
pub struct PGate {
    radix: usize,
}

impl Function for PGate {
    #[inline]
    fn get_num_params(&self) -> usize {
        self.radix - 1
    }
}

impl BoundedFn for PGate {
    #[inline]
    fn get_bounds(&self) -> Vec<Range<f64>> {
        let pi = std::f64::consts::PI;
        vec![-pi..pi; self.get_num_params()]
    }
}

impl<C: ComplexScalar> UnitaryFn<C> for PGate {
    #[inline]
    fn write_unitary(&self, params: &[C::Re], utry: &mut MatMut<C>) {
        let phases = params.iter().map(|&p| C::cis(p));
        for (i, p) in phases.enumerate() {
            utry.write(i + 1, i + 1, p);
        }
    }
}

impl<C: ComplexScalar> DifferentiableUnitaryFn<C> for PGate {
    #[inline]
    fn write_gradient(&self, params: &[C::Re], grad: &mut MatGradMut<C>) {
        let dphases = params.iter().map(|p| C::complex(-p.sin(), p.cos()));
        for (i, dp) in dphases.enumerate() {
            grad.write(i, i + 1, i + 1, dp);
        }
    }

    #[inline]
    fn write_unitary_and_gradient(
        &self,
        params: &[C::Re],
        utry: &mut MatMut<C>,
        grad: &mut MatGradMut<C>,
    ) {
        // Add debug checks for param length
        let sines = params.iter().map(|p| p.sin());
        let cosines = params.iter().map(|p| p.cos());
        let phases = sines
            .zip(cosines)
            .map(|(s, c)| (C::complex(c, s), C::complex(-s, c)));
        for (i, (p, dp)) in phases.enumerate() {
            utry.write(i + 1, i + 1, p);
            grad.write(i, i + 1, i + 1, dp);
        }
    }
}

impl<C: ComplexScalar> DoublyDifferentiableUnitaryFn<C> for PGate {
    #[inline]
    fn write_hessian(&self, params: &[C::Re], hess: &mut MatHessMut<C>) {
        let ddphases = params.iter().map(|p| C::complex(-p.cos(), -p.sin()));
        for (i, ddp) in ddphases.enumerate() {
            hess.write(i, i, i + 1, i + 1, ddp);
        }
    }
}

#[cfg(test)]
mod test {
    use proptest::prelude::*;

    use super::*;
    use crate::gates::ZGate;
    use crate::math::c64;
    use crate::math::function::strategies::arbitrary_with_params_strategy;
    use crate::math::unitary::UnitaryMatrix;

    proptest! {
        #[test]
        fn test_p_gate_zero_params_equal_identity(radix in 2..10usize) {
            let gate = PGate::new(radix);
            let unitary = gate.get_unitary(&vec![0.0; radix - 1]);
            let correct: UnitaryMatrix<c64> = UnitaryMatrix::identity(radices![radix]);
            unitary.assert_close_to(&correct);
        }

        #[test]
        fn test_p_gate_unity_roots_params_equal_z_gate(radix in 2..10usize) {
            let gate = PGate::new(radix);
            let pi = std::f64::consts::PI;
            let params = Vec::from_iter((0..(radix-1)).map(|i| 2.0 * pi * ((i + 1) as f64) / (radix as f64)));
            let unitary = gate.get_unitary(&params);
            let correct: UnitaryMatrix<c64> = ZGate::new(radix).get_unitary(&[]);
            unitary.assert_close_to(&correct);
        }
    }

    use crate::math::unitary::function::test::test_differentiable_unitary_fn;
    use crate::math::unitary::function::test::test_doubly_differentiable_unitary_fn;
    use crate::math::unitary::function::test::test_unitary_fn;

    test_unitary_fn!(arbitrary_with_params_strategy::<PGate>());
    test_differentiable_unitary_fn!(arbitrary_with_params_strategy::<PGate>());
    test_doubly_differentiable_unitary_fn!(arbitrary_with_params_strategy::<
        PGate,
    >());
}
