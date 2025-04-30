use std::ops::Range;

use faer_core::MatMut;
use gate_macros::ConstantUnitaryFunction;
use gate_macros::SimpleSingleQuditGate;
use num_traits::FloatConst;

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

/// The one-qudit Z (clock) gate. This is a Clifford/Weyl-Heisenberg gate.
///
/// The qubit (radix = 2) Z gate is given by the following matrix:
///
/// $$
/// \begin{pmatrix}
///     1 & 0 \\\\
///     0 & -1 \\\\
/// \end{pmatrix}
/// $$
///
/// The qutrit (radix = 3) Z gate is given by the following matrix:
///
/// $$
/// \begin{pmatrix}
///     1 & 0 & 0 \\\\
///     0 & \omega & 0 \\\\
///     0 & 0 & \omega^{2} \\\\
/// \end{pmatrix}
/// $$
///
/// However, generally it is given by the following formula:
///
/// $$
/// Z = \sum_{i} \omega^{i} \ket{i}\bra{i}
/// $$
///
/// where
///
/// $$
/// \omega = \exp\Big(\frac{2\pi i}{d}\Big)
/// $$
///
/// and $d$ is the number of levels (2 levels is a qubit, 3 levels is a qutrit,
/// etc.)
///
/// References:
/// - <https://en.wikipedia.org/wiki/Generalizations_of_Pauli_matrices>
/// - <https://arxiv.org/pdf/2302.07966.pdf>
/// - <https://arxiv.org/pdf/1206.1598.pdf>
/// - <https://journals.aps.org/pra/abstract/10.1103/PhysRevA.86.022316>
/// - <https://arxiv.org/pdf/1701.07902.pdf>
#[derive(
    Hash,
    PartialEq,
    Eq,
    Clone,
    Debug,
    SimpleSingleQuditGate,
    ConstantUnitaryFunction,
)]
pub struct ZGate {
    radix: usize,
}

impl<C: ComplexScalar> UnitaryFn<C> for ZGate {
    #[inline]
    fn write_unitary(&self, _params: &[C::Re], utry: &mut MatMut<C>) {
        let omega =
            C::cis(C::real(2.0) * C::Re::PI() / C::real(self.radix as f64));
        for i in 1..self.radix {
            utry.write(i, i, omega.powi(i as i32))
        }
    }
}

#[cfg(test)]
mod test {
    use crate::math::c64;
    use crate::math::unitary::UnitaryMatrix;
    use faer_core::mat;

    use super::*;

    #[test]
    fn test_qubit_z_gate() {
        let gate = ZGate::new(2);
        let unitary: UnitaryMatrix<c64> = gate.get_unitary(&[]);
        let correct = mat![
            [c64::new(1.0, 0.0), c64::new(0.0, 0.0)],
            [c64::new(0.0, 0.0), c64::new(-1.0, 0.0)]
        ];
        unitary.assert_close_to(&correct);
    }

    #[test]
    fn test_qutrit_z_gate() {
        let gate = ZGate::new(3);
        let unitary: UnitaryMatrix<c64> = gate.get_unitary(&[]);
        let omega = c64::from_polar(1.0, 2.0 * std::f64::consts::PI / 3.0);
        let correct = mat![
            [c64::new(1.0, 0.0), c64::new(0.0, 0.0), c64::new(0.0, 0.0)],
            [c64::new(0.0, 0.0), omega, c64::new(0.0, 0.0)],
            [c64::new(0.0, 0.0), c64::new(0.0, 0.0), omega.powi(2)]
        ];
        unitary.assert_close_to(&correct);
    }
}
