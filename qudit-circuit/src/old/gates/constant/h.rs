use std::ops::Range;

use faer_core::MatMut;
use gate_macros::ConstantUnitaryFunction;
use gate_macros::SimpleSingleQuditGate;
use num_traits::Float;
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

/// The one-qudit Hadamard gate. This is a Clifford/Weyl-Heisenberg gate.
///
/// The qubit (radix = 2) Hadamard gate is given by the following matrix:
///
/// $$
/// \begin{pmatrix}
///     \frac{\sqrt{2}}{2} & \frac{\sqrt{2}}{2} \\\\
///     \frac{\sqrt{2}}{2} & -\frac{\sqrt{2}}{2} \\\\
/// \end{pmatrix}
/// $$
///
/// However, generally it is given by the following formula:
///
/// $$
/// H = \frac{1}{\sqrt{d}} \sum_{ij} \omega^{ij} \ket{i}\bra{j}
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
/// - <https://www.frontiersin.org/articles/10.3389/fphy.2020.589504/full>
/// - <https://pubs.aip.org/aip/jmp/article-abstract/56/3/032202/763827>
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
pub struct HGate {
    radix: usize,
}

impl<C: ComplexScalar> UnitaryFn<C> for HGate {
    #[inline]
    fn write_unitary(&self, _params: &[C::Re], utry: &mut MatMut<C>) {
        let omega =
            C::cis(C::real(2.0) * C::Re::PI() / C::real(self.radix as f64));
        let invsqrt = C::real(self.radix as f64).sqrt().recip();
        for i in 0..self.radix {
            for j in 0..self.radix {
                utry.write(
                    i,
                    j,
                    omega.powu((i * j).try_into().unwrap()) * invsqrt,
                );
            }
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
    fn test_qubit_h_gate() {
        let h_gate = HGate::new(2);
        let unitary: UnitaryMatrix<c64> = h_gate.get_unitary(&[]);
        let correct = mat![
            [
                c64::new(1.0 / 2.0_f64.sqrt(), 0.0),
                c64::new(1.0 / 2.0_f64.sqrt(), 0.0)
            ],
            [
                c64::new(1.0 / 2.0_f64.sqrt(), 0.0),
                c64::new(-1.0 / 2.0_f64.sqrt(), 0.0)
            ]
        ];
        unitary.assert_close_to(&correct);
    }

    #[test]
    fn test_qutrit_h_gate() {
        let h_gate = HGate::new(3);
        let unitary: UnitaryMatrix<c64> = h_gate.get_unitary(&[]);
        let omega = c64::from_polar(1.0, 2.0 * std::f64::consts::PI / 3.0);
        let correct = mat![
            [
                c64::new(1.0 / 3.0_f64.sqrt(), 0.0),
                c64::new(1.0 / 3.0_f64.sqrt(), 0.0),
                c64::new(1.0 / 3.0_f64.sqrt(), 0.0)
            ],
            [
                c64::new(1.0 / 3.0_f64.sqrt(), 0.0),
                omega * c64::new(1.0 / 3.0_f64.sqrt(), 0.0),
                omega.powi(2) * c64::new(1.0 / 3.0_f64.sqrt(), 0.0)
            ],
            [
                c64::new(1.0 / 3.0_f64.sqrt(), 0.0),
                omega.powi(2) * c64::new(1.0 / 3.0_f64.sqrt(), 0.0),
                omega * c64::new(1.0 / 3.0_f64.sqrt(), 0.0)
            ]
        ];
        unitary.assert_close_to(&correct);
    }
}
