use std::ops::Range;

use faer_core::MatMut;
use gate_macros::ConstantUnitaryFunction;
use gate_macros::SimpleQuditGate;

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

/// The qudit swap gate. This is a two-qudit Clifford/Weyl-Heisenberg gate
/// that swaps the state of two qudits.
///
/// The qubit (radix = 2) version is given by the following matrix:
///
/// $$
/// \begin{pmatrix}
//     1 & 0 & 0 & 0 \\\\
//     0 & 0 & 1 & 0 \\\\
//     0 & 1 & 0 & 0 \\\\
//     0 & 0 & 0 & 1 \\\\
/// \end{pmatrix}
/// $$
///
/// The qutrit (radix = 3) version is given by the following matrix:
///
/// $$
/// \begin{pmatrix}
///     1 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\\\
///     0 & 0 & 0 & 1 & 0 & 0 & 0 & 0 & 0 \\\\
///     0 & 0 & 0 & 0 & 0 & 0 & 1 & 0 & 0 \\\\
///     0 & 1 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\\\
///     0 & 0 & 0 & 0 & 1 & 0 & 0 & 0 & 0 \\\\
///     0 & 0 & 0 & 0 & 0 & 0 & 0 & 1 & 0 \\\\
///     0 & 0 & 1 & 0 & 0 & 0 & 0 & 0 & 0 \\\\
///     0 & 0 & 0 & 0 & 0 & 1 & 0 & 0 & 0 \\\\
///     0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 1 \\\\
/// \end{pmatrix}
/// $$
///
/// However, generally it is given by the following formula:
///
/// $$
/// SWAP_d = \sum_{a, b} \ket{ab}\bra{ba}
/// $$
///
/// where $d$ is the number of levels (2 levels is a qubit, 3 levels is a
/// qutrit, etc.)
///
/// References:
/// - <https://link.springer.com/article/10.1007/s11128-013-0621-x>
/// - <https://arxiv.org/pdf/1105.5485.pdf>
#[derive(
    Hash, PartialEq, Eq, Clone, Debug, SimpleQuditGate, ConstantUnitaryFunction,
)]
#[num_qudits(2)]
pub struct SwapGate {
    radix: usize,
}

impl<C: ComplexScalar> UnitaryFn<C> for SwapGate {
    #[inline]
    fn write_unitary(&self, _params: &[C::Re], utry: &mut MatMut<C>) {
        let dim = self.get_dimension();
        let one = C::one();
        let zero = C::zero();
        for col in 0..dim {
            utry.write(col, col, zero);
        }
        for col in 0..dim {
            let a = col / self.radix;
            let b = col % self.radix;
            let row = b * self.radix + a;
            utry.write(row, col, one);
        }
    }
}

#[cfg(test)]
mod test {
    use faer_core::mat;

    use super::*;
    use crate::math::c64;
    use crate::math::unitary::UnitaryMatrix;

    #[test]
    fn test_qubit_swap_gate() {
        let gate = SwapGate::new(2);
        let unitary: UnitaryMatrix<c64> = gate.get_unitary(&[]);
        let correct = mat![
            [
                c64::new(1.0, 0.0),
                c64::new(0.0, 0.0),
                c64::new(0.0, 0.0),
                c64::new(0.0, 0.0)
            ],
            [
                c64::new(0.0, 0.0),
                c64::new(0.0, 0.0),
                c64::new(1.0, 0.0),
                c64::new(0.0, 0.0)
            ],
            [
                c64::new(0.0, 0.0),
                c64::new(1.0, 0.0),
                c64::new(0.0, 0.0),
                c64::new(0.0, 0.0)
            ],
            [
                c64::new(0.0, 0.0),
                c64::new(0.0, 0.0),
                c64::new(0.0, 0.0),
                c64::new(1.0, 0.0)
            ]
        ];
        unitary.assert_close_to(&correct);
    }

    #[test]
    fn test_qutrit_swap_gate() {
        let gate = SwapGate::new(3);
        let unitary: UnitaryMatrix<c64> = gate.get_unitary(&[]);
        let correct = mat![
            [
                c64::new(1.0, 0.0),
                c64::new(0.0, 0.0),
                c64::new(0.0, 0.0),
                c64::new(0.0, 0.0),
                c64::new(0.0, 0.0),
                c64::new(0.0, 0.0),
                c64::new(0.0, 0.0),
                c64::new(0.0, 0.0),
                c64::new(0.0, 0.0)
            ],
            [
                c64::new(0.0, 0.0),
                c64::new(0.0, 0.0),
                c64::new(0.0, 0.0),
                c64::new(1.0, 0.0),
                c64::new(0.0, 0.0),
                c64::new(0.0, 0.0),
                c64::new(0.0, 0.0),
                c64::new(0.0, 0.0),
                c64::new(0.0, 0.0)
            ],
            [
                c64::new(0.0, 0.0),
                c64::new(0.0, 0.0),
                c64::new(0.0, 0.0),
                c64::new(0.0, 0.0),
                c64::new(0.0, 0.0),
                c64::new(0.0, 0.0),
                c64::new(1.0, 0.0),
                c64::new(0.0, 0.0),
                c64::new(0.0, 0.0)
            ],
            [
                c64::new(0.0, 0.0),
                c64::new(1.0, 0.0),
                c64::new(0.0, 0.0),
                c64::new(0.0, 0.0),
                c64::new(0.0, 0.0),
                c64::new(0.0, 0.0),
                c64::new(0.0, 0.0),
                c64::new(0.0, 0.0),
                c64::new(0.0, 0.0)
            ],
            [
                c64::new(0.0, 0.0),
                c64::new(0.0, 0.0),
                c64::new(0.0, 0.0),
                c64::new(0.0, 0.0),
                c64::new(1.0, 0.0),
                c64::new(0.0, 0.0),
                c64::new(0.0, 0.0),
                c64::new(0.0, 0.0),
                c64::new(0.0, 0.0)
            ],
            [
                c64::new(0.0, 0.0),
                c64::new(0.0, 0.0),
                c64::new(0.0, 0.0),
                c64::new(0.0, 0.0),
                c64::new(0.0, 0.0),
                c64::new(0.0, 0.0),
                c64::new(0.0, 0.0),
                c64::new(1.0, 0.0),
                c64::new(0.0, 0.0)
            ],
            [
                c64::new(0.0, 0.0),
                c64::new(0.0, 0.0),
                c64::new(1.0, 0.0),
                c64::new(0.0, 0.0),
                c64::new(0.0, 0.0),
                c64::new(0.0, 0.0),
                c64::new(0.0, 0.0),
                c64::new(0.0, 0.0),
                c64::new(0.0, 0.0)
            ],
            [
                c64::new(0.0, 0.0),
                c64::new(0.0, 0.0),
                c64::new(0.0, 0.0),
                c64::new(0.0, 0.0),
                c64::new(0.0, 0.0),
                c64::new(1.0, 0.0),
                c64::new(0.0, 0.0),
                c64::new(0.0, 0.0),
                c64::new(0.0, 0.0)
            ],
            [
                c64::new(0.0, 0.0),
                c64::new(0.0, 0.0),
                c64::new(0.0, 0.0),
                c64::new(0.0, 0.0),
                c64::new(0.0, 0.0),
                c64::new(0.0, 0.0),
                c64::new(0.0, 0.0),
                c64::new(0.0, 0.0),
                c64::new(1.0, 0.0)
            ],
        ];
        unitary.assert_close_to(&correct);
    }
}
