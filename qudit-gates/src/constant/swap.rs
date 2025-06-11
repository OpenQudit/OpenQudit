use qudit_expr::{UnitaryExpression, UnitaryExpressionGenerator};

/// The qudit swap gate. This is a two-qudit Clifford/Weyl-Heisenberg gate
/// that swaps the state of two qudits.
///
/// The qubit (radix = 2) version is given by the following matrix:
///
/// $$
/// \begin{pmatrix}
///     1 & 0 & 0 & 0 \\\\
///     0 & 0 & 1 & 0 \\\\
///     0 & 1 & 0 & 0 \\\\
///     0 & 0 & 0 & 1 \\\\
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
#[derive(Hash, PartialEq, Eq, Clone, Debug)]
pub struct SwapGate {
    pub radix: usize,
}

impl SwapGate {
    pub fn new(radix: usize) -> Self {
        Self { radix }
    }
}

impl UnitaryExpressionGenerator for SwapGate {
    fn gen_expr(&self) -> UnitaryExpression {
        let proto = format!("utry Swap<{}, {}>()", self.radix, self.radix);
        let mut body = "".to_string();
        body += "[";
        for i in 0..self.radix {
            body += "[";
            let a_i = i / self.radix;
            let b_i = i % self.radix;
            for j in 0..self.radix {
                let a_j = j / self.radix;
                let b_j = j % self.radix;
                if a_i == b_j && b_i == a_j {
                    body += "1,";
                } else {
                    body += "0,";
                }
            }
            body += "],";
        }
        body += "]";

        UnitaryExpression::new(proto + "{" + &body + "}")
    }
}

// #[cfg(test)]
// mod test {
//     use qudit_core::matrix::mat;

//     use super::*;
//     use qudit_core::c64;
//     use qudit_core::unitary::UnitaryMatrix;
//     use qudit_core::unitary::UnitaryFn;

//     #[test]
//     fn test_qubit_swap_gate() {
//         let gate = SwapGate::new(2);
//         let unitary: UnitaryMatrix<c64> = gate.get_unitary(&[]);
//         let correct = mat![
//             [
//                 c64::new(1.0, 0.0),
//                 c64::new(0.0, 0.0),
//                 c64::new(0.0, 0.0),
//                 c64::new(0.0, 0.0)
//             ],
//             [
//                 c64::new(0.0, 0.0),
//                 c64::new(0.0, 0.0),
//                 c64::new(1.0, 0.0),
//                 c64::new(0.0, 0.0)
//             ],
//             [
//                 c64::new(0.0, 0.0),
//                 c64::new(1.0, 0.0),
//                 c64::new(0.0, 0.0),
//                 c64::new(0.0, 0.0)
//             ],
//             [
//                 c64::new(0.0, 0.0),
//                 c64::new(0.0, 0.0),
//                 c64::new(0.0, 0.0),
//                 c64::new(1.0, 0.0)
//             ]
//         ];
//         unitary.assert_close_to(&correct);
//     }

//     #[test]
//     fn test_qutrit_swap_gate() {
//         let gate = SwapGate::new(3);
//         let unitary: UnitaryMatrix<c64> = gate.get_unitary(&[]);
//         let correct = mat![
//             [
//                 c64::new(1.0, 0.0),
//                 c64::new(0.0, 0.0),
//                 c64::new(0.0, 0.0),
//                 c64::new(0.0, 0.0),
//                 c64::new(0.0, 0.0),
//                 c64::new(0.0, 0.0),
//                 c64::new(0.0, 0.0),
//                 c64::new(0.0, 0.0),
//                 c64::new(0.0, 0.0)
//             ],
//             [
//                 c64::new(0.0, 0.0),
//                 c64::new(0.0, 0.0),
//                 c64::new(0.0, 0.0),
//                 c64::new(1.0, 0.0),
//                 c64::new(0.0, 0.0),
//                 c64::new(0.0, 0.0),
//                 c64::new(0.0, 0.0),
//                 c64::new(0.0, 0.0),
//                 c64::new(0.0, 0.0)
//             ],
//             [
//                 c64::new(0.0, 0.0),
//                 c64::new(0.0, 0.0),
//                 c64::new(0.0, 0.0),
//                 c64::new(0.0, 0.0),
//                 c64::new(0.0, 0.0),
//                 c64::new(0.0, 0.0),
//                 c64::new(1.0, 0.0),
//                 c64::new(0.0, 0.0),
//                 c64::new(0.0, 0.0)
//             ],
//             [
//                 c64::new(0.0, 0.0),
//                 c64::new(1.0, 0.0),
//                 c64::new(0.0, 0.0),
//                 c64::new(0.0, 0.0),
//                 c64::new(0.0, 0.0),
//                 c64::new(0.0, 0.0),
//                 c64::new(0.0, 0.0),
//                 c64::new(0.0, 0.0),
//                 c64::new(0.0, 0.0)
//             ],
//             [
//                 c64::new(0.0, 0.0),
//                 c64::new(0.0, 0.0),
//                 c64::new(0.0, 0.0),
//                 c64::new(0.0, 0.0),
//                 c64::new(1.0, 0.0),
//                 c64::new(0.0, 0.0),
//                 c64::new(0.0, 0.0),
//                 c64::new(0.0, 0.0),
//                 c64::new(0.0, 0.0)
//             ],
//             [
//                 c64::new(0.0, 0.0),
//                 c64::new(0.0, 0.0),
//                 c64::new(0.0, 0.0),
//                 c64::new(0.0, 0.0),
//                 c64::new(0.0, 0.0),
//                 c64::new(0.0, 0.0),
//                 c64::new(0.0, 0.0),
//                 c64::new(1.0, 0.0),
//                 c64::new(0.0, 0.0)
//             ],
//             [
//                 c64::new(0.0, 0.0),
//                 c64::new(0.0, 0.0),
//                 c64::new(1.0, 0.0),
//                 c64::new(0.0, 0.0),
//                 c64::new(0.0, 0.0),
//                 c64::new(0.0, 0.0),
//                 c64::new(0.0, 0.0),
//                 c64::new(0.0, 0.0),
//                 c64::new(0.0, 0.0)
//             ],
//             [
//                 c64::new(0.0, 0.0),
//                 c64::new(0.0, 0.0),
//                 c64::new(0.0, 0.0),
//                 c64::new(0.0, 0.0),
//                 c64::new(0.0, 0.0),
//                 c64::new(1.0, 0.0),
//                 c64::new(0.0, 0.0),
//                 c64::new(0.0, 0.0),
//                 c64::new(0.0, 0.0)
//             ],
//             [
//                 c64::new(0.0, 0.0),
//                 c64::new(0.0, 0.0),
//                 c64::new(0.0, 0.0),
//                 c64::new(0.0, 0.0),
//                 c64::new(0.0, 0.0),
//                 c64::new(0.0, 0.0),
//                 c64::new(0.0, 0.0),
//                 c64::new(0.0, 0.0),
//                 c64::new(1.0, 0.0)
//             ],
//         ];
//         unitary.assert_close_to(&correct);
//     }
// }
