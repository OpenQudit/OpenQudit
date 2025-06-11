use qudit_expr::{UnitaryExpression, UnitaryExpressionGenerator};

/// The one-qudit shift (X) gate. This is a Weyl-Heisenberg gate.
///
/// This gate shifts the state of a qudit up by one level modulo. For
/// example, the shift gate on a qubit is the Pauli-X gate. The shift
/// gate on a qutrit is the following matrix:
///
/// $$
/// \begin{pmatrix}
///     0 & 0 & 1 \\\\
///     1 & 0 & 0 \\\\
///     0 & 1 & 0 \\\\
/// \end{pmatrix}
/// $$
///
/// The shift gate is generally given by the following formula:
///
/// $$
/// \begin{equation}
///     X = \sum_a |a + 1 mod d ><a|
/// \end{equation}
/// $$
///
/// where d is the number of levels (2 levels is a qubit, 3 levels is
/// a qutrit, etc.)
///
/// References:
///     - https://arxiv.org/pdf/2302.07966.pdf
#[derive(Hash, PartialEq, Eq, Clone, Debug)]
pub struct XGate {
    pub radix: usize,
}

impl XGate {
    pub fn new(radix: usize) -> Self {
        Self { radix }
    }
}

impl UnitaryExpressionGenerator for XGate {
    fn gen_expr(&self) -> UnitaryExpression {
        let proto = format!("utry X<{}>()", self.radix);
        let mut body = "[".to_string();
        for i in 0..self.radix {
            body += "[";
            for j in 0..self.radix {
                if j + 1 % self.radix == i {
                    body += "1, ";
                } else {
                    body += "0, ";
                }
            }
            body += "],";
        }
        body += "]";

        UnitaryExpression::new(proto + "{" + &body + "}")
    }
}

#[cfg(test)]
mod tests {
    use qudit_core::{c32, matrix::mat, unitary::UnitaryMatrix};
    use qudit_core::unitary::UnitaryFn;

    use super::*;

    #[test]
    fn test_qutrit_x_gate() {
        let utry: UnitaryMatrix<c32> = XGate::new(3).gen_expr().get_unitary([]);
        let expected = mat![
            [0, 0, 1],
            [1, 0, 0],
            [0, 1, 0]
        ]; // TODO: Make a comparison function/macro for matrices
        // The compiler error is because type inference uses the matrix integer elements
        // to infer integers. This entire test paradigm should be greatly simplied...
        assert_eq!(utry, expected);
    }
}
