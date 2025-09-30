use qudit_expr::{UnitaryExpression, ExpressionGenerator};

/// The one-qudit clock (Z) gate. This is a Weyl-Heisenberg gate.
///
/// This gate shifts the state of a qudit up by one level modulo. For
/// example, the clock gate on a qubit is the Pauli-Z gate. The clock
/// gate on a qutrit is the following matrix:
///
/// $$
/// \begin{pmatrix}
///     1 & 0 & 0 \\\\
///     0 & e^{\frac{2\pi i}{3} & 0 \\\\
///     0 & 0 & e^{\frac{4\pi i}{3} \\\\
/// \end{pmatrix}
/// $$
///
/// The clock gate is generally given by the following formula:
///
/// $$
/// \begin{equation}
///     X = \sum_a e^{\frac{2a\pi i}{d}|a><a|
/// \end{equation}
/// $$
///
/// where d is the number of levels (2 levels is a qubit, 3 levels is
/// a qutrit, etc.)
///
/// References:
///     - https://arxiv.org/pdf/2302.07966.pdf
#[derive(Hash, PartialEq, Eq, Clone, Debug)]
pub struct ZGate {
    pub radix: usize,
}

impl ZGate {
    pub fn new(radix: usize) -> Self {
        Self { radix }
    }
}

impl ExpressionGenerator for ZGate {
    type ExpressionType = UnitaryExpression;

    fn generate_expression(&self) -> UnitaryExpression {
        let proto = format!("Z<{}>()", self.radix);
        let mut body = "[".to_string();
        for i in 0..self.radix {
            body += "[";
            for j in 0..self.radix {
                if i == j {
                    if i == 0 {
                        body += "1, ";
                    } else {
                        body += &format!("e^((2*{}*π*i)/{}), ", j, self.radix);
                    }
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

