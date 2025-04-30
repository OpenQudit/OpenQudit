use qudit_expr::UnitaryExpression;
use qudit_expr::UnitaryExpressionGenerator;

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
#[derive(Hash, PartialEq, Eq, Clone, Debug)]
pub struct PGate {
    pub radix: usize,
}

impl PGate {
    pub fn new(radix: usize) -> Self {
        Self { radix }
    }
}

impl UnitaryExpressionGenerator for PGate {
    fn gen_expr(&self) -> UnitaryExpression {
        let mut proto = format!("utry P<{}>(", self.radix);
        for i in 0..self.radix - 1 {
            proto += "θ";
            proto += &i.to_string();
            proto += ", ";
        }
        proto += ")";
        
        let mut body = "".to_string();
        body += "[";
        for i in 0..self.radix {
            body += "[";
            for j in 0..self.radix {
                if i == j {
                    if i == 0 {
                        body += "1, ";
                    } else {
                        body += &format!("e^(i*θ{}), ", i - 1);
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

