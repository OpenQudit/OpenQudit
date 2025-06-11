use qudit_expr::{UnitaryExpression, UnitaryExpressionGenerator};

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
#[derive(Hash, PartialEq, Eq, Clone, Debug)]
pub struct HGate {
    pub radix: usize,
}

impl HGate {
    pub fn new(radix: usize) -> Self {
        Self { radix }
    }
}

impl UnitaryExpressionGenerator for HGate {
    fn gen_expr(&self) -> UnitaryExpression {
        let proto = format!("utry H<{}>()", self.radix);
        let mut body = "".to_string();
        if self.radix == 2 {
            body += "[[1/sqrt(2), 1/sqrt(2)], [1/sqrt(2), ~1/sqrt(2)]]";
            return UnitaryExpression::new(proto + "{" + &body + "}");
        }
        let omega = format!("e^(2*π*i/{})", self.radix);
        let invsqrt = format!("1/sqrt({})", self.radix);
        body += invsqrt.as_str();
        body += " * ";
        body += "[";
        for i in 0..self.radix {
            body += "[";
            for j in 0..self.radix {
                body += &format!("{}^({}*{}), ", omega, i, j);
            }
            body += "],";
        }
        body += "]";

        UnitaryExpression::new(proto + "{" + &body + "}")
    }
}
