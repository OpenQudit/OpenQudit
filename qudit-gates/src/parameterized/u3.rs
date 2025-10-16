use qudit_core::QuditRadices;
use qudit_expr::UnitaryExpression;
use qudit_expr::ExpressionGenerator;
use qudit_core::QuditSystem;

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
#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub struct U3Gate;

impl ExpressionGenerator for U3Gate {
    type ExpressionType = UnitaryExpression;

    fn generate_expression(&self) -> UnitaryExpression {
        let proto = "U3(θ0, θ1, θ2)";
        let body = "[
                [cos(θ0/2), ~e^(i*θ2)*sin(θ0/2)],
                [e^(i*θ1)*sin(θ0/2), e^(i*(θ1+θ2))*cos(θ0/2)]
        ]";
        UnitaryExpression::new(proto.to_owned() + "{" + &body + "}")
    }
}

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct ParameterizedUnitary {
    radices: QuditRadices,
}

impl ParameterizedUnitary {

    pub fn new(radices: QuditRadices) -> Self {
        ParameterizedUnitary {
            radices,
        }
    }

    fn generate_su2() -> UnitaryExpression {
        let proto = "U(θ0, θ1, θ2)";
        let body = "[
                [cos(θ0/2), ~e^(i*θ2)*sin(θ0/2)],
                [e^(i*θ1)*sin(θ0/2), e^(i*(θ1+θ2))*cos(θ0/2)]
        ]";
        UnitaryExpression::new(proto.to_owned() + "{" + &body + "}")
    }

    fn generate_embedded_two_param_su2(dimension: usize, i: usize, j: usize) -> UnitaryExpression {
        let proto = format!("U<{dimension}>(θ0, θ1)");
        let mut body = String::from("[");
        for row in 0..dimension {
            body += "[";
            for col in 0..dimension {
                if row == i && col == i {
                    body += "e^(i*θ1)*cos(θ0/2)";
                } else if row == i && col == j {
                    body += "~sin(θ0/2)";
                } else if row == j && col == i {
                    body += "sin(θ0/2)";
                } else if row == j && col == j {
                    body += "e^(i*(~θ1))*cos(θ0/2)";
                } else if row == col {
                    body += "1";
                } else {
                    body += "0";
                }
                body += ",";
            }
            body += "]";
        }
        body += "]";
        UnitaryExpression::new(proto.to_owned() + "{" + &body + "}")
    }

    fn generate_embedded_su2(dimension: usize, i: usize, j: usize) -> UnitaryExpression {
        let proto = format!("U<{dimension}>(θ0, θ1, θ2)");
        let mut body = String::from("[");
        for row in 0..dimension {
            body += "[";
            for col in 0..dimension {
                if row == i && col == i {
                    body += "cos(θ0/2)";
                } else if row == i && col == j {
                    body += "~e^(i*θ2)*sin(θ0/2)";
                } else if row == j && col == i {
                    body += "e^(i*θ1)*sin(θ0/2)";
                } else if row == j && col == j {
                    body += "e^(i*(θ1+θ2))*cos(θ0/2)";
                } else if row == col {
                    body += "1";
                } else {
                    body += "0";
                }
                body += ",";
            }
            body += "]";
        }
        body += "]";
        UnitaryExpression::new(proto.to_owned() + "{" + &body + "}")
    }

    fn embed_one_larger(unitary: UnitaryExpression) -> UnitaryExpression {
        let dimension = unitary.dimension() + 1;
        let mut one_larger = UnitaryExpression::identity(unitary.name(), [dimension]);
        one_larger.embed(unitary, 1, 1);
        one_larger
    }

    fn generate(dimension: usize) -> UnitaryExpression {
        if dimension < 2 {
            panic!("Cannot generate parameterized unitary for dimension 1 or less");
        }

        if dimension == 2 {
            return Self::generate_su2();
        }

        let right = {
            let one_smaller = Self::generate(dimension - 1);
            Self::embed_one_larger(one_smaller)
        };

        let left = {
            let mut acm = Self::generate_embedded_su2(dimension, dimension - 2, dimension - 1);
            for i in (0..=(dimension - 3)).rev() {
                let j = i + 1;
                let two = Self::generate_embedded_two_param_su2(dimension, i, j);
                acm = acm.dot(two)
            }
            acm
        };

        left.dot(right)
    }
}

impl ExpressionGenerator for ParameterizedUnitary {
    type ExpressionType = UnitaryExpression;

    fn generate_expression(&self) -> Self::ExpressionType {
        let mut expression = Self::generate(self.radices.dimension());
        expression.set_radices(self.radices.clone());
        expression
    }
}
