use qudit_expr::TensorExpression;
use qudit_expr::TensorExpressionGenerator;

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

impl TensorExpressionGenerator for U3Gate {
    fn gen_expr(&self) -> TensorExpression {
        let proto = "utry U3(θ0, θ1, θ2)";
        let body = "[
                [cos(θ0/2), ~e^(i*θ2)*sin(θ0/2)],
                [e^(i*θ1)*sin(θ0/2), e^(i*(θ1+θ2))*cos(θ0/2)]
        ]";
        TensorExpression::new(proto.to_owned() + "{" + &body + "}")
    }
}
