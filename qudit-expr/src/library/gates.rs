use crate::UnitaryExpression;
use crate::UnitarySystemExpression;
use qudit_core::QuditSystem;
use qudit_core::QuditRadices;
use qudit_core::ToRadix;

/// The identity or no-op gate.
#[cfg_attr(feature = "python", pyo3::pyfunction)]
#[cfg_attr(feature = "python", pyo3(signature = (radix = 2)))]
pub fn IGate(radix: usize) -> UnitaryExpression {
    let proto = format!("I<{}>()", radix);
    let mut body = "".to_string();
    body += "[";
    for i in 0..radix {
        body += "[";
        for j in 0..radix {
            if i == j {
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
#[cfg_attr(feature = "python", pyo3::pyfunction)]
#[cfg_attr(feature = "python", pyo3(signature = (radix = 2)))]
pub fn HGate(radix: usize) -> UnitaryExpression {
    let proto = format!("H<{}>()", radix);
    let mut body = "".to_string();
    if radix == 2 {
        body += "[[1/sqrt(2), 1/sqrt(2)], [1/sqrt(2), ~1/sqrt(2)]]";
        return UnitaryExpression::new(proto + "{" + &body + "}");
    }
    let omega = format!("e^(2*π*i/{})", radix);
    let invsqrt = format!("1/sqrt({})", radix);
    body += invsqrt.as_str();
    body += " * ";
    body += "[";
    for i in 0..radix {
        body += "[";
        for j in 0..radix {
            body += &format!("{}^({}*{}), ", omega, i, j);
        }
        body += "],";
    }
    body += "]";

    UnitaryExpression::new(proto + "{" + &body + "}")
}

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
#[cfg_attr(feature = "python", pyo3::pyfunction)]
#[cfg_attr(feature = "python", pyo3(signature = (radix = 2)))]
pub fn SwapGate(radix: usize) -> UnitaryExpression {
    let proto = format!("Swap<{}, {}>()", radix, radix);
    let mut body = "".to_string();
    body += "[";
    for i in 0..radix {
        body += "[";
        let a_i = i / radix;
        let b_i = i % radix;
        for j in 0..radix {
            let a_j = j / radix;
            let b_j = j % radix;
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
#[cfg_attr(feature = "python", pyo3::pyfunction)]
#[cfg_attr(feature = "python", pyo3(signature = (radix = 2)))]
pub fn XGate(radix: usize) -> UnitaryExpression {
    let proto = format!("X<{}>()", radix);
    let mut body = "[".to_string();
    for i in 0..radix {
        body += "[";
        for j in 0..radix {
            if (j + 1) % radix == i {
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
#[cfg_attr(feature = "python", pyo3::pyfunction)]
#[cfg_attr(feature = "python", pyo3(signature = (radix = 2)))]
pub fn ZGate(radix: usize) -> UnitaryExpression {
    let proto = format!("Z<{}>()", radix);
    let mut body = "[".to_string();
    for i in 0..radix {
        body += "[";
        for j in 0..radix {
            if i == j {
                if i == 0 {
                    body += "1, ";
                } else {
                    body += &format!("e^((2*{}*π*i)/{}), ", j, radix);
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
#[cfg_attr(feature = "python", pyo3::pyfunction)]
#[cfg_attr(feature = "python", pyo3(signature = (radix = 2)))]
pub fn PGate(radix: usize) -> UnitaryExpression {
    let mut proto = format!("P<{}>(", radix);
    for i in 0..radix - 1 {
        proto += "θ";
        proto += &i.to_string();
        proto += ", ";
    }
    proto += ")";
    
    let mut body = "".to_string();
    body += "[";
    for i in 0..radix {
        body += "[";
        for j in 0..radix {
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

#[cfg_attr(feature = "python", pyo3::pyfunction)]
pub fn U3Gate() -> UnitaryExpression {
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

/// Generates a fully parameterized unitary expression
///
/// References:
/// - de Guise, Hubert, Olivia Di Matteo, and Luis L. Sánchez-Soto.
///     "Simple factorization of unitary transformations."
///     Physical Review A 97.2 (2018): 022328.
#[cfg_attr(feature = "python", pyo3::pyfunction)]
#[cfg_attr(feature = "python", pyo3(signature = (radices = QuditRadices::new(&[2]))))]
pub fn ParameterizedUnitary(radices: QuditRadices) -> UnitaryExpression {
    let dimension = radices.dimension();

    if dimension < 2 {
        panic!("Cannot generate parameterized unitary for dimension 1 or less");
    }

    if dimension == 2 {
        return U3Gate();
    }

    let right = {
        let one_smaller = ParameterizedUnitary(QuditRadices::new(&[dimension - 1]));
        embed_one_larger(one_smaller)
    };

    let left = {
        let mut acm = generate_embedded_su2(dimension, dimension - 2, dimension - 1);
        for i in (0..=(dimension - 3)).rev() {
            let j = i + 1;
            let two = generate_embedded_two_param_su2(dimension, i, j);
            acm = acm.dot(two)
        }
        acm
    };

    let mut expression = left.dot(right);
    expression.set_radices(radices);
    expression
}

/// Invert an expression
#[cfg_attr(feature = "python", pyo3::pyfunction)]
pub fn Invert(mut expr: UnitaryExpression) -> UnitaryExpression {
    expr.dagger();
    expr
}

/// Calculates the cartesian product of the control levels.
///
/// # Examples
///
/// ```
/// let control_levels = vec![vec![0, 1], vec![0, 1]];
/// let prod = cartesian_product(control_levels);
/// assert_eq!(prod, vec![
///    vec![0, 0],
///    vec![1, 0],
///    vec![0, 1],
///    vec![1, 1],
/// ]);
/// ```
fn cartesian_product(control_levels: Vec<Vec<usize>>) -> Vec<Vec<usize>> {
    let mut prod = vec![];
    for level in control_levels.into_iter() {
        if prod.len() == 0 {
            for l in level.into_iter() {
                prod.push(vec![l]);
            }
        } else {
            let mut new_prod = vec![];
            for l in level.into_iter() {
                for v in prod.iter() {
                    let mut v_new = v.clone();
                    v_new.push(l);
                    new_prod.push(v_new);
                }
            }
            prod = new_prod;
        }
    }
    prod
}


/// An arbitrary controlled gate.
///
/// Given any gate, ControlledGate can add control qudits.
///
/// A controlled gate adds arbitrarily controls, and is generalized
/// for qudit or even mixed-qudit representation.
///
/// A controlled gate has a circuit structure as follows:
///
/// ```text
///     controls ----/----■----
///                       |
///                      .-.
///     targets  ----/---|G|---
///                      '-'
/// ```
///
/// Where $G$ is the gate being controlled.
///
/// To calculate the unitary for a controlled gate, given the unitary of
/// the gate being controlled, we can use the following equation:
///
/// $$U_{control} = P_i \otimes I + P_c \otimes G$$
///
/// Where $P_i$ is the projection matrix for the states that don't
/// activate the gate, $P_c$ is the projection matrix for the
/// states that do activate the gate, $I$ is the identity matrix
/// of dimension equal to the gate being controlled, and $G$ is
/// the unitary matrix of the gate being controlled.
///
/// In the simple case of a normal qubit CNOT ($G = X$), $P_i$ and $P_c$
/// are defined as follows:
///
/// $$
///     P_i = \ket{0}\bra{0}
///     P_c = \ket{1}\bra{1}
/// $$
///
/// This is because the $\ket{0}$ state is the state that doesn't
/// activate the gate, and the $\ket{1}$ state is the state that
/// does activate the gate.
///
/// We can also decide to invert this, and have the $\ket{0}$
/// state activate the gate, and the $\ket{1}$ state not activate
/// the gate. This is equivalent to swapping $P_i$ and $P_c$,
/// and usually drawn diagrammatically as follows:
///
/// ```text
///     controls ----/----□----
///                       |
///                      .-.
///     targets  ----/---|G|---
///                      '-'
/// ```
///
/// When we add more controls the projection matrices become more complex,
/// but the basic idea stays the same: we have a projection matrix for
/// the states that activate the gate, and a projection matrix for the
/// states that don't activate the gate. As in the case of a toffoli gate,
/// the projection matrices are defined as follows:
///
/// $$
///     P_i = \ket{00}\bra{00} + \ket{01}\bra{01} + \ket{10}\bra{10}
///     P_c = \ket{11}\bra{11}
/// $$
///
/// This is because the $\ket{00}$, $\ket{01}$, and
/// $\ket{10}$ states are the states that don't activate the
/// gate, and the $\ket{11}$ state is the state that does
/// activate the gate.
///
/// With qudits, we have more states and as such, more complex
/// projection matrices; however, the basic idea is the same.
/// For example, a qutrit controlled-not gate that is activated by
/// the $\ket{2}$ state and not activated by the $\ket{0}$
/// and $\ket{1}$ states is defined as follows:
///
/// $$
///     P_i = \ket{0}\bra{0} + \ket{1}\bra{1}
///     P_c = \ket{2}\bra{2}
/// $$
///
/// One interesting concept with qudits is that we can have multiple
/// active control levels. For example, a qutrit controlled-not gate that
/// is activated by the $\ket{1}$ and $\ket{2}$ states
/// and not activated by the $\ket{0}$ state is defined similarly
/// as follows:
///
/// $$
///     P_i = \ket{0}\bra{0}
///     P_c = \ket{1}\bra{1} + \ket{2}\bra{2}
/// $$
///
/// Note that we can always define $P_i$ simply from $P_c$:
///
/// $$P_i = I_p - P_c$$
///
/// Where $I_p$ is the identity matrix of dimension equal to the
/// dimension of the control qudits. This leaves us with out final
/// equation:
///
///
/// $$U_{control} = (I_p - P_c) \otimes I + P_c \otimes G$$
///
/// If, G is a unitary-valued function of real parameters, then the
/// gradient of the controlled gate simply discards the constant half
/// of the equation:
///
/// $$
///     \frac{\partial U_{control}}{\partial \theta} =
///         P_c \otimes \frac{\partial G}{\partial \theta}
/// $$
///
/// # Arguments
///
/// * `expr` - The gate to control.
///
/// * `control_radixes` - The number of levels for each control qudit.
///
/// * `control_levels` - The levels of the control qudits that activate the
///   gate. If more than one level is selected, the subspace spanned by the
///   levels acts as a control subspace. If all levels are selected for a
///   given qudit, the operation is equivalent to the original gate without
///   controls.
/// # Panics
///
/// * If `control_radixes` and `control_levels` have different lengths.
///
/// * If `control_levels` contains an empty level.
///
/// * If any level in `control_levels` is greater than or equal to the
///   corresponding radix in `control_radixes`.
///
/// * If any level in `control_levels` is not unique.
#[cfg_attr(feature = "python", pyo3::pyfunction)]
#[cfg_attr(feature = "python", pyo3(signature = (expr, control_radices = QuditRadices::new(&[2]), control_levels = None)))]
pub fn Controlled(
    expr: UnitaryExpression,
    control_radices: QuditRadices,
    control_levels: Option<Vec<Vec<usize>>>,
) -> UnitaryExpression {
    let control_levels = match control_levels {
        Some(levels) => levels,
        None => {
            // Generate default control_levels: each control qudit activated by its highest level
            control_radices.iter().map(|&radix| vec![(radix - 1) as usize]).collect()
        }
    };
    
    if control_radices.len() != control_levels.len() {
        panic!("control_radices and control_levels must have the same length");
    }

    if control_levels.iter().any(|levels| levels.len() == 0) {
        panic!("control_levels must not contain empty levels");
    }

    if control_levels
        .iter()
        .map(|levels| levels.into_iter().map(|level| level.to_radix()))
        .zip(control_radices.iter())
        .any(|(mut levels, radix)| levels.any(|level| level >= *radix))
    {
        panic!("Expected control levels to be less than the number of levels.");
    }

    // Check that all levels in control_levels are unique
    let mut control_level_sets = control_levels.clone();
    for level in control_level_sets.iter_mut() {
        level.sort();
        level.dedup();
    }
    if control_level_sets
        .iter()
        .zip(control_levels.iter())
        .any(|(level_dedup, level)| level.len() != level_dedup.len())
    {
        panic!("Expected control levels to be unique.");
    }

    let gate_expr = expr;
    let gate_dim = gate_expr.dimension();

    // Build appropriately sized identity expression
    let name = format!("Controlled({})", gate_expr.name());
    let radices = control_radices.concat(&gate_expr.radices());
    let mut expr = UnitaryExpression::identity(&name, radices);

    // Embed gate expression into identity expression at correct spots
    let diagonal_indices: Vec<usize> = cartesian_product(control_levels)
            .into_iter()
            .map(|block_idx_expansion| {
                control_radices.compress(&block_idx_expansion)
            })
            .map(|block_diag_idx| block_diag_idx * gate_dim)
            .collect();

    for diagonal_idx in diagonal_indices.iter() {
        expr.embed(gate_expr.clone(), *diagonal_idx, *diagonal_idx);
    }

    expr
}
#[cfg_attr(feature = "python", pyo3::pyfunction)]
#[cfg_attr(feature = "python", pyo3(signature = (expr, control_radices = QuditRadices::new(&[2]), control_levels = None)))]
pub fn ClassicallyControlled(
    expr: UnitaryExpression,
    control_radices: QuditRadices,
    control_levels: Option<Vec<Vec<usize>>>,
) -> UnitarySystemExpression {
    let control_levels = match control_levels {
        Some(levels) => levels,
        None => {
            // Generate default control_levels: each control qudit activated by its highest level
            control_radices.iter().map(|&radix| vec![(radix - 1) as usize]).collect()
        }
    };
    
    if control_radices.len() != control_levels.len() {
        panic!("control_radices and control_levels must have the same length");
    }

    if control_levels.iter().any(|levels| levels.len() == 0) {
        panic!("control_levels must not contain empty levels");
    }

    if control_levels
        .iter()
        .map(|levels| levels.into_iter().map(|level| level.to_radix()))
        .zip(control_radices.iter())
        .any(|(mut levels, radix)| levels.any(|level| level >= *radix))
    {
        panic!("Expected control levels to be less than the number of levels.");
    }

    // Check that all levels in control_levels are unique
    let mut control_level_sets = control_levels.clone();
    for level in control_level_sets.iter_mut() {
        level.sort();
        level.dedup();
    }
    if control_level_sets
        .iter()
        .zip(control_levels.iter())
        .any(|(level_dedup, level)| level.len() != level_dedup.len())
    {
        panic!("Expected control levels to be unique.");
    }

    let diagonal_indices: Vec<usize> = cartesian_product(control_levels)
            .into_iter()
            .map(|block_idx_expansion| control_radices.compress(&block_idx_expansion))
            .collect();

    let new_radices = control_radices.into_iter().map(|&r| r as usize).collect::<Vec<_>>();
    expr.classically_control(&diagonal_indices, &new_radices)
}

#[cfg(feature = "python")]
mod python {
    use pyo3::prelude::*;
    use crate::python::PyExpressionRegistrar;

    /// Registers the gate library with the Python module.
    fn register(parent_module: &Bound<'_, PyModule>) -> PyResult<()> {
        parent_module.add_function(wrap_pyfunction!(super::IGate, parent_module)?)?;
        parent_module.add_function(wrap_pyfunction!(super::HGate, parent_module)?)?;
        parent_module.add_function(wrap_pyfunction!(super::SwapGate, parent_module)?)?;
        parent_module.add_function(wrap_pyfunction!(super::XGate, parent_module)?)?;
        parent_module.add_function(wrap_pyfunction!(super::ZGate, parent_module)?)?;
        parent_module.add_function(wrap_pyfunction!(super::PGate, parent_module)?)?;
        parent_module.add_function(wrap_pyfunction!(super::U3Gate, parent_module)?)?;
        parent_module.add_function(wrap_pyfunction!(super::ParameterizedUnitary, parent_module)?)?;
        parent_module.add_function(wrap_pyfunction!(super::Invert, parent_module)?)?;
        parent_module.add_function(wrap_pyfunction!(super::Controlled, parent_module)?)?;
        parent_module.add_function(wrap_pyfunction!(super::ClassicallyControlled, parent_module)?)?;
        Ok(())
    }
    inventory::submit!(PyExpressionRegistrar { func: register });
}
