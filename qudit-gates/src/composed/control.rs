use qudit_core::{HasParams, QuditRadices, QuditSystem, ToRadix};
use qudit_expr::{UnitaryExpression, UnitaryExpressionGenerator};


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
#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub struct ControlledGate {
    expr: UnitaryExpression,
}

impl ControlledGate {
    /// Construct a ControlledGate.
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
    ///
    /// # Returns
    ///
    /// A new ControlledGate.
    ///
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
    ///
    /// # Examples
    ///
    /// // TODO: Come back to later
    pub fn new<E: UnitaryExpressionGenerator>(
        expr: E,
        control_radices: QuditRadices, // TODO: Make a ToRadices generic
        control_levels: Vec<Vec<usize>>,
    ) -> Self {
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

        let gate_expr = expr.gen_expr();
        let gate_dim = gate_expr.dimension();

        // Build appropriately sized identity expression
        let name = format!("Controlled({})", gate_expr.name());
        let radices = control_radices.concat(&gate_expr.radices());
        let mut expr = UnitaryExpression::identity(&name, radices);

        // Embed gate expression into identity expression at correct spots
        let diagonal_indices: Vec<usize> =
            ControlledGate::cartesian_product(control_levels)
                .into_iter()
                .map(|block_idx_expansion| {
                    control_radices.compress(&block_idx_expansion)
                })
                .map(|block_diag_idx| block_diag_idx * gate_dim)
                .collect();

        for diagonal_idx in diagonal_indices.iter() {
            expr.embed(gate_expr.clone(), *diagonal_idx, *diagonal_idx);
        }

        ControlledGate { expr }
    }

    /// Calculates the cartesian product of the control levels.
    ///
    /// # Examples
    ///
    /// ```
    /// use qudit_gates::ControlledGate;
    /// let control_levels = vec![vec![0, 1], vec![0, 1]];
    /// let prod = ControlledGate::cartesian_product(control_levels);
    /// assert_eq!(prod, vec![
    ///    vec![0, 0],
    ///    vec![1, 0],
    ///    vec![0, 1],
    ///    vec![1, 1],
    /// ]);
    /// ```
    pub fn cartesian_product(control_levels: Vec<Vec<usize>>) -> Vec<Vec<usize>> {
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
}

impl HasParams for ControlledGate {
    #[inline]
    fn num_params(&self) -> usize {
        self.expr.num_params()
    }
}

impl QuditSystem for ControlledGate {
    #[inline]
    fn radices(&self) -> QuditRadices {
        self.expr.radices()
    }

    #[inline]
    fn num_qudits(&self) -> usize {
        self.expr.num_qudits()
    }

    #[inline]
    fn dimension(&self) -> usize {
        self.expr.dimension()
    }
}

impl UnitaryExpressionGenerator for ControlledGate {
    #[inline]
    fn gen_expr(&self) -> UnitaryExpression {
        self.expr.clone()
    }
}
