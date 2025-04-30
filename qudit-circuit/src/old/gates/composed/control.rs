use std::hash::Hash;
use std::ops::Range;

use faer_core::unzipped;
use faer_core::zipped;
use faer_core::MatMut;

use super::super::NamedGate;
use crate::math::matrix::MatGradMut;
use crate::math::matrix::MatHessMut;
use crate::math::unitary::DifferentiableUnitaryFn;
use crate::math::unitary::DoublyDifferentiableUnitaryFn;
use crate::math::unitary::UnitaryFn;
use crate::math::unitary::UnitaryGradient;
use crate::math::unitary::UnitaryMatrix;
use crate::math::BoundedFn;
use crate::math::ComplexScalar;
use crate::math::Function;
use crate::Gate;
use crate::QuditRadices;
use crate::QuditSystem;

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
#[derive(Clone, Debug)]
pub struct ControlledGate {
    /// The gate being controlled.
    gate: Box<Gate>,

    /// The indices of the final matrix that determine where to copy the
    /// gate's unitary/gradient/hessian.
    diagonal_copy_indices: Vec<usize>,

    /// The unitary dimension of the gate being controlled
    block_size: usize,

    // // The radices of the control qudits.
    // control_radices: QuditRadices,

    // // The levels of the control qudits that activate the gate.
    // control_levels: Vec<Vec<usize>>,

    // // The projection matrix for the states that activate the gate.
    // control_proj: Array2<usize>,

    // // The identity half of the final unitary equation.
    // ihalf: Array2<usize>,
    /// The radices of the entire controlled gate.
    radices: QuditRadices,
}

impl ControlledGate {
    /// Construct a ControlledGate.
    ///
    /// # Arguments
    ///
    /// * `gate` - The gate to control.
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
    pub fn new(
        gate: Gate,
        control_radices: QuditRadices,
        control_levels: Vec<Vec<usize>>,
    ) -> Self {
        if control_radices.len() != control_levels.len() {
            panic!(
                "control_radices and control_levels must have the same length"
            );
        }

        if control_levels.iter().any(|levels| levels.len() == 0) {
            panic!("control_levels must not contain empty levels");
        }

        if control_levels
            .iter()
            .zip(control_radices.iter())
            .any(|(levels, radix)| levels.iter().any(|level| *level >= *radix))
        {
            panic!(
                "Expected control levels to be less than the number of levels."
            );
        }

        // check that all levels in control_levels are unique
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

        let gate_dim = gate.get_dimension();

        let diagonal_copy_indices: Vec<usize> =
            ControlledGate::cartesian_product(control_levels)
                .into_iter()
                .map(|block_idx_expansion| {
                    control_radices.compress(
                        &block_idx_expansion, // .into_iter()
                                              // .rev() // TODO: I think this is wrong?
                                              // .collect::<Vec<usize>>(),
                    )
                })
                .map(|block_idx| block_idx * gate_dim)
                .collect();

        // // Identity is applied when controls are not properly activated.
        // let identity_gate: Array2 = Array2::eye(gate.get_dimension());

        // // Control projection matrix determines if the gate should activate.
        // let control_proj: Array2 =
        //     ControlledGate::build_control_projection(&control_radices,
        // &control_levels);

        // // Identity projection matrix determines if it shouldn't activate.
        // let identity_proj: Array2 = Array2::eye(control_proj.shape()[0]) -
        // control_proj.clone();

        // // Identity half of the final unitary equation.
        // let ihalf = identity_proj.kron(&identity_gate);

        ControlledGate {
            gate: Box::new(gate.clone()),
            diagonal_copy_indices: diagonal_copy_indices,
            block_size: gate_dim,
            // control_radices: control_radices.clone(),
            // control_levels: control_levels,
            // control_proj: control_proj,
            // ihalf: ihalf,
            radices: control_radices.concat(&gate.get_radices()),
        }
    }

    /// Calculates the cartesian product of the control levels.
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
                    for v in prod.iter_mut() {
                        v.push(l);
                        new_prod.push(v.clone());
                    }
                }
                prod = new_prod;
            }
        }
        prod
    }

    // fn build_control_projection(
    //     control_radices: &QuditRadices,
    //     control_levels: &Vec<Vec<usize>>,
    // ) -> Array2<C> {
    //     let elementary_projection_list = control_radices
    //         .iter()
    //         .map(|&radix| Array2::zeros((radix, radix)))
    //         .zip(control_levels.iter())
    //         .map(|(mut proj, levels)| {
    //             for level in levels.iter() {
    //                 proj[[*level, *level]] = T::one();
    //             }
    //             proj
    //         })
    //         .collect::<Vec<_>>();

    //     elementary_projection_list
    //         .into_iter()
    //         .fold(Array2::eye(1), |acc, proj| acc.kron(&proj))
    // }
}

impl NamedGate for ControlledGate {
    #[inline]
    fn get_name(&self) -> String {
        // TODO: better naming with control radices and control levels
        format!("Controlled({})", self.gate.get_name())
    }
}

impl Function for ControlledGate {
    #[inline]
    fn get_num_params(&self) -> usize {
        self.gate.get_num_params()
    }
}

impl BoundedFn for ControlledGate {
    #[inline]
    fn get_bounds(&self) -> Vec<Range<f64>> {
        self.gate.get_bounds()
    }
}

impl QuditSystem for ControlledGate {
    #[inline]
    fn get_radices(&self) -> QuditRadices {
        self.radices.clone()
    }

    #[inline]
    fn get_num_qudits(&self) -> usize {
        self.radices.get_num_qudits()
    }

    #[inline]
    fn get_dimension(&self) -> usize {
        self.radices.get_dimension()
    }
}

impl<C: ComplexScalar> UnitaryFn<C> for ControlledGate {
    #[inline]
    fn write_unitary(&self, params: &[C::Re], utry: &mut MatMut<C>) {
        // assign gate_unitary to every block in self.diagonal_copy_indices
        for &block_idx in self.diagonal_copy_indices.iter() {
            let mut out_unitary_block = utry.as_mut().submatrix_mut(
                block_idx,
                block_idx,
                self.block_size,
                self.block_size,
            );
            self.gate.write_unitary(params, &mut out_unitary_block);
        }
    }
}

impl<C: ComplexScalar> DifferentiableUnitaryFn<C> for ControlledGate {
    #[inline]
    fn write_gradient(&self, params: &[C::Re], grad: &mut MatGradMut<C>) {
        let gate_gradient: UnitaryGradient<C> = self.gate.get_gradient(params);
        for (grad_idx, d_m) in gate_gradient.iter().enumerate() {
            for &block_idx in self.diagonal_copy_indices.iter() {
                let out_block = grad.get_matmut(grad_idx).submatrix_mut(
                    block_idx,
                    block_idx,
                    self.block_size,
                    self.block_size,
                );
                zipped!(out_block, d_m.as_ref()).for_each(
                    |unzipped!(mut out, i)| {
                        out.write(i.read());
                    },
                );
            }
        }
    }

    #[inline]
    fn write_unitary_and_gradient(
        &self,
        params: &[C::Re],
        utry: &mut MatMut<C>,
        grad: &mut MatGradMut<C>,
    ) {
        let (gate_unitary, gate_gradient): (
            UnitaryMatrix<C>,
            UnitaryGradient<C>,
        ) = self.gate.get_unitary_and_gradient(params);

        for &block_idx in self.diagonal_copy_indices.iter() {
            for (grad_idx, d_m) in gate_gradient.iter().enumerate() {
                let out_block = grad.get_matmut(grad_idx).submatrix_mut(
                    block_idx,
                    block_idx,
                    self.block_size,
                    self.block_size,
                );
                zipped!(out_block, d_m.as_ref()).for_each(
                    |unzipped!(mut out, i)| {
                        out.write(i.read());
                    },
                );
            }
            let out_block = utry.as_mut().submatrix_mut(
                block_idx,
                block_idx,
                self.block_size,
                self.block_size,
            );
            zipped!(out_block, gate_unitary.as_ref()).for_each(
                |unzipped!(mut out, i)| {
                    out.write(i.read());
                },
            );
        }
    }
}

impl<C: ComplexScalar> DoublyDifferentiableUnitaryFn<C> for ControlledGate {
    #[inline]
    fn write_hessian(&self, params: &[C::Re], hess: &mut MatHessMut<C>) {
        let mut gate_hess = self.gate.get_hessian(params);
        let gate_hessian: MatHessMut<C> = gate_hess.partials.as_mut();
        for hess_idx_r in 0..gate_hessian.num_params() {
            for hess_idx_c in hess_idx_r..gate_hessian.num_params() {
                let in_block = gate_hessian.get_matref(hess_idx_r, hess_idx_c);
                for &block_idx in self.diagonal_copy_indices.iter() {
                    let out_block =
                        hess.get_matmut(hess_idx_r, hess_idx_c).submatrix_mut(
                            block_idx,
                            block_idx,
                            self.block_size,
                            self.block_size,
                        );
                    zipped!(out_block, in_block).for_each(
                        |unzipped!(mut out, i)| {
                            out.write(i.read());
                        },
                    );
                }
            }
        }
    }
}

impl Hash for ControlledGate {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.gate.hash(state);
        self.diagonal_copy_indices.hash(state);
        self.radices.hash(state);
    }
}

impl PartialEq for ControlledGate {
    fn eq(&self, other: &Self) -> bool {
        self.gate == other.gate
            && self.diagonal_copy_indices == other.diagonal_copy_indices
            && self.radices == other.radices
    }
}

impl Eq for ControlledGate {}

#[cfg(test)]
pub mod strategies {
    use proptest::prelude::*;
    use proptest::strategy::BoxedStrategy;
    use proptest::strategy::Strategy;

    use super::*;
    use crate::gates::strategies::ArbitraryGateWithRadices;
    use crate::QuditRadices;

    //new(Gate, control_radices: QuditRadices, control_levels: Vec<Vec<usize>>)

    impl Arbitrary for ControlledGate {
        type Parameters =
            (Option<Gate>, Option<QuditRadices>, Option<Vec<Vec<usize>>>);
        type Strategy = BoxedStrategy<Self>;

        fn arbitrary_with(args: Self::Parameters) -> Self::Strategy {
            let radix_strat = match args.1 {
                Some(radices) => Just(radices).boxed(),
                None => any_with::<QuditRadices>((2, 4, 2, 3)),
            };

            let radix_and_gate_strat = match args.0 {
                Some(gate) => radix_strat
                    .prop_flat_map(move |radices| {
                        (Just(radices.clone()), Just(gate.clone()))
                    })
                    .boxed(),
                None => radix_strat
                    .prop_flat_map(|radices| {
                        (
                            Just(radices.clone()),
                            1..=radices.get_num_qudits() - 1,
                        )
                    })
                    .prop_flat_map(|(radices, num_controls)| {
                        let control_radices = QuditRadices::new(
                            radices[..num_controls]
                                .iter()
                                .map(|&x| x)
                                .collect(),
                        );
                        let target_radices = QuditRadices::new(
                            radices[num_controls..]
                                .iter()
                                .map(|&x| x)
                                .collect(),
                        );
                        (
                            Just(control_radices),
                            Gate::arbitrary_with_radices_no_rec(target_radices)
                                .unwrap(),
                        )
                    })
                    .boxed(),
            };

            let radix_gate_and_level_strat = match args.2 {
                Some(levels) => radix_and_gate_strat
                    .prop_flat_map(move |(radices, gate)| {
                        (
                            Just(radices.clone()),
                            Just(gate.clone()),
                            Just(levels.clone()),
                        )
                    })
                    .boxed(),
                None => radix_and_gate_strat
                    .prop_flat_map(|(radices, gate)| {
                        let num_controls = radices.get_num_qudits();
                        let mut levels = vec![];
                        for i in 0..num_controls {
                            levels
                                .push(prop::collection::vec(0..radices[i], 1));
                        }
                        (Just(radices.clone()), Just(gate), levels)
                    })
                    .boxed(),
            };

            // let levels_strat = match args.2 {
            //     Some(levels) => Just(levels).boxed(),
            //     None => {
            //         radix_strat
            //             .clone()  // TODO: I think this is wrong, I need to
            // flat_map radices into levels
            // .prop_perturb(|radices, mut rng| {
            // let mut levels = vec![];                 for radix in
            // radices.iter() {                     let
            // num_ctrl_levels = rng.gen_range(0..*radix);
            //                     let mut level = vec![];

            //                     // generate num_ctrl_levels unique values
            // from the range [0, radix)                     while
            // level.len() < num_ctrl_levels {
            // let new_level = rng.gen_range(0..*radix);
            // if !level.contains(&new_level) {
            // level.push(new_level);                         }
            //                     }

            //                     levels.push(level);
            //                 }
            //                 levels
            //             })
            //             .boxed()
            //     }
            // };

            radix_gate_and_level_strat
                .prop_map(|(radices, gate, levels)| {
                    ControlledGate::new(gate, radices, levels)
                })
                .boxed()
        }
    }

    impl ArbitraryGateWithRadices for ControlledGate {
        fn arbitrary_with_radices(
            radices: QuditRadices,
        ) -> Option<BoxedStrategy<Gate>> {
            if radices.get_num_qudits() == 1 {
                None
            } else {
                Some(
                    ControlledGate::arbitrary_with((None, Some(radices), None))
                        .prop_map(|x| Gate::from(x))
                        .boxed(),
                )
            }
        }

        fn arbitrary_with_radices_no_rec(
            _radices: QuditRadices,
        ) -> Option<BoxedStrategy<Gate>> {
            None
        }
    }
}

// #[cfg(test)]
// mod test {
//     use super::*;

//     #[test]
//     fn test_qubit_h_gate() {
//         let h_gate = HGate::new(2);
//         let unitary = h_gate.get_unitary(&[]);
//         let correct = array![
//             [
//                 c64::new(1.0 / 2.0_f64.sqrt(), 0.0),
//                 c64::new(1.0 / 2.0_f64.sqrt(), 0.0)
//             ],
//             [
//                 c64::new(1.0 / 2.0_f64.sqrt(), 0.0),
//                 c64::new(-1.0 / 2.0_f64.sqrt(), 0.0)
//             ]
//         ];
//         unitary.assert_close_to(&correct);
//     }
// }
