use std::collections::HashSet;
use std::ops::Deref;

use faer_core::c64;
use faer_core::permutation::permute_cols;
use faer_core::permutation::permute_rows;
use faer_core::permutation::Permutation;
use faer_core::AsMatMut;
use faer_core::AsMatRef;
use faer_core::Mat;
use faer_entity::ComplexField;
use faer_entity::Entity;

use super::fused_reshape_permute_reshape_into;
use crate::circuit::CircuitLocation;
use crate::QuditRadices;
use crate::QuditSystem;

/// A permutation of qudits.
///
/// Qudit systems are often shuffled around in compiler pipelines. This
/// object captures a specific shuffling operation which can be represented
/// in many ways.
#[derive(Hash, PartialEq, Eq, Clone, Debug)]
pub struct QuditPermutation {
    /// The number of qudits in the permutation.
    num_qudits: usize,

    /// The radices of the qudit system being permuted.
    radices: QuditRadices,

    /// The permutation vector in the qudit space.
    perm: Vec<usize>,

    /// The permutation vector in the index space.
    index_perm: Vec<usize>,

    /// The inverse permutation vector in the index space.
    inverse_index_perm: Vec<usize>,
}

impl QuditPermutation {
    /// Returns a permutation mapping qudit `i` to `perm[i]`.
    ///
    /// # Arguments
    ///
    /// * `radices` - The radices of the system being permuted
    /// * `perm` - A vector describing the permutation. The resulting operation
    ///   will shift the i-th qudit to the `perm[i]`-th qudit.
    /// # Panics
    ///
    /// * If the supplied permutation is not a proper permutation. This can
    ///   happen when there is a duplicate or invalid qudit index.
    ///
    /// * If the permutation and radices describe systems with different qudit
    ///   counts, i.e. they have different lengths.
    ///
    /// # Examples
    ///
    /// ```
    /// use qudit_circuit::{QuditRadices, QuditPermutation};
    ///
    /// let three_qubits = QuditRadices::new(vec![2; 3]);
    /// let qudit_shift = QuditPermutation::new(three_qubits, vec![2, 0, 1]);
    /// ```
    pub fn new(radices: QuditRadices, perm: Vec<usize>) -> QuditPermutation {
        if perm.len() != radices.len() {
            panic!("Invalid qudit permutation: perm's qudit count doesn't match radices'.");
        }

        if !perm.iter().all(|x| x < &radices.len()) {
            panic!("Invalid qudit permutation: permutation has invalid qudit index.");
        }

        let mut uniq = HashSet::new();
        if !perm.iter().all(|x| uniq.insert(x)) {
            panic!(
                "Invalid qudit permutation: permutation has duplicate entries."
            );
        }

        // OLD Ndarray reference implementation:
        // let index_perm = Array::range(0., radices.get_dimension() as f32, 1.)
        //     .mapv(|x| x as usize)
        //     .into_shape(radices.as_slice())
        //     .unwrap()
        //     .permuted_axes(perm.as_slice())
        //     .to_shape(radices.get_dimension())
        //     .unwrap()
        //     .to_vec();

        // TODO: Clean up
        let vec = Mat::from_fn(radices.get_dimension(), 1, |i, _| {
            c64::new(i as f64, 0.0)
        });
        let mut vec_buf = Mat::zeros(radices.get_dimension(), 1);
        let mut extended_radices = radices.clone().to_vec();
        let mut extended_perm = perm.clone().to_vec();
        extended_radices.push(1);
        extended_perm.push(perm.len());
        fused_reshape_permute_reshape_into(
            vec.as_ref(),
            &extended_radices,
            &extended_perm,
            vec_buf.as_mut(),
        );

        let mut index_perm = vec![];
        for i in 0..radices.get_dimension() {
            index_perm.push(vec_buf.read(i, 0).re as usize);
        }

        let mut inverse_index_perm = vec![0; radices.get_dimension()];
        index_perm
            .iter()
            .enumerate()
            .for_each(|(s, d)| inverse_index_perm[*d] = s);

        QuditPermutation {
            num_qudits: radices.len(),
            radices: radices,
            perm: perm,
            index_perm: index_perm,
            inverse_index_perm: inverse_index_perm,
        }
    }

    /// True if the permutation has physical meaning. A permutation cannot
    /// be physically meaningful if it permutes qudits of different radices.
    ///
    /// # Examples
    /// ```
    /// use qudit_circuit::{QuditRadices, QuditPermutation};
    ///
    /// let three_qubits = QuditRadices::new(vec![2; 3]);
    /// let qudit_shift = QuditPermutation::new(three_qubits, vec![2, 0, 1]);
    /// assert!(qudit_shift.has_physical_meaning());
    ///
    /// let three_qutrits = QuditRadices::new(vec![3; 3]);
    /// let qudit_shift = QuditPermutation::new(three_qutrits, vec![2, 0, 1]);
    /// assert!(qudit_shift.has_physical_meaning());
    ///
    /// let three_qudits = QuditRadices::new(vec![2, 3, 4]);
    /// let qudit_shift = QuditPermutation::new(three_qudits, vec![2, 0, 1]);
    /// assert!(!qudit_shift.has_physical_meaning());
    /// ```
    pub fn has_physical_meaning(&self) -> bool {
        self.perm
            .iter()
            .enumerate()
            .all(|x| self.radices[x.0] == self.radices[*x.1])
    }

    /// Returns a qubit permutation specifed by `perm`.
    ///
    /// # Arguments
    ///
    /// * `perm` - The qubit permutation.
    /// # Panics
    ///
    /// * If the supplied permutation is not a proper permutation. This can
    ///   happen when there is a duplicate or invalid qudit index.
    ///
    /// # Examples
    ///
    /// ```
    /// use qudit_circuit::{QuditRadices, QuditPermutation};
    ///
    /// let qubit_shift = QuditPermutation::from_qubit_location(vec![2, 0, 1]);
    /// ```
    pub fn from_qubit_location(perm: Vec<usize>) -> QuditPermutation {
        let rdx = QuditRadices::new(vec![2; perm.len()]);
        QuditPermutation::new(rdx, perm)
    }

    /// Returns a qubit permutation specifed by `perm`.
    ///
    /// # Arguments
    ///
    /// * `perm` - The qubit permutation.
    /// # Panics
    ///
    /// * If the supplied permutation is not a proper permutation. This can
    ///   happen when there is a duplicate or invalid qudit index.
    ///
    /// # Examples
    ///
    /// ```
    /// use qudit_circuit::{QuditRadices, QuditPermutation};
    ///
    /// let qutrit_swap = QuditPermutation::from_qudit_location(3, vec![1, 0]);
    /// ```
    pub fn from_qudit_location(
        radix: usize,
        perm: Vec<usize>,
    ) -> QuditPermutation {
        if radix < 2 {
            panic!("Invalid radix, must be greater than 2, got {}.", radix);
        }

        let rdx = QuditRadices::new(vec![radix; perm.len()]);
        QuditPermutation::new(rdx, perm)
    }

    /// Returns a permutation that sorts the circuit location
    pub fn locally_invert_location(
        radices: QuditRadices,
        loc: &CircuitLocation,
    ) -> QuditPermutation {
        let mut perm: Vec<usize> = (0..loc.len()).collect();
        perm.sort_by_key(|&i| &loc.qudits()[i]);
        QuditPermutation::new(radices, perm)
    }

    /// Returns the permutation in the index space.
    ///
    /// # Examples
    /// ```
    /// use qudit_circuit::{QuditRadices, QuditPermutation};
    ///
    /// let qubit_swap = QuditPermutation::from_qubit_location(vec![1, 0]);
    /// assert_eq!(qubit_swap.get_index_perm(), &vec![0, 2, 1, 3]);
    /// ```
    pub fn get_index_perm(&self) -> &Vec<usize> {
        return &self.index_perm;
    }

    /// Returns the number of qudits being permuted.
    pub fn get_num_qudits(&self) -> usize {
        self.num_qudits
    }

    /// Returns the dimension of the system being permuted.
    pub fn get_dimension(&self) -> usize {
        self.index_perm.len()
    }

    /// Returns the radices of the system after being permuted.
    ///
    /// # Examples
    /// ```
    /// use qudit_circuit::{QuditRadices, QuditPermutation};
    /// let qudit_swap = QuditPermutation::new(QuditRadices::new(vec![2, 3]), vec![1, 0]);
    /// assert_eq!(qudit_swap.get_permuted_radices(), QuditRadices::new(vec![3, 2]));
    /// ```
    ///
    /// ```
    /// use qudit_circuit::{QuditRadices, QuditPermutation};
    /// let qudit_swap = QuditPermutation::new(QuditRadices::new(vec![2, 3, 4]), vec![1, 0, 2]);
    /// assert_eq!(qudit_swap.get_permuted_radices(), QuditRadices::new(vec![3, 2, 4]));
    /// ```
    pub fn get_permuted_radices(&self) -> QuditRadices {
        QuditRadices::new(self.perm.iter().map(|&i| self.radices[i]).collect())
    }

    /// Returns true is this permutation does not permute any qudit.
    ///
    /// # Examples
    /// ```
    /// use qudit_circuit::math::QuditPermutation;
    /// let identity = QuditPermutation::from_qubit_location(vec![0, 1]);
    /// assert!(identity.is_identity());
    /// ```
    ///
    /// ```
    /// use qudit_circuit::math::QuditPermutation;
    /// let identity = QuditPermutation::from_qubit_location(vec![1, 0]);
    /// assert!(!identity.is_identity());
    /// ```
    pub fn is_identity(&self) -> bool {
        self.perm.iter().enumerate().all(|(s, d)| s == *d)
    }

    /// Returns a new permutation that composes `self` with `arg_perm`.
    ///
    /// # Panics
    ///
    /// If the permutations have incompatible radices.
    ///
    /// # Examples
    ///
    /// ```
    /// use qudit_circuit::math::QuditPermutation;
    /// let p1 = QuditPermutation::from_qubit_location(vec![1, 0]);
    /// let p2 = QuditPermutation::from_qubit_location(vec![1, 0]);
    /// assert!(p1.compose(&p2).is_identity());
    /// ```
    ///
    /// ```
    /// use qudit_circuit::math::QuditPermutation;
    /// let p1 = QuditPermutation::from_qubit_location(vec![0, 2, 1, 3]);
    /// let p2 = QuditPermutation::from_qubit_location(vec![1, 0, 3, 2]);
    /// assert_eq!(p1.compose(&p2), QuditPermutation::from_qubit_location(vec![2, 0, 3, 1]));
    /// ```
    pub fn compose(&self, arg_perm: &QuditPermutation) -> QuditPermutation {
        if self.radices != arg_perm.radices {
            panic!("Permutations being composed have incompatible radices.");
        }

        let mut composed_perm = vec![0; self.get_num_qudits()];
        arg_perm
            .iter()
            .enumerate()
            .for_each(|(s, d)| composed_perm[s] = self.perm[*d]);
        QuditPermutation::new(self.radices.clone(), composed_perm)
    }

    /// Returns a new permutation that inverts `self`.
    ///
    /// # Examples
    ///
    /// ```
    /// use qudit_circuit::math::QuditPermutation;
    /// let p1 = QuditPermutation::from_qubit_location(vec![1, 0]);
    /// assert_eq!(p1.invert(), p1);
    /// ```
    ///
    /// ```
    /// use qudit_circuit::math::QuditPermutation;
    /// let p1 = QuditPermutation::from_qubit_location(vec![2, 0, 3, 1]);
    /// assert_eq!(p1.invert(), QuditPermutation::from_qubit_location(vec![1, 3, 0, 2]));
    /// ```
    pub fn invert(&self) -> QuditPermutation {
        let mut inverted_perm = vec![0; self.get_num_qudits()];
        self.perm
            .iter()
            .enumerate()
            .for_each(|(s, d)| inverted_perm[*d] = s);
        QuditPermutation::new(self.radices.clone(), inverted_perm)
    }

    /// Return the permutation as a sequence of qudit swaps
    pub fn get_cycles(&self) -> Vec<(usize, usize)> {
        let mut cycles_vec = Vec::new();
        let mut current_perm = self.perm.clone();

        for in_qudit_index in 0..self.num_qudits {
            let out_qudit_index = current_perm[in_qudit_index];

            // If this qudit is in the wrong position
            if in_qudit_index != out_qudit_index {
                // Find correct position
                let cur_pos = current_perm
                    .iter()
                    .position(|&i| i == in_qudit_index)
                    .unwrap();

                // Record the swap
                cycles_vec.push((in_qudit_index, cur_pos));

                // Apply the swap to the permutation
                let tmp = current_perm[in_qudit_index];
                current_perm[in_qudit_index] = current_perm[cur_pos];
                current_perm[cur_pos] = tmp;
            }
        }

        cycles_vec.into_iter().rev().collect()
    }

    /// Return the permutation as a sequence of index swaps
    pub fn get_index_cycles(&self) -> Vec<(usize, usize)> {
        let mut cycles_vec = Vec::new();
        let mut current_perm = self.get_index_perm().clone();

        for in_index in 0..self.get_dimension() {
            let out_index = current_perm[in_index];

            // If this qudit is in the wrong position
            if in_index != out_index {
                // Find correct position
                let cur_pos =
                    current_perm.iter().position(|&i| i == in_index).unwrap();

                // Record the swap
                cycles_vec.push((in_index, cur_pos));

                // Apply the swap to the permutation
                let tmp = current_perm[in_index];
                current_perm[in_index] = current_perm[cur_pos];
                current_perm[cur_pos] = tmp;
            }
        }

        cycles_vec.into_iter().rev().collect()
    }

    pub fn swap_rows<E: Entity + ComplexField>(
        &self,
        a: impl AsMatRef<E>,
    ) -> Mat<E> {
        let in_mat = a.as_mat_ref();
        let p_mat = self.get_matrix();
        let mut out = Mat::zeros(in_mat.nrows(), in_mat.ncols());
        permute_rows(out.as_mut(), in_mat, p_mat.as_ref());
        out
    }

    // TODO: swap_rows_in_place

    pub fn swap_rows_to_buf<E: Entity + ComplexField>(
        &self,
        a: impl AsMatRef<E>,
        mut b: impl AsMatMut<E>,
    ) {
        let p_mat = self.get_matrix();
        permute_rows(b.as_mat_mut(), a.as_mat_ref(), p_mat.as_ref());
    }

    pub fn swap_cols<E: Entity + ComplexField>(
        &self,
        a: impl AsMatRef<E>,
    ) -> Mat<E> {
        let in_mat = a.as_mat_ref();
        let p_mat = self.get_matrix();
        let mut out = Mat::zeros(in_mat.nrows(), in_mat.ncols());
        permute_cols(out.as_mut(), in_mat, p_mat.as_ref());
        out
    }

    // TODO: swap_cols_in_place

    pub fn swap_cols_to_buf<E: Entity + ComplexField>(
        &self,
        a: impl AsMatRef<E>,
        mut b: impl AsMatMut<E>,
    ) {
        let p_mat = self.get_matrix();
        permute_cols(b.as_mat_mut(), a.as_mat_ref(), p_mat.as_ref());
    }

    pub fn apply<E: Entity + ComplexField>(
        &self,
        a: impl AsMatRef<E>,
    ) -> Mat<E> {
        let in_mat = a.as_mat_ref();
        let p_mat = self.get_matrix();
        let mut out = Mat::zeros(in_mat.nrows(), in_mat.ncols());
        permute_rows(out.as_mut(), in_mat, p_mat.as_ref());
        permute_cols(out.as_mut(), in_mat, p_mat.as_ref());
        out
    }

    // TODO: apply_in_place

    pub fn apply_to_buf<E: Entity + ComplexField>(
        &self,
        a: impl AsMatRef<E>,
        mut b: impl AsMatMut<E>,
    ) {
        let p_mat = self.get_matrix();
        permute_rows(b.as_mat_mut(), a.as_mat_ref(), p_mat.as_ref());
        permute_cols(b.as_mat_mut(), a.as_mat_ref(), p_mat.as_ref());
    }

    /// Returns the permutation as a unitary matrix.
    ///
    /// # Examples
    /// ```
    /// use qudit_circuit::{QuditRadices, QuditPermutation};
    ///
    /// let qubit_swap = QuditPermutation::from_qubit_location(vec![1, 0]);
    /// assert_eq!(qubit_swap.get_matrix(), Array2::from_shape_vec((4, 4), vec![
    ///    c64::new(1.0, 0.0), c64::new(0.0, 0.0), c64::new(0.0, 0.0), c64::new(0.0, 0.0),
    ///     c64::new(0.0, 0.0), c64::new(0.0, 0.0), c64::new(1.0, 0.0), c64::new(0.0, 0.0),
    ///     c64::new(0.0, 0.0), c64::new(1.0, 0.0), c64::new(0.0, 0.0), c64::new(0.0, 0.0),
    ///     c64::new(0.0, 0.0), c64::new(0.0, 0.0), c64::new(0.0, 0.0), c64::new(1.0, 0.0),
    /// ]).unwrap());
    /// ```
    pub fn get_matrix<E: Entity>(&self) -> Permutation<usize, E> {
        Permutation::new_checked(
            self.index_perm.clone().into(),
            self.inverse_index_perm.clone().into(),
        )
    }
}

/// QuditPermutation can be dereferenced into a vector of usizes.
impl Deref for QuditPermutation {
    type Target = Vec<usize>;

    fn deref(&self) -> &Vec<usize> {
        &self.perm
    }
}

impl QuditSystem for QuditPermutation {
    /// Returns the radices of the system before being permuted.
    fn get_radices(&self) -> QuditRadices {
        self.radices.clone()
    }
}

impl std::fmt::Display for QuditPermutation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[")?;
        for r in &self.perm {
            write!(f, "{}", r).unwrap();
        }
        write!(f, "]")?;
        Ok(())
    }
}

#[cfg(test)]
pub mod strategies {
    // use super::*;
    // // use crate::radices::strategies::fixed_length_radices;
    // use crate::radices::QuditRadices;
    // use proptest::prelude::*;

    // pub fn perms(num_qudits: usize) -> impl Strategy<Value =
    // QuditPermutation> {     (
    //         fixed_length_radices(num_qudits, 5),
    //         Just(Vec::from_iter(0..num_qudits)).prop_shuffle(),
    //     )
    //         .prop_map(|(radices, perm)| QuditPermutation::new(radices, perm))
    // }

    // pub fn perms_with_radices(radices: QuditRadices) -> impl Strategy<Value =
    // QuditPermutation> {     Just(Vec::from_iter(0..radices.
    // get_num_qudits()))         .prop_shuffle()
    //         .prop_map(move |perm| QuditPermutation::new(radices.clone(),
    // perm)) }

    // pub fn physical_perms(num_qudits: usize) -> impl Strategy<Value =
    // QuditPermutation> {     (
    //         fixed_length_radices(num_qudits, 5),
    //         Just(Vec::from_iter(0..num_qudits)).prop_shuffle(),
    //     )
    //         .prop_map(|(radices, perm)| QuditPermutation::new(radices, perm))
    //         .prop_filter("Permutation is not physical", |perm| {
    //             perm.has_physical_meaning()
    //         })
    // }

    // pub fn qubit_perms(num_qudits: usize) -> impl Strategy<Value =
    // QuditPermutation> {     Just(Vec::from_iter(0..num_qudits))
    //         .prop_shuffle()
    //         .prop_map(move |perm| {
    //             QuditPermutation::new(QuditRadices::new(vec![2; num_qudits]),
    // perm)         })
    // }

    // pub fn qudit_perms(num_qudits: usize, radix: usize) -> impl
    // Strategy<Value = QuditPermutation> {     Just(Vec::from_iter(0..
    // num_qudits))         .prop_shuffle()
    //         .prop_map(move |perm| {
    //             QuditPermutation::new(QuditRadices::new(vec![radix;
    // num_qudits]), perm)         })
    // }

    // pub fn arbitrary_qudit_perms(num_qudits: usize) -> impl Strategy<Value =
    // QuditPermutation> {     (
    //         2..5usize,
    //         Just(Vec::from_iter(0..num_qudits)).prop_shuffle(),
    //     )
    //         .prop_map(move |(radix, perm)| {
    //             QuditPermutation::new(QuditRadices::new(vec![radix;
    // num_qudits]), perm)         })
    // }
}

#[cfg(test)]
mod tests {
    // use super::QuditPermutation;
    // // use crate::math::hsd;
    // use crate::Gate;
    // use crate::QuditRadices;

    // #[test]
    // fn test_swap_as_perm() {
    //     for radix in 2..5 {
    //         let radices = QuditRadices::new(vec![radix, radix]);
    //         let perm = QuditPermutation::new(radices, vec![1, 0]);
    //         let perm_mat = perm.get_matrix();
    //         assert_eq!(Gate::QuditSwap(radix).get_unitary(&[]), perm_mat);
    //     }
    // }

    // #[test]
    // fn test_double_swap_as_perm() {
    //     for radix in 2..5 {
    //         let radices = QuditRadices::new(vec![radix; 4]);
    //         let perm = QuditPermutation::new(radices, vec![1, 0, 3, 2]);
    //         let perm_mat = perm.get_matrix();
    //         let swap_utry = Gate::QuditSwap(radix).get_unitary(&[]);
    //         assert_eq!(kron(&swap_utry, &swap_utry), perm_mat);
    //     }
    // }

    // #[test]
    // fn test_complicated_perm() {
    //     for radix in 2..3 {
    //         let radices = QuditRadices::new(vec![radix; 4]);
    //         let perm = QuditPermutation::new(radices, vec![1, 3, 0, 2]);
    //         let perm_mat = perm.get_matrix();
    //         // for r in perm_mat.outer_iter() {
    //         //     let mut line_str = String::new();
    //         //     for c in r.iter() {
    //         //         line_str += &format!("{} + {}i, ", c.re, c.im);
    //         //     }
    //         //     println!("{}", line_str);
    //         // }
    //         let swap_utry = Gate::QuditSwap(radix).get_unitary(&[]);
    //         let id = Array::eye(radix);
    //         let c1 = kron(&id, &kron(&swap_utry, &id));
    //         let c2 = kron(&swap_utry, &swap_utry);
    //         println!("{}", hsd(c1.dot(&c2).view(), perm_mat.view()));
    //         assert!(hsd(c1.dot(&c2).view(), perm_mat.view()) < 1e-7);
    //     }
    // }
}
