use std::collections::HashSet;

use faer::Mat;
use faer::mat::AsMatMut;
use faer::mat::AsMatRef;
use faer::perm::permute_cols;
use faer::perm::permute_rows;
use faer::perm::Perm;
use faer::reborrow::Reborrow;
use faer::reborrow::ReborrowMut;
use faer_traits::ComplexField;

use crate::Radices;
use crate::QuditSystem;
use crate::Radix;

// TODO: TEST with FRPR
// let vec = Mat::from_fn(radices.get_dimension(), 1, |i, _| {
//     c64::new(i as f64, 0.0)
// });
// let mut vec_buf = Mat::zeros(radices.get_dimension(), 1);
// let mut extended_radices = radices.clone().to_vec();
// let mut extended_perm = perm.clone().to_vec();
// extended_radices.push(1);
// extended_perm.push(perm.len());
// fused_reshape_permute_reshape_into(
//     vec.as_ref(),
//     &extended_radices,
//     &extended_perm,
//     vec_buf.as_mut(),
// );

// let mut index_perm = vec![];
// for i in 0..radices.get_dimension() {
//     index_perm.push(vec_buf.read(i, 0).re as usize);
// }

/// Calculate the index permutation from a qudit permutation for a given radices.
///
/// # Arguments
///
/// * `radices` - The radices of the system being permuted.
///
/// * `perm` - A vector describing the permutation.
///
/// # Returns
///
/// A vector of indices that describes the permutation in the index space.
///
/// # Examples
///
/// ```
/// use qudit_core::{QuditRadices, radices, calc_index_permutation};
/// let radices = radices![2; 2];
/// let perm = vec![1, 0];
/// let index_perm = calc_index_permutation(&radices, &perm);
/// assert_eq!(index_perm, vec![0, 2, 1, 3]);
/// ```
pub fn calc_index_permutation(radices: &Radices, perm: &[usize]) -> Vec<usize> {
    let mut index_perm = Vec::with_capacity(radices.dimension());
    let place_values = radices.place_values();
    let perm_place_values = perm
        .iter()
        .map(|&x| place_values[x])
        .collect::<Vec<usize>>();
    let perm_radices = perm.iter().map(|&x| radices[x]).collect::<Radices>();

    for i in 0..radices.dimension() {
        let expansion = perm_radices.expand(i);

        let mut acm_val = 0usize;
        for i in 0..radices.len() {
            acm_val += expansion[i] as usize * perm_place_values[i] as usize;
        }
        index_perm.push(acm_val);
    }

    index_perm
}

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
    radices: Radices,

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
    ///
    /// * `perm` - A vector describing the permutation. The resulting operation
    ///   will shift the i-th qudit to the `perm[i]`-th qudit.
    ///
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
    /// use qudit_core::{QuditRadices, QuditPermutation};
    ///
    /// let three_qubits = QuditRadices::new(&vec![2; 3]);
    /// let qudit_shift = QuditPermutation::new(three_qubits, &vec![2, 0, 1]);
    /// ```
    ///
    /// # See Also
    ///
    /// * [`from_qubit_location`] - A convenience constructor for qubit permutations.
    /// * [`from_qudit_location`] - A convenience constructor for qudit permutations.
    pub fn new<R: Into<Radices>>(radices: R, perm: &[usize]) -> QuditPermutation {
        fn __new_impl(radices: Radices, perm: &[usize]) -> QuditPermutation {
            if perm.len() != radices.len() {
                panic!("Invalid qudit permutation: perm's qudit count doesn't match radices'.");
            }

            if !perm.iter().all(|x| x < &radices.len()) {
                panic!("Invalid qudit permutation: permutation has invalid qudit index.");
            }

            let mut uniq = HashSet::new();
            if !perm.iter().all(|x| uniq.insert(x)) {
                panic!("Invalid qudit permutation: permutation has duplicate entries.");
            }

            let index_perm = calc_index_permutation(&radices, &perm);

            let mut inverse_index_perm = vec![0; radices.dimension()];
            index_perm
                .iter()
                .enumerate()
                .for_each(|(s, d)| inverse_index_perm[*d] = s);

            QuditPermutation {
                num_qudits: radices.len(),
                radices,
                perm: perm.to_vec(),
                index_perm,
                inverse_index_perm,
            }
        }
        __new_impl(radices.into(), perm)
    }

    /// Returns a qubit permutation specifed by `perm`.
    ///
    /// # Arguments
    ///
    /// * `perm` - The qubit permutation.
    ///
    /// # Panics
    ///
    /// * If the supplied permutation is not a proper permutation. This can
    ///   happen when there is a duplicate or invalid qudit index.
    ///
    /// # Examples
    ///
    /// ```
    /// use qudit_core::{QuditRadices, QuditPermutation};
    ///
    /// let qubit_shift = QuditPermutation::from_qubit_location(&vec![2, 0, 1]);
    /// ```
    ///
    /// # See Also
    ///
    /// * [`new`] - A general constructor for qudit permutations.
    /// * [`from_qudit_location`] - A convenience constructor for qudit permutations.
    #[inline]
    pub fn from_qubit_location(perm: &[usize]) -> QuditPermutation {
        Self::from_qudit_location(2u8, perm)
    }

    /// Returns a qudit permutation specifed by `perm` with a uniform `radix`.
    ///
    /// # Arguments
    ///
    /// * `perm` - The qubit permutation.
    ///
    /// * `radix` - The radix of the qudits being permuted.
    ///
    /// # Panics
    ///
    /// * If the supplied permutation is not a proper permutation. This can
    ///   happen when there is a duplicate or invalid qudit index.
    ///
    /// * If the radix is not a valid radix.
    ///
    /// # Examples
    ///
    /// ```
    /// use qudit_core::{QuditRadices, QuditPermutation};
    ///
    /// let qutrit_swap = QuditPermutation::from_qudit_location(3, &vec![1, 0]);
    /// ```
    ///
    /// # See Also
    ///
    /// * [`new`] - A general constructor for qudit permutations.
    /// * [`from_qubit_location`] - A convenience constructor for qubit permutations.
    #[inline]
    pub fn from_qudit_location<T: Into<Radix>>(radix: T, perm: &[usize]) -> QuditPermutation {
        let qudit_iter = core::iter::repeat(radix.into()).take(perm.len());
        let rdx = Radices::from_iter(qudit_iter);
        QuditPermutation::new(rdx, perm)
    }

    /// Returns true if the permutation has physical meaning.
    ///
    /// A permutation cannot be physically implemented if it permutes
    /// qudits of different radices.
    ///
    /// # Examples
    /// ```
    /// use qudit_core::{QuditRadices, QuditPermutation};
    ///
    /// let three_qubits = QuditRadices::new(&vec![2; 3]);
    /// let qudit_shift = QuditPermutation::new(three_qubits, &vec![2, 0, 1]);
    /// assert!(qudit_shift.is_physically_implementable());
    ///
    /// let three_qutrits = QuditRadices::new(&vec![3; 3]);
    /// let qudit_shift = QuditPermutation::new(three_qutrits, &vec![2, 0, 1]);
    /// assert!(qudit_shift.is_physically_implementable());
    ///
    /// let three_qudits = QuditRadices::new(&vec![2, 3, 4]);
    /// let qudit_shift = QuditPermutation::new(three_qudits, &vec![2, 0, 1]);
    /// assert!(!qudit_shift.is_physically_implementable());
    /// ```
    pub fn is_physically_implementable(&self) -> bool {
        self.perm
            .iter()
            .enumerate()
            .all(|x| self.radices[x.0] == self.radices[*x.1])
    }

    /// Returns a permutation that sorts the circuit location
    ///
    /// # Arguments
    ///
    /// * `loc` - The index list to be sorted.
    ///
    /// * `radices` - The radices of the system being permuted.
    ///
    /// # Panics
    ///
    /// * If the supplied radices are invalid.
    ///
    /// # Examples
    ///
    /// ```
    /// use qudit_core::{QuditRadices, QuditPermutation};
    /// let radices = QuditRadices::new(&vec![2, 3]);
    /// let loc = vec![37, 24];
    /// let hybrid_swap = QuditPermutation::locally_invert_location(radices.clone(), &loc);
    /// assert_eq!(hybrid_swap, QuditPermutation::new(radices, &vec![1, 0]));
    /// ```
    pub fn locally_invert_location(radices: Radices, loc: &[usize]) -> QuditPermutation {
        let mut perm: Vec<usize> = (0..loc.len()).collect();
        perm.sort_by_key(|&i| &loc[i]);
        QuditPermutation::new(radices, &perm)
    }

    /// Returns the permutation in the index space.
    ///
    /// The qudit permutation describes how the qudits of a system are permuted.
    /// This ultimately results in the permutation of the index space of the system.
    /// For example, two qubits have four basis states: |00>, |01>, |10>, and |11>.
    /// If the qubits are permuted, the basis states are permuted as well.
    /// The index permutation describes this permutation of the basis states.
    ///
    /// # Examples
    /// ```
    /// use qudit_core::{QuditRadices, QuditPermutation};
    ///
    /// let qubit_swap = QuditPermutation::from_qubit_location(&vec![1, 0]);
    /// assert_eq!(qubit_swap.index_perm(), &vec![0, 2, 1, 3]);
    /// ```
    pub fn index_perm(&self) -> &[usize] {
        return &self.index_perm;
    }

    /// Returns the radices of the system after being permuted.
    ///
    /// # Examples
    /// ```
    /// use qudit_core::{QuditRadices, QuditPermutation};
    /// let qudit_swap = QuditPermutation::new(QuditRadices::new(&vec![2, 3]), &vec![1, 0]);
    /// assert_eq!(qudit_swap.permuted_radices(), QuditRadices::new(&vec![3, 2]));
    ///
    /// let qudit_swap = QuditPermutation::new(QuditRadices::new(&vec![2, 3, 4]), &vec![1, 0, 2]);
    /// assert_eq!(qudit_swap.permuted_radices(), QuditRadices::new(&vec![3, 2, 4]));
    /// ```
    pub fn permuted_radices(&self) -> Radices {
        self.perm.iter().map(|&i| self.radices[i]).collect()
    }

    /// Returns true is this permutation does not permute any qudit.
    ///
    /// # Examples
    /// ```
    /// use qudit_core::QuditPermutation;
    /// let identity = QuditPermutation::from_qubit_location(&vec![0, 1]);
    /// assert!(identity.is_identity());
    /// ```
    ///
    /// ```
    /// use qudit_core::QuditPermutation;
    /// let identity = QuditPermutation::from_qubit_location(&vec![1, 0]);
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
    /// use qudit_core::QuditPermutation;
    /// let p1 = QuditPermutation::from_qubit_location(&vec![1, 0]);
    /// let p2 = QuditPermutation::from_qubit_location(&vec![1, 0]);
    /// assert!(p1.compose(&p2).is_identity());
    ///
    /// let p1 = QuditPermutation::from_qubit_location(&vec![0, 2, 1, 3]);
    /// let p2 = QuditPermutation::from_qubit_location(&vec![1, 0, 3, 2]);
    /// assert_eq!(p1.compose(&p2), QuditPermutation::from_qubit_location(&vec![2, 0, 3, 1]));
    /// ```
    pub fn compose(&self, arg_perm: &QuditPermutation) -> QuditPermutation {
        if self.radices != arg_perm.radices {
            panic!("Permutations being composed have incompatible radices.");
        }

        let mut composed_perm = vec![0; self.num_qudits()];
        arg_perm
            .iter()
            .enumerate()
            .for_each(|(s, d)| composed_perm[s] = self.perm[*d]);
        QuditPermutation::new(self.radices.clone(), &composed_perm)
    }

    /// Returns a new permutation that inverts `self`.
    ///
    /// # Examples
    ///
    /// ```
    /// use qudit_core::QuditPermutation;
    /// let p1 = QuditPermutation::from_qubit_location(&vec![1, 0]);
    /// assert_eq!(p1.invert(), p1);
    /// ```
    ///
    /// ```
    /// use qudit_core::QuditPermutation;
    /// let p1 = QuditPermutation::from_qubit_location(&vec![2, 0, 3, 1]);
    /// assert_eq!(p1.invert(), QuditPermutation::from_qubit_location(&vec![1, 3, 0, 2]));
    /// ```
    pub fn invert(&self) -> QuditPermutation {
        let mut inverted_perm = vec![0; self.num_qudits()];
        self.perm
            .iter()
            .enumerate()
            .for_each(|(s, d)| inverted_perm[*d] = s);
        QuditPermutation::new(self.radices.clone(), &inverted_perm)
    }

    /// Returns the permutation as a sequence of cycles.
    ///
    /// Any permutation can be represented as a product of disjoint cycles.
    /// This function performs that decompositions.
    ///
    /// # Returns
    ///
    /// A vector of cycles, where each cycle is a vector of qudit indices.
    ///
    /// # Examples
    /// ```
    /// use qudit_core::{QuditRadices, QuditPermutation};
    /// let qubit_swap = QuditPermutation::from_qubit_location(&vec![1, 0]);
    /// assert_eq!(qubit_swap.cycles(), vec![vec![0, 1]]);
    ///
    /// let complex_perm = QuditPermutation::from_qudit_location(3, &vec![1, 2, 0]);
    /// assert_eq!(complex_perm.cycles(), vec![vec![0, 1, 2]]);
    ///
    /// let complex_perm = QuditPermutation::from_qubit_location(&vec![1, 0, 3, 2]);
    /// assert_eq!(complex_perm.cycles(), vec![vec![0, 1], vec![2, 3]]);
    /// ```
    ///
    /// # Notes
    ///
    /// The cycles are not unique. For example, the permutation (0, 1, 2) can
    /// be represented as either (0, 1, 2) or (2, 0, 1). This function returns
    /// the former. Each new cycle starts with the smallest index.
    ///
    /// # See Also
    ///
    /// * [`transpositions`] - Returns the permutation as a sequence of qudit swaps.
    /// * [`index_cycles`] - Returns the permutation as a sequence of index swaps.
    /// * [`index_transpositions`] - Returns the permutation as a sequence of index swaps.
    pub fn cycles(&self) -> Vec<Vec<usize>> {
        let mut cycles_vec = Vec::new();
        let mut visited = vec![false; self.num_qudits];

        for new_cycle_start_index in 0..self.num_qudits {
            if visited[new_cycle_start_index] {
                continue;
            }

            let mut cycle = Vec::new();
            let mut cycle_iter_index = new_cycle_start_index;

            while !visited[cycle_iter_index] {
                visited[cycle_iter_index] = true;
                cycle.push(cycle_iter_index);
                cycle_iter_index = self.perm[cycle_iter_index];
            }

            if cycle.len() > 1 {
                cycles_vec.push(cycle);
            }
        }

        cycles_vec
    }

    /// Returns the index-space permutation as a sequence of cycles.
    ///
    /// See [`cycles`] for more information.
    ///
    /// # Returns
    ///
    /// A vector of cycles, where each cycle is a vector of qudit indices.
    ///
    /// # Examples
    ///
    /// ```
    /// use qudit_core::{QuditRadices, QuditPermutation};
    /// let qubit_swap = QuditPermutation::from_qubit_location(&vec![1, 0]);
    /// assert_eq!(qubit_swap.index_cycles(), vec![vec![1, 2]]);
    /// ```
    ///
    /// # See Also
    ///
    /// * [`cycles`] - Returns the permutation as a sequence of cycles.
    /// * [`transpositions`] - Returns the permutation as a sequence of qudit swaps.
    /// * [`index_transpositions`] - Returns the permutation as a sequence of index swaps.
    pub fn index_cycles(&self) -> Vec<Vec<usize>> {
        let mut cycles_vec = Vec::new();
        let mut visited = vec![false; self.dimension()];

        for new_cycle_start_index in 0..self.dimension() {
            if visited[new_cycle_start_index] {
                continue;
            }

            let mut cycle = Vec::new();
            let mut cycle_iter_index = new_cycle_start_index;

            while !visited[cycle_iter_index] {
                visited[cycle_iter_index] = true;
                cycle.push(cycle_iter_index);
                cycle_iter_index = self.index_perm[cycle_iter_index];
            }

            if cycle.len() > 1 {
                cycles_vec.push(cycle);
            }
        }

        cycles_vec
    }

    /// Return the permutation as a sequence of qudit swaps
    ///
    /// Any permutation can be represented as a product of transpositions
    /// (2-cycles or swaps). This function decomposes the permutation as a
    /// sequence of tranpositions.
    ///
    /// # Returns
    ///
    /// A vector of transpositions, where each swap is a qudit index tuple.
    ///
    /// # Examples
    ///
    /// ```
    /// use qudit_core::{QuditRadices, QuditPermutation};
    /// let qubit_swap = QuditPermutation::from_qubit_location(&vec![1, 0]);
    /// assert_eq!(qubit_swap.transpositions(), vec![(0, 1)]);
    ///
    /// let complex_perm = QuditPermutation::from_qudit_location(3, &vec![1, 2, 0]);
    /// assert_eq!(complex_perm.transpositions(), vec![(0, 1), (1, 2)]);
    /// ```
    ///
    /// # See Also
    ///
    /// * [`cycles`] - Returns the permutation as a sequence of cycles.
    /// * [`index_cycles`] - Returns the permutation as a sequence of index swaps.
    /// * [`index_transpositions`] - Returns the index permutation as a swap sequence.
    pub fn transpositions(&self) -> Vec<(usize, usize)> {
        let mut swaps_vec = Vec::new();
        for cycle in self.cycles() {
            for i in 0..cycle.len() - 1 {
                swaps_vec.push((cycle[i], cycle[i + 1]));
            }
        }
        swaps_vec
    }

    /// Return the index-space permutation as a sequence of swaps
    ///
    /// See [`transpositions`] for more information.
    ///
    /// # Returns
    ///
    /// A vector of swaps, where each swap is a tuple of qudit indices.
    ///
    /// # Examples
    ///
    /// ```
    /// use qudit_core::{QuditRadices, QuditPermutation};
    /// let qubit_swap = QuditPermutation::from_qubit_location(&vec![1, 0]);
    /// assert_eq!(qubit_swap.index_transpositions(), vec![(1, 2)]);
    /// ```
    ///
    /// # See Also
    ///
    /// * [`cycles`] - Returns the permutation as a sequence of cycles.
    /// * [`transpositions`] - Returns the permutation as a sequence of qudit swaps.
    /// * [`index_cycles`] - Returns the permutation as a sequence of index swaps.
    pub fn index_transpositions(&self) -> Vec<(usize, usize)> {
        let mut swaps_vec = Vec::new();
        for cycle in self.index_cycles() {
            for i in 0..cycle.len() - 1 {
                swaps_vec.push((cycle[i], cycle[i + 1]));
            }
        }
        swaps_vec
    }

    /// Swap the rows of a matrix according to the permutation.
    ///
    /// This function allocates and returns a new matrix. The permutation of
    /// the rows is done in index space.
    ///
    /// # Arguments
    ///
    /// * `a` - The matrix to be permuted.
    ///
    /// # Returns
    ///
    /// A new matrix with the rows permuted.
    ///
    /// # Examples
    ///
    /// ```
    /// use qudit_core::{QuditRadices, QuditPermutation};
    /// use qudit_core::matrix::{mat, Mat};
    /// let qubit_swap = QuditPermutation::from_qubit_location(&vec![1, 0]);
    /// let mat = mat![
    ///     [1., 2., 3., 4.],
    ///     [5., 6., 7., 8.],
    ///     [9., 10., 11., 12.],
    ///     [13., 14., 15., 16.]
    /// ];
    /// let permuted_mat = qubit_swap.swap_rows(&mat);
    /// assert_eq!(permuted_mat, mat![
    ///     [1., 2., 3., 4.],
    ///     [9., 10., 11., 12.],
    ///     [5., 6., 7., 8.],
    ///     [13., 14., 15., 16.]
    /// ]);
    /// ```
    ///
    /// # See Also
    ///
    /// * [`swap_rows_to_buf`] - Swaps the rows of a matrix witout allocations.
    /// * [`swap_cols`] - Swaps the columns of a matrix.
    pub fn swap_rows<E: ComplexField>(
        &self,
        a: impl AsMatRef<T = E, Rows = usize, Cols = usize>,
    ) -> Mat<E> {
        let in_mat = a.as_mat_ref();
        let p_mat = self.as_faer();
        let mut out = Mat::zeros(in_mat.nrows(), in_mat.ncols());
        permute_rows(out.as_mut(), in_mat, p_mat.as_ref());
        out
    }

    /// Swap the rows of a matrix in place according to the permutation.
    ///
    /// # Arguments
    ///
    /// * `a` - The matrix to be permuted.
    ///
    /// # Panics
    ///
    /// * If the matrix dimensions do not match the dimension of the permutation.
    ///
    /// # Examples
    ///
    /// ```
    /// use qudit_core::{QuditRadices, QuditPermutation};
    /// use qudit_core::matrix::{mat, Mat};
    /// let qubit_swap = QuditPermutation::from_qubit_location(&vec![1, 0]);
    /// let mut mat = mat![
    ///     [1., 2., 3., 4.],
    ///     [5., 6., 7., 8.],
    ///     [9., 10., 11., 12.],
    ///     [13., 14., 15., 16.]
    /// ];
    /// qubit_swap.swap_rows_in_place(&mut mat);
    /// assert_eq!(mat, mat![
    ///     [1., 2., 3., 4.],
    ///     [9., 10., 11., 12.],
    ///     [5., 6., 7., 8.],
    ///     [13., 14., 15., 16.]
    /// ]);
    /// ```
    ///
    /// # See Also
    ///
    /// * [`swap_rows`] - Swaps the rows of a matrix and returns a new matrix.
    /// * [`swap_rows_to_buf`] - Swaps the rows of a matrix without allocations.
    /// * [`swap_cols_in_place`] - Swaps the columns of a matrix in place.
    /// * [`swap_cols_to_buf`] - Swaps the columns of a matrix without allocations.
    /// * [`apply`] - Applies the permutation to a matrix and returns a new matrix.
    ///
    /// # Notes
    ///
    /// This is not entirely allocation free. The list of transpositions is calculated
    /// resulting in a small allocation. To be entirely allocation free, one should
    /// use the `swap_rows_to_buf` method instead.
    pub fn swap_rows_in_place<E: ComplexField>(
        &self,
        mut a: impl AsMatMut<T = E, Rows = usize, Cols = usize>,
    ) {
        let mut in_mat = a.as_mat_mut();
        let ncols = in_mat.ncols().clone();
        assert_eq!(in_mat.nrows(), self.dimension());
        assert_eq!(in_mat.ncols(), self.dimension());
        for cycle in self.index_transpositions() {
            for col in 0..ncols {
                // SAFETY: The matrix dimensions have been checked.
                unsafe {
                    let val0 = in_mat.rb().get_unchecked(cycle.0, col).clone();
                    let val1 = in_mat.rb().get_unchecked(cycle.1, col).clone();
                    *(in_mat.rb_mut().get_mut_unchecked(cycle.1, col)) = val0;
                    *(in_mat.rb_mut().get_mut_unchecked(cycle.0, col)) = val1;
                }
            }
        }
    }

    /// Swap the rows of a matrix from an input into an output buffer.
    ///
    /// This is an allocation free version of `swap_rows`.
    ///
    /// # Arguments
    ///
    /// * `a` - The matrix to be permuted.
    ///
    /// * `b` - The buffer to write the permuted matrix into.
    ///
    /// # Panics
    ///
    /// * If the matrix dimensions do not match the dimension of the permutation.
    ///
    /// # See Also
    ///
    /// * [`swap_rows`] - Swaps the rows of a matrix and returns a new matrix.
    /// * [`swap_rows_in_place`] - Swaps the rows of a matrix in place.
    /// * [`swap_cols_to_buf`] - Swaps the columns of a matrix without allocations.
    /// * [`apply_to_buf`] - Applies the permutation to a matrix and writes the result into a buffer.
    pub fn swap_rows_to_buf<E: ComplexField>(
        &self,
        a: impl AsMatRef<T = E, Rows = usize, Cols = usize>,
        mut b: impl AsMatMut<T = E, Rows = usize, Cols = usize>,
    ) {
        let p_mat = self.as_faer();
        permute_rows(b.as_mat_mut(), a.as_mat_ref(), p_mat.as_ref());
    }

    /// Swap the columns of a matrix according to the permutation.
    ///
    /// This function allocates and returns a new matrix. The permutation of
    /// the columns is done in index space.
    ///
    /// # Arguments
    ///
    /// * `a` - The matrix to be permuted.
    ///
    /// # Returns
    ///
    /// A new matrix with the columns permuted.
    ///
    /// # Examples
    ///
    /// ```
    /// use qudit_core::{QuditRadices, QuditPermutation};
    /// use qudit_core::matrix::{mat, Mat};
    /// let qubit_swap = QuditPermutation::from_qubit_location(&vec![1, 0]);
    /// let mat = mat![
    ///     [1., 2., 3., 4.],
    ///     [5., 6., 7., 8.],
    ///     [9., 10., 11., 12.],
    ///     [13., 14., 15., 16.]
    /// ];
    /// let permuted_mat = qubit_swap.swap_cols(&mat);
    /// assert_eq!(permuted_mat, mat![
    ///     [1., 3., 2., 4.],
    ///     [5., 7., 6., 8.],
    ///     [9., 11., 10., 12.],
    ///     [13., 15., 14., 16.]
    /// ]);
    /// ```
    ///
    /// # See Also
    ///
    /// * [`swap_cols_to_buf`] - Swaps the columns of a matrix without allocations.
    /// * [`swap_rows`] - Swaps the rows of a matrix.
    /// * [`apply`] - Applies the permutation to a matrix and returns a new matrix.
    pub fn swap_cols<E: ComplexField>(
        &self,
        a: impl AsMatRef<T = E, Rows = usize, Cols = usize>,
    ) -> Mat<E> {
        let in_mat = a.as_mat_ref();
        let p_mat = self.as_faer();
        let mut out = Mat::zeros(in_mat.nrows(), in_mat.ncols());
        permute_cols(out.as_mut(), in_mat, p_mat.as_ref());
        out
    }

    /// Swap the columns of a matrix in place according to the permutation.
    ///
    /// # Arguments
    ///
    /// * `a` - The matrix to be permuted.
    ///
    /// # Panics
    ///
    /// * If the matrix dimensions do not match the dimension of the permutation.
    ///
    /// # Examples
    ///
    /// ```
    /// use qudit_core::{QuditRadices, QuditPermutation};
    /// use qudit_core::matrix::{mat, Mat};
    /// let qubit_swap = QuditPermutation::from_qubit_location(&vec![1, 0]);
    /// let mut mat = mat![
    ///     [1., 2., 3., 4.],
    ///     [5., 6., 7., 8.],
    ///     [9., 10., 11., 12.],
    ///     [13., 14., 15., 16.]
    /// ];
    /// qubit_swap.swap_cols_in_place(mat.as_mut());
    /// assert_eq!(mat, mat![
    ///    [1., 3., 2., 4.],
    ///    [5., 7., 6., 8.],
    ///    [9., 11., 10., 12.],
    ///    [13., 15., 14., 16.]
    /// ]);
    /// ```
    ///
    /// # See Also
    ///
    /// * [`swap_rows`] - Swaps the rows of a matrix and returns a new matrix.
    /// * [`swap_rows_to_buf`] - Swaps the rows of a matrix without allocations.
    /// * [`swap_rows_in_place`] - Swaps the rows of a matrix in place.
    /// * [`swap_cols`] - Swaps the columns of a matrix with allocations.
    /// * [`swap_cols_to_buf`] - Swaps the columns of a matrix without allocations.
    /// * [`apply`] - Applies the permutation to a matrix and returns a new matrix.
    ///
    /// # Notes
    ///
    /// This is not entirely allocation free. The list of transpositions is calculated
    /// resulting in a small allocation. To be entirely allocation free, one should
    /// use the `swap_cols_to_buf` method instead.
    pub fn swap_cols_in_place<E: ComplexField>(
        &self,
        mut a: impl AsMatMut<T = E, Rows = usize, Cols = usize>,
    ) {
        let mut in_mat = a.as_mat_mut();
        assert_eq!(in_mat.nrows(), self.dimension());
        assert_eq!(in_mat.ncols(), self.dimension());
        for cycle in self.index_transpositions() {
            for row in 0..in_mat.nrows() {
                // SAFETY: The matrix dimensions have been checked.
                unsafe {
                    let val0 = in_mat.rb().get_unchecked(row, cycle.0).clone();
                    let val1 = in_mat.rb().get_unchecked(row, cycle.1).clone();
                    *(in_mat.rb_mut().get_mut_unchecked(row, cycle.1)) = val0;
                    *(in_mat.rb_mut().get_mut_unchecked(row, cycle.0)) = val1;
                }
            }
        }
    }

    /// Swap the columns of a matrix from an input into an output buffer.
    ///
    /// This is an allocation free version of `swap_cols`.
    ///
    /// # Arguments
    ///
    /// * `a` - The matrix to be permuted.
    ///
    /// * `b` - The buffer to write the permuted matrix into.
    ///
    /// # Panics
    ///
    /// * If the matrix dimensions do not match the dimension of the permutation.
    ///
    /// # See Also
    ///
    /// * [`swap_cols`] - Swaps the columns of a matrix and returns a new matrix.
    /// * [`swap_cols_in_place`] - Swaps the columns of a matrix in place.
    /// * [`swap_rows_to_buf`] - Swaps the rows of a matrix without allocations.
    /// * [`apply_to_buf`] - Applies the permutation to a matrix and writes the result into a buffer.
    /// * [`apply`] - Applies the permutation to a matrix and returns a new matrix.
    pub fn swap_cols_to_buf<E: ComplexField>(
        &self,
        a: impl AsMatRef<T = E, Rows = usize, Cols = usize>,
        mut b: impl AsMatMut<T = E, Rows = usize, Cols = usize>,
    ) {
        let p_mat = self.as_faer();
        permute_cols(b.as_mat_mut(), a.as_mat_ref(), p_mat.as_ref());
    }

    /// Apply the permutation to a matrix and return a new matrix.
    ///
    /// This is equivalent to calling `swap_rows` and then `swap_cols`.
    ///
    /// # Arguments
    ///
    /// * `a` - The matrix to be permuted.
    ///
    /// # Returns
    ///
    /// A new matrix with the rows and columns permuted.
    ///
    /// # Examples
    ///
    /// ```
    /// use qudit_core::{QuditRadices, QuditPermutation};
    /// use qudit_core::matrix::{mat, Mat};
    /// let qubit_swap = QuditPermutation::from_qubit_location(&vec![1, 0]);
    /// let mat = mat![
    ///     [1., 2., 3., 4.],
    ///     [5., 6., 7., 8.],
    ///     [9., 10., 11., 12.],
    ///     [13., 14., 15., 16.]
    /// ];
    /// let permuted_mat = qubit_swap.apply(&mat);
    /// assert_eq!(permuted_mat, mat![
    ///    [1., 3., 2., 4.],
    ///    [9., 11., 10., 12.],
    ///    [5., 7., 6., 8.],
    ///    [13., 15., 14., 16.]
    /// ]);
    ///
    /// // Applying qudit permutations can help convert between little
    /// // and big endian gates.
    /// let big_endian_cnot = mat![
    ///     [1., 0., 0., 0.],
    ///     [0., 1., 0., 0.],
    ///     [0., 0., 0., 1.],
    ///     [0., 0., 1., 0.]
    /// ];
    /// let little_endian_cnot = mat![
    ///     [1., 0., 0., 0.],
    ///     [0., 0., 0., 1.],
    ///     [0., 0., 1., 0.],
    ///     [0., 1., 0., 0.]
    /// ];
    /// assert_eq!(qubit_swap.apply(&big_endian_cnot), little_endian_cnot);
    /// ```
    ///
    /// # See Also
    ///
    /// * [`swap_rows`] - Swaps the rows of a matrix and returns a new matrix.
    /// * [`swap_cols`] - Swaps the columns of a matrix and returns a new matrix.
    /// * [`apply_to_buf`] - Applies the permutation to a matrix and writes the result into a buffer.
    /// * [`apply_in_place`] - Swaps the rows of a matrix in place.
    pub fn apply<E: ComplexField>(
        &self,
        a: impl AsMatRef<T = E, Rows = usize, Cols = usize>,
    ) -> Mat<E> {
        self.swap_cols(self.swap_rows(a))
    }

    /// Apply the permutation to a matrix in place.
    ///
    /// # Arguments
    ///
    /// * `a` - The matrix to be permuted.
    ///
    /// # Returns
    ///
    /// A new matrix with the rows and columns permuted.
    ///
    /// # See Also
    ///
    /// * [`swap_rows_in_place`] - Swaps the rows of a matrix in place.
    /// * [`swap_cols_in_place`] - Swaps the columns of a matrix in place.
    /// * [`apply_to_buf`] - Applies the permutation to a matrix and writes the result into a buffer.
    /// * [`apply`] - Applies the permutation to a matrix and returns a new matrix.
    ///
    /// # Notes
    ///
    /// This is not entirely allocation free. The list of transpositions is calculated
    /// resulting in a small allocation. To be entirely allocation free, one should
    /// use the `apply_to_buf` method instead.
    pub fn apply_in_place<E: ComplexField>(
        &self,
        mut a: impl AsMatMut<T = E, Rows = usize, Cols = usize>,
    ) {
        self.swap_rows_in_place(a.as_mat_mut());
        self.swap_cols_in_place(a.as_mat_mut());
    }

    /// Apply the permutation to a matrix and write the result into a buffer.
    ///
    /// This is an allocation free version of `apply`.
    ///
    /// # Arguments
    ///
    /// * `a` - The matrix to be permuted.
    ///
    /// * `b` - The buffer to write the permuted matrix into.
    ///
    /// # See Also
    ///
    /// * [`swap_rows_to_buf`] - Swaps the rows of a matrix without allocations.
    /// * [`swap_cols_to_buf`] - Swaps the columns of a matrix without allocations.
    /// * [`apply`] - Applies the permutation to a matrix and returns a new matrix.
    /// * [`apply_in_place`] - Swaps the rows of a matrix in place.
    pub fn apply_to_buf<E: ComplexField>(
        &self,
        a: impl AsMatRef<T = E, Rows = usize, Cols = usize>,
        mut b: impl AsMatMut<T = E, Rows = usize, Cols = usize>,
    ) {
        let p_mat = self.as_faer();
        permute_rows(b.as_mat_mut(), a.as_mat_ref(), p_mat.as_ref());
        permute_cols(b.as_mat_mut(), a.as_mat_ref(), p_mat.as_ref());
    }

    /// Returns the permutation as an faer object.
    fn as_faer(&self) -> Perm<usize> {
        Perm::new_checked(
            self.index_perm.clone().into(),
            self.inverse_index_perm.clone().into(),
            self.dimension(),
        )
    }

    // Returns the permutation as a unitary matrix.
    // // TODO: Update
    // pub fn get_matrix<E: Entity>(&self) -> UnitaryMatrix<E> {
    //     todo!()
    // }
}

/// QuditPermutation can be dereferenced as a usize slice.
impl core::ops::Deref for QuditPermutation {
    type Target = [usize];

    fn deref(&self) -> &Self::Target {
        &self.perm
    }
}

/// QuditPermutations permute a qudit system.
impl QuditSystem for QuditPermutation {
    /// Returns the radices of the system before being permuted.
    fn radices(&self) -> Radices {
        self.radices.clone()
    }

    /// Returns the number of qudits being permuted.
    fn num_qudits(&self) -> usize {
        self.num_qudits
    }

    /// Returns the dimension of the system being permuted.
    fn dimension(&self) -> usize {
        self.index_perm.len()
    }
}

impl core::fmt::Display for QuditPermutation {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
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
