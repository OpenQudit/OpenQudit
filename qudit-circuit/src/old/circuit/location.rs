use std::collections::HashSet;

/// A CircuitLocation describes where an instruction is spatial executed.
///
/// In other words, it describes which circuit qudits and clbits an instruction
/// acts on. It also describes the order of the qudits and clbits, i.e., the
/// desired permutation of the gate by the order of the qudits.
///
/// Consider a four-qubit circuit, the location with qubits = [0, 2], and
/// clbits = [] describes the application of a two-qudit purely-quantum
/// instruction to the first and third qudits in the circuit. This location
/// is not equivalent to the location with qubits = [2, 0], and clbits = [],
/// as this describes a permutation of the first example.
#[derive(Hash, PartialEq, Eq, Clone, Debug)]
pub struct CircuitLocation {
    /// The described qudits in the circuit.
    qudits: Vec<usize>,

    /// The described clbits in the circuit.
    clbits: Vec<usize>,
}

impl CircuitLocation {
    /// Returns a purely-quantum CircuitLocation object from the given vector.
    ///
    /// A purely-quantum location is one that does not contain any clbits.
    ///
    /// # Arguments
    ///
    /// * `location` - A vector describing the qudits in a circuit.
    ///
    /// # Panics
    ///
    /// If `location` is not a valid location. This can happen if a qudit
    /// index appears twice in the location.
    ///
    /// # Examples
    ///
    /// ```
    /// use qudit_circuit::circuit::CircuitLocation;
    /// let loc = CircuitLocation::pure(vec![3, 0]);
    /// ```
    pub fn pure(location: Vec<usize>) -> CircuitLocation {
        CircuitLocation::new(location, Vec::new())
    }

    /// Returns a purely-classical CircuitLocation object from the given vector.
    ///
    /// A purely-classical location is one that does not contain any qudits.
    ///
    /// # Arguments
    ///
    /// * `location` - A vector describing the clbits in a circuit.
    ///
    /// # Panics
    ///
    /// If `location` is not a valid location. This can happen if a clbit
    /// index appears twice in the location.
    ///
    /// # Examples
    ///
    /// ```
    /// use qudit_circuit::circuit::CircuitLocation;
    /// let loc = CircuitLocation::classical(vec![1, 2]);
    /// ```
    pub fn classical(location: Vec<usize>) -> CircuitLocation {
        CircuitLocation::new(Vec::new(), location)
    }

    /// Returns a CircuitLocation object from the given vectors.
    ///
    /// # Arguments
    ///
    /// * `qudits` - A vector describing the qudits in a circuit.
    ///
    /// * `clbits` - A vector describing the clbits in a circuit.
    ///
    /// # Panics
    ///
    /// If `qudits` or `clbits` are not valid locations. This can happen if a
    /// qudit or clbit index appears twice in the location.
    ///
    /// # Examples
    ///
    /// ```
    /// use qudit_circuit::circuit::CircuitLocation;
    /// let loc = CircuitLocation::new(vec![3, 0], vec![1, 2]);
    /// ```
    pub fn new(qudits: Vec<usize>, clbits: Vec<usize>) -> CircuitLocation {
        // Uniqueness check // TODO: re-evaluate 20 below; maybe surround in
        // debug
        if qudits.len() < 20 && clbits.len() < 20 {
            // Performance: Locatins are typically small, so this O(N^2)
            // uniqueness check is faster than the O(N) HashSet check.
            // This speed up is because of a low constant factor and
            // the fact that the HashSet check has to allocate memory.

            for i in 0..qudits.len() {
                for j in (i + 1)..qudits.len() {
                    if qudits[i] == qudits[j] {
                        panic!("Duplicate indices in given circuit location.");
                    }
                }
            }

            for i in 0..clbits.len() {
                for j in (i + 1)..clbits.len() {
                    if clbits[i] == clbits[j] {
                        panic!("Duplicate indices in given circuit location.");
                    }
                }
            }
        } else {
            let mut uniq = HashSet::new();
            if !qudits.iter().all(|x| uniq.insert(x)) {
                panic!("Duplicate indices in given circuit location.");
            }
            uniq.clear();
            if !clbits.iter().all(|x| uniq.insert(x)) {
                panic!("Duplicate indices in given circuit location.");
            }
        }

        CircuitLocation {
            qudits: qudits,
            clbits: clbits,
        }
    }

    /// Get the qudits in this location.
    ///
    /// # Examples
    ///
    /// ```
    /// use qudit_circuit::circuit::CircuitLocation;
    /// let loc = CircuitLocation::new(vec![3, 0], vec![1, 2]);
    /// assert_eq!(loc.qudits(), &[3, 0]);
    /// ```
    pub fn qudits(&self) -> &[usize] {
        &self.qudits
    }

    /// Get the clbits in this location.
    ///
    /// # Examples
    ///
    /// ```
    /// use qudit_circuit::circuit::CircuitLocation;
    /// let loc = CircuitLocation::new(vec![3, 0], vec![1, 2]);
    /// assert_eq!(loc.clbits(), &[1, 2]);
    /// ```
    pub fn clbits(&self) -> &[usize] {
        &self.clbits
    }

    /// Returns a new location containing all elements in `self` or `other`
    ///
    /// # Arguments
    ///
    /// * `other` - The other location to union.
    ///
    /// # Notes
    ///
    /// * The output orders the elements from `self` first, then the ones from
    ///   `other`.
    ///
    /// # Examples
    ///
    /// ```
    /// use qudit_circuit::circuit::CircuitLocation;
    /// let loc1 = CircuitLocation::pure(vec![0, 2]);
    /// let loc2 = CircuitLocation::pure(vec![2, 3]);
    /// assert_eq!(loc1.union(&loc2), CircuitLocation::pure(vec![0, 2, 3]));
    /// ```
    ///
    /// ```
    /// use qudit_circuit::circuit::CircuitLocation;
    /// let loc1 = CircuitLocation::new(vec![3, 0], vec![1, 2]);
    /// let loc2 = CircuitLocation::new(vec![2, 3], vec![3, 2]);
    /// assert_eq!(loc1.union(&loc2), CircuitLocation::new(vec![3, 0, 2], vec![1, 2, 3]));
    /// ```
    pub fn union(&self, other: &CircuitLocation) -> CircuitLocation {
        let mut union_qudits = self.qudits.clone();
        for qudit_index in &other.qudits {
            if !union_qudits.contains(qudit_index) {
                union_qudits.push(*qudit_index);
            }
        }

        let mut union_clbits = self.clbits.clone();
        for clbit_index in &other.clbits {
            if !union_clbits.contains(clbit_index) {
                union_clbits.push(*clbit_index);
            }
        }

        CircuitLocation {
            qudits: union_qudits,
            clbits: union_clbits,
        }
    }

    /// Returns a new location containing all elements in `self` and `other`
    ///
    /// # Arguments
    ///
    /// * `other` - The other location to intersect.
    ///
    /// # Notes
    ///
    /// * The elements in output are ordered the same way they are ordered in
    ///   `self`.
    ///
    /// # Examples
    ///
    /// ```
    /// use qudit_circuit::circuit::CircuitLocation;
    /// let loc1 = CircuitLocation::pure(vec![0, 2]);
    /// let loc2 = CircuitLocation::pure(vec![2, 3]);
    /// assert_eq!(loc1.intersect(&loc2), CircuitLocation::pure(vec![2]));
    /// ```
    ///
    /// ```
    /// use qudit_circuit::circuit::CircuitLocation;
    /// let loc1 = CircuitLocation::classical(vec![0, 2]);
    /// let loc2 = CircuitLocation::classical(vec![2, 0]);
    /// assert_eq!(loc1.intersect(&loc2), CircuitLocation::classical(vec![0, 2]));
    /// ```
    ///
    /// ```
    /// use qudit_circuit::circuit::CircuitLocation;
    /// let loc1 = CircuitLocation::new(vec![3, 0], vec![1, 2]);
    /// let loc2 = CircuitLocation::new(vec![2, 3], vec![3, 2]);
    /// assert_eq!(loc1.intersect(&loc2), CircuitLocation::new(vec![3], vec![2]));
    /// ```
    pub fn intersect(&self, other: &CircuitLocation) -> CircuitLocation {
        let mut inter_qudits = Vec::new();
        for qudit_index in &self.qudits {
            if other.qudits.contains(qudit_index) {
                inter_qudits.push(*qudit_index);
            }
        }

        let mut inter_clbits = Vec::new();
        for clbit_index in &self.clbits {
            if other.clbits.contains(clbit_index) {
                inter_clbits.push(*clbit_index);
            }
        }

        CircuitLocation {
            qudits: inter_qudits,
            clbits: inter_clbits,
        }
    }

    /// Returns a new location containing elements in `self` but not in `other`
    ///
    /// # Arguments
    ///
    /// * `other` - The other location to subtract.
    ///
    /// # Notes
    ///
    /// * The elements in output are ordered the same way they are ordered in
    ///   `self`.
    ///
    /// # Examples
    ///
    /// ```
    /// use qudit_circuit::circuit::CircuitLocation;
    /// let loc1 = CircuitLocation::pure(vec![0, 2]);
    /// let loc2 = CircuitLocation::pure(vec![2, 3]);
    /// assert_eq!(loc1.difference(&loc2), CircuitLocation::pure(vec![0]));
    /// ```
    ///
    /// ```
    /// use qudit_circuit::circuit::CircuitLocation;
    /// let loc1 = CircuitLocation::classical(vec![0, 2]);
    /// let loc2 = CircuitLocation::classical(vec![2, 0]);
    /// assert_eq!(loc1.difference(&loc2), CircuitLocation::classical(vec![]));
    /// ```
    ///
    /// ```
    /// use qudit_circuit::circuit::CircuitLocation;
    /// let loc1 = CircuitLocation::new(vec![3, 0], vec![1, 2]);
    /// let loc2 = CircuitLocation::new(vec![2, 3], vec![3, 2]);
    /// assert_eq!(loc1.difference(&loc2), CircuitLocation::new(vec![0], vec![1]));
    /// ```
    pub fn difference(&self, other: &CircuitLocation) -> CircuitLocation {
        let mut diff_qudits = Vec::new();
        for qudit_index in &self.qudits {
            if !other.qudits.contains(qudit_index) {
                diff_qudits.push(*qudit_index);
            }
        }

        let mut diff_clbits = Vec::new();
        for clbit_index in &self.clbits {
            if !other.clbits.contains(clbit_index) {
                diff_clbits.push(*clbit_index);
            }
        }

        CircuitLocation {
            qudits: diff_qudits,
            clbits: diff_clbits,
        }
    }

    /// Returns all possible pairs of qudits in this location.
    ///
    /// These are returned as sorted pairs, i.e., the first element of the
    /// pair is always less than the second element.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use qudit_circuit::circuit::CircuitLocation;
    /// let loc = CircuitLocation::pure(vec![0, 3, 2]);
    ///
    /// let all_pairs = loc.get_qudit_pairs();
    ///
    /// for pair in vec![(0, 2), (0, 3), (2, 3)] {
    ///     assert!(all_pairs.contains(&pair))
    /// }
    /// ```
    pub fn get_qudit_pairs(&self) -> Vec<(usize, usize)> {
        let mut to_return =
            Vec::with_capacity(self.qudits.len() * (self.qudits.len() - 1) / 2);
        for qudit_index1 in &self.qudits {
            for qudit_index2 in &self.qudits {
                if qudit_index1 < qudit_index2 {
                    to_return.push((*qudit_index1, *qudit_index2));
                }
            }
        }
        to_return
    }

    /// Returns a sorted copy of the location.
    ///
    /// # Examples
    ///
    /// ```
    /// use qudit_circuit::circuit::CircuitLocation;
    /// let loc = CircuitLocation::pure(vec![2, 0, 3]);
    /// assert_eq!(loc.to_sorted(), CircuitLocation::pure(vec![0, 2, 3]));
    /// ```
    ///
    /// ```
    /// use qudit_circuit::circuit::CircuitLocation;
    /// let loc = CircuitLocation::new(vec![3, 0], vec![2, 1]);
    /// assert_eq!(loc.to_sorted(), CircuitLocation::new(vec![0, 3], vec![1, 2]));
    /// ```
    pub fn to_sorted(&self) -> CircuitLocation {
        let mut qudits_sorted = self.qudits.clone();
        let mut clbits_sorted = self.clbits.clone();
        qudits_sorted.sort();
        clbits_sorted.sort();
        CircuitLocation {
            qudits: qudits_sorted,
            clbits: clbits_sorted,
        }
    }

    /// Returns true if the location is sorted and has trivial ordering.
    ///
    /// # Examples
    ///
    /// ```
    /// use qudit_circuit::circuit::CircuitLocation;
    /// let loc = CircuitLocation::pure(vec![0, 2, 3]);
    /// assert!(loc.is_sorted());
    /// ```
    ///
    /// ```
    /// use qudit_circuit::circuit::CircuitLocation;
    /// let loc = CircuitLocation::new(vec![3, 0], vec![2, 1]);
    /// assert!(!loc.is_sorted());
    /// ```
    pub fn is_sorted(&self) -> bool {
        self.is_qudit_sorted() && self.is_clbit_sorted()
    }

    /// Returns true if the qudits are sorted and have trivial ordering.
    ///
    /// # Examples
    ///
    /// ```
    /// use qudit_circuit::circuit::CircuitLocation;
    /// let loc = CircuitLocation::pure(vec![0, 2, 3]);
    /// assert!(loc.is_qudit_sorted());
    /// ```
    ///
    /// ```
    /// use qudit_circuit::circuit::CircuitLocation;
    /// let loc = CircuitLocation::new(vec![0, 3], vec![2, 1]);
    /// assert!(loc.is_qudit_sorted());
    /// ```
    pub fn is_qudit_sorted(&self) -> bool {
        if self.qudits.len() < 2 {
            return true;
        }

        (0..(self.qudits.len() - 1))
            .all(|i| self.qudits[i] < self.qudits[i + 1])
    }

    /// Returns true if the clbits are sorted and have trivial ordering.
    ///
    /// # Examples
    ///
    /// ```
    /// use qudit_circuit::circuit::CircuitLocation;
    /// let loc = CircuitLocation::classical(vec![0, 2, 3]);
    /// assert!(loc.is_clbit_sorted());
    /// ```
    ///
    /// ```
    /// use qudit_circuit::circuit::CircuitLocation;
    /// let loc = CircuitLocation::new(vec![0, 3], vec![2, 1]);
    /// assert!(!loc.is_clbit_sorted());
    /// ```
    pub fn is_clbit_sorted(&self) -> bool {
        if self.clbits.len() < 2 {
            return true;
        }

        (0..(self.clbits.len() - 1))
            .all(|i| self.clbits[i] < self.clbits[i + 1])
    }

    /// Returns true if `qudit_index` is in the location.
    ///
    /// # Examples
    ///
    /// ```
    /// use qudit_circuit::circuit::CircuitLocation;
    /// let loc = CircuitLocation::pure(vec![0, 2, 3]);
    /// assert!(loc.contains_qudit(&0));
    /// assert!(!loc.contains_qudit(&1));
    /// ```
    pub fn contains_qudit(&self, qudit_index: &usize) -> bool {
        self.qudits.contains(&qudit_index)
    }

    /// Returns true if `clbit_index` is in the location.
    ///
    /// # Examples
    ///
    /// ```
    /// use qudit_circuit::circuit::CircuitLocation;
    /// let loc = CircuitLocation::classical(vec![0, 2, 3]);
    /// assert!(loc.contains_clbit(&0));
    /// assert!(!loc.contains_clbit(&1));
    /// ```
    pub fn contains_clbit(&self, clbit_index: &usize) -> bool {
        self.clbits.contains(&clbit_index)
    }

    /// Returns the number of qudits and clbits in the location.
    ///
    /// # Examples
    ///
    /// ```
    /// use qudit_circuit::circuit::CircuitLocation;
    /// let loc = CircuitLocation::pure(vec![0, 2, 3]);
    /// assert_eq!(loc.len(), 3);
    /// ```
    ///
    /// ```
    /// use qudit_circuit::circuit::CircuitLocation;
    /// let loc = CircuitLocation::new(vec![0, 3], vec![2, 1]);
    /// assert_eq!(loc.len(), 4);
    /// ```
    pub fn len(&self) -> usize {
        self.get_num_qudits() + self.get_num_clbits()
    }

    /// Returns the number of qudits in the location.
    ///
    /// # Examples
    ///
    /// ```
    /// use qudit_circuit::circuit::CircuitLocation;
    /// let loc = CircuitLocation::pure(vec![0, 2, 3]);
    /// assert_eq!(loc.get_num_qudits(), 3);
    /// ```
    pub fn get_num_qudits(&self) -> usize {
        self.qudits.len()
    }

    /// Returns the number of clbits in the location.
    ///
    /// # Examples
    ///
    /// ```
    /// use qudit_circuit::circuit::CircuitLocation;
    /// let loc = CircuitLocation::classical(vec![0, 2, 3]);
    /// assert_eq!(loc.get_num_clbits(), 3);
    /// ```
    pub fn get_num_clbits(&self) -> usize {
        self.clbits.len()
    }

    pub fn get_qudit_index(&self, index: &usize) -> Option<usize> {
        self.qudits.iter().position(|x| x == index)
    }

    pub fn get_clbit_index(&self, index: &usize) -> Option<usize> {
        self.clbits.iter().position(|x| x == index)
    }
}

/// A macro to create a CircuitLocation object.
///
/// This macro is used to create CircuitLocation objects in a more
/// convenient way. It is used in a similar way to the vec! macro.
///
/// # Examples
///
/// ```
/// use qudit_circuit::circuit::CircuitLocation;
/// use qudit_circuit::loc;
/// let loc = loc![0, 2, 3];
/// assert_eq!(loc, CircuitLocation::pure(vec![0, 2, 3]));
/// ```
///
/// ```
/// use qudit_circuit::circuit::CircuitLocation;
/// use qudit_circuit::loc;
/// let loc = loc![0, 2, 3; 1, 2];
/// assert_eq!(loc, CircuitLocation::new(vec![0, 2, 3], vec![1, 2]));
/// ```
///
/// ```
/// use qudit_circuit::circuit::CircuitLocation;
/// use qudit_circuit::loc;
/// let loc = loc![; 1, 2];
/// assert_eq!(loc, CircuitLocation::classical(vec![1, 2]));
/// ```
#[macro_export]
macro_rules! loc {
    ($($x:expr),*;$($y:expr),*) => {
        CircuitLocation::new(vec![$($x),*], vec![$($y),*])
    };
    ($($x:expr),*) => {
        CircuitLocation::pure(vec![$($x),*])
    };
    (;$($x:expr),*) => {
        CircuitLocation::classical(vec![$($x),*])
    };
}

#[cfg(test)]
pub mod strategies {
    use proptest::prelude::*;

    use super::*;

    impl Arbitrary for CircuitLocation {
        type Parameters =
            (usize, usize, usize, usize, usize, usize, usize, usize);
        type Strategy = BoxedStrategy<CircuitLocation>;

        fn arbitrary() -> Self::Strategy {
            Self::arbitrary_with((0, 5, 0, 5, 0, 5, 0, 2))
        }

        fn arbitrary_with(args: Self::Parameters) -> Self::Strategy {
            let min_qudit = args.0;
            let max_qudit = args.1;
            let min_num_qudits = args.2;
            let max_num_qudits = args.3;
            let min_clbit = args.4;
            let max_clbit = args.5;
            let min_num_clbits = args.6;
            let max_num_clbits = args.7;

            (
                prop::collection::vec(
                    min_qudit..=max_qudit,
                    min_num_qudits..=max_num_qudits,
                ),
                prop::collection::vec(
                    min_clbit..=max_clbit,
                    min_num_clbits..=max_num_clbits,
                ),
            )
                .prop_map(|(qudits, clbits)| {
                    CircuitLocation::new(qudits, clbits)
                })
                .boxed()
        }
    }
}

// #[cfg(test)]
// mod tests {
//     use super::CircuitLocation;
//     use proptest::prelude::*;

//     proptest! {
//         /// The union operation is idempotent.
//         #[test]
//         fn test_union_self(loc in arbitrary_location(5, 4)) {
//             assert_eq!(loc.union(&loc), loc);
//         }
//     }

//     proptest! {
//         #[test]
//         fn test_union_other(loc1 in arbitrary_location(5, 4), loc2 in
// arbitrary_location(5, 4)) {             let union_loc = loc1.union(&loc2);
//             assert!(union_loc.len() >= loc1.len());
//             assert!(union_loc.len() >= loc2.len());
//             assert!(union_loc.len() <= loc1.len() + loc2.len());
//             assert!(loc1.iter().all(|x| union_loc.contains(x)));
//             assert!(loc2.iter().all(|x| union_loc.contains(x)));
//         }
//     }

//     proptest! {
//         /// The intersect operation is idempotent.
//         #[test]
//         fn test_intersect_self(loc in arbitrary_location(5, 4)) {
//             assert_eq!(loc.intersect(&loc), loc);
//         }
//     }

//     proptest! {
//         #[test]
//         fn test_intersect_other(loc1 in arbitrary_location(5, 4), loc2 in
// arbitrary_location(5, 4)) {             let intersect_loc =
// loc1.intersect(&loc2);             assert!(intersect_loc.len() <=
// loc1.len());             assert!(intersect_loc.len() <= loc2.len());
//             assert!(intersect_loc.iter().all(|x| loc1.contains(x)));
//             assert!(intersect_loc.iter().all(|x| loc2.contains(x)));
//         }
//     }

//     #[test]
//     fn test_vec_ops() {
//         let loc = CircuitLocation::new(vec![2, 3, 4]);
//         assert_eq!(loc.len(), 3);
//         assert_eq!(loc[1], 3);
//         assert_eq!(loc[1..], [3, 4]);
//         assert_eq!(loc.clone(), loc);
//     }
// }
