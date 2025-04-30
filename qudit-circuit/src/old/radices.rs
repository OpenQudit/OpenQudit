use std::ops::Add;
use std::ops::Deref;

/// The base of each qudit in a qudit system.
///
/// This object represents the radix -- sometimes called the base, level
/// or ditness -- of each qudit in a qudit system. A qubit is a two-level
/// qudit or a qudit with radix two, while a qutrit is a three-level qudit.
/// Two qutrits together are represented by the [3, 3] radices object.
/// Qudit indices are counted left to right, for example a [2, 3]
/// radices is interpreted as a qubit as the first qudit and a qutrit as the
/// second one.
#[derive(Hash, PartialEq, Eq, Clone, Debug)]
pub struct QuditRadices {
    /// The stored radices object.
    radices: Vec<usize>,
}

impl QuditRadices {
    /// Returns a radices object from the given vector.
    ///
    /// # Arguments
    ///
    /// * `radices` - A vector detailing the radices of a qudit system.
    ///
    /// # Panics
    ///
    /// If radices does not represent a valid system. This can happen
    /// if any of the radices are 0 or 1.
    ///
    /// # Examples
    ///
    /// ```
    /// use qudit_circuit::QuditRadices;
    /// let three_qubits = QuditRadices::new(vec![2; 3]);
    /// let qubit_qutrit = QuditRadices::new(vec![2, 3]);
    /// ```
    pub fn new(radices: Vec<usize>) -> QuditRadices {
        if radices.iter().any(|r| *r < 2) {
            panic!("Radices must all be >= 2.")
        }

        QuditRadices { radices }
    }

    /// Construct the expanded form of an index in this numbering system.
    ///
    /// # Arguments
    ///
    /// * `index` - The number to expand.
    ///
    /// # Panics
    ///
    /// If `index` is too large for this system.
    ///
    /// # Examples
    ///
    /// ```
    /// use qudit_circuit::QuditRadices;
    /// let two_qutrits = QuditRadices::new(vec![3, 3]);
    /// assert_eq!(two_qutrits.expand(7), vec![1, 2]);
    ///
    /// let hybrid_system = QuditRadices::new(vec![3, 2, 3]);
    /// assert_eq!(hybrid_system.expand(17), vec![2, 1, 2]);
    /// ```
    pub fn expand(&self, mut index: usize) -> Vec<usize> {
        if index >= self.get_dimension() {
            panic!(
                "Provided index {} is too large for this system with radices: {:#?}",
                index, self
            );
        }

        let mut expansion = Vec::with_capacity(self.len());

        for radix in self.radices.iter() {
            let coef = index % radix;
            index = index - coef;
            index = index / radix;
            expansion.push(coef);
        }

        expansion
    }

    /// Destruct an expanded form of an index back into its base 10 number.
    ///
    /// # Arguments
    ///
    /// * `expansion` - The expansion to compress.
    ///
    /// # Panics
    ///
    /// If `expansion` has a mismatch in length or radices.
    ///
    /// # Examples
    ///
    /// ```
    /// use qudit_circuit::QuditRadices;
    /// let two_qutrits = QuditRadices::new(vec![3, 3]);
    /// assert_eq!(two_qutrits.compress(&vec![1, 2]), 7);
    ///
    /// let hybrid_system = QuditRadices::new(vec![3, 2, 3]);
    /// assert_eq!(hybrid_system.compress(&vec![2, 1, 2]), 17);
    /// ```
    pub fn compress(&self, expansion: &[usize]) -> usize {
        if self.len() != expansion.len() {
            panic!("Invalid expansion: incorrect number of qudits.")
        }

        if expansion
            .iter()
            .enumerate()
            .any(|(index, coef)| *coef >= self.radices[index])
        {
            panic!("Invalid expansion: mismatch in qudit radices.")
        }

        if expansion.len() == 0 {
            return 0;
        }

        let mut acm_val = expansion[0];
        let mut acm_base = self.radices[0];

        for coef_index in 1..expansion.len() {
            let coef = expansion[coef_index];
            acm_val += coef * acm_base;
            acm_base *= self.radices[coef_index];
        }

        acm_val
    }

    /// Returns the dimension of a system described by these radices.
    ///
    /// # Examples
    ///
    /// ```
    /// use qudit_circuit::QuditRadices;
    /// let two_qubits = QuditRadices::new(vec![2, 2]);
    /// assert_eq!(two_qubits.get_dimension(), 4);
    ///
    /// let two_qutrits = QuditRadices::new(vec![3, 3]);
    /// assert_eq!(two_qutrits.get_dimension(), 9);
    ///
    /// let hybrid_system = QuditRadices::new(vec![3, 2, 3]);
    /// assert_eq!(hybrid_system.get_dimension(), 18);
    /// ```
    pub fn get_dimension(&self) -> usize {
        self.radices.iter().product()
    }

    /// Returns the number of qudits represented by these radices.
    ///
    /// # Examples
    ///
    /// ```
    /// use qudit_circuit::QuditRadices;
    /// let two_qubits = QuditRadices::new(vec![2, 2]);
    /// assert_eq!(two_qubits.get_num_qudits(), 2);
    ///
    /// let two_qutrits = QuditRadices::new(vec![3, 3]);
    /// assert_eq!(two_qutrits.get_num_qudits(), 2);
    ///
    /// let hybrid_system = QuditRadices::new(vec![3, 2, 3]);
    /// assert_eq!(hybrid_system.get_num_qudits(), 3);
    /// ```
    pub fn get_num_qudits(&self) -> usize {
        self.radices.len()
    }

    /// Concatenates two QuditRadices objects into a new object.
    ///
    /// # Arguments
    ///
    /// * `other` - The other QuditRadices object to concatenate with `self`.
    ///
    /// # Examples
    ///
    /// ```
    /// use qudit_circuit::QuditRadices;
    /// let two_qubits = QuditRadices::new(vec![2, 2]);
    /// let two_qutrits = QuditRadices::new(vec![3, 3]);
    /// let four_qudits = QuditRadices::new(vec![2, 2, 3, 3,]);
    /// assert_eq!(two_qubits.concat(&two_qutrits), four_qudits);
    ///
    /// let hybrid_system = QuditRadices::new(vec![3, 2, 3]);
    /// let two_qutrits = QuditRadices::new(vec![3, 3]);
    /// let five_qudits = QuditRadices::new(vec![3, 2, 3, 3, 3]);
    /// assert_eq!(hybrid_system.concat(&two_qutrits), five_qudits);
    /// ```
    pub fn concat(&self, other: &QuditRadices) -> QuditRadices {
        let mut radices = self.radices.clone();
        radices.extend(&other.radices);
        QuditRadices::new(radices)
    }

    /// Returns true if these radices describe a qubit-only system.
    ///
    /// # Examples
    ///
    /// ```
    /// use qudit_circuit::{QuditRadices, radices};
    /// let two_qubits = radices![2, 2];
    /// assert!(two_qubits.is_qubit_only());
    /// ```
    ///
    /// ```
    /// use qudit_circuit::{QuditRadices, radices};
    /// let two_qutrits = radices![3, 3];
    /// assert!(!two_qutrits.is_qubit_only());
    /// ```
    pub fn is_qubit_only(&self) -> bool {
        self.radices.iter().all(|r| *r == 2)
    }

    /// Returns true if these radices describe a qutrit-only system.
    ///
    /// # Examples
    ///
    /// ```
    /// use qudit_circuit::{QuditRadices, radices};
    /// let two_qubits = radices![2, 2];
    /// assert!(!two_qubits.is_qutrit_only());
    /// ```
    ///
    /// ```
    /// use qudit_circuit::{QuditRadices, radices};
    /// let two_qutrits = radices![3, 3];
    /// assert!(two_qutrits.is_qutrit_only());
    /// ```
    pub fn is_qutrit_only(&self) -> bool {
        self.radices.iter().all(|r| *r == 3)
    }

    /// Returns true if these radices describe a `radix`-only system.
    ///
    /// # Arguments
    ///
    /// * `radix` - The radix to check for.
    ///
    /// # Examples
    ///
    /// ```
    /// use qudit_circuit::{QuditRadices, radices};
    /// let two_qubits = radices![2, 2];
    /// assert!(two_qubits.is_qudit_only(2));
    /// assert!(!two_qubits.is_qudit_only(3));
    /// ```
    ///
    /// ```
    /// use qudit_circuit::{QuditRadices, radices};
    /// let mixed_qudits = radices![2, 3];
    /// assert!(!mixed_qudits.is_qudit_only(2));
    /// assert!(!mixed_qudits.is_qudit_only(3));
    /// ```
    pub fn is_qudit_only(&self, radix: usize) -> bool {
        self.radices.iter().all(|r| *r == radix)
    }

    /// Returns true if these radices describe a homogenous system.
    ///
    /// # Examples
    ///
    /// ```
    /// use qudit_circuit::{QuditRadices, radices};
    /// let two_qubits = radices![2, 2];
    /// assert!(two_qubits.is_homogenous());
    /// ```
    ///
    /// ```
    /// use qudit_circuit::{QuditRadices, radices};
    /// let mixed_qudits = radices![2, 3];
    /// assert!(!mixed_qudits.is_homogenous());
    /// ```
    pub fn is_homogenous(&self) -> bool {
        self.is_qudit_only(self.radices[0])
    }
}

/// QuditRadices can be dereferenced into a vector of usizes.
impl Deref for QuditRadices {
    type Target = Vec<usize>;

    fn deref(&self) -> &Vec<usize> {
        &self.radices
    }
}

impl Add for QuditRadices {
    type Output = QuditRadices;

    fn add(self, other: QuditRadices) -> QuditRadices {
        self.concat(&other)
    }
}

impl<'a, 'b> Add<&'b QuditRadices> for &'a QuditRadices {
    type Output = QuditRadices;

    fn add(self, other: &'b QuditRadices) -> QuditRadices {
        self.concat(other)
    }
}

impl From<Vec<usize>> for QuditRadices {
    fn from(radices: Vec<usize>) -> QuditRadices {
        QuditRadices::new(radices)
    }
}

impl From<&[usize]> for QuditRadices {
    fn from(radices: &[usize]) -> QuditRadices {
        QuditRadices::new(radices.to_vec())
    }
}

impl std::fmt::Display for QuditRadices {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[")?;
        for r in &self.radices {
            write!(f, "{}", r).unwrap();
        }
        write!(f, "]")?;
        Ok(())
    }
}

/// A macro to create a QuditRadices object from a list of radices.
///
/// # Examples
///
/// ```
/// use qudit_circuit::radices;
/// use qudit_circuit::QuditRadices;
///
/// let two_qubits = radices![2, 2];
/// assert_eq!(two_qubits.get_dimension(), 4);
/// assert_eq!(two_qubits, QuditRadices::new(vec![2, 2]));
///
/// let two_qutrits = radices![3, 3];
/// assert_eq!(two_qutrits.get_dimension(), 9);
/// assert_eq!(two_qutrits, QuditRadices::new(vec![3, 3]));
///
/// let hybrid_system = radices![3, 2, 3];
/// assert_eq!(hybrid_system.get_dimension(), 18);
/// assert_eq!(hybrid_system, QuditRadices::new(vec![3, 2, 3]));
///
/// let ten_qubits = radices![2; 10];
/// assert_eq!(ten_qubits.get_dimension(), 1024);
/// assert_eq!(ten_qubits, QuditRadices::new(vec![2; 10]));
/// ```
#[macro_export]
macro_rules! radices {
    ($($e:expr),*) => {
        QuditRadices::new(vec![$($e),*])
    };
    ($elem:expr; $n:expr) => {
        QuditRadices::new(vec![$elem; $n])
    };
}

#[cfg(test)]
pub mod strategies {
    use proptest::prelude::*;

    use super::*;

    impl Arbitrary for QuditRadices {
        type Parameters = (usize, usize, usize, usize);
        type Strategy = BoxedStrategy<Self>;

        fn arbitrary() -> Self::Strategy {
            Self::arbitrary_with((2, 4, 1, 4))
        }

        fn arbitrary_with(args: Self::Parameters) -> Self::Strategy {
            let min_radix = args.0;
            let max_radix = args.1;
            let min_count = args.2;
            let max_count = args.3;
            prop::collection::vec(min_radix..=max_radix, min_count..=max_count)
                .prop_map(|v| QuditRadices::new(v))
                .boxed()
        }
    }
}

// #[cfg(test)]
// mod tests {
//     // use super::strategies::arbitrary_length_radices;
//     use super::QuditRadices;
//     use proptest::prelude::*;

//     /// An expansion should solve for the original index.
//     fn check_expansion(index: usize, rdx: &QuditRadices, exp: &[usize]) {
//         // expansion and radices must have same length
//         assert_eq!(rdx.len(), exp.len());

//         if rdx.len() == 0 {
//             assert_eq!(index, 0);
//         }

//         let mut acm_val = exp[0];
//         let mut acm_base = rdx[0];

//         for coef_index in 1..exp.len() {
//             let coef = exp[coef_index];
//             acm_val += coef * acm_base;
//             acm_base *= rdx[coef_index];
//         }

//         assert_eq!(acm_val, index);
//     }

//     // proptest! {
//     //     /// An expansion should compress to the same value.
//     //     #[test]
//     //     fn test_expand_compress(rdx in arbitrary_length_radices(5, 4)) {
//     //         for index in 0..rdx.get_dimension() {
//     //             let exp = rdx.expand(index);
//     //             check_expansion(index, &rdx, &exp);
//     //             assert_eq!(rdx.compress(&exp), index);
//     //         }
//     //     }
//     // }

//     #[test]
//     fn test_vec_ops() {
//         let rdx = QuditRadices::new(vec![2, 3, 4]);
//         assert_eq!(rdx.len(), 3);
//         assert_eq!(rdx[1], 3);
//         assert_eq!(rdx[1..], [3, 4]);
//         assert_eq!(rdx.clone(), rdx);
//     }
// }
