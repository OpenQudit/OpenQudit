use std::collections::HashMap;

use crate::radix::{FromRadix, ToRadix};
use crate::QuditSystem;

/// The size of the statically allocated inline array of radices.
///
/// Radices objects with fewer than this number of qudits can avoid heap
/// allocation by storing the radices in a statically allocated array.
/// This can be useful for performance in some cases. This size is
/// determined such that there is no wasted space:
///
/// ```rust
/// use std::mem::size_of;
/// use qudit_core::QuditRadices;
///
/// assert!(size_of::<QuditRadices>() == size_of::<Vec<u8>>());
/// ```
///
/// This is only possible because of rust's enum discriminant optimizations.
/// In this case, the discriminant is stored in the pointer field of the
/// vector. Since the null pointer is invalid for a Vec, a null pointer
/// implies that the enum is a QuditRadices::Array, leaving 16 bytes for
/// data. This number is one less than that to make room for the length.
#[cfg(target_pointer_width = "64")]
const RADICES_INLINE_SIZE: usize = 15;

#[cfg(target_pointer_width = "32")]
const RADICES_INLINE_SIZE: usize = 7;

/// The number of basis states for each qudit in a qudit system.
///
/// This object represents the radix -- sometimes called the base, level
/// or ditness -- of each qudit in a qudit system, and is implemented as a
/// sequence of unsigned, byte-sized integers. A qubit is a two-level
/// qudit or a qudit with radix two, while a qutrit is a three-level qudit.
/// Two qutrits together are represented by the [3, 3] radices object.
///
/// No radix can be less than 2, as this would not be a valid qudit system.
/// While we support upto 8-bytes for the number of qudits (4-bytes on a
/// 32 bit machine), this implementation does not support individual
/// radices greater than 255, as this would require a larger data type.
///
/// ## Ordering and Endianness
///
/// Qudit indices are counted left to right, for example a [2, 3]
/// radices is interpreted as a qubit as the first qudit and a qutrit as
/// the second one. Openqudit uses big-endian ordering, so the qubit in
/// the previous example is the most significant qudit and the qutrit is
/// the least significant qudit. For example, in the same system, a state
/// |10> would be represented by the decimal number 3.
#[derive(Hash, PartialEq, Eq, Clone)]
pub enum QuditRadices {
    /// Small systems are stored on the stack in a fixed-size array.
    /// This can avoid heap allocation for small systems and the size of
    /// the array is determined such that there is no wasted space, see
    /// the [RADICES_INLINE_SIZE] constant for more information.
    /// The array is always fully initialized, and the number of qudits
    /// is stored in the first byte.
    Array(u8, [u8; RADICES_INLINE_SIZE]),

    /// Large systems are stored on the heap in a standard Vec.
    Heap(Vec<u8>),
}

impl QuditRadices {
    /// Constructs a radices object from the given vector.
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
    /// use qudit_core::QuditRadices;
    /// let three_qubits = QuditRadices::new(&vec![2; 3]);
    /// let qubit_qutrit = QuditRadices::new(&vec![2, 3]);
    /// ```
    ///
    /// # Note
    ///
    /// This function is usually not necessary, at least directly. Instead, use
    /// the shorthand [`radices!`] macro to create a QuditRadices object.
    pub fn new<T: ToRadix>(radices: &[T]) -> QuditRadices {
        if radices.len() < RADICES_INLINE_SIZE {
            let mut array = [0; RADICES_INLINE_SIZE];
            for (i, r) in radices.iter().enumerate() {
                if r.is_less_than_two() {
                    panic!(
                        "Qudit radix in position {} is invalid since it is < 2: {}",
                        i, *r
                    );
                }
                array[i] = r.to_radix();
            }
            QuditRadices::Array(radices.len() as u8, array)
        } else {
            let mut heap = Vec::with_capacity(radices.len());
            for (i, r) in radices.iter().enumerate() {
                if r.is_less_than_two() {
                    panic!(
                        "Qudit radix in position {} is invalid since it is < 2: {}",
                        i, *r
                    );
                }
                heap.push(r.to_radix());
            }
            QuditRadices::Heap(heap)
        }
    }

    /// Constructs a radices object without checking invariants.
    ///
    /// The caller must ensure that the radices are valid, i.e. that
    /// they are all greater than 1.
    ///
    /// See [`new`] for a safe constructor or more information.
    ///
    // TODO: Maybe make unchecked u8 only... because have to specify type or default to i32
    // or maybe make i32 conversion print debug message/warning
    // make sure macro uses u8; put in temp with size u8
    pub unsafe fn new_unchecked<T: ToRadix>(radices: &[T]) -> QuditRadices {
        if radices.len() < RADICES_INLINE_SIZE {
            let mut array = [0; RADICES_INLINE_SIZE];
            for (i, r) in radices.iter().enumerate() {
                array[i] = r.to_radix();
            }
            QuditRadices::Array(radices.len() as u8, array)
        } else {
            let mut heap = Vec::with_capacity(radices.len());
            for r in radices {
                heap.push(r.to_radix());
            }
            QuditRadices::Heap(heap)
        }
    }

    /// Construct the expanded form of an index in this numbering system.
    ///
    /// # Arguments
    ///
    /// * `index` - The number to expand.
    ///
    /// # Returns
    ///
    /// A vector of coefficients for each qudit in the system. Note that
    /// the coefficients are in big-endian order, that is, the first
    /// coefficient is for the most significant qudit.
    ///
    /// # Panics
    ///
    /// If `index` is too large for this system, that is, if it is greater
    /// than the product of the radices.
    ///
    /// # Examples
    ///
    /// ```
    /// use qudit_core::radices;
    /// use qudit_core::QuditRadices;
    ///
    /// let hybrid_system = radices![2, 3];
    /// assert_eq!(hybrid_system.expand(3), vec![1, 0]);
    ///
    /// let four_qubits = radices![2, 2, 2, 2];
    /// assert_eq!(four_qubits.expand(7), vec![0, 1, 1, 1]);
    ///
    /// let two_qutrits = radices![3, 3];
    /// assert_eq!(two_qutrits.expand(7), vec![2, 1]);
    ///
    /// let hybrid_system = radices![3, 2, 3];
    /// assert_eq!(hybrid_system.expand(17), vec![2, 1, 2]);
    /// ```
    ///
    /// # See Also
    ///
    /// * [`compress`] - The inverse of this function.
    /// * [`place_values`] - The place values for each position in the expansion.
    pub fn expand(&self, mut index: usize) -> Vec<u8> {
        if index >= self.dimension() {
            panic!(
                "Provided index {} is too large for this system with radices: {:#?}",
                index, self
            );
        }

        let mut expansion = vec![0u8; self.num_qudits()];

        for (idx, radix) in self.iter().enumerate().rev() {
            let casted_radix = *radix as usize;
            let coef = index % casted_radix;
            index = index - coef;
            index = index / casted_radix;
            expansion[idx] = coef as u8;
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
    /// use qudit_core::radices;
    /// use qudit_core::QuditRadices;
    ///
    /// let hybrid_system = radices![2, 3];
    /// assert_eq!(hybrid_system.compress(&vec![1, 0]), 3);
    ///
    /// let four_qubits = radices![2, 2, 2, 2];
    /// assert_eq!(four_qubits.compress(&vec![0, 1, 1, 1]), 7);
    ///
    /// let two_qutrits = radices![3, 3];
    /// assert_eq!(two_qutrits.compress(&vec![2, 1]), 7);
    ///
    /// let hybrid_system = radices![3, 2, 3];
    /// assert_eq!(hybrid_system.compress(&vec![2, 1, 2]), 17);
    /// ```
    ///
    /// # See Also
    ///
    /// * [`expand`] - The inverse of this function.
    /// * [`place_values`] - The place values for each position in the expansion.
    // TODO: use track_caller in user-facing code (any pub fn)
    pub fn compress<T: ToRadix>(&self, expansion: &[T]) -> usize {
        if self.len() != expansion.len() {
            panic!("Invalid expansion: incorrect number of qudits.")
        }
        assert_eq!(self.len(), expansion.len(), "msg here"); // TODO: Use Sarah's crate: equator

        if expansion
            .iter()
            .enumerate()
            .any(|(index, coef)| coef.to_radix() >= self[index])
        {
            panic!("Invalid expansion: mismatch in qudit radices.")
        }

        if expansion.len() == 0 {
            return 0;
        }

        let mut acm_val = expansion[self.num_qudits() - 1].to_radix() as usize;
        let mut acm_base = self[self.num_qudits() - 1] as usize;

        for coef_index in (0..expansion.len() - 1).rev() {
            let coef = expansion[coef_index];
            acm_val += (coef.to_radix() as usize) * acm_base;
            acm_base *= self[coef_index] as usize;
        }

        acm_val
    }

    /// Calculate the value for each expansion position in this numbering system.
    ///
    ///
    /// # Examples
    ///
    /// ```
    /// use qudit_core::radices;
    /// use qudit_core::QuditRadices;
    ///
    /// let two_qubits = radices![2, 2];
    /// assert_eq!(two_qubits.place_values(), vec![2, 1]);
    ///
    /// let two_qutrits = radices![3, 3];
    /// assert_eq!(two_qutrits.place_values(), vec![3, 1]);
    ///
    /// let hybrid_system = radices![3, 2, 3];
    /// assert_eq!(hybrid_system.place_values(), vec![6, 3, 1]);
    /// ```
    ///
    /// # See Also
    /// * [`expand`] - Expand an decimal value into this numbering system.
    /// * [`compress`] - Compress an expansion in this numbering system to decimal.
    pub fn place_values(&self) -> Vec<usize> {
        let mut place_values = vec![0; self.num_qudits()];
        let mut acm = 1;
        for (idx, r) in self.iter().enumerate().rev() {
            place_values[idx] = acm;
            acm *= *r as usize;
        }
        place_values
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
    /// use qudit_core::radices;
    /// use qudit_core::QuditRadices;
    ///
    /// let two_qubits = radices![2, 2];
    /// let two_qutrits = radices![3, 3];
    /// let four_qudits = radices![2, 2, 3, 3];
    /// assert_eq!(two_qubits.concat(&two_qutrits), four_qudits);
    ///
    /// let hybrid_system = radices![3, 2, 3];
    /// let two_qutrits = radices![3, 3];
    /// let five_qudits = radices![3, 2, 3, 3, 3];
    /// assert_eq!(hybrid_system.concat(&two_qutrits), five_qudits);
    /// ```
    #[inline(always)]
    pub fn concat(&self, other: &QuditRadices) -> QuditRadices {
        self.iter().chain(other.iter()).collect()
    }

    /// Returns the number of each radix in the system.
    ///
    /// # Examples
    ///
    /// ```
    /// use qudit_core::radices;
    /// use qudit_core::QuditRadices;
    /// use std::collections::HashMap;
    /// let two_qubits = radices![2, 2];
    /// let counts = two_qubits.counts();
    /// assert_eq!(counts.get(&2), Some(&2));
    ///
    /// let two_qutrits = radices![3, 3];
    /// let counts = two_qutrits.counts();
    /// assert_eq!(counts.get(&3), Some(&2));
    ///
    /// let hybrid_system = radices![3, 2, 3];
    /// let counts = hybrid_system.counts();
    /// let mut expected_counts = HashMap::new();
    /// expected_counts.insert(2, 1);
    /// expected_counts.insert(3, 2);
    /// assert_eq!(counts, expected_counts);
    /// ```
    pub fn counts(&self) -> HashMap<u8, usize> {
        let mut counts = HashMap::new();
        for radix in self.iter() {
            *counts.entry(*radix).or_insert(0) += 1;
        }
        counts
    }


    /// Returns the length of the radices object.
    pub fn len(&self) -> usize {
        self.num_qudits()
    }

    /// Returns true is the system is empty, false otherwise.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl crate::QuditSystem for QuditRadices {
    /// Returns the radices of the system.
    ///
    /// See [`QuditSystem`] for more information.
    #[inline(always)]
    fn radices(&self) -> QuditRadices {
        self.clone()
    }

    /// Returns the dimension of a system described by these radices.
    ///
    /// # Examples
    ///
    /// ```
    /// use qudit_core::radices;
    /// use qudit_core::QuditRadices;
    /// use qudit_core::QuditSystem;
    ///
    /// let two_qubits = radices![2, 2];
    /// assert_eq!(two_qubits.dimension(), 4);
    ///
    /// let two_qutrits = radices![3, 3];
    /// assert_eq!(two_qutrits.dimension(), 9);
    ///
    /// let hybrid_system = radices![3, 2, 3];
    /// assert_eq!(hybrid_system.dimension(), 18);
    /// ```
    #[inline(always)]
    fn dimension(&self) -> usize {
        self.iter().map(|&x| x as usize).product::<usize>()
    }

    /// Returns the number of qudits represented by these radices.
    ///
    /// # Examples
    ///
    /// ```
    /// use qudit_core::radices;
    /// use qudit_core::QuditRadices;
    /// use qudit_core::QuditSystem;
    ///
    /// let two_qubits = radices![2, 2];
    /// assert_eq!(two_qubits.num_qudits(), 2);
    ///
    /// let two_qutrits = radices![3, 3];
    /// assert_eq!(two_qutrits.num_qudits(), 2);
    ///
    /// let hybrid_system = radices![3, 2, 3];
    /// assert_eq!(hybrid_system.num_qudits(), 3);
    ///
    /// let ten_qubits = radices![2; 10];
    /// assert_eq!(ten_qubits.num_qudits(), 10);
    /// ```
    #[inline(always)]
    fn num_qudits(&self) -> usize {
        match &self {
            QuditRadices::Array(len, _) => *len as usize,
            QuditRadices::Heap(vec) => vec.len(),
        }
    }

    /// Returns true if these radices describe a qubit-only system.
    ///
    /// # Examples
    ///
    /// ```
    /// use qudit_core::radices;
    /// use qudit_core::QuditRadices;
    /// use qudit_core::QuditSystem;
    ///
    /// let two_qubits = radices![2, 2];
    /// assert!(two_qubits.is_qubit_only());
    ///
    /// let qubit_qutrit = radices![2, 3];
    /// assert!(!qubit_qutrit.is_qubit_only());
    ///
    /// let two_qutrits = radices![3, 3];
    /// assert!(!two_qutrits.is_qubit_only());
    /// ```
    #[inline(always)]
    fn is_qubit_only(&self) -> bool {
        self.iter().all(|r| *r == 2)
    }

    /// Returns true if these radices describe a qutrit-only system.
    ///
    /// # Examples
    ///
    /// ```
    /// use qudit_core::radices;
    /// use qudit_core::QuditRadices;
    /// use qudit_core::QuditSystem;
    ///
    /// let two_qubits = radices![2, 2];
    /// assert!(!two_qubits.is_qutrit_only());
    ///
    /// let qubit_qutrit = radices![2, 3];
    /// assert!(!qubit_qutrit.is_qutrit_only());
    ///
    /// let two_qutrits = radices![3, 3];
    /// assert!(two_qutrits.is_qutrit_only());
    /// ```
    #[inline(always)]
    fn is_qutrit_only(&self) -> bool {
        self.iter().all(|r| *r == 3)
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
    /// use qudit_core::radices;
    /// use qudit_core::QuditRadices;
    /// use qudit_core::QuditSystem;
    ///
    /// let two_qubits = radices![2, 2];
    /// assert!(two_qubits.is_qudit_only(2));
    /// assert!(!two_qubits.is_qudit_only(3));
    ///
    /// let mixed_qudits = radices![2, 3];
    /// assert!(!mixed_qudits.is_qudit_only(2));
    /// assert!(!mixed_qudits.is_qudit_only(3));
    /// ```
    #[inline(always)]
    fn is_qudit_only<T: ToRadix>(&self, radix: T) -> bool {
        self.iter().all(|r| *r == radix.to_radix())
    }

    /// Returns true if these radices describe a homogenous system.
    ///
    /// # Examples
    ///
    /// ```
    /// use qudit_core::radices;
    /// use qudit_core::QuditRadices;
    /// use qudit_core::QuditSystem;
    ///
    /// let two_qubits = radices![2, 2];
    /// assert!(two_qubits.is_homogenous());
    ///
    /// let mixed_qudits = radices![2, 3];
    /// assert!(!mixed_qudits.is_homogenous());
    /// ```
    #[inline(always)]
    fn is_homogenous(&self) -> bool {
        self.is_qudit_only(self[0])
    }
}

impl<T: ToRadix> core::iter::FromIterator<T> for QuditRadices {
    /// Creates a new QuditRadices object from an iterator.
    ///
    /// # Arguments
    ///
    /// * `iter` - An iterator over the radices of a qudit system.
    ///
    /// # Panics
    ///
    /// If radices does not represent a valid system. This can happen
    /// if any of the radices are 0 or 1.
    ///
    /// # Examples
    ///
    /// ```
    /// use qudit_core::radices;
    /// use qudit_core::QuditRadices;
    ///
    /// let qubit_qutrit = QuditRadices::from_iter(2..4);
    /// assert_eq!(qubit_qutrit, radices![2, 3]);
    ///
    /// let two_qutrits = QuditRadices::from_iter(vec![3, 3]);
    /// assert_eq!(two_qutrits, radices![3, 3]);
    ///
    /// // Ten qubits then ten qutrits
    /// let mixed_system = QuditRadices::from_iter(vec![2; 10].iter()
    ///                         .chain(vec![3; 10].iter()));
    /// ```
    ///
    /// # Note
    ///
    /// This will attempt to avoid an allocation when possible.
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        let iter = iter.into_iter();
        let size_hint = iter.size_hint().0;

        if size_hint > RADICES_INLINE_SIZE {
            let mut vec = Vec::with_capacity(size_hint);
            for (i, r) in iter.enumerate() {
                if r.is_less_than_two() {
                    panic!(
                        "Qudit radix in position {} is invalid since it is < 2: {}",
                        i, r
                    );
                }
                vec.push(r.to_radix());
            }

            // The size hint may not be correct; however, in this case,
            // returning a heap-allocated vector is still valid, but may
            // not be the most efficient. Since we have already allocated
            // the vector, we might as well use it.
            return QuditRadices::Heap(vec);
        }

        // Attempt to drain the iterator into an inline array without allocation.
        let mut array = [0u8; RADICES_INLINE_SIZE];
        let mut len = 0;
        let mut enum_iter = iter.enumerate();
        let mut overflow_option = None;

        while let Some((i, r)) = enum_iter.next() {
            if i == RADICES_INLINE_SIZE {
                overflow_option = Some(enum_iter);
                break;
            }

            if r.is_less_than_two() {
                panic!(
                    "Qudit radix in position {} is invalid since it is < 2: {}",
                    i, r
                );
            }

            array[i] = r.to_radix();
            len += 1;
        }

        // If we spill over into the heap, we need to allocate a new
        // vector and copy the data into it.
        if let Some(overflow_iter) = overflow_option {
            let mut vec: Vec<u8> = Vec::with_capacity(RADICES_INLINE_SIZE * 2);
            vec.extend_from_slice(&array);
            for (i, r) in overflow_iter {
                if r.is_less_than_two() {
                    panic!(
                        "Qudit radix in position {} is invalid since it is < 2: {}",
                        i, r
                    );
                }
                vec.push(r.to_radix());
            }
            return QuditRadices::Heap(vec);
        }

        QuditRadices::Array(len, array)
    }
}

impl core::ops::Deref for QuditRadices {
    type Target = [u8];

    #[inline(always)]
    fn deref(&self) -> &[u8] {
        match self {
            QuditRadices::Array(len, array) => &array[0..*len as usize],
            QuditRadices::Heap(vec) => &vec,
        }
    }
}

impl core::ops::Add for QuditRadices {
    type Output = QuditRadices;

    #[inline(always)]
    fn add(self, other: QuditRadices) -> QuditRadices {
        self.concat(&other)
    }
}

impl<'a, 'b> core::ops::Add<&'b QuditRadices> for &'a QuditRadices {
    type Output = QuditRadices;

    #[inline(always)]
    fn add(self, other: &'b QuditRadices) -> QuditRadices {
        self.concat(other)
    }
}

impl<T: ToRadix> From<Vec<T>> for QuditRadices {
    #[inline(always)]
    fn from(radices: Vec<T>) -> QuditRadices {
        QuditRadices::new(&radices)
    }
}

impl<T: ToRadix> From<&[T]> for QuditRadices {
    #[inline(always)]
    fn from(radices: &[T]) -> QuditRadices {
        QuditRadices::new(radices)
    }
}

impl core::fmt::Debug for QuditRadices {
    /// Formats the radices as a string.
    ///
    /// See Display for more information.
    #[inline(always)]
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> std::fmt::Result {
        <QuditRadices as core::fmt::Display>::fmt(self, f)
    }
}

impl core::fmt::Display for QuditRadices {
    /// Formats the radices as a string.
    ///
    /// # Examples
    ///
    /// ```
    /// use qudit_core::radices;
    /// use qudit_core::QuditRadices;
    /// let two_qubits = radices![2, 2];
    /// let two_qutrits = radices![3, 3];
    /// let qubit_qutrit = radices![2, 3];
    ///
    /// assert_eq!(format!("{}", two_qubits), "[2, 2]");
    /// assert_eq!(format!("{}", two_qutrits), "[3, 3]");
    /// assert_eq!(format!("{}", qubit_qutrit), "[2, 3]");
    /// ```
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[")?;
        let enum_iter = self.iter().enumerate();
        for (i, r) in enum_iter {
            if i == self.len() - 1 {
                write!(f, "{}", r)?;
            } else {
                write!(f, "{}, ", r)?;
            }
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
/// use qudit_core::radices;
/// use qudit_core::QuditRadices;
/// use qudit_core::QuditSystem;
///
/// // Similar to the vec! macro, we can use a comma separated list:
/// let two_qubits = radices![2, 2];
/// assert_eq!(two_qubits.dimension(), 4);
/// assert_eq!(two_qubits, QuditRadices::new(&vec![2, 2]));
///
/// let two_qutrits = radices![3, 3];
/// assert_eq!(two_qutrits.dimension(), 9);
/// assert_eq!(two_qutrits, QuditRadices::new(&vec![3, 3]));
///
/// let hybrid_system = radices![3, 2, 3];
/// assert_eq!(hybrid_system.dimension(), 18);
/// assert_eq!(hybrid_system, QuditRadices::new(&vec![3, 2, 3]));
///
/// // We can also use a single radix and a count to create a radices object:
/// let ten_qubits = radices![2; 10]; // 2 qubits repeated 10 times
/// assert_eq!(ten_qubits.dimension(), 1024);
/// assert_eq!(ten_qubits, QuditRadices::new(&vec![2; 10]));
/// ```
#[macro_export]
macro_rules! radices {
    ($($e:expr),*) => {
        QuditRadices::new(&vec![$($e),*])
    };
    ($elem:expr; $n:expr) => {
        QuditRadices::new(&vec![$elem; $n])
    };
}

/// An iterable that can be converted into a QuditRadices object.
pub trait ToRadices {
    /// Convert the iterable into a QuditRadices object.
    fn to_radices(self) -> QuditRadices;
}

impl ToRadices for QuditRadices {
    #[inline(always)]
    fn to_radices(self) -> QuditRadices {
        self
    }
}

impl<'a> ToRadices for &'a QuditRadices {
    #[inline(always)]
    fn to_radices(self) -> QuditRadices {
        self.clone()
    }
}

impl<T: ToRadix, I: IntoIterator<Item = T>> ToRadices for I {
    #[inline(always)]
    fn to_radices(self) -> QuditRadices {
        QuditRadices::from_iter(self)
    }
}

impl<T: FromRadix> From<QuditRadices> for Vec<T> {
    #[inline(always)]
    fn from(value: QuditRadices) -> Self {
        match value {
            QuditRadices::Array(len, array) => array[0..len as usize].into_iter().map(|r| T::from_radix(*r)).collect(),
            QuditRadices::Heap(vec) => vec.into_iter().map(|r| T::from_radix(r)).collect(),
        }
    }
}

#[cfg(test)]
pub mod strategies {
    use proptest::prelude::*;

    use super::*;

    impl Arbitrary for QuditRadices {
        type Parameters = (core::ops::Range<u8>, core::ops::Range<usize>);
        type Strategy = BoxedStrategy<Self>;

        /// Generate a random QuditRadices object.
        ///
        /// By default, the number of radices is chosen randomly between
        /// 1 and 4, and the radices themselves are chosen randomly
        /// between 2 and 4.
        fn arbitrary() -> Self::Strategy {
            Self::arbitrary_with((2..5u8, 1..5))
        }

        /// Generate a random QuditRadices object with the given parameters.
        ///
        /// # Arguments
        ///
        /// * `args` - A tuple of ranges for the number of radices and the
        ///           radices themselves. The first range is for the number
        ///           of radices, and the second range is for the radices
        ///           themselves.
        fn arbitrary_with(args: Self::Parameters) -> Self::Strategy {
            prop::collection::vec(args.0, args.1)
                .prop_map(|v| QuditRadices::new(&v))
                .boxed()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        /// An expansion should compress to the same value.
        #[test]
        fn test_expand_compress(radices in any::<QuditRadices>()) {
            for index in 0..radices.dimension() {
                let exp = radices.expand(index);
                assert_eq!(radices.compress(&exp), index);
            }
        }
    }

    #[test]
    fn test_slice_ops() {
        let rdx = QuditRadices::new(&vec![2, 3, 4usize]);
        assert_eq!(rdx.len(), 3);
        assert_eq!(rdx[1], 3);
        assert_eq!(rdx[1..], [3, 4]);
        assert_eq!(rdx.clone(), rdx);
    }
}
