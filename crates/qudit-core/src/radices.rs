// TODO: remove all mentions of the old macro radices!
// TODO: Order mods: implementations; python; strategies; test
use std::collections::HashMap;

use crate::radix::Radix;
use crate::{CompactStorage, CompactVec, QuditSystem};

/// The number of basis states for each qudit (or dit) in a quantum system.
///
/// This object represents the radix -- sometimes called the base, level
/// or ditness -- of each qudit in a qudit system or dit in a classical system,
/// and is implemented as a sequence of unsigned, byte-sized integers.
/// A qubit is a two-level qudit or a qudit with radix two, while a
/// qutrit is a three-level qudit. Two qutrits together are represented
/// by the [3, 3] radices object.
///
/// No radix can be less than 2, as this would not be a valid.
/// As each radix is represented by a byte, this implementation does not
/// support individual radices greater than 255.
///
/// ## Ordering and Endianness
///
/// Indices are counted left to right, for example a [2, 3] qudit
/// radices is interpreted as a qubit as the first qudit and a qutrit as
/// the second one. OpenQudit uses big-endian ordering, so the qubit in
/// the previous example is the most significant qudit and the qutrit is
/// the least significant qudit. For example, in the same system, a state
/// |10> would be represented by the decimal number 3.
#[derive(Hash, PartialEq, Eq, Clone)]
pub struct Radices(CompactVec<Radix>);

impl Radices {
    /// Constructs a Radices object from the given input.
    ///
    /// # Arguments
    ///
    /// * `radices` - A slice detailing the radices of a qudit system.
    ///
    /// # Panics
    ///
    /// If radices does not represent a valid system. This can happen
    /// if any of the radices are 0 or 1.
    ///
    /// # Examples
    ///
    /// ```
    /// # use qudit_core::Radices;
    /// let three_qubits = Radices::new([2; 3]);
    /// let qubit_qutrit = Radices::new([2, 3]);
    /// ```
    pub fn new<T: Into<Radices>>(radices: T) -> Self {
        radices.into()
    }

    /// Attempts to determine the radices for a qudit system with given dimension.
    ///
    /// It first tries powers of two, failing that powers of three. If neither
    /// then it will return a QuditRadices with only one qudit of dimension
    /// equal to the input.
    pub fn guess(dimension: usize) -> Radices {
        if dimension < 2 {
            panic!("Invalid dimension in Radices");
        }

        // Is a power of two?
        if dimension & (dimension - 1) == 0 {
            let num_qudits = dimension.trailing_zeros() as usize;
            return vec![2; num_qudits].into();
        }

        let mut n = dimension;
        let mut power = 0;
        while n > 1 {
            if n % 3 != 0 {
                return [dimension].into();
            }
            n /= 3;
            power += 1;
        }
        return vec![3; power].into();
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
    /// # use qudit_core::Radices;
    ///
    /// let hybrid_system = Radices::new([2, 3]);
    /// assert_eq!(hybrid_system.expand(3), vec![1, 0]);
    ///
    /// let four_qubits = Radices::new([2, 2, 2, 2]);
    /// assert_eq!(four_qubits.expand(7), vec![0, 1, 1, 1]);
    ///
    /// let two_qutrits = Radices::new([3, 3]);
    /// assert_eq!(two_qutrits.expand(7), vec![2, 1]);
    ///
    /// let hybrid_system = Radices::new([3, 2, 3]);
    /// assert_eq!(hybrid_system.expand(17), vec![2, 1, 2]);
    /// ```
    ///
    /// # See Also
    ///
    /// * [`compress`] - The inverse of this function.
    /// * [`place_values`] - The place values for each position in the expansion.
    #[track_caller]
    pub fn expand(&self, mut index: usize) -> Vec<usize> {
        if index >= self.dimension() {
            panic!(
                "Provided index {} is too large for this system with radices: {:#?}",
                index, self
            );
        }

        let mut expansion = vec![0; self.num_qudits()];

        for (idx, radix) in self.iter().enumerate().rev() {
            let casted_radix: usize = (*radix).into();
            let coef: usize = index % casted_radix;
            index = index - coef;
            index = index / casted_radix;
            expansion[idx] = coef;
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
    /// # use qudit_core::Radices;
    ///
    /// let hybrid_system = Radices::new([2, 3]);
    /// assert_eq!(hybrid_system.compress(&vec![1, 0]), 3);
    ///
    /// let four_qubits = Radices::new([2, 2, 2, 2]);
    /// assert_eq!(four_qubits.compress(&vec![0, 1, 1, 1]), 7);
    ///
    /// let two_qutrits = Radices::new([3, 3]);
    /// assert_eq!(two_qutrits.compress(&vec![2, 1]), 7);
    ///
    /// let hybrid_system = Radices::new([3, 2, 3]);
    /// assert_eq!(hybrid_system.compress(&vec![2, 1, 2]), 17);
    /// ```
    ///
    /// # See Also
    ///
    /// * [`expand`] - The inverse of this function.
    /// * [`place_values`] - The place values for each position in the expansion.
    #[track_caller]
    pub fn compress(&self, expansion: &[usize]) -> usize {
        if self.len() != expansion.len() {
            panic!("Invalid expansion: incorrect number of qudits.")
        }

        if expansion
            .iter()
            .enumerate()
            .any(|(index, coef)| *coef >= usize::from(self[index]))
        {
            panic!("Invalid expansion: mismatch in qudit radices.")
        }

        if expansion.len() == 0 {
            return 0;
        }

        let mut acm_val = expansion[self.num_qudits() - 1];
        let mut acm_base = usize::from(self[self.num_qudits() - 1]);

        for coef_index in (0..expansion.len() - 1).rev() {
            let coef = expansion[coef_index];
            acm_val += coef * acm_base;
            acm_base *= usize::from(self[coef_index]);
        }

        acm_val
    }

    /// Calculate the value for each expansion position in this numbering system.
    ///
    /// # Examples
    ///
    /// ```
    /// # use qudit_core::Radices;
    ///
    /// let two_qubits = Radices::new([2, 2]);
    /// assert_eq!(two_qubits.place_values(), vec![2, 1]);
    ///
    /// let two_qutrits = Radices::new([3, 3]);
    /// assert_eq!(two_qutrits.place_values(), vec![3, 1]);
    ///
    /// let hybrid_system = Radices::new([3, 2, 3]);
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
            acm *= usize::from(*r);
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
    /// # use qudit_core::Radices;
    ///
    /// let two_qubits = Radices::new([2, 2]);
    /// let two_qutrits = Radices::new([3, 3]);
    /// let four_qudits = Radices::new([2, 2, 3, 3]);
    /// assert_eq!(two_qubits.concat(&two_qutrits), four_qudits);
    ///
    /// let hybrid_system = Radices::new([3, 2, 3]);
    /// let two_qutrits = Radices::new([3, 3]);
    /// let five_qudits = Radices::new([3, 2, 3, 3, 3]);
    /// assert_eq!(hybrid_system.concat(&two_qutrits), five_qudits);
    /// ```
    #[inline(always)]
    pub fn concat(&self, other: &Radices) -> Radices {
        self.iter().chain(other.iter()).copied().collect()
    }

    /// Returns the number of each radix in the system.
    ///
    /// # Examples
    ///
    /// ```
    /// # use qudit_core::Radices;
    /// # use std::collections::HashMap;
    /// let two_qubits = Radices::new([2, 2]);
    /// let counts = two_qubits.counts();
    /// assert_eq!(counts.get(&(2.into())), Some(&2));
    ///
    /// let two_qutrits = Radices::new([3, 3]);
    /// let counts = two_qutrits.counts();
    /// assert_eq!(counts.get(&(3.into())), Some(&2));
    ///
    /// let hybrid_system = Radices::new([3, 2, 3]);
    /// let counts = hybrid_system.counts();
    /// assert_eq!(counts.get(&(3.into())), Some(&2));
    /// assert_eq!(counts.get(&(2.into())), Some(&1));
    /// ```
    pub fn counts(&self) -> HashMap<Radix, usize> {
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

impl crate::QuditSystem for Radices {
    /// Returns the radices of the system.
    ///
    /// See [`QuditSystem`] for more information.
    #[inline(always)]
    fn radices(&self) -> Radices {
        self.clone()
    }

    /// Returns the dimension of a system described by these radices.
    ///
    /// # Examples
    ///
    /// ```
    /// # use qudit_core::Radices;
    /// # use qudit_core::QuditSystem;
    ///
    /// let two_qubits = Radices::new([2, 2]);
    /// assert_eq!(two_qubits.dimension(), 4);
    ///
    /// let two_qutrits = Radices::new([3, 3]);
    /// assert_eq!(two_qutrits.dimension(), 9);
    ///
    /// let hybrid_system = Radices::new([3, 2, 3]);
    /// assert_eq!(hybrid_system.dimension(), 18);
    /// ```
    #[inline(always)]
    fn dimension(&self) -> usize {
        self.iter().product::<usize>()
    }

    /// Returns the number of qudits represented by these radices.
    ///
    /// # Examples
    ///
    /// ```
    /// # use qudit_core::Radices;
    /// # use qudit_core::QuditSystem;
    ///
    /// let two_qubits = Radices::new([2, 2]);
    /// assert_eq!(two_qubits.num_qudits(), 2);
    ///
    /// let two_qutrits = Radices::new([3, 3]);
    /// assert_eq!(two_qutrits.num_qudits(), 2);
    ///
    /// let hybrid_system = Radices::new([3, 2, 3]);
    /// assert_eq!(hybrid_system.num_qudits(), 3);
    ///
    /// let ten_qubits = Radices::new(vec![2; 10]);
    /// assert_eq!(ten_qubits.num_qudits(), 10);
    /// ```
    #[inline(always)]
    fn num_qudits(&self) -> usize {
        self.0.len()
    }

    /// Returns true if these radices describe a qubit-only system.
    ///
    /// # Examples
    ///
    /// ```
    /// # use qudit_core::Radices;
    /// # use qudit_core::QuditSystem;
    ///
    /// let two_qubits = Radices::new([2, 2]);
    /// assert!(two_qubits.is_qubit_only());
    ///
    /// let qubit_qutrit = Radices::new([2, 3]);
    /// assert!(!qubit_qutrit.is_qubit_only());
    ///
    /// let two_qutrits = Radices::new([3, 3]);
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
    /// # use qudit_core::Radices;
    /// # use qudit_core::QuditSystem;
    ///
    /// let two_qubits = Radices::new([2, 2]);
    /// assert!(!two_qubits.is_qutrit_only());
    ///
    /// let qubit_qutrit = Radices::new([2, 3]);
    /// assert!(!qubit_qutrit.is_qutrit_only());
    ///
    /// let two_qutrits = Radices::new([3, 3]);
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
    /// # use qudit_core::Radices;
    /// # use qudit_core::QuditSystem;
    ///
    /// let two_qubits = Radices::new([2, 2]);
    /// assert!(two_qubits.is_qudit_only(2));
    /// assert!(!two_qubits.is_qudit_only(3));
    ///
    /// let mixed_qudits = Radices::new([2, 3]);
    /// assert!(!mixed_qudits.is_qudit_only(2));
    /// assert!(!mixed_qudits.is_qudit_only(3));
    /// ```
    #[inline(always)]
    fn is_qudit_only<T: Into<Radix>>(&self, radix: T) -> bool {
        let radix = radix.into();
        self.iter().all(|r| *r == radix)
    }

    /// Returns true if these radices describe a homogenous system.
    ///
    /// # Examples
    ///
    /// ```
    /// # use qudit_core::Radices;
    /// # use qudit_core::QuditSystem;
    ///
    /// let two_qubits = Radices::new([2, 2]);
    /// assert!(two_qubits.is_homogenous());
    ///
    /// let mixed_qudits = Radices::new([2, 3]);
    /// assert!(!mixed_qudits.is_homogenous());
    /// ```
    #[inline(always)]
    fn is_homogenous(&self) -> bool {
        self.is_qudit_only(self[0])
    }
}

impl<T: Into<Radix>> core::iter::FromIterator<T> for Radices {
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
    /// # use qudit_core::Radices;
    ///
    /// let qubit_qutrit = Radices::from_iter(2..4);
    /// assert_eq!(qubit_qutrit, Radices::new([2, 3]));
    ///
    /// let two_qutrits = Radices::from_iter(vec![3, 3]);
    /// assert_eq!(two_qutrits, Radices::new([3, 3]));
    ///
    /// // Ten qubits then ten qutrits
    /// let mixed_system = Radices::from_iter(vec![2; 10].into_iter()
    ///                         .chain(vec![3; 10].into_iter()));
    ///
    /// // Using .collect()
    /// let from_collect: Radices = (2..5).collect();
    /// assert_eq!(from_collect, Radices::new([2, 3, 4]));
    /// ```
    ///
    /// # Note
    ///
    /// This will attempt to avoid an allocation when possible.
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        let vec: CompactVec<Radix> = iter.into_iter().map(|r| r.into()).collect();
        Radices(vec)
    }
}

impl core::ops::Deref for Radices {
    type Target = [Radix];

    #[inline(always)]
    fn deref(&self) -> &[Radix] {
        // Safety Radix is a transparent wrapper around u8
        unsafe { std::mem::transmute(std::mem::transmute::<_, &CompactVec<u8>>(&self.0).deref()) }
    }
}

impl<T: Into<Radix> + CompactStorage> From<CompactVec<T>> for Radices {
    #[inline(always)]
    fn from(value: CompactVec<T>) -> Self {
        value.into_iter().map(|x| x.into()).collect()
    }
}

impl<T: Into<Radix> + 'static> From<Vec<T>> for Radices {
    #[inline(always)]
    fn from(value: Vec<T>) -> Self {
        let vec = if std::any::TypeId::of::<T>() == std::any::TypeId::of::<Radix>() {
            // If T is radix, then doing CompactVec::from will provide move semantics efficiently
            // Safety: T == Radix, so Vec<T> == Vec<Radix>
            CompactVec::<Radix>::from(unsafe { std::mem::transmute::<_, Vec<Radix>>(value) })
        } else {
            value.into_iter().map(|x| x.into()).collect()
        };

        Radices(vec)
    }
}

impl<T: Into<Radix> + Copy> From<&[T]> for Radices {
    #[inline(always)]
    fn from(value: &[T]) -> Self {
        value.iter().copied().collect()
    }
}

impl<T: Into<Radix>, const N: usize> From<[T; N]> for Radices {
    #[inline(always)]
    fn from(value: [T; N]) -> Self {
        value.into_iter().collect()
    }
}

impl<'a, T: Into<Radix> + Copy, const N: usize> From<&'a [T; N]> for Radices {
    #[inline(always)]
    fn from(value: &'a [T; N]) -> Self {
        value.iter().copied().collect()
    }
}

impl From<Radix> for Radices {
    fn from(value: Radix) -> Self {
        [value].into()
    }
}

impl core::fmt::Debug for Radices {
    /// Formats the radices as a string.
    ///
    /// See Display for more information.
    #[inline(always)]
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> std::fmt::Result {
        <Radices as core::fmt::Display>::fmt(self, f)
    }
}

impl core::fmt::Display for Radices {
    /// Formats the radices as a string.
    ///
    /// # Examples
    ///
    /// ```
    /// # use qudit_core::Radices;
    /// let two_qubits = Radices::new([2, 2]);
    /// let two_qutrits = Radices::new([3, 3]);
    /// let qubit_qutrit = Radices::new([2, 3]);
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

#[cfg(test)]
pub mod strategies {
    use proptest::prelude::*;

    use super::*;

    impl Arbitrary for Radices {
        type Parameters = (core::ops::Range<u8>, core::ops::Range<usize>);
        type Strategy = BoxedStrategy<Self>;

        /// Generate a random Radices object.
        ///
        /// By default, the number of radices is chosen randomly between
        /// 1 and 4, and the radices themselves are chosen randomly
        /// between 2 and 4.
        fn arbitrary() -> Self::Strategy {
            Self::arbitrary_with((2..5u8, 1..5))
        }

        /// Generate a random Radices object with the given parameters.
        ///
        /// # Arguments
        ///
        /// * `args` - A tuple of ranges for the number of radices and the
        ///           radices themselves. The first range is for the number
        ///           of radices, and the second range is for the radices
        ///           themselves.
        fn arbitrary_with(args: Self::Parameters) -> Self::Strategy {
            prop::collection::vec(args.0, args.1)
                .prop_map(|v| Radices::new(v))
                .boxed()
        }
    }
}

#[cfg(feature = "python")]
mod python {
    use super::*;
    use pyo3::{exceptions::PyTypeError, prelude::*};

    impl<'py> FromPyObject<'py> for Radices {
        // const INPUT_TYPE: &'static str = "int | typing.Iterable[int]";

        fn extract_bound(ob: &Bound<'py, PyAny>) -> PyResult<Self> {
            if let Ok(num) = ob.extract::<usize>() {
                Ok(Radices::new(&[num]))
            } else if let Ok(nums) = ob.extract::<Vec<usize>>() {
                Ok(Radices::new(nums))
            } else {
                Err(PyTypeError::new_err(
                    "Expected a list of integers for radices.",
                ))
            }
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
        fn test_expand_compress(radices in any::<Radices>()) {
            for index in 0..radices.dimension() {
                let exp = radices.expand(index);
                assert_eq!(radices.compress(&exp), index);
            }
        }
    }

    #[test]
    fn test_slice_ops() {
        let rdx = Radices::new(vec![2, 3, 4usize]);
        assert_eq!(rdx.len(), 3);
        assert_eq!(rdx[1], 3);
        assert_eq!(rdx[1..], [3, 4]);
        assert_eq!(rdx.clone(), rdx);
    }
}
