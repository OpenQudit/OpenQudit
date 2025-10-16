use crate::{utils::CompactVec, wire::Wire};
use std::collections::HashSet;

/// A WireList describes where an instruction is spatially executed.
///
/// In other words, it describes an unnamed register of circuit wires (both quantum and classical).
/// This is commonly used to indicate "where" an instruction acts. Semantically,
/// this represents two separate lists for quantum and classical wires, so two wire
/// lists are equal if their quantum and classical sub-lists are equal separately.
/// 
/// Order matters. For example, consider the wire list with wires = [q0, q2] 
/// representing the two-qudit purely-quantum register of the first
/// and third qudits in the circuit. This wire list is not equivalent to the
/// wire list with wires = [q2, q0], as this describes a permutation of the former.
#[derive(Clone)]
pub struct WireList {
    /// The wires in the circuit, stored in a space-efficient vector.
    inner: CompactVec<Wire>,
}

impl WireList {
    #[cold]
    #[inline(never)]
    fn panic_duplicate_quantum() -> ! {
        panic!("Duplicate indices in quantum wires.");
    }

    #[cold]
    #[inline(never)]
    fn panic_duplicate_classical() -> ! {
        panic!("Duplicate indices in classical wires.");
    }

    #[cold]
    #[inline(never)]
    fn panic_duplicate_wires() -> ! {
        panic!("Duplicate wires in wire list.");
    }

    /// Helper function to check for duplicate indices in quantum and classical wire collections.
    fn check_uniqueness(quantum_indices: &[usize], classical_indices: &[usize]) {
        // Performance: For small collections, O(N^2) check is faster than HashSet
        if quantum_indices.len() < 20 && classical_indices.len() < 20 {
            for i in 0..quantum_indices.len() {
                for j in (i + 1)..quantum_indices.len() {
                    if quantum_indices[i] == quantum_indices[j] {
                        Self::panic_duplicate_quantum();
                    }
                }
            }

            for i in 0..classical_indices.len() {
                for j in (i + 1)..classical_indices.len() {
                    if classical_indices[i] == classical_indices[j] {
                        Self::panic_duplicate_classical();
                    }
                }
            }
        } else {
            let mut uniq = HashSet::new();
            if !quantum_indices.iter().all(|x| uniq.insert(x)) {
                Self::panic_duplicate_quantum();
            }
            uniq.clear();
            if !classical_indices.iter().all(|x| uniq.insert(x)) {
                Self::panic_duplicate_classical();
            }
        }
    }

    /// Helper function to check for duplicate wires.
    #[cold]
    fn check_wire_uniqueness(wires: &[Wire]) {
        if wires.len() < 20 {
            for i in 0..wires.len() {
                for j in (i + 1)..wires.len() {
                    if wires[i] == wires[j] {
                        Self::panic_duplicate_wires();
                    }
                }
            }
        } else {
            let mut uniq = HashSet::new();
            if !wires.iter().all(|x| uniq.insert(x)) {
                Self::panic_duplicate_wires();
            }
        }
    }

    /// Returns a purely-quantum WireList object from the given vector.
    ///
    /// A purely-quantum wire list is one that does not contain any classical wires.
    ///
    /// # Arguments
    ///
    /// * `indices` - A collection describing the quantum wire indices in a circuit.
    ///
    /// # Panics
    ///
    /// If `indices` contains duplicate indices.
    ///
    /// # Examples
    ///
    /// ```
    /// # use qudit_circuit::WireList;
    /// let wires = WireList::pure(&vec![3, 0]);
    /// ```
    #[inline]
    pub fn pure<T: AsRef<[usize]>>(indices: T) -> WireList {
        let indices = indices.as_ref();
        Self::check_uniqueness(indices, &[]);
        
        let mut vec: CompactVec<Wire> = CompactVec::new();
        for &idx in indices {
            vec.push(Wire::quantum(idx));
        }
        
        WireList {
            inner: vec
        }
    }

    /// Returns a purely-classical WireList object from the given vector.
    ///
    /// A purely-classical wire list is one that does not contain any quantum wires.
    ///
    /// # Arguments
    ///
    /// * `indices` - A collection describing the classical wire indices in a circuit.
    ///
    /// # Panics
    ///
    /// If `indices` contains duplicate indices.
    ///
    /// # Examples
    ///
    /// ```
    /// # use qudit_circuit::WireList;
    /// let wires = WireList::classical(&vec![1, 2]);
    /// ```
    #[inline]
    pub fn classical<T: AsRef<[usize]>>(indices: T) -> WireList {
        let indices = indices.as_ref();
        Self::check_uniqueness(&[], indices);
        
        let mut vec: CompactVec<Wire> = CompactVec::new();
        for &idx in indices {
            vec.push(Wire::classical(idx));
        }
        
        WireList {
            inner: vec
        }
    }

    /// Returns a WireList object from the given vectors of quantum and classical indices.
    ///
    /// # Arguments
    ///
    /// * `qudits` - A collection describing the quantum wire indices in a circuit.
    /// * `dits` - A collection describing the classical wire indices in a circuit.
    ///
    /// # Panics
    ///
    /// If `qudits` or `dits` contain duplicate indices.
    ///
    /// # Examples
    ///
    /// ```
    /// # use qudit_circuit::WireList;
    /// let wires = WireList::new(&vec![3, 0], &vec![1, 2]);
    /// ```
    #[inline]
    pub fn new<S: AsRef<[usize]>, T: AsRef<[usize]>>(qudits: S, dits: T) -> WireList {
        let quantum_indices = qudits.as_ref();
        let classical_indices = dits.as_ref();
        Self::check_uniqueness(quantum_indices, classical_indices);
        
        let mut vec: CompactVec<Wire> = CompactVec::new();
        for &idx in quantum_indices {
            vec.push(Wire::quantum(idx));
        }
        for &idx in classical_indices {
            vec.push(Wire::classical(idx));
        }

        WireList {
            inner: vec,
        }
    }

    /// Creates a WireList directly from a collection of wires.
    ///
    /// # Arguments
    ///
    /// * `wires` - A collection of Wire objects.
    ///
    /// # Panics
    ///
    /// If there are duplicate wires in the collection.
    ///
    /// # Examples
    ///
    /// ```
    /// # use qudit_circuit::{Wire, WireList};
    /// let wire_vec = vec![Wire::quantum(0), Wire::classical(1), Wire::quantum(2)];
    /// let wires = WireList::from_wires(wire_vec);
    /// ```
    pub fn from_wires<T: AsRef<[Wire]>>(wires: T) -> WireList {
        let wires = wires.as_ref();
        Self::check_wire_uniqueness(wires);
        WireList {
            inner: CompactVec::from(wires),
        }
    }

    /// Creates a `WireList` from its raw parts.
    ///
    /// # Safety
    ///
    /// The caller must ensure that the provided `inner` `CompactVec<Wire>` represents a valid
    /// wire list with no duplicate or null wires. Failing to do so can lead to unexpected behavior
    /// or panics in other `WireList` methods that assume uniqueness.
    pub unsafe fn from_raw_inner(inner: CompactVec<Wire>) -> Self {
        WireList { inner }
    }

    /// Get all wires in this list as a slice.
    ///
    /// # Examples
    ///
    /// ```
    /// # use qudit_circuit::{Wire, WireList};
    /// let wires = WireList::pure(vec![3, 0]);
    /// let wire_slice = wires.wires();
    /// ```
    #[inline]
    pub fn wires(&self) -> <&CompactVec<Wire> as IntoIterator>::IntoIter {
        self.inner.iter()
    }

    /// Get the quantum wire indices in this list.
    ///
    /// # Examples
    ///
    /// ```
    /// # use qudit_circuit::WireList;
    /// let wires = WireList::new(vec![3, 0], vec![1, 2]);
    /// let qudits: Vec<usize> = wires.qudits().collect();
    /// assert_eq!(&qudits, &[3, 0]);
    /// ```
    #[inline]
    pub fn qudits(&self) -> impl Iterator<Item = usize> + '_ {
        self.inner.iter().filter_map(|w| if w.is_quantum() { Some(w.index()) } else { None })
    }

    /// Get the classical wire indices in this list.
    ///
    /// # Examples
    ///
    /// ```
    /// # use qudit_circuit::WireList;
    /// let wires = WireList::new(vec![3, 0], vec![1, 2]);
    /// let dits: Vec<usize> = wires.dits().collect();
    /// assert_eq!(&dits, &[1, 2]);
    /// ```
    #[inline]
    pub fn dits(&self) -> impl Iterator<Item = usize> + '_ {
        self.inner.iter().filter_map(|w| if w.is_classical() { Some(w.index()) } else { None })
    }

    /// Returns a new wire list containing all elements in `self` or `other`
    ///
    /// # Arguments
    ///
    /// * `other` - The other list to union.
    ///
    /// # Notes
    ///
    /// * The output orders the elements from `self` first, then the ones from
    ///   `other`.
    ///
    /// # Examples
    ///
    /// ```
    /// # use qudit_circuit::WireList;
    /// let loc1 = WireList::pure([0, 2]);
    /// let loc2 = WireList::pure([2, 3]);
    /// assert_eq!(loc1.union(&loc2), WireList::pure([0, 2, 3]));
    /// ```
    ///
    /// ```
    /// # use qudit_circuit::WireList;
    /// let loc1 = WireList::new([3, 0], [1, 2]);
    /// let loc2 = WireList::new([2, 3], [3, 2]);
    /// assert_eq!(loc1.union(&loc2), WireList::new([3, 0, 2], [1, 2, 3]));
    /// ```
    pub fn union(&self, other: &WireList) -> WireList {
        let mut union: CompactVec<Wire> = self.inner.clone();
        for wire in other {
            if !union.contains(&wire) {
                union.push(wire);
            }
        }
        WireList { inner: union }
    }

    /// Returns a new wire list containing all elements in `self` and `other`
    ///
    /// # Arguments
    ///
    /// * `other` - The other list to intersect.
    ///
    /// # Notes
    ///
    /// * The elements in output are ordered the same way they are ordered in
    ///   `self`.
    ///
    /// # Examples
    ///
    /// ```
    /// # use qudit_circuit::WireList;
    /// let loc1 = WireList::pure([0, 2]);
    /// let loc2 = WireList::pure([2, 3]);
    /// assert_eq!(loc1.intersect(&loc2), WireList::pure([2]));
    /// ```
    ///
    /// ```
    /// # use qudit_circuit::WireList;
    /// let loc1 = WireList::classical([0, 2]);
    /// let loc2 = WireList::classical([2, 0]);
    /// assert_eq!(loc1.intersect(&loc2), WireList::classical([0, 2]));
    /// ```
    ///
    /// ```
    /// # use qudit_circuit::WireList;
    /// let loc1 = WireList::new([3, 0], [1, 2]);
    /// let loc2 = WireList::new([2, 3], [3, 2]);
    /// assert_eq!(loc1.intersect(&loc2), WireList::new([3], [2]));
    /// ```
    pub fn intersect(&self, other: &WireList) -> WireList {
        let mut intersection_inner: CompactVec<Wire> = CompactVec::new();
        for wire in self {
            if other.inner.contains(&wire) {
                intersection_inner.push(wire);
            }
        }
        WireList { inner: intersection_inner }
    }

    /// Returns a new wire list containing elements in `self` but not in `other`
    ///
    /// # Arguments
    ///
    /// * `other` - The other list to subtract.
    ///
    /// # Notes
    ///
    /// * The elements in output are ordered the same way they are ordered in
    ///   `self`.
    ///
    /// # Examples
    ///
    /// ```
    /// # use qudit_circuit::WireList;
    /// let loc1 = WireList::pure([0, 2]);
    /// let loc2 = WireList::pure([2, 3]);
    /// assert_eq!(loc1.difference(&loc2), WireList::pure([0]));
    /// ```
    ///
    /// ```
    /// # use qudit_circuit::WireList;
    /// let loc1 = WireList::classical([0, 2]);
    /// let loc2 = WireList::classical([2, 0]);
    /// assert_eq!(loc1.difference(&loc2), WireList::classical([]));
    /// ```
    ///
    /// ```
    /// # use qudit_circuit::{Wire, WireList};
    /// let loc1 = WireList::from_wires(vec![Wire::quantum(3), Wire::quantum(0), Wire::classical(1), Wire::classical(2)]);
    /// let loc2 = WireList::from_wires(vec![Wire::quantum(2), Wire::quantum(3), Wire::classical(3), Wire::classical(2)]);
    /// let expected = WireList::from_wires(vec![Wire::quantum(0), Wire::classical(1)]);
    /// assert_eq!(loc1.difference(&loc2), expected);
    /// ```
    pub fn difference(&self, other: &WireList) -> WireList {
        let mut difference_inner: CompactVec<Wire> = CompactVec::new();
        for wire in self {
            if !other.inner.contains(&wire) {
                difference_inner.push(wire);
            }
        }
        WireList { inner: difference_inner }
    }

    /// Returns all possible pairs of qudits in this list.
    ///
    /// These are returned as sorted pairs, i.e., the first element of the
    /// pair is always less than the second element.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use qudit_circuit::WireList;
    /// let loc = WireList::pure([0, 3, 2]);
    ///
    /// let all_pairs = loc.get_qudit_pairs();
    ///
    /// for pair in [(0, 2), (0, 3), (2, 3)] {
    ///     assert!(all_pairs.contains(&pair))
    /// }
    /// ```
    pub fn get_qudit_pairs(&self) -> Vec<(usize, usize)> {
        let qudits_vec: Vec<usize> = self.qudits().collect();
        if qudits_vec.is_empty() {
            return Vec::new();
        }
        let num_pairs = qudits_vec.len() * (qudits_vec.len() - 1) / 2;
        let mut to_return = Vec::with_capacity(num_pairs);
        for &qudit_index1 in &qudits_vec {
            for &qudit_index2 in &qudits_vec {
                if qudit_index1 < qudit_index2 {
                    to_return.push((qudit_index1, qudit_index2));
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
    /// # use qudit_circuit::WireList;
    /// let loc = WireList::pure([2, 0, 3]);
    /// assert_eq!(loc.to_sorted(), WireList::pure([0, 2, 3]));
    /// ```
    ///
    /// ```
    /// # use qudit_circuit::WireList;
    /// let loc = WireList::new([3, 0], [2, 1]);
    /// assert_eq!(loc.to_sorted(), WireList::new([0, 3], [1, 2]));
    /// ```
    pub fn to_sorted(&self) -> WireList {
        let mut sorted_inner = self.inner.clone();
        sorted_inner.sort();
        WireList {
            inner: sorted_inner,
        }
    }

    /// Returns true if the location is sorted and has trivial ordering.
    ///
    /// # Examples
    ///
    /// ```
    /// # use qudit_circuit::WireList;
    /// let loc = WireList::pure([0, 2, 3]);
    /// assert!(loc.is_sorted());
    /// ```
    ///
    /// ```
    /// # use qudit_circuit::WireList;
    /// let loc = WireList::new([3, 0], [2, 1]);
    /// assert!(!loc.is_sorted());
    /// ```
    pub fn is_sorted(&self) -> bool {
        if self.len() < 2 {
            return true;
        }

        (0..(self.len() - 1))
            .all(|i| self.inner.get(i).unwrap() < self.inner.get(i + 1).unwrap())
    }

    /// Returns true if the qudits are sorted and have trivial ordering.
    ///
    /// # Examples
    ///
    /// ```
    /// # use qudit_circuit::WireList;
    /// let loc = WireList::pure([0, 2, 3]);
    /// assert!(loc.is_qudit_sorted());
    /// ```
    ///
    /// ```
    /// # use qudit_circuit::WireList;
    /// let loc = WireList::new([0, 3], [2, 1]);
    /// assert!(loc.is_qudit_sorted());
    /// ```
    pub fn is_qudit_sorted(&self) -> bool {
        let qudits_vec: Vec<usize> = self.qudits().collect();
        if qudits_vec.len() < 2 {
            return true;
        }

        (0..(qudits_vec.len() - 1))
            .all(|i| qudits_vec.get(i).unwrap() < qudits_vec.get(i + 1).unwrap())
    }

    /// Returns true if the dits are sorted and have trivial ordering.
    ///
    /// # Examples
    ///
    /// ```
    /// # use qudit_circuit::WireList;
    /// let loc = WireList::classical([0, 2, 3]);
    /// assert!(loc.is_dit_sorted());
    /// ```
    ///
    /// ```
    /// # use qudit_circuit::WireList;
    /// let loc = WireList::new([0, 3], [2, 1]);
    /// assert!(!loc.is_dit_sorted());
    /// ```
    pub fn is_dit_sorted(&self) -> bool {
        let dits_vec: Vec<usize> = self.dits().collect();
        if dits_vec.len() < 2 {
            return true;
        }

        (0..(dits_vec.len() - 1))
            .all(|i| dits_vec.get(i).unwrap() < dits_vec.get(i + 1).unwrap())
    }

    /// Returns true if the given wire is in the list.
    ///
    /// # Arguments
    ///
    /// * `wire` - The wire to check for.
    ///
    /// # Examples
    ///
    /// ```
    /// # use qudit_circuit::{Wire, WireList};
    /// let loc = WireList::pure([0, 2, 3]);
    /// assert!(loc.contains(Wire::quantum(0)));
    /// assert!(!loc.contains(Wire::classical(0)));
    /// ```
    #[inline]
    pub fn contains(&self, wire: Wire) -> bool {
        self.inner.contains(&wire)
    }

    /// Returns true if `qudit_index` is in the list.
    ///
    /// # Examples
    ///
    /// ```
    /// # use qudit_circuit::{Wire, WireList};
    /// let loc = WireList::pure([0, 2, 3]);
    /// assert!(loc.contains_qudit(0));
    /// assert!(!loc.contains_qudit(1));
    /// ```
    #[inline]
    pub fn contains_qudit(&self, qudit_index: usize) -> bool {
        self.contains(Wire::quantum(qudit_index))
    }

    /// Returns true if `dit_index` is in the list.
    ///
    /// # Examples
    ///
    /// ```
    /// # use qudit_circuit::{Wire, WireList};
    /// let loc = WireList::classical([0, 2, 3]);
    /// assert!(loc.contains_dit(0));
    /// assert!(!loc.contains_dit(1));
    /// ```
    #[inline]
    pub fn contains_dit(&self, dit_index: usize) -> bool {
        self.contains(Wire::classical(dit_index))
    }

    /// Returns the total number of wires in the list.
    ///
    /// # Examples
    ///
    /// ```
    /// # use qudit_circuit::WireList;
    /// let loc = WireList::pure([0, 2, 3]);
    /// assert_eq!(loc.len(), 3);
    /// ```
    ///
    /// ```
    /// # use qudit_circuit::WireList;
    /// let loc = WireList::new([0, 3], [2, 1]);
    /// assert_eq!(loc.len(), 4);
    /// ```
    #[inline]
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// Returns the number of qudits in the list.
    ///
    /// # Examples
    ///
    /// ```
    /// # use qudit_circuit::WireList;
    /// let loc = WireList::pure([0, 2, 3]);
    /// assert_eq!(loc.get_num_qudits(), 3);
    /// ```
    #[inline]
    pub fn get_num_qudits(&self) -> usize {
        self.qudits().count()
    }

    /// Returns the number of dits in the list.
    ///
    /// # Examples
    ///
    /// ```
    /// # use qudit_circuit::WireList;
    /// let loc = WireList::classical([0, 2, 3]);
    /// assert_eq!(loc.get_num_dits(), 3);
    /// ```
    #[inline]
    pub fn get_num_dits(&self) -> usize {
        self.dits().count()
    }

    /// Returns the position of the given quantum wire within the sub-list of quantum wires.
    ///
    /// This is not the same as get_position, as that's over the whole list.
    ///
    /// # Arguments
    ///
    /// * `qudit_index` - The index of the quantum wire to find.
    ///
    /// # Examples
    ///
    /// ```
    /// # use qudit_circuit::WireList;
    /// let loc = WireList::pure([5, 0, 3]);
    /// assert_eq!(loc.get_qudit_position(0), Some(1)); // 0 is at position 1 in [5, 0, 3]
    /// assert_eq!(loc.get_qudit_position(5), Some(0)); // 5 is at position 0 in [5, 0, 3]
    /// assert_eq!(loc.get_qudit_position(1), None);
    /// ```
    #[inline]
    pub fn get_qudit_position(&self, qudit_index: usize) -> Option<usize> {
        self.qudits().position(|x| x == qudit_index)
    }

    /// Returns the position of the given classical wire within the sub-list of classical wires.
    ///
    /// This is not the same as get_position, as that's over the whole list.
    ///
    /// # Arguments
    ///
    /// * `dit_index` - The index of the classical wire to find.
    ///
    /// # Examples
    ///
    /// ```
    /// # use qudit_circuit::WireList;
    /// let loc = WireList::classical([5, 0, 3]);
    /// assert_eq!(loc.get_dit_position(0), Some(1)); // 0 is at position 1 in [5, 0, 3]
    /// assert_eq!(loc.get_dit_position(5), Some(0)); // 5 is at position 0 in [5, 0, 3]
    /// assert_eq!(loc.get_dit_position(1), None);
    /// ```
    #[inline]
    pub fn get_dit_position(&self, dit_index: usize) -> Option<usize> {
        self.dits().position(|x| x == dit_index)
    }

    /// Returns a new, owned `WireList` with the same contents as `self`.
    ///
    /// This creates a deep copy of the `WireList`.
    ///
    /// # Examples
    ///
    /// ```
    /// # use qudit_circuit::WireList;
    /// let original = WireList::pure([0, 1, 2]);
    /// let owned_copy = original.to_owned();
    /// assert_eq!(original, owned_copy);
    /// assert!(!std::ptr::eq(&original, &owned_copy)); // Ensure it's a new allocation
    /// ```
    #[inline]
    pub fn to_owned(&self) -> WireList {
        self.clone()
    }
}

impl PartialEq for WireList {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        // Compare quantum wires (checks both length and content in one pass)
        if !self.qudits().eq(other.qudits()) {
            return false;
        }
        
        // Compare classical wires (checks both length and content in one pass)
        self.dits().eq(other.dits())
    }
}

impl Eq for WireList {}

impl std::hash::Hash for WireList {
    #[inline]
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        // Hash quantum wires in order
        for q in self.qudits() {
            q.hash(state);
        }
        
        // Hash classical wires in order
        for c in self.dits() {
            c.hash(state);
        }
    }
}

impl<'a> IntoIterator for &'a WireList {
    type Item = Wire;
    type IntoIter = <&'a CompactVec<Wire> as IntoIterator>::IntoIter;

    fn into_iter(self) -> Self::IntoIter {
        self.wires()
    }
}

impl std::fmt::Debug for WireList {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let wires: Vec<String> = self
            .wires()
            .map(|wire| format!("{}", wire))
            .collect();
        
        write!(f, "WireList [{}]", wires.join(", "))
    }
}

impl std::fmt::Display for WireList {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let wires: Vec<String> = self
            .wires()
            .map(|wire| format!("{}", wire))
            .collect();
        
        write!(f, "[{}]", wires.join(", "))
    }
}

impl From<usize> for WireList {
    fn from(index: usize) -> Self {
        WireList::pure([index])
    }
}

impl From<Vec<usize>> for WireList {
    fn from(indices: Vec<usize>) -> Self {
        WireList::pure(indices)
    }
}

impl<'a> From<&'a [usize]> for WireList {
    fn from(indices: &'a [usize]) -> Self {
        WireList::pure(indices)
    }
}

impl<const N: usize> From<[usize; N]> for WireList {
    fn from(indices: [usize; N]) -> Self {
        WireList::pure(indices)
    }
}

impl<const N: usize> From<&[usize; N]> for WireList {
    fn from(indices: &[usize; N]) -> Self {
        WireList::pure(indices)
    }
}

impl From<(usize, usize)> for WireList {
    fn from(indices: (usize, usize)) -> Self {
        WireList::new([indices.0], [indices.1])
    }
}

impl From<(Vec<usize>, Vec<usize>)> for WireList {
    fn from(indices: (Vec<usize>, Vec<usize>)) -> Self {
        WireList::new(indices.0, indices.1)
    }
}

impl<W: Into<Wire>> From<W> for WireList {
    fn from(wire: W) -> Self {
        WireList::from_wires([wire.into()])
    }
}

impl<W: Into<Wire>> From<Vec<W>> for WireList {
    fn from(wires: Vec<W>) -> Self {
        let converted_wires = wires.into_iter().map(|w| w.into()).collect::<Vec<Wire>>();
        WireList::from_wires(converted_wires)
    }
}

impl<'a, W> From<&'a [W]> for WireList
where
    W: Clone + Into<Wire>,
{
    fn from(wires: &'a [W]) -> Self {
        let converted_wires: Vec<Wire> = wires.iter().map(|w_ref| w_ref.clone().into()).collect();
        WireList::from_wires(converted_wires)
    }
}

impl<W, const N: usize> From<[W; N]> for WireList
where
    W: Into<Wire>,
{
    fn from(wires: [W; N]) -> Self {
        let converted_wires: Vec<Wire> = wires.into_iter().map(|w| w.into()).collect();
        WireList::from_wires(converted_wires)
    }
}

impl<'a, W, const N: usize> From<&'a [W; N]> for WireList
where
    W: Clone + Into<Wire>,
{
    fn from(wires: &'a [W; N]) -> Self {
        let converted_wires: Vec<Wire> = wires.iter().map(|w_ref| w_ref.clone().into()).collect();
        WireList::from_wires(converted_wires)
    }
}

impl<'a, 'b> From<(&'a [usize], &'b [usize])> for WireList {
    fn from(indices: (&'a [usize], &'b [usize])) -> Self {
        WireList::new(indices.0, indices.1)
    }
}

impl<const N: usize, const M: usize> From<([usize; N], [usize; M])> for WireList {
    fn from(indices: ([usize; N], [usize; M])) -> Self {
        WireList::new(indices.0, indices.1)
    }
}

impl<'a, 'b, const N: usize, const M: usize> From<(&'a [usize; N], &'b [usize; M])> for WireList {
    fn from(indices: (&'a [usize; N], &'b [usize; M])) -> Self {
        WireList::new(indices.0, indices.1)
    }
}

impl From<WireList> for CompactVec<Wire> {
    #[inline(always)]
    fn from(value: WireList) -> Self {
        value.inner
    }
}

#[cfg(feature = "python")]
mod python {
    use super::*;
    use pyo3::{prelude::*, types::PyTuple};
    use crate::python::PyCircuitRegistrar;
    use pyo3::types::PyList;

    /// Python wrapper for WireList
    #[pyclass(name = "WireList", frozen, hash, eq)]
    #[derive(Clone, Hash, PartialEq, Eq)]
    pub struct PyWireList {
        inner: WireList,
    }

    impl From<WireList> for PyWireList {
        fn from(wire_list: WireList) -> Self {
            PyWireList { inner: wire_list }
        }
    }

    impl From<PyWireList> for WireList {
        fn from(py_wire_list: PyWireList) -> Self {
            py_wire_list.inner
        }
    }

    impl AsRef<WireList> for PyWireList {
        fn as_ref(&self) -> &WireList {
            &self.inner
        }
    }

    #[pymethods]
    impl PyWireList {
        /// Create a purely quantum WireList from Python
        #[staticmethod]
        fn pure(indices: Vec<usize>) -> PyResult<Self> {
            Ok(PyWireList {
                inner: WireList::pure(indices),
            })
        }

        /// Create a purely classical WireList from Python
        #[staticmethod]
        fn classical(indices: Vec<usize>) -> PyResult<Self> {
            Ok(PyWireList {
                inner: WireList::classical(indices),
            })
        }

        /// Create a mixed WireList from Python
        #[new]
        #[pyo3(signature = (qudits_or_wires, dits = None))]
        fn new(qudits_or_wires: &Bound<'_, PyAny>, dits: Option<Vec<usize>>) -> PyResult<Self> {
            let mut all_wires = Vec::new();
            
            // Process the main iterable - can contain usize or Wire objects
            for item in pyo3::types::PyIterator::from_object(qudits_or_wires)? {
                let item = item?;
                
                // Try to extract as Wire first
                if let Ok(wire) = item.extract::<Wire>() {
                    all_wires.push(wire);
                }
                // Fall back to usize (convert to quantum wire)
                else if let Ok(index) = item.extract::<usize>() {
                    all_wires.push(Wire::quantum(index));
                }
                else {
                    return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                        "Items in qudits_or_wires must be integers or Wire objects"
                    ));
                }
            }
            
            // Add classical wires if dits is provided
            if let Some(classical_indices) = dits {
                for index in classical_indices {
                    all_wires.push(Wire::classical(index));
                }
            }
            
            Ok(PyWireList {
                inner: WireList::from_wires(all_wires),
            })
        }

        /// Get quantum wire indices as Python list
        #[getter]
        fn qudits<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
            let qudits: Vec<usize> = self.inner.qudits().collect();
            PyList::new(py, qudits)
        }

        /// Get classical wire indices as Python list  
        #[getter]
        fn dits<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
            let dits: Vec<usize> = self.inner.dits().collect();
            PyList::new(py, dits)
        }

        /// Get number of quantum wires
        #[getter]
        fn num_qudits(&self) -> usize {
            self.inner.get_num_qudits()
        }

        /// Get number of classical wires
        #[getter]
        fn num_dits(&self) -> usize {
            self.inner.get_num_dits()
        }

        /// Check if wire list contains a quantum wire
        fn contains_qudit(&self, index: usize) -> bool {
            self.inner.contains_qudit(index)
        }

        /// Check if wire list contains a classical wire
        fn contains_dit(&self, index: usize) -> bool {
            self.inner.contains_dit(index)
        }

        /// Union operation
        fn union(&self, other: &PyWireList) -> PyWireList {
            PyWireList {
                inner: self.inner.union(&other.inner),
            }
        }

        /// Intersection operation
        fn intersect(&self, other: &PyWireList) -> PyWireList {
            PyWireList {
                inner: self.inner.intersect(&other.inner),
            }
        }

        /// Difference operation
        fn difference(&self, other: &PyWireList) -> PyWireList {
            PyWireList {
                inner: self.inner.difference(&other.inner),
            }
        }

        /// Python len() function - returns total number of wires
        fn __len__(&self) -> usize {
            self.inner.len()
        }

        /// Python bool() function - empty wire lists are falsy
        fn __bool__(&self) -> bool {
            self.inner.len() > 0
        }

        /// Python 'in' operator - checks if wire exists in list
        fn __contains__(&self, wire: Wire) -> bool {
            self.inner.contains(wire)
        }

        /// Python '&' operator - intersection operation
        fn __and__(&self, other: &PyWireList) -> PyWireList {
            self.intersect(other)
        }

        /// Python '|' operator - union operation (alternative)
        fn __or__(&self, other: &PyWireList) -> PyWireList {
            self.union(other)
        }

        /// Python '-' operator - difference operation
        fn __sub__(&self, other: &PyWireList) -> PyWireList {
            self.difference(other)
        }

        /// Python iterator - yields Wire objects
        fn __iter__(slf: PyRef<'_, Self>) -> PyWireListIterator {
            let wires: Vec<Wire> = slf.inner.wires().collect();
            PyWireListIterator::new(wires)
        }

        /// Python string representation
        fn __str__(&self) -> String {
            format!("{}", self.inner)
        }

        /// Python debug representation
        fn __repr__(&self) -> String {
            format!("{:?}", self.inner)
        }
    }

    #[pyclass]
    pub struct PyWireListIterator {
        wires: Vec<Wire>,
        index: usize,
    }

    impl PyWireListIterator {
        fn new(wires: Vec<Wire>) -> Self {
            PyWireListIterator { wires, index: 0 }
        }
    }

    #[pymethods]
    impl PyWireListIterator {
        fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
            slf
        }

        fn __next__(mut slf: PyRefMut<'_, Self>) -> Option<Wire> {
            if slf.index >= slf.wires.len() {
                None
            } else {
                let result = slf.wires[slf.index].clone();
                slf.index += 1;
                Some(result)
            }
        }
    }

    impl<'py> FromPyObject<'py> for WireList {
        fn extract_bound(ob: &Bound<'py, PyAny>) -> PyResult<Self> {
            if let Ok(plist) = ob.extract::<PyWireList>() {
                return Ok(plist.inner)
            }

            // Try to extract as a tuple of two lists
            if let Ok(tuple) = ob.downcast::<PyTuple>() {
                if tuple.len() == 2 {
                    let qudits: Vec<usize> = tuple.get_item(0)?.extract()?;
                    let dits: Vec<usize> = tuple.get_item(1)?.extract()?;
                    return Ok(WireList::new(qudits, dits));
                }
            }
            
            // Try to extract as a single list (pure quantum)
            if let Ok(list) = ob.extract::<Vec<usize>>() {
                return Ok(WireList::pure(list));
            }
            
            Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                "WireList expects either a list of indices or a tuple of (qudits, dits)"
            ))
        }
    }

    impl<'py> IntoPyObject<'py> for WireList {
        type Target = <PyWireList as IntoPyObject<'py>>::Target;
        type Output = <PyWireList as IntoPyObject<'py>>::Output;
        type Error = <PyWireList as IntoPyObject<'py>>::Error;

        fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
            PyWireList::from(self).into_pyobject(py)
        }
    }

    /// Registers the Wire class with the Python module.
    fn register(parent_module: &Bound<'_, PyModule>) -> PyResult<()> {
        parent_module.add_class::<PyWireList>()?;
        Ok(())
    }
    inventory::submit!(PyCircuitRegistrar { func: register });
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;
    use std::hash::{Hash, Hasher};

    #[test]
    fn test_constructors() {
        // Test pure quantum constructor
        let pure_list = WireList::pure(vec![0, 2, 1]);
        assert_eq!(pure_list.get_num_qudits(), 3);
        assert_eq!(pure_list.get_num_dits(), 0);
        assert_eq!(pure_list.qudits().collect::<Vec<_>>(), vec![0, 2, 1]);

        // Test classical constructor
        let classical_list = WireList::classical(vec![1, 3]);
        assert_eq!(classical_list.get_num_qudits(), 0);
        assert_eq!(classical_list.get_num_dits(), 2);
        assert_eq!(classical_list.dits().collect::<Vec<_>>(), vec![1, 3]);

        // Test mixed constructor
        let mixed_list = WireList::new(vec![0, 2], vec![1, 3]);
        assert_eq!(mixed_list.get_num_qudits(), 2);
        assert_eq!(mixed_list.get_num_dits(), 2);
        assert_eq!(mixed_list.qudits().collect::<Vec<_>>(), vec![0, 2]);
        assert_eq!(mixed_list.dits().collect::<Vec<_>>(), vec![1, 3]);

        // Test from_wires constructor
        let wires = vec![Wire::quantum(0), Wire::classical(1), Wire::quantum(2)];
        let from_wires_list = WireList::from_wires(wires);
        assert_eq!(from_wires_list.get_num_qudits(), 2);
        assert_eq!(from_wires_list.get_num_dits(), 1);
    }

    #[test]
    #[should_panic(expected = "Duplicate indices in quantum wires")]
    fn test_pure_duplicate_panic() {
        WireList::pure(vec![0, 1, 0]);
    }

    #[test]
    #[should_panic(expected = "Duplicate indices in classical wires")]
    fn test_classical_duplicate_panic() {
        WireList::classical(vec![1, 2, 1]);
    }

    #[test]
    #[should_panic(expected = "Duplicate wires in wire list")]
    fn test_from_wires_duplicate_panic() {
        let wires = vec![Wire::quantum(0), Wire::quantum(0)];
        WireList::from_wires(wires);
    }

    #[test]
    fn test_equality_semantics() {
        // Two lists with same quantum and classical wires but different interleaving
        let list1 = WireList::from_wires(vec![Wire::quantum(0), Wire::classical(1), Wire::quantum(2)]);
        let list2 = WireList::from_wires(vec![Wire::quantum(0), Wire::quantum(2), Wire::classical(1)]);
        
        // Should be equal because quantum lists [0, 2] and classical lists [1] are the same
        assert_eq!(list1, list2);

        // Different quantum order should not be equal
        let list3 = WireList::from_wires(vec![Wire::quantum(2), Wire::quantum(0), Wire::classical(1)]);
        assert_ne!(list1, list3);

        // Different classical order should not be equal
        let list4 = WireList::from_wires(vec![
            Wire::quantum(0), Wire::quantum(2), Wire::classical(1), Wire::classical(3)
        ]);
        let list5 = WireList::from_wires(vec![
            Wire::quantum(0), Wire::quantum(2), Wire::classical(3), Wire::classical(1)
        ]);
        assert_ne!(list4, list5);
    }

    #[test]
    fn test_hash_consistency_with_equality() {
        let list1 = WireList::from_wires(vec![Wire::quantum(0), Wire::classical(1), Wire::quantum(2)]);
        let list2 = WireList::from_wires(vec![Wire::quantum(0), Wire::quantum(2), Wire::classical(1)]);
        
        // Equal objects must have equal hashes
        assert_eq!(list1, list2);
        
        let mut hasher1 = std::collections::hash_map::DefaultHasher::new();
        let mut hasher2 = std::collections::hash_map::DefaultHasher::new();
        list1.hash(&mut hasher1);
        list2.hash(&mut hasher2);
        assert_eq!(hasher1.finish(), hasher2.finish());

        // Test in HashSet to ensure hash consistency
        let mut set = HashSet::new();
        set.insert(list1.clone());
        assert!(set.contains(&list2));
    }

    #[test]
    fn test_set_operations() {
        let list1 = WireList::new(vec![0, 2], vec![1]);
        let list2 = WireList::new(vec![2, 3], vec![1, 4]);

        // Union
        let union = list1.union(&list2);
        let expected_union = WireList::new(vec![0, 2, 3], vec![1, 4]);
        assert_eq!(union, expected_union);

        // Intersection
        let intersection = list1.intersect(&list2);
        let expected_intersection = WireList::new(vec![2], vec![1]);
        assert_eq!(intersection, expected_intersection);

        // Difference
        let difference = list1.difference(&list2);
        let expected_difference = WireList::new(vec![0], vec![]);
        assert_eq!(difference, expected_difference);
    }

    #[test]
    fn test_sorting_methods() {
        let unsorted = WireList::new(vec![3, 0, 2], vec![5, 1]);
        let sorted = unsorted.to_sorted();
        let expected = WireList::new(vec![0, 2, 3], vec![1, 5]);
        assert_eq!(sorted, expected);

        // Test sorting checks
        assert!(!unsorted.is_sorted());
        assert!(sorted.is_sorted());
        assert!(!unsorted.is_qudit_sorted());
        assert!(sorted.is_qudit_sorted());
        assert!(!unsorted.is_dit_sorted());
        assert!(sorted.is_dit_sorted());

        // Test edge cases
        let empty = WireList::new(vec![], vec![]);
        assert!(empty.is_sorted());
        assert!(empty.is_qudit_sorted());
        assert!(empty.is_dit_sorted());

        let single = WireList::pure(vec![5]);
        assert!(single.is_sorted());
        assert!(single.is_qudit_sorted());
    }

    #[test]
    fn test_contains_methods() {
        let list = WireList::new(vec![0, 2], vec![1, 3]);

        // Test contains wire
        assert!(list.contains(Wire::quantum(0)));
        assert!(list.contains(Wire::classical(1)));
        assert!(!list.contains(Wire::quantum(1)));
        assert!(!list.contains(Wire::classical(0)));

        // Test contains qudit/dit
        assert!(list.contains_qudit(0));
        assert!(list.contains_qudit(2));
        assert!(!list.contains_qudit(1));
        
        assert!(list.contains_dit(1));
        assert!(list.contains_dit(3));
        assert!(!list.contains_dit(0));
    }

    #[test]
    fn test_position_methods() {
        let list = WireList::new(vec![5, 0, 3], vec![7, 1, 4]);

        // Test qudit positions
        assert_eq!(list.get_qudit_position(5), Some(0));
        assert_eq!(list.get_qudit_position(0), Some(1));
        assert_eq!(list.get_qudit_position(3), Some(2));
        assert_eq!(list.get_qudit_position(1), None);

        // Test dit positions
        assert_eq!(list.get_dit_position(7), Some(0));
        assert_eq!(list.get_dit_position(1), Some(1));
        assert_eq!(list.get_dit_position(4), Some(2));
        assert_eq!(list.get_dit_position(0), None);
    }

    #[test]
    fn test_get_qudit_pairs() {
        let list = WireList::pure(vec![0, 3, 1]);
        let pairs = list.get_qudit_pairs();
        let expected = vec![(0, 1), (0, 3), (1, 3)];
        
        assert_eq!(pairs.len(), expected.len());
        for pair in expected {
            assert!(pairs.contains(&pair));
        }

        // Test empty and single qubit
        let empty = WireList::pure(vec![]);
        assert!(empty.get_qudit_pairs().is_empty());

        let single = WireList::pure(vec![0]);
        assert!(single.get_qudit_pairs().is_empty());
    }

    #[test]
    fn test_from_implementations() {
        // Test usize
        let from_usize: WireList = 5usize.into();
        assert_eq!(from_usize, WireList::pure(vec![5]));

        // Test Vec<usize>
        let from_vec: WireList = vec![0usize, 2, 1].into();
        assert_eq!(from_vec, WireList::pure(vec![0, 2, 1]));

        // Test array
        let from_array: WireList = [1usize, 3, 2].into();
        assert_eq!(from_array, WireList::pure(vec![1, 3, 2]));

        // Test tuple (quantum, classical)
        let from_tuple: WireList = (5, 3).into();
        assert_eq!(from_tuple, WireList::new(vec![5], vec![3]));

        // Test tuple of vectors
        let from_vec_tuple: WireList = (vec![0, 1], vec![2, 3]).into();
        assert_eq!(from_vec_tuple, WireList::new(vec![0, 1], vec![2, 3]));

        // Test Wire
        let from_wire: WireList = Wire::quantum(7).into();
        assert_eq!(from_wire, WireList::pure(vec![7]));

        // Test Vec<Wire>
        let wires = vec![Wire::quantum(0), Wire::classical(1)];
        let from_wire_vec: WireList = wires.into();
        assert_eq!(from_wire_vec, WireList::new(vec![0], vec![1]));
    }

    #[test]
    fn test_display_and_debug() {
        let list = WireList::new(vec![0, 2], vec![1]);
        
        // Test Display (clean format)
        let display_output = format!("{}", list);
        assert!(display_output.contains("q0") && display_output.contains("q2") && display_output.contains("c1"));
        assert!(!display_output.contains("WireList")); // Display shouldn't include type name

        // Test Debug (includes type name)
        let debug_output = format!("{:?}", list);
        assert!(debug_output.contains("WireList"));
        assert!(debug_output.contains("q0") && debug_output.contains("q2") && debug_output.contains("c1"));
    }

    #[test]
    fn test_iterator() {
        let list = WireList::new(vec![0, 2], vec![1]);
        
        let collected: Vec<Wire> = (&list).into_iter().collect();
        assert_eq!(collected.len(), 3);
        
        // Test that iteration preserves order
        let mut iter = (&list).into_iter();
        let first = iter.next().unwrap();
        assert!(first.is_quantum() && first.index() == 0);
    }

    #[test]
    fn test_clone_and_to_owned() {
        let original = WireList::new(vec![0, 1], vec![2]);
        let cloned = original.clone();
        let owned = original.to_owned();
        
        assert_eq!(original, cloned);
        assert_eq!(original, owned);
        
        // Ensure they're separate objects
        assert!(!std::ptr::eq(&original.inner, &cloned.inner));
        assert!(!std::ptr::eq(&original.inner, &owned.inner));
    }

    #[test]
    fn test_large_collections_performance() {
        // Test that large collections use HashSet for duplicate checking
        let large_indices: Vec<usize> = (0..25).collect();
        let large_list = WireList::pure(&large_indices);
        assert_eq!(large_list.get_num_qudits(), 25);
        
        // This should work without panicking
        let _mixed_large = WireList::new(&large_indices[..15], &large_indices[10..]);
    }

    #[test]
    fn test_edge_cases() {
        // Empty lists
        let empty_pure = WireList::pure(vec![]);
        let empty_classical = WireList::classical(vec![]);
        let empty_mixed = WireList::new(vec![], vec![]);
        
        assert_eq!(empty_pure.len(), 0);
        assert_eq!(empty_classical.len(), 0);
        assert_eq!(empty_mixed.len(), 0);
        
        assert_eq!(empty_pure, empty_classical);
        assert_eq!(empty_pure, empty_mixed);
        
        // Single wire lists
        let single_q = WireList::pure(vec![0]);
        let single_c = WireList::classical(vec![0]);
        
        assert_ne!(single_q, single_c); // Different wire types
        assert_eq!(single_q.len(), 1);
        assert_eq!(single_c.len(), 1);
        
        // Union with self should be identity
        let list = WireList::new(vec![0, 1], vec![2]);
        assert_eq!(list.union(&list), list);
        
        // Intersection with empty should be empty
        assert_eq!(list.intersect(&empty_mixed), empty_mixed);
        
        // Difference with self should be empty
        assert_eq!(list.difference(&list), empty_mixed);
    }
}
