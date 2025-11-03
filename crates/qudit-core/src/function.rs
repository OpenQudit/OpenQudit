use std::ops::Range;

use bit_set::BitSet;
use std::hash::{Hash, Hasher};

use crate::RealScalar;

/// A data structure representing indices of parameters (e.g. for a function).
/// There are optimized methods for consecutive and disjoint parameters.
#[derive(Clone, Debug)]
pub enum ParamIndices {
    /// The index of the first parameter (consecutive parameters)
    Joint(usize, usize), // start, length

    /// The index of each parameter (disjoint parameters)
    Disjoint(Vec<usize>),
}

impl PartialEq for ParamIndices {
    fn eq(&self, other: &Self) -> bool {
        if self.len() != other.len() {
            return false;
        }

        match (self, other) {
            // Same variants - delegate to inner equality
            (ParamIndices::Joint(start1, len1), ParamIndices::Joint(start2, len2)) => {
                start1 == start2 && len1 == len2
            }
            (ParamIndices::Disjoint(v1), ParamIndices::Disjoint(v2)) => v1 == v2,
            // Cross variants - compare iterators element by element
            _ => self.iter().eq(other.iter()),
        }
    }
}

impl Eq for ParamIndices {}

impl Hash for ParamIndices {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.len().hash(state);
        for index in self.iter() {
            index.hash(state);
        }
    }
}

impl ParamIndices {
    /// Converts the `ParamIndices` to a `BitSet`.
    ///
    /// This function creates a `BitSet` where each bit corresponds to a parameter index.
    /// If a parameter index is present in the `ParamIndices`, the corresponding bit in the
    /// `BitSet` is set to 1.
    ///
    /// # Returns
    ///
    /// A `BitSet` representing the parameter indices.
    pub fn to_bitset(&self) -> BitSet {
        let mut bitset = BitSet::new();
        match self {
            ParamIndices::Joint(start, length) => {
                for i in 0..*length {
                    bitset.insert(*start + i);
                }
            }
            ParamIndices::Disjoint(indices) => {
                for index in indices {
                    bitset.insert(*index);
                }
            }
        }
        bitset
    }

    /// Creates a `ParamIndices` representing a constant value (no parameters).
    ///
    /// # Returns
    ///
    /// A `ParamIndices` representing an empty set of parameter indices.
    pub fn empty() -> ParamIndices {
        ParamIndices::Joint(0, 0)
    }

    /// Checks if the `ParamIndices` is empty (contains no parameters).
    ///
    /// # Returns
    ///
    /// `true` if the `ParamIndices` is empty, `false` otherwise.
    pub fn is_empty(&self) -> bool {
        match self {
            ParamIndices::Joint(_, length) => *length == 0,
            ParamIndices::Disjoint(v) => v.is_empty(),
        }
    }

    /// Checks if the `ParamIndices` represents a consecutive range of parameters.
    ///
    /// # Returns
    ///
    /// `true` if the `ParamIndices` is consecutive, `false` otherwise.
    pub fn is_consecutive(&self) -> bool {
        match self {
            ParamIndices::Joint(_, _) => true,
            ParamIndices::Disjoint(v) => {
                if v.len() <= 1 {
                    return true;
                }
                let mut clone_v = v.clone();
                clone_v.sort();
                for i in 1..clone_v.len() {
                    if clone_v[i] != clone_v[i - 1] + 1 {
                        return false;
                    }
                }
                true
            }
        }
    }

    /// Returns the number of parameters represented by the `ParamIndices`.
    ///
    /// # Returns
    ///
    /// The number of parameters.
    ///
    pub fn num_params(&self) -> usize {
        match self {
            ParamIndices::Joint(_, length) => *length,
            ParamIndices::Disjoint(v) => v.len(),
        }
    }

    /// Returns the starting index of the parameters.
    ///
    /// For `ParamIndices::Joint`, this is the index of the first parameter in the consecutive range.
    /// For `ParamIndices::Disjoint`, this is the index of the first parameter in the vector, or 0 if the vector is empty.
    ///
    /// # Returns
    ///
    /// The starting index of the parameters.
    pub fn start(&self) -> usize {
        match self {
            ParamIndices::Joint(start, _) => *start,
            ParamIndices::Disjoint(v) => *v.first().unwrap_or(&0),
        }
    }

    /// Concatenates two `ParamIndices` into a single `ParamIndices`.
    ///
    /// This function combines the parameter indices from two `ParamIndices` instances.
    /// If the two `ParamIndices` represent overlapping ranges, they are merged into a single `ParamIndices::Joint` if possible.
    /// Otherwise, the parameter indices are combined into a `ParamIndices::Disjoint`.
    ///
    /// # Arguments
    ///
    /// * `other` - The other `ParamIndices` to concatenate with.
    ///
    /// # Returns
    ///
    /// A new `ParamIndices` representing the concatenation of the two input `ParamIndices`.
    pub fn concat(&self, other: &ParamIndices) -> ParamIndices {
        match (self, other) {
            (ParamIndices::Joint(start1, length1), ParamIndices::Joint(start2, length2)) => {
                if *start2 > *start1 && *start2 < *start1 + *length1 {
                    ParamIndices::Joint(*start1, *start2 + *length2 - *start1)
                } else if *start1 > *start2 && *start1 < *start2 + *length2 {
                    ParamIndices::Joint(*start2, *start1 + *length1 - *start2)
                } else {
                    let mut indices = Vec::new();
                    for i in *start1..*start1 + *length1 {
                        indices.push(i);
                    }
                    for i in *start2..*start2 + *length2 {
                        indices.push(i);
                    }
                    ParamIndices::Disjoint(indices)
                }
            }
            (ParamIndices::Joint(start, length), ParamIndices::Disjoint(v)) => {
                let mut indices = Vec::new();
                for i in *start..*start + *length {
                    if v.contains(&i) {
                        continue;
                    }
                    indices.push(i);
                }
                indices.extend(v.iter());
                ParamIndices::Disjoint(indices)
            }
            (ParamIndices::Disjoint(v), ParamIndices::Joint(start, length)) => {
                let mut indices = Vec::new();
                for i in *start..*start + *length {
                    if v.contains(&i) {
                        continue;
                    }
                    indices.push(i);
                }
                indices.extend(v.iter());
                ParamIndices::Disjoint(indices)
            }
            (ParamIndices::Disjoint(v1), ParamIndices::Disjoint(v2)) => {
                let mut indices = Vec::new();
                for i in v1 {
                    if v2.contains(i) {
                        continue;
                    }
                    indices.push(*i);
                }
                indices.extend(v2.iter());
                ParamIndices::Disjoint(indices)
            }
        }
    }

    /// Computes the intersection of two `ParamIndices`.
    ///
    /// This function returns a new `ParamIndices` containing only the parameter indices that are present in both input `ParamIndices`.
    /// The result is always a `ParamIndices::Disjoint`.
    ///
    /// # Arguments
    ///
    /// * `other` - The other `ParamIndices` to intersect with.
    ///
    /// # Returns
    ///
    /// A new `ParamIndices` representing the intersection of the two input `ParamIndices`.
    pub fn intersect(&self, other: &ParamIndices) -> ParamIndices {
        match (self, other) {
            (ParamIndices::Joint(start1, length1), ParamIndices::Joint(start2, length2)) => {
                let mut indices = Vec::new();
                for i in *start1..*start1 + *length1 {
                    if i >= *start2 && i < *start2 + *length2 {
                        indices.push(i);
                    }
                }
                ParamIndices::Disjoint(indices)
            }
            (ParamIndices::Joint(start, length), ParamIndices::Disjoint(v)) => {
                let mut indices = Vec::new();
                for i in *start..*start + *length {
                    if v.contains(&i) {
                        indices.push(i);
                    }
                }
                ParamIndices::Disjoint(indices)
            }
            (ParamIndices::Disjoint(v), ParamIndices::Joint(start, length)) => {
                let mut indices = Vec::new();
                for i in *start..*start + *length {
                    if v.contains(&i) {
                        indices.push(i);
                    }
                }
                ParamIndices::Disjoint(indices)
            }
            (ParamIndices::Disjoint(v1), ParamIndices::Disjoint(v2)) => {
                let mut indices = Vec::new();
                for i in v1 {
                    if v2.contains(i) {
                        indices.push(*i);
                    }
                }
                ParamIndices::Disjoint(indices)
            }
        }
    }

    /// Sorts the indices.
    ///
    /// If the `ParamIndices` is `Joint`, this method does nothing.
    ///
    /// # Returns
    ///
    /// A mutable reference to `self` for chaining.
    pub fn sort(&mut self) -> &mut Self {
        match self {
            ParamIndices::Joint(_, _) => {}
            ParamIndices::Disjoint(indices) => indices.sort(),
        }
        self
    }

    /// Returns a new `ParamIndices` with the indices sorted.
    pub fn sorted(&self) -> Self {
        match self {
            ParamIndices::Joint(s, l) => ParamIndices::Joint(*s, *l),
            ParamIndices::Disjoint(indices) => {
                let mut indices_out = indices.clone();
                indices_out.sort();
                ParamIndices::Disjoint(indices_out)
            }
        }
    }

    /// Creates an iterator over the parameter indices.
    ///
    /// # Returns
    ///
    /// A `ParamIndicesIter` that yields the parameter indices.
    pub fn iter<'a>(&'a self) -> ParamIndicesIter<'a> {
        match self {
            ParamIndices::Joint(start, length) => ParamIndicesIter::Joint {
                start: *start,
                length: *length,
                current: 0,
            },
            ParamIndices::Disjoint(indices) => ParamIndicesIter::Disjoint {
                indices: &indices,
                current: 0,
            },
        }
    }

    /// Checks if the `ParamIndices` contains the given index.
    ///
    /// # Arguments
    ///
    /// * `index` - The index to check.
    ///
    /// # Returns
    ///
    /// `true` if the `ParamIndices` contains the index, `false` otherwise.
    pub fn contains(&self, index: usize) -> bool {
        match self {
            ParamIndices::Joint(start, length) => index >= *start && index < *start + *length,
            ParamIndices::Disjoint(indices) => indices.contains(&index),
        }
    }

    /// Convert the indices to a vector
    pub fn to_vec(self) -> Vec<usize> {
        match self {
            ParamIndices::Joint(_, _) => self.iter().collect(),
            ParamIndices::Disjoint(vec) => vec,
        }
    }

    /// Convert the indices to a vector without consuming the indices object.
    pub fn as_vec(&self) -> Vec<usize> {
        self.iter().collect()
    }

    /// Returns the number of indices tracked by this object; alias for num_params()
    pub fn len(&self) -> usize {
        match self {
            ParamIndices::Joint(_, length) => *length,
            ParamIndices::Disjoint(indices) => indices.len(),
        }
    }
}

/// An iterator over the parameter indices in a `ParamIndices`.
pub enum ParamIndicesIter<'a> {
    /// Iterator for `ParamIndices::Joint`.
    Joint {
        /// The starting index of the consecutive range.
        start: usize,
        /// The length of the consecutive range.
        length: usize,
        /// The current index in the range.
        current: usize,
    },
    /// Iterator for `ParamIndices::Disjoint`.
    Disjoint {
        /// A slice of the indices.
        indices: &'a [usize],
        /// The current index in the slice.
        current: usize,
    },
}

impl<'a> Iterator for ParamIndicesIter<'a> {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            ParamIndicesIter::Joint {
                start,
                length,
                current,
            } => {
                if *current < *length {
                    let result = *start + *current;
                    *current += 1;
                    Some(result)
                } else {
                    None
                }
            }
            ParamIndicesIter::Disjoint { indices, current } => {
                if *current < indices.len() {
                    let result = indices[*current];
                    *current += 1;
                    Some(result)
                } else {
                    None
                }
            }
        }
    }
}

impl<T: AsRef<[usize]>> From<T> for ParamIndices {
    fn from(indices: T) -> Self {
        ParamIndices::Disjoint(indices.as_ref().to_vec())
    }
}

/// A parameterized object.
pub trait HasParams {
    /// The number of parameters this object requires.
    fn num_params(&self) -> usize;
}

/// A bounded, parameterized object.
pub trait HasBounds<R: RealScalar>: HasParams {
    /// The bounds for each variable of the function
    fn bounds(&self) -> Vec<Range<R>>;
}

/// A periodic, parameterized object
pub trait HasPeriods<R: RealScalar>: HasParams {
    /// The core period for each variable of the function
    fn periods(&self) -> Vec<Range<R>>;
}

// #[cfg(test)]
// pub mod strategies {
//     use std::ops::Range;

//     use proptest::prelude::*;

//     use super::BoundedFn;

//     pub fn params(num_params: usize) -> impl Strategy<Value = Vec<f64>> {
//         prop::collection::vec(
//             prop::num::f64::POSITIVE
//                 | prop::num::f64::NEGATIVE
//                 | prop::num::f64::NORMAL
//                 | prop::num::f64::SUBNORMAL
//                 | prop::num::f64::ZERO,
//             num_params,
//         )
//     }

//     pub fn params_with_bounds(
//         bounds: Vec<Range<f64>>,
//     ) -> impl Strategy<Value = Vec<f64>> {
//         bounds
//     }

//     pub fn arbitrary_with_params_strategy<F: Clone + BoundedFn + Arbitrary>(
//     ) -> impl Strategy<Value = (F, Vec<f64>)> {
//         any::<F>().prop_flat_map(|f| (Just(f.clone()), f.get_bounds()))
//     }
// }

#[cfg(feature = "python")]
mod python {
    use super::*;
    use pyo3::prelude::*;
    use pyo3::types::PyIterator;

    /// Python wrapper for parameter indices.
    ///
    /// This provides parameter index functionality to Python, supporting both
    /// consecutive (Joint) and disjoint parameter representations.
    #[pyclass(name = "ParamIndices", frozen, eq, hash)]
    #[derive(Clone, Debug, PartialEq, Eq, Hash)]
    pub struct PyParamIndices {
        inner: ParamIndices,
    }

    #[pymethods]
    impl PyParamIndices {
        /// Creates new parameter indices.
        ///
        /// # Arguments
        ///
        /// * `indices_or_start` - Either an iterable of parameter indices, or the starting index for a consecutive range
        /// * `length` - Optional length for consecutive range (only used if `indices_or_start` is an integer)
        ///
        /// # Examples
        ///
        /// ```python
        /// # Disjoint parameters
        /// params1 = ParamIndices([0, 2, 5, 7])
        ///
        /// # Consecutive parameters starting at index 10 with length 5
        /// params2 = ParamIndices(10, 5)  # represents [10, 11, 12, 13, 14]
        /// ```
        #[new]
        #[pyo3(signature = (indices_or_start, length = None))]
        fn new<'py>(indices_or_start: &Bound<'py, PyAny>, length: Option<usize>) -> PyResult<Self> {
            if let Some(len) = length {
                // Joint case: start + length
                let start: usize = indices_or_start.extract()?;
                Ok(PyParamIndices {
                    inner: ParamIndices::Joint(start, len),
                })
            } else {
                // Try to extract as an integer first (single parameter)
                if let Ok(single_index) = indices_or_start.extract::<usize>() {
                    Ok(PyParamIndices {
                        inner: ParamIndices::Joint(single_index, 1),
                    })
                } else {
                    // Try to extract as an iterable of indices
                    let iter = PyIterator::from_object(indices_or_start)?;
                    let mut indices = Vec::new();
                    for item in iter {
                        let index: usize = item?.extract()?;
                        indices.push(index);
                    }
                    Ok(PyParamIndices {
                        inner: ParamIndices::Disjoint(indices),
                    })
                }
            }
        }

        /// Returns the number of parameters.
        #[getter]
        fn num_params(&self) -> usize {
            self.inner.num_params()
        }

        /// Returns the starting index (for Joint) or first index (for Disjoint).
        #[getter]
        fn start(&self) -> usize {
            self.inner.start()
        }

        /// Returns whether the parameters are consecutive.
        #[getter]
        fn is_consecutive(&self) -> bool {
            self.inner.is_consecutive()
        }

        /// Returns whether the parameter indices are empty.
        #[getter]
        fn is_empty(&self) -> bool {
            self.inner.is_empty()
        }

        /// Checks if the parameter indices contain the given index.
        fn contains(&self, index: usize) -> bool {
            self.inner.contains(index)
        }

        /// Returns all parameter indices as a list.
        fn to_list(&self) -> Vec<usize> {
            self.inner.as_vec()
        }

        /// Returns a sorted copy of these parameter indices.
        fn sorted(&self) -> PyParamIndices {
            PyParamIndices {
                inner: self.inner.sorted(),
            }
        }

        /// Concatenates with another ParamIndices.
        fn concat(&self, other: &PyParamIndices) -> PyParamIndices {
            PyParamIndices {
                inner: self.inner.concat(&other.inner),
            }
        }

        /// Intersects with another ParamIndices.
        fn intersect(&self, other: &PyParamIndices) -> PyParamIndices {
            PyParamIndices {
                inner: self.inner.intersect(&other.inner),
            }
        }

        /// Creates parameter indices for a constant (no parameters).
        #[staticmethod]
        fn constant() -> PyParamIndices {
            PyParamIndices {
                inner: ParamIndices::empty(),
            }
        }

        fn __repr__(&self) -> String {
            match &self.inner {
                ParamIndices::Joint(start, length) => {
                    if *length == 1 {
                        format!("ParamIndices({})", start)
                    } else {
                        format!("ParamIndices({}, {})", start, length)
                    }
                }
                ParamIndices::Disjoint(indices) => {
                    format!("ParamIndices({:?})", indices)
                }
            }
        }

        fn __str__(&self) -> String {
            let indices = self.inner.as_vec();
            if indices.len() <= 3 {
                format!("{:?}", indices)
            } else {
                format!(
                    "[{}, ..., {}]",
                    indices.first().unwrap(),
                    indices.last().unwrap()
                )
            }
        }

        fn __len__(&self) -> usize {
            self.inner.len()
        }

        fn __iter__(slf: PyRef<'_, Self>) -> PyParamIndicesIterator {
            PyParamIndicesIterator {
                indices: slf.inner.as_vec(),
                index: 0,
            }
        }

        fn __contains__(&self, index: usize) -> bool {
            self.inner.contains(index)
        }
    }

    #[pyclass]
    struct PyParamIndicesIterator {
        indices: Vec<usize>,
        index: usize,
    }

    #[pymethods]
    impl PyParamIndicesIterator {
        fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
            slf
        }

        fn __next__(mut slf: PyRefMut<'_, Self>) -> Option<usize> {
            if slf.index < slf.indices.len() {
                let result = slf.indices[slf.index];
                slf.index += 1;
                Some(result)
            } else {
                None
            }
        }
    }

    impl From<ParamIndices> for PyParamIndices {
        fn from(indices: ParamIndices) -> Self {
            PyParamIndices { inner: indices }
        }
    }

    impl From<PyParamIndices> for ParamIndices {
        fn from(py_indices: PyParamIndices) -> Self {
            py_indices.inner
        }
    }

    impl<'py> IntoPyObject<'py> for ParamIndices {
        type Target = PyParamIndices;
        type Output = Bound<'py, Self::Target>;
        type Error = PyErr;

        fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
            let py_indices = PyParamIndices::from(self);
            Bound::new(py, py_indices)
        }
    }

    impl<'a, 'py> FromPyObject<'a, 'py> for ParamIndices {
        type Error = PyErr;

        fn extract(ob: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
            let py_indices: PyParamIndices = ob.extract()?;
            Ok(py_indices.into())
        }
    }
}
