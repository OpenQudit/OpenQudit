use std::ops::Range;

use bit_set::BitSet;

use crate::RealScalar;

/// A data structure representing indices of parameters (e.g. for a function).
/// There are optimized methods for consecutive and disjoint parameters.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum ParamIndices {
    /// The index of the first parameter (consecutive parameters)
    Joint(usize, usize), // start, length

    /// The index of each parameter (disjoint parameters)
    Disjoint(Vec<usize>),
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
    /// This function returns a `ParamIndices::Disjoint` with an empty vector,
    /// indicating that there are no parameters associated with this value.
    ///
    /// # Returns
    ///
    /// A `ParamIndices` representing a constant value.
    pub fn constant() -> ParamIndices {
        ParamIndices::Disjoint(vec![])
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
            },
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

    /// Creates an iterator over the parameter indices.
    ///
    /// # Returns
    ///
    /// A `ParamIndicesIter` that yields the parameter indices.
    pub fn iter<'a>(&'a self) -> ParamIndicesIter<'a> {
        match self {
            ParamIndices::Joint(start, length) => {
                ParamIndicesIter::Joint {
                    start: *start,
                    length: *length,
                    current: 0,
                }
            }
            ParamIndices::Disjoint(indices) => {
                ParamIndicesIter::Disjoint {
                    indices: &indices,
                    current: 0,
                }
            }
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
            ParamIndices::Joint(start, length) => {
                index >= *start && index < *start + *length
            }
            ParamIndices::Disjoint(indices) => {
                indices.contains(&index)
            }
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
        indices: &'a[usize],
        /// The current index in the slice.
        current: usize,
    },
}

impl<'a> Iterator for ParamIndicesIter<'a> {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            ParamIndicesIter::Joint { start, length, current } => {
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

// impl From<Range<usize>> for ParamIndices {
//     fn from(index: Range<usize>) -> Self {
//         ParamIndices::Joint(index.start, index.end - index.start)
//     }
// }

// impl From<usize> for ParamIndices {
//     fn from(index: usize) -> Self {
//         ParamIndices::Joint(index, 1)
//     }
// }

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
