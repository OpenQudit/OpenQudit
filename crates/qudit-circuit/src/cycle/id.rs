use slotmap::new_key_type;
use std::ops::{Add, Sub, AddAssign, SubAssign};

new_key_type! { 
    /// A unique internal identifier for instruction instances within a cycle.
    ///
    /// Generated automatically by the cycle when instructions are inserted,
    /// providing persistent, type-safe access and preventing use-after-free errors.
    pub struct InstId;
}

/// A persistent identifier for a cycle.
///
/// Note: imposes a maximum number of cycles as identifiers are never reused.
/// There can only `std::u64::MAX - 1` cycles ever created for a single circuit.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct CycleId(pub u64);

/// A index for a cycle describing execution order of cycles.
///
/// These may change and should not be used to identify cycles.
/// Instead, use CycleId.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct CycleIndex(pub u64);

/// A sentinel value representing an invalid or uninitialized cycle ID.
pub const INVALID_CYCLE_ID: CycleId = CycleId(std::u64::MAX);

impl CycleId {
    /// Creates a new cycle identifier from the given value.
    pub const fn new(id: u64) -> Self {
        Self(id)
    }

    /// Extracts the underlying numeric value of this cycle identifier.
    pub const fn get(self) -> u64 {
        self.0
    }
}

impl CycleIndex {
    /// Creates a new cycle index from the given value.
    pub const fn new(index: u64) -> Self {
        Self(index)
    }

    /// Extracts the underlying numeric value of this cycle index.
    pub const fn get(self) -> u64 {
        self.0
    }
}

impl From<u64> for CycleId {
    #[inline(always)]
    fn from(value: u64) -> Self {
        Self(value)
    }
}

impl From<CycleId> for u64 {
    #[inline(always)]
    fn from(id: CycleId) -> Self {
        id.0
    }
}

impl From<usize> for CycleId {
    #[inline(always)]
    fn from(value: usize) -> Self {
        Self(value as u64)
    }
}

impl From<CycleId> for usize {
    #[inline(always)]
    fn from(id: CycleId) -> Self {
        id.0 as usize
    }
}

impl From<u64> for CycleIndex {
    #[inline(always)]
    fn from(value: u64) -> Self {
        Self(value)
    }
}

impl From<CycleIndex> for u64 {
    #[inline(always)]
    fn from(index: CycleIndex) -> Self {
        index.0
    }
}

impl From<usize> for CycleIndex {
    #[inline(always)]
    fn from(value: usize) -> Self {
        Self(value as u64)
    }
}

impl From<CycleIndex> for usize {
    #[inline(always)]
    fn from(index: CycleIndex) -> Self {
        index.0 as usize
    }
}

impl<T: Into<CycleIndex>> Add<T> for CycleIndex {
    type Output = CycleIndex;

    fn add(self, rhs: T) -> Self::Output {
        let rhs_index = rhs.into();
        CycleIndex(self.0 + rhs_index.0)
    }
}

impl<T: Into<CycleIndex>> Sub<T> for CycleIndex {
    type Output = CycleIndex;

    fn sub(self, rhs: T) -> Self::Output {
        let rhs_index = rhs.into();
        CycleIndex(self.0 - rhs_index.0)
    }
}

impl<T: Into<CycleIndex>> AddAssign<T> for CycleIndex {
    fn add_assign(&mut self, rhs: T) {
        let rhs_index = rhs.into();
        self.0 += rhs_index.0;
    }
}

impl<T: Into<CycleIndex>> SubAssign<T> for CycleIndex {
    fn sub_assign(&mut self, rhs: T) {
        let rhs_index = rhs.into();
        self.0 -= rhs_index.0;
    }
}

impl PartialEq<u64> for CycleIndex {
    fn eq(&self, other: &u64) -> bool {
        self.0 == *other
    }
}

impl PartialEq<usize> for CycleIndex {
    fn eq(&self, other: &usize) -> bool {
        self.0 as usize == *other
    }
}

impl PartialEq<i32> for CycleIndex {
    fn eq(&self, other: &i32) -> bool {
        if *other < 0 {
            return false;
        }
        self.0 == *other as u64
    }
}

#[cfg(feature = "python")]
mod python {
    use super::*;
    use pyo3::prelude::*;
    use slotmap::{Key, KeyData};

    impl<'py> IntoPyObject<'py> for InstId {
        type Target = <u64 as IntoPyObject<'py>>::Target;
        type Output = <u64 as IntoPyObject<'py>>::Output;
        type Error = <u64 as IntoPyObject<'py>>::Error;

        fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
            self.data().as_ffi().into_pyobject(py)
        }
    }

    impl<'a, 'py> FromPyObject<'a, 'py> for InstId {
        type Error = PyErr;

        fn extract(obj: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
            let value: u64 = obj.extract()?;
            Ok(InstId::from(KeyData::from_ffi(value)))
        }
    }

    impl<'py> IntoPyObject<'py> for CycleId {
        type Target = <u64 as IntoPyObject<'py>>::Target;
        type Output = <u64 as IntoPyObject<'py>>::Output;
        type Error = <u64 as IntoPyObject<'py>>::Error;

        fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
            self.0.into_pyobject(py)
        }
    }

    impl<'a, 'py> FromPyObject<'a, 'py> for CycleId {
        type Error = PyErr;

        fn extract(obj: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
            let value: u64 = obj.extract()?;
            Ok(CycleId(value))
        }
    }

    impl<'py> IntoPyObject<'py> for CycleIndex {
        type Target = <u64 as IntoPyObject<'py>>::Target;
        type Output = <u64 as IntoPyObject<'py>>::Output;
        type Error = <u64 as IntoPyObject<'py>>::Error;

        fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
            self.0.into_pyobject(py)
        }
    }

    impl<'a, 'py> FromPyObject<'a, 'py> for CycleIndex {
        type Error = PyErr;

        fn extract(obj: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
            let value: u64 = obj.extract()?;
            Ok(CycleIndex(value))
        }
    }
}

