use slotmap::Key;

use crate::cycle::{CycleId, INVALID_CYCLE_ID};
use crate::cycle::InstId;

/// A persitent identifier uniquely identifying an instruction within a specific cycle.
///
/// Combines a persistent cycle identifier with an persistant, in-cycle, instruction
/// identifier to create a globally unique reference to a specific instruction
/// instance.
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq, Ord, PartialOrd)]
pub struct InstructionId(CycleId, InstId);

impl InstructionId {
    /// Creates a new instruction identifier from a cycle and instruction ID.
    ///
    /// This method is internal to the crate to maintain control over instruction
    /// identifier creation and ensure consistency.
    pub(crate) fn new(cycle_id: CycleId, inst_id: InstId) -> Self {
        InstructionId(cycle_id, inst_id)
    }

    /// Returns the cycle identifier component of this instruction ID.
    pub fn cycle(&self) -> CycleId {
        self.0
    }

    /// Returns the instruction identifier component within the cycle.
    pub fn inner(&self) -> InstId {
        self.1
    }

    /// Returns `true` if this has a valid cycle identifier component.
    pub fn is_valid(&self) -> bool {
        self.0 != INVALID_CYCLE_ID
    }

    /// Returns `true` if this has an invalid cycle identifier component.
    pub fn is_invalid(&self) -> bool {
        self.0 == INVALID_CYCLE_ID
    }
}

impl std::fmt::Display for InstructionId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "({}, 0x{:016x})", self.0.get(), self.1.data().as_ffi())
    }
}

#[cfg(feature = "python")]
mod python {
    use super::*;
    use pyo3::prelude::*;

    /// Python wrapper for `InstructionId` providing access to instruction identifiers.
    ///
    /// This wrapper allows Python code to work with instruction identifiers while
    /// maintaining type safety and preventing invalid construction.
    #[pyclass(name = "InstructionId", frozen, hash, eq, ord)]
    #[derive(Clone, Copy, Debug, Hash, PartialEq, Eq, Ord, PartialOrd)]
    pub struct PyInstructionId {
        inner: InstructionId,
    }

    #[pymethods]
    impl PyInstructionId {
        /// Returns the cycle identifier component of this instruction ID.
        #[getter]
        pub fn cycle(&self) -> CycleId {
            self.inner.cycle()
        }

        /// Returns the instruction identifier component within the cycle.
        #[getter]
        pub fn inner(&self) -> InstId {
            self.inner.inner()
        }

        /// Returns `True` if this instruction identifier references a valid cycle.
        pub fn is_valid(&self) -> bool {
            self.inner.is_valid()
        }

        /// Returns `True` if this instruction identifier references an invalid cycle.
        pub fn is_invalid(&self) -> bool {
            self.inner.is_invalid()
        }

        fn __repr__(&self) -> String {
            format!("InstructionId(cycle={:?}, inner={:?})", self.inner.cycle(), self.inner.inner())
        }

        fn __str__(&self) -> String {
            format!("({}, 0x{:016x})", self.inner.cycle().get(), self.inner.inner().data().as_ffi())
        }
    }

    impl<'py> IntoPyObject<'py> for InstructionId {
        type Target = PyInstructionId;
        type Output = Bound<'py, Self::Target>;
        type Error = PyErr;

        fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
            Bound::new(py, PyInstructionId { inner: self })
        }
    }

    impl<'a, 'py> FromPyObject<'a, 'py> for InstructionId {
        type Error = PyErr;

        fn extract(obj: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
            let py_inst_id: PyInstructionId = obj.extract()?;
            Ok(py_inst_id.inner)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cycle::InstId;
    use slotmap::KeyData;

    #[test]
    fn test_instruction_id_creation_and_getters() {
        let cycle_id = CycleId::new(42);
        let inst_id = InstId::from(KeyData::from_ffi(0x1234567890abcdef));
        let instruction_id = InstructionId::new(cycle_id, inst_id);

        assert_eq!(instruction_id.cycle(), cycle_id);
        assert_eq!(instruction_id.inner(), inst_id);
    }

    #[test]
    fn test_is_valid() {
        let valid_id = InstructionId::new(CycleId::new(0), InstId::default());
        let invalid_id = InstructionId::new(INVALID_CYCLE_ID, InstId::default());

        assert!(valid_id.is_valid());
        assert!(!valid_id.is_invalid());
        
        assert!(!invalid_id.is_valid());
        assert!(invalid_id.is_invalid());
    }

    #[test]
    fn test_display_format() {
        let cycle_id = CycleId::new(42);
        let inst_id = InstId::from(KeyData::from_ffi(0x1234567990abcdef));
        let instruction_id = InstructionId::new(cycle_id, inst_id);
        
        let display_str = format!("{}", instruction_id);
        assert_eq!(display_str, "(42, 0x1234567990abcdef)");
    }

    #[test]
    fn test_equality_and_ordering() {
        let id1 = InstructionId::new(CycleId::new(1), InstId::from(KeyData::from_ffi(100)));
        let id2 = InstructionId::new(CycleId::new(1), InstId::from(KeyData::from_ffi(100)));
        let id3 = InstructionId::new(CycleId::new(2), InstId::from(KeyData::from_ffi(100)));

        assert_eq!(id1, id2);
        assert_ne!(id1, id3);
        assert!(id1 < id3);
    }
}
