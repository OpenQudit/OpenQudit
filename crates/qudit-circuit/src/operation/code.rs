use super::kind::OpKind;

/// Encodes an operation kind and persistent reference in a 64-bit value.
#[derive(Clone, Debug, Hash, PartialEq, Eq, Copy, PartialOrd, Ord)]
#[repr(transparent)]
pub struct OpCode(u64);

impl OpCode {
    /// Number of bits reserved for operation kind
    const KIND_BITS: u32 = 3;

    /// Bit position where the operation kind is stored.
    const KIND_SHIFT: u32 = 64 - Self::KIND_BITS;

    /// Mask for extracting the operation kind bits.
    const KIND_MASK: u64 = ((1 << Self::KIND_BITS) - 1) << Self::KIND_SHIFT;

    /// Mask for extracting the reference ID bits.
    const REF_MASK: u64 = !Self::KIND_MASK;

    /// Creates an OpCode from an operation kind and reference ID.
    ///
    /// Packs the operation kind into the upper 3 bits and the reference ID into
    /// the remaining bits of a 64-bit value.
    ///
    /// # Arguments
    ///
    /// * `kind` - Operation kind, stored in upper 3 bits
    /// * `id` - Persistent reference ID, must fit in remaining bits
    ///
    /// # Panics
    ///
    /// Panics if `id` exceeds capacity and would overflow into kind bits.
    #[inline]
    pub(crate) fn new(kind: OpKind, id: u64) -> Self {
        if id & Self::KIND_MASK != 0 {
            panic!("Operation reference overflow.");
        }

        let kind = ((kind as u8) as u64) << Self::KIND_SHIFT;
        let id = id & Self::REF_MASK;
        OpCode(kind | id)
    }

    #[inline(always)]
    fn kind_bits(code: u64) -> u8 {
        ((code & Self::KIND_MASK) >> Self::KIND_SHIFT) as u8
    }

    /// Returns the operation kind encoded in this OpCode.
    #[inline(always)]
    pub fn kind(&self) -> OpKind {
        // SAFETY: OpKind is repr(u8) and we only store valid values (0-2)
        unsafe { std::mem::transmute(Self::kind_bits(self.0)) }
    }

    /// Returns the persistent reference ID encoded in this OpCode.
    #[inline(always)]
    pub fn id(&self) -> u64 {
        self.0 & Self::REF_MASK
    }
}

#[cfg(feature = "python")]
mod python {
    use super::*;
    use crate::python::PyCircuitRegistrar;
    use pyo3::prelude::*;

    /// Python wrapper for the OpCode struct.
    ///
    /// Provides access to quantum operation code functionality from Python.
    #[pyclass(name = "OpCode", frozen, eq, ord, hash)]
    #[derive(Clone, Debug, Hash, PartialEq, Eq, PartialOrd, Ord)]
    pub struct PyOpCode {
        inner: OpCode,
    }

    #[pymethods]
    impl PyOpCode {
        /// Returns the operation kind.
        #[getter]
        fn kind(&self) -> OpKind {
            self.inner.kind()
        }

        /// Returns the operation reference ID.
        #[getter]
        fn id(&self) -> u64 {
            self.inner.id()
        }

        fn __repr__(&self) -> String {
            format!("OpCode({:016x})", self.inner.0)
        }

        fn __str__(&self) -> String {
            format!("{:016x}", self.inner.0)
        }
    }

    impl From<OpCode> for PyOpCode {
        fn from(opcode: OpCode) -> Self {
            PyOpCode { inner: opcode }
        }
    }

    impl From<PyOpCode> for OpCode {
        fn from(py_opcode: PyOpCode) -> Self {
            py_opcode.inner
        }
    }

    impl<'py> IntoPyObject<'py> for OpCode {
        type Target = PyOpCode;
        type Output = Bound<'py, Self::Target>;
        type Error = PyErr;

        fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
            Bound::new(py, PyOpCode::from(self))
        }
    }

    impl<'a, 'py> FromPyObject<'a, 'py> for OpCode {
        type Error = PyErr;

        fn extract(obj: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
            if let Ok(py_opcode) = obj.extract::<PyOpCode>() {
                Ok(py_opcode.into())
            } else if let Ok(value) = obj.extract::<u64>() {
                // Validate that the kind bits represent a valid OpKind
                let kind_bits = OpCode::kind_bits(value);
                if OpKind::from_u8(kind_bits).is_none() {
                    return Err(pyo3::exceptions::PyValueError::new_err(format!(
                        "Invalid OpKind value {} in OpCode",
                        kind_bits
                    )));
                }
                Ok(OpCode(value))
            } else {
                Err(pyo3::exceptions::PyTypeError::new_err(
                    "Expected OpCode or int",
                ))
            }
        }
    }

    /// Registers the OpCode class with the Python module.
    fn register(parent_module: &Bound<'_, PyModule>) -> PyResult<()> {
        parent_module.add_class::<PyOpCode>()?;
        Ok(())
    }
    inventory::submit!(PyCircuitRegistrar { func: register });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn creates_and_extracts_correctly() {
        let opcode = OpCode::new(OpKind::Subcircuit, 12345);
        assert_eq!(opcode.kind(), OpKind::Subcircuit);
        assert_eq!(opcode.id(), 12345);
    }

    #[test]
    fn handles_max_valid_id() {
        let max_id = OpCode::REF_MASK;
        let opcode = OpCode::new(OpKind::Directive, max_id);
        assert_eq!(opcode.kind(), OpKind::Directive);
        assert_eq!(opcode.id(), max_id);
    }

    #[test]
    #[should_panic(expected = "Operation reference overflow")]
    fn panics_on_id_overflow() {
        OpCode::new(OpKind::Expression, OpCode::REF_MASK + 1);
    }
}
