/// Represents the kind of operation.
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq, PartialOrd, Ord)]
#[repr(u8)]
pub enum OpKind {
    /// Expression operation.
    Expression = 0,
    /// Subcircuit operation.
    Subcircuit = 1,
    /// Directive operation.
    Directive = 2,
}

impl OpKind {
    /// Attempts to create an OpKind from a u8 value.
    pub const fn from_u8(value: u8) -> Option<Self> {
        match value {
            0 => Some(OpKind::Expression),
            1 => Some(OpKind::Subcircuit),
            2 => Some(OpKind::Directive),
            _ => None,
        }
    }
}

#[cfg(feature = "python")]
mod python {
    use super::*;
    use crate::python::PyCircuitRegistrar;
    use pyo3::prelude::*;

    /// Python wrapper for OpKind.
    #[pyclass(frozen, eq, hash, ord)]
    #[derive(Clone, Copy, Debug, Hash, PartialEq, Eq, PartialOrd, Ord)]
    pub struct PyOpKind(OpKind);

    #[pymethods]
    impl PyOpKind {
        /// Get integer representation.
        fn __int__(&self) -> u8 {
            self.0 as u8
        }

        /// String representation.
        fn __str__(&self) -> &'static str {
            match self.0 {
                OpKind::Expression => "Expression",
                OpKind::Subcircuit => "Subcircuit",
                OpKind::Directive => "Directive",
            }
        }

        /// Debug representation.
        fn __repr__(&self) -> String {
            format!("OpKind.{}", self.__str__())
        }
    }

    impl From<OpKind> for PyOpKind {
        fn from(kind: OpKind) -> Self {
            Self(kind)
        }
    }

    impl From<PyOpKind> for OpKind {
        fn from(py_kind: PyOpKind) -> Self {
            py_kind.0
        }
    }

    impl<'py> IntoPyObject<'py> for OpKind {
        type Target = PyOpKind;
        type Output = Bound<'py, Self::Target>;
        type Error = PyErr;

        fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
            Bound::new(py, PyOpKind::from(self))
        }
    }

    impl<'a, 'py> FromPyObject<'a, 'py> for OpKind {
        type Error = PyErr;

        fn extract(obj: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
            if let Ok(py_kind) = obj.extract::<PyOpKind>() {
                Ok(py_kind.0)
            } else if let Ok(value) = obj.extract::<u8>() {
                match value {
                    0 => Ok(OpKind::Expression),
                    1 => Ok(OpKind::Subcircuit),
                    2 => Ok(OpKind::Directive),
                    _ => Err(pyo3::exceptions::PyValueError::new_err("Invalid OpKind value")),
                }
            } else {
                Err(pyo3::exceptions::PyTypeError::new_err("Expected OpKind or int"))
            }
        }
    }

    /// Registers the PyOpKind class with the Python module.
    fn register(parent_module: &Bound<'_, PyModule>) -> PyResult<()> {
        parent_module.add_class::<PyOpKind>()?;
        Ok(())
    }
    inventory::submit!(PyCircuitRegistrar { func: register });
}

