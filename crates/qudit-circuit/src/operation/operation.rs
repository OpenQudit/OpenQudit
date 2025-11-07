use qudit_core::HasParams;
use qudit_core::QuditSystem;

use crate::operation::directive::DirectiveOperation;
use crate::operation::expression::ExpressionOperation;
use crate::operation::subcircuit::CircuitOperation;

/// An operation in a circuit
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Operation {
    /// Operations described by a symbolical expression
    Expression(ExpressionOperation),

    /// Operations capturing an entire subcircuit
    Subcircuit(CircuitOperation),

    /// Compiler directives
    Directive(DirectiveOperation),
}

impl Operation {
    /// Return the number of qudits this operation acts on
    pub fn num_qudits(&self) -> Option<usize> {
        match self {
            Operation::Expression(e) => Some(e.num_qudits()),
            Operation::Subcircuit(c) => Some(c.num_qudits()),
            Operation::Directive(_) => None,
        }
    }
}

impl<T: Into<ExpressionOperation>> From<T> for Operation {
    fn from(value: T) -> Self {
        Operation::Expression(value.into())
    }
}

impl HasParams for Operation {
    fn num_params(&self) -> usize {
        match self {
            Operation::Expression(e) => e.num_params(),
            Operation::Subcircuit(c) => c.num_params(),
            Operation::Directive(_) => 0,
        }
    }
}

#[cfg(feature = "python")]
mod python {
    use super::*;
    use crate::python::PyCircuitRegistrar;
    use pyo3::{exceptions::PyTypeError, prelude::*};

    #[pyclass(name = "Operation")]
    #[derive(Clone)]
    pub struct PyOperation {
        pub(crate) inner: Operation,
    }

    #[pymethods]
    impl PyOperation {
        fn num_qudits(&self) -> Option<usize> {
            self.inner.num_qudits()
        }

        fn num_params(&self) -> usize {
            self.inner.num_params()
        }

        fn __repr__(&self) -> String {
            format!("Operation({:?})", self.inner)
        }
    }

    impl From<Operation> for PyOperation {
        fn from(operation: Operation) -> Self {
            PyOperation { inner: operation }
        }
    }

    impl From<PyOperation> for Operation {
        fn from(py_operation: PyOperation) -> Self {
            py_operation.inner
        }
    }

    impl<'a, 'py> FromPyObject<'a, 'py> for Operation {
        type Error = PyErr;

        fn extract(obj: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
            if let Ok(py_operation) = obj.extract::<PyOperation>() {
                Ok(py_operation.inner)
            } else if let Ok(expr_op) = obj.extract::<ExpressionOperation>() {
                Ok(Operation::Expression(expr_op))
            } else {
                Err(PyTypeError::new_err("Unrecognized operation type."))
            }
        }
    }

    impl<'py> IntoPyObject<'py> for Operation {
        type Target = PyOperation;
        type Output = Bound<'py, Self::Target>;
        type Error = PyErr;

        fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
            let py_operation = PyOperation { inner: self };
            Bound::new(py, py_operation)
        }
    }

    // Registers the Operation class with the Python module.
    fn register(parent_module: &Bound<'_, PyModule>) -> PyResult<()> {
        parent_module.add_class::<PyOperation>()?;
        Ok(())
    }
    inventory::submit!(PyCircuitRegistrar { func: register });
}
