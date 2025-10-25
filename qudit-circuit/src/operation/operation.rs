use qudit_core::HasParams;
use qudit_core::QuditSystem;

use crate::operation::directive::DirectiveOperation;
use crate::operation::expression::ExpressionOperation;
use crate::operation::subcircuit::CircuitOperation;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Operation {
    Expression(ExpressionOperation),
    Subcircuit(CircuitOperation),
    Directive(DirectiveOperation),
}

impl Operation {
    pub fn num_qudits(&self) -> Option<usize> {
        match self {
            Operation::Expression(e) => Some(e.num_qudits()),
            Operation::Subcircuit(c) => Some(c.num_qudits()),
            Operation::Directive(d) => None,
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
    use pyo3::{exceptions::PyTypeError, prelude::*};
    use crate::python::PyCircuitRegistrar;

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

    impl<'py> FromPyObject<'py> for Operation {
        fn extract_bound(ob: &Bound<'py, PyAny>) -> PyResult<Self> {
            if let Ok(py_operation) = ob.extract::<PyOperation>() {
                Ok(py_operation.inner)
            } else if let Ok(expr_op) = ob.extract::<ExpressionOperation>() {
                Ok(Operation::Expression(expr_op))
            } else {
                return Err(PyTypeError::new_err("Unrecognized operation type."));
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
