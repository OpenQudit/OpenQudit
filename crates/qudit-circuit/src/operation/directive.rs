use qudit_core::{ParamIndices, Radices};

use crate::Result;
use crate::param::{IntoArgumentList, ParameterVector};
use crate::{OpCode, circuit::InternableOperation, operation::OperationSet};

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq, PartialOrd, Ord)]
#[repr(u64)]
pub enum DirectiveOperation {
    Barrier = 0,
}
impl DirectiveOperation {
    pub fn specialize(self, _args: crate::ArgumentList) -> Result<DirectiveOperation> {
        Ok(self)
    }
}

impl InternableOperation for DirectiveOperation {
    fn intern_operation(
        self,
        operation_set: &mut OperationSet,
        _parameter_vector: &mut ParameterVector,
        _args: impl IntoArgumentList,
        _qudit_radices: Radices,
        _dit_radices: Radices,
    ) -> Result<(OpCode, ParamIndices)> {
        let op_code = operation_set.convert_directive(self);
        operation_set.increment(op_code);
        match self {
            DirectiveOperation::Barrier => Ok((op_code, ParamIndices::empty())),
        }
    }
}

impl TryFrom<u64> for DirectiveOperation {
    type Error = crate::Error;

    fn try_from(value: u64) -> Result<Self> {
        match value {
            0 => Ok(DirectiveOperation::Barrier),
            _ => Err(crate::Error::GenericError(String::from(
                "Invalid directive operation discriminant provided.",
            ))),
        }
    }
}

impl TryFrom<usize> for DirectiveOperation {
    type Error = crate::Error;

    fn try_from(value: usize) -> Result<Self> {
        Self::try_from(value as u64)
    }
}

#[cfg(feature = "python")]
pub mod python {
    use super::*;
    use crate::python::PyCircuitRegistrar;
    use pyo3::prelude::*;
    use pyo3_stub_gen::{derive::gen_stub_pyclass_enum, impl_stub_type};

    impl_stub_type!(DirectiveOperation = PyDirectiveOperation);

    #[gen_stub_pyclass_enum]
    #[pyclass(
        name = "DirectiveOperation",
        module = "openqudit.circuit",
        eq,
        eq_int,
        from_py_object
    )]
    #[derive(Clone, Copy, PartialEq)]
    pub enum PyDirectiveOperation {
        Barrier,
    }

    impl From<PyDirectiveOperation> for DirectiveOperation {
        fn from(op: PyDirectiveOperation) -> Self {
            match op {
                PyDirectiveOperation::Barrier => DirectiveOperation::Barrier,
            }
        }
    }

    impl From<DirectiveOperation> for PyDirectiveOperation {
        fn from(op: DirectiveOperation) -> Self {
            match op {
                DirectiveOperation::Barrier => PyDirectiveOperation::Barrier,
            }
        }
    }

    impl<'py> IntoPyObject<'py> for DirectiveOperation {
        type Target = PyDirectiveOperation;
        type Output = Bound<'py, Self::Target>;
        type Error = PyErr;

        fn into_pyobject(self, py: Python<'py>) -> std::result::Result<Self::Output, Self::Error> {
            match self {
                DirectiveOperation::Barrier => Bound::new(py, PyDirectiveOperation::Barrier),
            }
        }
    }

    impl<'a, 'py> FromPyObject<'a, 'py> for DirectiveOperation {
        type Error = PyErr;

        fn extract(obj: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
            let py_dir_ref = obj.cast::<PyDirectiveOperation>()?;
            match py_dir_ref.borrow().to_owned() {
                PyDirectiveOperation::Barrier => Ok(DirectiveOperation::Barrier),
            }
        }
    }

    // Registers the Barrier class with the Python module.
    fn register(parent_module: &Bound<'_, PyModule>) -> PyResult<()> {
        parent_module.add_class::<PyDirectiveOperation>()?;
        Ok(())
    }
    inventory::submit!(PyCircuitRegistrar { func: register });
}
