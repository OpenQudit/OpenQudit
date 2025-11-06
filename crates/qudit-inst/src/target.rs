use qudit_circuit::QuditCircuit;
use qudit_core::ComplexScalar;
use qudit_core::UnitaryMatrix;

#[derive(Clone)]
pub enum InstantiationTarget<C: ComplexScalar> {
    // Ket,
    // StateSystem
    // MixedState (Future, but worth thinking about in API)
    UnitaryMatrix(UnitaryMatrix<C>),
    // Kraus Operators (Future, but worth thinking about in API)
}

#[cfg(feature = "python")]
mod python {
    use super::InstantiationTarget;
    use crate::python::PyInstantiationRegistrar;
    use pyo3::prelude::*;
    use qudit_core::ComplexScalar;
    use qudit_core::UnitaryMatrix;
    use qudit_core::c64;

    #[pyclass(name = "InstantiationTarget")]
    #[derive(Clone)]
    pub struct PyInstantiationTarget {
        inner: InstantiationTarget<c64>,
    }

    #[pymethods]
    impl PyInstantiationTarget {
        #[new]
        pub fn new<'py>(obj: &Bound<'py, PyAny>) -> PyResult<Self> {
            // Try to extract as UnitaryMatrix
            if let Ok(unitary) = obj.extract::<UnitaryMatrix<c64>>() {
                return Ok(Self {
                    inner: InstantiationTarget::UnitaryMatrix(unitary),
                });
            }

            // Future: Add other variants here
            // if let Ok(ket) = obj.extract::<Ket<c64>>() {
            //     return Ok(Self {
            //         inner: InstantiationTarget::Ket(ket),
            //     });
            // }

            Err(pyo3::exceptions::PyTypeError::new_err(
                "Cannot convert object to InstantiationTarget. Expected UnitaryMatrix.",
            ))
        }

        pub fn is_unitary_matrix(&self) -> bool {
            matches!(self.inner, InstantiationTarget::UnitaryMatrix(_))
        }
    }

    impl<'py> IntoPyObject<'py> for InstantiationTarget<c64> {
        type Target = PyInstantiationTarget;
        type Output = Bound<'py, PyInstantiationTarget>;
        type Error = PyErr;

        fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
            let py_target = PyInstantiationTarget { inner: self };
            Bound::new(py, py_target)
        }
    }

    impl<'a, 'py> FromPyObject<'a, 'py> for InstantiationTarget<c64> {
        type Error = PyErr;

        fn extract(obj: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
            if let Ok(py_target) = obj.extract::<PyInstantiationTarget>() {
                return Ok(py_target.inner);
            }

            let py_target = PyInstantiationTarget::new(&*obj)?;
            Ok(py_target.inner)
        }
    }

    /// Registers the InstantiationTarget class with the Python module.
    fn register(parent_module: &Bound<'_, PyModule>) -> PyResult<()> {
        parent_module.add_class::<PyInstantiationTarget>()?;
        Ok(())
    }
    inventory::submit!(PyInstantiationRegistrar { func: register });
}
