use std::any::Any;
use std::collections::HashMap;

use qudit_core::ComplexScalar;
use qudit_circuit::QuditCircuit;

use crate::instantiater;
use crate::InstantiationResult;
use crate::InstantiationTarget;

pub trait DataItem: Any + ToString {}

pub type DataMap = HashMap<String, Box<dyn DataItem>>;

// pub trait Instantiater<C: ComplexScalar> {
//     fn instantiate(
//         &self,
//         circuit: &QuditCircuit<C>,
//         target: &InstantiationTarget<C>,
//         data: &DataMap,
//     ) -> InstantiationResult<C>;


//     fn batched_instantiate(
//         &self,
//         circuit: &QuditCircuit<C>,
//         targets: &[&InstantiationTarget<C>],
//         data: &DataMap,
//     ) -> Vec<InstantiationResult<C>> {
//         targets.iter().map(|t| self.instantiate(circuit, t, data)).collect()
//     }
// }

pub trait Instantiater<'a, C: ComplexScalar> {
    fn instantiate(
        &'a self,
        circuit: &'a QuditCircuit,
        target: &'a InstantiationTarget<C>,
        data: &'a DataMap,
    ) -> InstantiationResult<C>;


    fn batched_instantiate(
        &'a self,
        circuit: &'a QuditCircuit,
        targets: &'a[&'a InstantiationTarget<C>],
        data: &'a DataMap,
    ) -> Vec<InstantiationResult<C>> {
        targets.iter().map(|t| self.instantiate(circuit, t, data)).collect()
    }
}

#[cfg(feature = "python")]
pub mod python {
    use pyo3::{exceptions::{PyNotImplementedError, PyTypeError}, prelude::*, types::{PyDict, PyList}};
    use super::*;
    use qudit_core::c64;
    use crate::python::PyInstantiationRegistrar;
    use dyn_clone::DynClone;

    pub trait InstantiaterWrapper: for <'a> Instantiater<'a, c64> + Send + Sync + DynClone {}

    #[pyclass(name = "NativeInstantiater")]
    pub struct BoxedInstantiater {
        pub inner: Box<dyn InstantiaterWrapper>,
    }

    impl<'a> Instantiater<'a, c64> for BoxedInstantiater {
        fn instantiate(
                &'a self,
                circuit: &'a QuditCircuit,
                target: &'a InstantiationTarget<c64>,
                data: &'a DataMap,
            ) -> InstantiationResult<c64> {
            self.inner.instantiate(circuit, target, data)
        }

        fn batched_instantiate(
                &'a self,
                circuit: &'a QuditCircuit,
                targets: &'a[&'a InstantiationTarget<c64>],
                data: &'a DataMap,
            ) -> Vec<InstantiationResult<c64>> {
            self.inner.batched_instantiate(circuit, targets, data)
        }
    }

    #[pyclass(name = "Instantiater", subclass)]
    struct PyInstantiaterABC;

    #[pymethods]
    impl PyInstantiaterABC {
        fn instantiate(&self, circuit: QuditCircuit, target: InstantiationTarget<c64>, data: &Bound<'_, PyDict>) -> PyResult<InstantiationResult<c64>> {
            Err(PyNotImplementedError::new_err(
                "Instantiaters must implement the instantiate method.",
            ))
        }
    }

    struct PyInstantiaterTrampoline {
        instantiater: Py<PyAny>,
    }

    impl<'a> Instantiater<'a, c64> for PyInstantiaterTrampoline {
        fn instantiate(
                &'a self,
                circuit: &'a QuditCircuit,
                target: &'a InstantiationTarget<c64>,
                data: &'a DataMap,
            ) -> InstantiationResult<c64> {
            // TODO: handle failures by not panicking, and propagating a python error
            Python::attach(|py| {

                let py_data = PyDict::new(py);
                for (key, val) in data.iter() {
                    py_data.set_item(key, val.to_string());
                }

                self.instantiater
                    .bind(py)
                    .call_method("instantiate", (circuit.clone(), target.clone(), py_data), None)
                    .unwrap()
                    .extract()
                    .expect("Invalid return type from instantiate.")
            })
        }

        fn batched_instantiate(
                &'a self,
                circuit: &'a QuditCircuit,
                targets: &'a[&'a InstantiationTarget<c64>],
                data: &'a DataMap,
            ) -> Vec<InstantiationResult<c64>> {
            // TODO: handle failures by not panicking, and propagating a python error
            Python::attach(|py| {
                let bound = self.instantiater.bind(py);

                let py_data = PyDict::new(py);
                for (key, val) in data.iter() {
                    py_data.set_item(key, val.to_string());
                }

                if bound.hasattr("batched_instantiate").is_ok_and(|x| x) {
                    let py_targets = PyList::new(py, targets.into_iter().map(|&t| t.clone())).unwrap();
                    bound.call_method("batched_instantiate", (circuit.clone(), py_targets, py_data), None)
                        .unwrap()
                        .extract()
                        .expect("Invalid return type from batched instantiate.")
                } else {
                    let circuit = circuit.clone().into_pyobject(py).unwrap();
                    targets.iter().map(|&t| bound.call_method("instantiate", (&circuit, t.clone(), &py_data), None)
                        .unwrap()
                        .extract()
                        .expect("Invalid return type from instantiate.")
                    ).collect()
                }
            })
        }
    }

    /// Other pyo3 code can use PyInstantiater as a parameter's type in pyfunctions and
    /// pymethods, and this can populated with either Python defined instantiaters
    /// or boxed rust ones. The GIL is not held with this object, and calls to the
    /// rust version are direct through the box without attaching to the GIL.
    pub enum PyInstantiater {
        Python(PyInstantiaterTrampoline),
        Native(BoxedInstantiater),
    }

    impl<'a> Instantiater<'a, c64> for PyInstantiater {
        fn instantiate(
                &'a self,
                circuit: &'a QuditCircuit,
                target: &'a InstantiationTarget<c64>,
                data: &'a DataMap,
            ) -> InstantiationResult<c64> {
            match self {
                PyInstantiater::Python(inner) => inner.instantiate(circuit, target, data),
                PyInstantiater::Native(inner) => inner.instantiate(circuit, target, data),
            }
        }

        fn batched_instantiate(
                &'a self,
                circuit: &'a QuditCircuit,
                targets: &'a[&'a InstantiationTarget<c64>],
                data: &'a DataMap,
            ) -> Vec<InstantiationResult<c64>> { 
            match self {
                PyInstantiater::Python(inner) => inner.batched_instantiate(circuit, targets, data),
                PyInstantiater::Native(inner) => inner.batched_instantiate(circuit, targets, data),
            }
        }
    }

    impl<'py> FromPyObject<'py> for PyInstantiater {
        fn extract_bound(ob: &Bound<'py, PyAny>) -> PyResult<Self> {
            if let Ok(dyn_trait) = ob.extract::<PyRef<BoxedInstantiater>>() {
                Ok(PyInstantiater::Native(BoxedInstantiater { inner: dyn_clone::clone_box(&*dyn_trait.inner) }))
            } else if ob.hasattr("instantiate")? {
                let trampoline = PyInstantiaterTrampoline { instantiater: ob.to_owned().unbind() };
                Ok(PyInstantiater::Python(trampoline))
            } else {
                Err(PyTypeError::new_err(
                    "Cannot extract an 'Instantiater' during conversion to native code.",
                )) 
            }

        }
    }
}

