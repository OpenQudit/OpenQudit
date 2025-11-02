use std::any::Any;
use std::collections::HashMap;
use std::sync::Arc;

use qudit_core::ComplexScalar;
use qudit_circuit::QuditCircuit;

use crate::instantiater;
use crate::InstantiationResult;
use crate::InstantiationTarget;

pub trait DataItem: Any + ToString {}

impl<T: Any + ToString> DataItem for T {}

pub type DataMap = HashMap<String, Box<dyn DataItem>>;

pub trait Instantiater<C: ComplexScalar> {
    fn instantiate(
        &self,
        circuit: Arc<QuditCircuit>,
        target: Arc<InstantiationTarget<C>>,
        data: Arc<DataMap>,
    ) -> InstantiationResult<C>;


    fn batched_instantiate(
        &self,
        circuit: Arc<QuditCircuit>,
        targets: &[Arc<InstantiationTarget<C>>],
        data: Arc<DataMap>,
    ) -> Vec<InstantiationResult<C>> {
        targets.iter().map(|t| self.instantiate(circuit.clone(), t.clone(), data.clone())).collect()
    }
}

#[cfg(feature = "python")]
pub mod python {
    use pyo3::{exceptions::{PyNotImplementedError, PyTypeError}, prelude::*, types::{PyDict, PyList}};
    use super::*;
    use qudit_core::c64;
    use crate::python::PyInstantiationRegistrar;
    use dyn_clone::DynClone;
   
    fn pydict_to_datamap(py_dict: Option<&Bound<'_, PyDict>>) -> PyResult<Arc<DataMap>> {
        let mut data_map = HashMap::new();

        match py_dict {
            None => Ok(Arc::new(data_map)),
            Some(py_dict) => {
                for (key, value) in py_dict.iter() {
                    let key_str: String = key.extract()?;
                    let value_str: String = value.extract()?;
                    data_map.insert(key_str, Box::new(value_str) as Box<dyn DataItem>);
                }
                Ok(Arc::new(data_map))
            }
        }
    }

    pub trait InstantiaterWrapper: Instantiater<c64> + Send + Sync + DynClone {}

    #[pyclass(name = "NativeInstantiater")]
    pub struct BoxedInstantiater {
        pub inner: Box<dyn InstantiaterWrapper>,
    }

    #[pymethods]
    impl BoxedInstantiater {

        #[pyo3(name = "instantiate")]
        #[pyo3(signature = (circuit, target, data = None))]
        fn instantiate_python(
            &self,
            circuit: QuditCircuit,
            target: InstantiationTarget<c64>,
            data: Option<&Bound<'_, PyDict>>,
        ) -> PyResult<InstantiationResult<c64>> {
            let data_map = pydict_to_datamap(data)?;
            let result = Instantiater::instantiate(
                self,
                Arc::new(circuit),
                Arc::new(target),
                data_map,
            );
            Ok(result)
        }

        #[pyo3(name = "batched_instantiate")]
        #[pyo3(signature = (circuit, targets, data = None))]
        fn batched_instantiate_python(
            &self,
            circuit: QuditCircuit,
            targets: Vec<InstantiationTarget<c64>>,
            data: Option<&Bound<'_, PyDict>>,
        ) -> PyResult<Vec<InstantiationResult<c64>>> {
            let data_map = pydict_to_datamap(data)?;
            let target_arcs: Vec<Arc<InstantiationTarget<c64>>> = targets
                .into_iter()
                .map(Arc::new)
                .collect();
            let result = Instantiater::batched_instantiate(
                self,
                Arc::new(circuit),
                &target_arcs,
                data_map,
            );
            Ok(result)
        }
    }

    impl Instantiater<c64> for BoxedInstantiater {
        fn instantiate(
            &self,
            circuit: Arc<QuditCircuit>,
            target: Arc<InstantiationTarget<c64>>,
            data: Arc<DataMap>,
        ) -> InstantiationResult<c64> {
            self.inner.instantiate(circuit, target, data)
        }

        fn batched_instantiate(
            &self,
            circuit: Arc<QuditCircuit>,
            targets: &[Arc<InstantiationTarget<c64>>],
            data: Arc<DataMap>,
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

    impl Instantiater<c64> for PyInstantiaterTrampoline {
        fn instantiate(
            &self,
            circuit: Arc<QuditCircuit>,
            target: Arc<InstantiationTarget<c64>>,
            data: Arc<DataMap>,
        ) -> InstantiationResult<c64> {
            // TODO: handle failures by not panicking, and propagating a python error
            Python::attach(|py| {

                let py_data = PyDict::new(py);
                for (key, val) in data.iter() {
                    py_data.set_item(key, val.to_string());
                }

                self.instantiater
                    .bind(py)
                    .call_method("instantiate", ((*circuit).clone(), (*target).clone(), py_data), None)
                    .unwrap()
                    .extract()
                    .expect("Invalid return type from instantiate.")
            })
        }

        fn batched_instantiate(
            &self,
            circuit: Arc<QuditCircuit>,
            targets: &[Arc<InstantiationTarget<c64>>],
            data: Arc<DataMap>,
        ) -> Vec<InstantiationResult<c64>> {
            // TODO: handle failures by not panicking, and propagating a python error
            Python::attach(|py| {
                let bound = self.instantiater.bind(py);

                let py_data = PyDict::new(py);
                for (key, val) in data.iter() {
                    py_data.set_item(key, val.to_string());
                }

                if bound.hasattr("batched_instantiate").is_ok_and(|x| x) {
                    let py_targets = PyList::new(py, targets.into_iter().map(|t| (**t).clone())).unwrap();
                    bound.call_method("batched_instantiate", ((*circuit).clone(), py_targets, py_data), None)
                        .unwrap()
                        .extract()
                        .expect("Invalid return type from batched instantiate.")
                } else {
                    let circuit = (*circuit).clone().into_pyobject(py).unwrap();
                    targets.iter().map(|t| bound.call_method("instantiate", (&circuit, (**t).clone(), &py_data), None)
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

    impl Instantiater<c64> for PyInstantiater {
        fn instantiate(
            &self,
            circuit: Arc<QuditCircuit>,
            target: Arc<InstantiationTarget<c64>>,
            data: Arc<DataMap>,
        ) -> InstantiationResult<c64> {
            match self {
                PyInstantiater::Python(inner) => inner.instantiate(circuit, target, data),
                PyInstantiater::Native(inner) => inner.instantiate(circuit, target, data),
            }
        }

        fn batched_instantiate(
            &self,
            circuit: Arc<QuditCircuit>,
            targets: &[Arc<InstantiationTarget<c64>>],
            data: Arc<DataMap>,
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

