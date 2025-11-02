use std::ops::{Deref, DerefMut};

use crate::{
    expressions::JittableExpression,
    index::{IndexDirection, TensorIndex},
    GenerationShape, TensorExpression,
};

use super::NamedExpression;
use qudit_core::QuditSystem;
use qudit_core::Radices;

#[derive(PartialEq, Eq, Debug, Clone)]
pub struct UnitarySystemExpression {
    inner: NamedExpression,
    radices: Radices,
    num_unitaries: usize,
}

impl UnitarySystemExpression {
    pub fn new<T: AsRef<str>>(input: T) -> Self {
        TensorExpression::new(input).try_into().unwrap()
    }

    pub fn num_qudits(&self) -> usize {
        self.radices.num_qudits()
    }
}

impl JittableExpression for UnitarySystemExpression {
    fn generation_shape(&self) -> GenerationShape {
        GenerationShape::Tensor3D(
            self.num_unitaries,
            self.radices.dimension(),
            self.radices.dimension(),
        )
    }
}

impl AsRef<NamedExpression> for UnitarySystemExpression {
    fn as_ref(&self) -> &NamedExpression {
        &self.inner
    }
}

impl From<UnitarySystemExpression> for NamedExpression {
    fn from(value: UnitarySystemExpression) -> Self {
        value.inner
    }
}

impl Deref for UnitarySystemExpression {
    type Target = NamedExpression;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl DerefMut for UnitarySystemExpression {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.inner
    }
}

impl From<UnitarySystemExpression> for TensorExpression {
    fn from(value: UnitarySystemExpression) -> Self {
        let UnitarySystemExpression {
            inner,
            radices,
            num_unitaries,
        } = value;
        // TODO: add a proper implementation of into_iter for QuditRadices
        let indices = [num_unitaries]
            .into_iter()
            .map(|r| (IndexDirection::Batch, r))
            .chain(
                radices
                    .into_iter()
                    .map(|r| (IndexDirection::Output, usize::from(*r))),
            )
            .chain(
                radices
                    .into_iter()
                    .map(|r| (IndexDirection::Input, usize::from(*r))),
            )
            .enumerate()
            .map(|(i, (d, r))| TensorIndex::new(d, i, r))
            .collect();
        TensorExpression::from_raw(indices, inner)
    }
}

impl TryFrom<TensorExpression> for UnitarySystemExpression {
    // TODO: Come up with proper error handling
    type Error = String;

    fn try_from(value: TensorExpression) -> Result<Self, Self::Error> {
        let mut num_unitaries = None;
        let mut input_radices = vec![];
        let mut output_radices = vec![];
        for idx in value.indices() {
            match idx.direction() {
                IndexDirection::Batch => match &mut num_unitaries {
                    Some(n) => *n *= idx.index_size(),
                    None => num_unitaries = Some(idx.index_size()),
                },
                IndexDirection::Input => {
                    input_radices.push(idx.index_size());
                }
                IndexDirection::Output => {
                    output_radices.push(idx.index_size());
                }
                _ => unreachable!(),
            }
        }

        if input_radices != output_radices {
            return Err(String::from(
                "Non-square matrix tensor cannot be converted to a unitary.",
            ));
        }

        Ok(UnitarySystemExpression {
            inner: value.into(),
            radices: input_radices.into(),
            num_unitaries: num_unitaries.unwrap_or(1),
        })
    }
}

#[cfg(feature = "python")]
mod python {
    use super::*;
    use crate::python::PyExpressionRegistrar;
    use ndarray::ArrayViewMut3;
    use numpy::PyArray3;
    use numpy::PyArrayMethods;
    use pyo3::prelude::*;
    use pyo3::types::PyTuple;
    use qudit_core::c64;
    use qudit_core::Radix;

    #[pyclass]
    #[pyo3(name = "UnitarySystemExpression")]
    pub struct PyUnitarySystemExpression {
        expr: UnitarySystemExpression,
    }

    #[pymethods]
    impl PyUnitarySystemExpression {
        #[new]
        fn new(expr: String) -> Self {
            Self {
                expr: UnitarySystemExpression::new(expr),
            }
        }

        fn num_params(&self) -> usize {
            self.expr.num_params()
        }

        fn name(&self) -> String {
            self.expr.name().to_string()
        }

        fn radices(&self) -> Vec<Radix> {
            self.expr.radices.to_vec()
        }

        fn num_qudits(&self) -> usize {
            self.expr.num_qudits()
        }

        fn num_unitaries(&self) -> usize {
            self.expr.num_unitaries
        }

        fn dimension(&self) -> usize {
            self.expr.radices.dimension()
        }

        fn __repr__(&self) -> String {
            format!(
                "UnitarySystemExpression(name='{}', radices={:?}, num_unitaries={}, params={})",
                self.expr.name(),
                self.expr.radices.to_vec(),
                self.expr.num_unitaries,
                self.expr.num_params()
            )
        }
    }

    impl From<UnitarySystemExpression> for PyUnitarySystemExpression {
        fn from(value: UnitarySystemExpression) -> Self {
            PyUnitarySystemExpression { expr: value }
        }
    }

    impl From<PyUnitarySystemExpression> for UnitarySystemExpression {
        fn from(value: PyUnitarySystemExpression) -> Self {
            value.expr
        }
    }

    impl<'py> IntoPyObject<'py> for UnitarySystemExpression {
        type Target = <PyUnitarySystemExpression as IntoPyObject<'py>>::Target;
        type Output = Bound<'py, Self::Target>;
        type Error = PyErr;

        fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
            let py_expr = PyUnitarySystemExpression::from(self);
            Bound::new(py, py_expr)
        }
    }

    impl<'py> FromPyObject<'py> for UnitarySystemExpression {
        fn extract_bound(ob: &Bound<'py, PyAny>) -> PyResult<Self> {
            let py_expr: PyRef<PyUnitarySystemExpression> = ob.extract()?;
            Ok(py_expr.expr.clone())
        }
    }

    /// Registers the UnitarySystemExpression class with the Python module.
    fn register(parent_module: &Bound<'_, PyModule>) -> PyResult<()> {
        parent_module.add_class::<PyUnitarySystemExpression>()?;
        Ok(())
    }
    inventory::submit!(PyExpressionRegistrar { func: register });
}
