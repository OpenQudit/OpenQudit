use std::ops::Deref;
use std::ops::DerefMut;

use crate::GenerationShape;
use crate::TensorExpression;
use crate::expressions::JittableExpression;
use crate::index::IndexDirection;
use crate::index::TensorIndex;

use super::NamedExpression;
use qudit_core::QuditSystem;
use qudit_core::Radices;

#[derive(PartialEq, Eq, Debug, Clone)]
pub struct BraExpression {
    inner: NamedExpression,
    radices: Radices,
}

impl BraExpression {
    pub fn new<T: AsRef<str>>(input: T) -> Self {
        TensorExpression::new(input).try_into().unwrap()
    }
}

impl JittableExpression for BraExpression {
    fn generation_shape(&self) -> GenerationShape {
        GenerationShape::Vector(self.radices.dimension())
    }
}

impl AsRef<NamedExpression> for BraExpression {
    fn as_ref(&self) -> &NamedExpression {
        &self.inner
    }
}

impl From<BraExpression> for NamedExpression {
    fn from(value: BraExpression) -> Self {
        value.inner
    }
}

impl Deref for BraExpression {
    type Target = NamedExpression;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl DerefMut for BraExpression {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.inner
    }
}

impl From<BraExpression> for TensorExpression {
    fn from(value: BraExpression) -> Self {
        let BraExpression { inner, radices } = value;
        // TODO: add a proper implementation of into_iter for QuditRadices
        let indices = radices
            .into_iter()
            .enumerate()
            .map(|(i, r)| TensorIndex::new(IndexDirection::Input, i, usize::from(*r)))
            .collect();
        TensorExpression::from_raw(indices, inner)
    }
}

impl TryFrom<TensorExpression> for BraExpression {
    // TODO: Come up with proper error handling
    type Error = String;

    fn try_from(value: TensorExpression) -> Result<Self, Self::Error> {
        if value
            .indices()
            .iter()
            .any(|idx| idx.direction() != IndexDirection::Input)
        {
            return Err(String::from(
                "Cannot convert a tensor with non-input indices to a bra.",
            ));
        }
        let radices = Radices::from_iter(value.indices().iter().map(|idx| idx.index_size()));
        Ok(BraExpression {
            inner: value.into(),
            radices,
        })
    }
}

impl QuditSystem for BraExpression {
    fn radices(&self) -> Radices {
        self.radices.clone()
    }

    fn num_qudits(&self) -> usize {
        self.radices.num_qudits()
    }
}

#[cfg(feature = "python")]
mod python {
    use super::*;
    use crate::python::PyExpressionRegistrar;
    use pyo3::prelude::*;
    use qudit_core::Radix;
    

    #[pyclass]
    #[pyo3(name = "BraExpression")]
    pub struct PyBraExpression {
        expr: BraExpression,
    }

    #[pymethods]
    impl PyBraExpression {
        #[new]
        fn new(expr: String) -> Self {
            Self {
                expr: BraExpression::new(expr),
            }
        }

        fn num_params(&self) -> usize {
            self.expr.num_params()
        }

        fn name(&self) -> String {
            self.expr.name().to_string()
        }

        fn radices(&self) -> Vec<Radix> {
            self.expr.radices().to_vec()
        }

        fn dimension(&self) -> usize {
            self.expr.dimension()
        }

        fn __repr__(&self) -> String {
            format!(
                "BraExpression(name='{}', radices={:?}, params={})",
                self.expr.name(),
                self.expr.radices().to_vec(),
                self.expr.num_params()
            )
        }
    }

    impl From<BraExpression> for PyBraExpression {
        fn from(value: BraExpression) -> Self {
            PyBraExpression { expr: value }
        }
    }

    impl From<PyBraExpression> for BraExpression {
        fn from(value: PyBraExpression) -> Self {
            value.expr
        }
    }

    impl<'py> IntoPyObject<'py> for BraExpression {
        type Target = <PyBraExpression as IntoPyObject<'py>>::Target;
        type Output = Bound<'py, Self::Target>;
        type Error = PyErr;

        fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
            let py_expr = PyBraExpression::from(self);
            Bound::new(py, py_expr)
        }
    }

    impl<'a, 'py> FromPyObject<'a, 'py> for BraExpression {
        type Error = PyErr;

        fn extract(ob: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
            let py_expr: PyRef<PyBraExpression> = ob.extract()?;
            Ok(py_expr.expr.clone())
        }
    }

    /// Registers the BraExpression class with the Python module.
    fn register(parent_module: &Bound<'_, PyModule>) -> PyResult<()> {
        parent_module.add_class::<PyBraExpression>()?;
        Ok(())
    }
    inventory::submit!(PyExpressionRegistrar { func: register });
}
