use std::ops::{Deref, DerefMut};

use crate::{
    expressions::JittableExpression,
    index::{IndexDirection, TensorIndex},
    GenerationShape, TensorExpression,
};

use super::ComplexExpression;
use super::NamedExpression;
use qudit_core::QuditSystem;
use qudit_core::Radices;

#[derive(PartialEq, Eq, Debug, Clone)]
pub struct KetExpression {
    inner: NamedExpression,
    radices: Radices,
}

impl KetExpression {
    pub fn new<T: AsRef<str>>(input: T) -> Self {
        TensorExpression::new(input).try_into().unwrap()
    }

    pub fn zero<R: Into<Radices>>(radices: R) -> Self {
        let name = "zero";
        let radices = radices.into();
        let mut body = vec![ComplexExpression::zero(); radices.dimension()];
        body[0] = ComplexExpression::one();
        let variables = vec![];
        let inner = NamedExpression::new(name, variables, body);
        KetExpression { inner, radices }
    }
}

impl JittableExpression for KetExpression {
    fn generation_shape(&self) -> GenerationShape {
        GenerationShape::Matrix(self.radices.dimension(), 1)
    }
}

impl AsRef<NamedExpression> for KetExpression {
    fn as_ref(&self) -> &NamedExpression {
        &self.inner
    }
}

impl From<KetExpression> for NamedExpression {
    fn from(value: KetExpression) -> Self {
        value.inner
    }
}

impl Deref for KetExpression {
    type Target = NamedExpression;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl DerefMut for KetExpression {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.inner
    }
}

impl QuditSystem for KetExpression {
    fn radices(&self) -> Radices {
        self.radices.clone()
    }

    fn num_qudits(&self) -> usize {
        self.radices().num_qudits()
    }
}

impl From<KetExpression> for TensorExpression {
    fn from(value: KetExpression) -> Self {
        let KetExpression { inner, radices } = value;
        // TODO: add a proper implementation of into_iter for QuditRadices
        let indices = radices
            .into_iter()
            .enumerate()
            .map(|(i, r)| TensorIndex::new(IndexDirection::Output, i, usize::from(*r)))
            .collect();
        TensorExpression::from_raw(indices, inner)
    }
}

impl TryFrom<TensorExpression> for KetExpression {
    // TODO: Come up with proper error handling
    type Error = String;

    fn try_from(value: TensorExpression) -> Result<Self, Self::Error> {
        if value
            .indices()
            .iter()
            .any(|idx| idx.direction() != IndexDirection::Output)
        {
            return Err(String::from(
                "Cannot convert a tensor with non-output indices to a ket.",
            ));
        }
        let radices = Radices::from_iter(value.indices().iter().map(|idx| idx.index_size()));
        Ok(KetExpression {
            inner: value.into(),
            radices,
        })
    }
}

#[cfg(feature = "python")]
mod python {
    use super::*;
    use crate::python::PyExpressionRegistrar;
    use ndarray::ArrayViewMut2;
    use numpy::PyArray2;
    use numpy::PyArrayMethods;
    use pyo3::prelude::*;
    use pyo3::types::PyTuple;
    use qudit_core::c64;
    use qudit_core::Radix;

    #[pyclass]
    #[pyo3(name = "KetExpression")]
    pub struct PyKetExpression {
        expr: KetExpression,
    }

    #[pymethods]
    impl PyKetExpression {
        #[new]
        fn new(expr: String) -> Self {
            Self {
                expr: KetExpression::new(expr),
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
                "KetExpression(name='{}', radices={:?}, params={})",
                self.expr.name(),
                self.expr.radices().to_vec(),
                self.expr.num_params()
            )
        }
    }

    impl From<KetExpression> for PyKetExpression {
        fn from(value: KetExpression) -> Self {
            PyKetExpression { expr: value }
        }
    }

    impl From<PyKetExpression> for KetExpression {
        fn from(value: PyKetExpression) -> Self {
            value.expr
        }
    }

    impl<'py> IntoPyObject<'py> for KetExpression {
        type Target = <PyKetExpression as IntoPyObject<'py>>::Target;
        type Output = Bound<'py, Self::Target>;
        type Error = PyErr;

        fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
            let py_expr = PyKetExpression::from(self);
            Bound::new(py, py_expr)
        }
    }

    impl<'py> FromPyObject<'py> for KetExpression {
        fn extract_bound(ob: &Bound<'py, PyAny>) -> PyResult<Self> {
            let py_expr: PyRef<PyKetExpression> = ob.extract()?;
            Ok(py_expr.expr.clone())
        }
    }

    /// Registers the KetExpression class with the Python module.
    fn register(parent_module: &Bound<'_, PyModule>) -> PyResult<()> {
        parent_module.add_class::<PyKetExpression>()?;
        Ok(())
    }
    inventory::submit!(PyExpressionRegistrar { func: register });
}
