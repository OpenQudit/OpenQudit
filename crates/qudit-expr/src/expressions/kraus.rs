use std::ops::{Deref, DerefMut};

use crate::{
    GenerationShape, TensorExpression,
    expressions::JittableExpression,
    index::{IndexDirection, TensorIndex},
};

use super::NamedExpression;
use qudit_core::QuditSystem;
use qudit_core::Radices;

#[derive(PartialEq, Eq, Debug, Clone)]
pub struct KrausOperatorsExpression {
    inner: NamedExpression,
    input_radices: Radices,
    output_radices: Radices,
    num_operators: usize,
}

impl KrausOperatorsExpression {
    pub fn new<T: AsRef<str>>(input: T) -> Self {
        TensorExpression::new(input).try_into().unwrap()
    }

    pub fn num_qudits(&self) -> usize {
        if self.input_radices == self.output_radices {
            self.input_radices.num_qudits()
        } else {
            panic!("Input and output number of qudits are different for kraus operator.")
        }
    }
}

impl JittableExpression for KrausOperatorsExpression {
    fn generation_shape(&self) -> GenerationShape {
        GenerationShape::Tensor3D(
            self.num_operators,
            self.output_radices.dimension(),
            self.input_radices.dimension(),
        )
    }
}

impl AsRef<NamedExpression> for KrausOperatorsExpression {
    fn as_ref(&self) -> &NamedExpression {
        &self.inner
    }
}

impl From<KrausOperatorsExpression> for NamedExpression {
    fn from(value: KrausOperatorsExpression) -> Self {
        value.inner
    }
}

impl Deref for KrausOperatorsExpression {
    type Target = NamedExpression;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl DerefMut for KrausOperatorsExpression {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.inner
    }
}

impl From<KrausOperatorsExpression> for TensorExpression {
    fn from(value: KrausOperatorsExpression) -> Self {
        let KrausOperatorsExpression {
            inner,
            input_radices,
            output_radices,
            num_operators,
        } = value;
        // TODO: add a proper implementation of into_iter for QuditRadices
        let indices = [num_operators]
            .into_iter()
            .map(|r| (IndexDirection::Batch, r))
            .chain(
                output_radices
                    .into_iter()
                    .map(|r| (IndexDirection::Output, usize::from(*r))),
            )
            .chain(
                input_radices
                    .into_iter()
                    .map(|r| (IndexDirection::Input, usize::from(*r))),
            )
            .enumerate()
            .map(|(i, (d, r))| TensorIndex::new(d, i, r))
            .collect();
        TensorExpression::from_raw(indices, inner)
    }
}

impl TryFrom<TensorExpression> for KrausOperatorsExpression {
    // TODO: Come up with proper error handling
    type Error = String;

    fn try_from(value: TensorExpression) -> Result<Self, Self::Error> {
        let mut num_operators = None;
        let mut input_radices = vec![];
        let mut output_radices = vec![];
        for idx in value.indices() {
            match idx.direction() {
                IndexDirection::Batch => match num_operators {
                    Some(n) => num_operators = Some(n * idx.index_size()),
                    None => num_operators = Some(idx.index_size()),
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

        Ok(KrausOperatorsExpression {
            inner: value.into(),
            input_radices: input_radices.into(),
            output_radices: output_radices.into(),
            num_operators: num_operators.unwrap_or(1),
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
    use qudit_core::Radix;
    use qudit_core::c64;

    #[pyclass]
    #[pyo3(name = "KrausOperatorsExpression")]
    pub struct PyKrausOperatorsExpression {
        expr: KrausOperatorsExpression,
    }

    #[pymethods]
    impl PyKrausOperatorsExpression {
        #[new]
        fn new(expr: String) -> Self {
            Self {
                expr: KrausOperatorsExpression::new(expr),
            }
        }

        fn num_params(&self) -> usize {
            self.expr.num_params()
        }

        fn name(&self) -> String {
            self.expr.name().to_string()
        }

        fn radices(&self) -> Vec<Radix> {
            self.expr.input_radices.to_vec()
        }

        fn num_qudits(&self) -> usize {
            self.expr.num_qudits()
        }

        fn num_operators(&self) -> usize {
            self.expr.num_operators
        }

        fn dimension(&self) -> usize {
            self.expr.input_radices.dimension()
        }

        fn __repr__(&self) -> String {
            format!(
                "KrausOperatorsExpression(name='{}', radices={:?}, num_operators={}, params={})",
                self.expr.name(),
                self.expr.input_radices.to_vec(),
                self.expr.num_operators,
                self.expr.num_params()
            )
        }
    }

    impl From<KrausOperatorsExpression> for PyKrausOperatorsExpression {
        fn from(value: KrausOperatorsExpression) -> Self {
            PyKrausOperatorsExpression { expr: value }
        }
    }

    impl From<PyKrausOperatorsExpression> for KrausOperatorsExpression {
        fn from(value: PyKrausOperatorsExpression) -> Self {
            value.expr
        }
    }

    impl<'py> IntoPyObject<'py> for KrausOperatorsExpression {
        type Target = <PyKrausOperatorsExpression as IntoPyObject<'py>>::Target;
        type Output = Bound<'py, Self::Target>;
        type Error = PyErr;

        fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
            let py_expr = PyKrausOperatorsExpression::from(self);
            Bound::new(py, py_expr)
        }
    }

    impl<'a, 'py> FromPyObject<'a, 'py> for KrausOperatorsExpression {
        type Error = PyErr;

        fn extract(ob: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
            let py_expr: PyRef<PyKrausOperatorsExpression> = ob.extract()?;
            Ok(py_expr.expr.clone())
        }
    }

    /// Registers the KrausOperatorsExpression class with the Python module.
    fn register(parent_module: &Bound<'_, PyModule>) -> PyResult<()> {
        parent_module.add_class::<PyKrausOperatorsExpression>()?;
        Ok(())
    }
    inventory::submit!(PyExpressionRegistrar { func: register });
}
