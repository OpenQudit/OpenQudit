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
pub struct BraSystemExpression {
    inner: NamedExpression,
    radices: Radices,
    num_states: usize,
}

impl BraSystemExpression {
    pub fn new<T: AsRef<str>>(input: T) -> Self {
        TensorExpression::new(input).try_into().unwrap()
    }

    pub fn num_qudits(&self) -> usize {
        self.radices.num_qudits()
    }
}

impl JittableExpression for BraSystemExpression {
    fn generation_shape(&self) -> GenerationShape {
        GenerationShape::Tensor3D(self.num_states, 1, self.radices.dimension())
    }
}

impl AsRef<NamedExpression> for BraSystemExpression {
    fn as_ref(&self) -> &NamedExpression {
        &self.inner
    }
}

impl From<BraSystemExpression> for NamedExpression {
    fn from(value: BraSystemExpression) -> Self {
        value.inner
    }
}

impl Deref for BraSystemExpression {
    type Target = NamedExpression;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl DerefMut for BraSystemExpression {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.inner
    }
}

// TODO: replace individual From<X> for TensorExpression impls with blanket one
// pub trait HasIndices {
//     fn indices(&self) -> &[TensorIndex];
// }

// impl<T: HasIndices + Into<NamedExpression>> From<T> for TensorExpression {
//     fn from(value: T) -> Self {
//         let indices = value.indices().iter().cloned().collect();
//         let inner = value.into();
//         TensorExpression::from_raw(indices, inner)
//     }
// }

impl From<BraSystemExpression> for TensorExpression {
    fn from(value: BraSystemExpression) -> Self {
        let BraSystemExpression {
            inner,
            radices,
            num_states,
        } = value;
        // TODO: add a proper implementation of into_iter for QuditRadices
        let indices = [num_states]
            .into_iter()
            .map(|r| (IndexDirection::Batch, r))
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

impl TryFrom<TensorExpression> for BraSystemExpression {
    // TODO: Come up with proper error handling
    type Error = String;

    fn try_from(value: TensorExpression) -> Result<Self, Self::Error> {
        let mut num_states = None;
        let mut radices = vec![];
        for idx in value.indices() {
            match idx.direction() {
                IndexDirection::Batch => match num_states {
                    Some(n) => num_states = Some(n * idx.index_size()),
                    None => num_states = Some(idx.index_size()),
                },
                IndexDirection::Input => {
                    radices.push(idx.index_size());
                }
                _ => {
                    if idx.index_size() > 1 {
                        return Err(String::from(
                            "Cannot convert a tensor with non-input or batch indices to a bra system.",
                        ));
                    }
                }
            }
        }

        Ok(BraSystemExpression {
            inner: value.into(),
            radices: radices.into(),
            num_states: num_states.unwrap_or(1),
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
    #[pyo3(name = "BraSystemExpression")]
    pub struct PyBraSystemExpression {
        expr: BraSystemExpression,
    }

    #[pymethods]
    impl PyBraSystemExpression {
        #[new]
        fn new(expr: String) -> Self {
            Self {
                expr: BraSystemExpression::new(expr),
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

        fn num_states(&self) -> usize {
            self.expr.num_states
        }

        fn dimension(&self) -> usize {
            self.expr.radices.dimension()
        }

        fn __repr__(&self) -> String {
            format!(
                "BraSystemExpression(name='{}', radices={:?}, num_states={}, params={})",
                self.expr.name(),
                self.expr.radices.to_vec(),
                self.expr.num_states,
                self.expr.num_params()
            )
        }
    }

    impl From<BraSystemExpression> for PyBraSystemExpression {
        fn from(value: BraSystemExpression) -> Self {
            PyBraSystemExpression { expr: value }
        }
    }

    impl From<PyBraSystemExpression> for BraSystemExpression {
        fn from(value: PyBraSystemExpression) -> Self {
            value.expr
        }
    }

    impl<'py> IntoPyObject<'py> for BraSystemExpression {
        type Target = <PyBraSystemExpression as IntoPyObject<'py>>::Target;
        type Output = Bound<'py, Self::Target>;
        type Error = PyErr;

        fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
            let py_expr = PyBraSystemExpression::from(self);
            Bound::new(py, py_expr)
        }
    }

    impl<'py> FromPyObject<'py> for BraSystemExpression {
        fn extract_bound(ob: &Bound<'py, PyAny>) -> PyResult<Self> {
            let py_expr: PyRef<PyBraSystemExpression> = ob.extract()?;
            Ok(py_expr.expr.clone())
        }
    }

    /// Registers the BraSystemExpression class with the Python module.
    fn register(parent_module: &Bound<'_, PyModule>) -> PyResult<()> {
        parent_module.add_class::<PyBraSystemExpression>()?;
        Ok(())
    }
    inventory::submit!(PyExpressionRegistrar { func: register });
}
