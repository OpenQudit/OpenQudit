use std::ops::{Deref, DerefMut};

use crate::{
    GenerationShape, TensorExpression,
    expressions::JittableExpression,
    index::{IndexDirection, TensorIndex},
};

use super::NamedExpression;
use qudit_core::QuditSystem;
use qudit_core::Radices;

#[derive(PartialEq, Eq, Hash, Debug, Clone)]
pub struct KetSystemExpression {
    inner: NamedExpression,
    radices: Radices,
    num_states: usize,
}

impl KetSystemExpression {
    pub fn new<T: AsRef<str>>(input: T) -> Self {
        TensorExpression::new(input).try_into().unwrap()
    }

    pub fn num_qudits(&self) -> usize {
        self.radices.num_qudits()
    }
}

impl JittableExpression for KetSystemExpression {
    fn generation_shape(&self) -> GenerationShape {
        GenerationShape::Tensor3D(self.num_states, self.radices.dimension(), 1)
    }
}

impl AsRef<NamedExpression> for KetSystemExpression {
    fn as_ref(&self) -> &NamedExpression {
        &self.inner
    }
}

impl From<KetSystemExpression> for NamedExpression {
    fn from(value: KetSystemExpression) -> Self {
        value.inner
    }
}

impl Deref for KetSystemExpression {
    type Target = NamedExpression;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl DerefMut for KetSystemExpression {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.inner
    }
}

impl From<KetSystemExpression> for TensorExpression {
    fn from(value: KetSystemExpression) -> Self {
        let KetSystemExpression {
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
                    .iter()
                    .map(|r| (IndexDirection::Output, usize::from(*r))),
            )
            .enumerate()
            .map(|(i, (d, r))| TensorIndex::new(d, i, r))
            .collect();
        TensorExpression::from_raw(indices, inner)
    }
}

impl TryFrom<TensorExpression> for KetSystemExpression {
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
                IndexDirection::Output => {
                    radices.push(idx.index_size());
                }
                _ => {
                    if idx.index_size() > 1 {
                        return Err(String::from(
                            "Cannot convert a tensor with non-output or batch indices to a ket system.",
                        ));
                    }
                }
            }
        }

        Ok(KetSystemExpression {
            inner: value.into(),
            radices: radices.into(),
            num_states: num_states.unwrap_or(1),
        })
    }
}

#[cfg(feature = "python")]
mod python {
    use std::hash::DefaultHasher;
    use std::hash::Hash;
    use std::hash::Hasher;

    use super::*;
    use crate::ComplexExpression;
    use crate::python::PyExpressionRegistrar;
    use pyo3::prelude::*;
    use pyo3_stub_gen::derive::*;
    use pyo3_stub_gen::impl_stub_type;
    use qudit_core::Radix;

    /// A symbolic expression representing a batched system of ket (column) vectors.
    #[gen_stub_pyclass]
    #[pyclass(name = "KetSystemExpression", module = "openqudit.expressions")]
    pub struct PyKetSystemExpression {
        expr: KetSystemExpression,
    }

    #[gen_stub_pymethods]
    #[pymethods]
    impl PyKetSystemExpression {
        /// Parses a ket system expression from its string representation.
        ///
        /// # Arguments
        ///
        /// * `expr` - The textual definition of the ket system expression.
        #[new]
        fn new(expr: String) -> Self {
            Self {
                expr: KetSystemExpression::new(expr),
            }
        }

        /// Returns the number of free (unbound) parameters in this expression.
        fn num_params(&self) -> usize {
            self.expr.num_params()
        }

        /// Returns the name assigned to this expression.
        fn name(&self) -> String {
            self.expr.name().to_string()
        }

        /// Returns the radix of each qudit that this ket system acts on.
        fn radices(&self) -> Vec<Radix> {
            self.expr.radices.to_vec()
        }

        /// Returns the number of qudits in the system.
        fn num_qudits(&self) -> usize {
            self.expr.num_qudits()
        }

        /// Returns the number of individual ket states batched in this system.
        fn num_states(&self) -> usize {
            self.expr.num_states
        }

        /// Returns the total Hilbert space dimension of the underlying qudit system.
        fn dimension(&self) -> usize {
            self.expr.radices.dimension()
        }

        /// Returns the names of the free parameters in this tree, in the order
        /// they are first encountered.
        fn variables(&self) -> Vec<String> {
            self.expr.variables().to_vec()
        }

        fn elements(&self) -> Vec<ComplexExpression> {
            self.expr.elements().to_vec()
        }

        fn conjugate(&self) -> Self {
            let mut new = self.expr.clone();
            new.conjugate();
            Self { expr: new }
        }

        /// Returns a hash of this expression's body and qudit structure.
        ///
        /// Mirrors Rust equality, which compares element bodies and qudit
        /// structure and ignores the name the expression was given.
        fn __hash__(&self) -> u64 {
            let mut hasher = DefaultHasher::new();
            self.expr.hash(&mut hasher);
            hasher.finish()
        }

        fn __eq__(&self, other: &PyKetSystemExpression) -> bool {
            self.expr == other.expr
        }

        fn __repr__(&self) -> String {
            format!(
                "KetSystemExpression(name='{}', radices={:?}, num_states={}, params={})",
                self.expr.name(),
                self.expr.radices.to_vec(),
                self.expr.num_states,
                self.expr.num_params()
            )
        }
    }

    impl From<KetSystemExpression> for PyKetSystemExpression {
        fn from(value: KetSystemExpression) -> Self {
            PyKetSystemExpression { expr: value }
        }
    }

    impl From<PyKetSystemExpression> for KetSystemExpression {
        fn from(value: PyKetSystemExpression) -> Self {
            value.expr
        }
    }

    impl<'py> IntoPyObject<'py> for KetSystemExpression {
        type Target = <PyKetSystemExpression as IntoPyObject<'py>>::Target;
        type Output = Bound<'py, Self::Target>;
        type Error = PyErr;

        fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
            let py_expr = PyKetSystemExpression::from(self);
            Bound::new(py, py_expr)
        }
    }

    impl<'a, 'py> FromPyObject<'a, 'py> for KetSystemExpression {
        type Error = PyErr;

        fn extract(ob: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
            let py_expr: PyRef<PyKetSystemExpression> = ob.extract()?;
            Ok(py_expr.expr.clone())
        }
    }

    impl_stub_type!(KetSystemExpression = PyKetSystemExpression);

    /// Registers the KetSystemExpression class with the Python module.
    fn register(parent_module: &Bound<'_, PyModule>) -> PyResult<()> {
        parent_module.add_class::<PyKetSystemExpression>()?;
        Ok(())
    }
    inventory::submit!(PyExpressionRegistrar { func: register });
}
