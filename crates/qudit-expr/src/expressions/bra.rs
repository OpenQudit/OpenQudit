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

#[derive(PartialEq, Eq, Hash, Debug, Clone)]
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
            .iter()
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
    use std::hash::DefaultHasher;
    use std::hash::Hash;
    use std::hash::Hasher;

    use super::*;
    use crate::{ComplexExpression, python::PyExpressionRegistrar};
    use pyo3::prelude::*;
    use pyo3_stub_gen::derive::*;
    use pyo3_stub_gen::impl_stub_type;
    use qudit_core::Radix;

    /// A symbolic bra (row) vector expression over a qudit system.
    #[gen_stub_pyclass]
    #[pyclass(name = "BraExpression", module = "openqudit.expressions")]
    pub struct PyBraExpression {
        expr: BraExpression,
    }

    #[gen_stub_pymethods]
    #[pymethods]
    impl PyBraExpression {
        /// Parses a bra expression from its string representation.
        ///
        /// # Arguments
        ///
        /// * `expr` - The textual definition of the bra expression.
        #[new]
        fn new(expr: String) -> Self {
            Self {
                expr: BraExpression::new(expr),
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

        /// Returns the radix of each qudit that this bra acts on.
        fn radices(&self) -> Vec<Radix> {
            self.expr.radices().to_vec()
        }

        /// Returns the number of qudits this unitary acts on.
        fn num_qudits(&self) -> usize {
            self.expr.num_qudits()
        }

        /// Returns true if this unitary acts on a qubit-only system.
        fn is_qubit_only(&self) -> bool {
            self.expr.is_qubit_only()
        }

        /// Returns true if this unitary acts on a qutrit-only system
        fn is_qutrit_only(&self) -> bool {
            self.expr.is_qutrit_only()
        }

        /// Returns true if this unitary acts on a `radix`-only system
        fn is_qudit_only(&self, radix: Radix) -> bool {
            self.expr.is_qudit_only(radix)
        }

        /// Returns true if this unitary acts on a homogenous system
        fn is_homogenous(&self) -> bool {
            self.expr.is_homogenous()
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

        /// Returns the total Hilbert space dimension of the underlying qudit system.
        fn dimension(&self) -> usize {
            self.expr.dimension()
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

        fn __eq__(&self, other: &PyBraExpression) -> bool {
            self.expr == other.expr
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

    impl_stub_type!(BraExpression = PyBraExpression);

    /// Registers the BraExpression class with the Python module.
    fn register(parent_module: &Bound<'_, PyModule>) -> PyResult<()> {
        parent_module.add_class::<PyBraExpression>()?;
        Ok(())
    }
    inventory::submit!(PyExpressionRegistrar { func: register });
}
