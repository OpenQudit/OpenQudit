use std::ops::{Deref, DerefMut};

use crate::{
    GenerationShape, TensorExpression,
    expressions::JittableExpression,
    index::{IndexDirection, TensorIndex},
};

use super::ComplexExpression;
use super::NamedExpression;
use qudit_core::QuditSystem;
use qudit_core::Radices;

#[derive(PartialEq, Eq, Hash, Debug, Clone)]
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
            .iter()
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
            .any(|idx| idx.direction() != IndexDirection::Output && idx.index_size() != 1)
        {
            return Err(String::from(
                "Cannot convert a tensor with non-output indices to a ket.",
            ));
        }
        let radices = Radices::from_iter(
            value
                .indices()
                .iter()
                .filter(|idx| idx.index_size() > 1)
                .map(|idx| idx.index_size()),
        );
        Ok(KetExpression {
            inner: value.into(),
            radices,
        })
    }
}

#[cfg(feature = "python")]
mod python {
    use std::hash::DefaultHasher;
    use std::hash::Hash;
    use std::hash::Hasher;

    use super::*;
    use crate::python::PyExpressionRegistrar;
    use pyo3::prelude::*;
    use pyo3_stub_gen::derive::*;
    use pyo3_stub_gen::impl_stub_type;
    use qudit_core::Radix;

    /// A symbolic ket (column) vector expression over a qudit system.
    #[gen_stub_pyclass]
    #[pyclass(name = "KetExpression", module = "openqudit.expressions")]
    pub struct PyKetExpression {
        expr: KetExpression,
    }

    #[gen_stub_pymethods]
    #[pymethods]
    impl PyKetExpression {
        /// Parses a ket expression from its string representation.
        ///
        /// # Arguments
        ///
        /// * `expr` - The textual definition of the ket expression.
        #[new]
        fn new(expr: String) -> Self {
            Self {
                expr: KetExpression::new(expr),
            }
        }

        /// Constructs a zero ket expression over the given radices.
        ///
        /// # Arguments
        ///
        /// * `radices` - The radix of each qudit in the system.
        #[staticmethod]
        fn zero(radices: Vec<usize>) -> Self {
            Self {
                expr: KetExpression::zero(radices),
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

        /// Returns the radix of each qudit that this ket acts on.
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

        fn __eq__(&self, other: &PyKetExpression) -> bool {
            self.expr == other.expr
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

    impl<'a, 'py> FromPyObject<'a, 'py> for KetExpression {
        type Error = PyErr;

        fn extract(ob: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
            let py_expr: PyRef<PyKetExpression> = ob.extract()?;
            Ok(py_expr.expr.clone())
        }
    }

    impl_stub_type!(KetExpression = PyKetExpression);

    /// Registers the KetExpression class with the Python module.
    fn register(parent_module: &Bound<'_, PyModule>) -> PyResult<()> {
        parent_module.add_class::<PyKetExpression>()?;
        Ok(())
    }
    inventory::submit!(PyExpressionRegistrar { func: register });
}
