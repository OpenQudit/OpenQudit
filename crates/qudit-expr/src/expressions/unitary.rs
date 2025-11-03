use std::collections::{BTreeSet, HashMap};
use std::ops::{Deref, DerefMut};

use faer::Mat;
use qudit_core::ComplexScalar;
use qudit_core::QuditSystem;
use qudit_core::Radices;
use qudit_core::UnitaryMatrix;

use crate::{ComplexExpression, UnitarySystemExpression};
use crate::{
    GenerationShape, TensorExpression,
    expressions::JittableExpression,
    index::{IndexDirection, TensorIndex},
};

use super::NamedExpression;

#[derive(PartialEq, Eq, Debug, Clone)]
pub struct UnitaryExpression {
    inner: NamedExpression,
    radices: Radices,
}

// pub trait ExpressionContainer {
//     pub fn elements(&self) -> &[ComplexExpression];
//     pub fn elements_mut(&mut self) -> &mut [ComplexExpression];

//     pub fn conjugate(&mut self) {
//         todo!()
//     }
// }

// pub trait MatrixExpression: ExpressionContainer {
//     pub fn nrows(&self) -> usize;
//     pub fn ncols(&self) -> usize;

//     pub fn dagger(&mut self) {
//         todo!()
//     }

//     pub fn transpose(&mut self) {
//         todo!()
//     }
// }

impl UnitaryExpression {
    pub fn new<T: AsRef<str>>(input: T) -> Self {
        TensorExpression::new(input).try_into().unwrap()
    }

    // TODO: Change ToRadices by implementing appropriate Froms
    pub fn identity<S: Into<String>, T: Into<Radices>>(name: S, radices: T) -> Self {
        let radices = radices.into();
        let dim = radices.dimension();
        let mut body = Vec::with_capacity(dim * dim);
        for i in 0..dim {
            for j in 0..dim {
                if i == j {
                    body.push(ComplexExpression::one());
                } else {
                    body.push(ComplexExpression::zero());
                }
            }
        }
        let inner = NamedExpression::new(name, vec![], body);

        UnitaryExpression { inner, radices }
    }

    pub fn set_radices(&mut self, new_radices: Radices) {
        assert_eq!(self.radices.dimension(), new_radices.dimension());
        self.radices = new_radices;
    }

    pub fn transpose(&mut self) {
        let dim = self.dimension();
        for i in 1..dim {
            for j in i..dim {
                let (head, tail) = self.split_at_mut(j * dim + i);
                std::mem::swap(&mut head[i * dim + j], &mut tail[0]);
            }
        }
    }

    pub fn dagger(&mut self) {
        self.conjugate();
        self.transpose();
    }

    pub fn classically_multiplex(
        expressions: &[&UnitaryExpression],
        new_dim_radices: &[usize],
    ) -> UnitarySystemExpression {
        let new_dim = new_dim_radices.iter().product();
        assert!(
            expressions.len() == new_dim,
            "Cannot multiplex a number of expressions not equal to the length of new dimension."
        );
        assert!(expressions.len() > 0);
        for expression in expressions {
            assert_eq!(
                expression.radices(),
                expressions[0].radices(),
                "All expressions must have equal radices."
            );
        }

        // construct larger tensor
        let mut new_expressions =
            Vec::with_capacity(expressions[0].dimension() * expressions[0].dimension() * new_dim);
        let mut var_map_number = 0;
        let mut new_variables = Vec::new();
        for i in 0..new_dim {
            // TODO: double clone
            let mut renamed_expression = expressions[i].clone();
            renamed_expression.alpha_rename(Some(var_map_number));
            new_expressions.extend(renamed_expression.elements().iter().cloned());
            for _ in 0..expressions[i].variables().len() {
                new_variables.push(format!("alpha_{}", var_map_number));
                var_map_number += 1;
            }
        }

        let new_indices = new_dim_radices
            .iter()
            .map(|r| (IndexDirection::Batch, *r as usize))
            .chain(
                expressions[0]
                    .radices()
                    .iter()
                    .map(|r| (IndexDirection::Output, usize::from(*r))),
            )
            .chain(
                expressions[0]
                    .radices()
                    .iter()
                    .map(|r| (IndexDirection::Input, usize::from(*r))),
            )
            .enumerate()
            .map(|(id, (dir, size))| TensorIndex::new(dir, id, size))
            .collect();

        let name = {
            // if all expressions have same name: multiplex_name
            // else: multiplex_name_name_...
            if expressions
                .iter()
                .all(|e| e.name() == expressions[0].name())
            {
                format!("Multiplexed_{}", expressions[0].name())
            } else {
                "Multiplexed_".to_string()
                    + &expressions
                        .iter()
                        .map(|e| e.name())
                        .collect::<Vec<_>>()
                        .join("_")
            }
        };

        let inner = NamedExpression::new(name, new_variables, new_expressions);
        TensorExpression::from_raw(new_indices, inner)
            .try_into()
            .unwrap()
    }

    // TODO: better API for user-facing thoughts
    pub fn classically_control(
        &self,
        positions: &[usize],
        new_dim_radices: &[usize],
    ) -> UnitarySystemExpression {
        let new_dim = new_dim_radices.iter().product();
        assert!(
            positions.len() <= new_dim,
            "Cannot place unitary in more locations than length of new dimension."
        );

        // Ensure positions are unique
        let mut sorted_positions = positions.to_vec();
        sorted_positions.sort_unstable();
        assert!(
            sorted_positions.iter().collect::<BTreeSet<_>>().len() == sorted_positions.len(),
            "Positions must be unique"
        );

        // Construct identity expression
        let mut identity = Vec::with_capacity(self.dimension() * self.dimension());
        for i in 0..self.dimension() {
            for j in 0..self.dimension() {
                if i == j {
                    identity.push(ComplexExpression::one());
                } else {
                    identity.push(ComplexExpression::zero());
                }
            }
        }

        // construct larger tensor
        let mut expressions = Vec::with_capacity(self.dimension() * self.dimension() * new_dim);
        for i in 0..new_dim {
            if positions.contains(&i) {
                expressions.extend(self.elements().iter().cloned());
            } else {
                expressions.extend(identity.iter().cloned());
            }
        }

        let new_indices = new_dim_radices
            .iter()
            .map(|r| (IndexDirection::Batch, *r as usize))
            .chain(
                self.radices
                    .iter()
                    .map(|r| (IndexDirection::Output, usize::from(*r))),
            )
            .chain(
                self.radices
                    .iter()
                    .map(|r| (IndexDirection::Input, usize::from(*r))),
            )
            .enumerate()
            .map(|(id, (dir, size))| TensorIndex::new(dir, id, size))
            .collect();

        let inner = NamedExpression::new(
            format!("Stacked_{}", self.name()),
            self.variables().to_owned(),
            expressions,
        );
        TensorExpression::from_raw(new_indices, inner)
            .try_into()
            .unwrap()
    }

    pub fn embed(
        &mut self,
        sub_matrix: UnitaryExpression,
        top_left_row_idx: usize,
        top_left_col_idx: usize,
    ) {
        let nrows = self.dimension();
        let ncols = self.dimension();
        let sub_nrows = sub_matrix.dimension();
        let sub_ncols = sub_matrix.dimension();

        if top_left_row_idx + sub_nrows > nrows || top_left_col_idx + sub_ncols > ncols {
            panic!("Embedding matrix is too large");
        }

        for i in 0..sub_nrows {
            for j in 0..sub_ncols {
                // TODO: remove clone by building into_iter for sub_matrix
                let sub_expr = sub_matrix[i * sub_ncols + j].clone();
                self[(top_left_row_idx + i) * ncols + (top_left_col_idx + j)] = sub_expr;
            }
        }

        // Update variables: collect all unique variables from self and sub_matrix
        let mut new_variables: Vec<String> = self.variables().to_vec();
        for var in sub_matrix.variables().iter() {
            if !new_variables.contains(var) {
                new_variables.push(var.clone());
            }
        }
        self.set_variables(new_variables);
    }

    pub fn otimes<U: AsRef<UnitaryExpression>>(&self, other: U) -> Self {
        let other = other.as_ref();
        let mut variables = Vec::new();
        let mut var_map_self = HashMap::new();
        let mut i = 0;
        for var in self.variables().iter() {
            variables.push(format!("x{}", i));
            var_map_self.insert(var.clone(), format!("x{}", i));
            i += 1;
        }
        let mut var_map_other = HashMap::new();
        for var in other.variables().iter() {
            variables.push(format!("x{}", i));
            var_map_other.insert(var.clone(), format!("x{}", i));
            i += 1;
        }

        let lhs_nrows = self.dimension();
        let lhs_ncols = self.dimension();
        let rhs_nrows = other.dimension();
        let rhs_ncols = other.dimension();

        let mut out_body = Vec::with_capacity(lhs_nrows * lhs_ncols * rhs_ncols * rhs_nrows);

        for i in 0..lhs_nrows {
            for j in 0..rhs_nrows {
                for k in 0..lhs_ncols {
                    for l in 0..rhs_ncols {
                        out_body.push(
                            self[i * lhs_ncols + k].map_var_names(&var_map_self)
                                * other[j * rhs_ncols + l].map_var_names(&var_map_other),
                        );
                    }
                }
            }
        }

        let inner = NamedExpression::new(
            format!("{} ⊗ {}", self.name(), other.name()),
            variables,
            out_body,
        );

        UnitaryExpression {
            inner,
            radices: self.radices.concat(&other.radices),
        }
    }

    pub fn dot<U: AsRef<UnitaryExpression>>(&self, other: U) -> Self {
        let other = other.as_ref();
        let lhs_nrows = self.dimension();
        let lhs_ncols = self.dimension();
        let rhs_nrows = other.dimension();
        let rhs_ncols = other.dimension();

        if lhs_ncols != rhs_nrows {
            panic!(
                "Matrix dimensions do not match for dot product: {} != {}",
                lhs_ncols, rhs_nrows
            );
        }

        let mut variables = Vec::new();
        let mut var_map_self = HashMap::new();
        let mut i = 0;
        for var in self.variables().iter() {
            variables.push(format!("x{}", i));
            var_map_self.insert(var.clone(), format!("x{}", i));
            i += 1;
        }
        let mut var_map_other = HashMap::new();
        for var in other.variables().iter() {
            variables.push(format!("x{}", i));
            var_map_other.insert(var.clone(), format!("x{}", i));
            i += 1;
        }

        let mut out_body = Vec::with_capacity(lhs_nrows * rhs_ncols);

        for i in 0..lhs_nrows {
            for j in 0..rhs_ncols {
                let mut sum = self[i * lhs_ncols].map_var_names(&var_map_self)
                    * other[j].map_var_names(&var_map_other);
                for k in 1..lhs_ncols {
                    sum += self[i * lhs_ncols + k].map_var_names(&var_map_self)
                        * other[k * rhs_ncols + j].map_var_names(&var_map_other);
                }
                out_body.push(sum);
            }
        }

        let inner = NamedExpression::new(
            format!("{} ⋅ {}", self.name(), other.name()),
            variables,
            out_body,
        );

        UnitaryExpression {
            inner,
            radices: self.radices.clone(),
        }
    }

    pub fn get_arg_map<C: ComplexScalar>(&self, args: &[C::R]) -> HashMap<&str, C::R> {
        self.variables()
            .iter()
            .zip(args.iter())
            .map(|(a, b)| (a.as_str(), *b))
            .collect()
    }

    pub fn eval<C: ComplexScalar>(&self, args: &[C::R]) -> UnitaryMatrix<C> {
        let arg_map = self.get_arg_map::<C>(args);
        let dim = self.radices.dimension();
        let mut mat = Mat::zeros(dim, dim);
        for i in 0..dim {
            for j in 0..dim {
                *mat.get_mut(i, j) = self[i * self.dimension() + j].eval(&arg_map);
            }
        }
        UnitaryMatrix::new(self.radices.clone(), mat)
    }
}

impl JittableExpression for UnitaryExpression {
    fn generation_shape(&self) -> GenerationShape {
        GenerationShape::Matrix(self.radices.dimension(), self.radices.dimension())
    }
}

impl AsRef<UnitaryExpression> for UnitaryExpression {
    fn as_ref(&self) -> &UnitaryExpression {
        &self
    }
}

impl AsRef<NamedExpression> for UnitaryExpression {
    fn as_ref(&self) -> &NamedExpression {
        &self.inner
    }
}

impl From<UnitaryExpression> for NamedExpression {
    fn from(value: UnitaryExpression) -> Self {
        value.inner
    }
}

impl Deref for UnitaryExpression {
    type Target = NamedExpression;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl DerefMut for UnitaryExpression {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.inner
    }
}

impl From<UnitaryExpression> for TensorExpression {
    fn from(value: UnitaryExpression) -> Self {
        let UnitaryExpression { inner, radices } = value;
        let indices = radices
            .iter()
            .map(|r| (IndexDirection::Output, usize::from(*r)))
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

impl TryFrom<TensorExpression> for UnitaryExpression {
    // TODO: Come up with proper error handling
    type Error = String;

    fn try_from(value: TensorExpression) -> Result<Self, Self::Error> {
        let mut input_radices = vec![];
        let mut output_radices = vec![];
        for idx in value.indices() {
            match idx.direction() {
                IndexDirection::Input => {
                    input_radices.push(idx.index_size());
                }
                IndexDirection::Output => {
                    output_radices.push(idx.index_size());
                }
                _ => {
                    return Err(String::from(
                        "Cannot convert a tensor with non-input, non-output indices to an isometry.",
                    ));
                }
            }
        }

        if input_radices != output_radices {
            return Err(String::from(
                "Non-square matrix tensor cannot be converted to a unitary.",
            ));
        }

        Ok(UnitaryExpression {
            inner: value.into(),
            radices: input_radices.into(),
        })
    }
}

impl<C: ComplexScalar> From<UnitaryMatrix<C>> for UnitaryExpression {
    fn from(value: UnitaryMatrix<C>) -> Self {
        let mut body = vec![Vec::with_capacity(value.ncols()); value.nrows()];
        for col in value.col_iter() {
            for (row_id, elem) in col.iter().enumerate() {
                body[row_id].push(ComplexExpression::from(*elem));
            }
        }
        let mut flat_body = Vec::with_capacity(value.ncols() * value.nrows());
        for row in body.into_iter() {
            for elem in row.into_iter() {
                flat_body.push(elem);
            }
        }
        let inner = NamedExpression::new("Constant", vec![], flat_body);

        UnitaryExpression {
            inner,
            radices: value.radices(),
        }
    }
}

impl QuditSystem for UnitaryExpression {
    fn radices(&self) -> Radices {
        self.radices.clone()
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
    use qudit_core::Radix;
    use qudit_core::c64;

    #[pyclass]
    #[pyo3(name = "UnitaryExpression")]
    pub struct PyUnitaryExpression {
        expr: UnitaryExpression,
    }

    #[pymethods]
    impl PyUnitaryExpression {
        #[new]
        fn new(expr: String) -> Self {
            Self {
                expr: UnitaryExpression::new(expr),
            }
        }

        #[staticmethod]
        fn identity(name: String, radices: Vec<usize>) -> Self {
            Self {
                expr: UnitaryExpression::identity(name, radices),
            }
        }

        #[pyo3(signature = (*args))]
        fn __call__<'py>(&self, args: &Bound<'py, PyTuple>) -> PyResult<Bound<'py, PyArray2<c64>>> {
            let py = args.py();
            let args: Vec<f64> = args.extract()?;
            let unitary = self.expr.eval(&args);
            let py_array: Bound<'py, PyArray2<c64>> =
                PyArray2::zeros(py, (unitary.dimension(), unitary.dimension()), false);

            {
                let mut readwrite = py_array.readwrite();
                let mut py_array_view: ArrayViewMut2<c64> = readwrite.as_array_mut();

                for (j, col) in unitary.col_iter().enumerate() {
                    for (i, val) in col.iter().enumerate() {
                        py_array_view[[i, j]] = *val;
                    }
                }
            }

            Ok(py_array)
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

        fn transpose(&mut self) {
            self.expr.transpose();
        }

        fn dagger(&mut self) {
            self.expr.dagger();
        }

        fn otimes(&self, other: &PyUnitaryExpression) -> Self {
            Self {
                expr: self.expr.otimes(&other.expr),
            }
        }

        fn dot(&self, other: &PyUnitaryExpression) -> Self {
            Self {
                expr: self.expr.dot(&other.expr),
            }
        }

        fn embed(
            &mut self,
            sub_matrix: &PyUnitaryExpression,
            top_left_row_idx: usize,
            top_left_col_idx: usize,
        ) {
            self.expr
                .embed(sub_matrix.expr.clone(), top_left_row_idx, top_left_col_idx);
        }

        // fn classically_control(&self, positions: Vec<usize>, new_dim_radices: Vec<usize>) -> crate::python::tensor::PyTensorExpression {
        //     let result = self.expr.classically_control(&positions, &new_dim_radices);
        //     result.into()
        // }

        fn __repr__(&self) -> String {
            format!(
                "UnitaryExpression(name='{}', radices={:?}, params={})",
                self.expr.name(),
                self.expr.radices().to_vec(),
                self.expr.num_params()
            )
        }
    }

    impl From<UnitaryExpression> for PyUnitaryExpression {
        fn from(value: UnitaryExpression) -> Self {
            PyUnitaryExpression { expr: value }
        }
    }

    impl From<PyUnitaryExpression> for UnitaryExpression {
        fn from(value: PyUnitaryExpression) -> Self {
            value.expr
        }
    }

    impl<'py> IntoPyObject<'py> for UnitaryExpression {
        type Target = <PyUnitaryExpression as IntoPyObject<'py>>::Target;
        type Output = Bound<'py, Self::Target>;
        type Error = PyErr;

        fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
            let py_expr = PyUnitaryExpression::from(self);
            Bound::new(py, py_expr)
        }
    }

    impl<'a, 'py> FromPyObject<'a, 'py> for UnitaryExpression {
        type Error = PyErr;

        fn extract(ob: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
            let py_expr: PyRef<PyUnitaryExpression> = ob.extract()?;
            Ok(py_expr.expr.clone())
        }
    }

    /// Registers the UnitaryExpression class with the Python module.
    fn register(parent_module: &Bound<'_, PyModule>) -> PyResult<()> {
        parent_module.add_class::<PyUnitaryExpression>()?;
        Ok(())
    }
    inventory::submit!(PyExpressionRegistrar { func: register });
}
