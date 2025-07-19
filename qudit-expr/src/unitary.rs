use std::collections::HashMap;
use std::ops::Range;

use itertools::Itertools;
use qudit_core::matrix::Mat;
use qudit_core::matrix::MatMut;
use qudit_core::matrix::MatVecMut;
use qudit_core::unitary::DifferentiableUnitaryFn;
use qudit_core::unitary::UnitaryFn;
use qudit_core::unitary::UnitaryMatrix;
use qudit_core::state::StateVector;
use qudit_core::ComplexScalar;
use qudit_core::HasParams;
use qudit_core::HasPeriods;
use qudit_core::QuditPermutation;
use qudit_core::QuditRadices;
use qudit_core::QuditSystem;
use qudit_core::RealScalar;
use qudit_core::ToRadices;
use faer::reborrow::ReborrowMut;
use crate::tensor::TensorExpression;

use crate::analysis::simplify_expressions;
use crate::analysis::simplify_matrix;
use crate::complex::ComplexExpression;
use crate::expression::Expression;
use crate::qgl::parse_qobj;
// use crate::qgl::parse_unitary;
use crate::qgl::Expression as CiscExpression;
use crate::DifferentiationLevel;
use qudit_core::TensorShape;

pub struct DerivedExpression {
    pub name: String,
    pub shape: TensorShape,
    pub variables: Vec<String>,
    pub body: Vec<Expression>,
    pub dimensions: Vec<usize>,
    pub flattened_gradient: Vec<Expression>,
    pub flattened_hessian: Vec<Expression>,
}

impl DerivedExpression {
    pub fn new(expr: TensorExpression, diff_lvl: DifferentiationLevel) -> Self {
        let TensorExpression { name, shape, variables, body, dimensions } = expr;
        let exprs: Vec<Expression> = body.into_iter().flat_map(|c| vec![c.real, c.imag]).collect();

        let mut grad_exprs = vec![];
        if diff_lvl.gradient_capable() {
            for variable in &variables {
                for expr in &exprs {
                    let grad_expr = expr.differentiate(variable);
                    grad_exprs.push(grad_expr);
                }
            }
        }

        let mut hess_exprs = vec![];
        if diff_lvl.hessian_capable() {
            for variable1 in &variables {
                for variable2 in &variables {
                    // todo: symsq
                    for expr in &exprs {
                        let hess_expr = expr.differentiate(variable1).differentiate(variable2);
                        hess_exprs.push(hess_expr);
                    }
                }
            }
        }

        let expr_len = exprs.len();
        let grad_len = grad_exprs.len();

        let simplified_exprs = simplify_expressions(
            exprs.into_iter()
                .chain(grad_exprs)
                .chain(hess_exprs)
                .collect()
        );

        let exprs = simplified_exprs[0..expr_len].to_vec();
        let grad_exprs = simplified_exprs[expr_len..expr_len + grad_len].to_vec();
        let hess_exprs = simplified_exprs[expr_len + grad_len..].to_vec();

        DerivedExpression {
            name,
            shape,
            variables,
            body: exprs,
            dimensions,
            flattened_gradient: grad_exprs,
            flattened_hessian: hess_exprs,
        }
    }
}


#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct StateExpression {
    pub name: String,
    pub radices: QuditRadices,
    pub variables: Vec<String>,
    pub body: Vec<ComplexExpression>,
}

impl StateExpression {
    /// Creates a new `StateExpression` from a QGL string representation.
    ///
    /// This function parses the input string as a QGL object, then converts it
    /// into a `TensorExpression` and subsequently into a `StateExpression`.
    ///
    /// # Arguments
    ///
    /// * `input` - A type that can be converted to a string reference,
    ///             representing the QGL definition of the state expression.
    ///
    /// # Returns
    ///
    /// A new `StateExpression` instance.
    pub fn new<T: AsRef<str>>(input: T) -> Self {
        TensorExpression::new(input).to_state_expression()
}

    /// Evaluates the state expression with the given arguments and returns a `StateVector`.
    ///
    /// This function substitutes the provided real scalar arguments into the complex
    /// expressions that define the state vector's body, and then constructs a
    /// `StateVector` from the evaluated complex values.
    ///
    /// # Type Parameters
    ///
    /// * `C`: A type that implements `ComplexScalar`, representing the complex number
    ///        type for the state vector elements.
    ///
    /// # Arguments
    ///
    /// * `args` - A slice of real scalar values (`C::R`) to substitute for the
    ///            variables in the expression. The order of arguments must match
    ///            the order of `self.variables`.
    ///
    /// # Returns
    ///
    /// A `StateVector<C>` containing the evaluated complex elements of the state.
    pub fn eval<C: ComplexScalar>(&self, args: &[C::R]) -> StateVector<C> {
        let arg_map = self.variables.iter().zip(args.iter()).map(|(a, b)| (a.as_str(), *b)).collect();
        let evaluated_body: Vec<C> = self.body.iter().map(|expr| expr.eval(&arg_map)).collect();
        StateVector::new(self.radices.clone(), evaluated_body)
    }

    pub fn conjugate(&self) -> Self {
        let new_body = self.body.iter().map(|expr| expr.conjugate()).collect();
        StateExpression {
            name: format!("{}^_", self.name),
            radices: self.radices.clone(),
            variables: self.variables.clone(),
            body: new_body,
        }
    }

    pub fn name(&self) -> String {
        self.name.clone()
    }
}


#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct UnitaryExpression {
    pub name: String,
    pub radices: QuditRadices,
    pub variables: Vec<String>,
    pub body: Vec<Vec<ComplexExpression>>,
}

// #[test]
// fn test_state_expression() {
//     let expr = VectorExpression::new("
//         |0> + 1.0i * |1> + 2.0 * |2> + 3.0i * |3>
//     ");
// }

impl UnitaryExpression {
    pub fn new<T: AsRef<str>>(input: T) -> Self {
        TensorExpression::new(input).to_unitary_expression()
    }

    pub fn identity<S: AsRef<str>, T: ToRadices>(name: S, radices: T) -> Self {
        let radices = radices.to_radices();
        let dim = radices.dimension();
        let mut body = vec![Vec::new(); dim];
        for i in 0..dim {
            for j in 0..dim {
                if i == j {
                    body[i].push(ComplexExpression::one());
                } else {
                    body[i].push(ComplexExpression::zero());
                }
            }
        }
        UnitaryExpression {
            name: name.as_ref().to_string(),
            radices,
            variables: vec![],
            body,
        }
    }

    pub fn name(&self) -> String {
        self.name.clone()
    }

    pub fn get_arg_map<C: ComplexScalar>(&self, args: &[C::R]) -> HashMap<&str, C::R> {
        self.variables.iter().zip(args.iter()).map(|(a, b)| (a.as_str(), *b)).collect()
    }

    pub fn eval<C: ComplexScalar>(&self, args: &[C::R]) -> UnitaryMatrix<C> {
        let arg_map = self.get_arg_map::<C>(args);
        let dim = self.radices.dimension();
        let mut mat = Mat::zeros(dim, dim);
        for i in 0..dim {
            for j in 0..dim {
                *mat.get_mut(i, j) = self.body[i][j].eval(&arg_map);
            }
        }
        UnitaryMatrix::new(self.radices.clone(), mat)
    }



    pub fn embed(&mut self, sub_matrix: UnitaryExpression, top_left_row_idx: usize, top_left_col_idx: usize) {
        let nrows = self.body.len();
        let ncols = self.body[0].len();
        let sub_nrows = sub_matrix.body.len();
        let sub_ncols = sub_matrix.body[0].len();

        if top_left_row_idx + sub_nrows > nrows || top_left_col_idx + sub_ncols > ncols {
            panic!("Embedding matrix is too large");
        }

        for (i, row) in sub_matrix.body.into_iter().enumerate() {
            for (j, expr) in row.into_iter().enumerate() {
                self.body[top_left_row_idx + i][top_left_col_idx + j] = expr;
            }
        }

        // Update variables: collect all unique variables from self and sub_matrix
        let mut new_variables: Vec<String> = self.variables.iter().cloned().collect();
        for var in sub_matrix.variables.iter() {
            if !new_variables.contains(var) {
                new_variables.push(var.clone());
            }
        }
        self.variables = new_variables;
    }

    pub fn otimes<U: AsRef<UnitaryExpression>>(&self,  other: U) -> Self {
        let other = other.as_ref();
        let mut variables = Vec::new();
        let mut var_map_self = HashMap::new();
        let mut i = 0;
        for var in self.variables.iter() {
            variables.push(format!("x{}", i));
            var_map_self.insert(var.clone(), format!("x{}", i));
            i += 1;
        }
        let mut var_map_other = HashMap::new();
        for var in other.variables.iter() {
            variables.push(format!("x{}", i));
            var_map_other.insert(var.clone(), format!("x{}", i));
            i += 1;
        }

        let lhs_nrows = self.body.len();
        let lhs_ncols = self.body[0].len();
        let rhs_nrows = other.body.len();
        let rhs_ncols = other.body[0].len();

        let mut out_body = vec![vec![]; lhs_nrows * rhs_nrows];
        
        for i in 0..lhs_nrows {
            for j in 0..rhs_nrows {
                for k in 0..lhs_ncols {
                    for l in 0..rhs_ncols {
                        out_body[i * rhs_nrows + j].push(self.body[i][k].map_var_names(&var_map_self) * other.body[j][l].map_var_names(&var_map_other));
                    }
                }
            }
        }

        UnitaryExpression {
            name: format!("{} ⊗ {}", self.name, other.name),
            radices: self.radices.concat(&other.radices),
            variables,
            body: out_body,
        }
    }

    // TODO: remove when MatrixExpression a thing
    pub fn dot_no_rename<U: AsRef<UnitaryExpression>>(&self, other: U) -> Self {
        let other = other.as_ref();
        let lhs_nrows = self.body.len();
        let lhs_ncols = self.body[0].len();
        let rhs_nrows = other.body.len();
        let rhs_ncols = other.body[0].len();

        if lhs_ncols != rhs_nrows {
            panic!("Matrix dimensions do not match for dot product: {} != {}", lhs_ncols, rhs_nrows);
        }

        let mut out_body = vec![vec![]; lhs_nrows];

        for i in 0..lhs_nrows {
            for j in 0..rhs_ncols {
                let mut sum = &self.body[i][0] * &other.body[0][j];
                for k in 1..lhs_ncols {
                    sum = sum + &self.body[i][k] * &other.body[k][j];
                }
                out_body[i].push(sum);
            }
        }

        UnitaryExpression {
            name: format!("{} ⋅ {}", self.name, other.name),
            radices: self.radices.clone(),
            variables: self.variables.clone(),
            body: out_body,
        }
    }

    pub fn dot<U: AsRef<UnitaryExpression>>(&self, other: U) -> Self {
        let other = other.as_ref();
        let lhs_nrows = self.body.len();
        let lhs_ncols = self.body[0].len();
        let rhs_nrows = other.body.len();
        let rhs_ncols = other.body[0].len();

        if lhs_ncols != rhs_nrows {
            panic!("Matrix dimensions do not match for dot product: {} != {}", lhs_ncols, rhs_nrows);
        }

        let mut variables = Vec::new();
        let mut var_map_self = HashMap::new();
        let mut i = 0;
        for var in self.variables.iter() {
            variables.push(format!("x{}", i));
            var_map_self.insert(var.clone(), format!("x{}", i));
            i += 1;
        }
        let mut var_map_other = HashMap::new();
        for var in other.variables.iter() {
            variables.push(format!("x{}", i));
            var_map_other.insert(var.clone(), format!("x{}", i));
            i += 1;
        }

        let mut out_body = vec![vec![]; lhs_nrows];

        for i in 0..lhs_nrows {
            for j in 0..rhs_ncols {
                let mut sum = self.body[i][0].map_var_names(&var_map_self) * other.body[0][j].map_var_names(&var_map_other);
                for k in 1..lhs_ncols {
                    sum = sum + self.body[i][k].map_var_names(&var_map_self) * other.body[k][j].map_var_names(&var_map_other);
                }
                out_body[i].push(sum);
            }
        }

        UnitaryExpression {
            name: format!("{} ⋅ {}", self.name, other.name),
            radices: self.radices.clone(),
            variables,
            body: out_body,
        }
    }

    pub fn conjugate(&self) -> Self {
        let mut out_body = Vec::new();
        for row in &self.body {
            let mut new_row = Vec::new();
            for expr in row {
                new_row.push(expr.conjugate());
            }
            out_body.push(new_row);
        }
        UnitaryExpression {
            name: format!("{}^_", self.name),
            radices: self.radices.clone(),
            variables: self.variables.clone(),
            body: out_body,
        }
    }

    pub fn transpose(&self) -> Self {
        let mut out_body = Vec::new();
        for j in 0..self.body[0].len() {
            let mut new_row = Vec::new();
            for i in 0..self.body.len() {
                new_row.push(self.body[i][j].clone());
            }
            out_body.push(new_row);
        }
        UnitaryExpression {
            name: format!("{}^T", self.name),
            radices: self.radices.clone(),
            variables: self.variables.clone(),
            body: out_body,
        }
    }

    pub fn differentiate(&self) -> MatVecExpression {
        let mut grad_exprs = vec![vec![Vec::with_capacity(self.body.len()); self.body.len()]; self.variables.len()];
        for (m, var) in self.variables.iter().enumerate() {
            for (i, row) in self.body.iter().enumerate() {
                for (_j, expr) in row.iter().enumerate() {
                    grad_exprs[m][i].push(expr.differentiate(var));
                }
            }
        }

        MatVecExpression {
            name: format!("∇{}", self.name),
            variables: self.variables.clone(),
            body: grad_exprs,
        }
    }

    pub fn to_tensor_expression(&self) -> TensorExpression {
        let nrows = self.body.len();
        let ncols = self.body[0].len();
        let flattened_body = self.body.clone().into_iter().flat_map(|row| row.into_iter()).collect();
        TensorExpression {
            name: self.name.clone(),
            shape: TensorShape::Matrix(nrows, ncols),
            variables: self.variables.clone(),
            body: flattened_body,
            dimensions: self.radices.concat(&self.radices).iter().map(|&x| x as usize).collect(),
        }
    }

    pub fn simplify(&self) -> Self {
        let new_body = simplify_matrix(&self.body);
        UnitaryExpression {
            name: self.name.clone(),
            radices: self.radices.clone(),
            variables: self.variables.clone(),
            body: new_body,
        }
    }

    pub fn determinant(&self) -> ComplexExpression {
        if self.body.len() != self.body[0].len() {
            panic!("Determinant can only be calculated for square matrices");
        }

        if self.body.len() == 1 {
            return self.body[0][0].clone();
        }

        if self.body.len() == 2 {
            return self.body[0][0].clone() * self.body[1][1].clone() - self.body[0][1].clone() * self.body[1][0].clone();
        }
        
        todo!()
    }

    pub fn greatest_common_ancestors(&self) -> HashMap<String, Expression> {
        // Find greatest common ancestor
        // for every variable in expression:
        // - create vector of all ancestors containing the variable
        // - for every element in expression
        //     - if it contains the variable, walk up the tree storing all ancestors in the vector
        //     - if the vector is populated, remove anything that disagrees, lets figure it out
        
        let mut gca = HashMap::new();
        for var in &self.variables {
            let mut ancestors = None;
            for row in &self.body {
                for expr in row {
                    let new_ancestors = expr.get_ancestors(var);
                    if new_ancestors.is_empty() {
                        continue;
                    }
                    match ancestors {
                        None => ancestors = Some(new_ancestors),
                        Some(ref mut a) => a.retain(|ancestor| new_ancestors.contains(ancestor)),
                    }
                }
            }
            gca.insert(var.clone(), ancestors.unwrap().pop().unwrap());
        }
        
        gca
    }

    pub fn substitute<S: AsRef<Expression>, T: AsRef<Expression>>(&self, original: S, substitution: T) -> Self {
        let original = original.as_ref();
        let substitution = substitution.as_ref();

        let mut new_body = vec![vec![]; self.body.len()];
        for (i, row) in self.body.iter().enumerate() {
            for expr in row {
                new_body[i].push(expr.substitute(original, substitution));
            }
        }

        // TODO: Check if we removed a variable from the expression, then we need to remove it from the variables list

        UnitaryExpression {
            name: format!("{}", self.name),
            radices: self.radices.clone(),
            variables: self.variables.clone(),
            body: new_body,
        }
    }

    pub fn rename_variable<S: AsRef<str>, T: AsRef<str>>(&self, original: S, new: T) -> Self {
        let original = original.as_ref();
        let new = new.as_ref();

        let mut new_body = vec![vec![]; self.body.len()];
        for (i, row) in self.body.iter().enumerate() {
            for expr in row {
                new_body[i].push(expr.rename_variable(original, new));
            }
        }

        let mut new_variables = self.variables.clone();
        for var in &mut new_variables {
            if var == original {
                *var = new.to_string();
            }
        }

        UnitaryExpression {
            name: format!("{}", self.name),
            radices: self.radices.clone(),
            variables: new_variables,
            body: new_body,
        }
    }

    pub fn alpha_rename(&self, start_index: usize) -> Self {
        let mut new_vars = Vec::new();
        let mut var_map = HashMap::new();
        for (i, var) in self.variables.iter().enumerate() {
            let new_var = format!("x{}", i + start_index);
            new_vars.push(new_var.clone());
            var_map.insert(var.clone(), new_var);
        }

        let mut new_body = vec![vec![]; self.body.len()];
        for (i, row) in self.body.iter().enumerate() {
            for expr in row {
                new_body[i].push(expr.map_var_names(&var_map));
            }
        }

        UnitaryExpression {
            name: self.name.clone(),
            radices: self.radices.clone(),
            variables: new_vars,
            body: new_body,
        }
    }

    /// replace all variables with values, new variables given in variables input
    pub fn substitute_parameters<S: AsRef<str>, E: AsRef<Expression>>(&self, variables: &[S], values: &[E]) -> Self {
        let sub_map: HashMap<_, _> = self.variables.iter().zip(values.iter()).map(|(k, v)| (Expression::Variable(k.to_string()), v.as_ref())).collect();

        let mut new_body = vec![vec![]; self.body.len()];
        for (i, row) in self.body.iter().enumerate() {
            for expr in row {
                let mut new_expr = None; 
                for (var, value) in &sub_map {
                    match new_expr {
                        None => {
                            new_expr = Some(expr.substitute(var, value))
                        }
                        Some(ref mut e) => *e = e.substitute(var, value),
                    }
                }
                match new_expr {
                    None => new_body[i].push(expr.clone()),
                    Some(e) => new_body[i].push(e),
                }
            }
        }

        let new_variables = variables.iter().map(|s| s.as_ref().to_string()).collect();

        UnitaryExpression {
            name: format!("{}", self.name),
            radices: self.radices.clone(),
            variables: new_variables,
            body: new_body,
        }
    }

    pub fn permute(&self, perm: &QuditPermutation) -> Self {
        let mut new_body = vec![vec![]; self.body.len()];
        // TODO: is it index_perm or inverse_index_perm
        let index_perm = perm.index_perm();

        // swap rows
        for (i, row) in self.body.iter().enumerate() {
            for expr in row {
                new_body[index_perm[i]].push(expr.clone());
            }
        }

        // swap cols
        for (_i, row) in new_body.iter_mut().enumerate() {
            let new_row = (0..row.len()).map(|j| row[index_perm[j]].clone()).collect();
            *row = new_row;
        }

        UnitaryExpression {
            name: format!("P({}){}", perm, self.name),
            radices: perm.permuted_radices(),
            variables: self.variables.clone(),
            body: new_body,
        }
    }

    //TODO: Rename this to emplace
    pub fn extend(&self, placement: &[usize], radices: &QuditRadices) -> Self {
        assert!(placement.len() == self.num_qudits());
        assert!(placement.iter().enumerate().all(|(i, &p)| self.radices[i] == radices[p]));

        // kron with idenity to make this appropriately sized
        let missing: QuditRadices = radices.clone().into_iter().enumerate().filter(|(i, _)| !placement.contains(i)).map(|(_, r)| *r).collect::<QuditRadices>();
        let extended = if missing.is_empty() {
            self.clone()
        } else {
            let id = UnitaryExpression::identity("I", missing);
            self.otimes(id)
        };

        // permute according to placement
        // want to move first |placement| qudits to placement
        let extended_placement = placement.iter().map(|&r| r).chain((0..radices.num_qudits()).filter(|i| !placement.contains(i))).collect::<Vec<_>>();
        let perm = QuditPermutation::new(radices, &extended_placement);
        let permuted = extended.permute(&perm);
        permuted
    }
}

impl QuditSystem for UnitaryExpression {
    fn radices(&self) -> QuditRadices {
        self.radices.clone()
    }

    fn dimension(&self) -> usize {
        self.body.len()
    }

    fn num_qudits(&self) -> usize {
        self.radices.len()
    }
}

// pub struct MatrixExpression {
//     pub name: String,
//     pub variables: Vec<String>,
//     pub body: Vec<Vec<ComplexExpression>>,
// }

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct StateSystemExpression {
    pub name: String,
    pub radices: QuditRadices,
    pub variables: Vec<String>,
    pub body: Vec<Vec<ComplexExpression>>,
}

impl StateSystemExpression {
    /// Creates a new `StateSystemExpression` from a QGL string representation.
    ///
    /// This function parses the input string as a QGL object, then converts it
    /// into a `TensorExpression` and subsequently into a `StateSystemExpression`.
    ///
    /// # Arguments
    ///
    /// * `input` - A type that can be converted to a string reference,
    ///             representing the QGL definition of the state system expression.
    ///
    /// # Returns
    ///
    /// A new `StateSystemExpression` instance.
    pub fn new<T: AsRef<str>>(input: T) -> Self {
        TensorExpression::new(input).to_state_system_expression()
    }

    /// Evaluates the state system expression with the given arguments and returns a `Mat`.
    ///
    /// This function substitutes the provided real scalar arguments into the complex
    /// expressions that define the state system's body, and then constructs a
    /// `Mat` from the evaluated complex values.
    ///
    /// # Type Parameters
    ///
    /// * `C`: A type that implements `ComplexScalar`, representing the complex number
    ///        type for the state system elements.
    ///
    /// # Arguments
    ///
    /// * `args` - A slice of real scalar values (`C::R`) to substitute for the
    ///            variables in the expression. The order of arguments must match
    ///            the order of `self.variables`.
    ///
    /// # Returns
    ///
    /// A `Mat<C>` containing the evaluated complex elements of the state system.
    pub fn eval<C: ComplexScalar>(&self, args: &[C::R]) -> Mat<C> {
        let arg_map = self.variables.iter().zip(args.iter()).map(|(a, b)| (a.as_str(), *b)).collect();
        let nrows = self.body.len();
        let ncols = self.body[0].len();
        let mut mat = Mat::zeros(nrows, ncols);
        for i in 0..nrows {
            for j in 0..ncols {
                *mat.get_mut(i, j) = self.body[i][j].eval(&arg_map);
            }
        }
        mat
    }

    pub fn name(&self) -> String {
        self.name.clone()
    }
}

pub struct MatVecExpression {
    pub name: String,
    pub variables: Vec<String>,
    pub body: Vec<Vec<Vec<ComplexExpression>>>,
}

pub trait TensorExpressionGenerator {
    fn gen_expr(&self) -> TensorExpression;
}

impl TensorExpressionGenerator for TensorExpression {
    fn gen_expr(&self) -> TensorExpression {
        self.clone()
    }
}

pub trait UnitaryExpressionGenerator {
    fn gen_expr(&self) -> UnitaryExpression;
}

impl UnitaryExpressionGenerator for UnitaryExpression {
    fn gen_expr(&self) -> UnitaryExpression {
        self.clone()
    }
}

impl HasParams for UnitaryExpression {
    fn num_params(&self) -> usize {
        self.variables.len()
    }
}

impl<R: RealScalar> HasPeriods<R> for UnitaryExpression {
    fn periods(&self) -> Vec<Range<R>> {
        todo!()
    }
}

impl<C: ComplexScalar> UnitaryFn<C> for UnitaryExpression {
    fn write_unitary(&self, params: &[<C as ComplexScalar>::R], mut utry: MatMut<C>) {
        let arg_map = self.get_arg_map::<C>(params);
        let dim = self.radices.dimension();
        for i in 0..dim {
            for j in 0..dim {
                *utry.rb_mut().get_mut(i, j) = self.body[i][j].eval(&arg_map);
            }
        }
    }
}

impl<C: ComplexScalar> DifferentiableUnitaryFn<C> for UnitaryExpression {
    fn write_unitary_and_gradient(
        &self,
        params: &[C::R],
        mut out_utry: MatMut<C>,
        mut out_grad: MatVecMut<C>,
    ) {
        let arg_map = self.get_arg_map::<C>(params);
        let dim = self.radices.dimension();
        for i in 0..dim {
            for j in 0..dim {
                *out_utry.rb_mut().get_mut(i, j) = self.body[i][j].eval(&arg_map);
            }
        }

        let grad_expr = self.differentiate();

        for (i, grad_mat) in grad_expr.body.iter().enumerate() {
            for (j, grad_row) in grad_mat.iter().enumerate() {
                for (k, grad_elem) in grad_row.iter().enumerate() {
                    out_grad.write(i, j, k, grad_elem.eval(&arg_map));
                }
            }
        }
    }
}

impl AsRef<UnitaryExpression> for UnitaryExpression {
    fn as_ref(&self) -> &UnitaryExpression {
        self
    }
}

#[cfg(test)]
mod tests {
    use crate::{Module, ModuleBuilder};

    use qudit_core::{c64, matrix::MatVec};
    use super::*;

    #[test]
    fn test_cnot_reshape2() {
        let mut cnot = TensorExpression::new("CNOT() {
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 0, 1],
                [0, 0, 1, 0],
            ]
        }");

        let name = cnot.name().to_owned();
        let reshaped = cnot.reshape(TensorShape::Matrix(2, 8)).permute(&vec![2, 0, 1, 3]);
        // println!("{:?}", reshaped);

        let module: Module<c64> = ModuleBuilder::new("test", DifferentiationLevel::None)
            .add_tensor_expression(reshaped.clone())
            .build();

        // let mut out_tensor: MatVec<c64> = MatVec::zeros(4, 4, 2);
        let col_stride = qudit_core::memory::calc_col_stride::<c64>(2, 8);
        let mut memory = qudit_core::memory::alloc_zeroed_memory::<c64>(col_stride*8);
        
        let matmut = unsafe {
            qudit_core::matrix::MatMut::from_raw_parts_mut(memory.as_mut_ptr(), 2, 8, 1, col_stride as isize)
        };
        // let mut out_tensor: Mat<c64> = Mat::zeros(2, 8);
        // let mut out_ptr: *mut f64 = out_tensor.as_ptr_mut() as *mut f64;
        let mut out_ptr: *mut f64 = memory.as_mut_ptr() as *mut f64;
        let func = module.get_function(&name).unwrap();

        let null_ptr = std::ptr::null() as *const f64;

        unsafe { func.call(null_ptr, out_ptr); }

        println!("{:?}", matmut);
        // for r in 0..matmut.nrows() {
        //     for c in 0..matmut.ncols() {
        //         println!("({}, {}): {}", r, c, matmut.get(r, c));
        //     }
        // }
    }

    #[test]
    fn test_cnot_reshape() {
        let mut cnot = TensorExpression::new("CNOT() {
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 0, 1],
                [0, 0, 1, 0],
            ]
        }");

        let name = cnot.name().to_owned();
        let reshaped = cnot.reshape(TensorShape::Matrix(8, 2)).permute(&vec![0, 1, 3, 2]);

        let module: Module<c64> = ModuleBuilder::new("test", DifferentiationLevel::None)
            .add_tensor_expression(reshaped.clone())
            .build();

        // let mut out_tensor: MatVec<c64> = MatVec::zeros(4, 4, 2);
        let mut out_tensor: Mat<c64> = Mat::zeros(8, 2);
        let mut out_ptr: *mut f64 = out_tensor.as_ptr_mut() as *mut f64;
        let func = module.get_function(&name).unwrap();

        let null_ptr = std::ptr::null() as *const f64;

        unsafe { func.call(null_ptr, out_ptr); }

        println!("{:?}", out_tensor);

    }


    #[test]
    fn test_tensor_gen() {
        let expr = TensorExpression::new("ZZParity() {
            [
                [
                    [ 1, 0, 0, 0 ], 
                    [ 0, 0, 0, 0 ],
                    [ 0, 0, 0, 0 ],
                    [ 0, 0, 0, 1 ],
                ],
                [
                    [ 0, 0, 0, 0 ], 
                    [ 0, 1, 0, 0 ],
                    [ 0, 0, 1, 0 ],
                    [ 0, 0, 0, 0 ],
                ],
            ]
        }");

        // for elem in expr.body.iter() {
        //     println!("{:?}", elem);
        // }

        let name = expr.name().to_owned();

        let module: Module<c64> = ModuleBuilder::new("test", DifferentiationLevel::None)
            .add_tensor_expression(expr)
            .build();

        let mut out_tensor: MatVec<c64> = MatVec::zeros(4, 4, 2);
        let mut out_ptr: *mut f64 = out_tensor.as_mut_ptr() as *mut f64;
        let func = module.get_function(&name).unwrap();

        let null_ptr = std::ptr::null() as *const f64;

        unsafe { func.call(null_ptr, out_ptr); }

        println!("{:?}", out_tensor);

//         let params = vec![1.7, 2.3, 3.1];
//         let mut out_utry: Mat<c64> = Mat::zeros(2, 2);
//         let mut out_grad: MatVec<c64> = MatVec::zeros(2, 2, 3);

//         let module: Module<c64> = ModuleBuilder::new("test", DifferentiationLevel::Gradient)
//             .add_expression_with_stride(expr, out_utry.col_stride().try_into().unwrap())
//             .build();

//         println!("{}", module);

//         let u3_grad_combo_func = module.get_function_and_gradient("U3").unwrap();
//         let out_ptr = out_utry.as_mut().as_ptr_mut() as *mut f64;
//         let out_grad_ptr = out_grad.as_mut().as_mut_ptr().as_ptr() as *mut f64;

//         let start = std::time::Instant::now();
//         for _ in 0..1000000 {
//             unsafe { u3_grad_combo_func.call(params.as_ptr(), out_ptr, out_grad_ptr); }
//         }
//         let duration = start.elapsed();
//         println!("Time elapsed in expensive_function() is: {:?}", duration);
//         println!("Average time: {:?}", duration / 1000000);

//         println!("{:?}", out_utry);
//         println!("{:?}", out_grad);
    }
}
