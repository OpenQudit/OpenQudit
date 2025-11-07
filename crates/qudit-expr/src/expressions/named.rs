use itertools::Itertools;
use qudit_core::RealScalar;
use std::{
    collections::HashMap,
    ops::{Deref, DerefMut},
};

use crate::ComplexExpression;

#[derive(PartialEq, Eq, Debug, Clone)]
pub struct ExpressionBody {
    body: Vec<ComplexExpression>,
}

impl ExpressionBody {
    pub fn new(body: Vec<ComplexExpression>) -> Self {
        Self { body }
    }

    pub fn num_elements(&self) -> usize {
        self.body.len()
    }

    pub fn elements(&self) -> &[ComplexExpression] {
        &self.body
    }

    pub fn elements_mut(&mut self) -> &mut [ComplexExpression] {
        &mut self.body
    }

    pub fn conjugate(&mut self) {
        for expr in self.body.iter_mut() {
            expr.conjugate_in_place()
        }
    }

    pub fn apply_element_permutation(&mut self, elem_perm: &[usize]) {
        // TODO: do physical element permutation in place via transpositions
        let mut swap_vec = vec![];
        std::mem::swap(&mut swap_vec, &mut self.body);
        self.body = swap_vec
            .into_iter()
            .enumerate()
            .sorted_by(|(old_idx_a, _), (old_idx_b, _)| {
                elem_perm[*old_idx_a].cmp(&elem_perm[*old_idx_b])
            })
            .map(|(_, expr)| expr)
            .collect();
    }

    pub fn rename(&mut self, var_map: HashMap<String, String>) {
        for elem in self.body.iter_mut() {
            *elem = elem.map_var_names(&var_map);
        }
    }
}

impl From<Vec<ComplexExpression>> for ExpressionBody {
    fn from(value: Vec<ComplexExpression>) -> Self {
        ExpressionBody::new(value)
    }
}

impl From<ExpressionBody> for Vec<ComplexExpression> {
    fn from(value: ExpressionBody) -> Self {
        value.body
    }
}

impl AsRef<[ComplexExpression]> for ExpressionBody {
    fn as_ref(&self) -> &[ComplexExpression] {
        self.elements()
    }
}

impl Deref for ExpressionBody {
    type Target = [ComplexExpression];

    fn deref(&self) -> &Self::Target {
        &self.body
    }
}

impl DerefMut for ExpressionBody {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.body
    }
}

#[derive(PartialEq, Eq, Debug, Clone)]
pub struct BoundExpressionBody {
    variables: Vec<String>,
    body: ExpressionBody,
}

impl BoundExpressionBody {
    pub fn new<B: Into<ExpressionBody>>(variables: Vec<String>, body: B) -> Self {
        Self {
            variables,
            body: body.into(),
        }
    }

    pub fn num_params(&self) -> usize {
        self.variables.len()
    }

    pub fn variables(&self) -> &[String] {
        &self.variables
    }

    pub fn set_variables(&mut self, new_variables: Vec<String>) {
        self.variables = new_variables;
    }

    pub fn num_elements(&self) -> usize {
        self.body.num_elements()
    }

    pub fn elements(&self) -> &[ComplexExpression] {
        self.body.elements()
    }

    pub fn elements_mut(&mut self) -> &mut [ComplexExpression] {
        self.body.elements_mut()
    }

    pub fn conjugate(&mut self) {
        self.body.conjugate()
    }

    pub fn destruct(self) -> (Vec<String>, Vec<ComplexExpression>) {
        let Self { variables, body } = self;
        (variables, body.into())
    }

    pub fn apply_element_permutation(&mut self, elem_perm: &[usize]) {
        self.body.apply_element_permutation(elem_perm);
    }

    pub fn alpha_rename(&mut self, starting_number: Option<usize>) {
        let mut var_id = starting_number.unwrap_or_default();

        let mut var_map = HashMap::new();
        let mut new_vars = Vec::new();

        for var in self.variables() {
            let new_var_name = format!("alpha_{}", var_id);
            new_vars.push(new_var_name.clone());
            var_map.insert(var.clone(), new_var_name);
            var_id += 1;
        }

        self.body.rename(var_map);
        self.variables = new_vars;
    }

    pub fn get_arg_map<R: RealScalar>(&self, args: &[R]) -> HashMap<&str, R> {
        self.variables()
            .iter()
            .zip(args.iter())
            .map(|(a, b)| (a.as_str(), *b))
            .collect()
    }
}

impl AsRef<[ComplexExpression]> for BoundExpressionBody {
    fn as_ref(&self) -> &[ComplexExpression] {
        self.elements()
    }
}

impl Deref for BoundExpressionBody {
    type Target = ExpressionBody;

    fn deref(&self) -> &Self::Target {
        &self.body
    }
}

impl DerefMut for BoundExpressionBody {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.body
    }
}

#[derive(Debug, Clone)]
pub struct NamedExpression {
    name: String,
    body: BoundExpressionBody,
}

impl NamedExpression {
    pub fn new<S: Into<String>, B: Into<ExpressionBody>>(
        name: S,
        variables: Vec<String>,
        body: B,
    ) -> Self {
        Self {
            name: name.into(),
            body: BoundExpressionBody::new(variables, body),
        }
    }

    pub fn from_body_with_name(name: String, body: BoundExpressionBody) -> Self {
        Self { name, body }
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn set_name(&mut self, new_name: impl Into<String>) {
        self.name = new_name.into();
    }

    pub fn set_variables(&mut self, new_variables: Vec<String>) {
        self.body.set_variables(new_variables);
    }

    pub fn destruct(self) -> (String, Vec<String>, Vec<ComplexExpression>) {
        let Self { name, body } = self;
        let (variables, body) = body.destruct();
        (name, variables, body)
    }

    pub fn apply_element_permutation(&mut self, elem_perm: &[usize]) {
        self.body.apply_element_permutation(elem_perm);
    }
}

impl AsRef<[ComplexExpression]> for NamedExpression {
    fn as_ref(&self) -> &[ComplexExpression] {
        self.elements()
    }
}

impl AsRef<BoundExpressionBody> for NamedExpression {
    fn as_ref(&self) -> &BoundExpressionBody {
        &self.body
    }
}

impl<B: AsRef<[ComplexExpression]>> PartialEq<B> for NamedExpression {
    fn eq(&self, other: &B) -> bool {
        self.elements() == other.as_ref()
    }
}

impl Eq for NamedExpression {}

impl Deref for NamedExpression {
    type Target = BoundExpressionBody;

    fn deref(&self) -> &Self::Target {
        &self.body
    }
}

impl DerefMut for NamedExpression {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.body
    }
}
