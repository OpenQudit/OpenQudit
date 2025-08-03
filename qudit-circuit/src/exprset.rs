use std::collections::BTreeMap;

use qudit_expr::TensorExpression;

use crate::operation::{Operation, OperationReference};

#[derive(Clone, Debug, PartialEq)]
pub struct ExpressionSet {
    data: BTreeMap<(String, usize), Vec<(TensorExpression, usize)>>,
    idx_counter: usize,
}

impl ExpressionSet {
    pub fn new() -> Self {
        ExpressionSet { data: BTreeMap::new(), idx_counter: 0 }
    }

    pub fn insert(&mut self, expr: TensorExpression) -> usize {
        let name = expr.name().to_string();
        let dimension_len = expr.dimensions().len();
        let key = (name, dimension_len);

        // Get a mutable reference to the vector associated with the key.
        // If the key does not exist, insert a new empty vector.
        let exprs_vec = self.data.entry(key).or_insert_with(Vec::new);

        for (existing_expr, idx) in exprs_vec.iter() {
            if existing_expr == &expr {
                return *idx;
            }
        }

        // If the expression is not found, assign a new index,
        // add it to the vector, and increment the counter.
        let current_idx = self.idx_counter;
        exprs_vec.push((expr, current_idx));
        self.idx_counter += 1;
        current_idx
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct OperationSet {
    data: BTreeMap<String, Vec<(Operation, OperationReference)>>,
    map: BTreeMap<OperationReference, Operation>,
    idx_counter: u64,
}

impl OperationSet {
    pub fn new() -> Self {
        OperationSet {
            data: BTreeMap::new(),
            map: BTreeMap::new(),
            idx_counter: 0
        }
    }

    pub fn insert(&mut self, expr: Operation) -> OperationReference {
        // TODO: this name mapping implements label-based identification
        // replace it, once expression data structures are optimized with
        // fast equal
        let name = expr.name().to_string();

        // Get a mutable reference to the vector associated with the key.
        // If the key does not exist, insert a new empty vector.
        let exprs_vec = self.data.entry(name).or_insert_with(Vec::new);

        for (existing_expr, idx) in exprs_vec.iter() {
            if existing_expr == &expr {
                return *idx;
            }
        }

        // If the expression is not found, assign a new index,
        // add it to the vector, and increment the counter.
        let current_idx = OperationReference::new(self.idx_counter);
        exprs_vec.push((expr.clone(), current_idx));
        self.map.insert(current_idx, expr);
        self.idx_counter += 1;
        current_idx
    }

    pub fn get(&self, index: &OperationReference) -> Option<&Operation> {
        self.map.get(index)
    }
}
