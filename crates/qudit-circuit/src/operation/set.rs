use std::{
    collections::{BTreeMap, hash_map::Entry},
    sync::{Arc, Mutex},
};

use qudit_expr::{
    ExpressionCache, TensorExpression,
    index::{IndexDirection, TensorIndex},
};
use rustc_hash::FxHashMap;
use slotmap::{Key, KeyData};

use super::kind::OpKind;
use crate::{
    OpCode,
    operation::{
        Operation,
        directive::DirectiveOperation,
        expression::{ExpressionOpKind, ExpressionOperation},
        subcircuit::{CircuitCache, CircuitId, CircuitOperation},
    },
};

#[derive(Clone)]
pub struct OperationSet {
    expressions: Arc<Mutex<ExpressionCache>>,
    subcircuits: CircuitCache,
    op_to_expr_map: BTreeMap<OpCode, ExpressionOpKind>,
    op_counts: FxHashMap<OpCode, usize>,
}

pub struct OperationSetIter<'a> {
    op_codes: std::collections::btree_set::IntoIter<OpCode>,
    operation_set: &'a OperationSet,
}

impl<'a> Iterator for OperationSetIter<'a> {
    type Item = Operation;

    fn next(&mut self) -> Option<Self::Item> {
        while let Some(op_code) = self.op_codes.next() {
            if let Some(operation) = self.operation_set.get(op_code) {
                return Some(operation);
            }
            // Skip invalid opcodes (shouldn't happen)
        }
        None
    }
}

pub struct OpCodesIter {
    op_codes: std::collections::btree_set::IntoIter<OpCode>,
}

impl Iterator for OpCodesIter {
    type Item = OpCode;

    fn next(&mut self) -> Option<Self::Item> {
        self.op_codes.next()
    }
}

pub struct OperationsWithCountsIter<'a> {
    ops_iter: std::collections::btree_map::IntoIter<OpCode, usize>,
    operation_set: &'a OperationSet,
}

impl<'a> Iterator for OperationsWithCountsIter<'a> {
    type Item = (Operation, usize);

    fn next(&mut self) -> Option<Self::Item> {
        while let Some((op_code, count)) = self.ops_iter.next() {
            if let Some(operation) = self.operation_set.get(op_code) {
                return Some((operation, count));
            }
            // Skip invalid opcodes (shouldn't happen)
        }
        None
    }
}

// Optional: Implement IntoIterator for convenience
impl<'a> IntoIterator for &'a OperationSet {
    type Item = Operation;
    type IntoIter = OperationSetIter<'a>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}
impl OperationSet {
    pub fn new() -> Self {
        OperationSet {
            expressions: ExpressionCache::new_shared(),
            subcircuits: CircuitCache::new(),
            op_to_expr_map: BTreeMap::new(),
            op_counts: FxHashMap::default(),
        }
    }

    pub fn count(&self, op_code: OpCode) -> usize {
        *self.op_counts.get(&op_code).unwrap()
    }

    /// Increment internal instruction type counter.
    pub(crate) fn increment(&mut self, op_code: OpCode) {
        self.op_counts
            .entry(op_code)
            .and_modify(|count| *count += 1)
            .or_insert(1);
    }

    /// Decrement internal instruction type counter.
    pub(crate) fn decrement(&mut self, op_code: OpCode) {
        if let Entry::Occupied(mut entry) = self.op_counts.entry(op_code) {
            let count = entry.get_mut();
            *count -= 1;
            if *count == 0 {
                entry.remove();
                match op_code.kind() {
                    OpKind::Expression => self.expressions.lock().unwrap().remove(op_code.id()),
                    OpKind::Subcircuit => {
                        self.subcircuits
                            .remove(CircuitId::from(KeyData::from_ffi(op_code.id())));
                    }
                    OpKind::Directive => {}
                };
            }
        }
    }

    pub fn num_operations(&self) -> usize {
        self.op_counts.values().sum()
    }

    pub fn expressions(&self) -> Arc<Mutex<ExpressionCache>> {
        self.expressions.clone()
    }

    pub fn insert(&mut self, op: Operation) -> crate::Result<OpCode> {
        let code = match op {
            Operation::Expression(e) => self.insert_expression(e)?,
            Operation::Subcircuit(c) => self.insert_subcircuit(c),
            Operation::Directive(d) => self.convert_directive(d),
        };
        self.increment(code);
        Ok(code)
    }

    pub fn insert_expression(&mut self, op: ExpressionOperation) -> crate::Result<OpCode> {
        let expression_type = op.expr_type();
        let expr_id = self.expressions.lock().unwrap().insert(op);
        let op_ref = OpCode::new(OpKind::Expression, expr_id);
        match self.op_to_expr_map.get(&op_ref) {
            None => {
                self.op_to_expr_map.insert(op_ref, expression_type);
            }
            Some(expr_type) => assert_eq!(&expression_type, expr_type), // TODO: Make an error...
        }
        Ok(op_ref)
    }

    pub fn insert_subcircuit(&mut self, op: CircuitOperation) -> OpCode {
        let circuit_id = self.subcircuits.insert(op);

        OpCode::new(OpKind::Subcircuit, circuit_id.data().as_ffi())
    }

    pub fn convert_directive(&self, op: DirectiveOperation) -> OpCode {
        OpCode::new(OpKind::Directive, op as u64)
    }

    //     pub fn get_expression(&self, index: &OperationReference) -> Option<ExpressionId> {
    //         self.op_to_expr_map.get(index).map(|cached| match cached {
    //             CachedExpressionOperation::UnitaryGate(e) => *e,
    //             CachedExpressionOperation::KrausOperators(e) => *e,
    //             CachedExpressionOperation::TerminatingMeasurement(e) => *e,
    //             CachedExpressionOperation::ClassicallyControlledUnitary(e) => *e,
    //             CachedExpressionOperation::QuditInitialization(e) => *e,
    //         })
    //     }

    pub fn get(&self, op_code: OpCode) -> Option<Operation> {
        match op_code.kind() {
            OpKind::Expression => self
                .expressions
                .lock()
                .unwrap()
                .get(op_code.id())
                .and_then(|expr| {
                    match self.op_to_expr_map.get(&op_code) {
                        Some(ExpressionOpKind::UnitaryGate) => {
                            expr.try_into().ok().map(ExpressionOperation::UnitaryGate)
                        },
                        Some(ExpressionOpKind::KrausOperators) => {
                            expr.try_into().ok().map(ExpressionOperation::KrausOperators)
                        },
                        Some(ExpressionOpKind::TerminatingMeasurement) => {
                            expr.try_into().ok().map(ExpressionOperation::TerminatingMeasurement)
                        },
                        Some(ExpressionOpKind::ClassicallyControlledUnitary) => {
                            expr.try_into().ok().map(ExpressionOperation::ClassicallyControlledUnitary)
                        },
                        Some(ExpressionOpKind::QuditInitialization) => {
                            expr.try_into().ok().map(ExpressionOperation::QuditInitialization)
                        },
                        None => unreachable!("Expression kind not found for cached expression. This indicates an inconsistency in the OperationSet's state."),
                    }
                })
                .map(Operation::Expression),
            OpKind::Subcircuit => {
                let circuit_id = CircuitId::from(KeyData::from_ffi(op_code.id()));
                self.subcircuits.get(circuit_id).map(|c| Operation::Subcircuit(c.clone()))
            }
            OpKind::Directive => {
                crate::operation::directive::DirectiveOperation::try_from(op_code.id())
                    .ok()
                    .map(Operation::Directive)
            }
        }
    }

    pub fn indices(&self, op_code: OpCode) -> Vec<TensorIndex> {
        match op_code.kind() {
            OpKind::Expression => self.expressions.lock().unwrap().indices(op_code.id()),
            OpKind::Subcircuit => todo!(),
            OpKind::Directive => todo!(),
        }
    }

    pub fn num_params(&self, index: &OpCode) -> Option<usize> {
        match index.kind() {
            OpKind::Expression => {
                let expr_id = index.id();
                Some(self.expressions.lock().unwrap().num_params(expr_id))
            }
            OpKind::Subcircuit => {
                let circuit_id = index.id();
                self.subcircuits
                    .num_params(KeyData::from_ffi(circuit_id).into())
            }
            OpKind::Directive => Some(0),
        }
    }

    pub fn name(&self, op_code: OpCode) -> String {
        match op_code.kind() {
            OpKind::Expression => self.expressions.lock().unwrap().base_name(op_code.id()),
            OpKind::Subcircuit => todo!(),
            OpKind::Directive => todo!(),
        }
    }

    pub fn iter(&self) -> OperationSetIter<'_> {
        use std::collections::BTreeSet;
        
        // Collect unique OpCodes in a consistent order
        let op_codes: BTreeSet<OpCode> = self.op_counts.keys().copied().collect();
        
        OperationSetIter {
            op_codes: op_codes.into_iter(),
            operation_set: self,
        }
    }

    pub fn op_codes(&self) -> OpCodesIter {
        use std::collections::BTreeSet;
        
        // Collect unique OpCodes in a consistent order
        let op_codes: BTreeSet<OpCode> = self.op_counts.keys().copied().collect();
        
        OpCodesIter {
            op_codes: op_codes.into_iter(),
        }
    }

    pub fn operations_with_counts(&self) -> OperationsWithCountsIter<'_> {
        // Convert to BTreeMap for consistent ordering
        let ordered_ops: BTreeMap<OpCode, usize> = self.op_counts.iter()
            .map(|(&k, &v)| (k, v))
            .collect();
        
        OperationsWithCountsIter {
            ops_iter: ordered_ops.into_iter(),
            operation_set: self,
        }
    }
}
