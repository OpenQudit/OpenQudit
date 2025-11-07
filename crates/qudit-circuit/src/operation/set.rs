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

    pub fn insert(&mut self, op: Operation) -> OpCode {
        let code = match op {
            Operation::Expression(e) => self.insert_expression(e),
            Operation::Subcircuit(c) => self.insert_subcircuit(c),
            Operation::Directive(d) => self.convert_directive(d),
        };
        self.increment(code);
        code
    }

    pub fn insert_expression(&mut self, op: ExpressionOperation) -> OpCode {
        let expression_type = op.expr_type();
        let expr_id = self.expressions.lock().unwrap().insert(op);
        let op_ref = OpCode::new(OpKind::Expression, expr_id as u64);
        match self.op_to_expr_map.get(&op_ref) {
            None => {
                self.op_to_expr_map.insert(op_ref.clone(), expression_type);
            }
            Some(expr_type) => assert_eq!(&expression_type, expr_type),
        }
        op_ref
    }

    pub fn insert_expression_with_dits(
        &mut self,
        op: ExpressionOperation,
        dit_radices: &[usize],
    ) -> OpCode {
        if dit_radices.len() == 0 {
            let op_ref = self.insert_expression(op);
            self.increment(op_ref); // Yeah, it's a mess.
            return op_ref;
        }

        let expression_type = op.expr_type();
        let mut tensor_expr: TensorExpression = op.into();

        // Reindex the expression's batch dimensions to match dits
        let batch = dit_radices.iter().map(|r| (IndexDirection::Batch, *r));
        let outs = tensor_expr
            .indices()
            .iter()
            .filter(|idx| idx.direction() == IndexDirection::Output)
            .map(|idx| (idx.direction(), idx.index_size()));
        let ins = tensor_expr
            .indices()
            .iter()
            .filter(|idx| idx.direction() == IndexDirection::Input)
            .map(|idx| (idx.direction(), idx.index_size()));
        let new_indices = batch
            .chain(outs)
            .chain(ins)
            .enumerate()
            .map(|(i, (d, s))| TensorIndex::new(d, i, s))
            .collect();
        tensor_expr.reindex(new_indices);

        let expr_id = self.expressions.lock().unwrap().insert(tensor_expr);

        let op_ref = OpCode::new(OpKind::Expression, expr_id as u64);
        match self.op_to_expr_map.get(&op_ref) {
            None => {
                self.op_to_expr_map.insert(op_ref.clone(), expression_type);
            }
            Some(expr_type) => assert_eq!(&expression_type, expr_type),
        }
        self.increment(op_ref); // Yeah, it's a mess.
        op_ref
    }

    pub fn insert_subcircuit(&mut self, op: CircuitOperation) -> OpCode {
        let circuit_id = self.subcircuits.insert(op);
        let op_ref = OpCode::new(OpKind::Subcircuit, circuit_id.data().as_ffi());
        op_ref
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

    pub fn indices(&self, op_code: OpCode) -> Vec<TensorIndex> {
        match op_code.kind() {
            OpKind::Expression => self.expressions.lock().unwrap().indices(op_code.id()),
            OpKind::Subcircuit => todo!(),
            OpKind::Directive => todo!(),
        }
    }

    #[allow(dead_code)]
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

    #[allow(dead_code)]
    pub fn name(&self, op_code: OpCode) -> String {
        match op_code.kind() {
            OpKind::Expression => self.expressions.lock().unwrap().name(op_code.id()),
            OpKind::Subcircuit => todo!(),
            OpKind::Directive => todo!(),
        }
    }

    // pub fn get_cached_expression(&self, index: &OpCode) -> Option<&CachedExpressionOperation> {
    //     self.op_to_expr_map.get(index)
    // }
}
