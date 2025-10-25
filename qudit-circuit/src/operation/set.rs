use std::{cell::RefCell, collections::BTreeMap, rc::Rc, sync::{Arc, Mutex}};

use qudit_expr::{index::{IndexDirection, TensorIndex}, ExpressionCache, ExpressionId, TensorExpression};
use slotmap::{Key, KeyData};

use super::kind::OpKind;
use crate::{operation::{directive::DirectiveOperation, expression::{ExpressionOpKind, ExpressionOperation}, subcircuit::{CircuitCache, CircuitOperation}, Operation}, OpCode};

#[derive(Clone)]
pub struct OperationSet {
    expressions: Arc<Mutex<ExpressionCache>>,
    subcircuits: CircuitCache,
    op_to_expr_map: BTreeMap<OpCode, ExpressionOpKind>,
}

impl OperationSet {
    pub fn new() -> Self {
        OperationSet {
            expressions: ExpressionCache::new_shared(),
            subcircuits: CircuitCache::new(),
            op_to_expr_map: BTreeMap::new(),
        }
    }

    pub fn expressions(&self) -> Arc<Mutex<ExpressionCache>> {
        self.expressions.clone()
    }

    pub fn insert(&mut self, op: Operation) -> OpCode {
        match op {
            Operation::Expression(e) => self.insert_expression(e),
            Operation::Subcircuit(c) => self.insert_subcircuit(c),
            Operation::Directive(d) => self.convert_directive(d),
        }
    }

    pub fn insert_expression(&mut self, op: ExpressionOperation) -> OpCode {
        let expression_type = op.expr_type();
        let expr_id = self.expressions.lock().unwrap().insert(op);
        let op_ref = OpCode::new(OpKind::Expression, expr_id as u64);
        match self.op_to_expr_map.get(&op_ref) {
            None => { self.op_to_expr_map.insert(op_ref.clone(), expression_type); },
            Some(expr_type) => assert_eq!(&expression_type, expr_type),
        }
        op_ref
    }

    pub fn insert_expression_with_dits(&mut self, op: ExpressionOperation, dit_radices: &[usize]) -> OpCode {
        let expression_type = op.expr_type();
        let mut tensor_expr: TensorExpression = op.into();

        // Reindex the expression's batch dimensions to match dits
        let batch = dit_radices.iter()
            .map(|r| (IndexDirection::Batch, *r));
        let outs = tensor_expr.indices()
            .iter()
            .filter(|idx| idx.direction() == IndexDirection::Output)
            .map(|idx| (idx.direction(), idx.index_size()));
        let ins = tensor_expr.indices()
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
            None => { self.op_to_expr_map.insert(op_ref.clone(), expression_type); },
            Some(expr_type) => assert_eq!(&expression_type, expr_type),
        }
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
    
    pub fn num_params(&self, index: &OpCode) -> Option<usize> {
        match index.kind() {
            OpKind::Expression => {
                let expr_id = index.id();
                Some(self.expressions.lock().unwrap().num_params(expr_id))
            },
            OpKind::Subcircuit => {
                let circuit_id = index.id();
                self.subcircuits.num_params(KeyData::from_ffi(circuit_id).into())
            },
            OpKind::Directive => Some(0),
        }
    }

    // pub fn get_cached_expression(&self, index: &OpCode) -> Option<&CachedExpressionOperation> {
    //     self.op_to_expr_map.get(index)
    // }
}
