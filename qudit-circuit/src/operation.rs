use std::cell::RefCell;
use std::collections::BTreeMap;
use std::rc::Rc;

use qudit_core::{ComplexScalar, HasParams, QuditRadices, RealScalar, ToRadix};
use qudit_expr::index::TensorIndex;
use qudit_expr::{ExpressionCache, ExpressionId, UnitaryExpression};
use qudit_expr::TensorExpression;
use qudit_expr::BraSystemExpression;
use qudit_expr::KetExpression;
use qudit_expr::KrausOperatorsExpression;
use qudit_expr::NamedExpression;
use qudit_gates::Gate;
use qudit_core::state::StateVector;
use bit_set::BitSet;
// use qudit_tree::ExpressionTree;

use crate::circuit::QuditCircuit;

pub enum OperationType {
    Gate,
    Subcircuit,
    Control,
}

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub struct DitState(u8);

impl std::fmt::Display for DitState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }
}

impl ToRadix for DitState {
    fn to_radix(self) -> u8 {
        self.0
    }

    fn is_less_than_two(self) -> bool {
        self.0 < 2
    }
}

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub struct ControlState {
    pub state: Vec<DitState>,
    pub radices: QuditRadices,
} // TODO: Make this compact on stack
//

impl ControlState {
    // TODO: look into a better name
    pub fn to_measurement_kraus_position(&self) -> usize {
        self.radices.compress(&self.state)
    }

    pub fn from_binary_state<S: AsRef<[u8]>>(state: S) -> Self {
        let state = state.as_ref();
        // TODO: Error checking
        let radices = QuditRadices::new(&vec![2; state.len()]);
        let mut dit_state = vec![];
        for s in state {
            dit_state.push(match s {
                0 => DitState(0),
                1 => DitState(1),
                _ => panic!("Bad dit state"),
            });
        }

        Self {
            state: dit_state,
            radices,
        }
    }
}

// pub something ControlState {
//      How many clbits
//      the desired state of each clbit?
// }

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Operation {

    UnitaryGate(UnitaryExpression),
    KrausOperators(KrausOperatorsExpression),
    TerminatingMeasurement(BraSystemExpression),
    ClassicallyControlledUnitary(KrausOperatorsExpression),
    QuditInitialization(KetExpression),

    // TODO: Delay
    // Subcircuit(ImmutableCircuit),
    // Barrier,
}

impl AsRef<NamedExpression> for Operation {
    fn as_ref(&self) -> &NamedExpression {
        match self {
            Operation::UnitaryGate(e) => e.as_ref(),
            Operation::KrausOperators(e) => e.as_ref(),
            Operation::TerminatingMeasurement(e) => e.as_ref(),
            Operation::ClassicallyControlledUnitary(e) => e.as_ref(),
            Operation::QuditInitialization(e) => e.as_ref(),
        }
    }
}

impl From<Operation> for TensorExpression {
    fn from(value: Operation) -> Self {
        match value {
            Operation::UnitaryGate(e) => e.into(),
            Operation::KrausOperators(e) => e.into(),
            Operation::TerminatingMeasurement(e) => e.into(),
            Operation::ClassicallyControlledUnitary(e) => e.into(),
            Operation::QuditInitialization(e) => e.into(),
        }
    }
}

impl Operation {
    pub fn discriminant(&self) -> usize {
        match self {
            Operation::UnitaryGate(_) => 0,
            Operation::KrausOperators(_) => 1,
            Operation::TerminatingMeasurement(_) => 2,
            Operation::ClassicallyControlledUnitary(_) => 3,
            Operation::QuditInitialization(_) => 4,
        }
    }
}

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub enum CachedOperation {
    UnitaryGate(ExpressionId),
    KrausOperators(ExpressionId),
    TerminatingMeasurement(ExpressionId),
    ClassicallyControlledUnitary(ExpressionId),
    QuditInitialization(ExpressionId),
}


#[derive(Clone, Debug, Hash, PartialEq, Eq, Copy, PartialOrd, Ord)]
pub struct OperationReference(u64);

impl OperationReference {
    #[inline(always)]
    pub(super) fn new(id: u64) -> Self {
        OperationReference(id)
    }
}

impl From<u64> for OperationReference {
    fn from(value: u64) -> OperationReference {
        OperationReference(value)
    }
}

impl HasParams for Operation {
    fn num_params(&self) -> usize {
        match self {
            Operation::UnitaryGate(e) => e.num_params(),
            Operation::KrausOperators(e) => e.num_params(),
            Operation::TerminatingMeasurement(e) => e.num_params(),
            Operation::ClassicallyControlledUnitary(e) => e.num_params(),
            Operation::QuditInitialization(e) => e.num_params(),
        }
    }
}

#[derive(Clone)]
pub struct OperationSet {
    expressions: Rc<RefCell<ExpressionCache>>,
    op_to_expr_map: BTreeMap<OperationReference, CachedOperation>,
    expr_to_op_map: BTreeMap<ExpressionId, OperationReference>,
    idx_counter: u64,
}

impl OperationSet {
    pub fn new() -> Self {
        OperationSet {
            expressions: ExpressionCache::new_shared(),
            op_to_expr_map: BTreeMap::new(),
            expr_to_op_map: BTreeMap::new(),
            idx_counter: 0
        }
    }

    pub fn expressions(&self) -> Rc<RefCell<ExpressionCache>> {
        self.expressions.clone()
    }

    pub fn insert(&mut self, op: Operation) -> OperationReference {
        let disc = op.discriminant();
        let expr: TensorExpression = op.into();
        let expr_id = self.expressions.borrow_mut().insert(expr);

        match self.expr_to_op_map.get(&expr_id) {
            None => {
                let op_ref = OperationReference(self.idx_counter);
                self.expr_to_op_map.insert(expr_id, op_ref);

                // TODO: replce this hack with something actually good
                let cached = match disc {
                    0 => CachedOperation::UnitaryGate(expr_id),
                    1 => CachedOperation::KrausOperators(expr_id),
                    2 => CachedOperation::TerminatingMeasurement(expr_id),
                    3 => CachedOperation::ClassicallyControlledUnitary(expr_id),
                    4 => CachedOperation::QuditInitialization(expr_id),
                    _ => unreachable!(),
                };
                self.op_to_expr_map.insert(op_ref, cached);
                self.idx_counter += 1;
                op_ref
            }
            Some(op_ref) => op_ref.clone(),
        }
    }

    pub fn get_expression(&self, index: &OperationReference) -> Option<ExpressionId> {
        self.op_to_expr_map.get(index).map(|cached| match cached {
            CachedOperation::UnitaryGate(e) => *e,
            CachedOperation::KrausOperators(e) => *e,
            CachedOperation::TerminatingMeasurement(e) => *e,
            CachedOperation::ClassicallyControlledUnitary(e) => *e,
            CachedOperation::QuditInitialization(e) => *e,
        })
    }

    pub fn indices(&self, expr_id: ExpressionId) -> Vec<TensorIndex> {
        self.expressions.borrow().indices(expr_id)
    }
    
    pub fn num_params(&self, index: &OperationReference) -> Option<usize> {
        self.get_expression(index).map(|expr_id| self.expressions.borrow().num_params(expr_id))
    }

    pub fn get_cached(&self, index: &OperationReference) -> Option<&CachedOperation> {
        self.op_to_expr_map.get(index)
    }
}

use qudit_expr::ExpressionGenerator;

impl From<qudit_gates::Gate> for Operation {
    fn from(value: qudit_gates::Gate) -> Self {
        Operation::UnitaryGate(value.generate_expression())
    }
}

impl From<UnitaryExpression> for Operation {
    fn from(value: UnitaryExpression) -> Self {
        Operation::UnitaryGate(value)
    }
}
