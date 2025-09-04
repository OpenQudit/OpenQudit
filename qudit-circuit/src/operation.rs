use std::cell::RefCell;
use std::collections::BTreeMap;
use std::rc::Rc;

use qudit_core::{ComplexScalar, HasParams, QuditRadices, RealScalar, ToRadix};
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

// impl Operation {
//     pub fn name(&self) -> String {
//         match self {
//             Operation::UnitaryGate(gate) => gate.name().to_string(),
//             Operation::KrausOperators(t) => format!("ProjectiveMeasurement({})", t.name()),
//             Operation::TerminatingMeasurement(s) => format!("TerminatingMeasurement({})", s.name()),
//             Operation::ClassicallyControlledUnitary(g, _) => format!("ClassicallyControlled({})", g.name()),
//             Operation::QuditInitialization(s) => format!("Initialization({})", s.name()),
//         }
//     }
// }

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
        todo!()
        // match self {
        //     Operation::Gate(gate) => gate.num_params(),
        //     Operation::TerminatingMeasurement(s) => s.variables.len(),
        //     Operation::ClassicallyControlled(s, _) => s.num_params(),
        //     _ => todo!()
        //     // Operation::Subcircuit(subcircuit) => subcircuit.num_params(),
        //     // Operation::Control(_) => 0,
        // }
    }
}

#[derive(Clone)]
pub struct OperationSet {
    expressions: Rc<RefCell<ExpressionCache>>,
    map: BTreeMap<OperationReference, CachedOperation>,
    idx_counter: u64,
}

impl OperationSet {
    pub fn new() -> Self {
        OperationSet {
            expressions: ExpressionCache::new_shared(),
            map: BTreeMap::new(),
            idx_counter: 0
        }
    }

    pub fn insert(&mut self, expr: Operation) -> OperationReference {
        todo!()
        // // TODO: this name mapping implements label-based identification
        // // replace it, once expression data structures are optimized with
        // // fast equal
        // let name = expr.name().to_string();

        // // Get a mutable reference to the vector associated with the key.
        // // If the key does not exist, insert a new empty vector.
        // let exprs_vec = self.data.entry(name).or_insert_with(Vec::new);

        // for (existing_expr, idx) in exprs_vec.iter() {
        //     if existing_expr == &expr {
        //         return *idx;
        //     }
        // }

        // // If the expression is not found, assign a new index,
        // // add it to the vector, and increment the counter.
        // let current_idx = OperationReference::new(self.idx_counter);
        // exprs_vec.push((expr.clone(), current_idx));
        // self.map.insert(current_idx, expr);
        // self.idx_counter += 1;
        // current_idx
    }

    pub fn get(&self, index: &OperationReference) -> Option<&Operation> {
        todo!()
        // self.map.get(index)
    }
}
