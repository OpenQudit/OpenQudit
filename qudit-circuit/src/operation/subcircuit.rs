use qudit_core::{ClassicalSystem, HybridSystem, QuditSystem};
use qudit_core::{HasParams, QuditRadices};

use slotmap::new_key_type;
use slotmap::SlotMap;
use slotmap::Key;
new_key_type! { pub struct CircuitId; }

use crate::instruction::Instruction;

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub struct CircuitOperation {
    qudit_radices: QuditRadices,
    dit_radices: QuditRadices,
    instructions: Vec<Instruction>,
    num_params: usize,
}

impl HasParams for CircuitOperation {
    fn num_params(&self) -> usize {
        self.num_params
    }
}

impl QuditSystem for CircuitOperation {
    fn radices(&self) -> QuditRadices {
        self.qudit_radices.clone()
    }
}

impl ClassicalSystem for CircuitOperation {
    fn radices(&self) -> QuditRadices {
        self.dit_radices.clone()
    }
}

impl HybridSystem for CircuitOperation {}

#[derive(Clone)]
pub struct CircuitCache {
    circuits: SlotMap<CircuitId, CircuitOperation>,
}

impl CircuitCache {
    pub fn new() -> Self {
        CircuitCache {
            circuits: SlotMap::with_key(),
        }
    }

    pub fn insert(&mut self, circuit: CircuitOperation) -> CircuitId {
        let id = self.circuits.insert(circuit);
        if id.data().as_ffi() & (0b111 << 61) != 0 {
            panic!("CircuitOperation cache overflow.");
        }
        id
    }

    pub fn get(&self, circuit_id: CircuitId) -> Option<&CircuitOperation> {
        self.circuits.get(circuit_id)
    }

    pub fn num_params(&self, circuit_id: CircuitId) -> Option<usize> {
        self.get(circuit_id).map(|c| c.num_params())
    }
}

