use qudit_core::{ClassicalSystem, HybridSystem, QuditSystem};
use qudit_core::{HasParams, Radices};

use slotmap::Key;
use slotmap::SlotMap;
use slotmap::new_key_type;
new_key_type! { pub struct CircuitId; }

use crate::instruction::Instruction;

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub struct CircuitOperation {
    qudit_radices: Radices,
    dit_radices: Radices,
    instructions: Vec<Instruction>,
    num_params: usize,
}

impl HasParams for CircuitOperation {
    fn num_params(&self) -> usize {
        self.num_params
    }
}

impl QuditSystem for CircuitOperation {
    fn radices(&self) -> Radices {
        self.qudit_radices.clone()
    }
}

impl ClassicalSystem for CircuitOperation {
    fn radices(&self) -> Radices {
        self.dit_radices.clone()
    }
}

impl HybridSystem for CircuitOperation {}

#[derive(Clone)]
pub struct CircuitCache {
    circuits: SlotMap<CircuitId, CircuitOperation>,
}

impl CircuitCache {
    /// Initialize a circuit cache
    pub fn new() -> Self {
        CircuitCache {
            circuits: SlotMap::with_key(),
        }
    }

    /// Insert a circuit into the cache
    pub fn insert(&mut self, circuit: CircuitOperation) -> CircuitId {
        let id = self.circuits.insert(circuit);
        if id.data().as_ffi() & (0b111 << 61) != 0 {
            panic!("CircuitOperation cache overflow.");
        }
        id
    }

    pub fn remove(&mut self, circuit_id: CircuitId) -> Option<CircuitOperation> {
        self.circuits.remove(circuit_id)
    }

    #[allow(dead_code)]
    pub fn get(&self, circuit_id: CircuitId) -> Option<&CircuitOperation> {
        self.circuits.get(circuit_id)
    }

    #[allow(dead_code)]
    pub fn num_params(&self, circuit_id: CircuitId) -> Option<usize> {
        self.get(circuit_id).map(|c| c.num_params())
    }
}
