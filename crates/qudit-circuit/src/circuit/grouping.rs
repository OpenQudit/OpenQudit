use crate::cycle::CycleList;
use crate::cycle::{CycleId, CycleIndex};
use crate::instruction::{Instruction, InstructionId};
use crate::operation::OpCode;
use crate::operation::OperationSet;
use crate::operation::{
    CircuitOperation, DirectiveOperation, ExpressionOperation, OpKind, Operation,
};
use crate::Result;
use crate::param::{Argument as ParameterEntry, ArgumentList, Parameter, ParameterVector};
use crate::wire::Wire;
use crate::wire::WireList;
use qudit_core::Radices;
use qudit_core::array::Tensor;
use qudit_core::{
    ClassicalSystem, ComplexScalar, HasParams, HybridSystem, ParamIndices, ParamInfo, QuditSystem,
};
use qudit_expr::index::IndexDirection;
use qudit_expr::{
    BraSystemExpression, FUNCTION, KetExpression, KrausOperatorsExpression, TensorExpression,
    UnitaryExpression, UnitarySystemExpression,
};
use qudit_tensor::{QuditCircuitTensorNetworkBuilder, QuditTensor, QuditTensorNetwork};
use rustc_hash::{FxHashMap, FxHashSet};
use std::collections::HashMap;
use super::*;



pub struct CircuitRegion {
    inst_ids: Vec<InstructionId>,
}

impl CircuitRegion {
    pub fn new(inst_ids: Vec<InstructionId>) -> Self {
        if inst_ids.len() > 1 {
            if inst_ids.len() <= 16 {
                for i in 0..inst_ids.len() {
                    for j in (i + 1)..inst_ids.len() {
                        if inst_ids[i] == inst_ids[j] {
                            panic!("Duplicate InstructionId found in CircuitRegion.");
                        }
                    }
                }
            } else {
                let mut seen = rustc_hash::FxHashSet::default();
                for id in &inst_ids {
                    if !seen.insert(*id) {
                        panic!("Duplicate InstructionId found in CircuitRegion.");
                    }
                }
            }
        }

        CircuitRegion { inst_ids }
    }

    pub fn inst_ids(&self) -> &[InstructionId] {
        &self.inst_ids
    }

    pub fn max_cycle(&self, circuit: &QuditCircuit) -> Option<CycleIndex> {
        match (self.inst_ids.iter()
                .map(|inst_id| circuit.cycle_id_to_index(inst_id.cycle()))
                .max()) {
            Some(Some(cycle)) => Some(cycle),
            _ => None,
        }
    }

    pub fn min_cycle(&self, circuit: &QuditCircuit) -> Option<CycleIndex> {
        match (self.inst_ids.iter()
                .map(|inst_id| circuit.cycle_id_to_index(inst_id.cycle()))
                .min()) {
            Some(Some(cycle)) => Some(cycle),
            _ => None,
        }
    }

    pub fn min_qudit(&self, circuit: &QuditCircuit) -> Option<usize> {
        self.inst_ids
            .iter()
            .filter_map(|inst_id| circuit.get(*inst_id))
            .flat_map(|inst| inst.wires().qudits().min())
            .min()
    }

    pub fn max_qudit(&self, circuit: &QuditCircuit) -> Option<usize> {
        self.inst_ids
            .iter()
            .filter_map(|inst_id| circuit.get(*inst_id))
            .flat_map(|inst| inst.wires().qudits().max())
            .max()
    }

    pub fn min_dit(&self, circuit: &QuditCircuit) -> Option<usize> {
        self.inst_ids
            .iter()
            .filter_map(|inst_id| circuit.get(*inst_id))
            .flat_map(|inst| inst.wires().dits().min())
            .min()
    }

    pub fn max_dit(&self, circuit: &QuditCircuit) -> Option<usize> {
        self.inst_ids
            .iter()
            .filter_map(|inst_id| circuit.get(*inst_id))
            .flat_map(|inst| inst.wires().dits().max())
            .max()
    }

    pub fn wires(&self, circuit: &QuditCircuit) -> WireList {
        let mut qudits = rustc_hash::FxHashSet::default();
        let mut dits = rustc_hash::FxHashSet::default();

        for inst_id in &self.inst_ids {
            if let Some(inst) = circuit.get(*inst_id) {
                qudits.extend(inst.wires().qudits());
                dits.extend(inst.wires().dits());
            }
        }

        let mut sorted_qudits: Vec<usize> = qudits.into_iter().collect();
        sorted_qudits.sort_unstable();
        let mut sorted_dits: Vec<usize> = dits.into_iter().collect();
        sorted_dits.sort_unstable();

        WireList::new(sorted_qudits, sorted_dits)
    }
}

impl std::ops::Deref for CircuitRegion {
    type Target = [InstructionId];

    fn deref(&self) -> &Self::Target {
        &self.inst_ids
    }
}

impl AsRef<CircuitRegion> for CircuitRegion {
    fn as_ref(&self) -> &Self {
        self
    }
}

/// Grouping and Subcircuits
impl QuditCircuit {
    /// Logically group instruction ids into a circuit region while checking validity
    pub fn form_region(&self, inst_ids: &[InstructionId]) -> Option<CircuitRegion> {
        // Check if all instruction IDs are valid in the circuit.
        if !inst_ids.iter().all(|&id| self.is_valid_id(id)) {
            return None;
        }

        let region = CircuitRegion::new(inst_ids.to_vec());
        let check = self.is_convex(&region);
        match check {
            true => Some(region),
            false => None,
        }
    }

    fn slice<R: Into<CircuitRegion>>(&self, region: R) -> QuditCircuit {
        // Copy the gates in the region into a new circuit and return it
        todo!()
    }

    // fn surround(&self, inst_id: InstructionId, block_size: usize) -> CircuitRegion {
    //     todo!()
    // }

    // fn group(&mut self, inst_ids: &[InstructionId]) -> Option<InstructionId> {
    //     if !self.is_convex(inst_ids) {
    //         return None;
    //     }
        
    //     //  - The gates in the region are collected into a subcircuit
    //     // let wire_map = []
    //     // let params = {}
    //     // for inst_id in inst_ids {
    //     //      let inst = circuit + inst_id;
    //     //      let subcircuit_inst = inst.copy().map_thru(qudit_map)  # do I have to map
    //     //      parameters?
    //     //      params.update(inst.params())
    //     // }
    //     // let qudit_radices = 
    //     // let dit_radices =
    //     // let num_params = params.len()
    //     // let subcircuit = {qudit_radices, dit_radices, subcircuit_insts, num_params)
        

    //     //  - The region is replaced with a subcirucit
    //     //      - Find or create a cycle to insert
    //     //          - Check easy case, there exists a cycle in the region that contains
    //     //              all qudits and dits -> first one becomes cycle to insert
    //     todo!()
    // }

    // fn batch_group<R: Into<CircuitRegion>>(&mut self, regions: &[R]) -> Vec<InstructionId> {
    //     todo!()
    // }

    // fn ungroup(&self, inst_id: InstructionId) {
    //     todo!()
    // }

    // fn batch_ungroup(&self, inst_id: InstructionId) {
    //     todo!()
    // }

    // fn ungroup_all(&self, inst_id: InstructionId) {
    //     todo!()
    // }

    /// Return true if a region is convex.
    ///
    /// A region is convex if all gates between two members are also members.
    fn is_convex<R: AsRef<CircuitRegion>>(&self, region: R) -> bool {
        let region_ref = region.as_ref();

        if !self.is_region_in_circuit(region_ref) {
            return false;
        }

        let region_inst_ids: FxHashSet<InstructionId> = region_ref.inst_ids().iter().cloned().collect();
        if region_inst_ids.is_empty() {
            return true; // An empty region is convex
        }

        // Get the maximum cycle index of the region.
        // This is guaranteed to exist if the region is valid and non-empty.
        let region_max_cycle_idx = region_ref
            .max_cycle(self)
            .expect("Non-empty region in circuit must have a max cycle.");

        // A set to store instructions outside the region that are confirmed
        // not to lead back into the region. This prevents redundant checks.
        let mut known_to_never_reenter = FxHashSet::default();

        // Sort instruction IDs based on their cycle index, largest first.
        let mut sorted_region_inst_ids: Vec<InstructionId> = region_ref.inst_ids().to_vec();
        sorted_region_inst_ids.sort_unstable_by_key(|id| {
            std::cmp::Reverse(self.cycle_id_to_index(id.cycle()).expect("Instruction cycle must exist").0)
        });

        // Walk all instructions in the region starting from max cycle (right-to-left).
        for inst_id in sorted_region_inst_ids {
            let cycle_idx = self
                .cycle_id_to_index(inst_id.cycle())
                .expect("Instruction cycle must exist.");

            // Instructions furthest to the right in the region can never violate convexity.
            if cycle_idx == region_max_cycle_idx {
                continue;
            }

            // Start a frontier for exhaustive search
            let mut frontier: FxHashSet<InstructionId> = self.next(inst_id)
                .values()
                // Only iterate instructions that exit the region, since we are
                // looking for an instruction in the expansion of an iterate that
                // "re-enters" the set.
                .filter(|inst_id| !region_inst_ids.contains(&inst_id))
                .copied()
                .collect();

            while let Some(inst_id2) = frontier.iter().next().copied() {
                frontier.remove(&inst_id2);

                // Stop walking if we reach or pass the maximum cycle of the region.
                let inst_id2_cycle_idx = self
                    .cycle_id_to_index(inst_id2.cycle())
                    .expect("Instruction cycle must exist.");

                if inst_id2_cycle_idx >= region_max_cycle_idx {
                    continue;
                }

                let expansion = self.next(inst_id2);

                // If any instruction in the expansion is inside the region,
                // it means there's a path that exited the region and re-entered it.
                // This violates convexity.
                if expansion.values().any(|p| region_inst_ids.contains(p)) {
                    return false;
                }

                frontier.extend(expansion.values()
                    .filter(|inst_id| !region_inst_ids.contains(inst_id))
                    .filter(|inst_id| !known_to_never_reenter.contains(*inst_id)));
                known_to_never_reenter.insert(inst_id2);
            }
        }

        true
    }

    fn is_region_in_circuit<R: AsRef<CircuitRegion>>(&self, region: R) -> bool {
        let region = region.as_ref();

        if region.inst_ids().is_empty() {
            return true;
        }

        let cycle_check = match region.max_cycle(self) {
            None => false,
            Some(max_cycle) => usize::from(max_cycle) < self.num_cycles(),
        };

        let qudit_check = match region.max_qudit(self) {
            None => false,
            Some(max_qudit) => max_qudit < self.num_qudits(),
        };

        let dit_check = match region.max_dit(self) {
            None => false,
            Some(max_dit) => max_dit < self.num_dits(),
        };

        cycle_check && qudit_check && dit_check
    }
}

