use super::*;
use crate::instruction::InstructionId;
use crate::wire::Wire;
use std::collections::HashMap;

/// DAG Methods
impl QuditCircuit {
    /// Distill the circuit front nodes into a hashmap.
    ///
    /// # Returns
    ///
    /// A mapping from qudit or dit index to a circuit point of the first
    /// operation in the circuit on that qudit or clbit.
    ///
    /// # Performance
    ///
    /// This method is O(|width|) where width includes both the number of
    /// qudits and number of classical dits in the circuit.
    ///
    /// # Notes
    ///
    /// The same instruction id may be pointed to
    /// by two different keys in the hash map if it is at the front of
    /// the circuit at multiple spots. For example, if a cnot was at the
    /// front of the circuit, then it would be pointed to by both the
    /// control and target qudit indices.
    ///
    /// # Examples
    ///
    /// ```
    /// # use qudit_circuit::QuditCircuit;
    /// # use qudit_circuit::Wire;
    /// # use qudit_expr::library::{PGate, HGate};
    /// let mut circuit = QuditCircuit::pure([2, 2]);
    /// let p_id = circuit.append(PGate(2), 0, None);
    /// let h_id = circuit.append(HGate(2), 1, None);
    /// assert_eq!(circuit.front().len(), 2);
    /// assert_eq!(circuit.front()[&Wire::quantum(0)], p_id);
    /// assert_eq!(circuit.front()[&Wire::quantum(1)], h_id);
    /// ```
    pub fn front(&self) -> HashMap<Wire, InstructionId> {
        self.front
            .iter()
            .map(|(wire, front_cycle_id)| {
                let front_cycle = self
                    .cycles
                    .get_from_id(*front_cycle_id)
                    .expect("Expected cycle to exist.");
                let front_inst_id = front_cycle
                    .get_id_from_wire(*wire)
                    .expect("Expected there to be an instruction here?");
                (*wire, InstructionId::new(*front_cycle_id, front_inst_id))
            })
            .collect()
    }

    /// Distill the circuit rear nodes into a hashmap.
    ///
    /// See [`QuditCircuit::front`] for more information.
    pub fn rear(&self) -> HashMap<Wire, InstructionId> {
        self.rear
            .iter()
            .map(|(wire, rear_cycle_id)| {
                let rear_cycle = self
                    .cycles
                    .get_from_id(*rear_cycle_id)
                    .expect("Expected cycle to exist.");
                let rear_inst_id = rear_cycle
                    .get_id_from_wire(*wire)
                    .expect("Expected there to be an instruction here?");
                (*wire, InstructionId::new(*rear_cycle_id, rear_inst_id))
            })
            .collect()
    }

    /// Get the first instruction on a wire.
    ///
    /// # Returns
    ///
    /// An instruction id pointing to the first instruction on the specified wire
    /// if it exists. None, otherwise.
    ///
    /// # Performance
    ///
    /// This method is O(1)
    ///
    /// # Examples
    ///
    /// ```
    /// # use qudit_circuit::QuditCircuit;
    /// # use qudit_circuit::Wire;
    /// # use qudit_expr::library::PGate;
    /// let mut circuit = QuditCircuit::pure([2]);
    /// let p_id = circuit.append(PGate(2), 0, None);
    /// assert_eq!(circuit.first_on(Wire::quantum(0)), Some(p_id));
    /// ```
    pub fn first_on<W: Into<Wire>>(&self, wire: W) -> Option<InstructionId> {
        let wire = wire.into();
        self.front.get(&wire).map(|cycle_id| {
            InstructionId::new(
                *cycle_id,
                self.cycles[*cycle_id]
                    .get_id_from_wire(wire)
                    .expect("Expected instruction to exist."),
            )
        })
    }

    /// Get the last instruction on a wire.
    ///
    /// See [`QuditCircuit::first_on`] for more information.
    pub fn last_on<W: Into<Wire>>(&self, wire: W) -> Option<InstructionId> {
        let wire = wire.into();
        self.rear.get(&wire).map(|cycle_id| {
            InstructionId::new(
                *cycle_id,
                self.cycles[*cycle_id]
                    .get_id_from_wire(wire)
                    .expect("Expected instruction to exist."),
            )
        })
    }

    /// Gather the points of the next operations from the point of an operation.
    ///
    /// # Arguments
    ///
    /// * `point` - The point to get the next operations from. This needs refer
    ///   to a valid point in the circuit.
    ///
    /// # Returns
    ///
    /// A mapping from qudit or clbit index to the point of the next operation
    /// on that qudit or clbit.
    ///
    /// # Performance
    ///
    /// This method is O(|op-width|) where op-width includes both the number of
    /// qudits and number of classical dits in the operation referred to by
    /// `point`.
    ///
    /// # Panics
    ///
    /// If `point` is not a valid point in the circuit.
    ///
    /// # Notes
    ///
    /// The same operation may be pointed to by two different keys in the hash
    /// map if it is the next of the operation at multiple spots. For
    /// example, if a cnot is after the pointed operation, then it would be
    /// pointed to by both the control and target qudit indices in the returned
    /// map.
    ///
    /// # Examples
    ///
    /// ```
    /// # use qudit_circuit::QuditCircuit;
    /// # use qudit_circuit::Wire;
    /// # use qudit_expr::library::{PGate, HGate};
    /// let mut circuit = QuditCircuit::pure([2]);
    /// let first_inst = circuit.append(PGate(2), 0, None);
    /// let second_inst = circuit.append(HGate(2), 0, None);
    ///
    /// let next_insts = circuit.next(first_inst);
    /// assert_eq!(next_insts.len(), 1);
    /// assert_eq!(next_insts[&Wire::quantum(0)], second_inst);
    /// ```
    pub fn next(&self, inst_id: InstructionId) -> HashMap<Wire, InstructionId> {
        let cycle = self
            .cycles
            .get_from_id(inst_id.cycle())
            .expect("Invalid instruction id.");
        let wires = cycle.get_wires_from_id(inst_id.inner());

        match wires {
            Some(wires) => wires
                .wires()
                .filter_map(|wire| {
                    cycle.get_next(wire).map(|next_cycle_id| {
                        let next_cycle = self
                            .cycles
                            .get_from_id(next_cycle_id)
                            .expect("Expected cycle to exist.");
                        let next_inst_id = next_cycle
                            .get_id_from_wire(wire)
                            .expect("Expected there to be an instruction here?");
                        (wire, InstructionId::new(next_cycle_id, next_inst_id))
                    })
                })
                .collect(),
            None => HashMap::new(),
        }
    }

    /// Gather the points of the previous operations from the point of an
    /// operation.
    ///
    /// See [`QuditCircuit::next`] for more information.
    pub fn prev(&self, inst_id: InstructionId) -> HashMap<Wire, InstructionId> {
        let cycle = self
            .cycles
            .get_from_id(inst_id.cycle())
            .expect("Invalid instruction id.");
        let wires = cycle.get_wires_from_id(inst_id.inner());

        match wires {
            Some(wires) => wires
                .wires()
                .filter_map(|wire| {
                    cycle.get_prev(wire).map(|prev_cycle_id| {
                        let prev_cycle = self
                            .cycles
                            .get_from_id(prev_cycle_id)
                            .expect("Expected cycle to exist.");
                        let prev_inst_id = prev_cycle
                            .get_id_from_wire(wire)
                            .expect("Expected there to be an instruction here?");
                        (wire, InstructionId::new(prev_cycle_id, prev_inst_id))
                    })
                })
                .collect(),
            None => HashMap::new(),
        }
    }
}
