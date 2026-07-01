use super::*;
use crate::cycle::CycleIndex;
use crate::instruction::{Instruction, InstructionId};
use crate::operation::OpCode;
use crate::operation::Operation;
use crate::wire::Wire;
use crate::wire::WireList;
use qudit_core::{ClassicalSystem, QuditSystem};

impl QuditCircuit {
    /// Counts the amount of an operation in the circuit.
    pub fn count(&self, op_code: OpCode) -> usize {
        self.operations.count(op_code)
    }

    /// Checks if `wires` is a valid set of wires in the circuit.
    ///
    /// A wire list is valid if all qudit indices are less than the
    /// number of qudits in the circuit and all classical dit indices
    /// are less than the number of classical dits in the circuit.
    ///
    /// # Arguments
    ///
    /// * `wires` - The set of wires to check.
    ///
    /// # Returns
    ///
    /// `true` if `wires` is valid, `false` otherwise.
    ///
    /// # Performance
    ///
    /// This method is O(w) where
    ///     - `w` is the number of wires in the input list.
    ///
    /// # Examples
    ///
    /// ```
    /// # use qudit_circuit::QuditCircuit;
    /// # use qudit_circuit::WireList;
    /// let circuit = QuditCircuit::new([2, 2], [2, 2]);
    /// assert!(circuit.is_valid_wires(WireList::from([0, 1])));
    /// assert!(circuit.is_valid_wires(WireList::from(([0, 1], [0, 1]))));
    /// assert!(circuit.is_valid_wires(WireList::from((0, 0))));
    /// assert!(!circuit.is_valid_wires(WireList::from(([0, 1], [0, 2]))));
    /// assert!(!circuit.is_valid_wires(WireList::from([0, 1, 2])));
    /// assert!(!circuit.is_valid_wires(WireList::from(([0, 1], [2]))));
    /// ```
    pub fn is_valid_wires<W: AsRef<WireList>>(&self, wires: W) -> bool {
        let wires = wires.as_ref();
        wires.qudits().all(|q| q < self.num_qudits()) && wires.dits().all(|c| c < self.num_dits())
    }

    /// Check if an instruction identifier points to a valid instruction in the circuit.
    ///
    /// # Arguments
    ///
    /// * `inst_id` - The instruction id to check.
    ///
    /// # Returns
    ///
    /// `true` if `inst_id` is valid, `false` otherwise.
    ///
    /// # Performance
    ///
    /// This method is O(1).
    ///
    /// # Examples
    ///
    /// ```
    /// # use qudit_circuit::QuditCircuit;
    /// # use qudit_expr::library::PGate;
    /// let mut circuit = QuditCircuit::new([2, 2], [2, 2]);
    /// let p_id = circuit.append(PGate(2), 0, None).unwrap();
    /// assert!(circuit.is_valid_id(p_id));
    /// ```
    pub fn is_valid_id<P: Into<InstructionId>>(&self, inst_id: P) -> bool {
        let inst_id = inst_id.into();
        match self.cycles.get_from_id(inst_id.cycle()) {
            None => false,
            Some(cycle) => cycle.is_valid_id(inst_id.inner()),
        }
    }

    /// Finds the first available cycle for qudits in `location`.
    ///
    /// An available cycle for `location` is one where it and all
    /// cycles after it are unoccupied for `location`.
    ///
    /// # Arguments
    ///
    /// * `location` - The location to check for cycle availability.
    ///
    /// # Returns
    ///
    /// The index of the first available cycle for `location` or `None`
    ///
    /// # Performance
    ///
    /// This method is O(|location|).
    ///
    /// # Panics
    ///
    /// If `location` is not a valid location in the circuit.
    ///
    /// # Examples
    ///
    /// ```
    /// # use qudit_circuit::QuditCircuit;
    /// # use qudit_circuit::WireList;
    /// # use qudit_expr::library::PGate;
    /// let mut circuit = QuditCircuit::new([2, 2], [2, 2]);
    /// circuit.append(PGate(2), 0, None);
    /// assert!(circuit.find_available_cycle(WireList::from([0])).is_none());
    /// assert!(circuit.find_available_cycle(WireList::from([1])).is_some_and(|c| c == 0));
    /// ```
    pub fn find_available_cycle<W: AsRef<WireList>>(&self, wires: W) -> Option<CycleIndex> {
        if !self.is_valid_wires(&wires) {
            // TODO: Really, panic over this? This it totally a recoverable error.
            // Cmon, you know better now. Get some proper error handling going already.
            panic!("Cannot find available cycle for invalid location.");
        }

        if self.cycles.is_empty() {
            return None;
        }

        let last_occupied_cycle_option = wires
            .as_ref()
            .wires()
            .filter_map(|w| self.rear.get(&w))
            .map(|cycle_id| {
                self.cycles
                    .id_to_index(*cycle_id)
                    .expect("Expected cycle to exist.")
            })
            .max_by(Ord::cmp);

        match last_occupied_cycle_option {
            Some(cycle_index) => {
                if cycle_index + 1u64 < CycleIndex(self.num_cycles() as u64) {
                    Some(cycle_index + 1u64)
                } else {
                    None
                }
            }
            None => {
                // Circuit is not empty due to above check,
                // but no gates on location
                Some(CycleIndex(0))
            }
        }
    }

    /// Find first or create new available cycle and return its index
    pub(super) fn find_available_or_append_cycle<W: AsRef<WireList>>(
        &mut self,
        wires: W,
    ) -> CycleIndex {
        // Location validity implicitly checked in find_available_cycle
        if let Some(cycle_index) = self.find_available_cycle(wires) {
            cycle_index
        } else {
            self.cycles.push()
        }
    }

    /// Intern an operation in the circuit's operation cache
    ///
    /// This allows further additions by OpCodes.
    pub fn cache_operation<O: Into<Operation>>(&mut self, op: O) -> OpCode {
        self.operations.insert(op.into()).expect("TODO")
    }

    //     fn _insert_ref(
    //         &mut self,
    //         cycle_index: CycleIndex,
    //         op_code: OpCode,
    //         wires: WireList,
    //         params: ParamIndices,
    //     ) -> InstructionId {
    //         // TODO: Check cycle_index is valid

    //         // Two options: cycle_index is available at location
    //         // or not
    //         //
    //         // if available at location, then insert there, but then we need to update
    //         // prev and next of the prev and next without knowing exactly where they are at: costly
    //         //
    //         // if not available at location, need to insert a new cycle, but now I know who
    //         // prev and next are
    //         for wire in wires.wires() {

    //         }

    //         todo!()
    //     }

    /// Checks if a qudit is inactive
    pub fn is_qudit_inactive(&self, index: usize) -> bool {
        !self.front.contains_key(&Wire::quantum(index))
    }

    /// Remove the operation at `point` from the circuit
    pub fn remove(&mut self, inst_id: InstructionId) -> Option<Instruction> {
        if !self.is_valid_id(inst_id) {
            // TODO: log warning?
            return None;
        }

        let wires = self
            .cycles
            .get_from_id(inst_id.cycle())
            .expect("Expected valid cycle.")
            .get_wires_from_id(inst_id.inner())
            .expect("Expected valid instruction.");

        // Update circuit quantum DAG info
        for wire in &wires {
            let cycle = self
                .cycles
                .get_from_id(inst_id.cycle())
                .expect("Expected valid cycle.");
            let next = cycle.get_next(wire);
            let prev = cycle.get_prev(wire);

            match (next, prev) {
                (Some(next_cycle_id), Some(prev_cycle_id)) => {
                    self.cycles
                        .get_mut_from_id(next_cycle_id)
                        .expect("Expected valid cycle.")
                        .set_prev(wire, prev_cycle_id);
                    self.cycles
                        .get_mut_from_id(prev_cycle_id)
                        .expect("Expected valid cycle.")
                        .set_next(wire, next_cycle_id);
                }
                (Some(next_cycle_id), None) => {
                    self.cycles
                        .get_mut_from_id(next_cycle_id)
                        .expect("Expected valid cycle.")
                        .reset_prev(wire);
                    debug_assert!(*self.front.get(&wire).unwrap() == inst_id.cycle());
                    self.front.insert(wire, next_cycle_id);
                }
                (None, Some(prev_cycle_id)) => {
                    self.cycles
                        .get_mut_from_id(prev_cycle_id)
                        .expect("Expected valid cycle.")
                        .reset_next(wire);
                    debug_assert!(*self.rear.get(&wire).unwrap() == inst_id.cycle());
                    self.rear.insert(wire, prev_cycle_id);
                }
                (None, None) => {
                    debug_assert!(*self.front.get(&wire).unwrap() == inst_id.cycle());
                    debug_assert!(*self.rear.get(&wire).unwrap() == inst_id.cycle());
                    self.front.remove(&wire);
                    self.rear.remove(&wire);
                }
            }
        }

        let cycle = self
            .cycles
            .get_mut_from_id(inst_id.cycle())
            .expect("Expected valid cycle.");
        let inst = cycle
            .remove(
                wires
                    .wires()
                    .next()
                    .expect("Corrupted instruction acting on no wires."),
            )
            .expect("Expected instruction to remove.");

        if cycle.num_ops() == 0 {
            // Empty cycles cannot exist; must be removed
            self.cycles.remove_id(inst_id.cycle());
        }

        for param in &inst.params() {
            self.params.decrement(param);
        }

        self.operations.decrement(inst.op_code());

        Some(inst)
    }
}
