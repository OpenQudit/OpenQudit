use std::collections::HashMap;
use qudit_core::{ComplexScalar, HasParams, QuditSystem};

use indexmap::IndexSet;
use qudit_core::{QuditRadices, RealScalar, c64};
use qudit_gates::Gate;
use qudit_expr::UnitaryExpressionGenerator;
// use qudit_tensor::{BuilderExpressionInput, ExpressionTree, TreeBuilder, TreeOptimizer};
use qudit_tensor::QuditCircuitNetwork;

use crate::{compact::CompactIntegerVector, cpoint, cyclelist::CycleList, instruction::{Instruction, InstructionReference}, location::CircuitLocation, operation::{Operation, OperationReference}, qpoint, CircuitPoint, DitOrBit};

/// A quantum circuit that can be defined with qudits and classical bits.
#[derive(Clone)]
pub struct QuditCircuit<C: ComplexScalar = c64> {
    /// The number of qudits in the circuit.
    num_qudits: usize,

    /// The number of classical bits in the circuit.
    num_clbits: usize,

    /// The QuditRadices object that describes the dimension of the circuit.
    radices: QuditRadices,

    /// All instructions in the circuit stored in cycles.
    cycles: CycleList,

    /// A map that stores information on each type of operation in the circuit.
    /// Currently, counts for each type of operation is stored.
    op_info: HashMap<OperationReference, usize>,

    /// A map that stores information on the connections between qudits in the circuit.
    /// Currently, gate counts on each pair of qudits are stored.
    graph_info: HashMap<(usize, usize), usize>,

    /// A pointer to the first operation on each qudit. These are stored as
    /// physical cycle indices.
    qfront: Vec<Option<usize>>,

    /// A pointer to the last operation on each qudit. These are stored as
    /// physical cycle indices.
    qrear: Vec<Option<usize>>,

    /// A pointer to the first operation on each classical bit. These are stored as
    /// physical cycle indices.
    cfront: Vec<Option<usize>>,

    /// A pointer to the last operation on each classical bit. These are stored as
    /// physical cycle indices.
    crear: Vec<Option<usize>>,

    /// The set of gates in the circuit.
    pub gates: IndexSet<Gate>,

    /// The set of subcircuits in the circuit.
    // pub subcircuits: IndexSet<ExpressionTree>,

    /// The stored parameters of the circuit.
    params: Vec<C::R>,
}

impl<C: ComplexScalar> QuditCircuit<C> {

    /// Creates a new QuditCircuit object.
    ///
    /// # Arguments
    ///
    /// * `radices` - The QuditRadices object that describes the qudit system.
    ///
    /// * `num_clbits` - The number of classical bits in the circuit.
    ///
    /// # Examples
    ///
    /// We can define purely quantum kernels without classical bits:
    /// ```
    /// use qudit_circuit::QuditCircuit;
    /// use qudit_core::{radices, QuditRadices};
    ///
    /// let two_qubit_circuit = QuditCircuit::new(radices![2, 2], 0);
    /// let two_qutrit_circuit = QuditCircuit::new(radices![3, 3], 0);
    /// let hybrid_circuit = QuditCircuit::new(radices![2, 2, 3, 3], 0);
    /// ```
    ///
    /// We can also define hybrid quantum-classical circuits:
    /// ```
    /// use qudit_circuit::QuditCircuit;
    /// use qudit_core::{radices, QuditRadices};
    /// let two_qubit_circuit = QuditCircuit::new(radices![2, 2], 2);
    /// let two_qutrit_circuit = QuditCircuit::new(radices![3, 3], 4);
    /// ```
    ///
    /// Note in the `two_qutrit_circuit` example, we have four classical bits
    /// even though we only have two qudits. This is because each qudit has
    /// three possible values, so we need two bits to store/measure each qudit.
    pub fn new(radices: QuditRadices, num_clbits: usize) -> QuditCircuit<C> {
        QuditCircuit::with_capacity(radices, num_clbits, 1)
    }

    /// Creates a new QuditCircuit object with a given cycle capacity.
    ///
    /// # Arguments
    ///
    /// * `radices` - The QuditRadices object that describes the qudit system.
    ///
    /// * `num_clbits` - The number of classical bits in the circuit.
    ///
    /// * `capacity` - The number of cycles to pre-allocate.
    ///
    /// # Examples
    ///
    /// ```
    /// use qudit_circuit::QuditCircuit;
    /// use qudit_core::{radices, QuditRadices};
    /// let two_qubit_circuit = QuditCircuit::with_capacity(radices![2, 2], 2, 10);
    /// ```
    pub fn with_capacity(
        radices: QuditRadices,
        num_clbits: usize,
        capacity: usize,
    ) -> QuditCircuit<C> {
        QuditCircuit {
            num_qudits: radices.num_qudits(),
            num_clbits: num_clbits,
            cycles: CycleList::with_capacity(capacity),
            op_info: HashMap::new(),
            graph_info: HashMap::new(),
            qfront: vec![None; radices.num_qudits()],
            qrear: vec![None; radices.num_qudits()],
            cfront: vec![None; num_clbits],
            crear: vec![None; num_clbits],
            radices: radices,
            gates: IndexSet::new(),
            // subcircuits: IndexSet::new(),
            params: Vec::new(),
        }
    }

    /// Returns the number of cycles in the circuit.
    ///
    /// # Performance
    ///
    /// This method is O(1).
    pub fn num_cycles(&self) -> usize {
        self.cycles.len()
    }

    /// Returns the number of parameters in the circuit.
    ///
    /// # Performance
    ///
    /// This method is 0(1).
    pub fn num_params(&self) -> usize {
        self.params.len()
    }

    /// Returns the number of operations in the circuit.
    ///
    /// # Performance
    ///
    /// This method is O(|t|) where `t` is the number of distinct instruction
    /// types in the circuit.
    pub fn num_operations(&self) -> usize {
        self.op_info.iter().map(|(_, count)| count).sum()
    }

    /// A reference to the parameters of the circuit.
    pub fn params(&self) -> &Vec<C::R> {
        &self.params
    }

    /// Increment internal instruction type counter.
    fn inc_op_counter(&mut self, op_type: &OperationReference) {
        match self.op_info.get_mut(op_type) {
            Some(count) => *count += 1,
            None => {
                self.op_info.insert(*op_type, 1);
            },
        }
    }

    /// Increment internal graph counter.
    fn inc_graph_counter(&mut self, location: &CircuitLocation) {
        for pair in location.get_qudit_pairs() {
            match self.graph_info.get_mut(&pair) {
                Some(count) => *count += 1,
                None => {
                    self.graph_info.insert(pair, 1);
                },
            }
        }
    }

    /// Decrement internal instruction type counter.
    fn dec_inst_counter(&mut self, op_type: &OperationReference) -> bool {
        if !self.op_info.contains_key(op_type) {
            panic!(
                "Cannot decrement instruction counter for instruction type that does not exist."
            );
        }

        let count = self.op_info.get_mut(op_type).unwrap();
        *count -= 1;

        if *count == 0 {
            self.op_info.remove(op_type);
            true
        }
        else {
            false
        }
    }

    /// Decrement internal graph counter.
    fn dec_graph_counter(&mut self, location: &CircuitLocation) {
        for pair in location.get_qudit_pairs() {
            if !self.graph_info.contains_key(&pair) {
                panic!("Cannot decrement graph counter for qudit pair that does not exist.")
            }

            let count = self.graph_info.get_mut(&pair).unwrap();
            *count -= 1;

            if *count == 0 {
                self.graph_info.remove(&pair);
            }
        }
    }

    /// Checks if `location` is a valid location in the circuit.
    ///
    /// A location is valid if all qudit indices are less than the
    /// number of qudits in the circuit and all classical bit indices
    /// are less than the number of classical bits in the circuit.
    ///
    /// # Arguments
    ///
    /// * `location` - The location to check.
    ///
    /// # Returns
    ///
    /// `true` if `location` is valid, `false` otherwise.
    ///
    /// # Performance
    ///
    /// This method is O(|location|).
    ///
    /// # Examples
    ///
    /// ```
    /// use qudit_circuit::QuditCircuit;
    /// use qudit_circuit::{radices, loc, QuditRadices};
    /// use qudit_circuit::circuit::location::CircuitLocation;
    /// let circuit = QuditCircuit::new(radices![2, 2], 2);
    /// assert!(circuit.is_valid_location(&loc![0, 1]));
    /// assert!(circuit.is_valid_location(&loc![0, 1; 0, 1]));
    /// assert!(circuit.is_valid_location(&loc![0; 0]));
    /// assert!(!circuit.is_valid_location(&loc![0, 1; 0, 2]));
    /// assert!(!circuit.is_valid_location(&loc![0, 1, 2]));
    /// assert!(!circuit.is_valid_location(&loc![0, 1; 2]));
    /// ```
    pub fn is_valid_location(&self, location: &CircuitLocation) -> bool {
        location.qudits().iter().all(|q| q < self.num_qudits)
            && location.clbits().iter().all(|c| c < self.num_clbits)
    }

    /// Checks if `point` is a valid point in the circuit.
    ///
    /// A point is valid if its cycle index is in bounds and its
    /// qudit or classical bit index is in bounds.
    ///
    /// # Arguments
    ///
    /// * `point` - The point to check.
    ///
    /// # Returns
    ///
    /// `true` if `point` is valid, `false` otherwise.
    ///
    /// # Performance
    ///
    /// This method is O(1).
    ///
    /// # Examples
    ///
    /// ```
    /// use qudit_circuit::{QuditCircuit, Gate, QuditRadices};
    /// use qudit_circuit::{radices, loc, point};
    /// use qudit_circuit::circuit::location::CircuitLocation;
    /// use qudit_circuit::circuit::point::{CircuitPoint, DitOrBit};
    /// let mut circuit = QuditCircuit::new(radices![2, 2], 2);
    /// circuit.append_gate(Gate::Z(2), loc![0], vec![]);
    /// assert!(circuit.is_valid_point(&point!(0, 0)));
    /// assert!(!circuit.is_valid_point(&point!(1, 0)));
    /// assert!(circuit.is_valid_point(&point!(0; 1)));
    /// assert!(!circuit.is_valid_point(&point!(0; 2)));
    /// ```
    pub fn is_valid_point(&self, point: &CircuitPoint) -> bool {
        point.cycle < self.cycles.len()
            && match point.dit_or_bit {
                DitOrBit::Qudit(q) => q < self.num_qudits,
                DitOrBit::Clbit(c) => c < self.num_clbits,
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
    /// use qudit_circuit::{QuditCircuit, Gate};
    /// use qudit_circuit::{radices, loc, QuditRadices};
    /// use qudit_circuit::circuit::location::CircuitLocation;
    /// let mut circuit = QuditCircuit::new(radices![2, 2], 2);
    /// circuit.append_gate(Gate::Z(2), loc![0], vec![]);
    /// assert!(circuit.find_available_cycle(&loc![0]).is_none());
    /// assert_eq!(circuit.find_available_cycle(&loc![1]), Some(0));
    /// ```
    pub fn find_available_cycle(
        &self,
        location: &CircuitLocation,
    ) -> Option<usize> {
        if !self.is_valid_location(location) {
            panic!("Cannot find available cycle for invalid location.");
        }

        if self.cycles.is_empty() {
            return None;
        }

        let last_occupied_cycle = location
            .qudits()
            .iter()
            .map(|q| match self.qrear[q] {
                Some(cycle_index) => {
                    Some(self.cycles.map_physical_to_logical_idx(cycle_index))
                },
                None => None,
            })
            .chain(location.clbits().iter().map(|c| match self.crear[c] {
                Some(cycle_index) => {
                    Some(self.cycles.map_physical_to_logical_idx(cycle_index))
                },
                None => None,
            }))
            .max()
            .unwrap();

        if let None = last_occupied_cycle {
            return Some(0);
        }
        let last_occupied_cycle = last_occupied_cycle.unwrap();

        if last_occupied_cycle + 1 < self.cycles.len() {
            Some(last_occupied_cycle + 1)
        } else {
            None
        }
    }

    /// Find the first available cycle's physical id, if none exist, append a
    /// new cycle.
    fn find_available_or_append_cycle(
        &mut self,
        location: &CircuitLocation,
    ) -> usize {
        // Location validity implicitly checked in find_available_cycle
        if let Some(cycle_index) = self.find_available_cycle(location) {
            self.cycles.map_logical_to_physical_idx(cycle_index)
        } else {
            self.cycles.push()
        }
    }

    /// Append an instruction to the end of the circuit.
    ///
    /// # Arguments
    ///
    /// * `inst` - The instruction to append.
    pub fn append_instruction(&mut self, inst: Instruction<C::R>) {
        // check valid operation for radix match, measurement bandwidth etc
        let Instruction { op, location, params } = inst;

        // Find cycle placement (location validity implicitly checked here)
        let cycle_index = self.find_available_or_append_cycle(&location);

        // Build operation reference
        let op_ref = OperationReference::new(self, op);

        // Update counters
        self.inc_op_counter(&op_ref);
        self.inc_graph_counter(&location);

        // Update qudit DAG info
        for qudit_index in location.qudits() {
            if let Some(rear_cycle_index) = self.qrear[qudit_index] {
                self.cycles[rear_cycle_index].set_qnext(
                    qudit_index,
                    cycle_index,
                );
                self.cycles[cycle_index].set_qprev(
                    qudit_index,
                    rear_cycle_index,
                );
            } else {
                // If qrear is none then no op exists on this qudit from before,
                // so we update qfront too.
                self.qfront[qudit_index] = Some(cycle_index);
            }
            // Update qrear
            self.qrear[qudit_index] = Some(cycle_index);
        }

        // Update clbit DAG info
        for clbit_index in location.clbits() {
            if let Some(rear_cycle_index) = self.crear[clbit_index] {
                self.cycles[rear_cycle_index].set_cnext(
                    clbit_index,
                    cycle_index,
                );
                self.cycles[cycle_index].set_cprev(
                    clbit_index,
                    rear_cycle_index,
                );
            } else {
                // If crear is none then no op exists on this clbit from before,
                // so we update cfront too.
                self.cfront[clbit_index] = Some(cycle_index);
            }
            // Update crear
            self.crear[clbit_index] = Some(cycle_index);
        }

        // update params
        let param_indices = CompactIntegerVector::from_range((self.params.len() - params.len())..self.params.len());
        self.params.extend(params);

        // Build instruction reference
        let inst_ref = InstructionReference::new(op_ref, location, param_indices);

        // Add op to cycle
        self.cycles[cycle_index].push(inst_ref);
    }

    /// Shorthand for appending an instruction
    pub fn append_gate(
        &mut self,
        gate: Gate,
        location: CircuitLocation,
        params: Vec<C::R>,
    ) {
        self.append_instruction(Instruction::new(Operation::Gate(gate), location, params));
    }

    /// Remove the operation at `point` from the circuit.
    pub fn remove(&mut self, point: CircuitPoint) {
        if !self.is_valid_point(&point) {
            panic!("Cannot remove operation at invalid point.");
        }

        let cycle_index = self.cycles.map_logical_to_physical_idx(point.cycle);
        let inst = match self.cycles[cycle_index].remove(point.dit_or_bit) {
            Some(inst) => inst,
            None => panic!("Operation not found at {} in cycle {}", point.dit_or_bit, point.cycle),
        };
        let location = &inst.location;

        // if cycle is empty; remove it
        if self.cycles[cycle_index].is_empty() {
            self.cycles.remove(cycle_index);
        }

        // Update circuit qudit DAG info
        for qudit_index in location.qudits() {
            let qnext = self.cycles[cycle_index].get_qnext(qudit_index);
            let qprev = self.cycles[cycle_index].get_qprev(qudit_index);

            match (qnext, qprev) {
                (Some(next_cycle_index), Some(prev_cycle_index)) => {
                    self.cycles[next_cycle_index].set_qprev(qudit_index, prev_cycle_index);
                    self.cycles[prev_cycle_index].set_qnext(qudit_index, next_cycle_index);
                },
                (Some(next_cycle_index), None) => {
                    self.cycles[next_cycle_index].reset_qprev(qudit_index);
                    self.qfront[qudit_index] = Some(next_cycle_index);
                },
                (None, Some(prev_cycle_index)) => {
                    self.qrear[qudit_index] = Some(prev_cycle_index);
                    self.cycles[prev_cycle_index].reset_qnext(qudit_index);
                },
                (None, None) => {
                    self.qrear[qudit_index] = None;
                    self.qfront[qudit_index] = None;
                },
            }
        }

        // Update circuit qudit DAG info
        for clbit_index in location.clbits() {
            let cnext = self.cycles[cycle_index].get_cnext(clbit_index);
            let cprev = self.cycles[cycle_index].get_cprev(clbit_index);

            match (cnext, cprev) {
                (Some(next_cycle_index), Some(prev_cycle_index)) => {
                    self.cycles[next_cycle_index].set_cprev(clbit_index, prev_cycle_index);
                    self.cycles[prev_cycle_index].set_cnext(clbit_index, next_cycle_index);
                },
                (Some(next_cycle_index), None) => {
                    self.cycles[next_cycle_index].reset_cprev(clbit_index);
                    self.cfront[clbit_index] = Some(next_cycle_index);
                },
                (None, Some(prev_cycle_index)) => {
                    self.crear[clbit_index] = Some(prev_cycle_index);
                    self.cycles[prev_cycle_index].reset_cnext(clbit_index);
                },
                (None, None) => {
                    self.crear[clbit_index] = None;
                    self.cfront[clbit_index] = None;
                },
            }
        }

        // Update counters
        self.dec_graph_counter(location);
        self.dec_inst_counter(&inst.op);

        // Explicitly do not remove from index sets, as this would require updating
        // all operation references in the circuit, since the indices shift.
        // TODO: look into dummy values and memory overhead? With gates probably not
        // much, but subcircuits could be a problem.
    }

    // /////////////////////////////////////////////////////////////////
    // DAG Methods
    // /////////////////////////////////////////////////////////////////

    /// Distill the circuit front nodes into a hashmap.
    ///
    /// # Returns
    ///
    /// A mapping from qudit or clbit index to a circuit point of the first
    /// operation in the circuit on that qudit or clbit.
    ///
    /// # Performance
    ///
    /// This method is O(|width|) where width includes both the number of
    /// qudits and number of classical bits in the circuit.
    ///
    /// # Notes
    ///
    /// The same operation may be pointed to by two different keys in the hash
    /// map if it is at the front of the circuit at multiple spots. For
    /// example, if a cnot was at the front of the circuit, then it would be
    /// pointed to by both the control and target qudit indices.
    ///
    /// # Examples
    ///
    /// ```
    /// use qudit_circuit::{QuditCircuit, Gate, QuditRadices};
    /// use qudit_circuit::{radices, loc, point};
    /// use qudit_circuit::circuit::location::CircuitLocation;
    /// use qudit_circuit::circuit::point::{CircuitPoint, DitOrBit};
    /// let mut circuit = QuditCircuit::new(radices![2, 2], 2);
    /// circuit.append_gate(Gate::Z(2), loc![0], vec![]);
    /// circuit.append_gate(Gate::H(2), loc![1], vec![]);
    /// assert_eq!(circuit.front().len(), 2);
    /// assert_eq!(circuit.front()[&DitOrBit::Qudit(0)], point!(0, 0));
    /// assert_eq!(circuit.front()[&DitOrBit::Qudit(1)], point!(0, 1));
    /// ```
    pub fn front(&self) -> HashMap<DitOrBit, CircuitPoint> {
        self.qfront
            .iter()
            .enumerate()
            .filter(|x| x.1.is_some())
            .map(|x| (DitOrBit::Qudit(x.0), qpoint!(self.cycles[x.1.unwrap()].logical_index, x.0)))
            .chain(
                self.cfront
                    .iter()
                    .enumerate()
                    .filter(|x| x.1.is_some())
                    .map(|x| (DitOrBit::Clbit(x.0), cpoint!(self.cycles[x.1.unwrap()].logical_index, x.0))),
            )
            .collect()
    }

    /// Distill the circuit rear nodes into a hashmap.
    ///
    /// See [`QuditCircuit::front`] for more information.
    pub fn rear(&self) -> HashMap<DitOrBit, CircuitPoint> {
        self.qrear
            .iter()
            .enumerate()
            .filter(|x| x.1.is_some())
            .map(|x| (DitOrBit::Qudit(x.0), qpoint!(self.cycles[x.1.unwrap()].logical_index, x.0)))
            .chain(
                self.crear
                    .iter()
                    .enumerate()
                    .filter(|x| x.1.is_some())
                    .map(|x| (DitOrBit::Clbit(x.0), cpoint!(self.cycles[x.1.unwrap()].logical_index, x.0))),
            )
            .collect()
    }

    /// Gather the points of the next operations from the point of an operation.
    ///
    /// # Arguments
    ///
    /// * `point` - The point to get the next operations from. This needs refer
    /// to a valid point in the circuit.
    ///
    /// # Returns
    ///
    /// A mapping from qudit or clbit index to the point of the next operation
    /// on that qudit or clbit.
    ///
    /// # Performance
    ///
    /// This method is O(|op-width|) where op-width includes both the number of
    /// qudits and number of classical bits in the operation referred to by
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
    /// use qudit_circuit::{QuditCircuit, Gate};
    /// use qudit_circuit::{radices, loc, point, QuditRadices};
    /// use qudit_circuit::circuit::location::CircuitLocation;
    /// use qudit_circuit::circuit::point::{CircuitPoint, DitOrBit};
    /// let mut circuit = QuditCircuit::new(radices![2, 2], 2);
    /// circuit.append_gate(Gate::Z(2), loc![0], vec![]);
    /// circuit.append_gate(Gate::H(2), loc![1], vec![]);
    /// circuit.append_gate(Gate::Z(2), loc![0], vec![]);
    /// assert_eq!(circuit.next(point!(0, 0)).len(), 1);
    /// assert_eq!(circuit.next(point!(0, 0))[&DitOrBit::Qudit(0)], point!(1, 0));
    /// ```
    pub fn next(&self, point: CircuitPoint) -> HashMap<DitOrBit, CircuitPoint> {
        let physical_cycle_index = self.cycles.map_logical_to_physical_idx(point.cycle);
        let location = self.cycles[physical_cycle_index].get_location(point.dit_or_bit);

        match location {
            Some(location) => {
                location.qudits()
                    .iter()
                    .map(|qudit_index|
                        match self.cycles[physical_cycle_index].get_qnext(qudit_index) {
                            Some(next_cycle_index) => {
                                Some((DitOrBit::Qudit(qudit_index), qpoint!(qudit_index, self.cycles[next_cycle_index].logical_index)))
                            },
                            None => {
                                None
                            }
                        }
                    )
                    .chain(
                        location.clbits()
                            .iter()
                            .map(|clbit_index|
                                match self.cycles[physical_cycle_index].get_cnext(clbit_index) {
                                    Some(next_cycle_index) => {
                                        Some((DitOrBit::Clbit(clbit_index), cpoint!(clbit_index, self.cycles[next_cycle_index].logical_index)))
                                    },
                                    None => {
                                        None
                                    }
                                }
                            )
                    )
                    .filter(|x| x.is_some())
                    .map(|x| x.unwrap())
                    .collect()

            },
            None => HashMap::new(),
        }
    }

    /// Gather the points of the previous operations from the point of an
    /// operation.
    ///
    /// See [`QuditCircuit::next`] for more information.
    pub fn prev(&self, point: CircuitPoint) -> HashMap<DitOrBit, CircuitPoint> {
        let physical_cycle_index = self.cycles.map_logical_to_physical_idx(point.cycle);
        let location = self.cycles[physical_cycle_index].get_location(point.dit_or_bit);

        match location {
            Some(location) => {
                location.qudits()
                    .iter()
                    .map(|qudit_index|
                        match self.cycles[physical_cycle_index].get_qprev(qudit_index) {
                            Some(prev_cycle_index) => {
                                Some((DitOrBit::Qudit(qudit_index), qpoint!(qudit_index, self.cycles[prev_cycle_index].logical_index)))
                            },
                            None => {
                                None
                            }
                        }
                    )
                    .chain(
                        location.clbits()
                            .iter()
                            .map(|clbit_index|
                                match self.cycles[physical_cycle_index].get_cprev(clbit_index) {
                                    Some(prev_cycle_index) => {
                                        Some((DitOrBit::Clbit(clbit_index), cpoint!(clbit_index, self.cycles[prev_cycle_index].logical_index)))
                                    },
                                    None => {
                                        None
                                    }
                                }
                            )
                    )
                    .filter(|x| x.is_some())
                    .map(|x| x.unwrap())
                    .collect()

            },
            None => HashMap::new(),
        }
    }


    // Convert the circuit to an expression tree.
    // pub fn to_tree(&self) -> ExpressionTree {
    //     let mut point_to_index_map = HashMap::new();
    //     let mut op_index_count = 0;
    //     for (cycle_index, cycle) in self.cycles.iter().enumerate() {
    //         for inst in cycle {
    //             for qudit_index in inst.location.qudits() {
    //                 let point = qpoint![cycle_index, qudit_index];
    //                 point_to_index_map.insert(point, op_index_count);
    //             }
    //             op_index_count += 1;
    //         }
    //     }

    //     let mut expressions = Vec::new();
    //     let mut qudits_list = Vec::new();
    //     let mut next_list = Vec::new();
    //     let mut prev_list = Vec::new();

    //     for cycle in &self.cycles {
    //         for inst in cycle {
    //             let op = inst.op.dereference(self);
    //             match op {
    //                 Operation::Gate(gate) => {
    //                     expressions.push(BuilderExpressionInput::Unitary(gate.gen_expr()));
    //                 },
    //                 Operation::Subcircuit(subcircuit) => {
    //                     expressions.push(BuilderExpressionInput::Tree(subcircuit));
    //                 },
    //                 Operation::Control(_) => {
    //                     panic!("Control operations are not supported in expression trees currently.");
    //                 },
    //             }

    //             let mut qudits = Vec::new();
    //             for qudit_index in inst.location.qudits() {
    //                 qudits.push(qudit_index);
    //             }
    //             qudits_list.push(qudits);

    //             let mut op_nexts = Vec::new();
    //             let mut op_prevs = Vec::new();

    //             for qudit_index in inst.location.qudits() {
    //                 let physical_cycle_index = cycle.get_qnext(qudit_index);
    //                 match physical_cycle_index {
    //                     Some(next_cycle_index) => {
    //                         let next_point = qpoint![next_cycle_index, qudit_index];
    //                         op_nexts.push(Some(point_to_index_map[&next_point]));
    //                     },
    //                     None => op_nexts.push(None),
    //                 }

    //                 let physical_cycle_index = cycle.get_qprev(qudit_index);
    //                 match physical_cycle_index {
    //                     Some(prev_cycle_index) => {
    //                         let prev_point = qpoint![prev_cycle_index, qudit_index];
    //                         op_prevs.push(Some(point_to_index_map[&prev_point]));
    //                     },
    //                     None => op_prevs.push(None),
    //                 }
    //             }

    //             next_list.push(op_nexts);
    //             prev_list.push(op_prevs);
    //         }
    //     }

    //     let tree = TreeBuilder::new(
    //         self.num_qudits(),
    //         expressions,
    //         qudits_list,
    //         next_list,
    //         prev_list,
    //     ).build_tree();
    //     TreeOptimizer::new().optimize(tree)
    // }
}

impl<C: ComplexScalar> QuditSystem for QuditCircuit<C> {
    fn num_qudits(&self) -> usize {
        self.num_qudits
    }

    fn dimension(&self) -> usize {
        self.radices.dimension()
    }

    fn radices(&self) -> QuditRadices {
        self.radices.clone()
    }
}

impl<C: ComplexScalar> HasParams for QuditCircuit<C> {
    fn num_params(&self) -> usize {
        self.params.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use faer::reborrow::ReborrowMut;
    use qudit_core::c32;
    use qudit_core::c64;
    use qudit_core::radices;
    use qudit_core::QuditRadices;
    use qudit_expr::DifferentiationLevel;
    use qudit_tree::QVM;
    use crate::loc;
    use crate::CircuitLocation;
    use qudit_tree::compile;
    use qudit_core::unitary::UnitaryMatrix;
    use qudit_core::unitary::UnitaryFn;

    pub fn build_qsearch_thin_step_circuit(n: usize) -> QuditCircuit {
        let mut circ = QuditCircuit::new(radices![2; n], 0);
        for i in 0..n {
            circ.append_gate(Gate::U3(), loc![i], vec![]);
        }
        for _ in 0..n {
            for i in 0..(n - 1) {
                circ.append_gate(Gate::CP(), loc![i, i + 1], vec![]);
                circ.append_gate(Gate::U3(), loc![i], vec![]);
                circ.append_gate(Gate::U3(), loc![i + 1], vec![]);
            }
        }
        circ
    }

    #[test]
    fn test_u3_mul_circuit() {
        let mut circ: QuditCircuit<f64> = QuditCircuit::new(radices![2; 1], 0);
        circ.append_gate(Gate::U3(), loc![0], vec![1.7, 1.7, 1.7]);
        circ.append_gate(Gate::U3(), loc![0], vec![1.7, 1.7, 1.7]);
        let tree = circ.to_tree();
        println!("{:?}", tree);
        let code = compile(&tree);
        println!("{:?}", code);
        let mut qvm: QVM<c64> = QVM::new(code, DifferentiationLevel::None);
        let params = vec![1.7; 6];
        let utry = qvm.get_unitary(&params);

        let u3utry: UnitaryMatrix<c64> = Gate::U3().gen_expr().get_unitary(&[1.7, 1.7, 1.7]);
        let expected = u3utry.dot(&u3utry);
        let dist = UnitaryMatrix::new([2], utry.to_owned()).get_distance_from(expected);
        println!("{:?}", dist);
        assert!(dist < 1e-7);
    }

    // #[test]
    // fn test_u3_kron_circuit() {
    //     let mut circ: QuditCircuit<f64> = QuditCircuit::new(radices![2; 2], 0);
    //     circ.append_gate(Gate::U3(), loc![0], vec![1.7, 1.7, 1.7]);
    //     circ.append_gate(Gate::U3(), loc![1], vec![1.7, 1.7, 1.7]);
    //     let tree = circ.to_tree();
    //     println!("{:?}", tree);
    //     let code = compile(&tree);
    //     println!("{:?}", code);
    //     let mut qvm: QVM<c64> = QVM::new(code, DifferentiationLevel::None);
    //     let params = vec![1.7; 6];
    //     let utry = qvm.get_unitary(&params);

    //     let u3utry: UnitaryMatrix<c64> = Gate::U3().gen_expr().get_unitary(&[1.7, 1.7, 1.7]);
    //     let expected = u3utry.kron(&u3utry);
    //     let dist = UnitaryMatrix::new([2, 2], utry.to_owned()).get_distance_from(expected);
    //     println!("{:?}", dist);
    //     assert!(dist < 1e-7);
    // }

    #[test]
    fn test_qsearch_block_circuit() {
        let mut circ: QuditCircuit<f64> = QuditCircuit::new(radices![2; 2], 0);
        circ.append_gate(Gate::CP(), loc![0, 1], vec![]);
        circ.append_gate(Gate::U3(), loc![0], vec![]);
        circ.append_gate(Gate::U3(), loc![1], vec![]);
        let tree = circ.to_tree();
        let code = compile(&tree);
        println!("{:?}", tree);
        println!("{:?}", code);
        let mut qvm: QVM<c64> = QVM::new(code, DifferentiationLevel::None);
        let params = vec![1.7; 6];
        let utry = qvm.get_unitary(&params);


        let u3utry: UnitaryMatrix<c64> = Gate::U3().gen_expr().get_unitary(&[1.7, 1.7, 1.7]);
        let cnotutry: UnitaryMatrix<c64> = Gate::CP().gen_expr().get_unitary(&[]);
        let u3utry2 = u3utry.kron(&u3utry);
        let expected = u3utry2.dot(&cnotutry);
        let dist = UnitaryMatrix::new([2], utry.to_owned()).get_distance_from(expected);
        println!("{:?}", dist);
        assert!(dist < 1e-7);
    }

    #[test]
    fn test_gen_code_for_paper() {
        let mut circ: QuditCircuit<f64> = QuditCircuit::new(radices![2; 3], 0);
        circ.append_gate(Gate::U3(), loc![0], vec![]);
        circ.append_gate(Gate::U3(), loc![1], vec![]);
        circ.append_gate(Gate::U3(), loc![2], vec![]);
        circ.append_gate(Gate::CP(), loc![1, 2], vec![]);
        circ.append_gate(Gate::CP(), loc![0, 1], vec![]);
        circ.append_gate(Gate::U3(), loc![0], vec![]);
        circ.append_gate(Gate::CP(), loc![0, 1], vec![]);
        circ.append_gate(Gate::CP(), loc![1, 2], vec![]);
        circ.append_gate(Gate::U3(), loc![0], vec![]);
        circ.append_gate(Gate::U3(), loc![1], vec![]);
        circ.append_gate(Gate::U3(), loc![2], vec![]);
        let tree = circ.to_tree();
        println!("{:?}", tree);
        let code = compile(&tree);
        println!("{:?}", code);
    }

    #[test]
    fn test_qsearch_2block_circuit() {
        let mut circ: QuditCircuit<f64> = QuditCircuit::new(radices![2; 3], 0);
        circ.append_gate(Gate::CP(), loc![0, 1], vec![]);
        circ.append_gate(Gate::U3(), loc![0], vec![]);
        circ.append_gate(Gate::U3(), loc![1], vec![]);
        circ.append_gate(Gate::CP(), loc![1, 2], vec![]);
        circ.append_gate(Gate::U3(), loc![1], vec![]);
        circ.append_gate(Gate::U3(), loc![2], vec![]);
        let tree = circ.to_tree();
        println!("{:?}", tree);
        let code = compile(&tree);
        println!("{:?}", code);
        let mut qvm: QVM<c64> = QVM::new(code, DifferentiationLevel::None);
        let params = vec![1.7; 12];
        let utry = qvm.get_unitary(&params);


        let u3utry: UnitaryMatrix<c64> = Gate::U3().gen_expr().get_unitary(&[1.7, 1.7, 1.7]);
        let cnotutry: UnitaryMatrix<c64> = Gate::CP().gen_expr().get_unitary(&[]);
        let u3utry2 = u3utry.kron(&u3utry);
        let block = u3utry2.dot(&cnotutry);
        let block_i = block.kron(&UnitaryMatrix::identity([2]));
        let i_block = UnitaryMatrix::identity([2]).kron(&block);
        let expected = i_block.dot(&block_i);
        let dist = UnitaryMatrix::new([2], utry.to_owned()).get_distance_from(expected);
        println!("{:?}", dist);
        assert!(dist < 1e-7);
    }

    #[test]
    fn test_qsearch_thin_step_circuit() {
        let circ = build_qsearch_thin_step_circuit(3);
        // assert_eq!(circ.num_cycles(), 2 * 2 * 3 + 1);
        // assert_eq!(circ.num_operations(), 3 * 2 * 3 + 3);

        let tree = circ.to_tree();
        println!("{:?}", tree);

        let code = compile(&tree);
        println!("{:?}", code);

        let mut qvm: QVM<c32> = QVM::new(code, DifferentiationLevel::Gradient);
        let params = vec![1.7; 3 * 2 * 3 * 4 + 12];
        let start = std::time::Instant::now();
        let mut utry = qudit_core::matrix::Mat::zeros(8, 8);
        let mut grad = qudit_core::matrix::MatVec::zeros(8, 8, params.len());
        let n = 1000;
        for _ in 0..n {
            qvm.write_unitary_and_gradient(&params, utry.as_mut(), grad.as_mut());
        }
        let elapsed = start.elapsed();
        println!("Time per unitary: {:?}", elapsed / n as u32);
        // let utry = qvm.get_unitary(&params);
        // println!("{:?}", utry);
    }

    use qudit_expr::Module;
    use qudit_expr::ModuleBuilder;
    use qudit_core::memory::alloc_zeroed_memory;
    use qudit_core::memory::calc_col_stride;
    use qudit_core::matrix::MatMut;

    #[test]
    fn test_cnot_expr_utry() {
        let cnot = Gate::CP().gen_expr();
        let col_stride = calc_col_stride::<c64>(4, 4);
        let mut memory = alloc_zeroed_memory::<c64>(4 * col_stride);
        let name = cnot.name();

        let module: Module<c64> = ModuleBuilder::new("test", DifferentiationLevel::None)
            .add_expression(cnot)
            .build();

        println!("{}", module);

        unsafe {
            let mut matmut: MatMut<c64> = faer::MatMut::from_raw_parts_mut(
                memory.as_mut_ptr() as *mut c64,
                4,
                4,
                1,
                col_stride as isize,
            );

            for i in 0..4 {
                *matmut.rb_mut().get_mut(i, i) = c64::new(1.0, 0.0);
            }
        }

        let matmut: MatMut<c64> = unsafe {
            let cnot_func = module.get_function_raw(&name);
            cnot_func([].as_ptr(), memory.as_ptr() as *mut f64);

            faer::MatMut::from_raw_parts_mut(
                memory.as_mut_ptr() as *mut c64,
                4,
                4,
                1,
                col_stride as isize,
            )
        };
        println!("{:?}", matmut);
        // let utry: UnitaryMatrix<c64> = cnot.get_unitary(&[]);
        // println!("{:?}", utry);
    }
}
