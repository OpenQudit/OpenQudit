use std::collections::HashMap;
use qudit_core::{ClassicalSystem, ComplexScalar, HasParams, HybridSystem, ParamIndices, QuditSystem, ToRadices};

use indexmap::IndexSet;
use qudit_core::{QuditRadices, RealScalar, c64};
use qudit_expr::index::{IndexDirection, TensorIndex};
use qudit_gates::Gate;
use qudit_expr::{TensorExpression, UnitaryExpressionGenerator};
// use qudit_tensor::{BuilderExpressionInput, ExpressionTree, TreeBuilder, TreeOptimizer};
use qudit_tensor::{QuditCircuitTensorNetworkBuilder, QuditTensor, QuditTensorNetwork};
use crate::exprset::{ExpressionSet, OperationSet};
use crate::location::{self, ToLocation};
use crate::param::ParamList;
use crate::point::CircuitDitId;
use crate::ParamEntry;
use crate::{compact::CompactIntegerVector, cycle::QuditCycle, cyclelist::CycleList, instruction::{Instruction, InstructionReference}, iterator::{QuditCircuitBFIterator, QuditCircuitDFIterator, QuditCircuitFastIterator}, location::CircuitLocation, operation::{Operation, OperationReference}, CircuitPoint};


/// A quantum circuit that can be defined with qudits and classical bits.
#[derive(Clone)]
pub struct QuditCircuit<C: ComplexScalar = c64> {
    /// The number of qudits in the circuit.
    num_qudits: usize,

    /// The number of classical dits in the circuit.
    num_dits: usize,

    /// The QuditRadices object that describes the quantum dimension of the circuit.
    qudit_radices: QuditRadices,

    /// The QuditRadices object that describes the classical dimension of the circuit.
    dit_radices: QuditRadices,

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
    // pub gates: IndexSet<Gate>,
    // pub expression_set: IndexSet<TensorExpression>,
    pub expression_set: OperationSet,

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
    /// * `qudit_radices` - The QuditRadices object that describes the qudit system.
    ///
    /// * `dit_radices` - The QuditRadices object that describes the classical system. 
    ///
    /// # Examples
    ///
    /// We can define hybrid quantum-classical circuits:
    /// ```
    /// use qudit_core::c64;
    /// use qudit_circuit::QuditCircuit;
    ///
    /// let two_qubit_circuit = QuditCircuit::<c64>::new([2, 2], [2, 2]);
    /// let two_qutrit_circuit = QuditCircuit::<c64>::new([3, 3], [3, 3]);
    /// ```
    pub fn new<T1: ToRadices, T2: ToRadices>(qudit_radices: T1, dit_radices: T2) -> QuditCircuit<C> {
        QuditCircuit::with_capacity(qudit_radices, dit_radices, 1)
    }

    /// Creates a new purely-quantum QuditCircuit object.
    ///
    /// # Arguments
    ///
    /// * `qudit_radices` - The QuditRadices object that describes the qudit system.
    ///
    /// # Examples
    ///
    /// We can define purely quantum kernels without classical bits:
    /// ```
    /// use qudit_core::c64;
    /// use qudit_circuit::QuditCircuit;
    ///
    /// let two_qubit_circuit = QuditCircuit::<c64>::pure([2, 2]);
    /// let two_qutrit_circuit = QuditCircuit::<c64>::pure([3, 3]);
    /// let hybrid_circuit = QuditCircuit::<c64>::pure([2, 2, 3, 3]);
    /// ```
    pub fn pure<T: ToRadices>(qudit_radices: T) -> QuditCircuit<C> {
        QuditCircuit::with_capacity(qudit_radices, QuditRadices::new::<usize>(&[]), 1)
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
    /// use qudit_core::c64;
    /// use qudit_circuit::QuditCircuit;
    /// let two_qubit_circuit = QuditCircuit::<c64>::with_capacity([2, 2], [2, 2], 10);
    /// ```
    pub fn with_capacity<T1: ToRadices, T2: ToRadices>(
        qudit_radices: T1,
        dit_radices: T2,
        capacity: usize,
    ) -> QuditCircuit<C> {
        let qudit_radices = qudit_radices.to_radices();
        let dit_radices = dit_radices.to_radices();
        let num_qudits = qudit_radices.len();
        let num_dits = dit_radices.len();
        QuditCircuit {
            num_qudits,
            num_dits,
            cycles: CycleList::with_capacity(capacity),
            op_info: HashMap::new(),
            graph_info: HashMap::new(),
            qfront: vec![None; num_qudits],
            qrear: vec![None; num_qudits],
            cfront: vec![None; num_dits],
            crear: vec![None; num_dits],
            qudit_radices,
            dit_radices,
            expression_set: OperationSet::new(),
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
    /// This method is O(1).
    pub fn num_params(&self) -> usize {
        self.params.len()
    }

    /// Returns the number of operations in the circuit.
    ///
    /// # Performance
    ///
    /// This method is O(|t|) where
    ///     - `t` is the number of distinct instruction types in the circuit.
    pub fn num_operations(&self) -> usize {
        self.op_info.iter().map(|(_, count)| count).sum()
    }

    /// A reference to the parameters of the circuit.
    pub fn params(&self) -> &[C::R] {
        &self.params
    }

    /// Increment internal instruction type counter.
    fn inc_op_counter(&mut self, op_type: &OperationReference) {
        *self.op_info.entry(*op_type).or_insert(0) += 1;
    }

    /// Increment internal graph counter.
    fn inc_graph_counter(&mut self, location: &CircuitLocation) {
        for pair in location.get_qudit_pairs() {
            *self.graph_info.entry(pair).or_insert(0) += 1;
        }
    }

    /// Decrement internal instruction type counter.
    fn dec_inst_counter(&mut self, op_type: &OperationReference) -> bool {
        // TODO: sort inst and op names
        if !self.op_info.contains_key(op_type) {
            panic!(
                "Cannot decrement instruction counter for instruction type that does not exist."
            );
        }

        let count = self.op_info.get_mut(op_type).unwrap();
        *count -= 1;

        if *count == 0 {
            self.op_info.remove(op_type);
            // TODO: self.expression_set.remove(op_type)
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

    /// Retrieve the cycle at the logical `index` in the circuit.
    pub(super) fn get_cycle(&self, index: usize) -> &QuditCycle {
        &self.cycles[index]
    }

    /// Checks if `location` is a valid location in the circuit.
    ///
    /// A location is valid if all qudit indices are less than the
    /// number of qudits in the circuit and all classical dit indices
    /// are less than the number of classical dits in the circuit.
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
    /// use qudit_core::c64;
    /// use qudit_circuit::QuditCircuit;
    /// let circuit = QuditCircuit::<c64>::new([2, 2], [2, 2]);
    /// assert!(circuit.is_valid_location([0, 1]));
    /// assert!(circuit.is_valid_location(([0, 1], [0, 1])));
    /// assert!(circuit.is_valid_location((0, 0)));
    /// assert!(!circuit.is_valid_location(([0, 1], [0, 2])));
    /// assert!(!circuit.is_valid_location([0, 1, 2]));
    /// assert!(!circuit.is_valid_location(([0, 1], [2])));
    /// ```
    pub fn is_valid_location<L: ToLocation>(&self, location: L) -> bool {
        let location = location.to_location();
        location.qudits().iter().all(|q| q < self.num_qudits)
            && location.dits().iter().all(|c| c < self.num_dits())
    }

    /// Checks if `point` is a valid point in the circuit.
    ///
    /// A point is valid if its cycle index is in bounds and its
    /// qudit or classical dit index is in bounds.
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
    /// use qudit_core::c64;
    /// use qudit_gates::Gate;
    /// use qudit_circuit::QuditCircuit;
    /// let mut circuit: QuditCircuit<c64> = QuditCircuit::new([2, 2], [2, 2]);
    /// circuit.append_uninit_gate(Gate::P(2), [0]);
    /// assert!(circuit.is_valid_point((0, 0)));
    /// assert!(!circuit.is_valid_point((1, 0)));
    ///
    /// // Negative dit indices (like the -1 below) imply classical indices.
    /// assert!(circuit.is_valid_point((0, -1))); // Cycle 0, Classical dit 1
    /// assert!(!circuit.is_valid_point((0, -2)));
    /// ```
    pub fn is_valid_point<P: Into<CircuitPoint>>(&self, point: P) -> bool {
        let point = point.into();
        point.cycle < self.cycles.len()
            && match point.dit_id {
                CircuitDitId::Quantum(q) => q < self.num_qudits,
                CircuitDitId::Classical(c) => c < self.num_dits,
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
    /// use qudit_core::c64;
    /// use qudit_gates::Gate;
    /// use qudit_circuit::QuditCircuit;
    /// let mut circuit = QuditCircuit::<c64>::new([2, 2], [2, 2]);
    /// circuit.append_uninit_gate(Gate::P(2), [0]);
    /// assert!(circuit.find_available_cycle([0]).is_none());
    /// assert_eq!(circuit.find_available_cycle([1]), Some(0));
    /// ```
    pub fn find_available_cycle<L: ToLocation>(&self, location: L) -> Option<usize> {
        let location = location.to_location();
        if !self.is_valid_location(&location) { // TODO: cleanup ToLocation; a lot of unnecessary
            // copies/allocations in statements like this; think through Into/From/AsRef maybe?
            //
            // TODO: Really, panic over this? This it totally a recoverable error.
            // Cmon, you know better now. Get some proper error handling going already.
            panic!("Cannot find available cycle for invalid location.");
        }

        if self.cycles.is_empty() {
            return None;
        }

        let last_occupied_cycle_option = location
            .qudits()
            .iter()
            .filter_map(|q| self.qrear[q])
            .chain(location.dits().iter().filter_map(|c| self.crear[c]))
            .map(|cycle_index| self.cycles.map_physical_to_logical_idx(cycle_index))
            .max_by(Ord::cmp);

        match last_occupied_cycle_option {
            Some(cycle_index) if cycle_index + 1 < self.num_cycles() => {
                Some(cycle_index + 1)
            }
            None => {
                // Circuit is not empty due to above check,
                // but no gates on location
                Some(0)
            }
            _ => {
                None
            }
        }
    }

    /// Find first or create new available cycle and return its physical index
    fn find_available_or_append_cycle<L: ToLocation>(&mut self, location: L) -> usize {
        // Location validity implicitly checked in find_available_cycle
        if let Some(cycle_index) = self.find_available_cycle(location) {
            self.cycles.map_logical_to_physical_idx(cycle_index)
        } else {
            self.cycles.push()
        }
    }

    // TODO: prepend + insert
    // TODO: param_indices_mapping

    /// Append an instruction to the end of the circuit.
    ///
    /// # Arguments
    ///
    /// * `inst` - The instruction to append.
    pub fn append_instruction(&mut self, inst: Instruction<C::R>) {
        let Instruction { op, location, params } = inst;

        // Build operation reference
        let op_ref = self.expression_set.insert(op);

        // let param_indices = ParamIndices::Joint(self.params.len(), params.len());
        // self.params.extend(params);

        self.append(op_ref, location, params.into());
    }

    fn append(&mut self, op_ref: OperationReference, location: CircuitLocation, params: ParamList<C::R>) {
        // TODO: check valid operation for radix match, measurement bandwidth etc
        
        // TODO: check params is valid: length is equal to op_params, existing exist, etc..
        let param_map = params.to_param_indices(self.params.len());
        self.params.extend(params.new_entries_unwrapped());
        // TODO: have to something about static entries...
        // TODO: have to do something about gate parameters mapped within same gate

        // Find cycle placement (location validity implicitly checked here)
        let cycle_index = self.find_available_or_append_cycle(&location);

        // Update counters
        self.inc_op_counter(&op_ref);
        self.inc_graph_counter(&location);

        // Update quantum DAG info
        for qudit_index in location.qudits() {
            if let Some(rear_cycle_index) = self.qrear[qudit_index] {
                self.cycles[rear_cycle_index].set_qnext(qudit_index, cycle_index);
                self.cycles[cycle_index].set_qprev(qudit_index, rear_cycle_index);
            } else {
                // If qrear is none, no instruction exists on this wire
                // so we update qfront too.
                self.qfront[qudit_index] = Some(cycle_index);
            }
            self.qrear[qudit_index] = Some(cycle_index);
        }

        // Update classical DAG info
        for dit_index in location.dits() {
            if let Some(rear_cycle_index) = self.crear[dit_index] {
                self.cycles[rear_cycle_index].set_cnext(dit_index, cycle_index);
                self.cycles[cycle_index].set_cprev(dit_index, rear_cycle_index);
            } else {
                // If crear is none, no instruction exists on this wire
                // so we update cfront too.
                self.cfront[dit_index] = Some(cycle_index);
            }
            self.crear[dit_index] = Some(cycle_index);
        }

        // Build instruction reference
        let inst_ref = InstructionReference::new(op_ref, location, param_map);

        // Add op to cycle
        self.cycles[cycle_index].push(inst_ref);
    }

    /// Shorthand for appending a gate without caring about the gate parameters
    pub fn append_uninit_gate<L: ToLocation>(&mut self, gate: Gate, location: L) {
        let params = vec![None; gate.num_params()];
        let op_ref = self.expression_set.insert(Operation::Gate(gate));
        self.append(op_ref, location.to_location(), params.into());
    }

    /// Shorthand for appending a gate with specific parameter entries 
    pub fn append_gate<L: ToLocation, P: Into<ParamList<C::R>>>(&mut self, gate: Gate, location: L, params: P) {
        let op_ref = self.expression_set.insert(Operation::Gate(gate));
        self.append(op_ref, location.to_location(), params.into());
    }

    /// Remove the operation at `point` from the circuit.
    pub fn remove(&mut self, point: CircuitPoint) {
        if !self.is_valid_point(point) {
            // Don't need to panic here... TODO
            panic!("Cannot remove operation at invalid point.");
        }

        let location = match self.cycles[point.cycle].get_location(point.dit_id) {
            Some(location) => location.clone(),
            // TODO: Error handling
            None => panic!("Operation not found at {} in cycle {}", point.dit_id, point.cycle),
        };

        // Update circuit quantum DAG info
        for qudit_index in location.qudits() {
            let qnext = self.cycles[point.cycle].get_qnext(qudit_index);
            let qprev = self.cycles[point.cycle].get_qprev(qudit_index);

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
        for clbit_index in location.dits() {
            let cnext = self.cycles[point.cycle].get_cnext(clbit_index);
            let cprev = self.cycles[point.cycle].get_cprev(clbit_index);

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

        // Remove the instruction from the cycle
        match self.cycles[point.cycle].remove(point.dit_id) {
            Some(inst_ref) => {
                // Update counters
                self.dec_graph_counter(&inst_ref.location);
                self.dec_inst_counter(&inst_ref.op);
            },
            // TODO: Error handling
            None => panic!("Operation not found at {} in cycle {}", point.dit_id, point.cycle),
        }

        // If cycle is empty, remove it
        if self.cycles[point.cycle].is_empty() {
            self.cycles.remove(point.cycle);
        }
    }

    // /////////////////////////////////////////////////////////////////
    // DAG Methods
    // /////////////////////////////////////////////////////////////////

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
    /// The same operation point (value in return map) may be pointed to
    /// by two different keys in the hash map if it is at the front of
    /// the circuit at multiple spots. For example, if a cnot was at the
    /// front of the circuit, then it would be pointed to by both the
    /// control and target qudit indices.
    ///
    /// # Examples
    ///
    /// ```
    /// use qudit_core::c64;
    /// use qudit_gates::Gate;
    /// use qudit_circuit::QuditCircuit;
    /// use qudit_circuit::{CircuitDitId, CircuitPoint};
    /// let mut circuit = QuditCircuit::<c64>::new([2, 2], [2, 2]);
    /// circuit.append_uninit_gate(Gate::P(2), [0]);
    /// circuit.append_uninit_gate(Gate::H(2), [1]);
    /// assert_eq!(circuit.front().len(), 2);
    /// assert_eq!(circuit.front()[&CircuitDitId::Quantum(0)], CircuitPoint::new(0, 0));
    /// assert_eq!(circuit.front()[&CircuitDitId::Quantum(1)], CircuitPoint::new(0, 1));
    /// ```
    pub fn front(&self) -> HashMap<CircuitDitId, CircuitPoint> {
        let quantum = self.qfront.iter().enumerate()
            .filter_map(|q| q.1.map(|phy_cidx| {
                let id = CircuitDitId::quantum(q.0);
                (id, CircuitPoint::new(self.cycles.map_physical_to_logical_idx(phy_cidx), id))
            }));

        let classical = self.cfront.iter().enumerate()
            .filter_map(|c| c.1.map(|phy_cidx| { 
                let id = CircuitDitId::classical(c.0);
                (id, CircuitPoint::new(self.cycles.map_physical_to_logical_idx(phy_cidx), id))
            }));

        quantum.chain(classical).collect()
    }

    /// Distill the circuit rear nodes into a hashmap.
    ///
    /// See [`QuditCircuit::front`] for more information.
    pub fn rear(&self) -> HashMap<CircuitDitId, CircuitPoint> {
        let quantum = self.qrear.iter().enumerate()
            .filter_map(|q| q.1.map(|phy_cidx| {
                let id = CircuitDitId::quantum(q.0);
                (id, CircuitPoint::new(self.cycles.map_physical_to_logical_idx(phy_cidx), id))
            }));

        let classical = self.crear.iter().enumerate()
            .filter_map(|c| c.1.map(|phy_cidx| { 
                let id = CircuitDitId::classical(c.0);
                (id, CircuitPoint::new(self.cycles.map_physical_to_logical_idx(phy_cidx), id))
            }));

        quantum.chain(classical).collect()
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
    /// use qudit_core::c64;
    /// use qudit_gates::Gate;
    /// use qudit_circuit::QuditCircuit;
    /// use qudit_circuit::{CircuitPoint, CircuitDitId};
    /// let mut circuit: QuditCircuit<c64> = QuditCircuit::new([2, 2], [2, 2]);
    /// circuit.append_uninit_gate(Gate::P(2), [0]);
    /// circuit.append_uninit_gate(Gate::H(2), [1]);
    /// circuit.append_uninit_gate(Gate::P(2), [0]);
    /// assert_eq!(circuit.next(CircuitPoint::new(0, 0)).len(), 1);
    /// assert_eq!(circuit.next(CircuitPoint::new(0, 0))[&CircuitDitId::Quantum(0)],
    /// CircuitPoint::new(1, 0));
    /// ```
    pub fn next(&self, point: CircuitPoint) -> HashMap<CircuitDitId, CircuitPoint> {
        let location = self.cycles[point.cycle].get_location(point.dit_id); // TODO consider
        // extracting to an inlined function get_location(CircuitPointLike)

        match location {
            Some(location) => { 
                let quantum = location.qudits().iter().filter_map(|q_idx|
                    self.cycles[point.cycle].get_qnext(q_idx).map(|next_cidx| {
                        let id = CircuitDitId::quantum(q_idx);
                        (id, CircuitPoint::new(self.cycles.map_physical_to_logical_idx(next_cidx), id))
                    })
                );

                let classical = location.dits().iter().filter_map(|c_idx|
                    self.cycles[point.cycle].get_cnext(c_idx).map(|next_cidx| {
                        let id = CircuitDitId::classical(c_idx);
                        (id, CircuitPoint::new(self.cycles.map_physical_to_logical_idx(next_cidx), id))
                    })
                );
                quantum.chain(classical).collect()
            },
            None => HashMap::new(),
        }
    }

    /// Gather the points of the previous operations from the point of an
    /// operation.
    ///
    /// See [`QuditCircuit::next`] for more information.
    pub fn prev(&self, point: CircuitPoint) -> HashMap<CircuitDitId, CircuitPoint> {
        let location = self.cycles[point.cycle].get_location(point.dit_id); // TODO consider
        // extracting to an inlined function get_location(CircuitPointLike)

        match location {
            Some(location) => { 
                let quantum = location.qudits().iter().filter_map(|q_idx|
                    self.cycles[point.cycle].get_qprev(q_idx).map(|prev_cidx| {
                        let id = CircuitDitId::quantum(q_idx);
                        (id, CircuitPoint::new(self.cycles.map_physical_to_logical_idx(prev_cidx), id))
                    })
                );

                let classical = location.dits().iter().filter_map(|c_idx|
                    self.cycles[point.cycle].get_cprev(c_idx).map(|prev_cidx| {
                        let id = CircuitDitId::classical(c_idx);
                        (id, CircuitPoint::new(self.cycles.map_physical_to_logical_idx(prev_cidx), id))
                    })
                );
                quantum.chain(classical).collect()
            },
            None => HashMap::new(),
        }
    }

    /// Return an iterator over the operations in the circuit.
    ///
    /// The ordering is not guaranteed to be consistent, but it will
    /// be in a simulation/topological order. For more control over the
    /// ordering of iteration see [`QuditCircuit::iter_df`] or
    /// [`QuditCircuit::iter_bf`].
    pub fn iter(&self) -> QuditCircuitFastIterator<C> {
        QuditCircuitFastIterator::new(self)
    }

    /// Return a depth-first iterator over the operations in the circuit.
    ///
    /// See [`QuditCircuitDFIterator`] for more info.
    pub fn iter_df(&self) -> QuditCircuitDFIterator<C> {
        QuditCircuitDFIterator::new(self)
    }

    /// Return a breadth-first iterator over the operations in the circuit.
    ///
    /// See [`QuditCircuitBFIterator`] for more info.
    pub fn iter_bf(&self) -> QuditCircuitBFIterator<C> {
        QuditCircuitBFIterator::new(self)
    }

//     /// Return an iterator over the operations in the circuit with cycles.
//     pub fn iter_with_cycles(&self) -> QuditCircuitFastIteratorWithCycles<C> {
//         QuditCircuitFastIteratorWithCycles::new(self)
//     }

    /// Convert the circuit to a symbolic tensor network.
    pub fn to_tensor_network(&self) -> QuditTensorNetwork {
        self.as_tensor_network_builder().build()
    }

    pub fn as_tensor_network_builder(&self) -> QuditCircuitTensorNetworkBuilder {
        let mut network = QuditCircuitTensorNetworkBuilder::new(self.qudit_radices());
        for inst_ref in self.iter() {
            let op_ref = inst_ref.op.dereference(self);
            match op_ref {
                Operation::Gate(gate) => {
                    network = network.prepend(QuditTensor::new(gate.gen_expr().to_tensor_expression(), inst_ref.param_indices.clone()), inst_ref.location.qudits().to_vec(), inst_ref.location.qudits().to_vec(), vec![]);
                }
                Operation::ProjectiveMeasurement(t, a) => {
                    let clbit_indices = a.iter().map(|clbit| clbit.to_string()).collect();
                    network = network.prepend(QuditTensor::new(t.clone(), inst_ref.param_indices.clone()), inst_ref.location.qudits().to_vec(), inst_ref.location.qudits().to_vec(), clbit_indices);
                }
                Operation::TerminatingMeasurement(s, a) => {
                    let clbit_indices: Vec<String> = a.iter().map(|clbit| clbit.to_string()).collect();
                    let mut t = s.to_tensor_expression();
                    t.reindex(clbit_indices.iter().map(|_| (IndexDirection::Batch, 2)).chain(t.indices().iter().filter(|idx| idx.direction() != IndexDirection::Batch).map(|idx| (idx.direction(), idx.index_size())))
                        .enumerate()
                        .map(|(id, (dir, size))| TensorIndex::new(dir, id, size))
                        .collect());
                    network = network.prepend(QuditTensor::new(t, inst_ref.param_indices.clone()), inst_ref.location.qudits().to_vec(), vec![], clbit_indices);
                }
                Operation::ClassicallyControlled(g, a) => {
                    let clbit_indices: Vec<String> = a.iter().map(|clbit| clbit.to_string()).collect();
                    let mut t = g.gen_expr().to_tensor_expression().stack_with_identity(&[2usize.pow(clbit_indices.len() as u32) - 1], 2usize.pow(clbit_indices.len() as u32));
                    t.reindex(clbit_indices.iter().map(|_| (IndexDirection::Batch, 2)).chain(t.indices().iter().filter(|idx| idx.direction() != IndexDirection::Batch).map(|idx| (idx.direction(), idx.index_size())))
                        .enumerate()
                        .map(|(id, (dir, size))| TensorIndex::new(dir, id, size))
                        .collect());
                    network = network.prepend(QuditTensor::new(t.clone(), inst_ref.param_indices.clone()), inst_ref.location.qudits().to_vec(), inst_ref.location.qudits().to_vec(), clbit_indices);
                }
                Operation::Initialization(s) => { todo!() }
                Operation::Reset => { todo!() }
                Operation::Barrier => { /* NO-OP */ }
            }
        }
        network
    }
}

impl<C: ComplexScalar> QuditSystem for QuditCircuit<C> {
    fn num_qudits(&self) -> usize {
        self.num_qudits
    }

    fn dimension(&self) -> usize {
        self.qudit_radices.dimension()
    }

    fn radices(&self) -> QuditRadices {
        self.qudit_radices.clone()
    }
}

impl<C: ComplexScalar> HasParams for QuditCircuit<C> {
    fn num_params(&self) -> usize {
        self.params.len()
    }
}

impl<C: ComplexScalar> ClassicalSystem for QuditCircuit<C> {
    fn radices(&self) -> qudit_core::QuditRadices {
        self.dit_radices.clone()
    }

    fn num_dits(&self) -> usize {
        self.num_dits
    }
}

impl<C: ComplexScalar> HybridSystem for QuditCircuit<C> {}

#[cfg(test)]
mod tests {
    use super::*;
    use bit_set::BitSet;
    use faer::reborrow::ReborrowMut;
    use qudit_core::c32;
    use qudit_core::c64;
    use qudit_core::radices;
    use qudit_core::QuditRadices;
    use qudit_expr::DifferentiationLevel;
    use qudit_expr::StateSystemExpression;
    use qudit_tensor::Bytecode;
    use crate::CircuitLocation;
    use qudit_core::unitary::UnitaryMatrix;
    use qudit_core::unitary::UnitaryFn;
    use qudit_expr::{FUNCTION, GRADIENT};

    pub fn build_qsearch_thin_step_circuit(n: usize) -> QuditCircuit {
        let block_expr = Gate::U3().gen_expr().otimes(Gate::U3().gen_expr()).dot(Gate::CX().gen_expr());
        let mut circ = QuditCircuit::pure(vec![2; n]);
        for i in 0..n {
            circ.append_uninit_gate(Gate::U3(), [i]);
        }
        for _ in 0..n {
            for i in 0..(n - 1) {
                circ.append_uninit_gate(Gate::Expression(block_expr.clone()), [i, i+1]);
                // circ.append_gate(Gate::CX(), loc![i, i + 1], vec![]);
                // circ.append_gate(Gate::U3(), loc![i], vec![]);
                // circ.append_gate(Gate::U3(), loc![i + 1], vec![]);
            }
        }
        circ
    }

    #[test]
    fn build_qsearch_thin_step_circuit_test() {
        build_qsearch_thin_step_circuit(3);
        build_qsearch_thin_step_circuit(4);
        build_qsearch_thin_step_circuit(5);
        build_qsearch_thin_step_circuit(6);
        build_qsearch_thin_step_circuit(7);
    }

    #[test]
    fn build_qsearch_thin_step_circuit_to_tensor_test() {
        const n: usize = 3;

        // let mut circ: QuditCircuit<c64> = QuditCircuit::new(radices![2; n], 0);
        // for i in 0..n {
        //     circ.append_gate(Gate::U3(), loc![i], vec![]);
        // }
        // circ.append_gate(Gate::CP(), loc![0, 1], vec![]);
        let circ = build_qsearch_thin_step_circuit(n);

        // let start = std::time::Instant::now();
        // for _ in 0..1000 {
        //     let network = circ.to_tensor_network();
        //     let code = qudit_tensor::compile_network(network);
        // }
        // let elapsed = start.elapsed();
        // println!("Time to construct: {:?}", elapsed / 1000);

        let network = circ.to_tensor_network();
        let code = qudit_tensor::compile_network(network);


        // let Bytecode { expressions, const_code, dynamic_code, buffers } = code;
        // let code = Bytecode { expressions, const_code, dynamic_code: dynamic_code[..3].to_vec(), buffers };

        println!("{:?}", code);
        let mut tnvm = qudit_tensor::TNVM::<c32, GRADIENT>::new(&code);
        let start = std::time::Instant::now();
        for _ in 0..1000 {
            let result = tnvm.evaluate::<GRADIENT>(&[1.7; (3*n) + (7*(n-1)*n)]);
        }
        let elapsed = start.elapsed();
        println!("Time per evaluation: {:?}", elapsed / 1000);
        // let unitary = result.get_fn_result().unpack_matrix();
        // println!("{:?}", unitary);
    }

    // #[test]
    // fn qutrit_test() {
    //     const n: usize = 3;

    //     let mut circ: QuditCircuit<c64> = QuditCircuit::new(radices![3; n], 0);
    //     for i in 0..n {
    //         circ.append_gate(Gate::PGate(3), loc![i], vec![]);
    //     }

    //     let network = circ.to_tensor_network();
    //     let code = qudit_tensor::compile_network(network);

    //     println!("{:?}", code);
    //     let mut tnvm = qudit_tensor::TNVM::<c32, GRADIENT>::new(&code);
    //     let start = std::time::Instant::now();
    //     for _ in 0..1000 {
    //         let result = tnvm.evaluate(&[1.7; (3*n) + (7*(n-1)*n)]);
    //     }
    //     let elapsed = start.elapsed();
    //     println!("Time per evaluation: {:?}", elapsed / 1000);
    //     // let unitary = result.get_fn_result().unpack_matrix();
    //     // println!("{:?}", unitary);
    // }

    #[test]
    fn simple_instantiation_structure_test() {
        let mut circ: QuditCircuit<c32> = QuditCircuit::new([2], [2, 2, 2]);
        circ.append_uninit_gate(Gate::U3(), [0]);
        let mut builder = circ.as_tensor_network_builder();
        builder = builder.prepend_unitary(UnitaryMatrix::<c32>::identity(radices![2]), vec![0]);
        let network = builder.build();
        let code = qudit_tensor::compile_network(network);
        let mut tnvm = qudit_tensor::TNVM::<c32, GRADIENT>::new(&code);
        let result = tnvm.evaluate::<GRADIENT>(&[1.7; 3]);
        let fn_out = result.get_fn_result().unpack_matrix();
        let grad_out = result.get_grad_result().unpack_tensor3d();
    }

    #[test]
    fn dynamic_circuit_test() {
        let mut circ: QuditCircuit<c32> = QuditCircuit::new([2, 2, 2, 2], [2, 2]); // TODO:
        // allow dit level classical information

        for i in 0..4 {
            circ.append_uninit_gate(Gate::U3(), [i]);
        }

        let block_expr = Gate::U3().gen_expr().otimes(Gate::U3().gen_expr()).dot(Gate::CX().gen_expr());

        circ.append_instruction(Instruction::new(Operation::Gate(Gate::Expression(block_expr.clone())), [0, 1], vec![]));
        circ.append_instruction(Instruction::new(Operation::Gate(Gate::Expression(block_expr.clone())), [2, 3], vec![]));
        circ.append_instruction(Instruction::new(Operation::Gate(Gate::Expression(block_expr.clone())), [1, 2], vec![]));
        // circ.append_instruction(Instruction::new(Operation::Gate(Gate::Expression(block_expr.clone())), loc![3, 4], vec![]));

        // let clbits = BitSet::from_bit_vec(&[0b11]);
        let mut clbits = BitSet::new();
        clbits.insert(0);
        clbits.insert(1);
        // let two_qubit_basis_measurement = StateSystemExpression::new("ThreeQMeasure() {
        //     [
        //         [[ 1, 0, 0, 0, 0, 0, 0, 0 ]],
        //         [[ 0, 1, 0, 0, 0, 0, 0, 0 ]],
        //         [[ 0, 0, 1, 0, 0, 0, 0, 0 ]],
        //         [[ 0, 0, 0, 1, 0, 0, 0, 0 ]],
        //         [[ 0, 0, 0, 0, 1, 0, 0, 0 ]],
        //         [[ 0, 0, 0, 0, 0, 1, 0, 0 ]],
        //         [[ 0, 0, 0, 0, 0, 0, 1, 0 ]],
        //         [[ 0, 0, 0, 0, 0, 0, 0, 1 ]],
        //     ]
        // }");
        let two_qubit_basis_measurement = StateSystemExpression::new("ThreeQMeasure() {
            [
                [[ 1, 0, 0, 0 ]],
                [[ 0, 1, 0, 0 ]],
                [[ 0, 0, 1, 0 ]],
                [[ 0, 0, 0, 1 ]],
            ]
        }");
        circ.append_instruction(Instruction::new(Operation::TerminatingMeasurement(two_qubit_basis_measurement, clbits.clone()), [1, 2], vec![]));

        circ.append_instruction(Instruction::new(Operation::ClassicallyControlled(Gate::U3(), clbits.clone()), [0], vec![]));
        circ.append_instruction(Instruction::new(Operation::ClassicallyControlled(Gate::U3(), clbits.clone()), [3], vec![]));
        // circ.append_instruction(Instruction::new(Operation::ClassicallyControlled(Gate::U3(), clbits), loc![0, 3], vec![]));

        let network = circ.to_tensor_network();
        let code = qudit_tensor::compile_network(network);
        let mut tnvm = qudit_tensor::TNVM::<c32, GRADIENT>::new(&code);
        let start = std::time::Instant::now();
        for _ in 0..1000 {
            let result = tnvm.evaluate::<GRADIENT>(&[1.7; 36]);
        }
        let elapsed = start.elapsed();
        println!("Time per evaluation: {:?}", elapsed / 1000);
    }

    #[test]
    fn test_param_overlapping_cirucit() {
        let mut circ: QuditCircuit<c64> = QuditCircuit::pure([2, 2]);
        circ.append_uninit_gate(Gate::CX(), [0, 1]);
        circ.append_uninit_gate(Gate::P(2), [1]);
        circ.append_uninit_gate(Gate::CX(), [0, 1]);
        circ.append_gate(Gate::P(2), [1], [ParamEntry::Existing(0)]);
        circ.append_uninit_gate(Gate::CX(), [0, 1]);

        let network = circ.to_tensor_network();
        let code = qudit_tensor::compile_network(network);
        let mut tnvm = qudit_tensor::TNVM::<c64, GRADIENT>::new(&code);
        let start = std::time::Instant::now();
        for _ in 0..1000 {
            let result = tnvm.evaluate::<GRADIENT>(&[1.7]);
        }
        let result = tnvm.evaluate::<GRADIENT>(&[1.7]);
        dbg!(result.get_grad_result().unpack_tensor3d());
        let elapsed = start.elapsed();
        println!("Time per evaluation: {:?}", elapsed / 1000);
    }

    // #[test]
    // fn test_u3_mul_circuit() {
    //     let mut circ: QuditCircuit<f64> = QuditCircuit::new(radices![2; 1], 0);
    //     circ.append_gate(Gate::U3(), loc![0], vec![1.7, 1.7, 1.7]);
    //     circ.append_gate(Gate::U3(), loc![0], vec![1.7, 1.7, 1.7]);
    //     let tree = circ.to_tree();
    //     println!("{:?}", tree);
    //     let code = compile(&tree);
    //     println!("{:?}", code);
    //     let mut qvm: QVM<c64> = QVM::new(code, DifferentiationLevel::None);
    //     let params = vec![1.7; 6];
    //     let utry = qvm.get_unitary(&params);

    //     let u3utry: UnitaryMatrix<c64> = Gate::U3().gen_expr().get_unitary(&[1.7, 1.7, 1.7]);
    //     let expected = u3utry.dot(&u3utry);
    //     let dist = UnitaryMatrix::new([2], utry.to_owned()).get_distance_from(expected);
    //     println!("{:?}", dist);
    //     assert!(dist < 1e-7);
    // }

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

    // #[test]
    // fn test_qsearch_block_circuit() {
    //     let mut circ: QuditCircuit<f64> = QuditCircuit::new(radices![2; 2], 0);
    //     circ.append_gate(Gate::CP(), loc![0, 1], vec![]);
    //     circ.append_gate(Gate::U3(), loc![0], vec![]);
    //     circ.append_gate(Gate::U3(), loc![1], vec![]);
    //     let tree = circ.to_tree();
    //     let code = compile(&tree);
    //     println!("{:?}", tree);
    //     println!("{:?}", code);
    //     let mut qvm: QVM<c64> = QVM::new(code, DifferentiationLevel::None);
    //     let params = vec![1.7; 6];
    //     let utry = qvm.get_unitary(&params);


    //     let u3utry: UnitaryMatrix<c64> = Gate::U3().gen_expr().get_unitary(&[1.7, 1.7, 1.7]);
    //     let cnotutry: UnitaryMatrix<c64> = Gate::CP().gen_expr().get_unitary(&[]);
    //     let u3utry2 = u3utry.kron(&u3utry);
    //     let expected = u3utry2.dot(&cnotutry);
    //     let dist = UnitaryMatrix::new([2], utry.to_owned()).get_distance_from(expected);
    //     println!("{:?}", dist);
    //     assert!(dist < 1e-7);
    // }

    // #[test]
    // fn test_gen_code_for_paper() {
    //     let mut circ: QuditCircuit<f64> = QuditCircuit::new(radices![2; 3], 0);
    //     circ.append_gate(Gate::U3(), loc![0], vec![]);
    //     circ.append_gate(Gate::U3(), loc![1], vec![]);
    //     circ.append_gate(Gate::U3(), loc![2], vec![]);
    //     circ.append_gate(Gate::CP(), loc![1, 2], vec![]);
    //     circ.append_gate(Gate::CP(), loc![0, 1], vec![]);
    //     circ.append_gate(Gate::U3(), loc![0], vec![]);
    //     circ.append_gate(Gate::CP(), loc![0, 1], vec![]);
    //     circ.append_gate(Gate::CP(), loc![1, 2], vec![]);
    //     circ.append_gate(Gate::U3(), loc![0], vec![]);
    //     circ.append_gate(Gate::U3(), loc![1], vec![]);
    //     circ.append_gate(Gate::U3(), loc![2], vec![]);
    //     let tree = circ.to_tree();
    //     println!("{:?}", tree);
    //     let code = compile(&tree);
    //     println!("{:?}", code);
    // }

    // #[test]
    // fn test_qsearch_2block_circuit() {
    //     let mut circ: QuditCircuit<f64> = QuditCircuit::new(radices![2; 3], 0);
    //     circ.append_gate(Gate::CP(), loc![0, 1], vec![]);
    //     circ.append_gate(Gate::U3(), loc![0], vec![]);
    //     circ.append_gate(Gate::U3(), loc![1], vec![]);
    //     circ.append_gate(Gate::CP(), loc![1, 2], vec![]);
    //     circ.append_gate(Gate::U3(), loc![1], vec![]);
    //     circ.append_gate(Gate::U3(), loc![2], vec![]);
    //     let tree = circ.to_tree();
    //     println!("{:?}", tree);
    //     let code = compile(&tree);
    //     println!("{:?}", code);
    //     let mut qvm: QVM<c64> = QVM::new(code, DifferentiationLevel::None);
    //     let params = vec![1.7; 12];
    //     let utry = qvm.get_unitary(&params);


    //     let u3utry: UnitaryMatrix<c64> = Gate::U3().gen_expr().get_unitary(&[1.7, 1.7, 1.7]);
    //     let cnotutry: UnitaryMatrix<c64> = Gate::CP().gen_expr().get_unitary(&[]);
    //     let u3utry2 = u3utry.kron(&u3utry);
    //     let block = u3utry2.dot(&cnotutry);
    //     let block_i = block.kron(&UnitaryMatrix::identity([2]));
    //     let i_block = UnitaryMatrix::identity([2]).kron(&block);
    //     let expected = i_block.dot(&block_i);
    //     let dist = UnitaryMatrix::new([2], utry.to_owned()).get_distance_from(expected);
    //     println!("{:?}", dist);
    //     assert!(dist < 1e-7);
    // }

    // #[test]
    // fn test_qsearch_thin_step_circuit() {
    //     let circ = build_qsearch_thin_step_circuit(3);
    //     // assert_eq!(circ.num_cycles(), 2 * 2 * 3 + 1);
    //     // assert_eq!(circ.num_operations(), 3 * 2 * 3 + 3);

    //     let tree = circ.to_tree();
    //     println!("{:?}", tree);

    //     let code = compile(&tree);
    //     println!("{:?}", code);

    //     let mut qvm: QVM<c32> = QVM::new(code, DifferentiationLevel::Gradient);
    //     let params = vec![1.7; 3 * 2 * 3 * 4 + 12];
    //     let start = std::time::Instant::now();
    //     let mut utry = qudit_core::matrix::Mat::zeros(8, 8);
    //     let mut grad = qudit_core::matrix::MatVec::zeros(8, 8, params.len());
    //     let n = 1000;
    //     for _ in 0..n {
    //         qvm.write_unitary_and_gradient(&params, utry.as_mut(), grad.as_mut());
    //     }
    //     let elapsed = start.elapsed();
    //     println!("Time per unitary: {:?}", elapsed / n as u32);
    //     // let utry = qvm.get_unitary(&params);
    //     // println!("{:?}", utry);
    // }

    // use qudit_expr::Module;
    // use qudit_expr::ModuleBuilder;
    // use qudit_core::memory::alloc_zeroed_memory;
    // use qudit_core::memory::calc_col_stride;
    // use qudit_core::matrix::MatMut;

    // #[test]
    // fn test_cnot_expr_utry() {
    //     let cnot = Gate::CP().gen_expr();
    //     let col_stride = calc_col_stride::<c64>(4, 4);
    //     let mut memory = alloc_zeroed_memory::<c64>(4 * col_stride);
    //     let name = cnot.name();

    //     let module: Module<c64> = ModuleBuilder::new("test", DifferentiationLevel::None)
    //         .add_expression(cnot)
    //         .build();

    //     println!("{}", module);

    //     unsafe {
    //         let mut matmut: MatMut<c64> = faer::MatMut::from_raw_parts_mut(
    //             memory.as_mut_ptr() as *mut c64,
    //             4,
    //             4,
    //             1,
    //             col_stride as isize,
    //         );

    //         for i in 0..4 {
    //             *matmut.rb_mut().get_mut(i, i) = c64::new(1.0, 0.0);
    //         }
    //     }

    //     let matmut: MatMut<c64> = unsafe {
    //         let cnot_func = module.get_function_raw(&name);
    //         cnot_func([].as_ptr(), memory.as_ptr() as *mut f64);

    //         faer::MatMut::from_raw_parts_mut(
    //             memory.as_mut_ptr() as *mut c64,
    //             4,
    //             4,
    //             1,
    //             col_stride as isize,
    //         )
    //     };
    //     println!("{:?}", matmut);
    //     // let utry: UnitaryMatrix<c64> = cnot.get_unitary(&[]);
    //     // println!("{:?}", utry);
    // }
}
