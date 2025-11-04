use std::collections::HashMap;
use qudit_core::array::Tensor;
use qudit_core::{ClassicalSystem, ComplexScalar, HasParams, HybridSystem, ParamIndices, ParamInfo, QuditSystem};
use qudit_core::Radices;
use qudit_expr::index::IndexDirection;
use qudit_expr::{BraSystemExpression, KetExpression, KrausOperatorsExpression, TensorExpression, UnitaryExpression, UnitarySystemExpression, FUNCTION};
use qudit_tensor::{QuditCircuitTensorNetworkBuilder, QuditTensor, QuditTensorNetwork};
use rustc_hash::FxHashMap;
use slotmap::Key;
use crate::cycle::{CycleId, CycleIndex, InstId};
use crate::instruction::{Instruction, InstructionId};
use crate::param::{Argument as ParameterEntry, ArgumentList, Parameter, ParameterVector};
use crate::wire::Wire;
use crate::wire::WireList;
use crate::operation::{CircuitOperation, DirectiveOperation, ExpressionOperation, OpKind, Operation};
use crate::operation::OperationSet;
use crate::operation::OpCode;
use crate::cycle::CycleList;

/// A quantum circuit that can be defined with qudits and classical bits.
#[derive(Clone)]
pub struct QuditCircuit {
    /// The QuditRadices object that describes the quantum dimension of the circuit.
    qudit_radices: Radices,

    /// The QuditRadices object that describes the classical dimension of the circuit.
    dit_radices: Radices,

    /// All instructions in the circuit stored in cycles.
    cycles: CycleList,

    /// The set of cached operations in the circuit.
    operations: OperationSet,

    /// The stored parameters of the circuit.
    params: ParameterVector,

    /// A map that stores information on each type of operation in the circuit.
    /// Currently, counts for each type of operation is stored.
    op_info: HashMap<OpCode, usize>,

    /// A pointer to the first operation on each wire.
    front: FxHashMap<Wire, CycleId>,

    /// A pointer to the last operation on each wire.
    rear: FxHashMap<Wire, CycleId>,
}

impl QuditCircuit {

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
    /// use qudit_circuit::QuditCircuit;
    ///
    /// let two_qubit_circuit = QuditCircuit::new([2, 2], [2, 2]);
    /// let two_qutrit_circuit = QuditCircuit::new([3, 3], [3, 3]);
    /// ```
    pub fn new<T1: Into<Radices>, T2: Into<Radices>>(qudit_radices: T1, dit_radices: T2) -> QuditCircuit {
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
    /// use qudit_circuit::QuditCircuit;
    ///
    /// let two_qubit_circuit = QuditCircuit::pure([2, 2]);
    /// let two_qutrit_circuit = QuditCircuit::pure([3, 3]);
    /// let hybrid_circuit = QuditCircuit::pure([2, 2, 3, 3]);
    /// ```
    pub fn pure<T: Into<Radices>>(qudit_radices: T) -> QuditCircuit {
        QuditCircuit::with_capacity(qudit_radices, Radices::from(&[] as &[usize]), 1)
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
    /// let two_qubit_circuit = QuditCircuit::with_capacity([2, 2], [2, 2], 10);
    /// ```
    pub fn with_capacity<T1: Into<Radices>, T2: Into<Radices>>(
        qudit_radices: T1,
        dit_radices: T2,
        capacity: usize,
    ) -> QuditCircuit {
        let qudit_radices = qudit_radices.into();
        let dit_radices = dit_radices.into();
        QuditCircuit {
            qudit_radices,
            dit_radices,
            cycles: CycleList::with_capacity(capacity),
            op_info: HashMap::new(),
            front: FxHashMap::default(),
            rear: FxHashMap::default(),
            operations: OperationSet::new(),
            params: ParameterVector::default(),
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

    /// Returns a vector of active qudit indices.
    ///
    /// An active qudit is one that participates in at least one operation.
    ///
    /// # Returns
    ///
    /// A vector containing the indices of qudits that are active.
    ///
    /// # Performance
    ///
    /// This method is O(q) where
    ///     - `q` is the number of qudits in the circuit.
    pub fn active_qudits(&self) -> Vec<usize> {
        self.front.iter()
            .filter_map(|(wire, _)| if wire.is_quantum() { Some(wire.index()) } else { None })
            .collect()
    }

    /// Returns a vector of active classical dit indices.
    ///
    /// An active classical dit is one that participates in at least one operation.
    ///
    /// # Returns
    ///
    /// A vector containing the indices of classical dits that are active.
    ///
    /// # Performance
    ///
    /// This method is O(c) where
    ///     - `c` is the number of classical dits in the circuit.
    pub fn active_dits(&self) -> Vec<usize> {
        self.front.iter()
            .filter_map(|(wire, _)| if wire.is_classical() { Some(wire.index()) } else { None })
            .collect()
    }

    /// A reference to the parameters of the circuit.
    pub fn params(&self) -> &[Parameter] {
        &self.params
    }

    /// Checks if the circuit is empty.
    ///
    /// # Returns
    ///
    /// `true` if the circuit contains no cycles, `false` otherwise.
    ///
    /// # Examples
    ///
    /// ```
    /// use qudit_circuit::QuditCircuit;
    /// let circuit = QuditCircuit::new([2, 2], [2, 2]);
    /// assert!(circuit.is_empty());
    /// ```
    ///
    /// # Performance
    ///
    /// This method is O(1).
    pub fn is_empty(&self) -> bool {
        self.cycles.is_empty()
    }

    /// Increment internal instruction type counter.
    fn inc_op_counter(&mut self, op_type: &OpCode) {
        *self.op_info.entry(*op_type).or_insert(0) += 1;
    }

    /// Decrement internal instruction type counter.
    #[allow(dead_code)]
    fn dec_inst_counter(&mut self, op_type: &OpCode) -> bool {
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
    /// use qudit_circuit::QuditCircuit;
    /// let circuit = QuditCircuit::new([2, 2], [2, 2]);
    /// assert!(circuit.is_valid_location([0, 1]));
    /// assert!(circuit.is_valid_location(([0, 1], [0, 1])));
    /// assert!(circuit.is_valid_location((0, 0)));
    /// assert!(!circuit.is_valid_location(([0, 1], [0, 2])));
    /// assert!(!circuit.is_valid_location([0, 1, 2]));
    /// assert!(!circuit.is_valid_location(([0, 1], [2])));
    /// ```
    pub fn is_valid_wires(&self, wires: &WireList) -> bool {
        wires.qudits().all(|q| q < self.num_qudits())
            && wires.dits().all(|c| c < self.num_dits())
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
    /// circuit.append(Gate::P(2), [0], None);
    /// assert!(circuit.is_valid_point((0, 0)));
    /// assert!(!circuit.is_valid_point((1, 0)));
    ///
    /// // Negative dit indices (like the -1 below) imply classical indices.
    /// assert!(circuit.is_valid_point((0, -1))); // Cycle 0, Classical dit 1
    /// assert!(!circuit.is_valid_point((0, -2)));
    /// ```
    // pub fn is_valid_point<P: Into<CircuitPoint>>(&self, point: P) -> bool {
    //     let point = point.into();
    //     point.cycle < self.cycles.len()
    //         && match point.dit_id {
    //             CircuitDitId::Quantum(q) => q < self.num_qudits,
    //             CircuitDitId::Classical(c) => c < self.num_dits,
    //         }
    // }

    pub fn is_valid_id<P: Into<InstructionId>>(&self, _inst_id: P) -> bool {
        todo!()
        // let inst_id = inst_id.into();
        // let valid_dit = match inst_id.dit() {
        //     CircuitDitId::Quantum(dit) => dit < self.num_qudits(),
        //     CircuitDitId::Classical(dit) => dit < self.num_dits(),
        // };
        // self.cycles.is_id(inst_id.cycle()) && valid_dit
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
    pub fn find_available_cycle(&self, wires: &WireList) -> Option<CycleIndex> {
        if !self.is_valid_wires(&wires) { // TODO: cleanup ToLocation; a lot of unnecessary
            // copies/allocations in statements like this; think through Into/From/AsRef maybe?
            //
            // TODO: Really, panic over this? This it totally a recoverable error.
            // Cmon, you know better now. Get some proper error handling going already.
            panic!("Cannot find available cycle for invalid location.");
        }

        if self.cycles.is_empty() {
            return None;
        }

        let last_occupied_cycle_option = wires
            .wires()
            .filter_map(|w| self.rear.get(&w))
            .map(|cycle_id| self.cycles.id_to_index(*cycle_id))
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
    fn find_available_or_append_cycle(&mut self, wires: &WireList) -> CycleIndex {
        // Location validity implicitly checked in find_available_cycle
        if let Some(cycle_index) = self.find_available_cycle(wires) {
            cycle_index
        } else {
            self.cycles.push()
        }
    }

    // TODO: prepend + insert
    // TODO: param_indices_mapping

    fn _append_ref(&mut self, op_code: OpCode, wires: WireList, params: ParamIndices) {
        // TODO: check valid operation for radix match, measurement bandwidth etc
        
        // TODO: check params is valid: length is equal to op_params, existing exist, etc..
        // TODO: have to something about static entries...
        // TODO: have to do something about gate parameters mapped within same gate

        // Find cycle placement (location validity implicitly checked here)
        let cycle_index = self.find_available_or_append_cycle(&wires);
        let cycle_id = self.cycles.index_to_id(cycle_index);

        // Update counters
        self.inc_op_counter(&op_code);
        // self.inc_graph_counter(&location);

        // Update quantum DAG info
        for wire in wires.wires() {
            if let Some(&rear_cycle_id) = self.rear.get(&wire) {
                self.cycles.get_mut_from_id(rear_cycle_id).set_next(wire, cycle_id);
                self.cycles[cycle_index].set_prev(wire, rear_cycle_id);
            } else {
                // If rear is none, nothing exists on this wire, so update front too.
                self.front.insert(wire, cycle_id);
            }
            self.rear.insert(wire, cycle_id);
        }

        // Build instruction reference
        let inst_ref = Instruction::new(op_code, wires, params);

        // Add op to cycle
        self.cycles[cycle_index].push(inst_ref);
    }

    /// Intern an operation in the circuit's operation cache
    ///
    /// This allows further additions by OpCodes.
    pub fn cache_operation<O: Into<Operation>>(&mut self, op: O) -> OpCode {
        self.operations.insert(op.into())
    }

    /// Append an operation to the circuit
    pub fn append<O, W, A>(&mut self, op: O, wires: W, args: A) -> InstructionId
    where
        O: Into<Operation>,
        W: Into<WireList>,
        A: TryInto<Option<ArgumentList>>,
    {
        let op = op.into();
        let args = match args.try_into() {
            Err(_) => panic!("Get some proper error handling going already..."),
            Ok(Some(args)) => args,
            Ok(None) => ArgumentList::new(vec![ParameterEntry::Unspecified; op.num_params()]),
        };
        // let args: ArgumentList = if args.is_none() {
        //     ArgumentList::new(vec![ParameterEntry::Unspecified; op.num_params()])
        // } else {
        //     match args.unwrap().try_into() {
        //         Err(_) => panic!("Get some proper error handling going already..."),
        //         Ok(args) => args,
        //     }
        // };
        
        match op {
            Operation::Expression(e) => self.append_expression(e, wires, args),
            Operation::Subcircuit(s) => self.append_subcircuit(s, wires, args),
            Operation::Directive(d) => self.append_directive(d, wires, args),
        }
    }

    /// Append an operation already interned by the circuit provided by an OpCode
    pub fn append_by_code<W, A>(&mut self, op: OpCode, wires: W, args: A) -> InstructionId
    where
        W: Into<WireList>,
        A: TryInto<ArgumentList>,
    {
        let args: ArgumentList = match args.try_into() {
            Err(_) => panic!("Get some proper error handling going already..."),
            Ok(args) => args,
        };

        if args.requires_expression_modification() {
            todo!()
        }

        let param_ids = self.params.parse(&args); // persistent ids; not indices
        let wires = wires.into();
        self._append_ref(op, wires, param_ids);
        InstructionId::new(CycleId(0), InstId::null())
    }

    /// Append an expression to the circuit
    pub fn append_expression<O, L, P>(&mut self, op: O, loc: L, params: P) -> InstructionId
    where
        O: Into<ExpressionOperation>,
        L: Into<WireList>,
        P: Into<ArgumentList>,
    {
        let op: ExpressionOperation = op.into();
        let loc: WireList = loc.into();
        let args: ArgumentList = params.into();

        let param_ids = self.params.parse(&args); // persistent ids; not indices

        // Modify expression with new parameter expressions
        let new_variables = args.variables();
        let expressions = args.expressions();

        // Substitute params into op
        let subbed_op = match op {
            ExpressionOperation::UnitaryGate(e) => {
                let e: TensorExpression = e.into();
                let subbed_expr: UnitaryExpression = e.substitute_parameters(&new_variables, &expressions).try_into().unwrap();
                ExpressionOperation::UnitaryGate(subbed_expr)
            }
            ExpressionOperation::KrausOperators(e) => {
                let e: TensorExpression = e.into();
                let subbed_expr: KrausOperatorsExpression = e.substitute_parameters(&new_variables, &expressions).try_into().unwrap();
                ExpressionOperation::KrausOperators(subbed_expr)
            }
            ExpressionOperation::TerminatingMeasurement(e) => {
                let e: TensorExpression = e.into();
                let subbed_expr: BraSystemExpression = e.substitute_parameters(&new_variables, &expressions).try_into().unwrap();
                ExpressionOperation::TerminatingMeasurement(subbed_expr)
            }
            ExpressionOperation::ClassicallyControlledUnitary(e) => {
                let e: TensorExpression = e.into();
                let subbed_expr: UnitarySystemExpression = e.substitute_parameters(&new_variables, &expressions).try_into().unwrap();
                ExpressionOperation::ClassicallyControlledUnitary(subbed_expr)
            }
            ExpressionOperation::QuditInitialization(e) => {
                let e: TensorExpression = e.into();
                let subbed_expr: KetExpression = e.substitute_parameters(&new_variables, &expressions).try_into().unwrap();
                ExpressionOperation::QuditInitialization(subbed_expr)
            }
        };

        let op_code = self.operations.insert_expression_with_dits(subbed_op, &loc.dits().map(|d| self.dit_radices[d].into()).collect::<Vec<_>>());
        self._append_ref(op_code, loc, param_ids);
        InstructionId::new(CycleId(0), InstId::null())
    }

    /// Append a subcircuit to the circuit
    pub fn append_subcircuit<L, P>(&mut self, _op: CircuitOperation, loc: L, params: P) -> InstructionId
    where
        L: Into<WireList>,
        P: Into<ArgumentList>,
    {
        let _loc = loc.into();
        let _params = params.into();

        todo!()
    }

    /// Append a circuit directive to the circuit
    pub fn append_directive<L, P>(&mut self, _op: DirectiveOperation, loc: L, params: P) -> InstructionId
    where
        L: Into<WireList>,
        P: Into<ArgumentList>,
    {
        let _loc = loc.into();
        let _params = params.into();

        todo!()
    }


    /// Checks if a qudit is inactive
    pub fn is_qudit_inactive(&self, index: usize) -> bool {
        self.front.get(&Wire::quantum(index)).is_none()
    }
    
    /// Initialize the qudits specified in a zero state
    pub fn zero_initialize<W: Into<WireList>>(&mut self, wires: W) {
        let wires = wires.into();
        let location_radices = wires.qudits().map(|q| self.qudit_radices[q]).collect::<Radices>();
        let state = KetExpression::zero(location_radices);
        let op = ExpressionOperation::QuditInitialization(state);
        self.append(op, wires, None::<ArgumentList>);
    }

    /// Remove the operation at `point` from the circuit
    pub fn remove(&mut self, _inst_id: InstructionId) {
        todo!()
        //
        // Need to make sure I remove parameters as well; may need to reference count them
        // and potentially update every instruction reference
        //
        //
        // if !self.is_valid_id(inst_id) {
        //     // Don't need to panic here... TODO
        //     panic!("Cannot remove instruction with invalid id.");
        // }

        // let location = match self.cycles.get_inst_id( {
        //     Some(location) => location.clone(),
        //     // TODO: Error handling
        //     None => panic!("Operation not found at {} in cycle {}", point.dit_id, point.cycle),
        // };

        // // Update circuit quantum DAG info
        // for qudit_index in location.qudits() {
        //     let qnext = self.cycles[point.cycle].get_qnext(qudit_index);
        //     let qprev = self.cycles[point.cycle].get_qprev(qudit_index);

        //     match (qnext, qprev) {
        //         (Some(next_cycle_index), Some(prev_cycle_index)) => {
        //             self.cycles[next_cycle_index].set_qprev(qudit_index, prev_cycle_index);
        //             self.cycles[prev_cycle_index].set_qnext(qudit_index, next_cycle_index);
        //         },
        //         (Some(next_cycle_index), None) => {
        //             self.cycles[next_cycle_index].reset_qprev(qudit_index);
        //             self.qfront[qudit_index] = Some(next_cycle_index);
        //         },
        //         (None, Some(prev_cycle_index)) => {
        //             self.qrear[qudit_index] = Some(prev_cycle_index);
        //             self.cycles[prev_cycle_index].reset_qnext(qudit_index);
        //         },
        //         (None, None) => {
        //             self.qrear[qudit_index] = None;
        //             self.qfront[qudit_index] = None;
        //         },
        //     }
        // }

        // // Update circuit qudit DAG info
        // for clbit_index in location.dits() {
        //     let cnext = self.cycles[point.cycle].get_cnext(clbit_index);
        //     let cprev = self.cycles[point.cycle].get_cprev(clbit_index);

        //     match (cnext, cprev) {
        //         (Some(next_cycle_index), Some(prev_cycle_index)) => {
        //             self.cycles[next_cycle_index].set_cprev(clbit_index, prev_cycle_index);
        //             self.cycles[prev_cycle_index].set_cnext(clbit_index, next_cycle_index);
        //         },
        //         (Some(next_cycle_index), None) => {
        //             self.cycles[next_cycle_index].reset_cprev(clbit_index);
        //             self.cfront[clbit_index] = Some(next_cycle_index);
        //         },
        //         (None, Some(prev_cycle_index)) => {
        //             self.crear[clbit_index] = Some(prev_cycle_index);
        //             self.cycles[prev_cycle_index].reset_cnext(clbit_index);
        //         },
        //         (None, None) => {
        //             self.crear[clbit_index] = None;
        //             self.cfront[clbit_index] = None;
        //         },
        //     }
        // }

        // // Remove the instruction from the cycle
        // match self.cycles[point.cycle].remove(point.dit_id) {
        //     Some(inst_ref) => {
        //         // Update counters
        //         self.dec_graph_counter(&inst_ref.location);
        //         self.dec_inst_counter(&inst_ref.op);
        //     },
        //     // TODO: Error handling
        //     None => panic!("Operation not found at {} in cycle {}", point.dit_id, point.cycle),
        // }

        // // If cycle is empty, remove it
        // if self.cycles[point.cycle].is_empty() {
        //     self.cycles.remove(point.cycle);
        // }
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
    pub fn front(&self) -> HashMap<Wire, InstructionId> {
        todo!()
        // let quantum = self.qfront.iter().enumerate()
        //     .filter_map(|q| q.1.map(|cycle_id| {
        //         let q_id = CircuitDitId::quantum(q.0);
        //         (q_id, InstructionId::new(q_id, cycle_id))
        //     }));

        // let classical = self.cfront.iter().enumerate()
        //     .filter_map(|c| c.1.map(|cycle_id| { 
        //         let c_id = CircuitDitId::classical(c.0);
        //         (c_id, InstructionId::new(c_id, cycle_id))
        //     }));

        // quantum.chain(classical).collect()
    }

    /// Distill the circuit rear nodes into a hashmap.
    ///
    /// See [`QuditCircuit::front`] for more information.
    pub fn rear(&self) -> HashMap<Wire, InstructionId> {
        todo!()
        // let quantum = self.qrear.iter().enumerate()
        //     .filter_map(|q| q.1.map(|cycle_id| {
        //         let q_id = CircuitDitId::quantum(q.0);
        //         (q_id, InstructionId::new(q_id, cycle_id))
        //     }));

        // let classical = self.crear.iter().enumerate()
        //     .filter_map(|c| c.1.map(|cycle_id| { 
        //         let c_id = CircuitDitId::classical(c.0);
        //         (c_id, InstructionId::new(c_id, cycle_id))
        //     }));

        // quantum.chain(classical).collect()
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
    pub fn next(&self, _inst_id: InstructionId) -> HashMap<Wire, InstructionId> {
        todo!()
        // let cycle = self.cycles.get_from_id(inst_id.cycle());
        // let location = cycle.get_location_from_id(inst_id.inner());

        // match location {
        //     Some(location) => { 
        //         let quantum = location.qudits().iter().filter_map(|q_idx|
        //             cycle.get_qnext(q_idx).map(|next_cycle_id| {
        //                 let q_id = CircuitDitId::quantum(q_idx);
        //                 (q_id, InstructionId::new(q_id, next_cycle_id))
        //             })
        //         );

        //         let classical = location.dits().iter().filter_map(|c_idx|
        //             cycle.get_cnext(c_idx).map(|next_cycle_id| {
        //                 let c_id = CircuitDitId::classical(c_idx);
        //                 (c_id, InstructionId::new(c_id, next_cycle_id))
        //             })
        //         );
        //         quantum.chain(classical).collect()
        //     },
        //     None => HashMap::new(),
        // }
    }

    /// Gather the points of the previous operations from the point of an
    /// operation.
    ///
    /// See [`QuditCircuit::next`] for more information.
    pub fn prev(&self, _inst_id: InstructionId) -> HashMap<Wire, InstructionId> {
        todo!()
        // let cycle = self.cycles.get_from_id(inst_id.cycle());
        // let location = cycle.get_location(inst_id.dit());

        // match location {
        //     Some(location) => { 
        //         let quantum = location.qudits().iter().filter_map(|q_idx|
        //             cycle.get_qprev(q_idx).map(|prev_cycle_id| {
        //                 let q_id = CircuitDitId::quantum(q_idx);
        //                 (q_id, InstructionId::new(q_id, prev_cycle_id))
        //             })
        //         );

        //         let classical = location.dits().iter().filter_map(|c_idx|
        //             cycle.get_cprev(c_idx).map(|prev_cycle_id| {
        //                 let c_id = CircuitDitId::classical(c_idx);
        //                 (c_id, InstructionId::new(c_id, prev_cycle_id))
        //             })
        //         );
        //         quantum.chain(classical).collect()
        //     },
        //     None => HashMap::new(),
        // }
    }

    /// Return an iterator over the operations in the circuit.
    ///
    /// The ordering is not guaranteed to be consistent, but it will
    /// be in a simulation/topological order. For more control over the
    /// ordering of iteration see [`QuditCircuit::iter_df`] or
    /// [`QuditCircuit::iter_bf`].
    pub fn iter(&self) -> impl Iterator<Item = &Instruction> + '_{
        self.cycles.iter().flat_map(|cycle| cycle.iter())
    }

    // /// Return a depth-first iterator over the operations in the circuit.
    // ///
    // /// See [`QuditCircuitDFIterator`] for more info.
    // pub fn iter_df(&self) -> QuditCircuitDFIterator {
    //     QuditCircuitDFIterator::new(self)
    // }

    // /// Return a breadth-first iterator over the operations in the circuit.
    // ///
    // /// See [`QuditCircuitBFIterator`] for more info.
    // pub fn iter_bf(&self) -> QuditCircuitBFIterator {
    //     QuditCircuitBFIterator::new(self)
    // }

//     /// Return an iterator over the operations in the circuit with cycles.
//     pub fn iter_with_cycles(&self) -> QuditCircuitFastIteratorWithCycles<C> {
//         QuditCircuitFastIteratorWithCycles::new(self)
//     }

    /// Calculate the Kraus Operators that describe this circuit as a program
    pub fn kraus_ops<C: ComplexScalar>(&self, args: &[C::R]) -> Tensor<C, 3> {
        let network = self.to_tensor_network();
        let code = qudit_tensor::compile_network(network);
        let mut tnvm = qudit_tensor::TNVM::<C, FUNCTION>::new(&code, Some(&self.params.const_map()));
        let result = tnvm.evaluate::<FUNCTION>(args);
        result.get_fn_result2().unpack_tensor3d().to_owned()
    }

    /// Convert the circuit to a symbolic tensor network
    pub fn to_tensor_network(&self) -> QuditTensorNetwork {
        self.as_tensor_network_builder().build()
    }

    /// Convert the circuit to a tensor network builder
    pub fn as_tensor_network_builder(&self) -> QuditCircuitTensorNetworkBuilder {
        let mut network = QuditCircuitTensorNetworkBuilder::new(self.qudit_radices(), Some(self.operations.expressions()));

        for inst in self.iter() {
            if inst.op_code().kind() == OpKind::Expression {
                let indices = self.operations.indices(inst.op_code());
                let param_indices = self.params.convert_ids_to_indices(inst.params());
                let constant = param_indices.iter().map(|i| self.params[i].is_constant()).collect();
                let param_info = ParamInfo::new(param_indices, constant); 
                let input_index_map = if indices.iter().any(|idx| idx.direction() == IndexDirection::Input && idx.index_size() > 1) { inst.wires().qudits().collect() } else { vec![] };
                let output_index_map = if indices.iter().any(|idx| idx.direction() == IndexDirection::Output && idx.index_size() > 1) { inst.wires().qudits().collect() } else { vec![] };
                let batch_index_map: Vec<String> = inst.wires().dits()
                    .map(|id| id.to_string())
                    .collect();
                let tensor = QuditTensor::new(indices, inst.op_code().id(), param_info);
                // println!("Adding new tensor to network builder with in qudits: {:?}; out qudits: {:?}, batch indices: {:?}", input_index_map.clone(), output_index_map.clone(), batch_index_map.clone());
                network = network.prepend(tensor, input_index_map, output_index_map, batch_index_map)
            }
        }
        network
    }
}

impl QuditSystem for QuditCircuit {
    #[inline(always)]
    fn num_qudits(&self) -> usize {
        self.qudit_radices.num_qudits()
    }

    #[inline(always)]
    fn dimension(&self) -> usize {
        self.qudit_radices.dimension()
    }

    fn radices(&self) -> Radices {
        self.qudit_radices.clone()
    }
}

impl HasParams for QuditCircuit {
    #[inline(always)]
    fn num_params(&self) -> usize {
        self.params.len()
    }
}

impl ClassicalSystem for QuditCircuit {
    fn radices(&self) -> Radices {
        self.dit_radices.clone()
    }
    
    #[inline(always)]
    fn num_dits(&self) -> usize {
        self.dit_radices.num_qudits()
    }
}

impl HybridSystem for QuditCircuit {}

#[cfg(test)]
mod tests {
    use core::f32;

    use super::*;
    use bit_set::BitSet;
    use faer::reborrow::ReborrowMut;
    use qudit_core::c32;
    use qudit_core::c64;
    use qudit_core::radices;
    use qudit_core::QuditRadices;
    use qudit_expr::DifferentiationLevel;
    use qudit_tensor::Bytecode;
    // use crate::CircuitLocation;
    use qudit_core::unitary::UnitaryMatrix;
    use qudit_core::unitary::UnitaryFn;
    use qudit_expr::{FUNCTION, GRADIENT};
    use qudit_expr::ExpressionGenerator;

    pub fn different_ways_to_add_an_operation() -> QuditCircuit {
        let mut circ = QuditCircuit::new([2, 2, 3], [2, 2, 3]);
        
        // // Append a parameterized operation: this will append a fully parameterized expression to
        // // the circuit at the location specified
        // circ.append_parameterized(Gate::U3(), 0); // Standard gate; ez; simple
        // circ.append_parameterized(BraSystemExpression::new("Z_meas() { [
        //     [1, 0, 0, 0],
        //     [0, 1, 0, 0],
        //     [0, 0, 1, 0],
        //     [0, 0, 0, 1],
        // }"), ((0, 1), (0, 1))); // Terminating Z Measurement on qudit 0, 1
        // circ.append_parameterized(KetExpression::new("Zero_init<2, 3>() { [1, 0, 0, 0, 0, 0] }") (0, 2)); // Initialize qudit 0 and 2 in zero state
        // circ.append_parameterized(UnitaryExpression::new("..."), 2); // unitary expression on qudit 2
        // circ.append_parameterized(KrausExpression::new("..."), 1); // Channel (Modelled as a measurement to environment; specified by no classical dits in location) on qudit 1
        // circ.append_parameterized(KrausExpression::new("..."), (2, 2)); // Measurement Channel (specified by classical dit in location) on qudit 2
        // circ.append_parameterized((UnitaryExpression::new("..."), "111, 002"), ([0, 1, 2], [0, 1, 2])); // Classically controlled operation


        // Append an operation with a specified set of parameters
        // circ.append_specified(Gate::U3(), 0, ["pi/4", "alpha", "e^(i*theta)"]);
        // circ.append_specified(Gate::U3(), 0, [0.75, 0.123, 0.1414]);
        // circ.append_specified(Gate::U3(), 0, ["alpha", "alpha", "alpha"]);

        circ
    }

    pub fn build_qsearch_thin_step_circuit(n: usize) -> QuditCircuit {
        let block_expr = Gate::U3().generate_expression().otimes(&Gate::U3().generate_expression()).dot(&Gate::CX().generate_expression());
        let mut circ = QuditCircuit::pure(vec![2; n]);
        for i in 0..n {
            circ.append_parameterized(Gate::U3(), [i]);
        }
        for _ in 0..n {
            for i in 0..(n - 1) {
                circ.append_parameterized(Gate::Expression(block_expr.clone()), [i, i+1]);
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

        // let mut circ: QuditCircuit = QuditCircuit::pure(radices![2; n]);
        // for i in 0..n {
        //     circ.append_parameterized(Gate::U3(), i);
        // }
        // circ.append_parameterized(Gate::U3(), 1);
        // let block_expr = Gate::U3().generate_expression().otimes(&Gate::U3().generate_expression()).dot(&Gate::CX().generate_expression());
        // circ.append_parameterized(block_expr, [0, 1]);
        // circ.append_parameterized(Gate::CX(), [0, 1]);
        // for i in 0..n {
        //     circ.append_parameterized(Gate::U3(), i);
        // }
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

        // let result = tnvm.evaluate::<GRADIENT>(&[1.7; (3*n) + (7*(n-1)*n)]);
        // let unitary = result.get_fn_result().unpack_matrix();
        // println!("{:?}", unitary);
    }

    // pub fn build_simple_dynamic_circuit() -> QuditCircuit {
    //     let mut circ: QuditCircuit = QuditCircuit::new([2, 2, 2, 2], [2, 2]);

    //     circ.zero_initialize([1, 2]);
    //     circ.append_parameterized(Gate::H(2), [1]);
    //     circ.append_parameterized(Gate::CX(), [1, 2]);
    //     circ.append_parameterized(Gate::CX(), [0, 1]);
    //     circ.append_parameterized(Gate::CX(), [2, 3]);
    //     circ.append_parameterized(Gate::H(2), [2]);

    //     let one_qubit_basis_measurement = BraSystemExpression::new("OneQMeasure() {
    //         [
    //             [[ 1, 0, ]],
    //             [[ 0, 1, ]],
    //         ]
    //     }");

    //     circ.append_parameterized(Operation::TerminatingMeasurement(one_qubit_basis_measurement.clone()), ([1], [0]));
    //     circ.append_parameterized(Operation::TerminatingMeasurement(one_qubit_basis_measurement), ([2], [1]));

    //     let u3_block_expr = Gate::U3().generate_expression().otimes(Gate::U3().generate_expression());
    //     circ.append_parameterized(Operation::ClassicallyControlledUnitary(u3_block_expr.classically_control(&[0], &[2, 2])), ([0, 3], [0, 1]));
    //     circ.append_parameterized(Operation::ClassicallyControlledUnitary(u3_block_expr.classically_control(&[1], &[2, 2])), ([0, 3], [0, 1]));
    //     circ.append_parameterized(Operation::ClassicallyControlledUnitary(u3_block_expr.classically_control(&[2], &[2, 2])), ([0, 3], [0, 1]));
    //     circ.append_parameterized(Operation::ClassicallyControlledUnitary(u3_block_expr.classically_control(&[3], &[2, 2])), ([0, 3], [0, 1]));
    
    //     circ
    // }

    // #[test]
    // fn dynamic_circuit_tnvm_test() {
    //     let circ = build_simple_dynamic_circuit();
    //     dbg!(circ.num_params());

    //     let network = circ.to_tensor_network();
    //     let code = qudit_tensor::compile_network(network);
    //     println!("{:?}", code);

    //     let mut tnvm = qudit_tensor::TNVM::<c64, FUNCTION>::new(&code);
    //     let result = tnvm.evaluate::<FUNCTION>(&vec![std::f64::consts::FRAC_PI_2; circ.num_params()]);
    //     let unitary = result.get_fn_result().unpack_tensor3d();
    //     println!("{:?}", unitary);
    // }

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

    // #[test]
    // fn simple_instantiation_structure_test() {
    //     let mut circ: QuditCircuit<c32> = QuditCircuit::new([2], [2, 2, 2]);
    //     circ.append_uninit_gate(Gate::U3(), [0]);
    //     let mut builder = circ.as_tensor_network_builder();
    //     builder = builder.prepend_unitary(UnitaryMatrix::<c32>::identity(radices![2]), vec![0]);
    //     let network = builder.build();
    //     let code = qudit_tensor::compile_network(network);
    //     let mut tnvm = qudit_tensor::TNVM::<c32, GRADIENT>::new(&code);
    //     let result = tnvm.evaluate::<GRADIENT>(&[1.7; 3]);
    //     let fn_out = result.get_fn_result().unpack_matrix();
    //     let grad_out = result.get_grad_result().unpack_tensor3d();
    // }

    // #[test]
    // fn dynamic_circuit_test2() {
    //     let mut circ: QuditCircuit<c32> = QuditCircuit::new([2, 2, 2, 2], [2, 2]);

    //     circ.zero_initialize([1, 2]);

    //     for i in 0..4 {
    //         circ.append_uninit_gate(Gate::U3(), [i]);
    //     }

    //     let block_expr = Gate::U3().gen_expr().otimes(Gate::U3().gen_expr()).dot(Gate::CX().gen_expr());
    //     circ.append_instruction(Instruction::new(Operation::Gate(Gate::Expression(block_expr.clone())), [0, 1], vec![]));
    //     circ.append_instruction(Instruction::new(Operation::Gate(Gate::Expression(block_expr.clone())), [2, 3], vec![]));
    //     circ.append_instruction(Instruction::new(Operation::Gate(Gate::Expression(block_expr.clone())), [1, 2], vec![]));

    //     let two_qubit_basis_measurement = StateSystemExpression::new("TwoQMeasure() {
    //         [
    //             [[ 1, 0, 0, 0 ]],
    //             [[ 0, 1, 0, 0 ]],
    //             [[ 0, 0, 1, 0 ]],
    //             [[ 0, 0, 0, 1 ]],
    //         ]
    //     }");
    //     circ.append_instruction(Instruction::new(Operation::TerminatingMeasurement(two_qubit_basis_measurement), ([1, 2], [0,1]), vec![]));

    //     let cs1 = ControlState::from_binary_state([0,0]);
    //     circ.uninit_classically_control(Gate::U3(), cs1.clone(), ([0], [0, 1]));
    //     circ.uninit_classically_control(Gate::U3(), cs1.clone(), ([3], [0, 1]));

    //     let cs2 = ControlState::from_binary_state([0,1]);
    //     circ.uninit_classically_control(Gate::U3(), cs2.clone(), ([0], [0, 1]));
    //     circ.uninit_classically_control(Gate::U3(), cs2.clone(), ([3], [0, 1]));

    //     let cs3 = ControlState::from_binary_state([1,0]);
    //     circ.uninit_classically_control(Gate::U3(), cs3.clone(), ([0], [0, 1]));
    //     circ.uninit_classically_control(Gate::U3(), cs3.clone(), ([3], [0, 1]));

    //     let cs4 = ControlState::from_binary_state([1,1]);
    //     circ.uninit_classically_control(Gate::U3(), cs4.clone(), ([0], [0, 1]));
    //     circ.uninit_classically_control(Gate::U3(), cs4.clone(), ([3], [0, 1]));
        
    //     let network = circ.to_tensor_network();
    //     let code = qudit_tensor::compile_network(network);
    //     let mut tnvm = qudit_tensor::TNVM::<c32, GRADIENT>::new(&code);
    //     let start = std::time::Instant::now();
    //     for _ in 0..1000 {
    //         let result = tnvm.evaluate::<GRADIENT>(&vec![1.7; circ.num_params()]);
    //     }
    //     let elapsed = start.elapsed();
    //     println!("Time per evaluation: {:?}", elapsed / 1000);
    // }

    // #[test]
    // fn dynamic_circuit_test() {
    //     let mut circ: QuditCircuit<c32> = QuditCircuit::new([2, 2, 2, 2], [2, 2]);

    //     for i in 0..4 {
    //         circ.append_uninit_gate(Gate::U3(), [i]);
    //     }

    //     let block_expr = Gate::U3().gen_expr().otimes(Gate::U3().gen_expr()).dot(Gate::CX().gen_expr());

    //     circ.append_instruction(Instruction::new(Operation::Gate(Gate::Expression(block_expr.clone())), [0, 1], vec![]));
    //     circ.append_instruction(Instruction::new(Operation::Gate(Gate::Expression(block_expr.clone())), [2, 3], vec![]));
    //     circ.append_instruction(Instruction::new(Operation::Gate(Gate::Expression(block_expr.clone())), [1, 2], vec![]));
    //     // circ.append_instruction(Instruction::new(Operation::Gate(Gate::Expression(block_expr.clone())), loc![3, 4], vec![]));

    //     // let clbits = BitSet::from_bit_vec(&[0b11]);
    //     let mut clbits = BitSet::new();
    //     clbits.insert(0);
    //     clbits.insert(1);
    //     // let two_qubit_basis_measurement = StateSystemExpression::new("ThreeQMeasure() {
    //     //     [
    //     //         [[ 1, 0, 0, 0, 0, 0, 0, 0 ]],
    //     //         [[ 0, 1, 0, 0, 0, 0, 0, 0 ]],
    //     //         [[ 0, 0, 1, 0, 0, 0, 0, 0 ]],
    //     //         [[ 0, 0, 0, 1, 0, 0, 0, 0 ]],
    //     //         [[ 0, 0, 0, 0, 1, 0, 0, 0 ]],
    //     //         [[ 0, 0, 0, 0, 0, 1, 0, 0 ]],
    //     //         [[ 0, 0, 0, 0, 0, 0, 1, 0 ]],
    //     //         [[ 0, 0, 0, 0, 0, 0, 0, 1 ]],
    //     //     ]
    //     // }");
    //     let two_qubit_basis_measurement = StateSystemExpression::new("ThreeQMeasure() {
    //         [
    //             [[ 1, 0, 0, 0 ]],
    //             [[ 0, 1, 0, 0 ]],
    //             [[ 0, 0, 1, 0 ]],
    //             [[ 0, 0, 0, 1 ]],
    //         ]
    //     }");
    //     circ.append_instruction(Instruction::new(Operation::TerminatingMeasurement(two_qubit_basis_measurement), ([1, 2], [0,1]), vec![]));

    //     // circ.append_instruction(Instruction::new(Operation::ClassicallyControlled(Gate::U3(), clbits.clone()), [0], vec![]));
    //     // circ.append_instruction(Instruction::new(Operation::ClassicallyControlled(Gate::U3(), clbits.clone()), [3], vec![]));
    //     // circ.append_instruction(Instruction::new(Operation::ClassicallyControlled(Gate::U3(), clbits), loc![0, 3], vec![]));

    //     let network = circ.to_tensor_network();
    //     let code = qudit_tensor::compile_network(network);
    //     let mut tnvm = qudit_tensor::TNVM::<c32, GRADIENT>::new(&code);
    //     let start = std::time::Instant::now();
    //     for _ in 0..1000 {
    //         let result = tnvm.evaluate::<GRADIENT>(&[1.7; 36]);
    //     }
    //     let elapsed = start.elapsed();
    //     println!("Time per evaluation: {:?}", elapsed / 1000);
    // }

    // #[test]
    // fn test_param_overlapping_cirucit() {
    //     let mut circ: QuditCircuit<c64> = QuditCircuit::pure([2, 2]);
    //     circ.append_uninit_gate(Gate::CX(), [0, 1]);
    //     circ.append_uninit_gate(Gate::P(2), [1]);
    //     circ.append_uninit_gate(Gate::CX(), [0, 1]);
    //     circ.append_gate(Gate::P(2), [1], [ParamEntry::Existing(0)]);
    //     circ.append_uninit_gate(Gate::CX(), [0, 1]);

    //     // optional: circ.new_parameteric_param("alpha");
    //     // optional: circ.set_parameter_value("alpha", 0.5);
    //     //  - if parameteric, makes it dynamic
    //     //  - sets value
    //     // optional: circ.freeze_parameter_value("alpha", 0.5);
    //     //  - if parameteric, makes it static
    //     //  - sets value
    //     // circ.append_gate(Gate::P(2), [1], ["alpha"]);
    //     // circ.append_gate(Gate::P(2), [1], ["alpha/2"]);

    //     let network = circ.to_tensor_network();
    //     let code = qudit_tensor::compile_network(network);
    //     let mut tnvm = qudit_tensor::TNVM::<c64, GRADIENT>::new(&code);
    //     let start = std::time::Instant::now();
    //     for _ in 0..1000 {
    //         let result = tnvm.evaluate::<GRADIENT>(&[1.7]);
    //     }
    //     let result = tnvm.evaluate::<GRADIENT>(&[1.7]);
    //     dbg!(result.get_grad_result().unpack_tensor3d());
    //     let elapsed = start.elapsed();
    //     println!("Time per evaluation: {:?}", elapsed / 1000);
    // }

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

    #[test]
    fn test_gen_code_for_paper() {
        let mut circ = QuditCircuit::pure(vec![2; 3]);
        circ.append_parameterized(Gate::U3(), [0]);
        circ.append_parameterized(Gate::U3(), [1]);
        circ.append_parameterized(Gate::U3(), [2]);
        circ.append_parameterized(Gate::CX(), [1, 2]);
        circ.append_parameterized(Gate::CX(), [0, 1]);
        circ.append_parameterized(Gate::U3(), [0]);
        circ.append_parameterized(Gate::CX(), [0, 1]);
        circ.append_parameterized(Gate::CX(), [1, 2]);
        circ.append_parameterized(Gate::U3(), [0]);
        circ.append_parameterized(Gate::U3(), [1]);
        circ.append_parameterized(Gate::U3(), [2]);
        let code = qudit_tensor::compile_network(circ.to_tensor_network());
        println!("{:?}", code);
    }

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

#[cfg(feature = "python")]
mod python {
    use super::*;
    use crate::python::PyCircuitRegistrar;
    use pyo3::prelude::*;
    use pyo3::exceptions::PyTypeError;
    use pyo3::types::PyTuple;
    use qudit_core::c64;
    use numpy::PyArray3;
    use numpy::PyArrayMethods;
    use ndarray::ArrayViewMut3;

    /// Helper function to parse a Python object that can be
    /// either an integer or an iterable of integers.
    fn parse_int_or_iterable<'py>(input: &Bound<'py, PyAny>) -> PyResult<Vec<usize>> {
        // First, try to extract the input as a single integer.
        if let Ok(val) = input.extract::<usize>() {
            // If successful, create a Vec of that length filled with the value 2.
            Ok(vec![2; val])
        } else {
            // If it's not an integer, try to treat it as an iterable.
            // This will raise a TypeError if the object is not iterable
            // or its elements are not integers.
            pyo3::types::PyIterator::from_object(input)?
                .map(|item| item?.extract::<usize>())
                .collect::<PyResult<Vec<usize>>>()
        }
    }

    #[pyclass]
    #[pyo3(name = "QuditCircuit")]
    pub struct PyQuditCircuit {
        circuit: QuditCircuit
    }

    #[pymethods]
    impl PyQuditCircuit {
        /// Creates a new QuditCircuit instance.
        ///
        /// Args:
        ///     qudits (int | Iterable[int]): An integer specifying number of qudits 
        ///         or an iterable of qudit radices.
        ///     dits (int | Iterable[int] | None): An integer, an iterable, or None.
        ///         Defaults to None, which results in no classical dits.
        #[new]
        #[pyo3(signature = (qudits, dits = None))]
        fn new<'py>(qudits: &Bound<'py, PyAny>, dits: Option<&Bound<'py, PyAny>>) -> PyResult<PyQuditCircuit> {
            let qudits_vec = parse_int_or_iterable(qudits)?;

            let dits_vec = match dits {
                Some(pyany) => parse_int_or_iterable(pyany)?,
                None => Vec::new(),
            };

            Ok(PyQuditCircuit {
                circuit: QuditCircuit::new(qudits_vec, dits_vec),
            })
        }

        // --- Properties ---

        /// Returns the number of parameters in the circuit.
        #[getter]
        fn num_params(&self) -> PyResult<usize> {
            Ok(self.circuit.num_params())
        }

        /// Returns the number of operations in the circuit.
        #[getter]
        fn num_operations(&self) -> PyResult<usize> {
            Ok(self.circuit.num_operations())
        }

        /// Returns the number of cycles in the circuit.
        #[getter]
        fn num_cycles(&self) -> PyResult<usize> {
            Ok(self.circuit.num_cycles())
        }

        #[getter]
        fn is_empty(&self) -> PyResult<bool> {
            Ok(self.circuit.is_empty())
        }

        #[getter]
        fn active_qudits(&self) -> PyResult<Vec<usize>> {
            Ok(self.circuit.active_qudits())
        }

        #[getter]
        fn active_dits(&self) -> PyResult<Vec<usize>> {
            Ok(self.circuit.active_dits())
        }

        // @property def params(self) -> Parameters?
        // @property def coupling_graph(self) -> CouplingGraph?
        // @property def gate_set(self) -> GateSet?

        // Metrics
        //
        // def depth(self, *, filter: Optional[Callable] = None, recursive: bool = True) -> int:
        // def parallelism(self) -> float:
        // def gate_counts(self, *, filter: Optional[Callable] = None, recursive: bool = True) -> int:
        //
        // Qudit Methods
        //
        // def append_qudit(self, radix: int = 2) -> None:
        // def extend_qudits(self, radixes: Iterable[int]) -> None:
        // def insert_qudit(self, qudit_index: int, radix: int = 2) -> None:
        // def pop_qudit(self, qudit_index: int) -> None:
        // def is_qudit_in_range(self, qudit_index: int) -> bool:
        // def is_qudit_idle(self, qudit_index: int) -> bool:
        // def renumber_qudits(self, qudit_permutation: Iterable[int]) -> None:
        //
        // Cycle Methods
        //
        // def pop_cycle(self, cycle_index: int) -> None?
        //
        // DAG Methods:
        //
        // @property def front(self) -> set[CircuitPoint]:
        // @property def rear(self) -> set[CircuitPoint]:
        // @property def first_on(self) -> CircuitPoint | None:
        // @property def last_on(self) -> CircuitPoint | None:
        // def next(self, current: CircuitPointLike | CircuitRegionLike, /) -> set[CircuitPoint]:
        // def prev(self, current: CircuitPointLike | CircuitRegionLike, /) -> set[CircuitPoint]:
        //
        // At Methods:
        //
        // def get_operation(self, point: CircuitPointLike) -> Operation:
        // 
        // def point(
        //     self,
        //     op: Operation | Gate,
        //     start: CircuitPointLike = (0, 0),
        //     end: CircuitPointLike | None = None,
        // ) -> CircuitPoint:
        //
        // def append(self, op: Operation) -> int:

        /// Returns the Kraus operators of the circuit as a NumPy array.
        #[pyo3(signature = (args = None))]
        pub fn kraus_ops<'py>(
            &self,
            py: Python<'py>,
            args: Option<&Bound<'py, PyAny>>,
        ) -> PyResult<Bound<'py, PyArray3<c64>>> {
            let rust_args: Vec<f64> = match args {
                Some(py_args) => py_args.extract()?,
                None => {
                    if self.circuit.num_params() != 0 {
                        return Err(PyTypeError::new_err(
                            "Circuit has parameters, but no arguments were provided to kraus_ops."
                        ));
                    }
                    Vec::new()
                }
            };

            // Call the underlying Rust method
            let tensor: Tensor<c64, 3> = self.circuit.kraus_ops(&rust_args);
            let shape = tensor.dims();

            let py_array: Bound<'py, PyArray3<c64>> = PyArray3::zeros(py, shape.clone(), false);

            {
                let mut readwrite = py_array.readwrite();
                let mut py_array_view: ArrayViewMut3<c64> = readwrite.as_array_mut();

                for k in 0..shape[0] {
                    let kraus_op = tensor.subtensor_ref(k);
                    for (j, col) in kraus_op.col_iter().enumerate() {
                        for (i, val) in col.iter().enumerate() {
                            py_array_view[[k, i, j]] = *val;
                        }
                    }
                }
            }

            Ok(py_array)
        }

        #[pyo3(signature = (op, loc, args = None))]
        pub fn append<'py>(
            &mut self,
            op: Operation,
            loc: &Bound<'py, PyAny>,
            args: Option<ArgumentList>,
        ) -> PyResult<()> {
            let num_qudits = op.num_qudits();

            // 2. Parse 'loc' as an int, iterable of ints, or tuple of iterables
            let parsed_loc = if let Ok(single_loc) = loc.extract::<usize>() {
                WireList::pure(&[single_loc])
            } else if let Ok(tuple) = loc.downcast::<PyTuple>() {
                if tuple.len() == 2 {
                    let item0 = tuple.get_item(0)?;
                    let item1 = tuple.get_item(1)?;

                    // Try parsing as (iterable, iterable)
                    if let (Ok(vec0), Ok(vec1)) = (item0.extract::<Vec<usize>>(), item1.extract::<Vec<usize>>()) {
                        WireList::new(vec0, vec1)
                    } 
                    // Else, try parsing as (int, int)
                    else if let (Ok(int0), Ok(int1)) = (item0.extract::<usize>(), item1.extract::<usize>()) {
                        if num_qudits.is_some() && num_qudits == Some(1) {
                            WireList::new(vec![int0], vec![int1])
                        } else {
                            WireList::pure(&[int0, int1])
                        }
                    } else {
                        return Err(PyTypeError::new_err("A 2-element 'loc' tuple must contain (int, int) or (iterable, iterable)"));
                    }
                } else {
                    let mut qudit_register = tuple.extract::<Vec<usize>>()?;
                    let dit_register = qudit_register.split_off(num_qudits.unwrap());
                    WireList::new(qudit_register, dit_register)
                }
            } else if let Ok(list_loc) = loc.extract::<Vec<usize>>() {
                let mut qudit_register = list_loc.clone();
                let dit_register = qudit_register.split_off(num_qudits.unwrap());
                WireList::new(qudit_register, dit_register)
            } else {
                return Err(PyTypeError::new_err(
                    "Argument 'loc' must be an int, an iterable of ints, or a tuple of two iterables of ints",
                ));
            };

            self.circuit.append(op, parsed_loc, args);
            Ok(())
        }

        

        // def append_gate
        // def append_circuit
        // def extend
        // def insert
        // def insert_gate
        // def insert_circuit
        // def remove(op)
        // def remove_all
        // def count
        // def pop(point)
        // def batch_pop(points)
        // def replace(point, op)
        // def replace_all(op, op)
        // def batch_replace(points, ops)
        // def replace_gate
        // def replace_with_circuit(... as circuit_gate = False)
        //
        // But like also, initializations, barriers, measurements, kraus ops, classical controls
        //
        // Movement/Ownership
        //
        // def copy()
        // def become()
        // def clear()
        //
        // Parameter Methods?
        //
        // def un-constant?
        // def freeze
        // Specified vs Constant appends are funky, A user should edit their expression
        // before hand if they want to hardcode a constant into it, otherwise, circuits
        // should track constant parameters, whether they are added as specified or const.
        //
        // User's should have ability to safely `def freeze(self, parameter_index: int, value: Constant)` and
        // `def thaw(self, parameter_index: int)`
        //
        // For this, might need to add a new type of Parameter => Frozen({value: Constant, name:
        // Option<String>}), when a named parameter is frozen, the name is stored for future thaws
        // Also, if I add an expression with the same parameter, it should also be frozen.
        // 
        // Advanced Algorithms
        //
        // def compress()
        // def surround() // Need to seriously think about subcircuits/regions/grouping
        // def invert()
        // def evaluate(args)
        // def evaluate_gradient(args)
        // def evaluate_hessian(args)
        // def instantiate(...)
        //
        // dunder methods
        //
        // __getitem__
        // __iter__
        // __reversed__
        // __contains__
        // __len__
        // __invert__
        // __eq__
        // __ne__
        // __add__
        // __mul__
        // __radd__
        // __iadd__
        // __imul__
        // __str__
        // __repr__
        // operations/operations_with_cycles
        //
        // IO
        //
        // save
        // to
        // from_file
        // from_unitary
        // __reduce__
        // rebuild_circuit
    }

    impl<'py> IntoPyObject<'py> for QuditCircuit {
        type Target = <PyQuditCircuit as IntoPyObject<'py>>::Target;
        type Output = <PyQuditCircuit as IntoPyObject<'py>>::Output;
        type Error = <PyQuditCircuit as IntoPyObject<'py>>::Error;
        
        fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
            PyQuditCircuit::from(self).into_pyobject(py)
        }
    }

    impl<'py> FromPyObject<'py> for QuditCircuit {
        fn extract_bound(ob: &Bound<'py, PyAny>) -> PyResult<Self> {
            let py_circuit_ref = ob.downcast::<PyQuditCircuit>()?;
            Ok(py_circuit_ref.borrow().circuit.clone())
        }
    }

    impl From<QuditCircuit> for PyQuditCircuit {
        fn from(value: QuditCircuit) -> Self {
            PyQuditCircuit { circuit: value }
        }
    }

    impl From<PyQuditCircuit> for QuditCircuit {
        fn from(value: PyQuditCircuit) -> Self {
            value.circuit
        }
    }

    /// Registers the QuditCircuit class with the Python module.
    fn register(parent_module: &Bound<'_, PyModule>) -> PyResult<()> {
        parent_module.add_class::<PyQuditCircuit>()?;
        Ok(())
    }
    inventory::submit!(PyCircuitRegistrar { func: register });

}
