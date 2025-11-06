use crate::cycle::CycleList;
use crate::cycle::{CycleId, CycleIndex};
use crate::instruction::{Instruction, InstructionId};
use crate::operation::OpCode;
use crate::operation::OperationSet;
use crate::operation::{
    CircuitOperation, DirectiveOperation, ExpressionOperation, OpKind, Operation,
};
use crate::param::{Argument as ParameterEntry, ArgumentList, ParameterVector};
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
use rustc_hash::FxHashMap;
use std::collections::HashMap;

/// A quantum circuit that can be defined with qudits and classical bits.
///
/// The circuit is internally represented as a list of cycles, where each
/// cycles represents an abstract moment in time. In each cycle, instructions
/// can direct operations to be applied to quantum and/or classical wires.
/// This data structure holds invariant that there never will be an empty
/// cycle. As a result, cycles can be removed automatically from anywhere in
/// the circuit, when operations are removed. However, cycles are identified
/// by a persistant identifier, which never changes and enables fast O(1) lookup.
/// This additionally enables instruction identifiers that will always point
/// to their correct instruction, no matter how the circuit changes underneath.
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

    /// A pointer to the first operation on each wire.
    front: FxHashMap<Wire, CycleId>,

    /// A pointer to the last operation on each wire.
    rear: FxHashMap<Wire, CycleId>,
}

/// Constructors
impl QuditCircuit {
    /// Creates a new QuditCircuit object.
    ///
    /// # Arguments
    ///
    /// * `qudit_radices` - The radices that describes the qudit system.
    ///
    /// * `dit_radices` - The radices that describes the classical system.
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
    pub fn new<T1: Into<Radices>, T2: Into<Radices>>(
        qudit_radices: T1,
        dit_radices: T2,
    ) -> QuditCircuit {
        QuditCircuit::with_capacity(qudit_radices, dit_radices, 4)
    }

    /// Creates a new purely-quantum QuditCircuit object.
    ///
    /// # Arguments
    ///
    /// * `qudit_radices` - The radices that describes the qudit system.
    ///
    /// # Examples
    ///
    /// We can define purely quantum kernels without classical bits:
    /// ```
    /// # use qudit_circuit::QuditCircuit;
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
    /// * `qudit_radices` - The radices that describes the qudit system.
    ///
    /// * `dit_radices` - The radices that describes the classical system.
    ///
    /// * `capacity` - The number of cycles to pre-allocate.
    ///
    /// # Examples
    ///
    /// ```
    /// # use qudit_circuit::QuditCircuit;
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
            front: FxHashMap::default(),
            rear: FxHashMap::default(),
            operations: OperationSet::new(),
            params: ParameterVector::default(),
        }
    }
}

/// Properties
impl QuditCircuit {
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
        self.operations.num_operations()
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
    /// This method is O(w) where
    ///     - `w` is the number of wires in the circuit.
    pub fn active_qudits(&self) -> Vec<usize> {
        self.front
            .iter()
            .filter_map(|(wire, _)| {
                if wire.is_quantum() {
                    Some(wire.index())
                } else {
                    None
                }
            })
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
    /// This method is O(w) where
    ///     - `w` is the number of wires in the circuit.
    pub fn active_dits(&self) -> Vec<usize> {
        self.front
            .iter()
            .filter_map(|(wire, _)| {
                if wire.is_classical() {
                    Some(wire.index())
                } else {
                    None
                }
            })
            .collect()
    }

    /// A reference to the parameters of the circuit.
    pub fn params(&self) -> &ParameterVector {
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
    /// # use qudit_circuit::QuditCircuit;
    /// let circuit = QuditCircuit::pure([2, 2]);
    /// assert!(circuit.is_empty());
    /// ```
    ///
    /// # Performance
    ///
    /// This method is O(1).
    pub fn is_empty(&self) -> bool {
        self.cycles.is_empty()
    }
}

impl QuditCircuit {
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
    /// let p_id = circuit.append(PGate(2), 0, None);
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
    fn find_available_or_append_cycle<W: AsRef<WireList>>(&mut self, wires: W) -> CycleIndex {
        // Location validity implicitly checked in find_available_cycle
        if let Some(cycle_index) = self.find_available_cycle(wires) {
            cycle_index
        } else {
            self.cycles.push()
        }
    }

    // TODO: prepend + insert

    /// Intern an operation in the circuit's operation cache
    ///
    /// This allows further additions by OpCodes.
    pub fn cache_operation<O: Into<Operation>>(&mut self, op: O) -> OpCode {
        self.operations.insert(op.into())
    }

    fn _append_ref(
        &mut self,
        op_code: OpCode,
        wires: WireList,
        params: ParamIndices,
    ) -> InstructionId {
        // TODO: check valid operation for radix match, measurement bandwidth etc
        // TODO: check params is valid: length is equal to op_params, existing exist, etc..
        // TODO: have to something about static entries...
        // TODO: have to do something about gate parameters mapped within same gate

        // Find cycle placement (location validity implicitly checked here)
        let cycle_index = self.find_available_or_append_cycle(&wires);
        let cycle_id = self.cycles.index_to_id(cycle_index);

        // Update quantum DAG info
        for wire in wires.wires() {
            if let Some(&rear_cycle_id) = self.rear.get(&wire) {
                self.cycles
                    .get_mut_from_id(rear_cycle_id)
                    .expect("Expected cycle to exist.")
                    .set_next(wire, cycle_id);
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
        let inner_id = self.cycles[cycle_index].push(inst_ref);

        InstructionId::new(cycle_id, inner_id)
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
        self.operations.increment(op); // Need to inform self.operations
        self._append_ref(op, wires, param_ids)
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
                let subbed_expr: UnitaryExpression = e
                    .substitute_parameters(&new_variables, &expressions)
                    .try_into()
                    .unwrap();
                ExpressionOperation::UnitaryGate(subbed_expr)
            }
            ExpressionOperation::KrausOperators(e) => {
                let e: TensorExpression = e.into();
                let subbed_expr: KrausOperatorsExpression = e
                    .substitute_parameters(&new_variables, &expressions)
                    .try_into()
                    .unwrap();
                ExpressionOperation::KrausOperators(subbed_expr)
            }
            ExpressionOperation::TerminatingMeasurement(e) => {
                let e: TensorExpression = e.into();
                let subbed_expr: BraSystemExpression = e
                    .substitute_parameters(&new_variables, &expressions)
                    .try_into()
                    .unwrap();
                ExpressionOperation::TerminatingMeasurement(subbed_expr)
            }
            ExpressionOperation::ClassicallyControlledUnitary(e) => {
                let e: TensorExpression = e.into();
                let subbed_expr: UnitarySystemExpression = e
                    .substitute_parameters(&new_variables, &expressions)
                    .try_into()
                    .unwrap();
                ExpressionOperation::ClassicallyControlledUnitary(subbed_expr)
            }
            ExpressionOperation::QuditInitialization(e) => {
                let e: TensorExpression = e.into();
                let subbed_expr: KetExpression = e
                    .substitute_parameters(&new_variables, &expressions)
                    .try_into()
                    .unwrap();
                ExpressionOperation::QuditInitialization(subbed_expr)
            }
        };

        let op_code = self.operations.insert_expression_with_dits(
            subbed_op,
            &loc.dits()
                .map(|d| self.dit_radices[d].into())
                .collect::<Vec<_>>(),
        );
        self._append_ref(op_code, loc, param_ids)
    }

    /// Append a subcircuit to the circuit
    pub fn append_subcircuit<L, P>(
        &mut self,
        _op: CircuitOperation,
        loc: L,
        params: P,
    ) -> InstructionId
    where
        L: Into<WireList>,
        P: Into<ArgumentList>,
    {
        let _loc = loc.into();
        let _params = params.into();

        todo!()
    }

    /// Append a circuit directive to the circuit
    pub fn append_directive<L, P>(
        &mut self,
        _op: DirectiveOperation,
        loc: L,
        params: P,
    ) -> InstructionId
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
        let location_radices = wires
            .qudits()
            .map(|q| self.qudit_radices[q])
            .collect::<Radices>();
        let state = KetExpression::zero(location_radices);
        let op = ExpressionOperation::QuditInitialization(state);
        self.append(op, wires, None::<ArgumentList>);
    }

    /// Remove the operation at `point` from the circuit
    pub fn remove(&mut self, inst_id: InstructionId) {
        if !self.is_valid_id(inst_id) {
            // TODO: log warning?
            return;
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
    }
}

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

/// Iteration
impl QuditCircuit {
    /// Return an iterator over the operations in the circuit.
    ///
    /// The ordering is not guaranteed to be consistent, but it will
    /// be in a simulation/topological order. For more control over the
    /// ordering of iteration see [`QuditCircuit::iter_df`] or
    /// [`QuditCircuit::iter_bf`].
    pub fn iter(&self) -> impl Iterator<Item = &Instruction> + '_ {
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
}

/// Evaluation
impl QuditCircuit {
    /// Calculate the Kraus Operators that describe this circuit as a program
    pub fn kraus_ops<C: ComplexScalar>(&self, args: &[C::R]) -> Tensor<C, 3> {
        let network = self.to_tensor_network();
        let code = qudit_tensor::compile_network(network);
        let mut tnvm =
            qudit_tensor::TNVM::<C, FUNCTION>::new(&code, Some(&self.params.const_map()));
        let result = tnvm.evaluate::<FUNCTION>(args);
        result.get_fn_result2().unpack_tensor3d().to_owned()
    }

    /// Convert the circuit to a symbolic tensor network
    pub fn to_tensor_network(&self) -> QuditTensorNetwork {
        self.as_tensor_network_builder().build()
    }

    /// Convert the circuit to a tensor network builder
    pub fn as_tensor_network_builder(&self) -> QuditCircuitTensorNetworkBuilder {
        let mut network = QuditCircuitTensorNetworkBuilder::new(
            self.qudit_radices(),
            Some(self.operations.expressions()),
        );

        for inst in self.iter() {
            if inst.op_code().kind() == OpKind::Expression {
                let indices = self.operations.indices(inst.op_code());
                let param_indices = self.params.convert_ids_to_indices(inst.params());
                let constant = param_indices
                    .iter()
                    .map(|i| self.params[i].is_constant())
                    .collect();
                let param_info = ParamInfo::new(param_indices, constant);
                let input_index_map = if indices
                    .iter()
                    .any(|idx| idx.direction() == IndexDirection::Input && idx.index_size() > 1)
                {
                    inst.wires().qudits().collect()
                } else {
                    vec![]
                };
                let output_index_map = if indices
                    .iter()
                    .any(|idx| idx.direction() == IndexDirection::Output && idx.index_size() > 1)
                {
                    inst.wires().qudits().collect()
                } else {
                    vec![]
                };
                let batch_index_map: Vec<String> =
                    inst.wires().dits().map(|id| id.to_string()).collect();
                let tensor = QuditTensor::new(indices, inst.op_code().id(), param_info);
                // println!("Adding new tensor to network builder with in qudits: {:?}; out qudits: {:?}, batch indices: {:?}", input_index_map.clone(), output_index_map.clone(), batch_index_map.clone());
                network =
                    network.prepend(tensor, input_index_map, output_index_map, batch_index_map)
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
    use super::*;
    use qudit_core::c32;
    use qudit_expr::GRADIENT;
    use qudit_expr::library::Controlled;
    use qudit_expr::library::U3Gate;
    use qudit_expr::library::XGate;

    pub fn build_qsearch_thin_step_circuit(n: usize) -> QuditCircuit {
        let block_expr = U3Gate()
            .otimes(U3Gate())
            .dot(Controlled(XGate(2), [2].into(), None));
        let mut circ = QuditCircuit::pure(vec![2; n]);
        for i in 0..n {
            circ.append(U3Gate(), [i], None);
        }
        for _ in 0..2 {
            for i in 0..(n - 1) {
                circ.append(block_expr.clone(), [i, i + 1], None);
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
        const N: usize = 3;
        let circ = build_qsearch_thin_step_circuit(N);
        let network = circ.to_tensor_network();
        let code = qudit_tensor::compile_network(network);
        let mut tnvm =
            qudit_tensor::TNVM::<c32, GRADIENT>::new(&code, Some(&circ.params.const_map()));
        let result = tnvm.evaluate::<GRADIENT>(&[1.7; (3 * N) + (6 * (N - 1) * 2)]);
        let _unitary = result.get_fn_result().unpack_matrix();
    }
}

#[cfg(feature = "python")]
mod python {
    use super::*;
    use crate::python::PyCircuitRegistrar;
    use ndarray::ArrayViewMut3;
    use numpy::PyArray3;
    use numpy::PyArrayMethods;
    use pyo3::exceptions::PyTypeError;
    use pyo3::prelude::*;
    use pyo3::types::PyTuple;
    use qudit_core::c64;

    #[pyclass]
    #[pyo3(name = "QuditCircuit")]
    pub struct PyQuditCircuit {
        circuit: QuditCircuit,
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
        fn new<'py>(qudits: Radices, dits: Option<Radices>) -> PyResult<PyQuditCircuit> {
            match dits {
                None => Ok(PyQuditCircuit {
                    circuit: QuditCircuit::pure(qudits),
                }),
                Some(dits) => Ok(PyQuditCircuit {
                    circuit: QuditCircuit::new(qudits, dits),
                }),
            }
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
        #[getter]
        fn front(&self) -> HashMap<Wire, InstructionId> {
            self.circuit.front()
        }

        #[getter]
        fn rear(&self) -> HashMap<Wire, InstructionId> {
            self.circuit.rear()
        }

        fn first_on(&self, wire: Wire) -> Option<InstructionId> {
            self.circuit.first_on(wire)
        }

        fn last_on(&self, wire: Wire) -> Option<InstructionId> {
            self.circuit.last_on(wire)
        }

        fn next(&self, inst_id: InstructionId) -> HashMap<Wire, InstructionId> {
            self.circuit.next(inst_id)
        }

        fn prev(&self, inst_id: InstructionId) -> HashMap<Wire, InstructionId> {
            self.circuit.prev(inst_id)
        }

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
                            "Circuit has parameters, but no arguments were provided to kraus_ops.",
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
            } else if let Ok(tuple) = loc.cast::<PyTuple>() {
                if tuple.len() == 2 {
                    let item0 = tuple.get_item(0)?;
                    let item1 = tuple.get_item(1)?;

                    // Try parsing as (iterable, iterable)
                    if let (Ok(vec0), Ok(vec1)) =
                        (item0.extract::<Vec<usize>>(), item1.extract::<Vec<usize>>())
                    {
                        WireList::new(vec0, vec1)
                    }
                    // Else, try parsing as (int, int)
                    else if let (Ok(int0), Ok(int1)) =
                        (item0.extract::<usize>(), item1.extract::<usize>())
                    {
                        if num_qudits.is_some() && num_qudits == Some(1) {
                            WireList::new(vec![int0], vec![int1])
                        } else {
                            WireList::pure(&[int0, int1])
                        }
                    } else {
                        return Err(PyTypeError::new_err(
                            "A 2-element 'loc' tuple must contain (int, int) or (iterable, iterable)",
                        ));
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

    impl<'a, 'py> FromPyObject<'a, 'py> for QuditCircuit {
        type Error = PyErr;

        fn extract(obj: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
            let py_circuit_ref = obj.cast::<PyQuditCircuit>()?;
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
