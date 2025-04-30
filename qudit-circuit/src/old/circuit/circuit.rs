use std::collections::HashMap;
use std::fmt;
use std::ops::Range;

use super::cycle::CycleList;
use super::cycle::QuditCycle;
use super::instruction::Instruction;
use super::iterator::QuditCircuitBFIterator;
use super::iterator::QuditCircuitDFIterator;
use super::iterator::QuditCircuitFastIterator;
use super::iterator::QuditCircuitFastIteratorWithCycles;
use super::location::CircuitLocation;
use super::node::OperationNode;
use super::operation::Operation;
use super::point::CircuitPoint;
use super::point::DitOrBit;
// use crate::math::unitary::DifferentiableUnitaryFn;
// use crate::math::unitary::DoublyDifferentiableUnitaryFn;
// use crate::math::unitary::UnitaryFn;
// use crate::math::unitary::UnitaryGradient;
// use crate::math::unitary::UnitaryHessian;
// use crate::math::unitary::UnitaryMatrix;
// use crate::math::unitary::UnitaryTensor;
use crate::math::ComplexScalar;
use crate::math::*;
use crate::point;
use crate::sim::ExpressionTree;
use crate::sim::TreeBuilder;
use crate::sim::TreeOptimizer;
use crate::system::ClassicalSystem;
use crate::system::HybridSystem;
use crate::Gate;
use crate::QuditRadices;
use crate::QuditSystem;

#[derive(Clone)]
pub struct QuditCircuit<C: ComplexScalar = c64> {
    num_qudits: usize,
    num_clbits: usize,
    radices: QuditRadices,
    cycles: CycleList<C>,
    inst_info: HashMap<Instruction, usize>,
    graph_info: HashMap<(usize, usize), usize>,
    qfront: Vec<Option<usize>>,
    qrear: Vec<Option<usize>>,
    cfront: Vec<Option<usize>>,
    crear: Vec<Option<usize>>,
}

// Wants:
// **O(1) append** (Find or create first available cycle (O(1) append to it)
// O(1) prepend (Should be O(1) since the cycles are linked rather than in
// array) [Nope (O(N))] O(1)/O(N) remove (removing op is O(q) where q is size of
// op, removing cycle is O(N), finding cycle op is in is O(1)) O(1) pop (same as
// above) O(N) insert
// O(N) iteration/finding op
// O(1) dag ops

// Questions:
// How do you express a circuit region?
// How do you fold?

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
    /// use qudit_circuit::{radices, QuditRadices};
    ///
    /// let two_qubit_circuit = QuditCircuit::new(radices![2, 2], 0);
    /// let two_qutrit_circuit = QuditCircuit::new(radices![3, 3], 0);
    /// let hybrid_circuit = QuditCircuit::new(radices![2, 2, 3, 3], 0);
    /// ```
    ///
    /// We can also define hybrid quantum-classical circuits:
    /// ```
    /// use qudit_circuit::QuditCircuit;
    /// use qudit_circuit::{radices, QuditRadices};
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
    /// use qudit_circuit::{radices, QuditRadices};
    /// let two_qubit_circuit = QuditCircuit::with_capacity(radices![2, 2], 2, 10);
    /// ```
    pub fn with_capacity(
        radices: QuditRadices,
        num_clbits: usize,
        capacity: usize,
    ) -> QuditCircuit<C> {
        QuditCircuit {
            num_qudits: radices.get_num_qudits(),
            num_clbits: num_clbits,
            cycles: CycleList::with_capacity(capacity),
            inst_info: HashMap::new(),
            graph_info: HashMap::new(),
            qfront: vec![None; radices.get_num_qudits()],
            qrear: vec![None; radices.get_num_qudits()],
            radices: radices,
            cfront: vec![None; num_clbits],
            crear: vec![None; num_clbits],
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

    /// Returns the number of operations in the circuit.
    ///
    /// # Performance
    ///
    /// This method is O(|t|) where `t` is the number of distinct instruction
    /// types in the circuit.
    #[allow(dead_code)]
    fn get_num_operations(&self) -> usize {
        self.inst_info.iter().map(|(_, count)| count).sum()
    }

    // pub fn get_depth(&self) -> usize {
    //     let mut qudit_depths = vec![0; self.num_qudits];
    //     for op in self.ops.iter() {
    //         let new_depth = op
    //             .location
    //             .iter()
    //             .map(|&q| qudit_depths[q] + 1)
    //             .max()
    //             .unwrap();
    //         op.location
    //             .iter()
    //             .for_each(|&q| qudit_depths[q] = new_depth);
    //     }
    //     *qudit_depths.iter().max().unwrap()
    // }

    // pub fn get_params(&self) -> Vec<f64> {
    //     let mut params = Vec::with_capacity(self.get_num_params());
    //     self.ops
    //         .iter()
    //         .for_each(|op| params.extend(op.params.iter()));
    //     params
    // }

    // fn get_parallelism(&self) -> f64 {}
    // fn get_connectivity(&self) -> CouplingGraph {}
    // fn get_gate_counts(&self) -> HashMap<Gate, usize> {}

    // fn append_qudit(&mut self, radix: usize) {}
    // fn extend_qudits(&mut self, radices: &[usize]) {}
    // fn insert_qudit(&mut self, radix: usize) {}
    // fn pop_qudit(&mut self, qudit_index: usize) {}
    // fn is_qudit_idle(&self, qudit_index: usize) {}
    // fn renumber_qudits(&mut self, qudit_permutation: &[usize]) {}

    // fn index(&self, inst: Instruction) -> Option<usize> {}
    // fn append(&mut self, op: Operation) {}

    /// Increment internal instruction type counter.
    fn inc_inst_counter(&mut self, inst_type: &Instruction) {
        match self.inst_info.get_mut(inst_type) {
            Some(count) => *count += 1,
            None => {
                self.inst_info.insert(inst_type.clone(), 1);
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
    fn dec_inst_counter(&mut self, inst_type: &Instruction) {
        if !self.inst_info.contains_key(inst_type) {
            panic!(
                "Cannot decrement instruction counter for instruction type that does not exist."
            );
        }

        let count = self.inst_info.get_mut(inst_type).unwrap();
        *count -= 1;

        if *count == 0 {
            self.inst_info.remove(inst_type);
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
        location.qudits().iter().all(|&q| q < self.num_qudits)
            && location.clbits().iter().all(|&c| c < self.num_clbits)
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

    pub fn set_params(&mut self, params: &[C::Re]) {
        let mut param_index = 0;
        for cycle_index in 0..self.cycles.len() {
            let cycle = &mut self.cycles[cycle_index];
            for node_index in cycle.get_all_indices() {
                let op = &mut cycle.get_mut(node_index).unwrap().op;
                op.set_params(
                    &params[param_index..param_index + op.get_num_params()],
                );
                param_index += op.get_num_params();
            }
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
            .map(|&q| match self.qrear[q] {
                Some(cycle_index) => {
                    Some(self.cycles.map_physical_to_logical_idx(cycle_index))
                },
                None => None,
            })
            .chain(location.clbits().iter().map(|&c| match self.crear[c] {
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

    pub fn append_operation(&mut self, op: Operation<C>) {
        // check valid operation for radix match, measurement bandwidth etc
        let location = op.location().clone();

        // Build operation object (location validity implicitly checked here)
        let cycle_index = self.find_available_or_append_cycle(&location);
        let mut node = OperationNode::new(op, cycle_index); // physical index

        // Update counters
        self.inc_inst_counter(node.op.instruction());
        self.inc_graph_counter(&location);

        // Update qudit dag info
        for (loc_index, qudit_index) in location.qudits().iter().enumerate() {
            if let Some(rear_cycle_index) = self.qrear[*qudit_index] {
                let prev_node =
                    self.get_node_mut(point!(rear_cycle_index, *qudit_index));
                let qnext_index = prev_node
                    .op
                    .location()
                    .get_qudit_index(qudit_index)
                    .unwrap();
                prev_node.qnext[qnext_index] = Some(cycle_index);
                node.qprev[loc_index] = Some(rear_cycle_index);
            } else {
                // If qrear is none then no op exists on this qudit from before,
                // so we update qfront too.
                self.qfront[*qudit_index] = Some(cycle_index);
            }
            // Update qrear
            self.qrear[*qudit_index] = Some(cycle_index);
        }

        // Update clbit dag info
        for (loc_index, clbit_index) in location.clbits().iter().enumerate() {
            if let Some(rear_cycle_index) = self.crear[*clbit_index] {
                let prev_node =
                    self.get_node_mut(point!(rear_cycle_index, *clbit_index));
                let cnext_index = prev_node
                    .op
                    .location()
                    .get_clbit_index(clbit_index)
                    .unwrap();
                prev_node.qnext[cnext_index] = Some(cycle_index);
                node.qprev[loc_index] = Some(rear_cycle_index);
            } else {
                // If crear is none then no op exists on this clbit from before,
                // so we update cfront too.
                self.cfront[*clbit_index] = Some(cycle_index);
            }
            // Update crear
            self.crear[*clbit_index] = Some(cycle_index);
        }

        // Add op to cycle
        self.cycles.get_mut(cycle_index).push(node);
    }

    pub fn append_gate(
        &mut self,
        gate: Gate,
        location: CircuitLocation,
        params: Vec<C::Re>,
    ) {
        let inst_type = if location.clbits().is_empty() {
            Instruction::Quantum(gate)
        } else {
            Instruction::ClassicallyControlled(gate)
        };

        let op = Operation::new(inst_type, location, params);
        self.append_operation(op);
    }

    pub fn remove(&mut self, point: CircuitPoint) {
        if !self.is_valid_point(&point) {
            panic!("Cannot remove operation at invalid point.");
        }

        let cycle_index = self.cycles.map_logical_to_physical_idx(point.cycle);
        let node = self.cycles.get_mut(cycle_index).remove(point.dit_or_bit);
        let location = node.op.location();

        // if cycle is empty; remove it
        if self.cycles.get(cycle_index).is_empty() {
            self.cycles.remove(cycle_index);
        }

        // Update circuit qudit dag info
        for (i, qudit_index) in location.qudits().iter().enumerate() {
            if self.qfront[*qudit_index] == Some(cycle_index) {
                self.qfront[*qudit_index] = node.qnext[i];
            }
            if self.qrear[*qudit_index] == Some(cycle_index) {
                self.qrear[*qudit_index] = node.qprev[i];
            }
        }

        // Update circuit qudit dag info
        for (i, clbit_index) in location.clbits().iter().enumerate() {
            if self.cfront[*clbit_index] == Some(cycle_index) {
                self.cfront[*clbit_index] = node.cnext[i];
            }
            if self.crear[*clbit_index] == Some(cycle_index) {
                self.crear[*clbit_index] = node.cprev[i];
            }
        }

        // Update counters
        self.dec_inst_counter(node.op.instruction());
        self.dec_graph_counter(location);
    }

    // fn append_circuit(&mut self, circuit: Circuit, location: Vec<usize>) {}
    // fn extend(&mut self, ops: &[Operation]) {}
    // fn insert_before(&mut self, new_op: Operation, ptr_op: &Operation) {}
    // fn insert_after(&mut self, new_op: Operation, ptr_op: &Operation) {}
    // fn insert_gate_after(&mut self, gate: Gate, location: Vec<usize>, params:
    // Vec<f64>)

    // pub fn group(&mut self, ops: &[&Operation]) {}

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
            .map(|x| (DitOrBit::Qudit(x.0), point!(x.1.unwrap(), x.0)))
            .chain(
                self.cfront
                    .iter()
                    .enumerate()
                    .filter(|x| x.1.is_some())
                    .map(|x| (DitOrBit::Clbit(x.0), point!(x.1.unwrap(); x.0))),
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
            .map(|x| (DitOrBit::Qudit(x.0), point!(x.1.unwrap(), x.0)))
            .chain(
                self.crear
                    .iter()
                    .enumerate()
                    .filter(|x| x.1.is_some())
                    .map(|x| (DitOrBit::Clbit(x.0), point!(x.1.unwrap(); x.0))),
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
    /// qudits and number of classical bits in the operation refered to by
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
        let node = &self.get_node(point);
        node.qnext
            .iter()
            .enumerate()
            .filter(|x| x.1.is_some())
            .map(|x| {
                (
                    DitOrBit::Qudit(node.op.location().qudits()[x.0]),
                    point!(x.1.unwrap(), x.0),
                )
            })
            .chain(node.cnext.iter().enumerate().filter(|x| x.1.is_some()).map(
                |x| {
                    (
                        DitOrBit::Clbit(node.op.location().clbits()[x.0]),
                        point!(x.1.unwrap(); x.0),
                    )
                },
            ))
            .collect()
    }

    /// Gather the points of the previous operations from the point of an
    /// operation.
    ///
    /// See [`QuditCircuit::next`] for more information.
    pub fn prev(&self, point: CircuitPoint) -> HashMap<DitOrBit, CircuitPoint> {
        let node = &self.get_node(point);
        node.qprev
            .iter()
            .enumerate()
            .filter(|x| x.1.is_some())
            .map(|x| {
                (
                    DitOrBit::Qudit(node.op.location().qudits()[x.0]),
                    point!(x.1.unwrap(), x.0),
                )
            })
            .chain(node.cprev.iter().enumerate().filter(|x| x.1.is_some()).map(
                |x| {
                    (
                        DitOrBit::Clbit(node.op.location().clbits()[x.0]),
                        point!(x.1.unwrap(); x.0),
                    )
                },
            ))
            .collect()
    }

    /// Retrieve the operation at a point in the circuit.
    ///
    /// # Arguments
    ///
    /// * `point` - The point to get the operation from. This needs refer
    /// to a valid point in the circuit.
    ///
    /// # Returns
    ///
    /// A reference to the operation at `point`.
    ///
    /// # Performance
    ///
    /// This method is O(1).
    ///
    /// # Panics
    ///
    /// If `point` is not a valid point in the circuit.
    ///
    /// # Examples
    ///
    /// ```
    /// use qudit_circuit::{QuditCircuit, Gate, QuditRadices};
    /// use qudit_circuit::{radices, loc, point};
    /// use qudit_circuit::circuit::location::CircuitLocation;
    /// use qudit_circuit::circuit::point::{CircuitPoint, DitOrBit};
    /// use qudit_circuit::circuit::instruction::Instruction;
    /// let mut circuit = QuditCircuit::new(radices![2, 2], 2);
    /// circuit.append_gate(Gate::Z(2), loc![0], vec![]);
    /// circuit.append_gate(Gate::H(2), loc![1], vec![]);
    /// assert_eq!(circuit.get(point!(0, 0)).instruction(), &Instruction::Quantum(Gate::Z(2)));
    /// assert_eq!(circuit.get(point!(0, 1)).instruction(), &Instruction::Quantum(Gate::H(2)));
    /// ```
    pub fn get(&self, point: CircuitPoint) -> &Operation<C> {
        match self.cycles[point.cycle].get(point.dit_or_bit) {
            Some(node) => &node.op,
            None => panic!("Cannot get operation at invalid point."),
        }
    }

    /// Retrieve the operation node at a point in the circuit.
    pub(super) fn get_node(&self, point: CircuitPoint) -> &OperationNode<C> {
        match self.cycles[point.cycle].get(point.dit_or_bit) {
            Some(node) => &node,
            None => panic!("Cannot get operation at invalid point."),
        }
    }

    /// Retrieve the operation node at a point in the circuit.
    #[inline]
    pub(super) fn get_node_mut(
        &mut self,
        point: CircuitPoint,
    ) -> &mut OperationNode<C> {
        match self.cycles.get_mut(point.cycle).get_mut(point.dit_or_bit) {
            Some(node) => node,
            None => panic!("Cannot get operation at invalid point."),
        }
    }

    /// Retrieve the cycle at the logical `index` in the circuit.
    pub(super) fn get_cycle(&self, index: usize) -> &QuditCycle<C> {
        &self.cycles[index]
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

    /// Return an iterator over the operations in the circuit with cycles.
    pub fn iter_with_cycles(&self) -> QuditCircuitFastIteratorWithCycles<C> {
        QuditCircuitFastIteratorWithCycles::new(self)
    }

    pub fn expression_tree(&self) -> ExpressionTree {
        // TODO: panic if non quantum

        let mut point_to_index_map = HashMap::new();
        let mut op_index_count = 0;
        for (cycle_index, cycle) in self.cycles.iter().enumerate() {
            for node in cycle {
                let point = point![cycle_index, node.op.location().qudits()[0]];
                point_to_index_map.insert(point, op_index_count);
                op_index_count += 1;
            }
        }

        let mut op_list = Vec::new();
        let mut next_list = Vec::new();
        let mut prev_list = Vec::new();

        for cycle in &self.cycles {
            for node in cycle {
                op_list.push(&node.op);

                let mut op_nexts = Vec::new();
                let mut op_prevs = Vec::new();

                for (loc_index, next) in node.qnext.iter().enumerate() {
                    match next {
                        Some(next_cycle_index) => {
                            let next_point = point![
                                *next_cycle_index,
                                node.op.location().qudits()[loc_index]
                            ];
                            let std_next_point = point![
                                *next_cycle_index,
                                self.get_node(next_point)
                                    .op
                                    .location()
                                    .qudits()[0]
                            ];
                            op_nexts.push(Some(
                                point_to_index_map[&std_next_point],
                            ));
                        },
                        None => op_nexts.push(None),
                    }
                }

                for (loc_index, prev) in node.qprev.iter().enumerate() {
                    match prev {
                        Some(prev_cycle_index) => {
                            let prev_point = point![
                                *prev_cycle_index,
                                node.op.location().qudits()[loc_index]
                            ];
                            let std_prev_point = point![
                                *prev_cycle_index,
                                self.get_node(prev_point)
                                    .op
                                    .location()
                                    .qudits()[0]
                            ];
                            op_prevs.push(Some(
                                point_to_index_map[&std_prev_point],
                            ));
                        },
                        None => op_prevs.push(None),
                    }
                }

                next_list.push(op_nexts);
                prev_list.push(op_prevs);
            }
        }

        let tree = TreeBuilder::new(
            self.get_num_qudits(),
            op_list,
            next_list,
            prev_list,
        )
        .build_tree();
        TreeOptimizer::new().optimize(tree)
    }

    // pub fn with_gate
    // pub fn with_instruction
    // pub fn instantiate(target: InstantiationTarget, multistarts: usize) //
    // KISS, for more manually build instantiater
}

impl<C: ComplexScalar> QuditSystem for QuditCircuit<C> {
    fn get_num_qudits(&self) -> usize {
        self.num_qudits
    }

    fn get_radices(&self) -> QuditRadices {
        self.radices.clone()
    }
}

impl<C: ComplexScalar> ClassicalSystem for QuditCircuit<C> {
    fn get_num_clbits(&self) -> usize {
        self.num_clbits
    }
}

impl<C: ComplexScalar> HybridSystem for QuditCircuit<C> {}

impl<'a, C: ComplexScalar> IntoIterator for &'a QuditCircuit<C> {
    type IntoIter = QuditCircuitFastIterator<'a, C>;
    type Item = &'a Operation<C>;

    fn into_iter(self) -> Self::IntoIter {
        QuditCircuitFastIterator::new(self)
    }
}

impl<C: ComplexScalar> Function for QuditCircuit<C> {
    /// Returns the number of parameters in the circuit.
    ///
    /// # Performance
    ///
    /// This method is O(|t|) where `t` is the number of distinct instruction
    /// types in the circuit.
    fn get_num_params(&self) -> usize {
        self.inst_info
            .iter()
            .map(|(inst, count)| inst.get_num_params() * count)
            .sum()
    }
}

impl<C: ComplexScalar> BoundedFn for QuditCircuit<C> {
    fn get_bounds(&self) -> Vec<Range<f64>> {
        let mut bounds = Vec::with_capacity(self.get_num_params());
        for op in self {
            bounds.extend(op.get_bounds());
        }
        bounds
    }
}

// impl<C: ComplexScalar> UnitaryFn<C> for QuditCircuit
// {
//     fn get_unitary(
//         &self,
//         params: &[C::Re],
//     ) -> UnitaryMatrix<C>
//     {
//         if self.get_num_clbits() != 0
//         {
//             // TODO: Actually check for classical ops
//             panic!("Cannot get unitary of circuit with classical bits.");
//         }

//         // Construct a unitary builder
//         let mut builder = UnitaryTensor::new(self.radices.clone());

//         // If we are supplied parameters use them, otherwise use op's stored
// params.         if params.is_empty()
//         {
//             for op in self
//             {
//                 // TODO: Benchmark using depth first iterator here
//                 let utry: UnitaryMatrix<C> = op.get_unitary(&[]);
//                 builder.apply_right(utry.view(), op.location().qudits(),
// false);             }
//         }
//         else
//         {
//             if params.len() != self.get_num_params()
//             {
//                 panic!("Incorrect number of parameters supplied to
// circuit.");             }

//             let mut param_idx = 0;
//             for op in self
//             {
//                 // TODO: Benchmark using depth first iterator here
//                 let lower_bound = param_idx;
//                 let upper_bound = param_idx + op.get_num_params();
//                 let utry = op.get_unitary(&params[lower_bound..upper_bound]);
//                 param_idx += op.get_num_params();
//                 builder.apply_right(utry.view(), op.location().qudits(),
// false);             }
//         }

//         // Return built unitary
//         builder.get_unitary()
//     }
// }

// impl<C: ComplexScalar> DifferentiableUnitaryFn<C> for QuditCircuit
// {
//     fn get_gradient(
//         &self,
//         params: &[C::Re],
//     ) -> UnitaryGradient<C>
//     {
//         self.get_unitary_and_gradient(params).1
//     }

//     fn get_unitary_and_gradient(
//         &self,
//         params: &[C::Re],
//     ) -> (UnitaryMatrix<C>, UnitaryGradient<C>)
//     {
//         let num_ops = self.get_num_operations();
//         let mut utry_list = Vec::with_capacity(num_ops);
//         let mut grad_list = Vec::with_capacity(num_ops);
//         let mut loc_list = Vec::with_capacity(num_ops);
//         if params.is_empty()
//         {
//             for op in self
//             {
//                 // TODO: Benchmark using depth first iterator here
//                 let (utry, grad) = op.get_unitary_and_gradient(&[]);
//                 utry_list.push(utry);
//                 grad_list.push(grad);
//                 loc_list.push(op.location());
//             }
//         }
//         else
//         {
//             let mut param_idx = 0;
//             for op in self
//             {
//                 let param_slice = &params[param_idx..param_idx +
// op.get_num_params()];                 let (utry, grad) =
// op.get_unitary_and_gradient(param_slice);                 param_idx +=
// op.get_num_params();                 utry_list.push(utry);
//                 grad_list.push(grad);
//                 loc_list.push(op.location());
//             }
//         }

//         let mut left = UnitaryTensor::new(self.radices.clone());
//         let mut right: UnitaryTensor<C> =
// UnitaryTensor::new(self.radices.clone());         let mut out_grad =
// UnitaryGradient::zeros(self.radices.clone(), self.get_num_params());

//         let mut grad_idx = 0usize;

//         for (m, location) in utry_list.iter().zip(loc_list.iter())
//         {
//             right.apply_right(m.view(), location.qudits(), false);
//         }

//         for (m, (location, d_m)) in
// utry_list.iter().zip(loc_list.iter().zip(grad_list.iter()))         {
//             // Store left unitary and then apply gate to left on right
//             let left_utry = left.get_unitary();
//             left.apply_right(m.view(), location.qudits(), false);

//             // Remove gate from the left of the right tensor
//             right.apply_left(m.view(), location.qudits(), true);

//             // Calculate all gradients
//             if d_m.len() > 0
//             {
//                 // get the slice of the out_grad as mutable buffer from
// grad_idx to                 // d_m.shape()[0] + grad_idx:
//                 let mut grad_ref = out_grad[grad_idx..grad_idx +
// d_m.shape()[0]];                 right.squeeze_left(
//                     left_utry.view(),
//                     d_m.view(),
//                     &mut grad_ref,
//                     location.qudits(),
//                 );
//                 grad_idx += d_m.shape()[0];
//             }
//         }

//         (left.get_unitary(), out_grad)
//     }
// }

// impl<C: ComplexScalar> DoublyDifferentiableUnitaryFn<C> for QuditCircuit
// {
//     fn get_unitary_gradient_and_hessian(
//         &self,
//         params: &[C::Re],
//     ) -> (UnitaryMatrix<C>, UnitaryGradient<C>, UnitaryHessian<C>)
//     {
//         let num_ops = self.get_num_operations();
//         let mut utry_list = Vec::with_capacity(num_ops);
//         let mut grad_list = Vec::with_capacity(num_ops);
//         let mut hess_list = Vec::with_capacity(num_ops);
//         let mut loc_list = Vec::with_capacity(num_ops);
//         if params.is_empty()
//         {
//             for op in self
//             {
//                 // TODO: Benchmark using depth first iterator here
//                 let (utry, grad, hess) =
// op.get_unitary_gradient_and_hessian(&[]);
// utry_list.push(utry);                 grad_list.push(grad);
//                 hess_list.push(hess);
//                 loc_list.push(op.location());
//             }
//         }
//         else
//         {
//             let mut param_idx = 0;
//             for op in self
//             {
//                 let param_slice = &params[param_idx..param_idx +
// op.get_num_params()];                 let (utry, grad, hess) =
// op.get_unitary_gradient_and_hessian(param_slice);                 param_idx
// += op.get_num_params();                 utry_list.push(utry);
//                 grad_list.push(grad);
//                 hess_list.push(hess);
//                 loc_list.push(op.location());
//             }
//         }

//         let mut left = UnitaryTensor::new(self.radices.clone());
//         let mut right: UnitaryTensor<C> =
// UnitaryTensor::new(self.radices.clone());         let mut out_grad =
// UnitaryGradient::zeros(self.radices.clone(), self.get_num_params());
//         let mut out_hess = UnitaryHessian::zeros(self.radices.clone(),
// self.get_num_params());         let mut grad_idx = 0usize;
//         let mut hess_idx = 0usize;

//         for (m, location) in utry_list.iter().zip(loc_list.iter()).skip(1)
//         {
//             right.apply_right(m.view(), location.qudits(), false);
//         }

//         for i in 0..utry_list.len()
//         {
//             let d_m = &grad_list[i];
//             let num_gate_params = d_m.shape()[0];
//             let location = &loc_list[i];

//             // 0. Skip all hessian and gradient calculations if the gate is
// constant             if num_gate_params == 0
//             {
//                 if i != utry_list.len() - 1
//                 {
//                     right.apply_left(utry_list[i + 1].view(), loc_list[i +
// 1].qudits(), true);                 }
//                 left.apply_right(utry_list[i].view(), location.qudits(),
// false);                 continue;
//             }

//             // 1. Calculate block diagonal of circuit hessian, these are
// blocks             // use the hessian of the gate.
//             let left_utry = left.get_unitary();
//             for (k, dd_m_row) in hess_list[i].outer_iter().enumerate()
//             {
//                 let mut hess_ref = out_hess[(hess_idx + k, hess_idx..hess_idx
// + num_gate_params)];                 // right * dd_m_row (broadcast) *
// left_utry -> hess_ref                 right.squeeze_left(
//                     left_utry.view(),
//                     dd_m_row.view(),
//                     &mut hess_ref,
//                     location.qudits(),
//                 );
//             }

//             // 2. Calculate off-diagonal blocks of circuit hessian, these
// blocks             // are formed from two different gate gradients in the
// tensor string.             // The hessian super matrix is also block
// symmetric, so we just             // compute the upper triangular blocks and
// copy them to the lower             // triangular blocks.
//             if i != utry_list.len() - 1
//             {
//                 // 2a. remove utry[i+1] from right
//                 right.apply_left(utry_list[i + 1].view(), loc_list[i +
// 1].qudits(), true);

//                 // 2b. copy right tensor -> grad_right
//                 let mut grad_right = right.clone();

//                 // 2c. copy left tensor (x num_gate_params) -> grad_lefts
//                 let mut grad_lefts = vec![left.clone(); num_gate_params];

//                 // 2d. apply d_m[i] to grad_lefts[i] on right
//                 for (k, grad_left) in grad_lefts.iter_mut().enumerate()
//                 {
//                     grad_left.apply_right(d_m[k], location.qudits(), false);
//                 }

//                 // 2e Calculate the block for row i col j in the super-block
// hessian matrix                 let mut col_idx = hess_idx.clone() +
// num_gate_params;                 for j in (i + 1)..utry_list.len()
//                 {
//                     let other_num_gate_params = grad_list[j].shape()[0];

//                     if other_num_gate_params == 0
//                     {
//                         for k in 0..num_gate_params
//                         {
//                             grad_lefts[k].apply_right(
//                                 utry_list[j].view(),
//                                 loc_list[j].qudits(),
//                                 false,
//                             );
//                         }
//                         if j != utry_list.len() - 1
//                         {
//                             grad_right.apply_left(
//                                 utry_list[j + 1].view(),
//                                 loc_list[j + 1].qudits(),
//                                 true,
//                             );
//                         }
//                         continue;
//                     }

//                     // 2ei. calculate grad right unitary
//                     let grad_right_utry = grad_right.get_unitary();

//                     let col_range = col_idx..(col_idx +
// other_num_gate_params);                     for k in 0..num_gate_params
//                     {
//                         // 2eii. Slice out_hess to get row block and col
// block                         // let slice = s![hess_idx + k,
// col_range.clone(), .., ..];                         // let slice_t =
// s![col_range.clone(), hess_idx + k, .., ..];                         // let
// (mut hess_ref, mut hess_ref_t) =                         //
// out_hess.multi_slice_mut((slice, slice_t));                         let mut
// hess_ref = out_hess[(hess_idx + k, col_range.clone())];

//                         // 2eii. squeeze: grad_right * grad_list[j] *
// grads_left[k] -> hess_ref
// grad_lefts[k].squeeze_right(
// grad_right_utry.view(),                             grad_list[j].view(),
//                             &mut hess_ref,
//                             loc_list[j].qudits(),
//                         );

//                         // 2eiii. copy data from off-diagonal row to
// off-diagonal col                         // hess_ref.swap_axes(0, 1);
//                         // hess_ref_t.assign(&hess_ref);

//                         // 2eiv. apply utry_list[j] to grads_left[k] on the
// right                         grad_lefts[k].apply_right(utry_list[j].view(),
// loc_list[j].qudits(), false);                     }
//                     col_idx += other_num_gate_params;

//                     // 2ev.apply utry_list[j + 1] inverse to grad_right on
// the left                     if j != utry_list.len() - 1
//                     {
//                         grad_right.apply_left(
//                             utry_list[j + 1].view(),
//                             loc_list[j + 1].qudits(),
//                             true,
//                         );
//                     }
//                 }

//                 // 2f. Store resulting grad_left as complete gradient for
// this gate                 for k in 0..num_gate_params
//                 {
//                     out_grad[grad_idx] = &grad_lefts[k].get_square_data();
//                     grad_idx += 1;
//                 }
//             }
//             else
//             {
//                 // 2g. Calculate gradient for last gate
//                 let mut grad_ref = out_grad[grad_idx..grad_idx +
// num_gate_params];                 left.squeeze_right(
//                     right.get_unitary().view(),
//                     d_m.view(),
//                     &mut grad_ref,
//                     location.qudits(),
//                 );
//             }

//             // 3. Increment hess_idx and update left tensor
//             hess_idx += num_gate_params;
//             left.apply_right(utry_list[i].view(), location.qudits(), false);
//         }

//         (left.get_unitary(), out_grad, out_hess)
//     }

//     fn get_hessian(
//         &self,
//         params: &[C::Re],
//     ) -> UnitaryHessian<C>
//     {
//         self.get_unitary_gradient_and_hessian(params).2
//     }
// }

impl<C: ComplexScalar> fmt::Debug for QuditCircuit<C> {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt.debug_struct("QuditCircuit")
            .field("num_qudits", &self.num_qudits)
            .field("num_clbits", &self.num_clbits)
            .field("radices", &self.radices)
            .field("cycles", &self.cycles)
            .finish()
    }
}

#[cfg(test)]
pub mod strategies {
    use proptest::prelude::*;

    use super::*;
    use crate::gates::strategies::ArbitraryGateWithRadices;

    pub fn arbitrary_pure_circuit<C: ComplexScalar>(
        radices: Option<QuditRadices>,
        min_num_gates: Option<usize>,
        max_num_gates: Option<usize>,
    ) -> impl Strategy<Value = QuditCircuit<C>> {
        // TODO: I want this to shrink to smaller circuits with fewer gates
        // TODO: Rebuild as a base strategy and value tree
        let radix_strat = match radices {
            Some(radices) => Just(radices).boxed(),
            None => any_with::<QuditRadices>((2, 3, 2, 8)),
        };

        let num_gate_strat = match (min_num_gates, max_num_gates) {
            (Some(min), Some(max)) => min..max,
            (Some(min), None) => min..20,
            (None, Some(max)) => 0..max,
            (None, None) => 0..20,
        };

        (radix_strat, num_gate_strat)
            .prop_flat_map(|(radices, num_gates)| {
                (
                    Just(radices.clone()),
                    prop::collection::vec(
                        1..=radices.get_num_qudits(),
                        num_gates,
                    ),
                )
            })
            .prop_perturb(|(radices, gate_sizes), mut rng| {
                let mut gate_radices_and_locations = Vec::new();
                for gate_size in gate_sizes.iter() {
                    let mut gate_radices = Vec::new();
                    let mut gate_location = Vec::new();
                    for _ in 0..*gate_size {
                        let mut qudit_index =
                            rng.gen_range(0..radices.get_num_qudits());
                        while gate_location.contains(&qudit_index) {
                            qudit_index =
                                rng.gen_range(0..radices.get_num_qudits());
                        }
                        gate_radices.push(radices[qudit_index]);
                        gate_location.push(qudit_index);
                    }
                    gate_radices_and_locations.push((
                        QuditRadices::new(gate_radices),
                        CircuitLocation::pure(gate_location),
                    ));
                }
                (radices, gate_radices_and_locations)
            })
            .prop_flat_map(|(radices, gate_radices_and_locations)| {
                let mut gate_strats = Vec::new();
                for (grad, gloc) in gate_radices_and_locations {
                    gate_strats.push((
                        Gate::arbitrary_with_radices(grad).unwrap(),
                        Just(gloc),
                    ))
                }
                (Just(radices), gate_strats)
            })
            .prop_map(|(radices, gates_and_locs)| {
                let mut qc = QuditCircuit::new(radices, 0);
                for (gate, loc) in gates_and_locs {
                    qc.append_gate(gate, loc, vec![]);
                }
                qc
            })
    }

    pub fn arbitrary_pure_circuit_with_params<C: ComplexScalar>(
        radices: Option<QuditRadices>,
        min_num_gates: Option<usize>,
        max_num_gates: Option<usize>,
    ) -> impl Strategy<Value = (QuditCircuit<C>, Vec<f64>)> {
        arbitrary_pure_circuit(radices, min_num_gates, max_num_gates)
            .prop_flat_map(|qc| {
                let bounds = qc.get_bounds();
                (Just(qc), bounds)
            })
    }
}

#[cfg(test)]
mod test {
    use faer_core::mat;
    // use proptest::prelude::*;

    // use super::strategies::*;
    use super::*;
    use crate::loc;
    // use crate::math::unitary::function::test::assert_unitary_gradient_function_works;
    // use crate::math::unitary::function::test::assert_unitary_hessian_function_works;
    // use crate::math::unitary::UnitaryMatrix;
    use crate::radices;

    #[test]
    fn test_toffoli_circuit_equals_toffoli_unitary() {
        let mut qc: QuditCircuit<c64> = QuditCircuit::new(radices![2, 2, 2], 0);
        qc.append_gate(Gate::H(2), loc![2], vec![]);
        qc.append_gate(Gate::CX(), loc![1, 2], vec![]);
        qc.append_gate(Gate::Tdg(), loc![2], vec![]);
        qc.append_gate(Gate::CX(), loc![0, 2], vec![]);
        qc.append_gate(Gate::T(), loc![2], vec![]);
        qc.append_gate(Gate::CX(), loc![1, 2], vec![]);
        qc.append_gate(Gate::Tdg(), loc![2], vec![]);
        qc.append_gate(Gate::CX(), loc![0, 2], vec![]);
        qc.append_gate(Gate::T(), loc![1], vec![]);
        qc.append_gate(Gate::T(), loc![2], vec![]);
        qc.append_gate(Gate::CX(), loc![0, 1], vec![]);
        qc.append_gate(Gate::H(2), loc![2], vec![]);
        qc.append_gate(Gate::T(), loc![0], vec![]);
        qc.append_gate(Gate::Tdg(), loc![1], vec![]);
        qc.append_gate(Gate::CX(), loc![0, 1], vec![]);
        // let unitary: UnitaryMatrix<c64> = qc.get_unitary(&[]);
        // Assign a toffoli array to the variable correct
        let _correct = mat![
            [1., 0., 0., 0., 0., 0., 0., 0.],
            [0., 1., 0., 0., 0., 0., 0., 0.],
            [0., 0., 1., 0., 0., 0., 0., 0.],
            [0., 0., 0., 1., 0., 0., 0., 0.],
            [0., 0., 0., 0., 1., 0., 0., 0.],
            [0., 0., 0., 0., 0., 1., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 1.],
            [0., 0., 0., 0., 0., 0., 1., 0.]
        ];
        // unitary.assert_close_to(&correct);
    }

    // proptest! {
    //     #[test]
    //     fn test_gradient((qc, params) in arbitrary_pure_circuit_with_params(None, None, None)) {
    //         assert_unitary_gradient_function_works(qc, &params);
    //     }

    //     #[test]
    //     fn test_hessian((qc, params) in arbitrary_pure_circuit_with_params(None, None, None)) {
    //         assert_unitary_hessian_function_works(qc, &params);
    //     }
    // }
}
