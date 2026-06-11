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

    /// Returns the number of unassigned (variable) parameters in the circuit.
    ///
    /// # Performance
    ///
    /// This method is O(p) where
    ///     - `p` is the total number of parameters in the circuit.
    pub fn num_unassigned_params(&self) -> usize {
        self.params
            .iter()
            .filter(|&p| matches!(p, Parameter::Unassigned))
            .count()
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

    /// A reference to the operation set of the circuit.
    pub fn operations(&self) -> &OperationSet {
        &self.operations
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

