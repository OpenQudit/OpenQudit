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



/// Access and Iteration
impl QuditCircuit {
    /// Try to retrieve an instruction from its id.
    pub fn get(&self, inst_id: InstructionId) -> Option<&Instruction> {
        self.cycles
            .get_from_id(inst_id.cycle())
            .and_then(|cycle| cycle.get_from_id(inst_id.inner()))
    }

    /// Return an iterator over the identifiers of instructions in the circuit.
    ///
    /// The ordering is not guaranteed to be consistent, but it will
    /// be in a simulation/topological order.
    pub fn id_iter(&self) -> impl Iterator<Item = InstructionId> + '_ {
        self.cycles.iter().flat_map(|cycle| {
            cycle
                .id_iter()
                .map(|inner| InstructionId::new(cycle.id, inner))
        })
    }

    /// Return an iterator over the instructions in the circuit.
    ///
    /// The ordering is not guaranteed to be consistent, but it will
    /// be in a simulation/topological order. For more control over the
    /// ordering of iteration see [QuditCircuit::iter_sorted]
    pub fn iter(&self) -> impl Iterator<Item = &Instruction> + '_ {
        self.cycles.iter().flat_map(|cycle| cycle.iter())
    }

    /// Return a sorted iterator over the instructions in the circuit.
    ///
    /// Will always iterate over the instructions in the same order. This
    /// iteration is a valid simulation order.
    pub fn iter_sorted(&self) -> impl Iterator<Item = &Instruction> + '_ {
        self.cycles.iter().flat_map(|cycle| cycle.iter_sorted())
    }
 
    /// Convert a cycle ID to a cycle index
    #[inline]
    pub fn cycle_id_to_index(&self, cycle_id: CycleId) -> Option<CycleIndex> {
        self.cycles.id_to_index(cycle_id)
    }

    /// Convert a cycle index to a cycle ID  
    #[inline]
    pub fn cycle_index_to_id(&self, cycle_index: CycleIndex) -> CycleId {
        self.cycles.index_to_id(cycle_index)
    }
}

