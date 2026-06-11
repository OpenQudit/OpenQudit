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
        // self.params.len()
        self.params.num_unassigned()
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

