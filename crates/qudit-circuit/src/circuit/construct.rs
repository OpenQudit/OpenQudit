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
