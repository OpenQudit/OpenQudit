use crate::cycle::CycleList;
use crate::cycle::{CycleId, CycleIndex};
use crate::instruction::{Instruction, InstructionId};
use crate::operation::OpCode;
use crate::operation::OperationSet;
use crate::operation::{
    CircuitOperation, DirectiveOperation, ExpressionOperation, OpKind, Operation,
};
use crate::Result;
use crate::param::{Argument as ParameterEntry, ArgumentList, Parameter, ParameterId, ParameterVector};
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
            network = self.add_instruction_to_builder(
                network, 
                inst.op_code(), 
                inst.wires(), 
                inst.params(), 
            ).expect("TODO");
        }
       network 
    }

    fn add_instruction_to_builder(
        &self,
        mut network: QuditCircuitTensorNetworkBuilder,
        op_code: OpCode,
        wires: WireList,
        params: ParamIndices,
    ) -> Result<QuditCircuitTensorNetworkBuilder> {

        // Convert an expression operation to an tensor and add it to the network
        if op_code.kind() == OpKind::Expression {

            // Collect the current parameter information for this tensor.
            let param_indices = self.params.convert_ids_to_indices(params);
            let constant = param_indices
                .iter()
                .map(|i| self.params[i].is_assigned())
                .collect();
            let param_info = ParamInfo::new(param_indices, constant);

            // Collect the tensor indices/leg information for this tensor.
            let indices = self.operations.indices(op_code);
            let input_index_map = if indices
                .iter()
                .any(|idx| idx.direction() == IndexDirection::Input && idx.index_size() > 1)
            {
                wires.qudits().collect()
            } else {
                vec![]
            };
            let output_index_map = if indices
                .iter()
                .any(|idx| idx.direction() == IndexDirection::Output && idx.index_size() > 1)
            {
                wires.qudits().collect()
            } else {
                vec![]
            };
            let batch_index_map: Vec<String> =
            wires.dits().map(|id| id.to_string()).collect();
            let tensor = QuditTensor::new(indices, op_code.id(), param_info);
            // println!("Adding new tensor {} to network builder with in qudits: {:?}; out qudits: {:?}, batch indices: {:?}", self.operations.name(inst.op_code()), input_index_map.clone(), output_index_map.clone(), batch_index_map.clone());
            network =
                network.prepend(tensor, input_index_map, output_index_map, batch_index_map);
            
            Ok(network)
        }
        else
        {
            let op = self.operations.get(op_code).ok_or(crate::Error::MissingOperation(op_code))?;

            match op {
                Operation::Expression(_) => unreachable!("Already handled expressions."),

                Operation::Subcircuit(sub) => {
                    for sub_inst in sub.iter() {
                        let mapped_wires = sub_inst.wires().map_through(&wires);
                        let mapped_params = sub_inst.params().map_through(&params);
                        network = self.add_instruction_to_builder(
                            network,
                            sub_inst.op_code(),
                            mapped_wires,
                            mapped_params,
                        )?;
                    }
                    Ok(network)
                }

                // Directives don't contribute to the tensor network evaluation of a circuit.
                Operation::Directive(_) => Ok(network),
            }
        }
    }
}

