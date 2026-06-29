use std::collections::HashMap;
use std::path::Path;

use crate::instruction::{Instruction, InstructionId};
use crate::lang::all_parsers;
use crate::operation::{CircuitOperation, OpCode};
use crate::operation::ExpressionOperation;
use crate::param::{ArgumentList, Parameter};
use crate::{Argument, Operation, Result};
use crate::wire::WireList;
use qudit_core::Radices;
use qudit_core::ParamIndices;
use qudit_expr::{Expression, KetExpression};
use super::*;

pub trait InternableOperation {
    fn intern_operation(self, operation_set: &mut OperationSet, parameter_vector: &mut ParameterVector, args: impl IntoArgumentList, qudit_radices: Radices, dit_radices: Radices) -> Result<(OpCode, ParamIndices)>;
}

impl QuditCircuit {
    /// Append an operation to the circuit
    pub fn append<O, W, A>(&mut self, op: O, wires: W, args: A) -> Result<InstructionId>
    where
        O: InternableOperation,
        W: Into<WireList>,
        A: IntoArgumentList,
    {
        let wires = wires.into();
        let qudit_radices = wires.qudits()
                .map(|d| self.qudit_radices()[d])
                .collect::<Radices>();
        let dit_radices = wires.dits()
                .map(|d| self.dit_radices()[d])
                .collect::<Radices>();
        let (code, params) = op.intern_operation(&mut self.operations, &mut self.params, args, qudit_radices, dit_radices)?;
        Ok(self._append_ref(code, wires, params))

        
        // op.append_to(self, wires, args)
        // let op = op.into();
        // let args = match args.try_into() {
        //     Err(_) => panic!("Get some proper error handling going already..."),
        //     Ok(Some(args)) => args,
        //     Ok(None) => ArgumentList::new(vec![ParameterEntry::Unspecified; op.num_params()]),
        // };
        // // let args: ArgumentList = if args.is_none() {
        // //     ArgumentList::new(vec![ParameterEntry::Unspecified; op.num_params()])
        // // } else {
        // //     match args.unwrap().try_into() {
        // //         Err(_) => panic!("Get some proper error handling going already..."),
        // //         Ok(args) => args,
        // //     }
        // // };

        // match op {
        //     Operation::Expression(e) => self.append_expression(e, wires, args),
        //     Operation::Subcircuit(s) => self._append_subcircuit(s, wires, args),
        //     Operation::Directive(d) => self.append_directive(d, wires, args),
        // }
    }

    fn _append_ref(
        &mut self,
        op_code: OpCode,
        wires: WireList,
        params: ParamIndices,
    ) -> InstructionId {
        // TODO: check valid operation for radix match, measurement bandwidth etc
        // TODO: check params is valid: length is equal to op_params, existing exist, etc..

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

    /// Initialize the qudits specified in a zero state
    pub fn zero_initialize<W: Into<WireList>>(&mut self, wires: W) -> InstructionId {
        let wires = wires.into();
        let location_radices = wires
            .qudits()
            .map(|q| self.qudit_radices[q])
            .collect::<Radices>();
        let state = KetExpression::zero(location_radices);
        let op = ExpressionOperation::QuditInitialization(state);
        self.append(op, wires, None::<ArgumentList>).expect("zero_initialize args are always valid")
    }

    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let ext = path
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("");

        let source = std::fs::read_to_string(path).map_err(|e| crate::Error::GenericError(e.to_string()))?;

        for parser in all_parsers() {
            if parser.supported_extensions().contains(&ext) {
                return parser.parse(&source);
            }
        }

        Err(format!("no parser registered for extension '.{ext}'").into())
    }
}

impl InternableOperation for QuditCircuit {
    fn intern_operation(self, operation_set: &mut OperationSet, parameter_vector: &mut ParameterVector, args: impl IntoArgumentList, qudit_radices: Radices, dit_radices: Radices) -> Result<(OpCode, ParamIndices)> {
        // 1. Align provided args with the circuit's unassigned parameters
        let args = args.into_args(self.num_params())?;
        let mut complete_args = Vec::new();
        let mut unassigned_ptr = 0;

        for parameter in self.params().iter() {
            match parameter {
                Parameter::Assigned32(f) => complete_args.push(Argument::Float32(*f)),
                Parameter::Assigned64(f) => complete_args.push(Argument::Float64(*f)),
                Parameter::AssignedRatio(c) => complete_args.push(Argument::Expression(Expression::Constant(c.clone()))),
                Parameter::Unassigned => {
                    complete_args.push(args[unassigned_ptr].clone());
                    unassigned_ptr += 1;
                },
            }
        }

        let complete_argument_list = ArgumentList::new(complete_args);

        // 2. Flatten and specialize instructions
        let mut flattened_instructions = Vec::new();

        for inst in self.iter() {
            let op = self.operations().get(inst.op_code()).unwrap();
            
            // Get the specific arguments for this instruction
            // e.g. if the instruction used inner parameter [2], we get complete_argument_list[2]
            let op_args = complete_argument_list.slice_by_indices(&inst.params());

            // Specialize: this handles expression updates and nested subcircuits recursively
            let specialized_op = op.specialize(op_args, self.operations(), operation_set)?;
            let global_op_code = operation_set.insert(specialized_op)?;

            // Calculate the new local parameter indices
            let local_params = complete_argument_list.map_indices_for_instruction(&inst.params());

            // Build and push new instruction
            let globalized_instruction = Instruction::new(global_op_code, inst.wires(), local_params);
            flattened_instructions.push(globalized_instruction);
        }

        // 3. Resolve the circuit operation's parameter map
        let params = parameter_vector.parse(&complete_argument_list);
        
        // 4. Finally build circuit operation object and return
        let op = CircuitOperation::new(
            qudit_radices, 
            dit_radices, 
            flattened_instructions, 
            complete_argument_list.parameters().len()
        );

        let op_code = operation_set.insert(Operation::Subcircuit(op))?;
        
        Ok((op_code, params))
    }
}

#[cfg(test)]
mod tests {
    use qudit_expr::library::PGate;
    use crate::QuditCircuit;

    #[test]
    fn test_recursive_complex_subcircuit_append() {
        let p = PGate(2);

        let mut inner_subcircuit = QuditCircuit::pure([2]);
        // inner_subcircuit.append(p, 0, ["theta"]);
        inner_subcircuit.append(p, 0, ());

        println!("{:?}", inner_subcircuit.num_params());


        let mut outer_subcircuit = QuditCircuit::pure([2]);
        // outer_subcircuit.append(inner_subcircuit, 0, ["a + b"]);
        outer_subcircuit.append(inner_subcircuit, 0, ["a + b"]);

        println!("{:?}", outer_subcircuit.num_params());


        let mut global_circuit = QuditCircuit::pure([2]);
        global_circuit.append(outer_subcircuit, 0, ["c*d", "e + g"]);
        // global_circuit.append(outer_subcircuit, 0, ());

        println!("{:?}", global_circuit.num_params());
        println!("{:?}", global_circuit.kraus_ops::<qudit_core::c64>(&[0.5f64, 1.0, 0.25, 0.25]));
    }
}
