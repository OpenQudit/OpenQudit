use qudit_core::{ClassicalSystem, HybridSystem, ParamIndices, QuditSystem};
use qudit_core::{HasParams, Radices};

use slotmap::Key;
use slotmap::SlotMap;
use slotmap::new_key_type;
new_key_type! { pub struct CircuitId; }

use crate::circuit::InternableOperation;
use crate::instruction::Instruction;
use crate::operation::OperationSet;
use crate::param::{IntoArgumentList, ParameterVector};
use crate::OpCode;
use crate::Result;


/// A CircuitOperation is a circuit that has been converted to an operation.
///
/// It's parameter vector and operation set are removed and stored directly with
/// the Circuit object. These operations can be hierarchly, in that, a CircuitOperation
/// can have a CircuitOperation as an instruction. The operations and parameters are
/// always stored in the root Circuit object.
///
/// These cannot be constructed manually, but rather, are made implicitly when
/// appending or inserting a circuit into another circuit.
#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub struct CircuitOperation {
    qudit_radices: Radices,
    dit_radices: Radices,
    instructions: Vec<Instruction>,
    num_params: usize,
}

impl CircuitOperation {
    /// Create a new CircuitOperation.
    pub fn new(qudit_radices: Radices, dit_radices: Radices, instructions: Vec<Instruction>, num_params: usize) -> Self {
        CircuitOperation {
            qudit_radices,
            dit_radices,
            instructions,
            num_params,
        }
    }

    /// Specializes a subcircuit by recursively rebinding its internal instructions.
    pub fn specialize(
        self,
        args: crate::param::ArgumentList,
        source_ops: &OperationSet,
        target_ops: &mut OperationSet,
    ) -> Result<Self> {
        if args.len() != self.num_params {
            return Err(crate::Error::IncorrectNumberOfArguments(args.len(), self.num_params));
        }

        let mut specialized_instructions = Vec::with_capacity(self.instructions.len());

        for mut inst in self.instructions {
            // 1. Fetch the original operation from the Source context
            let op = source_ops
                .get(inst.op_code())
                .ok_or(crate::Error::MissingOperation(inst.op_code()))?;

            // 2. Map the sub-arguments for this specific instruction.
            // This takes the circuit operation's args passed here and 
            // selects the ones used by this specific internal instruction.
            let sub_args = args.slice_by_indices(&inst.params());

            // 3. Recurse: Specialize the inner operation.
            // This handles nested subcircuits or expression substitutions.
            let specialized_op = op.specialize(sub_args, source_ops, target_ops)?;

            // 4. Intern the specialized operation into the Target context
            let new_op_code = target_ops.insert(specialized_op)?;

            // 5. Calculate new parameter indices relative to new variable 
            // space created by the specialization of `args`.
            let new_params = args.map_indices_for_instruction(&inst.params());

            // 6. Create and push updated instruction
            let new_inst = Instruction::new(new_op_code, inst.wires(), new_params);
            
            specialized_instructions.push(new_inst);
        }

        let new_num_params = args.parameters().len();

        Ok(Self {
            qudit_radices: self.qudit_radices,
            dit_radices: self.dit_radices,
            instructions: specialized_instructions,
            num_params: new_num_params,
        })
    }
}

impl HasParams for CircuitOperation {
    fn num_params(&self) -> usize {
        self.num_params
    }
}

impl QuditSystem for CircuitOperation {
    fn radices(&self) -> Radices {
        self.qudit_radices.clone()
    }
}

impl ClassicalSystem for CircuitOperation {
    fn radices(&self) -> Radices {
        self.dit_radices.clone()
    }
}

impl HybridSystem for CircuitOperation {}

impl InternableOperation for CircuitOperation {
    fn intern_operation(self, operation_set: &mut OperationSet, parameter_vector: &mut ParameterVector, args: impl IntoArgumentList, qudit_radices: Radices, dit_radices: Radices) -> Result<(OpCode, ParamIndices)> {
        // CircuitOperations are tricky, since you add QuditCircuit objects directly to the
        // circuit. They get converted internally to CircuitOperations. At this point, the
        // information necessary to intern the operation is lost. All we can do at this point, is
        // to ensure that this CircuitOperation has been previously interned properly. So we need
        // to check if this operation exists inside operation_set, otherwise this CircuitOperation
        // corresponds to different circuit. We can also check that qudit_radices and dit_radices
        // matches what we need.
        todo!();
    }
}

impl std::ops::Deref for CircuitOperation {
    type Target = [Instruction];

    fn deref(&self) -> &Self::Target {
        &self.instructions
    }
}


#[derive(Clone)]
pub struct CircuitCache {
    circuits: SlotMap<CircuitId, CircuitOperation>,
}

impl CircuitCache {
    /// Initialize a circuit cache
    pub fn new() -> Self {
        CircuitCache {
            circuits: SlotMap::with_key(),
        }
    }

    /// Insert a circuit into the cache
    pub fn insert(&mut self, circuit: CircuitOperation) -> CircuitId {
        let id = self.circuits.insert(circuit);
        if id.data().as_ffi() & (0b111 << 61) != 0 {
            panic!("CircuitOperation cache overflow.");
        }
        id
    }

    pub fn remove(&mut self, circuit_id: CircuitId) -> Option<CircuitOperation> {
        self.circuits.remove(circuit_id)
    }

    #[allow(dead_code)]
    pub fn get(&self, circuit_id: CircuitId) -> Option<&CircuitOperation> {
        self.circuits.get(circuit_id)
    }

    #[allow(dead_code)]
    pub fn num_params(&self, circuit_id: CircuitId) -> Option<usize> {
        self.get(circuit_id).map(|c| c.num_params())
    }
}
