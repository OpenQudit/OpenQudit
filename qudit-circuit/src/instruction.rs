use qudit_core::{HasParams, RealScalar};

use crate::{compact::CompactIntegerVector, operation::{Operation, OperationReference}, CircuitLocation};

pub struct Instruction<R: RealScalar> {
    pub op: Operation,
    pub location: CircuitLocation,
    pub params: Vec<R>,
}

impl<R: RealScalar> Instruction<R> {
    pub fn new(op: Operation, location: CircuitLocation, mut params: Vec<R>) -> Instruction<R> {
        if params.len() != op.num_params() {
            for _ in params.len()..op.num_params() {
                params.push(R::zero());
            }
        }
        Instruction {
            op,
            location,
            params,
        }
    }
}

#[derive(Clone, Debug)]
pub struct InstructionReference {
    pub op: OperationReference,
    pub location: CircuitLocation,
    pub param_indices: CompactIntegerVector,
    _packing: u64,
}

impl InstructionReference {
    pub fn new(op: OperationReference, location: CircuitLocation, param_indices: CompactIntegerVector) -> InstructionReference {
        InstructionReference {
            op,
            location,
            param_indices,
            _packing: 0,
        }
    }

    pub fn to_owned(&mut self) -> InstructionReference {
        let op = self.op.clone();
        self.op = OperationReference::from(0);
        let location = self.location.to_owned();
        let param_indices = self.param_indices.to_owned();
        let _packing = self._packing.clone();
        self._packing = 0;
        InstructionReference {
            op,
            location,
            param_indices,
            _packing,
        }
    }
}
