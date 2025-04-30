use std::collections::HashMap;

use crate::{gates::NamedGate, Gate};

use super::{
    instructions::{FRPRStruct, KronStruct, MatmulStruct, WriteStruct},
    SizedMatrixBuffer, SpecializedInstruction,
};

#[derive(Clone)]
pub enum GeneralizedInstruction {
    Write(Gate, usize, usize),
    Matmul(usize, usize, usize),
    Kron(usize, usize, usize),
    FRPR(usize, Vec<usize>, Vec<usize>, usize),
}

impl std::fmt::Debug for GeneralizedInstruction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GeneralizedInstruction::Write(gate, _, index) => {
                write!(f, "Write {} {:?}", gate.get_name(), index)
            },
            GeneralizedInstruction::Matmul(a, b, c) => {
                write!(f, "Matmul {:?} {:?} {:?}", a, b, c)
            },
            GeneralizedInstruction::Kron(a, b, c) => {
                write!(f, "Kron {:?} {:?} {:?}", a, b, c)
            },
            GeneralizedInstruction::FRPR(a, _, _, d) => {
                write!(f, "FRPR {:?} {:?}", a, d)
            },
        }
    }
}

impl GeneralizedInstruction {
    pub fn offset_buffer_indices(&mut self, offset: usize) {
        match self {
            GeneralizedInstruction::Write(_, _, index) => {
                *index += offset;
            },
            GeneralizedInstruction::Matmul(a, b, c) => {
                *a += offset;
                *b += offset;
                *c += offset;
            },
            GeneralizedInstruction::Kron(a, b, c) => {
                *a += offset;
                *b += offset;
                *c += offset;
            },
            GeneralizedInstruction::FRPR(a, _, _, d) => {
                *a += offset;
                *d += offset;
            },
        }
    }

    pub fn replace_buffer_indices(
        &mut self,
        buffer_map: &HashMap<usize, usize>,
    ) {
        match self {
            GeneralizedInstruction::Write(_, _, index) => {
                if let Some(new_index) = buffer_map.get(index) {
                    *index = *new_index;
                }
            },
            GeneralizedInstruction::Matmul(a, b, c) => {
                if let Some(new_index) = buffer_map.get(a) {
                    *a = *new_index;
                }
                if let Some(new_index) = buffer_map.get(b) {
                    *b = *new_index;
                }
                if let Some(new_index) = buffer_map.get(c) {
                    *c = *new_index;
                }
            },
            GeneralizedInstruction::Kron(a, b, c) => {
                if let Some(new_index) = buffer_map.get(a) {
                    *a = *new_index;
                }
                if let Some(new_index) = buffer_map.get(b) {
                    *b = *new_index;
                }
                if let Some(new_index) = buffer_map.get(c) {
                    *c = *new_index;
                }
            },
            GeneralizedInstruction::FRPR(a, _, _, d) => {
                if let Some(new_index) = buffer_map.get(a) {
                    *a = *new_index;
                }
                if let Some(new_index) = buffer_map.get(d) {
                    *d = *new_index;
                }
            },
        }
    }

    pub fn specialize(
        &self,
        buffers: &Vec<SizedMatrixBuffer>,
    ) -> SpecializedInstruction {
        match self {
            GeneralizedInstruction::Write(gate, param_pointer, index) => {
                SpecializedInstruction::Write(WriteStruct::new(
                    gate.clone(),
                    *param_pointer,
                    buffers[*index].clone(),
                ))
            },
            GeneralizedInstruction::Matmul(a, b, c) => {
                let spec_a = buffers[*a].clone();
                let spec_b = buffers[*b].clone();
                let spec_c = buffers[*c].clone();
                SpecializedInstruction::Matmul(MatmulStruct::new(
                    spec_a, spec_b, spec_c,
                ))
            },
            GeneralizedInstruction::Kron(a, b, c) => {
                let spec_a = buffers[*a].clone();
                let spec_b = buffers[*b].clone();
                let spec_c = buffers[*c].clone();
                SpecializedInstruction::Kron(KronStruct::new(
                    spec_a, spec_b, spec_c,
                ))
            },
            GeneralizedInstruction::FRPR(in_index, shape, perm, out_index) => {
                let spec_a = buffers[*in_index].clone();
                let spec_b = buffers[*out_index].clone();
                SpecializedInstruction::FRPR(FRPRStruct::new(
                    spec_a, shape, perm, spec_b,
                ))
            },
        }
    }
}
