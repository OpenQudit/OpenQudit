use std::collections::HashMap;

use qudit_core::{ComplexScalar, ParamIndices};

#[derive(Clone)]
pub enum BytecodeInstruction {
    ConsecutiveParamWrite(String, usize, usize),
    SplitParamWrite(String, ParamIndices, usize),
    Matmul(usize, usize, usize, ParamIndices, ParamIndices),
    Kron(usize, usize, usize, ParamIndices, ParamIndices),
    Hadamard(usize, usize, usize, ParamIndices, ParamIndices),
    FRPR(usize, Vec<usize>, Vec<usize>, usize),
    Trace(usize, Vec<(usize, usize)>, usize),
}

impl std::fmt::Debug for BytecodeInstruction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BytecodeInstruction::ConsecutiveParamWrite(name, _, index) => {
                write!(f, "C-Write {} {:?}", name, index)
            },
            BytecodeInstruction::SplitParamWrite(name, _, index) => {
                write!(f, "S-Write {} {:?}", name, index)
            },
            BytecodeInstruction::Matmul(a, b, c, _, _) => {
                write!(f, "Matmul {:?} {:?} {:?}", a, b, c)
            },
            BytecodeInstruction::Kron(a, b, c, _, _) => {
                write!(f, "Kron {:?} {:?} {:?}", a, b, c)
            },
            BytecodeInstruction::Hadamard(a, b, c, _, _) => {
                write!(f, "Kron {:?} {:?} {:?}", a, b, c)
            },
            BytecodeInstruction::FRPR(a, _, _, d) => {
                write!(f, "FRPR {:?} {:?}", a, d)
            },
            BytecodeInstruction::Trace(a, _, c) => {
                write!(f, "Trace {:?} {:?}", a, c)
            },
        }
    }
}

impl BytecodeInstruction {
    pub fn offset_buffer_indices(&mut self, offset: usize) {
        match self {
            BytecodeInstruction::ConsecutiveParamWrite(_, _, index) => {
                *index += offset;
            },
            BytecodeInstruction::SplitParamWrite(_, _, index) => {
                *index += offset;
            },
            BytecodeInstruction::Matmul(a, b, c, _, _) => {
                *a += offset;
                *b += offset;
                *c += offset;
            },
            BytecodeInstruction::Kron(a, b, c, _, _) => {
                *a += offset;
                *b += offset;
                *c += offset;
            },
            BytecodeInstruction::Hadamard(a, b, c, _, _) => {
                *a += offset;
                *b += offset;
                *c += offset;
            },
            BytecodeInstruction::FRPR(a, _, _, d) => {
                *a += offset;
                *d += offset;
            },
            BytecodeInstruction::Trace(a, _, c) => {
                *a += offset;
                *c += offset;
            },
        }
    }

    pub fn replace_buffer_indices(
        &mut self,
        buffer_map: &HashMap<usize, usize>,
    ) {
        match self {
            BytecodeInstruction::ConsecutiveParamWrite(_, _, index) => {
                if let Some(new_index) = buffer_map.get(index) {
                    *index = *new_index;
                }
            },
            BytecodeInstruction::SplitParamWrite(_, _, index) => {
                if let Some(new_index) = buffer_map.get(index) {
                    *index = *new_index;
                }
            },
            BytecodeInstruction::Matmul(a, b, c, _, _) => {
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
            BytecodeInstruction::Kron(a, b, c, _, _) => {
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
            BytecodeInstruction::Hadamard(a, b, c, _, _) => {
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
            BytecodeInstruction::FRPR(a, _, _, d) => {
                if let Some(new_index) = buffer_map.get(a) {
                    *a = *new_index;
                }
                if let Some(new_index) = buffer_map.get(d) {
                    *d = *new_index;
                }
            },
            BytecodeInstruction::Trace(a, _, c) => {
                if let Some(new_index) = buffer_map.get(a) {
                    *a = *new_index;
                }
                if let Some(new_index) = buffer_map.get(c) {
                    *c = *new_index;
                }
            },
        }
    }
}
