use std::collections::HashMap;

use qudit_core::ParamInfo;
use qudit_expr::ExpressionId;

#[derive(Clone)]
pub enum BytecodeInstruction {
    Write(ExpressionId, ParamInfo, usize),
    Matmul(usize, usize, usize, ParamInfo, ParamInfo),
    Kron(usize, usize, usize, ParamInfo, ParamInfo),
    Hadamard(usize, usize, usize, ParamInfo, ParamInfo),
    FRPR(usize, Vec<usize>, Vec<usize>, usize),
    Trace(usize, Vec<(usize, usize)>, usize),
}

impl std::fmt::Debug for BytecodeInstruction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BytecodeInstruction::Write(id, _, index) => {
                write!(f, "Write {} {:?}", id, index)
            }
            BytecodeInstruction::Matmul(a, b, c, _, _) => {
                write!(f, "Matmul {:?} {:?} {:?}", a, b, c)
            }
            BytecodeInstruction::Kron(a, b, c, _, _) => {
                write!(f, "Kron {:?} {:?} {:?}", a, b, c)
            }
            BytecodeInstruction::Hadamard(a, b, c, _, _) => {
                write!(f, "Kron {:?} {:?} {:?}", a, b, c)
            }
            BytecodeInstruction::FRPR(a, _, _, d) => {
                write!(f, "FRPR {:?} {:?}", a, d)
            }
            BytecodeInstruction::Trace(a, _, c) => {
                write!(f, "Trace {:?} {:?}", a, c)
            }
        }
    }
}

impl BytecodeInstruction {
    pub fn offset_buffer_indices(&mut self, offset: usize) {
        match self {
            BytecodeInstruction::Write(_, _, index) => {
                *index += offset;
            }
            BytecodeInstruction::Matmul(a, b, c, _, _) => {
                *a += offset;
                *b += offset;
                *c += offset;
            }
            BytecodeInstruction::Kron(a, b, c, _, _) => {
                *a += offset;
                *b += offset;
                *c += offset;
            }
            BytecodeInstruction::Hadamard(a, b, c, _, _) => {
                *a += offset;
                *b += offset;
                *c += offset;
            }
            BytecodeInstruction::FRPR(a, _, _, d) => {
                *a += offset;
                *d += offset;
            }
            BytecodeInstruction::Trace(a, _, c) => {
                *a += offset;
                *c += offset;
            }
        }
    }

    pub fn replace_buffer_indices(&mut self, buffer_map: &HashMap<usize, usize>) {
        match self {
            BytecodeInstruction::Write(_, _, index) => {
                if let Some(new_index) = buffer_map.get(index) {
                    *index = *new_index;
                }
            }
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
            }
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
            }
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
            }
            BytecodeInstruction::FRPR(a, _, _, d) => {
                if let Some(new_index) = buffer_map.get(a) {
                    *a = *new_index;
                }
                if let Some(new_index) = buffer_map.get(d) {
                    *d = *new_index;
                }
            }
            BytecodeInstruction::Trace(a, _, c) => {
                if let Some(new_index) = buffer_map.get(a) {
                    *a = *new_index;
                }
                if let Some(new_index) = buffer_map.get(c) {
                    *c = *new_index;
                }
            }
        }
    }
}
