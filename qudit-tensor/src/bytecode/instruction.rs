use std::collections::HashMap;

use qudit_core::{ComplexScalar, ParamIndices};

#[derive(Clone)]
pub enum BytecodeInstruction {
    ConsecutiveParamWrite(String, usize, usize),
    SplitParamWrite(String, ParamIndices, usize),
    IndependentMatmul(usize, usize, usize),
    DependentMatmul(usize, usize, usize, Vec<usize>, Vec<usize>),
    IndependentKron(usize, usize, usize),
    DependentKron(usize, usize, usize, Vec<usize>, Vec<usize>),
    IndependentHadamard(usize, usize, usize),
    DependentHadamard(usize, usize, usize, Vec<usize>, Vec<usize>),
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
            BytecodeInstruction::IndependentMatmul(a, b, c) => {
                write!(f, "D-Matmul {:?} {:?} {:?}", a, b, c)
            },
            BytecodeInstruction::DependentMatmul(a, b, c, _, _) => {
                write!(f, "O-Matmul {:?} {:?} {:?}", a, b, c)
            },
            BytecodeInstruction::IndependentKron(a, b, c) => {
                write!(f, "D-Kron {:?} {:?} {:?}", a, b, c)
            },
            BytecodeInstruction::DependentKron(a, b, c, _, _) => {
                write!(f, "O-Kron {:?} {:?} {:?}", a, b, c)
            },
            BytecodeInstruction::IndependentHadamard(a, b, c) => {
                write!(f, "D-Kron {:?} {:?} {:?}", a, b, c)
            },
            BytecodeInstruction::DependentHadamard(a, b, c, _, _) => {
                write!(f, "O-Kron {:?} {:?} {:?}", a, b, c)
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
            BytecodeInstruction::DependentMatmul(a, b, c, _, _) => {
                *a += offset;
                *b += offset;
                *c += offset;
            },
            BytecodeInstruction::IndependentMatmul(a, b, c) => {
                *a += offset;
                *b += offset;
                *c += offset;
            },
            BytecodeInstruction::DependentKron(a, b, c, _, _) => {
                *a += offset;
                *b += offset;
                *c += offset;
            },
            BytecodeInstruction::IndependentKron(a, b, c) => {
                *a += offset;
                *b += offset;
                *c += offset;
            },
            BytecodeInstruction::DependentHadamard(a, b, c, _, _) => {
                *a += offset;
                *b += offset;
                *c += offset;
            },
            BytecodeInstruction::IndependentHadamard(a, b, c) => {
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
            BytecodeInstruction::IndependentMatmul(a, b, c) => {
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
            BytecodeInstruction::DependentMatmul(a, b, c, _, _) => {
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
            BytecodeInstruction::IndependentKron(a, b, c) => {
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
            BytecodeInstruction::DependentKron(a, b, c, _, _) => {
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
            BytecodeInstruction::IndependentHadamard(a, b, c) => {
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
            BytecodeInstruction::DependentHadamard(a, b, c, _, _) => {
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
