use std::collections::HashMap;

use qudit_core::{ComplexScalar, ParamIndices};
use qudit_expr::{DifferentiationLevel, Module, ModuleBuilder, TensorExpression};

use super::{instructions::{ConsecutiveParamWriteStruct, DisjointKronStruct, DisjointMatmulStruct, FRPRStruct, OverlappingKronStruct, OverlappingMatMulStruct, SplitParamWriteStruct}, SizedTensorBuffer, SpecializedInstruction};

#[derive(Clone)]
pub enum GeneralizedInstruction {
    ConsecutiveParamWrite(String, usize, usize),
    SplitParamWrite(String, ParamIndices, usize),
    DisjointMatmul(usize, usize, usize),
    OverlappingMatmul(usize, usize, usize, Vec<usize>, Vec<usize>),
    DisjointKron(usize, usize, usize),
    OverlappingKron(usize, usize, usize, Vec<usize>, Vec<usize>),
    FRPR(usize, Vec<usize>, Vec<usize>, usize),
    // Reshape(usize, usize),
}

impl std::fmt::Debug for GeneralizedInstruction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GeneralizedInstruction::ConsecutiveParamWrite(name, _, index) => {
                write!(f, "C-Write {} {:?}", name, index)
            },
            GeneralizedInstruction::SplitParamWrite(name, _, index) => {
                write!(f, "S-Write {} {:?}", name, index)
            },
            GeneralizedInstruction::DisjointMatmul(a, b, c) => {
                write!(f, "D-Matmul {:?} {:?} {:?}", a, b, c)
            },
            GeneralizedInstruction::OverlappingMatmul(a, b, c, _, _) => {
                write!(f, "O-Matmul {:?} {:?} {:?}", a, b, c)
            },
            GeneralizedInstruction::DisjointKron(a, b, c) => {
                write!(f, "D-Kron {:?} {:?} {:?}", a, b, c)
            },
            GeneralizedInstruction::OverlappingKron(a, b, c, _, _) => {
                write!(f, "O-Kron {:?} {:?} {:?}", a, b, c)
            },
            GeneralizedInstruction::FRPR(a, _, _, d) => {
                write!(f, "FRPR {:?} {:?}", a, d)
            },
            // GeneralizedInstruction::Reshape(a, b) => {
            //     write!(f, "Reshape {:?} {:?}", a, b)
            // },
        }
    }
}

impl GeneralizedInstruction {
    pub fn offset_buffer_indices(&mut self, offset: usize) {
        match self {
            GeneralizedInstruction::ConsecutiveParamWrite(_, _, index) => {
                *index += offset;
            },
            GeneralizedInstruction::SplitParamWrite(_, _, index) => {
                *index += offset;
            },
            GeneralizedInstruction::OverlappingMatmul(a, b, c, _, _) => {
                *a += offset;
                *b += offset;
                *c += offset;
            },
            GeneralizedInstruction::DisjointMatmul(a, b, c) => {
                *a += offset;
                *b += offset;
                *c += offset;
            },
            GeneralizedInstruction::OverlappingKron(a, b, c, _, _) => {
                *a += offset;
                *b += offset;
                *c += offset;
            },
            GeneralizedInstruction::DisjointKron(a, b, c) => {
                *a += offset;
                *b += offset;
                *c += offset;
            },
            GeneralizedInstruction::FRPR(a, _, _, d) => {
                *a += offset;
                *d += offset;
            },
            // GeneralizedInstruction::Reshape(a, b) => {
            //     *a += offset;
            //     *b += offset;
            // },
        }
    }

    pub fn replace_buffer_indices(
        &mut self,
        buffer_map: &HashMap<usize, usize>,
    ) {
        match self {
            GeneralizedInstruction::ConsecutiveParamWrite(_, _, index) => {
                if let Some(new_index) = buffer_map.get(index) {
                    *index = *new_index;
                }
            },
            GeneralizedInstruction::SplitParamWrite(_, _, index) => {
                if let Some(new_index) = buffer_map.get(index) {
                    *index = *new_index;
                }
            },
            GeneralizedInstruction::DisjointMatmul(a, b, c) => {
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
            GeneralizedInstruction::OverlappingMatmul(a, b, c, _, _) => {
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
            GeneralizedInstruction::DisjointKron(a, b, c) => {
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
            GeneralizedInstruction::OverlappingKron(a, b, c, _, _) => {
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
            // GeneralizedInstruction::Reshape(a, b) => {
            //     if let Some(new_index) = buffer_map.get(a) {
            //         *a = *new_index;
            //     }
            //     if let Some(new_index) = buffer_map.get(b) {
            //         *b = *new_index;
            //     }
            // },
        }
    }

    // pub fn build_expressions<C: ComplexScalar>(&self, builder: ModuleBuilder<C>) -> ModuleBuilder<C> {
    //     if let GeneralizedInstruction::ConsecutiveParamWrite(expr, _, _) = self {
    //         builder.add_tensor_expression(expr.clone())
    //     } else if let GeneralizedInstruction::SplitParamWrite(expr, param_indices, _) = self {
    //         builder.add_tensor_expression_with_param_indices(expr.clone(), param_indices.clone())
    //     } else {
    //         builder
    //     }
    // }

    pub fn specialize<C: ComplexScalar>(
        &self,
        buffers: &Vec<SizedTensorBuffer>,
        module: &Module<C>,
        diff_lvl: DifferentiationLevel,
    ) -> SpecializedInstruction<C> {
        match self {
            GeneralizedInstruction::ConsecutiveParamWrite(name, start, index) => {
                let (utry_fn, grad_fn) = unsafe {
                    let utry_fn = module.get_function_raw(&name);
                    let grad_fn = if diff_lvl != DifferentiationLevel::None {
                        Some(module.get_function_and_gradient_raw(&name))
                    } else {
                        None
                    };
                    (utry_fn, grad_fn)
                };
                SpecializedInstruction::CWrite(ConsecutiveParamWriteStruct::new(
                    utry_fn,
                    grad_fn,
                    *start,
                    buffers[*index].clone(),
                ))
            },
            GeneralizedInstruction::SplitParamWrite(name, param_indices, index) => {
                let (utry_fn, grad_fn) = unsafe {
                    let utry_fn = module.get_function_raw(&name);
                    let grad_fn = if diff_lvl != DifferentiationLevel::None {
                        Some(module.get_function_and_gradient_raw(&name))
                    } else {
                        None
                    };
                    (utry_fn, grad_fn)
                };
                SpecializedInstruction::SWrite(SplitParamWriteStruct::new(
                    utry_fn,
                    grad_fn,
                    buffers[*index].clone(),
                ))
            },
            GeneralizedInstruction::DisjointMatmul(a, b, c) => {
                let spec_a = buffers[*a].clone();
                let spec_b = buffers[*b].clone();
                let spec_c = buffers[*c].clone();
                SpecializedInstruction::DMatMul(DisjointMatmulStruct::new(
                    spec_a, spec_b, spec_c,
                ))
            },
            GeneralizedInstruction::OverlappingMatmul(a, b, c, a_shared_indices, b_shared_indices) => {
                let spec_a = buffers[*a].clone();
                let spec_b = buffers[*b].clone();
                let spec_c = buffers[*c].clone();
                SpecializedInstruction::OMatMul(OverlappingMatMulStruct::new(
                    spec_a, spec_b, spec_c, a_shared_indices.clone(), b_shared_indices.clone(),
                ))
            },
            GeneralizedInstruction::DisjointKron(a, b, c) => {
                let spec_a = buffers[*a].clone();
                let spec_b = buffers[*b].clone();
                let spec_c = buffers[*c].clone();
                SpecializedInstruction::DKron(DisjointKronStruct::new(
                    spec_a, spec_b, spec_c,
                ))
            },
            GeneralizedInstruction::OverlappingKron(a, b, c, a_shared_indices, b_shared_indices) => {
                let spec_a = buffers[*a].clone();
                let spec_b = buffers[*b].clone();
                let spec_c = buffers[*c].clone();
                SpecializedInstruction::OKron(OverlappingKronStruct::new(
                    spec_a, spec_b, spec_c, a_shared_indices.clone(), b_shared_indices.clone(),
                ))
            },
            GeneralizedInstruction::FRPR(in_index, shape, perm, out_index) => {
                let spec_a = buffers[*in_index].clone();
                let spec_b = buffers[*out_index].clone();
                SpecializedInstruction::FRPR(FRPRStruct::new(
                    spec_a, shape, perm, spec_b,
                ))
            },
            // GeneralizedInstruction::Reshape(in_index, out_index) => {
            //     let spec_a = buffers[*in_index].clone();
            //     let spec_b = buffers[*out_index].clone();
            //     SpecializedInstruction::Reshape(ReshapeStruct::new(spec_a, spec_b))
            // },
        }
    }
}
