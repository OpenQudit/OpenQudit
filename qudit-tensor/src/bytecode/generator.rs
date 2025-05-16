use std::collections::{HashMap, HashSet};

use super::buffer::TensorBuffer;
use super::{Bytecode, GeneralizedInstruction};
use qudit_core::{HasParams, ParamIndices};
use crate::tree::{ExpressionTree, LeafNode};
use qudit_expr::{TensorExpression, TensorGenerationShape, UnitaryExpression};
use qudit_core::QuditSystem;

pub struct BytecodeGenerator {
    expression_set: HashMap<TensorExpression, HashMap<Option<ParamIndices>, String>>,
    static_code: Vec<GeneralizedInstruction>,
    dynamic_code: Vec<GeneralizedInstruction>,
    buffers: Vec<TensorBuffer>,
    static_tree_cache: HashMap<ExpressionTree, usize>,
    gate_fn_counter: usize,
}

impl BytecodeGenerator {
    pub fn new() -> Self {
        Self {
            expression_set: HashMap::new(),
            static_code: Vec::new(),
            dynamic_code: Vec::new(),
            buffers: Vec::new(),
            static_tree_cache: HashMap::new(),
            gate_fn_counter: 0,
        }
    }

    pub fn get_new_buffer(
        &mut self,
        gen_shape: &TensorGenerationShape,
        num_params: usize,
    ) -> usize {
        let out = self.buffers.len();
        println!("Making new buffer with shape: {:?}", gen_shape);
        self.buffers.push(TensorBuffer {
            shape: gen_shape.clone(),
            num_params,
        });
        out
    }

    pub fn generate(mut self, tree: &ExpressionTree) -> Bytecode {
        self.parse(tree);

        Bytecode {
            // expression_set: self.expression_set.into_iter().collect(),
            expressions: self.expression_set
                .into_iter()
                .map(|(e, v)| v.into_iter()
                    .map(|(p, n)| (e.clone(), p.clone(), n.clone()))
                    .collect::<Vec<(TensorExpression, Option<ParamIndices>, String)>>()
                ).flatten()
                .collect(),
            static_code: self.static_code,
            dynamic_code: self.dynamic_code,
            buffers: self.buffers,
            merged_buffers: HashMap::new(),
        }
    }

    pub fn parse(&mut self, tree: &ExpressionTree) -> usize {
        match tree {
            ExpressionTree::Leaf(LeafNode { expr, param_indices } ) => {
                let out = self.get_new_buffer(&expr.generation_shape(), param_indices.num_params());

                // if this expression exists in set then
                //  get name and pass that to instructions
                //  otherwise create new name and add pair to set
                let param_key = if param_indices.is_consecutive() {
                    None
                } else {
                    Some(param_indices.clone())
                };
                let fn_name = if self.expression_set.contains_key(expr) {
                    let expression_map = self.expression_set.get_mut(expr).unwrap();
                    if expression_map.contains_key(&param_key) {
                        expression_map.get(&param_key).unwrap().to_owned()
                    } else {
                        let name = String::from("gate") + &self.gate_fn_counter.to_string();
                        self.gate_fn_counter += 1;
                        expression_map.insert(param_key, name.clone());
                        name
                    }
                } else {
                    let name = String::from("gate") + &self.gate_fn_counter.to_string();
                    self.gate_fn_counter += 1;
                    let mut expression_map = HashMap::new();
                    expression_map.insert(param_key, name.clone());
                    self.expression_set.insert(expr.clone(), expression_map);
                    name
                };

                if param_indices.is_consecutive() {
                    self.dynamic_code.push(GeneralizedInstruction::ConsecutiveParamWrite(
                        fn_name,
                        param_indices.start(),
                        out.clone(),
                    ));
                } else {
                    self.dynamic_code.push(GeneralizedInstruction::SplitParamWrite(
                        fn_name,
                        param_indices.clone(),
                        out.clone(),
                    ));
                }

                out
            },
            ExpressionTree::MatMul(node) => {
                let left = self.parse(&node.left);
                let right = self.parse(&node.right);
                let left_indices = node.left.param_indices();
                let right_indices = node.right.param_indices();
                let overlap = left_indices.intersect(&right_indices);
                let gen_shape = node.generation_shape();
                let out = self.get_new_buffer(
                    &gen_shape, left_indices.num_params() + right_indices.num_params() - overlap.num_params(),
                );
                if overlap.empty() {
                    self.dynamic_code.push(GeneralizedInstruction::DisjointMatmul(
                        left.clone(),
                        right.clone(),
                        out.clone(),
                    ));
                } else {
                    let left_shared_indices = left_indices.iter()
                        .enumerate()
                        .filter_map(|(i, x)| {
                            if overlap.contains(x) {
                                Some(i)
                            } else {
                                None
                            }
                        }).collect::<Vec<_>>();
                    let right_shared_indices = right_indices.iter()
                        .enumerate()
                        .filter_map(|(i, x)| {
                            if overlap.contains(x) {
                                Some(i)
                            } else {
                                None
                            }
                        }).collect::<Vec<_>>();
                    self.dynamic_code.push(GeneralizedInstruction::OverlappingMatmul(
                        left.clone(),
                        right.clone(),
                        out.clone(),
                        left_shared_indices,
                        right_shared_indices,
                    ));
                }
                out
            },
            ExpressionTree::Reshape(node) => {
                let child = self.parse(&node.child);
                let gen_shape = node.generation_shape();
                let out = self.get_new_buffer(
                    &gen_shape, node.param_indices().num_params(),
                );
                self.dynamic_code.push(GeneralizedInstruction::Reshape(
                    child.clone(),
                    out.clone(),
                ));
                out
            },
            ExpressionTree::Transpose(node) => {
                let child = self.parse(&node.child);
                let gen_shape = node.generation_shape();
                let out = self.get_new_buffer(
                    &gen_shape, node.param_indices().num_params(),
                );
                self.dynamic_code.push(GeneralizedInstruction::FRPR(
                    child.clone(),
                    node.child.dimensions().into(),
                    node.perm.clone(),
                    out.clone(),
                ));
                out
            },
            ExpressionTree::OuterProduct(node) => {
                todo!()
            },
            ExpressionTree::Constant(node) => {
                if self.static_tree_cache.contains_key(tree) {
                    return self.static_tree_cache[tree];
                }

                let code = BytecodeGenerator::new().generate(&node.child);

                let buffer_offset = self.buffers.len();
                for buffer in code.buffers {
                    self.buffers.push(buffer);
                }

                assert!(code.static_code.len() == 0);

                for mut inst in code.dynamic_code {
                    inst.offset_buffer_indices(buffer_offset);
                    self.static_code.push(inst);
                }

                let out = self.buffers.len() - 1;
                self.static_tree_cache.insert(tree.clone(), out);
                out
            },
        }
    }
}

pub struct StaticBytecodeOptimizer {
    bytecode: Bytecode,
    #[allow(dead_code)]
    gate_cache: HashMap<UnitaryExpression, usize>,
    replaced_buffers: HashMap<usize, usize>,
}

impl StaticBytecodeOptimizer {
    pub fn new(bytecode: Bytecode) -> Self {
        Self {
            bytecode,
            gate_cache: HashMap::new(),
            replaced_buffers: HashMap::new(),
        }
    }

    pub fn optimize(mut self) -> Bytecode {
        self.deduplicate_gate_gen();
        self.replace_buffers();
        self.bytecode
    }

    fn deduplicate_gate_gen(&mut self) {
        // TODO: This requires unitaryexpression equality, but not sure if it adds value
        // let mut out = Vec::new();
        // for inst in &self.bytecode.static_code {
        //     if let GeneralizedInstruction::Write(gate, param_offset, buffer) =
        //         inst
        //     {
        //         if let Some(index) = self.gate_cache.get(gate) {
        //             self.replaced_buffers.insert(*buffer, *index);
        //         } else {
        //             self.gate_cache.insert(gate.clone(), buffer.clone());
        //             out.push(GeneralizedInstruction::Write(
        //                 gate.clone(),
        //                 *param_offset,
        //                 *buffer,
        //             ));
        //         }
        //     } else {
        //         out.push(inst.clone());
        //     }
        // }

        // self.bytecode.static_code = out;
    }

    fn replace_buffers(&mut self) {
        for inst in &mut self.bytecode.static_code {
            inst.replace_buffer_indices(&self.replaced_buffers);
        }

        for inst in &mut self.bytecode.dynamic_code {
            inst.replace_buffer_indices(&self.replaced_buffers);
        }
    }
}
