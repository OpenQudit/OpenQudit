use std::collections::{HashMap, HashSet, BTreeSet};

use super::buffer::TensorBuffer;
use super::{Bytecode, BytecodeInstruction};
use qudit_core::{HasParams, ParamIndices};
use crate::tree::{ExpressionTree, LeafNode, TraceNode, TransposeNode};
use qudit_expr::{GenerationShape, TensorExpression, UnitaryExpression};
use qudit_core::QuditSystem;
use qudit_core::TensorShape;

#[derive(Default)]
pub struct BytecodeGenerator {
    expression_set: HashMap<TensorExpression, HashMap<Option<ParamIndices>, String>>,
    const_code: Vec<BytecodeInstruction>,
    dynamic_code: Vec<BytecodeInstruction>,
    buffers: Vec<TensorBuffer>,
    const_buffers: BTreeSet<usize>,
    gate_fn_counter: usize,
}

impl BytecodeGenerator {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn new_buffer(&mut self, shape: GenerationShape, nparams: usize) -> usize {
        let out = self.buffers.len();
        self.buffers.push(TensorBuffer::new(shape, nparams));
        out
    }

    pub fn generate(mut self, tree: ExpressionTree) -> Bytecode {
        self.parse(tree);

        Bytecode {
            expressions: self.expression_set
                .into_iter()
                .map(|(e, v)| v.into_iter()
                    .map(|(p, n)| (e.clone(), p.clone(), n.clone()))
                    .collect::<Vec<(TensorExpression, Option<ParamIndices>, String)>>()
                ).flatten()
                .collect(),
            const_code: self.const_code,
            dynamic_code: self.dynamic_code,
            buffers: self.buffers,
        }
    }

    pub fn parse(&mut self, tree: ExpressionTree) -> usize {
        match tree {
            ExpressionTree::Leaf(LeafNode { expr, param_indices } ) => {
                let out = self.new_buffer(expr.generation_shape(), param_indices.num_params());

                // if this expression exists in set then
                //  get name and pass that to instructions
                //  otherwise create new name and add pair to set
                let param_key = if param_indices.is_consecutive() {
                    None
                } else {
                    Some(param_indices.clone())
                };
                // TODO: Drop the param key nonsense, pass param_map directly and let the TNVM
                // figure it out; Also if the param_map and gate combo has already been written
                // too, then just reuse results
                //  write u3(a, b, c) -> 0
                //  write u3(a, b, c) -> 1
                //  kron 0 1 2
                //
                //  no need for second write, just make kron: kron 0 0 2
                let fn_name = if self.expression_set.contains_key(&expr) {
                    let expression_map = self.expression_set.get_mut(&expr).unwrap();
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

                if param_indices.is_empty() {
                    self.const_code.push(BytecodeInstruction::ConsecutiveParamWrite(fn_name, param_indices.start(), out.clone()));
                    self.const_buffers.insert(out.clone());
                } else if param_indices.is_consecutive() {
                    self.dynamic_code.push(BytecodeInstruction::ConsecutiveParamWrite(
                        fn_name,
                        param_indices.start(),
                        out.clone(),
                    ));
                } else {
                    self.dynamic_code.push(BytecodeInstruction::SplitParamWrite(
                        fn_name,
                        param_indices.clone(),
                        out.clone(),
                    ));
                }

                out
            },
            ExpressionTree::MatMul(node) => {
                let left_indices = node.left.param_indices();
                let right_indices = node.right.param_indices();
                let gen_shape = node.indices().into();
                let left = self.parse(*node.left);
                let right = self.parse(*node.right);
                let overlap = left_indices.intersect(&right_indices);
                let out = self.new_buffer(
                    gen_shape,
                    left_indices.num_params() + right_indices.num_params() - overlap.num_params(),
                );
                let instruction = BytecodeInstruction::Matmul(
                    left,
                    right,
                    out,
                    left_indices,
                    right_indices,
                );
                if self.const_buffers.contains(&left) && self.const_buffers.contains(&right) {
                    self.const_code.push(instruction);
                    self.const_buffers.insert(out);
                } else {
                    self.dynamic_code.push(instruction);
                }
                out
            },
            ExpressionTree::Transpose(node) => {
                let num_params = node.param_indices().num_params();
                let child_indices = node.child.indices();
                let TransposeNode { child, perm, indices } = node;
                let child_buffer = self.parse(*child);
                let out_buffer = self.new_buffer(
                    indices.into(),
                    num_params,
                );
                let instruction = BytecodeInstruction::FRPR(
                    child_buffer,
                    child_indices.iter().map(|idx| idx.index_size()).collect(),
                    perm,
                    out_buffer,
                );
                if self.const_buffers.contains(&child_buffer) {
                    self.const_code.push(instruction);
                    self.const_buffers.insert(out_buffer);
                } else {
                    self.dynamic_code.push(instruction);
                }
               out_buffer 
            },
            ExpressionTree::Trace(node) => {
                let num_params = node.param_indices().num_params();
                let TraceNode { child, dimension_pairs, indices } = node;
                let child_buffer = self.parse(*child);
                let out_buffer = self.new_buffer(
                    indices.into(),
                    num_params,
                );
                let instruction = BytecodeInstruction::Trace(
                    child_buffer,
                    dimension_pairs,
                    out_buffer,
                );
                if self.const_buffers.contains(&child_buffer) {
                    self.const_code.push(instruction);
                    self.const_buffers.insert(out_buffer);
                } else {
                    self.dynamic_code.push(instruction);
                }
                out_buffer
            },
            ExpressionTree::Outer(node) => {
                let left_indices = node.left.param_indices();
                let right_indices = node.right.param_indices();
                let gen_shape = node.indices().into();
                let left = self.parse(*node.left);
                let right = self.parse(*node.right);
                let overlap = left_indices.intersect(&right_indices);
                let out = self.new_buffer(
                    gen_shape,
                    left_indices.num_params() + right_indices.num_params() - overlap.num_params(),
                );
                let instruction = BytecodeInstruction::Kron(
                    left,
                    right,
                    out,
                    left_indices,
                    right_indices,
                );
                if self.const_buffers.contains(&left) && self.const_buffers.contains(&right) {
                    self.const_code.push(instruction);
                    self.const_buffers.insert(out);
                } else {
                    self.dynamic_code.push(instruction);
                }
                out
            },
            ExpressionTree::Hadamard(node) => {
                let left_indices = node.left.param_indices();
                let right_indices = node.right.param_indices();
                let gen_shape = node.indices().into();
                let left = self.parse(*node.left);
                let right = self.parse(*node.right);
                let overlap = left_indices.intersect(&right_indices);
                let out = self.new_buffer(
                    gen_shape,
                    left_indices.num_params() + right_indices.num_params() - overlap.num_params(),
                );
                let instruction = BytecodeInstruction::Hadamard(
                    left,
                    right,
                    out,
                    left_indices,
                    right_indices,
                );
                if self.const_buffers.contains(&left) && self.const_buffers.contains(&right) {
                    self.const_code.push(instruction);
                    self.const_buffers.insert(out);
                } else {
                    self.dynamic_code.push(instruction);
                }
                out
            },
        }
    }
}

