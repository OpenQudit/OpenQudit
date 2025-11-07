use std::collections::BTreeSet;
use std::sync::{Arc, Mutex};

use super::buffer::TensorBuffer;
use super::{Bytecode, BytecodeInstruction};
use crate::tree::{LeafNode, TTGTNode, TTGTTree, TraceNode, TransposeNode};
use qudit_expr::{ExpressionCache, GenerationShape};

#[derive(Default)]
pub struct BytecodeGenerator {
    const_code: Vec<BytecodeInstruction>,
    dynamic_code: Vec<BytecodeInstruction>,
    buffers: Vec<TensorBuffer>,
    const_buffers: BTreeSet<usize>,
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

    pub fn generate(mut self, tree: TTGTTree) -> Bytecode {
        let TTGTTree { root, expressions } = tree;
        let (out_buffer, num_params) = self.parse(root, &expressions);

        Bytecode {
            expressions,
            const_code: self.const_code,
            dynamic_code: self.dynamic_code,
            buffers: self.buffers,
            out_buffer,
            num_params,
        }
    }

    fn parse(
        &mut self,
        tree: TTGTNode,
        expressions: &Arc<Mutex<ExpressionCache>>,
    ) -> (usize, usize) {
        // TODO: Look out for potential recalculations:
        //  write u3(a, b, c) -> 0
        //  write u3(a, b, c) -> 1
        //  kron 0 1 2
        //
        //  no need for second write, just make kron: kron 0 0 2

        match tree {
            TTGTNode::Leaf(LeafNode {
                expr, param_info, ..
            }) => {
                let shape = expressions.lock().unwrap().generation_shape(expr);
                let out = self.new_buffer(shape, param_info.num_var_params());
                let total_num_params = param_info.num_params();
                let constant = param_info.is_empty();
                let inst = BytecodeInstruction::Write(expr, param_info, out);

                if constant {
                    self.const_code.push(inst);
                    self.const_buffers.insert(out);
                } else {
                    self.dynamic_code.push(inst);
                }

                (out, total_num_params)
            }
            TTGTNode::MatMul(node) => {
                let left_indices = node.left.param_info();
                let right_indices = node.right.param_info();
                let gen_shape = node.indices().into();
                let (left, _) = self.parse(*node.left, expressions);
                let (right, _) = self.parse(*node.right, expressions);
                let total_num_params = left_indices.union(&right_indices).num_params();
                let overlap = left_indices.intersect(&right_indices);
                let out = self.new_buffer(
                    gen_shape,
                    left_indices.num_var_params() + right_indices.num_var_params()
                        - overlap.num_var_params(),
                );
                let instruction =
                    BytecodeInstruction::Matmul(left, right, out, left_indices, right_indices);
                if self.const_buffers.contains(&left) && self.const_buffers.contains(&right) {
                    self.const_code.push(instruction);
                    self.const_buffers.insert(out);
                } else {
                    self.dynamic_code.push(instruction);
                }
                (out, total_num_params)
            }
            TTGTNode::Transpose(node) => {
                let num_params = node.param_info().num_var_params();
                let child_indices = node.child.indices();
                let TransposeNode {
                    child,
                    perm,
                    indices,
                } = node;
                let (child_buffer, total_num_params) = self.parse(*child, expressions);
                let out_buffer = self.new_buffer(indices.into(), num_params);
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
                (out_buffer, total_num_params)
            }
            TTGTNode::Trace(node) => {
                let num_params = node.param_info().num_var_params();
                let TraceNode {
                    child,
                    dimension_pairs,
                    indices,
                } = node;
                let (child_buffer, total_num_params) = self.parse(*child, expressions);
                let out_buffer = self.new_buffer(indices.into(), num_params);
                let instruction =
                    BytecodeInstruction::Trace(child_buffer, dimension_pairs, out_buffer);
                if self.const_buffers.contains(&child_buffer) {
                    self.const_code.push(instruction);
                    self.const_buffers.insert(out_buffer);
                } else {
                    self.dynamic_code.push(instruction);
                }
                (out_buffer, total_num_params)
            }
            TTGTNode::Outer(node) => {
                let left_indices = node.left.param_info();
                let right_indices = node.right.param_info();
                let gen_shape = node.indices().into();
                let (left, _) = self.parse(*node.left, expressions);
                let (right, _) = self.parse(*node.right, expressions);
                let total_num_params = left_indices.union(&right_indices).num_params();
                let overlap = left_indices.intersect(&right_indices);
                let out = self.new_buffer(
                    gen_shape,
                    left_indices.num_var_params() + right_indices.num_var_params()
                        - overlap.num_var_params(),
                );
                let instruction =
                    BytecodeInstruction::Kron(left, right, out, left_indices, right_indices);
                if self.const_buffers.contains(&left) && self.const_buffers.contains(&right) {
                    self.const_code.push(instruction);
                    self.const_buffers.insert(out);
                } else {
                    self.dynamic_code.push(instruction);
                }
                (out, total_num_params)
            }
            TTGTNode::Hadamard(node) => {
                let left_indices = node.left.param_info();
                let right_indices = node.right.param_info();
                let gen_shape = node.indices().into();
                let (left, _) = self.parse(*node.left, expressions);
                let (right, _) = self.parse(*node.right, expressions);
                let total_num_params = left_indices.union(&right_indices).num_params();
                let overlap = left_indices.intersect(&right_indices);
                let out = self.new_buffer(
                    gen_shape,
                    left_indices.num_var_params() + right_indices.num_var_params()
                        - overlap.num_var_params(),
                );
                let instruction =
                    BytecodeInstruction::Hadamard(left, right, out, left_indices, right_indices);
                if self.const_buffers.contains(&left) && self.const_buffers.contains(&right) {
                    self.const_code.push(instruction);
                    self.const_buffers.insert(out);
                } else {
                    self.dynamic_code.push(instruction);
                }
                (out, total_num_params)
            }
        }
    }
}
