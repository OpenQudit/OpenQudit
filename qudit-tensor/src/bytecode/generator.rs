use std::cell::RefCell;
use std::collections::{HashMap, HashSet, BTreeSet};
use std::rc::Rc;

use super::buffer::TensorBuffer;
use super::{Bytecode, BytecodeInstruction};
use qudit_core::{HasParams, ParamInfo};
use qudit_expr::index::TensorIndex;
use crate::tree::{TTGTTree, TTGTNode, LeafNode, TraceNode, TransposeNode};
use qudit_expr::{ExpressionCache, GenerationShape, TensorExpression, UnitaryExpression};
use qudit_core::QuditSystem;
use qudit_core::TensorShape;

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

    // pub fn from_ttgt_tree(tree: TTGTTree) -> Self {
    //     let TTGTTree { root, expressions } = tree;

    //     Self {
    //         root,
    //         expressions,
    //         const_code: Vec::default(),
    //         dynamic_code: Vec::default(),
    //         buffers: Vec::default(),
    //         const_buffers: BTreeSet::default(),
    //     }
    // }

    pub fn new_buffer(&mut self, shape: GenerationShape, nparams: usize) -> usize {
        let out = self.buffers.len();
        self.buffers.push(TensorBuffer::new(shape, nparams));
        out
    }

    pub fn generate(mut self, tree: TTGTTree) -> Bytecode {
        let TTGTTree { root, expressions } = tree;
        let out_buffer = self.parse(root, &expressions);

        Bytecode {
            // expressions: self.expression_set
            //     .into_iter()
            //     .map(|(e, v)| v.into_iter()
            //         .map(|(p, n)| (e.clone(), p.clone(), n.clone()))
            //         .collect::<Vec<(TensorExpression, Option<ParamInfo>, String)>>()
            //     ).flatten()
            //     .collect(),
            expressions,
            const_code: self.const_code,
            dynamic_code: self.dynamic_code,
            buffers: self.buffers,
            out_buffer,
        }
    }

    fn parse(&mut self, tree: TTGTNode, expressions: &Rc<RefCell<ExpressionCache>>) -> usize {
        // TODO: Look out for potential recalculations:
        //  write u3(a, b, c) -> 0
        //  write u3(a, b, c) -> 1
        //  kron 0 1 2
        //
        //  no need for second write, just make kron: kron 0 0 2

        match tree {
            TTGTNode::Leaf(LeafNode { expr, param_info, indices, .. } ) => {
                let shape = expressions.borrow().generation_shape(expr);
                let out = self.new_buffer(shape, param_info.num_params());
                let constant = param_info.is_empty();
                let inst = BytecodeInstruction::Write(expr, param_info, out.clone());

                if constant {
                    self.const_code.push(inst);
                    self.const_buffers.insert(out.clone());
                } else {
                    self.dynamic_code.push(inst);
                }

                out
            },
            TTGTNode::MatMul(node) => {
                let left_indices = node.left.param_info();
                let right_indices = node.right.param_info();
                let gen_shape = node.indices().into();
                let left = self.parse(*node.left, expressions);
                let right = self.parse(*node.right, expressions);
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
            TTGTNode::Transpose(node) => {
                let num_params = node.param_info().num_params();
                let child_indices = node.child.indices();
                let TransposeNode { child, perm, indices } = node;
                let child_buffer = self.parse(*child, expressions);
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
            TTGTNode::Trace(node) => {
                let num_params = node.param_info().num_params();
                let TraceNode { child, dimension_pairs, indices } = node;
                let child_buffer = self.parse(*child, expressions);
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
            TTGTNode::Outer(node) => {
                let left_indices = node.left.param_info();
                let right_indices = node.right.param_info();
                let gen_shape = node.indices().into();
                let left = self.parse(*node.left, expressions);
                let right = self.parse(*node.right, expressions);
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
            TTGTNode::Hadamard(node) => {
                let left_indices = node.left.param_info();
                let right_indices = node.right.param_info();
                let gen_shape = node.indices().into();
                let left = self.parse(*node.left, expressions);
                let right = self.parse(*node.right, expressions);
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

