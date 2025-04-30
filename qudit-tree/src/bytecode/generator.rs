use std::collections::{HashMap, HashSet};

use super::MatrixBuffer;
use super::{Bytecode, GeneralizedInstruction};
use qudit_core::HasParams;
use crate::tree::ExpressionTree;
use qudit_expr::UnitaryExpression;
use qudit_core::QuditSystem;

pub struct BytecodeGenerator {
    expression_set: HashSet<UnitaryExpression>,
    static_code: Vec<GeneralizedInstruction>,
    dynamic_code: Vec<GeneralizedInstruction>,
    matrix_buffers: Vec<MatrixBuffer>,
    param_counter: usize,
    static_tree_cache: HashMap<ExpressionTree, usize>,
}

impl BytecodeGenerator {
    pub fn new() -> Self {
        Self {
            expression_set: HashSet::new(),
            static_code: Vec::new(),
            dynamic_code: Vec::new(),
            matrix_buffers: Vec::new(),
            param_counter: 0, // TODO: Handle parameters way better
            static_tree_cache: HashMap::new(),
        }
    }

    pub fn get_new_buffer(
        &mut self,
        nrows: usize,
        ncols: usize,
        num_params: usize,
    ) -> usize {
        let out = self.matrix_buffers.len();
        self.matrix_buffers.push(MatrixBuffer {
            nrows,
            ncols,
            num_params,
        });
        out
    }

    pub fn generate(mut self, tree: &ExpressionTree) -> Bytecode {
        self.parse(tree);

        Bytecode {
            expression_set: self.expression_set.into_iter().collect(),
            static_code: self.static_code,
            dynamic_code: self.dynamic_code,
            matrix_buffers: self.matrix_buffers,
            merged_buffers: HashMap::new(),
        }
    }

    pub fn parse(&mut self, tree: &ExpressionTree) -> usize {
        match tree {
            ExpressionTree::Identity(_) => unreachable!(
                "Identity should not even exist. Like in the code base."
            ),
            ExpressionTree::Kron(n) => {
                let left = self.parse(&n.left);
                let right = self.parse(&n.right);
                // let out = self.get_free_to_clobber(n.get_dimension(), n.get_dimension(), n.get_num_params());
                let out = self.get_new_buffer(
                    n.dimension(),
                    n.dimension(),
                    n.num_params(),
                );
                self.dynamic_code.push(GeneralizedInstruction::Kron(
                    left.clone(),
                    right.clone(),
                    out.clone(),
                ));
                // self.free_buffer(left);
                // self.free_buffer(right);
                out
            },
            ExpressionTree::Mul(n) => {
                let left = self.parse(&n.left);
                let right = self.parse(&n.right);
                // let out = self.get_free_to_clobber(n.get_dimension(), n.get_dimension(), n.get_num_params());
                let out = self.get_new_buffer(
                    n.dimension(),
                    n.dimension(),
                    n.num_params(),
                );
                self.dynamic_code.push(GeneralizedInstruction::Matmul(
                    right.clone(),
                    left.clone(),
                    out.clone(),
                ));
                // self.free_buffer(left);
                // self.free_buffer(right);
                out
            },
            ExpressionTree::Leaf(g) => {
                // let out = self.get_gate_index(g.clone());
                // if g.get_num_params() != 0 && self.len() - 1 != out {
                let out = self.get_new_buffer(
                    g.dimension(),
                    g.dimension(),
                    g.num_params(),
                );
                self.dynamic_code.push(GeneralizedInstruction::Write(
                    g.clone(),
                    self.param_counter,
                    out.clone(),
                ));
                self.param_counter += g.num_params();
                self.expression_set.insert(g.clone());
                // }
                out
            },
            ExpressionTree::Constant(n) => {
                if self.static_tree_cache.contains_key(tree) {
                    return self.static_tree_cache[tree];
                }

                let code = BytecodeGenerator::new().generate(&n.child);

                let buffer_offset = self.matrix_buffers.len();
                for buffer in code.matrix_buffers {
                    self.matrix_buffers.push(buffer);
                }

                assert!(code.static_code.len() == 0);

                for mut inst in code.dynamic_code {
                    inst.offset_buffer_indices(buffer_offset);
                    self.static_code.push(inst);
                }

                for expr in code.expression_set {
                    self.expression_set.insert(expr);
                }

                let out = self.matrix_buffers.len() - 1;
                self.static_tree_cache.insert(tree.clone(), out);
                out
            },
            ExpressionTree::Perm(_n) => {
                unreachable!();
                // let child = self.parse(&n.child);
                // let out = self.get_free_to_clobber(n.get_dimension(), n.get_dimension(), n.get_num_params());
                // TODO: let (ins, outs, pshape) = n.get_permutation().as_frpr();
                // self.bytecode.push(GeneralizedInstruction::FRPR(ins, outs, pshape, child.clone(), out.clone()));
                // self.free_buffer(child);
                // out
            },
            ExpressionTree::Contract(n) => {
                let mut left = self.parse(&n.left);
                let mut right = self.parse(&n.right);

                if !n.skip_left {
                    let out = self.get_new_buffer(
                        n.left_contraction_shape.0,
                        n.left_contraction_shape.1,
                        n.left.num_params(),
                    );
                    self.dynamic_code.push(GeneralizedInstruction::FRPR(
                        left.clone(),
                        n.left_tensor_shape.clone().into_iter().map(|x| x.try_into().unwrap()).collect(),
                        n.left_perm.clone(),
                        out.clone(),
                    ));
                    // self.free_buffer(left);
                    left = out;
                }

                if !n.skip_right {
                    let out = self.get_new_buffer(
                        n.right_contraction_shape.0,
                        n.right_contraction_shape.1,
                        n.right.num_params(),
                    );
                    self.dynamic_code.push(GeneralizedInstruction::FRPR(
                        right.clone(),
                        n.right_tensor_shape.clone().into_iter().map(|x| x.try_into().unwrap()).collect(),
                        n.right_perm.clone(),
                        out.clone(),
                    ));
                    // self.free_buffer(right);
                    right = out;
                }

                let pre_out = self.get_new_buffer(
                    n.right_contraction_shape.0,
                    n.left_contraction_shape.1,
                    n.num_params(),
                );
                self.dynamic_code.push(GeneralizedInstruction::Matmul(
                    right.clone(),
                    left.clone(),
                    pre_out.clone(),
                ));
                // self.free_buffer(left);
                // self.free_buffer(right);

                let out = self.get_new_buffer(
                    n.out_matrix_shape.0,
                    n.out_matrix_shape.1,
                    n.num_params(),
                );
                self.dynamic_code.push(GeneralizedInstruction::FRPR(
                    pre_out.clone(),
                    n.pre_out_tensor_shape.clone().into_iter().map(|x| x.try_into().unwrap()).collect(),
                    n.pre_out_perm.clone().into_iter().map(|x| x.try_into().unwrap()).collect(),
                    out.clone(),
                ));
                // self.free_buffer(pre_out);
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
