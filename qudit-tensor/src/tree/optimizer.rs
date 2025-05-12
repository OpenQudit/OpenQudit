use super::constant::ConstantNode;
use super::leaf::LeafNode;
use super::matmul::MatMulNode;
use super::outer::OuterProductNode;
use super::reshape::ReshapeNode;
use super::transpose::TransposeNode;
use super::ExpressionTree;
use qudit_core::HasParams;

pub struct TreeOptimizer {}

impl TreeOptimizer {
    pub fn new() -> Self {
        Self {}
    }

    pub fn optimize(&self, mut tree: ExpressionTree) -> ExpressionTree {
        // tree = self.fuse_common_operations(tree);
        // tree.traverse_mut(&|n| self.fuse_contraction_pre_post_permutations(n));
        // self.constant_propagation(&mut tree);
        tree = self.remove_no_op_transposes(tree);
        tree
    }

    pub fn remove_no_op_transposes(&self, tree: ExpressionTree) -> ExpressionTree {
        match tree {
            ExpressionTree::Leaf(_) => tree,
            ExpressionTree::Constant(ConstantNode { child }) => self.remove_no_op_transposes(*child),
            ExpressionTree::MatMul(MatMulNode { left, right }) => {
                let left = self.remove_no_op_transposes(*left);
                let right = self.remove_no_op_transposes(*right);
                ExpressionTree::MatMul(MatMulNode::new(left, right))
            }
            ExpressionTree::OuterProduct(OuterProductNode { left, right }) => {
                let left = self.remove_no_op_transposes(*left);
                let right = self.remove_no_op_transposes(*right);
                ExpressionTree::OuterProduct(OuterProductNode::new(left, right))
            }
            ExpressionTree::Reshape(ReshapeNode { child, gen_shape }) => {
                let child = self.remove_no_op_transposes(*child);
                if gen_shape == child.generation_shape() {
                    // if the generation shape is the same, then we can just return the child
                    child
                } else {
                    // otherwise, we need to keep the reshape
                    ExpressionTree::reshape(child, gen_shape)
                }
            }
            ExpressionTree::Transpose(TransposeNode { child, perm, split_at, dimensions, generation_shape }) => {
                let child = self.remove_no_op_transposes(*child);
                if perm.is_sorted() {
                    // if the permutation is sorted, then we can remove the transpose
                    // and just return the child
                    match child {
                        ExpressionTree::Leaf(mut node) => {
                            node.set_generation_shape(generation_shape);
                            ExpressionTree::Leaf(node)
                        }
                        ExpressionTree::Transpose(mut node) => {
                            node.set_generation_shape(generation_shape);
                            ExpressionTree::Transpose(node)
                        }
                        ExpressionTree::Reshape(mut node) => {
                            node.set_generation_shape(generation_shape);
                            ExpressionTree::Reshape(node)
                        }
                        _ => {
                            if generation_shape == child.generation_shape() {
                                // if the generation shape is the same, then we can just return the child
                                child
                            } else {
                                // otherwise, we need to keep the reshape
                                ExpressionTree::reshape(child, generation_shape)
                            }
                        }
                    }
                } else {
                    // otherwise, we need to keep the transpose
                    ExpressionTree::Transpose(TransposeNode::new(child, perm, split_at))
                }
            }
        }
    }

    // fn fuse_common_operations(&self, tree: ExpressionTree) -> ExpressionTree {
    //     // traverse the tree, if all children of a kron or mul node or also kron, mul, or leaf then
    //     // fuse; not a good algorithm; TODO: be better...
    //     match tree {
    //         ExpressionTree::Identity(_) => tree,
    //         ExpressionTree::Kron(n) => {
    //             let left = self.fuse_common_operations(*n.left);
    //             let right = self.fuse_common_operations(*n.right);
    //             // if we can fuse, then both left and right are leafs
    //             if let (ExpressionTree::Leaf(left), ExpressionTree::Leaf(right)) = (&left, &right) {
    //                 ExpressionTree::Leaf(left.otimes(right))
    //             } else {
    //                 ExpressionTree::Kron(KronNode::new(left, right))
    //             }
    //         },
    //         ExpressionTree::Mul(n) => {
    //             let left = self.fuse_common_operations(*n.left);
    //             let right = self.fuse_common_operations(*n.right);
    //             // if we can fuse, then both left and right are leafs
    //             if let (ExpressionTree::Leaf(left), ExpressionTree::Leaf(right)) = (&left, &right) {
    //                 ExpressionTree::Leaf(right.dot(left))
    //             } else {
    //                 ExpressionTree::Mul(MulNode::new(left, right))
    //             }
    //         },
    //         ExpressionTree::Leaf(_) => tree,
    //         ExpressionTree::Constant(_) => tree,
    //         ExpressionTree::Perm(n) => {
    //             let child = self.fuse_common_operations(*n.child);
    //             ExpressionTree::Perm(PermNode::new(child, n.perm))
    //         },
    //         ExpressionTree::Contract(n) => {
    //             let left = self.fuse_common_operations(*n.left);
    //             let right = self.fuse_common_operations(*n.right);
    //             ExpressionTree::Contract(ContractNode::new(left, right, n.left_qudits, n.right_qudits))
    //         },
    //     }
    // }


    // fn fuse_contraction_pre_post_permutations(
    //     &self,
    //     tree: &mut ExpressionTree,
    // ) {
    //     if let ExpressionTree::Contract(node) = tree {
    //         // TODO: Double-check im getting the permutations correct
    //         let left_perm = node.left_perm.clone();
    //         let right_perm = node.right_perm.clone();
    //         let left_contraction_shape = node.left_contraction_shape.clone();
    //         let right_contraction_shape = node.right_contraction_shape.clone();

    //         if let ExpressionTree::Contract(left) = node.left.as_mut() {
    //             left.fuse_output_perm(left_perm, left_contraction_shape);
    //             node.skip_left_permutation();
    //         }
    //         if let ExpressionTree::Contract(right) = node.right.as_mut() {
    //             right.fuse_output_perm(right_perm, right_contraction_shape);
    //             node.skip_right_permutation();
    //         }
    //     }
    // }

    // fn constant_propagation(&self, tree: &mut ExpressionTree) {
    //     if tree.num_params() == 0 {
    //         *tree = ExpressionTree::Constant(ConstantNode::new(tree.clone()));
    //     } else {
    //         match tree {
    //             ExpressionTree::Identity(_) => {},
    //             ExpressionTree::Kron(n) => {
    //                 self.constant_propagation(&mut n.left);
    //                 self.constant_propagation(&mut n.right);
    //             },
    //             ExpressionTree::Mul(n) => {
    //                 self.constant_propagation(&mut n.left);
    //                 self.constant_propagation(&mut n.right);
    //             },
    //             ExpressionTree::Leaf(_) => {},
    //             ExpressionTree::Constant(_) => {},
    //             ExpressionTree::Perm(n) => {
    //                 self.constant_propagation(&mut n.child);
    //             },
    //             ExpressionTree::Contract(n) => {
    //                 self.constant_propagation(&mut n.left);
    //                 self.constant_propagation(&mut n.right);
    //             },
    //         }
    //     }
    // }

    // fn fuse_contraction_and_permutations<C: ComplexScalar>(
    //     &self,
    //     _tree: &mut ExpressionTree<C>,
    // ) {
    // walk tree
    // if node is contract and either child permute
    // remove permute and add to contract
    // }
}
