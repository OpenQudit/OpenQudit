use super::constant::ConstantNode;
use super::ExpressionTree;
use crate::math::Function;

pub struct TreeOptimizer {}

impl TreeOptimizer {
    pub fn new() -> Self {
        Self {}
    }

    pub fn optimize(&self, mut tree: ExpressionTree) -> ExpressionTree {
        tree.traverse_mut(&|n| self.fuse_contraction_pre_post_permutations(n));
        self.constant_propagation(&mut tree);
        tree
    }

    fn fuse_contraction_pre_post_permutations(
        &self,
        tree: &mut ExpressionTree,
    ) {
        if let ExpressionTree::Contract(node) = tree {
            // TODO: Double-check im getting the permutations correct
            let left_perm = node.left_perm.clone();
            let right_perm = node.right_perm.clone();
            let left_contraction_shape = node.left_contraction_shape.clone();
            let right_contraction_shape = node.right_contraction_shape.clone();

            if let ExpressionTree::Contract(left) = node.left.as_mut() {
                left.fuse_output_perm(left_perm, left_contraction_shape);
                node.skip_left_permutation();
            }
            if let ExpressionTree::Contract(right) = node.right.as_mut() {
                right.fuse_output_perm(right_perm, right_contraction_shape);
                node.skip_right_permutation();
            }
        }
    }

    fn constant_propagation(&self, tree: &mut ExpressionTree) {
        if tree.get_num_params() == 0 {
            *tree = ExpressionTree::Constant(ConstantNode::new(tree.clone()));
        } else {
            match tree {
                ExpressionTree::Identity(_) => {},
                ExpressionTree::Kron(n) => {
                    self.constant_propagation(&mut n.left);
                    self.constant_propagation(&mut n.right);
                },
                ExpressionTree::Mul(n) => {
                    self.constant_propagation(&mut n.left);
                    self.constant_propagation(&mut n.right);
                },
                ExpressionTree::Leaf(_) => {},
                ExpressionTree::Constant(_) => {},
                ExpressionTree::Perm(n) => {
                    self.constant_propagation(&mut n.child);
                },
                ExpressionTree::Contract(n) => {
                    self.constant_propagation(&mut n.left);
                    self.constant_propagation(&mut n.right);
                },
            }
        }
    }

    // fn fuse_contraction_and_permutations<C: ComplexScalar>(
    //     &self,
    //     _tree: &mut ExpressionTree<C>,
    // ) {
    // walk tree
    // if node is contract and either child permute
    // remove permute and add to contract
    // }
}
