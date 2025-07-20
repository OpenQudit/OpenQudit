
use crate::tree::HadamardProductNode;
use crate::tree::TraceNode;

use super::fmt::PrintTree;
use super::leaf::LeafNode;
use super::matmul::MatMulNode;
use super::outer::OuterProductNode;
use super::transpose::TransposeNode;

use qudit_core::HasPeriods;
use qudit_core::HasParams;
use qudit_core::ParamIndices;
use qudit_core::RealScalar;
use qudit_expr::index::IndexDirection;
use qudit_expr::index::TensorIndex;
use qudit_expr::GenerationShape;
use qudit_expr::TensorExpression;
use qudit_core::TensorShape;
use qudit_expr::UnitaryExpression;
use qudit_core::QuditRadices;
use qudit_core::QuditSystem;

// TODO: Rename to TensorTree
/// A tree structure representing a parameterized quantum expression.
#[derive(PartialEq, Clone)]
pub enum ExpressionTree {
    // kinds of products
    MatMul(MatMulNode),
    Outer(OuterProductNode),
    Hadamard(HadamardProductNode),

    // Permute/reshape
    Transpose(TransposeNode),

    // Partial Traces
    Trace(TraceNode),

    // Tensor generation nodes
    Leaf(LeafNode),
}


impl ExpressionTree {
    pub fn indices(&self) -> Vec<TensorIndex> {
        match self {
            Self::MatMul(s) => s.indices(),
            Self::Outer(s) => s.indices(),
            Self::Hadamard(s) => s.indices(),
            Self::Transpose(s) => s.indices(),
            Self::Trace(s) => s.indices(),
            Self::Leaf(s) => s.indices(),
        }
    }

    pub fn generation_shape(&self) -> GenerationShape {
        self.indices().into()
    }

    pub fn rank(&self) -> usize {
        self.indices().len()
    }

    pub fn param_indices(&self) -> ParamIndices {
        match self {
            Self::MatMul(s) => s.param_indices(),
            Self::Outer(s) => s.param_indices(),
            Self::Hadamard(s) => s.param_indices(),
            Self::Transpose(s) => s.param_indices(),
            Self::Trace(s) => s.param_indices(),
            Self::Leaf(s) => s.param_indices(),
        }
    }

    pub fn traverse_mut(&mut self, f: &impl Fn(&mut Self)) {
        f(self);
        match self {
            ExpressionTree::MatMul(n) => {
                n.left.traverse_mut(f);
                n.right.traverse_mut(f);
            },
            ExpressionTree::Outer(n) => {
                n.left.traverse_mut(f);
                n.right.traverse_mut(f);
            },
            ExpressionTree::Hadamard(n) => {
                n.left.traverse_mut(f);
                n.right.traverse_mut(f);
            },
            ExpressionTree::Transpose(n) => {
                n.child.traverse_mut(f);
            },
            ExpressionTree::Trace(n) => {
                n.child.traverse_mut(f);
            },
            ExpressionTree::Leaf(_) => {},
        }
    }

    pub fn leaf(expr: TensorExpression, param_indices: ParamIndices) -> Self {
        Self::Leaf(LeafNode::new(expr, param_indices))
    }

    pub fn outer(self, right: Self) -> Self {
        ExpressionTree::Outer(OuterProductNode::new(self, right))
    }

    pub fn hadamard(self, right: Self) -> Self {
        ExpressionTree::Hadamard(HadamardProductNode::new(self, right))
    }

    pub fn transpose(self, perm: Vec<usize>, redirection: Vec<IndexDirection>) -> Self {
        if let ExpressionTree::Leaf(mut n) = self {
            n.permute(&perm, redirection);
            ExpressionTree::Leaf(n)
        } else {
            // if it's already a transpose, can merge TODO
            Self::Transpose(TransposeNode::new(self, perm, redirection))
        }
    }

    pub fn reindex(self, new_indices: Vec<TensorIndex>) -> Self {
        if let ExpressionTree::Leaf(mut n) = self {
            n.reindex(new_indices);
            ExpressionTree::Leaf(n)
        } else {
            panic!("Cannot reindex non leaf nodes directly.");
        }
    }

    // pub fn contract(
    //     self,
    //     other: Self,
    //     left_unsummed_indices: Vec<usize>,
    //     right_unsummed_indices: Vec<usize>,
    //     left_contraction_indices: Vec<usize>,
    //     right_contraction_indices: Vec<usize>,
    // ) -> Self {
    //     assert_eq!(left_unsummed_indices.len(), right_unsummed_indices.len());
    //     assert_eq!(left_contraction_indices.len(), right_contraction_indices.len());

    //     let (left, right) = (self, other);

    //     if left_contraction_indices.is_empty() {
    //         if left.rank() == left_unsummed_indices.len() && right.rank() == left.rank() {
    //             return ExpressionTree::Hadamard(HadamardProductNode::new(left, right));
    //         }
    //         return ExpressionTree::Outer(OuterProductNode::new(left, right));
    //     }



    //     // else TTG


    //     todo!()
    // }

    pub fn matmul(self, other: Self) -> Self {
        Self::MatMul(MatMulNode::new(self, other))
    }
    
    pub fn trace(self, pairs: Vec<(usize, usize)>) -> Self {
        if let ExpressionTree::Leaf(mut n) = self {
            n.trace(&pairs);
            ExpressionTree::Leaf(n)
        } else {
            Self::Trace(TraceNode::new(self, pairs))
        }
    }
}

impl std::hash::Hash for ExpressionTree {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        match self {
            Self::MatMul(s) => s.hash(state),
            Self::Outer(s) => s.hash(state),
            Self::Hadamard(s) => s.hash(state),
            Self::Transpose(s) => s.hash(state),
            Self::Trace(s) => s.hash(state),
            Self::Leaf(s) => s.hash(state),
        }
    }
}

impl Eq for ExpressionTree {}

impl PrintTree for ExpressionTree {
    fn write_tree(&self, prefix: &str, fmt: &mut std::fmt::Formatter<'_>) {
        match self {
            Self::MatMul(s) => s.write_tree(prefix, fmt),
            Self::Outer(s) => s.write_tree(prefix, fmt),
            Self::Hadamard(s) => s.write_tree(prefix, fmt),
            Self::Transpose(s) => s.write_tree(prefix, fmt),
            Self::Trace(s) => s.write_tree(prefix, fmt),
            Self::Leaf(s) => s.write_tree(prefix, fmt),
        }
    }
}

impl std::fmt::Debug for ExpressionTree {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.write_tree("", f); // TODO: propogate results
        Ok(())
    }
}
