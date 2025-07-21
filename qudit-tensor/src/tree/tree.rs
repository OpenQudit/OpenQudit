
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
use qudit_expr::index::IndexId;
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
        let is_identity_permutation = perm.iter().enumerate().all(|(i, &p)| i == p);
        let original_directions: Vec<IndexDirection> = self.indices().iter().map(|idx| idx.direction()).collect();
        let is_identity_redirection = redirection == original_directions;

        if is_identity_permutation && is_identity_redirection {
            self
        } else if let ExpressionTree::Leaf(mut n) = self {
            n.permute(&perm, redirection);
            ExpressionTree::Leaf(n)
        } else if let ExpressionTree::Transpose(n) = self {
            let TransposeNode { child, perm, .. } = n;
            let composed_perm: Vec<usize> = perm.iter().map(|&idx| perm[idx]).collect();
            Self::Transpose(TransposeNode::new(*child, composed_perm, redirection))
        } else {
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

    pub fn contract(
        self,
        right: Self,
        shared_ids: Vec<IndexId>,
        contraction_ids: Vec<IndexId>,
    ) -> Self {
        let left = self;
        // TODO: assert all shared ids are in both left and right
        // TODO: assert all contracted ids are in both left and right
        // TODO: assert no overlap between shared and contracted

        if contraction_ids.is_empty() {
            if left.rank() == shared_ids.len() && right.rank() == left.rank() {
                return ExpressionTree::Hadamard(HadamardProductNode::new(left, right));
            }
            return ExpressionTree::Outer(OuterProductNode::new(left, right));
        }

        // First find the permutation and redirection for left that makes its tensor
        // order (shared_ids [Batch], non_contracted [Output], contracted[Input])
        let left_left_indices = left.indices().iter()
            .filter(|idx| !shared_ids.contains(&idx.index_id()))
            .filter(|idx| !contraction_ids.contains(&idx.index_id()))
            .copied()
            .collect::<Vec<TensorIndex>>();

        let left_index_transpose = shared_ids.iter().copied()
            .chain(left_left_indices.iter().map(|idx| idx.index_id()))
            .chain(contraction_ids.iter().copied())
            .map(|i| left.indices().iter().position(|x| x.index_id() == i).unwrap())
            .collect::<Vec<usize>>();

        let left_index_redirection = shared_ids.iter()
            .map(|_| IndexDirection::Batch)
            .chain(left_left_indices.iter().map(|_| IndexDirection::Output))
            .chain(contraction_ids.iter().map(|_| IndexDirection::Input))
            .collect();

        let left_transposed_tree = left.transpose(left_index_transpose, left_index_redirection);

        // same for right but (shared_ids, contracted, non_contracted)
        let right_right_indices = right.indices().iter()
            .filter(|idx| !shared_ids.contains(&idx.index_id()))
            .filter(|idx| !contraction_ids.contains(&idx.index_id()))
            .copied()
            .collect::<Vec<TensorIndex>>();

        let right_index_transpose = shared_ids.iter().copied()
            .chain(contraction_ids.iter().copied())
            .chain(right_right_indices.iter().map(|idx| idx.index_id()))
            .map(|i| right.indices().iter().position(|x| x.index_id() == i).unwrap())
            .collect::<Vec<usize>>();

        let right_index_redirection = shared_ids.iter()
            .map(|_| IndexDirection::Batch)
            .chain(contraction_ids.iter().map(|_| IndexDirection::Output))
            .chain(right_right_indices.iter().map(|_| IndexDirection::Input))
            .collect();

        let right_transposed_tree = right.transpose(right_index_transpose, right_index_redirection);

        // Contract
        left_transposed_tree.matmul(right_transposed_tree)
    }

    pub fn matmul(self, other: Self) -> Self {
        Self::MatMul(MatMulNode::new(self, other))
    }
    
    pub fn trace(self, pairs: Vec<(usize, usize)>) -> Self {
        if pairs.is_empty() {
            self
        } else if let ExpressionTree::Leaf(mut n) = self {
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
