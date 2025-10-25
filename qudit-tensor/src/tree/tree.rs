
use std::cell::RefCell;
use std::rc::Rc;
use std::sync::Arc;
use std::sync::Mutex;

use crate::tree::HadamardProductNode;
use crate::tree::TraceNode;

use super::fmt::PrintTree;
use super::leaf::LeafNode;
use super::matmul::MatMulNode;
use super::outer::OuterProductNode;
use super::transpose::TransposeNode;

use qudit_core::HasPeriods;
use qudit_core::HasParams;
use qudit_core::ParamInfo;
use qudit_core::RealScalar;
use qudit_expr::index::IndexDirection;
use qudit_expr::index::IndexId;
use qudit_expr::index::TensorIndex;
use qudit_expr::ExpressionId;
use qudit_expr::ExpressionCache;
use qudit_expr::GenerationShape;
use qudit_expr::TensorExpression;
use qudit_core::TensorShape;
use qudit_expr::UnitaryExpression;
use qudit_core::QuditRadices;
use qudit_core::QuditSystem;

// TODO: Rename to TensorTree
/// A tree structure representing a parameterized quantum expression.
#[derive(PartialEq, Eq, Clone)]
pub enum TTGTNode {
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

#[derive(Clone)]
pub struct TTGTTree {
    pub root: TTGTNode,
    pub expressions: Arc<Mutex<ExpressionCache>>,
}

impl TTGTTree {
    pub fn indices(&self) -> Vec<TensorIndex> {
        self.root.indices()
    }

    pub fn generation_shape(&self) -> GenerationShape {
        self.indices().into()
    }

    pub fn rank(&self) -> usize {
        self.indices().len()
    }

    pub fn contract(
        self,
        right: Self,
        shared_ids: Vec<IndexId>,
        contraction_ids: Vec<IndexId>,
    ) -> Self {
        if !Arc::ptr_eq(&self.expressions, &right.expressions) {
            panic!("Contracting two TTGT trees with different expression caches.");
        }

        // TODO: assert all shared ids are in both left and right
        // TODO: assert all contracted ids are in both left and right
        // TODO: assert no overlap between shared and contracted

        let left = self;

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

        // println!("Left transpose: {:?} redirection: {:?}", left_index_transpose, left_index_redirection);
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

        // println!("Right transpose: {:?} redirection: {:?}", right_index_transpose, right_index_redirection);
        let right_transposed_tree = right.transpose(right_index_transpose, right_index_redirection);

        // Contract
        if contraction_ids.is_empty() {
            if left_transposed_tree.rank() == shared_ids.len() && right_transposed_tree.rank() == left_transposed_tree.rank() {
                left_transposed_tree.hadamard(right_transposed_tree)
            } else {
                left_transposed_tree.outer(right_transposed_tree)
            }
        } else {
            left_transposed_tree.matmul(right_transposed_tree)
        }
    }

    pub fn outer(self, right: Self) -> Self {
        if !Arc::ptr_eq(&self.expressions, &right.expressions) {
            panic!("Contracting two TTGT trees with different expression caches.");
        }

        Self {
            root: TTGTNode::Outer(OuterProductNode::new(self.root, right.root)),
            expressions: self.expressions,
        }
    }

    pub fn hadamard(self, right: Self) -> Self {
        if !Arc::ptr_eq(&self.expressions, &right.expressions) {
            panic!("Contracting two TTGT trees with different expression caches.");
        }

        Self {
            root: TTGTNode::Hadamard(HadamardProductNode::new(self.root, right.root)),
            expressions: self.expressions,
        }
    }

    pub fn matmul(self, right: Self) -> Self {
        if !Arc::ptr_eq(&self.expressions, &right.expressions) {
            panic!("Contracting two TTGT trees with different expression caches.");
        }

        Self {
            root: TTGTNode::MatMul(MatMulNode::new(self.root, right.root)),
            expressions: self.expressions,
        }
    }

    pub fn transpose(mut self, perm: Vec<usize>, redirection: Vec<IndexDirection>) -> Self {
        let is_identity_permutation = perm.iter().enumerate().all(|(i, &p)| i == p);
        let original_directions: Vec<IndexDirection> = self.indices().iter().map(|idx| idx.direction()).collect();
        let is_identity_redirection = redirection == original_directions;

        let new_root = if is_identity_permutation && is_identity_redirection {
            self.root
        // } else if let TTGTNode::Leaf(mut n) = self.root {
        //     let new_id = self.expressions.borrow_mut().permute(n.expr, perm, redirection);
        //     TTGTNode::Leaf(LeafNode::new(new_id, n.param_info, self.expressions.borrow_mut().indices(new_id)))
        } else if let TTGTNode::Transpose(n) = self.root {
            let TransposeNode { child, perm: base_perm, .. } = n;
            let composed_perm: Vec<usize> = base_perm.iter().map(|&idx| perm[idx]).collect();
            TTGTNode::Transpose(TransposeNode::new(*child, composed_perm, redirection))
        } else if let TTGTNode::Leaf(n) = self.root {
            let expr_perm = n.convert_tensor_perm_to_expression_perm(&perm);
            let LeafNode { expr: expr_id, param_info, indices, tensor_to_expr_position_map } = n;
            let new_indices = perm.iter()
                .map(|p| indices[*p])
                .zip(redirection.iter())
                .map(|(idx, new_direction)| TensorIndex::new(*new_direction, idx.index_id(), idx.index_size()))
                .collect::<Vec<TensorIndex>>();
            let new_shape: GenerationShape = (&new_indices).into();
            let new_expr_id = self.expressions.lock().unwrap().permute_reshape(expr_id, expr_perm, new_shape);
            // println!("After permuting in tree transpose, id is {:?}", new_expr_id);
            TTGTNode::Leaf(LeafNode::new(new_expr_id, param_info, new_indices, tensor_to_expr_position_map))
        } else {
            TTGTNode::Transpose(TransposeNode::new(self.root, perm, redirection))
        };

        Self {
            root: new_root,
            expressions: self.expressions
        }
    }

    pub fn leaf(expressions: Arc<Mutex<ExpressionCache>>, expr: ExpressionId, param_info: ParamInfo, indices: Vec<TensorIndex>, tensor_to_expr_position_map: Vec<Vec<usize>>) -> Self {
        Self {
            root: TTGTNode::Leaf(LeafNode::new(expr, param_info, indices, tensor_to_expr_position_map)),
            expressions,
        }
    }

    // pub fn reindex(mut self, new_indices: Vec<TensorIndex>) -> Self {
    //     let new_root = if let TTGTNode::Leaf(mut n) = self.root {
    //         let new_id = self.expressions.borrow_mut().reindex(n.expr, new_indices);
    //         TTGTNode::Leaf(LeafNode::new(new_id, n.param_info, self.expressions.borrow_mut().indices(new_id)))
    //     } else {
    //         panic!("Cannot reindex non leaf nodes directly.");
    //     };

    //     Self {
    //         root: new_root,
    //         expressions: self.expressions
    //     }
    // }

    pub fn trace(mut self, pairs: Vec<(usize, usize)>) -> Self {
        let new_root = if pairs.is_empty() {
            self.root
        // } else if let TTGTNode::Leaf(mut n) = self.root {
        //     let new_id = self.expressions.borrow_mut().trace(n.expr, pairs);
        //     TTGTNode::Leaf(LeafNode::new(new_id, n.param_info, self.expressions.borrow_mut().indices(new_id)))
        } else {
            TTGTNode::Trace(TraceNode::new(self.root, pairs))
        };

        Self {
            root: new_root,
            expressions: self.expressions,
        }
    }
}

impl TTGTNode {
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

    pub fn param_info(&self) -> ParamInfo {
        match self {
            Self::MatMul(s) => s.param_info(),
            Self::Outer(s) => s.param_info(),
            Self::Hadamard(s) => s.param_info(),
            Self::Transpose(s) => s.param_info(),
            Self::Trace(s) => s.param_info(),
            Self::Leaf(s) => s.param_info(),
        }
    }
}

impl std::hash::Hash for TTGTNode {
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

impl PrintTree for TTGTTree {
    fn write_tree(&self, prefix: &str, fmt: &mut std::fmt::Formatter<'_>) {
        match &self.root {
            TTGTNode::MatMul(s) => s.write_tree(prefix, fmt),
            TTGTNode::Outer(s) => s.write_tree(prefix, fmt),
            TTGTNode::Hadamard(s) => s.write_tree(prefix, fmt),
            TTGTNode::Transpose(s) => s.write_tree(prefix, fmt),
            TTGTNode::Trace(s) => s.write_tree(prefix, fmt),
            TTGTNode::Leaf(s) => s.write_tree(prefix, fmt),
        }
    }
}

impl std::fmt::Debug for TTGTTree {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.write_tree("", f); // TODO: propogate results
        Ok(())
    }
}

impl PrintTree for TTGTNode {
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

impl std::fmt::Debug for TTGTNode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.write_tree("", f); // TODO: propogate results
        Ok(())
    }
}
