use std::hash::Hash;

use super::fmt::PrintTree;
use qudit_core::HasPeriods;
use qudit_core::HasParams;
use qudit_core::ParamIndices;
use qudit_core::RealScalar;
use qudit_core::QuditRadices;
use qudit_core::QuditSystem;
use qudit_expr::index::IndexDirection;
use qudit_expr::index::TensorIndex;
use qudit_expr::GenerationShape;
use super::tree::ExpressionTree;

#[derive(Clone, PartialEq, Eq, Debug, Hash)]
pub struct MatMulNode {
    pub left: Box<ExpressionTree>,
    pub right: Box<ExpressionTree>,
    param_map: ParamIndices,
    indices: Vec<TensorIndex>,
}

impl MatMulNode {
    pub fn new(left: ExpressionTree, right: ExpressionTree) -> MatMulNode {
        let left_indices = left.indices();
        let right_indices = right.indices();

        // assert left and right have the same batch dimensions
        let left_batch_size = left_indices.iter().filter(|idx| idx.direction() == IndexDirection::Batch).map(|idx| idx.index_size()).product::<usize>();
        let right_batch_size = right_indices.iter().filter(|idx| idx.direction() == IndexDirection::Batch).map(|idx| idx.index_size()).product::<usize>();
        assert_eq!(left_batch_size, right_batch_size);

        // new indices = batch (shared) | left::output | right::input
        let indices = left_indices.iter()
            .filter(|idx| idx.direction() == IndexDirection::Batch)
            .map(|idx| (idx.direction(), idx.index_size()))
            .chain(left_indices.iter()
                .filter(|idx| idx.direction() == IndexDirection::Output)
                .map(|idx| (idx.direction(), idx.index_size())))
            .chain(right_indices.iter()
                .filter(|idx| idx.direction() == IndexDirection::Input)
                .map(|idx| (idx.direction(), idx.index_size())))
            .enumerate()
            .map(|(id, (dir, size))| TensorIndex::new(dir, id, size))
            .collect();

        let param_map = left.param_indices().concat(&right.param_indices());

        MatMulNode {
            left: Box::new(left),
            right: Box::new(right),
            param_map,
            indices,
        }
    }

    pub fn indices(&self) -> Vec<TensorIndex> {
        self.indices.clone()
    }

    pub fn param_indices(&self) -> ParamIndices {
        self.param_map.clone()
    }
}

impl PrintTree for MatMulNode {
    fn write_tree(&self, prefix: &str, fmt: &mut std::fmt::Formatter<'_>) {
        writeln!(fmt, "{}MatMul", prefix).unwrap();
        let left_prefix = self.modify_prefix_for_child(prefix, false);
        let right_prefix = self.modify_prefix_for_child(prefix, true);
        self.left.write_tree(&left_prefix, fmt);
        self.right.write_tree(&right_prefix, fmt);
    }
}

