use std::hash::Hash;

use super::fmt::PrintTree;
use qudit_core::HasPeriods;
use qudit_core::HasParams;
use qudit_core::ParamInfo;
use qudit_core::RealScalar;
use qudit_core::Radices;
use qudit_core::QuditSystem;
use qudit_expr::index::IndexDirection;
use qudit_expr::index::TensorIndex;
use qudit_expr::GenerationShape;
use super::tree::TTGTNode;

/// A kron node in the computation tree that stacks two nodes.
#[derive(Clone, PartialEq, Eq, Debug, Hash)]
pub struct OuterProductNode {
    pub left: Box<TTGTNode>,
    pub right: Box<TTGTNode>,
    param_info: ParamInfo,
    indices: Vec<TensorIndex>,
}

impl OuterProductNode {
    pub fn new(left: TTGTNode, right: TTGTNode) -> OuterProductNode {
        let left_indices = left.indices();
        let right_indices = right.indices();

        // assert left and right have the same batch dimensions
        let left_batch_size = left_indices.iter().filter(|idx| idx.direction() == IndexDirection::Batch).map(|idx| idx.index_size()).product::<usize>();
        let right_batch_size = right_indices.iter().filter(|idx| idx.direction() == IndexDirection::Batch).map(|idx| idx.index_size()).product::<usize>();
        assert_eq!(left_batch_size, right_batch_size);

        // new indices = batch (shared) | (left, right)::output | (left, right)::input
        let indices = left_indices.iter()
            .filter(|idx| idx.direction() == IndexDirection::Batch)
            .chain(left_indices.iter()
                .filter(|idx| idx.direction() == IndexDirection::Output))
            .chain(right_indices.iter()
                .filter(|idx| idx.direction() == IndexDirection::Output))
            .chain(left_indices.iter()
                .filter(|idx| idx.direction() == IndexDirection::Input))
            .chain(right_indices.iter()
                .filter(|idx| idx.direction() == IndexDirection::Input))
            .copied()
            .collect();

        let param_info = left.param_info().concat(&right.param_info());

        OuterProductNode {
            left: Box::new(left),
            right: Box::new(right),
            param_info,
            indices,
        }
    }

    pub fn indices(&self) -> Vec<TensorIndex> {
        self.indices.clone()
    }

    pub fn param_info(&self) -> ParamInfo {
        self.param_info.clone()
    }
}

impl PrintTree for OuterProductNode {
    fn write_tree(&self, prefix: &str, fmt: &mut std::fmt::Formatter<'_>) {
        writeln!(fmt, "{}Outer", prefix).unwrap();
        let left_prefix = self.modify_prefix_for_child(prefix, false);
        let right_prefix = self.modify_prefix_for_child(prefix, true);
        self.left.write_tree(&left_prefix, fmt);
        self.right.write_tree(&right_prefix, fmt);
    }
}

