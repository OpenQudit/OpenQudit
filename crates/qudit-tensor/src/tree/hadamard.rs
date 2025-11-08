use std::hash::Hash;

use super::fmt::PrintTree;
use super::tree::TTGTNode;
use qudit_core::ParamInfo;
use qudit_expr::index::TensorIndex;

#[derive(Clone, PartialEq, Eq, Debug, Hash)]
pub struct HadamardProductNode {
    pub left: Box<TTGTNode>,
    pub right: Box<TTGTNode>,
    param_info: ParamInfo,
    indices: Vec<TensorIndex>,
}

impl HadamardProductNode {
    pub fn new(left: TTGTNode, right: TTGTNode) -> Self {
        let left_indices = left.indices();
        let right_indices = right.indices();

        assert!(
            left_indices
                .iter()
                .zip(right_indices.iter())
                .all(|(l, r)| l.index_size() == r.index_size() && l.direction() == r.direction())
        );

        let param_info = left.param_info().union(&right.param_info());

        HadamardProductNode {
            left: Box::new(left),
            right: Box::new(right),
            param_info,
            indices: left_indices,
        }
    }

    pub fn indices(&self) -> Vec<TensorIndex> {
        self.indices.clone()
    }

    pub fn param_info(&self) -> ParamInfo {
        self.param_info.clone()
    }
}

impl PrintTree for HadamardProductNode {
    fn write_tree(&self, prefix: &str, fmt: &mut std::fmt::Formatter<'_>) {
        writeln!(fmt, "{}Hadamard", prefix).unwrap();
        let left_prefix = self.modify_prefix_for_child(prefix, false);
        let right_prefix = self.modify_prefix_for_child(prefix, true);
        self.left.write_tree(&left_prefix, fmt);
        self.right.write_tree(&right_prefix, fmt);
    }
}
