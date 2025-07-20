use std::hash::Hash;

use super::fmt::PrintTree;
use qudit_core::HasPeriods;
use qudit_core::HasParams;
use qudit_core::ParamIndices;
use qudit_core::RealScalar;
use qudit_core::QuditRadices;
use qudit_core::QuditSystem;
use qudit_core::TensorShape;
use qudit_expr::index::IndexDirection;
use qudit_expr::index::TensorIndex;
use qudit_expr::GenerationShape;
use super::tree::ExpressionTree;

#[derive(Clone, PartialEq, Eq, Debug, Hash)]
pub struct HadamardProductNode {
    pub left: Box<ExpressionTree>,
    pub right: Box<ExpressionTree>,
    param_map: ParamIndices,
    indices: Vec<TensorIndex>,
}

impl HadamardProductNode {
    pub fn new(left: ExpressionTree, right: ExpressionTree) -> Self {
        let left_indices = left.indices();
        let right_indices = right.indices();

        assert!(left_indices.iter().zip(right_indices.iter()).all(|(l, r)| l.index_size() == r.index_size() && l.direction() == r.direction()));

        let param_map = left.param_indices().concat(&right.param_indices());

        HadamardProductNode {
            left: Box::new(left),
            right: Box::new(right),
            param_map,
            indices: left_indices,
        }
    }

    pub fn indices(&self) -> Vec<TensorIndex> {
        self.indices.clone()
    }

    pub fn param_indices(&self) -> ParamIndices {
        self.param_map.clone()
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

