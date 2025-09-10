use std::hash::Hash;

use super::fmt::PrintTree;
use qudit_core::{ParamIndices, ParamInfo, QuditRadices, RealScalar};
use qudit_expr::{index::{IndexDirection, TensorIndex}, ExpressionId, GenerationShape, TensorExpression};
use qudit_core::TensorShape;

#[derive(Clone, PartialEq, Eq, Debug, Hash)]
pub struct LeafNode {
    pub expr: ExpressionId,
    pub param_info: ParamInfo,
    pub indices: Vec<TensorIndex>,

    /// During leaf creation, some of the expression's indices might be grouped
    /// because they partcipate in a contraction together. This keeps track of that
    /// information for translating permutations.
    pub tensor_to_expr_position_map: Vec<Vec<usize>>,
}

impl LeafNode {
    pub fn new(expr: ExpressionId, param_info: ParamInfo, indices: Vec<TensorIndex>, tensor_to_expr_position_map: Vec<Vec<usize>>) -> LeafNode {
        LeafNode {
            expr,
            param_info,
            indices,
            tensor_to_expr_position_map,
        }
    }

    pub fn indices(&self) -> Vec<TensorIndex> {
        self.indices.clone()
    }

    pub fn param_info(&self) -> ParamInfo {
        self.param_info.clone()
    }

    pub fn convert_tensor_perm_to_expression_perm(&self, perm: &[usize]) -> Vec<usize> {
        perm.iter().flat_map(|p| &self.tensor_to_expr_position_map[*p]).cloned().collect()
    }
}

impl PrintTree for LeafNode {
    fn write_tree(&self, prefix: &str, fmt: &mut std::fmt::Formatter<'_>) {
        writeln!(fmt, "{}{}", prefix, self.expr).unwrap()
    }
}
