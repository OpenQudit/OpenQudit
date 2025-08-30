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
}

impl LeafNode {
    pub fn new(expr: ExpressionId, param_info: ParamInfo, indices: Vec<TensorIndex>) -> LeafNode {
        LeafNode {
            expr,
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

impl PrintTree for LeafNode {
    fn write_tree(&self, prefix: &str, fmt: &mut std::fmt::Formatter<'_>) {
        writeln!(fmt, "{}{}", prefix, self.expr).unwrap()
    }
}
