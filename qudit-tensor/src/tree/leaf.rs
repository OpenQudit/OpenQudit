use std::hash::Hash;

use super::fmt::PrintTree;
use qudit_core::{ParamIndices, QuditRadices, RealScalar};
use qudit_expr::{TensorExpression, TensorGenerationShape};
use super::tree::ExpressionTree;

#[derive(Clone, PartialEq, Eq, Debug, Hash)]
pub struct LeafNode {
    pub expr: TensorExpression,
    pub param_indices: ParamIndices,
    pub gen_shape: TensorGenerationShape,
}

impl LeafNode {
    pub fn new(expr: TensorExpression, param_indices: ParamIndices) -> LeafNode {
        let gen_shape = expr.generation_shape();

        LeafNode {
            expr,
            param_indices,
            gen_shape,
        }
    }

    pub fn dimensions(&self) -> QuditRadices {
        self.expr.dimensions()
    }

    pub fn generation_shape(&self) -> TensorGenerationShape {
        self.gen_shape.clone()
    }

    pub fn set_generation_shape(&mut self, gen_shape: TensorGenerationShape) {
        self.gen_shape = gen_shape;
    }

    pub fn param_indices(&self) -> ParamIndices {
        self.param_indices.clone()
    }
}

impl PrintTree for LeafNode {
    fn write_tree(&self, prefix: &str, fmt: &mut std::fmt::Formatter<'_>) {
        writeln!(fmt, "{}{}", prefix, self.expr.name).unwrap()
    }
}
