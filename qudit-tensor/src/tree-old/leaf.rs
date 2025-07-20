use std::hash::Hash;

use super::fmt::PrintTree;
use qudit_core::{ParamIndices, QuditRadices, RealScalar};
use qudit_expr::{GenerationShape, TensorExpression};
use qudit_core::TensorShape;
use super::tree::ExpressionTree;

#[derive(Clone, PartialEq, Eq, Debug, Hash)]
pub struct LeafNode {
    pub expr: TensorExpression,
    pub param_indices: ParamIndices,
}

impl LeafNode {
    pub fn new(expr: TensorExpression, param_indices: ParamIndices) -> LeafNode {
        LeafNode {
            expr,
            param_indices,
        }
    }

    pub fn dimensions(&self) -> Vec<usize> {
        self.expr.dimensions()
    }

    pub fn generation_shape(&self) -> GenerationShape {
        self.expr.generation_shape()
    }

    pub fn set_generation_shape(&mut self, gen_shape: GenerationShape) {
        todo!()
        // self.expr.reshape(gen_shape);
    }

    pub fn permute(&mut self, perm: &[usize], gen_shape: GenerationShape) {
        self.expr.permute(perm, gen_shape);
    }

    pub fn trace(&mut self, dimension_pairs: &[(usize, usize)]) {
        self.expr = self.expr.partial_trace(dimension_pairs);
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
