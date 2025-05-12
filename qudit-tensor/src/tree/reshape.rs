use std::hash::Hash;

use super::fmt::PrintTree;
use qudit_core::{ParamIndices, QuditRadices, RealScalar};
use qudit_expr::{TensorExpression, TensorGenerationShape};
use super::tree::ExpressionTree;

#[derive(Clone, PartialEq, Eq, Debug, Hash)]
pub struct ReshapeNode {
    pub child: Box<ExpressionTree>,
    pub gen_shape: TensorGenerationShape,
}

impl ReshapeNode {
    pub fn new(child: ExpressionTree, gen_shape: TensorGenerationShape) -> ReshapeNode {
        ReshapeNode {
            child: Box::new(child),
            gen_shape,
        }
    }

    pub fn dimensions(&self) -> QuditRadices {
        self.child.dimensions()
    }

    pub fn generation_shape(&self) -> TensorGenerationShape {
        self.gen_shape.clone()
    }

    pub fn set_generation_shape(&mut self, gen_shape: TensorGenerationShape) {
        self.gen_shape = gen_shape;
    }

    pub fn param_indices(&self) -> ParamIndices {
        self.child.param_indices()
    }
}

impl PrintTree for ReshapeNode {
    fn write_tree(&self, prefix: &str, fmt: &mut std::fmt::Formatter<'_>) {
        writeln!(fmt, "{}Reshape{:?}", prefix, self.generation_shape()).unwrap();
        let child_prefix = self.modify_prefix_for_child(prefix, false);
        self.child.write_tree(&child_prefix, fmt);
    }
}
