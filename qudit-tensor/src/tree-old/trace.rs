use std::fmt;
use std::hash::Hash;

use qudit_core::HasPeriods;
use qudit_core::HasParams;
use qudit_core::ParamIndices;
use qudit_core::RealScalar;
use qudit_core::QuditRadices;
use qudit_core::QuditSystem;
use qudit_core::QuditPermutation;
use qudit_core::TensorShape;
use qudit_expr::GenerationShape;

use super::fmt::PrintTree;
use super::tree::ExpressionTree;

/// A partial trace node in the computation tree.
#[derive(Clone, PartialEq, Eq, Hash)]
pub struct TraceNode {
    /// The child node to be permuted.
    pub child: Box<ExpressionTree>,

    /// The permutation to apply to the child node.
    pub dimension_pairs: Vec<(usize, usize)>,

    // num_params: usize,
    //
    /// The dimension of each of the resulting tensor's indices.
    pub dimensions: Vec<usize>,

    /// The shape of the tensor after computation.
    pub generation_shape: GenerationShape,
}

impl TraceNode {
    pub fn new(child: ExpressionTree, pairs: Vec<(usize, usize)>) -> TraceNode {
        let child_dimensions = child.dimensions();

        let mut new_dimensions = child_dimensions.clone();
        let mut indices_to_remove = Vec::new();

        for (idx1, idx2) in &pairs {
            if *idx1 >= child_dimensions.len() || *idx2 >= child_dimensions.len() {
                panic!("Dimension index out of bounds for trace operation. Child dimensions: {:?}, attempting to trace indices: ({}, {})", child_dimensions, idx1, idx2);
            }
            if child_dimensions[*idx1] != child_dimensions[*idx2] {
                panic!("Dimensions at trace indices must be equal. Found {} at index {} and {} at index {}.", child_dimensions[*idx1], idx1, child_dimensions[*idx2], idx2);
            }
            indices_to_remove.push(*idx1);
            indices_to_remove.push(*idx2);
        }

        indices_to_remove.sort_unstable();
        indices_to_remove.dedup();
        indices_to_remove.reverse();

        for &idx in &indices_to_remove {
            new_dimensions.remove(idx);
        }

        // let generation_shape = GenerationShape::from_dimensions(&new_dimensions); // TODO
        let generation_shape = GenerationShape::Scalar; // TODO:

        TraceNode {
            child: Box::new(child),
            dimension_pairs: pairs,
            // num_params,
            dimensions: new_dimensions,
            generation_shape,
        }
    }

    pub fn dimensions(&self) -> Vec<usize> {
        self.dimensions.clone()
    }

    pub fn generation_shape(&self) -> GenerationShape {
        self.generation_shape.clone()
    }

    pub fn param_indices(&self) -> ParamIndices {
        self.child.param_indices()
    }
}

impl fmt::Debug for TraceNode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Trace")
            .field("child", &self.child)
            .field("pairs", &self.dimension_pairs)
            .finish()
    }
}

impl PrintTree for TraceNode {
    fn write_tree(&self, prefix: &str, fmt: &mut std::fmt::Formatter<'_>) {
        writeln!(fmt, "{}Trace({:?})", prefix, self.dimension_pairs).unwrap();
        let child_prefix = self.modify_prefix_for_child(prefix, true);
        self.child.write_tree(&child_prefix, fmt);
    }
}
