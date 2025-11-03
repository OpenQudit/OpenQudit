use std::fmt;
use std::hash::Hash;

use qudit_core::ParamInfo;
use qudit_expr::index::TensorIndex;

use super::fmt::PrintTree;
use super::tree::TTGTNode;

/// A partial trace node in the computation tree.
#[derive(Clone, PartialEq, Eq, Hash)]
pub struct TraceNode {
    /// The child node to be permuted.
    pub child: Box<TTGTNode>,

    pub dimension_pairs: Vec<(usize, usize)>,

    pub indices: Vec<TensorIndex>,
}

impl TraceNode {
    pub fn new(child: TTGTNode, pairs: Vec<(usize, usize)>) -> TraceNode {
        let child_indices = child.indices();

        let mut indices_to_remove = Vec::new();

        for (idx1, idx2) in &pairs {
            if *idx1 >= child_indices.len() || *idx2 >= child_indices.len() {
                panic!("Dimension index out of bounds for trace operation. Child dimensions: {:?}, attempting to trace indices: ({}, {})", child_indices, idx1, idx2);
            }
            if child_indices[*idx1].index_size() != child_indices[*idx2].index_size() {
                panic!("Dimensions at trace indices must be equal. Found {} at index {} and {} at index {}.", child_indices[*idx1], idx1, child_indices[*idx2], idx2);
            }
            indices_to_remove.push(*idx1);
            indices_to_remove.push(*idx2);
        }

        let indices = child_indices.into_iter().enumerate().filter(|(i, idx)| !indices_to_remove.contains(i)).map(|(_, idx)| idx).collect();

        TraceNode {
            child: Box::new(child),
            dimension_pairs: pairs,
            indices,
        }
    }

    pub fn indices(&self) -> Vec<TensorIndex> {
        self.indices.clone()
    }

    pub fn param_info(&self) -> ParamInfo {
        self.child.param_info()
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
