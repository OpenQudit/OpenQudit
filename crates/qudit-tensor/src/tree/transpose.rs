use std::fmt;
use std::hash::Hash;

use qudit_core::ParamInfo;
use qudit_expr::index::TensorIndex;
use qudit_expr::index::IndexDirection;

use super::fmt::PrintTree;
use super::tree::TTGTNode;

/// A permutation node in the computation tree.
/// This node wraps another node and applies a permutation to its output.
#[derive(Clone, PartialEq, Eq, Hash)]
pub struct TransposeNode {
    /// The child node to be permuted.
    pub child: Box<TTGTNode>,

    /// The permutation to apply to the child node.
    pub perm: Vec<usize>,

    pub indices: Vec<TensorIndex>,
}

impl TransposeNode {
    pub fn new(child: TTGTNode, perm: Vec<usize>, redirection: Vec<IndexDirection>) -> TransposeNode {
        let child_indices = child.indices();
        let indices: Vec<TensorIndex> = (0..perm.len()).map(|x| TensorIndex::new(redirection[x], child_indices[perm[x]].index_id(), child_indices[perm[x]].index_size())).collect();

        TransposeNode {
            child: Box::new(child),
            perm,
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

impl fmt::Debug for TransposeNode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Transpose")
            .field("child", &self.child)
            .field("perm", &self.perm)
            .finish()
    }
}

impl PrintTree for TransposeNode {
    fn write_tree(&self, prefix: &str, fmt: &mut std::fmt::Formatter<'_>) {
        writeln!(fmt, "{}Transpose{:?}", prefix, self.perm).unwrap();
        let child_prefix = self.modify_prefix_for_child(prefix, true);
        self.child.write_tree(&child_prefix, fmt);
    }
}

