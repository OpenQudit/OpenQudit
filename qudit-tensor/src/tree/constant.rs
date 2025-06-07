use std::hash::Hash;

use qudit_core::HasPeriods;
use qudit_core::HasParams;
use qudit_core::ParamIndices;
use qudit_core::QuditRadices;
use qudit_core::RealScalar;
use qudit_core::QuditSystem;
use qudit_core::TensorShape;

use super::fmt::PrintTree;
use super::tree::ExpressionTree;

#[derive(Clone, PartialEq, Eq, Debug, Hash)]
pub struct ConstantNode {
    pub child: Box<ExpressionTree>,
}

impl ConstantNode {
    pub fn new(child: ExpressionTree) -> Self {
        Self {
            child: Box::new(child),
        }
    }

    pub fn dimensions(&self) -> QuditRadices {
        self.child.dimensions()
    }

    pub fn generation_shape(&self) -> TensorShape {
        self.child.generation_shape()
    }

    pub fn param_indices(&self) -> ParamIndices {
        self.child.param_indices()
    }
}

// impl HasParams for ConstantNode {
//     fn num_params(&self) -> usize {
//         0
//     }
// }

// impl<R: RealScalar> HasPeriods<R> for ConstantNode {
//     fn periods(&self) -> Vec<std::ops::Range<R>> {
//         return Vec::new();
//     }
// }

// impl QuditSystem for ConstantNode {
//     fn dimension(&self) -> usize {
//         self.child.dimension()
//     }

//     fn num_qudits(&self) -> usize {
//         self.child.num_qudits()
//     }

//     fn radices(&self) -> QuditRadices {
//         self.child.radices()
//     }
// }

impl PrintTree for ConstantNode {
    fn write_tree(&self, prefix: &str, fmt: &mut std::fmt::Formatter<'_>) {
        writeln!(fmt, "{}Constant", prefix).unwrap();
        let child_prefix = self.modify_prefix_for_child(prefix, true);
        self.child.write_tree(&child_prefix, fmt);
    }
}
