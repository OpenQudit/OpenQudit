use std::hash::Hash;

use crate::math::BoundedFn;
use crate::math::Function;
use crate::{QuditRadices, QuditSystem};

use super::{tree::PrintTree, ExpressionTree};

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
}

impl Function for ConstantNode {
    fn get_num_params(&self) -> usize {
        0
    }
}

impl BoundedFn for ConstantNode {
    fn get_bounds(&self) -> Vec<std::ops::Range<f64>> {
        return Vec::new();
    }
}

impl QuditSystem for ConstantNode {
    fn get_dimension(&self) -> usize {
        self.child.get_dimension()
    }

    fn get_num_qudits(&self) -> usize {
        self.child.get_num_qudits()
    }

    fn get_radices(&self) -> QuditRadices {
        self.child.get_radices()
    }
}

// impl Hash for ConstantNode {
//     fn hash<H: Hasher>(&self, state: &mut H) {
//         self.child.hash(state);
//     }
// }

impl PrintTree for ConstantNode {
    fn write_tree(&self, prefix: &str, fmt: &mut std::fmt::Formatter<'_>) {
        writeln!(fmt, "{}Constant", prefix).unwrap();
        let child_prefix = self.modify_prefix_for_child(prefix, true);
        self.child.write_tree(&child_prefix, fmt);
    }
}
