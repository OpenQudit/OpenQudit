
use crate::tree::TraceNode;

use super::constant::ConstantNode;
// use super::contract::ContractNode;
use super::fmt::PrintTree;
use super::leaf::LeafNode;
// use super::identity::IdentityNode;
// use super::kron::KronNode;
use super::matmul::MatMulNode;
// use super::mul::MulNode;
use super::outer::OuterProductNode;
use super::reshape::ReshapeNode;
// use super::perm::PermNode;
use super::transpose::TransposeNode;

use qudit_core::HasPeriods;
use qudit_core::HasParams;
use qudit_core::ParamIndices;
use qudit_core::RealScalar;
use qudit_expr::GenerationShape;
use qudit_expr::TensorExpression;
use qudit_core::TensorShape;
use qudit_expr::UnitaryExpression;
use qudit_core::QuditRadices;
use qudit_core::QuditSystem;

/// A tree structure representing a parameterized quantum expression.
#[derive(PartialEq, Clone)]
pub enum ExpressionTree {
    Constant(ConstantNode),
    Reshape(ReshapeNode),
    MatMul(MatMulNode),
    Transpose(TransposeNode),
    Trace(TraceNode),
    OuterProduct(OuterProductNode),
    Leaf(LeafNode),
}


impl ExpressionTree {
    pub fn traverse_mut(&mut self, f: &impl Fn(&mut Self)) {
        f(self);
        match self {
            ExpressionTree::OuterProduct(n) => {
                n.left.traverse_mut(f);
                n.right.traverse_mut(f);
            },
            ExpressionTree::MatMul(n) => {
                n.left.traverse_mut(f);
                n.right.traverse_mut(f);
            },
            ExpressionTree::Leaf(_) => {},
            ExpressionTree::Transpose(n) => {
                n.child.traverse_mut(f);
            },
            ExpressionTree::Trace(n) => {
                n.child.traverse_mut(f);
            },
            ExpressionTree::Constant(n) => {
                n.child.traverse_mut(f);
            },
            ExpressionTree::Reshape(n) => {
                n.child.traverse_mut(f);
            },
        }
    }

    pub fn leaf(expr: TensorExpression, param_indices: ParamIndices) -> Self {
        Self::Leaf(LeafNode::new(expr, param_indices))
    }

    pub fn transpose(self, perm: Vec<usize>, shape: GenerationShape) -> Self {
        Self::Transpose(TransposeNode::new(self, perm, shape))
    }

    pub fn product(self, other: Self) -> Self {
        todo!()
    }

    pub fn matmul(self, other: Self) -> Self {
        Self::MatMul(MatMulNode::new(self, other))
    }
    
    pub fn reshape(self, shape: GenerationShape) -> Self {
        Self::Reshape(ReshapeNode::new(self, shape))
    }
    
    pub fn trace(self, pairs: Vec<(usize, usize)>) -> Self {
        Self::Trace(TraceNode::new(self, pairs))
    }

    pub fn dimensions(&self) -> Vec<usize> {
        match self {
            Self::OuterProduct(s) => s.dimensions(),
            Self::MatMul(s) => s.dimensions(),
            Self::Leaf(s) => s.dimensions(),
            Self::Trace(s) => s.dimensions(),
            Self::Transpose(s) => s.dimensions(),
            Self::Constant(s) => s.dimensions(),
            Self::Reshape(s) => s.dimensions(),
        }
    }

    pub fn generation_shape(&self) -> GenerationShape {
        match self {
            Self::OuterProduct(s) => s.generation_shape(),
            Self::MatMul(s) => s.generation_shape(),
            Self::Leaf(s) => s.generation_shape(),
            Self::Trace(s) => s.generation_shape(),
            Self::Transpose(s) => s.generation_shape(),
            Self::Constant(s) => s.generation_shape(),
            Self::Reshape(s) => s.generation_shape(),
        }
    }

    pub fn param_indices(&self) -> ParamIndices {
        match self {
            Self::OuterProduct(s) => s.param_indices(),
            Self::MatMul(s) => s.param_indices(),
            Self::Leaf(s) => s.param_indices(),
            Self::Trace(s) => s.param_indices(),
            Self::Transpose(s) => s.param_indices(),
            Self::Constant(s) => s.param_indices(),
            Self::Reshape(s) => s.param_indices(),
        }
    }
}

// impl QuditSystem for ExpressionTree {
//     fn dimension(&self) -> usize {
//         match self {
//             Self::Identity(s) => s.dimension(),
//             Self::Kron(s) => s.dimension(),
//             Self::Mul(s) => s.dimension(),
//             Self::Leaf(s) => s.dimension(),
//             Self::Perm(s) => s.dimension(),
//             Self::Contract(s) => s.dimension(),
//             Self::Constant(s) => s.dimension(),
//         }
//     }

//     fn radices(&self) -> QuditRadices {
//         match self {
//             Self::Identity(s) => s.radices(),
//             Self::Kron(s) => s.radices(),
//             Self::Mul(s) => s.radices(),
//             Self::Leaf(s) => s.radices(),
//             Self::Perm(s) => s.radices(),
//             Self::Contract(s) => s.radices(),
//             Self::Constant(s) => s.radices(),
//         }
//     }
// }

// impl HasParams for ExpressionTree {
//     fn num_params(&self) -> usize {
//         match self {
//             Self::Identity(s) => s.num_params(),
//             Self::Kron(s) => s.num_params(),
//             Self::Mul(s) => s.num_params(),
//             Self::Leaf(s) => s.num_params(),
//             Self::Perm(s) => s.num_params(),
//             Self::Contract(s) => s.num_params(),
//             Self::Constant(s) => s.num_params(),
//         }
//     }
// }

// impl<R: RealScalar> HasPeriods<R> for ExpressionTree {
//     fn periods(&self) -> Vec<std::ops::Range<R>> {
//         match self {
//             Self::Identity(s) => s.periods(),
//             Self::Kron(s) => s.periods(),
//             Self::Mul(s) => s.periods(),
//             Self::Leaf(s) => s.periods(),
//             Self::Perm(s) => s.periods(),
//             Self::Contract(s) => s.periods(),
//             Self::Constant(s) => s.periods(),
//         }
//     }
// }

// impl From<UnitaryExpression> for ExpressionTree {
//     fn from(expr: UnitaryExpression) -> ExpressionTree {
//         Self::Leaf(expr)
//     }
// }

impl std::hash::Hash for ExpressionTree {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        match self {
            Self::OuterProduct(s) => s.hash(state),
            Self::MatMul(s) => s.hash(state),
            Self::Leaf(s) => s.hash(state),
            Self::Trace(s) => s.hash(state),
            Self::Transpose(s) => s.hash(state),
            Self::Constant(s) => s.hash(state),
            Self::Reshape(s) => s.hash(state),
        }
    }
}

impl Eq for ExpressionTree {}

impl PrintTree for ExpressionTree {
    fn write_tree(&self, prefix: &str, fmt: &mut std::fmt::Formatter<'_>) {
        match self {
            Self::OuterProduct(s) => s.write_tree(prefix, fmt),
            Self::MatMul(s) => s.write_tree(prefix, fmt),
            Self::Leaf(s) => s.write_tree(prefix, fmt),
            Self::Trace(s) => s.write_tree(prefix, fmt),
            Self::Transpose(s) => s.write_tree(prefix, fmt),
            Self::Constant(s) => s.write_tree(prefix, fmt),
            Self::Reshape(s) => s.write_tree(prefix, fmt),
        }
    }
}

impl std::fmt::Debug for ExpressionTree {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.write_tree("", f); // TODO: propogate results
        Ok(())
    }
}

// #[cfg(test)]
// pub mod strategies {
//     use super::*;
//     use crate::math::perm::strategies::perms_with_radices;
//     use crate::strategies::gates;
//     // use crate::strategies::params;
//     use proptest::prelude::*;

//     // TODO: implement this
//     // pub fn node_with_radices(radices: QuditRadices) -> impl Strategy<Value
// = Node> {

//     // }

//     pub fn arbitrary_nodes() -> impl Strategy<Value = Node> {
//         let leaf = gates().prop_map(|g| Node::Leaf(LeafStruct::new(g)));
//         leaf.prop_recursive(
//             3, // at most this many levels
//             4, // Shoot for maximum of this many nodes
//             2, // maximum branching factor
//             |inner| {
//                 prop_oneof![
//                     inner
//                         .clone()
//                         .prop_flat_map(|inner_node| (
//                             Just(inner_node.clone()),
//                             perms_with_radices(inner_node.get_radices())
//                         ))
//                         .prop_map(|(inner_node, perm)|
// Node::Perm(PermNode::new(                             inner_node, perm
//                         ))),
//                     (inner.clone(), inner.clone())
//                         .prop_filter("Size would be too large", |(left,
// right)| {                             let num_params = left.get_num_params()
// + right.get_num_params() + 1;                             let dimension =
// left.get_dimension() * right.get_dimension();
// num_params * dimension * dimension < 1024                         })
//                         .prop_map(|(left, right)|
// Node::Kron(KronNode::new(left, right))),                     inner
//                         .clone() // TODO: Write better mul case
//                         .prop_map(|node| Node::Mul(MulNode::new(node.clone(),
// node))),                 ]
//             },
//         )
//     }

//     pub fn nodes_and_params() -> impl Strategy<Value = (Node, Vec<f64>)> {
//         arbitrary_nodes().prop_flat_map(|node| {
//             let num_params = node.get_num_params();
//             (Just(node), params(num_params))
//         })
//     }
// }

#[cfg(test)]
mod tests {
    // use std::time::Instant;
    // use crate::math::c64;

    // use super::*;

    // #[test]
    // fn test_speed() {
    //     let cx: ExpressionTree<c64> = ExpressionTree::Leaf(Gate::CZ());
    //     println!("{:?}", cx);
    //     let rz1 = ExpressionTree::Leaf(Gate::P(2));
    //     let rz2 = ExpressionTree::Leaf(Gate::P(2));
    //     let k1 = ExpressionTree::Kron(KronNode::new(rz1, rz2));
    //     println!("{:?}", k1);
    //     let block = ExpressionTree::Mul(MulNode::new(cx, k1));
    //     println!("{:?}", block);
    //     let block1 = block.clone();
    //     let block2 = block.clone();
    //     let block3 = block.clone();
    //     let block4 = block.clone();
    //     let block5 = block.clone();
    //     let block6 = block.clone();
    //     let block7 = block.clone();
    //     let block8 = block.clone();

    //     let mulblock1 = ExpressionTree::Mul(MulNode::new(block1, block2));
    //     let mulblock2 = ExpressionTree::Mul(MulNode::new(block3, block4));
    //     let mulblock3 = ExpressionTree::Mul(MulNode::new(block5, block6));
    //     let mulblock4 = ExpressionTree::Mul(MulNode::new(block7, block8));

    //     let kronblock1 =
    //         ExpressionTree::Kron(KronNode::new(mulblock1, mulblock2));
    //     let kronblock2 =
    //         ExpressionTree::Kron(KronNode::new(mulblock3, mulblock4));

    //     let circ = ExpressionTree::Kron(KronNode::new(kronblock1, kronblock2));
    //     println!("{:?}", circ);

    //     let now = Instant::now();
    //     for _ in 0..100 {
    //         let _ = circ.get_unitary_and_gradient(&[
    //             1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
    //             1.0, 2.0, 3.0, 4.0,
    //         ]);
    //     }
    //     let elapsed = now.elapsed();
    //     println!("==================={:.2?}", elapsed);
    // }
}
