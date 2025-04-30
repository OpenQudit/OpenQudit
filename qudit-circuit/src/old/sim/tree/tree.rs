// use faer_core::MatMut;

use super::constant::ConstantNode;
use super::contract::ContractNode;
use super::identity::IdentityNode;
use super::kron::KronNode;
use super::mul::MulNode;
use super::perm::PermNode;
use crate::gates::NamedGate;
// use crate::math::unitary::DifferentiableUnitaryFn;
// use crate::math::unitary::DoublyDifferentiableUnitaryFn;
// use crate::math::unitary::UnitaryFn;
// use crate::math::unitary::UnitaryGradient;
// use crate::math::unitary::UnitaryHessian;
use crate::math::BoundedFn;
use crate::math::Function;
use crate::Gate;
use crate::QuditRadices;
use crate::QuditSystem;

#[derive(PartialEq, Clone)] // TODO: Display
pub enum ExpressionTree {
    Identity(IdentityNode),
    Kron(KronNode),
    Mul(MulNode),
    Perm(PermNode),
    Leaf(Gate),
    Contract(ContractNode),
    Constant(ConstantNode),
}

impl ExpressionTree {
    pub fn traverse_mut(&mut self, f: &impl Fn(&mut Self)) {
        f(self);
        match self {
            ExpressionTree::Identity(_) => {},
            ExpressionTree::Kron(n) => {
                n.left.traverse_mut(f);
                n.right.traverse_mut(f);
            },
            ExpressionTree::Mul(n) => {
                n.left.traverse_mut(f);
                n.right.traverse_mut(f);
            },
            ExpressionTree::Leaf(_) => {},
            ExpressionTree::Perm(n) => {
                n.child.traverse_mut(f);
            },
            ExpressionTree::Contract(n) => {
                n.left.traverse_mut(f);
                n.right.traverse_mut(f);
            },
            ExpressionTree::Constant(n) => {
                n.child.traverse_mut(f);
            },
        }
    }
}

impl QuditSystem for ExpressionTree {
    fn get_dimension(&self) -> usize {
        match self {
            Self::Identity(s) => s.get_dimension(),
            Self::Kron(s) => s.get_dimension(),
            Self::Mul(s) => s.get_dimension(),
            Self::Leaf(s) => s.get_dimension(),
            Self::Perm(s) => s.get_dimension(),
            Self::Contract(s) => s.get_dimension(),
            Self::Constant(s) => s.get_dimension(),
        }
    }

    fn get_radices(&self) -> QuditRadices {
        match self {
            Self::Identity(s) => s.get_radices(),
            Self::Kron(s) => s.get_radices(),
            Self::Mul(s) => s.get_radices(),
            Self::Leaf(s) => s.get_radices(),
            Self::Perm(s) => s.get_radices(),
            Self::Contract(s) => s.get_radices(),
            Self::Constant(s) => s.get_radices(),
        }
    }
}

impl Function for ExpressionTree {
    fn get_num_params(&self) -> usize {
        match self {
            Self::Identity(s) => s.get_num_params(),
            Self::Kron(s) => s.get_num_params(),
            Self::Mul(s) => s.get_num_params(),
            Self::Leaf(s) => s.get_num_params(),
            Self::Perm(s) => s.get_num_params(),
            Self::Contract(s) => s.get_num_params(),
            Self::Constant(s) => s.get_num_params(),
        }
    }
}

impl BoundedFn for ExpressionTree {
    fn get_bounds(&self) -> Vec<std::ops::Range<f64>> {
        match self {
            Self::Identity(s) => s.get_bounds(),
            Self::Kron(s) => s.get_bounds(),
            Self::Mul(s) => s.get_bounds(),
            Self::Leaf(s) => s.get_bounds(),
            Self::Perm(s) => s.get_bounds(),
            Self::Contract(s) => s.get_bounds(),
            Self::Constant(s) => s.get_bounds(),
        }
    }
}
//
// impl<C: ComplexScalar> UnitaryFn<C> for ExpressionTree<C> {
//     fn write_unitary(
//         &self,
//         params: &[C::Re],
//         mut out_utry: &mut MatMut<C>,
//     ) {
//         match self {
//             Self::Identity(s) => s.write_unitary(params, &mut out_utry),
//             Self::Kron(s) => s.write_unitary(params, &mut out_utry),
//             Self::Mul(s) => s.write_unitary(params, &mut out_utry),
//             Self::Leaf(s) => s.write_unitary(params, &mut out_utry),
//             Self::Perm(s) => s.write_unitary(params, &mut out_utry),
//             Self::Contract(s) => s.write_unitary(params, &mut out_utry),
//         }
//     }
// }
//
// impl<C: ComplexScalar> DifferentiableUnitaryFn<C> for ExpressionTree<C> {
//     fn write_gradient(
//         &self,
//         params: &[C::Re],
//         out_grad: &mut UnitaryGradient<C>,
//     ) {
//         match self {
//             Self::Identity(s) => s.write_gradient(params, out_grad),
//             Self::Kron(s) => s.write_gradient(params, out_grad),
//             Self::Mul(s) => s.write_gradient(params, out_grad),
//             Self::Leaf(s) => s.write_gradient(params, out_grad),
//             Self::Perm(s) => s.write_gradient(params, out_grad),
//             Self::Contract(s) => s.write_gradient(params, out_grad),
//         }
//     }
//
//     fn write_unitary_and_gradient(
//         &self,
//         params: &[C::Re],
//         mut out_utry: &mut MatMut<C>,
//         out_grad: &mut UnitaryGradient<C>,
//     ) {
//         match self {
//             Self::Identity(s) => {
//                 s.write_unitary_and_gradient(params, &mut out_utry, out_grad)
//             },
//             Self::Kron(s) => {
//                 s.write_unitary_and_gradient(params, &mut out_utry, out_grad)
//             },
//             Self::Mul(s) => {
//                 s.write_unitary_and_gradient(params, &mut out_utry, out_grad)
//             },
//             Self::Leaf(s) => {
//                 s.write_unitary_and_gradient(params, &mut out_utry, out_grad)
//             },
//             Self::Perm(s) => {
//                 s.write_unitary_and_gradient(params, &mut out_utry, out_grad)
//             },
//             Self::Contract(s) => {
//                 s.write_unitary_and_gradient(params, &mut out_utry, out_grad)
//             },
//         }
//     }
// }
//
// impl<C: ComplexScalar> DoublyDifferentiableUnitaryFn<C> for ExpressionTree<C> {
//     fn write_hessian(
//         &self,
//         params: &[C::Re],
//         out_hess: &mut UnitaryHessian<C>,
//     ) {
//         match self {
//             Self::Identity(s) => s.write_hessian(params, out_hess),
//             Self::Kron(s) => s.write_hessian(params, out_hess),
//             Self::Mul(s) => s.write_hessian(params, out_hess),
//             Self::Leaf(s) => s.write_hessian(params, out_hess),
//             Self::Perm(s) => s.write_hessian(params, out_hess),
//             Self::Contract(s) => s.write_hessian(params, out_hess),
//         }
//     }
//
//     fn write_unitary_and_hessian(
//         &self,
//         params: &[C::Re],
//         mut out_utry: &mut MatMut<C>,
//         out_hess: &mut UnitaryHessian<C>,
//     ) {
//         match self {
//             Self::Identity(s) => {
//                 s.write_unitary_and_hessian(params, &mut out_utry, out_hess)
//             },
//             Self::Kron(s) => {
//                 s.write_unitary_and_hessian(params, &mut out_utry, out_hess)
//             },
//             Self::Mul(s) => {
//                 s.write_unitary_and_hessian(params, &mut out_utry, out_hess)
//             },
//             Self::Leaf(s) => {
//                 s.write_unitary_and_hessian(params, &mut out_utry, out_hess)
//             },
//             Self::Perm(s) => {
//                 s.write_unitary_and_hessian(params, &mut out_utry, out_hess)
//             },
//             Self::Contract(s) => {
//                 s.write_unitary_and_hessian(params, &mut out_utry, out_hess)
//             },
//         }
//     }
//
//     fn write_gradient_and_hessian(
//         &self,
//         params: &[C::Re],
//         out_grad: &mut UnitaryGradient<C>,
//         out_hess: &mut UnitaryHessian<C>,
//     ) {
//         match self {
//             Self::Identity(s) => {
//                 s.write_gradient_and_hessian(params, out_grad, out_hess)
//             },
//             Self::Kron(s) => {
//                 s.write_gradient_and_hessian(params, out_grad, out_hess)
//             },
//             Self::Mul(s) => {
//                 s.write_gradient_and_hessian(params, out_grad, out_hess)
//             },
//             Self::Leaf(s) => {
//                 s.write_gradient_and_hessian(params, out_grad, out_hess)
//             },
//             Self::Perm(s) => {
//                 s.write_gradient_and_hessian(params, out_grad, out_hess)
//             },
//             Self::Contract(s) => {
//                 s.write_gradient_and_hessian(params, out_grad, out_hess)
//             },
//         }
//     }
//
//     fn write_unitary_gradient_and_hessian(
//         &self,
//         params: &[C::Re],
//         mut out_utry: &mut MatMut<C>,
//         out_grad: &mut UnitaryGradient<C>,
//         out_hess: &mut UnitaryHessian<C>,
//     ) {
//         match self {
//             Self::Identity(s) => s.write_unitary_gradient_and_hessian(
//                 params, &mut out_utry, out_grad, out_hess,
//             ),
//             Self::Kron(s) => s.write_unitary_gradient_and_hessian(
//                 params, &mut out_utry, out_grad, out_hess,
//             ),
//             Self::Mul(s) => s.write_unitary_gradient_and_hessian(
//                 params, &mut out_utry, out_grad, out_hess,
//             ),
//             Self::Leaf(s) => s.write_unitary_gradient_and_hessian(
//                 params, &mut out_utry, out_grad, out_hess,
//             ),
//             Self::Perm(s) => s.write_unitary_gradient_and_hessian(
//                 params, &mut out_utry, out_grad, out_hess,
//             ),
//             Self::Contract(s) => s.write_unitary_gradient_and_hessian(
//                 params, &mut out_utry, out_grad, out_hess,
//             ),
//         }
//     }
// }

impl From<Gate> for ExpressionTree {
    fn from(gate: Gate) -> ExpressionTree {
        Self::Leaf(gate)
    }
}

impl std::hash::Hash for ExpressionTree {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        match self {
            Self::Identity(s) => s.hash(state),
            Self::Kron(s) => s.hash(state),
            Self::Mul(s) => s.hash(state),
            Self::Leaf(s) => s.hash(state),
            Self::Perm(s) => s.hash(state),
            Self::Contract(s) => s.hash(state),
            Self::Constant(s) => s.hash(state),
        }
    }
}

impl Eq for ExpressionTree {}

pub trait PrintTree {
    fn modify_prefix_for_child(
        &self,
        prefix: &str,
        last_child: bool,
    ) -> String {
        let mut new_prefix = String::from(prefix);

        // replace last ╠═ or ╚═ with ║
        new_prefix.pop();
        new_prefix.pop();
        new_prefix.pop();

        match new_prefix.pop() {
            Some('╠') => new_prefix.push_str("║   "),
            Some('╚') => new_prefix.push_str("    "),
            _ => (),
        };

        if last_child {
            new_prefix.push_str("╚══ ");
        } else {
            new_prefix.push_str("╠══ ");
        }

        new_prefix
    }

    fn write_tree(&self, prefix: &str, fmt: &mut std::fmt::Formatter<'_>);
}

impl PrintTree for ExpressionTree {
    fn write_tree(&self, prefix: &str, fmt: &mut std::fmt::Formatter<'_>) {
        match self {
            Self::Identity(s) => s.write_tree(prefix, fmt),
            Self::Kron(s) => s.write_tree(prefix, fmt),
            Self::Mul(s) => s.write_tree(prefix, fmt),
            Self::Leaf(s) => {
                writeln!(fmt, "{}{}", prefix, s.get_name()).unwrap()
            },
            Self::Perm(s) => s.write_tree(prefix, fmt),
            Self::Contract(s) => s.write_tree(prefix, fmt),
            Self::Constant(s) => s.write_tree(prefix, fmt),
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
