use std::collections::{BTreeSet, HashMap};

use crate::tree::ExpressionTree;


mod path;
mod tensor;
mod contraction;
mod network;
pub use tensor::QuditTensor;
pub use network::QuditCircuitNetwork;

#[derive(Debug, Clone)]
pub(crate) enum Wire {
    Empty,
    Connected(usize, usize), // node_id, local_index_id
    Closed,
}

type QuditId = usize;
type TensorId = usize;
type IndexId = usize;
type IndexSize = usize;
type SubNetwork = u64;
type Cost = usize;

/// IndexDirection represents a tensor's leg direction from the quantum circuit perspective
///
/// Left corresponds to inputs, right to outputs, and up as parallel dimensions
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum IndexDirection {
    Input,
    Output,
    Up,
    // Down,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct LocalTensorIndex {
    direction: IndexDirection,
    index_id: IndexId,
    index_size: IndexSize,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum NetworkIndex {
    Open(usize, IndexDirection), // qudit_id for left and right and up_index_id for up
    Contracted(usize), // contraction_id 
    Shared(usize, IndexDirection), // batch index id
}

impl NetworkIndex {
    pub fn is_shared(&self) -> bool {
        matches!(self, &NetworkIndex::Shared(..))
    }
}

#[cfg(test)]
mod tests {
    use qudit_expr::DifferentiationLevel;

    use crate::{bytecode::BytecodeGenerator, tree::TreeOptimizer};
    use crate::qvm::QVM;
    use qudit_core::QuditRadices;
    use qudit_expr::TensorExpression;

    use super::*;

    #[test]
    fn test_projective_measurement() {
        let u3 = TensorExpression::new("U3(a, b, c) {
            [
                [cos(a/2), ~e^(c*i)*sin(a/2)],
                [e^(b*i)*sin(a/2), e^(i*(b+c))*cos(a/2)],
            ]
        }"); 
        let classically_controlled_u3 = u3.stack_with_identity(&[1], 2);
        let ZZ = TensorExpression::new("ZZParity() {
            [
                [
                    [ 1, 0, 0, 0 ], 
                    [ 0, 0, 0, 0 ],
                    [ 0, 0, 0, 0 ],
                    [ 0, 0, 0, 1 ],
                ],
                [
                    [ 0, 0, 0, 0 ], 
                    [ 0, 1, 0, 0 ],
                    [ 0, 0, 1, 0 ],
                    [ 0, 0, 0, 0 ],
                ],
            ]
        }");
        
        let mut network = QuditCircuitNetwork::new(QuditRadices::new(&vec![2, 2]));
        network.prepend(QuditTensor::new(ZZ.clone(), vec![].into()), vec![0, 1], vec![0, 1], vec!["a".to_string()]);
        network.prepend(QuditTensor::new(classically_controlled_u3.clone(), vec![0, 1, 2].into()), vec![0], vec![0], vec!["a".to_string()]);

        let optimal_path = network.optimize_optimal_simple();
        println!("Optimal Path: {:?}", optimal_path.path);
        let tree = network.path_to_expression_tree(&optimal_path);
        println!("Expression Tree: {:?}", tree);
        let tree = TreeOptimizer::new().optimize(tree);
        println!("Expression Tree: {:?}", tree);
        let code = BytecodeGenerator::new().generate(&tree);
        println!("Bytecode: {:?}", code);

        let mut qvm: QVM<qudit_core::c64> = QVM::new(code, DifferentiationLevel::None);
        let params = [1.7, 1.7, 1.7];
        let out_buffer = qvm.evaluate(&params);
        let out_fn = out_buffer.get_fn_result().unpack_matvec();
        // let out_grad = out_buffer.get_grad_result().unpack_tensor4d();
        println!("Output: {:?}", out_fn);
        // println!("Output grad: {:?}", out_grad);
    }

    // #[test]
    // fn test_optimize_optimal() {
    //     let mut network = QuditCircuitNetwork::new(QuditRadices::new(&vec![2, 2]));
    //     let cnot = TensorExpression::new("CNOT() {
    //         [
    //             [1, 0, 0, 0],
    //             [0, 1, 0, 0],
    //             [0, 0, 0, 1],
    //             [0, 0, 1, 0],
    //         ]
    //     }");

    //     let u3 = TensorExpression::new("U3(a, b, c) {
    //         [
    //             [cos(a/2), ~e^(c*i)*sin(a/2)],
    //             [e^(b*i)*sin(a/2), e^(i*(b+c))*cos(a/2)],
    //         ]
    //     }"); 
    //     let ZZ = TensorExpression::new("ZZParity() {
    //         [
    //             [
    //                 [ 1, 0, 0, 0 ], 
    //                 [ 0, 0, 0, 0 ],
    //                 [ 0, 0, 0, 0 ],
    //                 [ 0, 0, 0, 1 ],
    //             ],
    //             [
    //                 [ 0, 0, 0, 0 ], 
    //                 [ 0, 1, 0, 0 ],
    //                 [ 0, 0, 1, 0 ],
    //                 [ 0, 0, 0, 0 ],
    //             ],
    //         ]
    //     }");
    //     network.prepend(QuditTensor::new(u3.clone(), vec![0, 1, 2].into()), vec![0], vec![0]);
    //     network.prepend(QuditTensor::new(ZZ.clone(), vec![].into()), vec![0, 1], vec![0, 1]);
    //     // network.prepend(QuditTensor::new(cnot.clone(), ParamIndices::constant()), vec![0,1], vec![0,1]);
    //     // network.prepend(QuditTensor::new(cnot.clone(), ParamIndices::constant()), vec![1,2], vec![1,2]);
    //     // let expr = TensorExpression::new("A() {
    //     //     [
    //     //         [1, 0, 0, 0],
    //     //         [0, 1, 0, 0],
    //     //         [0, 0, 0, 1],
    //     //         [0, 0, 1, 0],
    //     //     ]
    //     // }");
    //     // network.prepend(
    //     //     QuditTensor::new(expr.clone(), vec![2, 2], vec![2, 2], vec![], ParamIndices::constant()),
    //     //     vec![0, 1],
    //     //     vec![0, 1],
    //     // );
    //     // network.prepend(
    //     //     QuditTensor::new(expr.clone(), vec![2, 2], vec![2, 2], vec![], ParamIndices::constant()),
    //     //     vec![1, 2],
    //     //     vec![1, 2],
    //     // );
    //     // network.prepend(
    //     //     QuditTensor::new(expr.clone(), vec![2, 2], vec![2, 2], vec![], ParamIndices::constant()),
    //     //     vec![0, 1],
    //     //     vec![0, 1],
    //     // );
    //     // network.prepend(
    //     //     QuditTensor::new(expr.clone(), vec![2, 2], vec![2, 2], vec![], ParamIndices::constant()),
    //     //     vec![1, 2],
    //     //     vec![1, 2],
    //     // );
    //     // println!("expr shape: {:?}", expr.shape);
    //     let optimal_path = network.optimize_optimal_simple();
    //     println!("Optimal Path: {:?}", optimal_path.path);
    //     let tree = network.path_to_expression_tree(&optimal_path);
    //     println!("Expression Tree: {:?}", tree);
    //     let tree = TreeOptimizer::new().optimize(tree);
    //     println!("Expression Tree: {:?}", tree);
    //     let code = BytecodeGenerator::new().generate(&tree);
    //     println!("Bytecode: {:?}", code);
    //     let mut qvm: QVM<qudit_core::c64> = QVM::new(code, DifferentiationLevel::Gradient);
    //     let params = [1.7, 1.7, 1.7];
    //     let out_buffer = qvm.evaluate(&params);
    //     let out_fn = out_buffer.get_fn_result().unpack_matvec();
    //     let out_grad = out_buffer.get_grad_result().unpack_tensor4d();
    //     println!("Output: {:?}", out_fn);
    //     println!("Output grad: {:?}", out_grad);
    //     // let out = qvm.get_unitary(&params);
    //     // println!("Unitary: {:?}", out);
    //     // network.prepend(
    //     //     QuditTensor::new(vec![2, 2], vec![2, 2], vec![]),
    //     //     vec![0, 1],
    //     //     vec![0, 1],
    //     // );
    //     // network.prepend(
    //     //     QuditTensor::new(vec![2, 2], vec![2, 2], vec![]),
    //     //     vec![1, 2],
    //     //     vec![1, 2],
    //     // );
    //     // network.prepend(
    //     //     QuditTensor::new(vec![2, 2], vec![2, 2], vec![]),
    //     //     vec![0, 1],
    //     //     vec![0, 1],
    //     // );
    //     // network.prepend(
    //     //     QuditTensor::new(vec![2, 2], vec![2, 2], vec![]),
    //     //     vec![1, 2],
    //     //     vec![1, 2],
    //     // );
    //     assert!(optimal_path.cost > 0);
    // }
}
