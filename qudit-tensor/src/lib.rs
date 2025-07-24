mod tree;
mod network;
mod bytecode;
mod cpu;

pub use network::QuditTensor;
pub use network::QuditTensorNetwork;
pub use network::QuditCircuitTensorNetworkBuilder;
pub use bytecode::Bytecode;
pub use cpu::TNVM;
pub use cpu::TNVMResult;
pub use cpu::TNVMReturnType;

pub fn compile_network(network: QuditTensorNetwork) -> Bytecode {
    let optimal_path = network.solve_for_path();
    let tree = network.path_to_expression_tree(optimal_path);
    // let tree = crate::tree::TreeOptimizer::new().optimize(tree);
    crate::bytecode::BytecodeGenerator::new().generate(tree)
}

#[cfg(test)]
mod tests {
    use qudit_expr::DifferentiationLevel;

    use crate::bytecode::BytecodeGenerator;
    // use crate::qvm::QVM;
    use qudit_core::QuditRadices;
    use qudit_expr::TensorExpression;
    use qudit_expr::{FUNCTION, GRADIENT, HESSIAN};

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
        
        let mut network = QuditCircuitTensorNetworkBuilder::new(QuditRadices::new(&vec![2, 2]))
            // .prepend(QuditTensor::new(ZZ.clone(), vec![].into()), vec![0, 1], vec![0, 1], vec!["a".to_string()])
            .prepend(QuditTensor::new(classically_controlled_u3.clone(), vec![0, 1, 2].into()), vec![0], vec![0], vec!["a".to_string()])
            .prepend(QuditTensor::new(classically_controlled_u3.clone(), vec![0, 1, 2].into()), vec![1], vec![1], vec!["a".to_string()])
            .build();

        let optimal_path = network.solve_for_path();
        println!("Optimal Path: {:?}", optimal_path.path);
        let tree = network.path_to_expression_tree(optimal_path);
        println!("Expression Tree: {:?}", tree);
        // let tree = TreeOptimizer::new().optimize(tree);
        // println!("Expression Tree: {:?}", tree);
        let code = BytecodeGenerator::new().generate(tree);
        println!("Bytecode: \n{:?}", code);
        // let Bytecode { expressions, const_code, dynamic_code, buffers } = code;
        // let code = Bytecode { expressions, const_code, dynamic_code: dynamic_code[..1].to_vec(), buffers };

        let params = [1.7, 1.7, 1.7];
        let mut tnvm = TNVM::<qudit_core::c64, GRADIENT>::new(&code);
        let out = tnvm.evaluate(&params);
        let out_fn = out.get_fn_result().unpack_tensor3d();
        println!("{:?}", out_fn);
        let out_fn = out.get_grad_result().unpack_tensor4d();
        println!("{:?}", out_fn);
//         // Option 1:
//         {
//             let mut tnvm: TNVM<qudit_core::c64, GRADIENT> = TNVM::new(code);
//             let out_buffer = tnvm.evaluate(&params);
//             let out_fn = out_buffer.get_fn_result().unpack_matvec(); // This might be unsafe tho?
//             let out_grad = out_buffer.get_grad_result().unpack_tensor4d();
//         }
//         // Option 2:
//         {
//             let mut tnvm: TNVM<qudit_core::c64, GRADIENT> = TNVM::new(code);
//             let (fn_buffer, grad_buffer) = tnvm.evaluate(&params);
//             let out_fn = fn_buffer.unpack_tensor3d();
//             let out_grad = grad_buffer.unpack_tensor4d();
//         }
//         println!("Output: {:?}", out_fn);
//         // println!("Output grad: {:?}", out_grad);
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
