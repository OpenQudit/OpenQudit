mod tree;
mod network;
mod bytecode;
mod cpu;

pub use network::QuditTensor;
pub use network::QuditTensorNetwork;
pub use network::QuditCircuitTensorNetworkBuilder;
pub use bytecode::Bytecode;
pub use cpu::TNVM;
pub use cpu::PinnedTNVM;
pub use cpu::TNVMResult;
pub use cpu::TNVMReturnType;

pub fn compile_network(network: QuditTensorNetwork) -> Bytecode {
    let optimal_path = network.solve_for_path();

    // println!("{:?}", optimal_path);
    let tree = network.path_to_ttgt_tree(optimal_path);
    // println!("{:?}", tree);
    crate::bytecode::BytecodeGenerator::new().generate(tree)
}

#[cfg(test)]
mod tests {
    use qudit_expr::DifferentiationLevel;

    use crate::bytecode::BytecodeGenerator;
    // use crate::qvm::QVM;
    use qudit_core::{ParamInfo, QuditRadices};
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
        // let p3 = TensorExpression::new("Phase<3>(a, b) {
        //     [
        //         [ 1, 0, 0 ],
        //         [ 0, e^(i*a), 0 ],
        //         [ 0, 0, e^(i*b) ]
        //     ]
        // }");
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
        
        let mut network = QuditCircuitTensorNetworkBuilder::new(QuditRadices::new(&vec![2, 2]), None)
            .prepend_expression(ZZ.clone(), ParamInfo::empty(), vec![0, 1], vec![0, 1], vec!["a".to_string()])
            .prepend_expression(classically_controlled_u3.clone(), ParamInfo::parameterized(vec![0, 1, 2]), vec![0], vec![0], vec!["a".to_string()])
            .build();

        let optimal_path = network.solve_for_path();
        println!("Optimal Path: {:?}", optimal_path.path);
        let tree = network.path_to_ttgt_tree(optimal_path);
        println!("Expression Tree: {:?}", tree);
        let code = BytecodeGenerator::new().generate(tree);
        println!("Bytecode: \n{:?}", code);
        // let Bytecode { expressions, const_code, dynamic_code, buffers } = code;
        // let code = Bytecode { expressions, const_code, dynamic_code: dynamic_code[..1].to_vec(), buffers };

        let params = [1.7, 1.7, 1.7];
        let mut tnvm = TNVM::<qudit_core::c64, GRADIENT>::new(&code, None);
        let out = tnvm.evaluate::<GRADIENT>(&params);
        let out_fn = out.get_fn_result().unpack_tensor3d();
        println!("{:?}", out_fn);
        let out_fn = out.get_grad_result().unpack_tensor4d();
        println!("{:?}", out_fn);
    }
}
