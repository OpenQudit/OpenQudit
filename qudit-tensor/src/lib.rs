mod tree;
mod network;
mod bytecode;
mod networknew;
mod qvm;

pub use network::QuditTensor;
pub use network::QuditCircuitNetwork;
pub use bytecode::Bytecode;
pub use qvm::QVM;
pub use qvm::QVMResult;
pub use qvm::QVMReturnType;

pub fn compile_network(network: &QuditCircuitNetwork) -> Bytecode {
    let optimal_path = network.optimize_optimal_simple();
    let tree = network.path_to_expression_tree(&optimal_path);
    let tree = crate::tree::TreeOptimizer::new().optimize(tree);
    crate::bytecode::BytecodeGenerator::new().generate(&tree)
}
