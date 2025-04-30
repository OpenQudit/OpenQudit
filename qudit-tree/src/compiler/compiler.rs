// use crate::compiler::{
//     bytecode::{remove_identity_frpr, Bytecode},
//     BufferOptimizer, BufferReuser, BytecodeGenerator, ExpressionTree,
//     StaticBytecodeOptimizer,
// };
use crate::tree::ExpressionTree;
use crate::bytecode::{Bytecode, BytecodeGenerator};
use crate::bytecode::StaticBytecodeOptimizer;
use crate::bytecode::remove_identity_frpr;
// use crate::bytecode::BufferOptimizer;

pub fn compile(tree: &ExpressionTree) -> Bytecode {
    let code = BytecodeGenerator::new().generate(tree);
    let code = StaticBytecodeOptimizer::new(code).optimize();
    let code = remove_identity_frpr(code);
    // let code = BufferOptimizer::new().optimize(code);
    // let code = BufferReuser::new().reuse_buffers(code);
    code
}
