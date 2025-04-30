mod tree;
mod bytecode;
mod compiler;
mod qvm;

pub use tree::TreeOptimizer;
pub use tree::BuilderExpressionInput;
pub use tree::TreeBuilder;
pub use tree::ExpressionTree;
pub use compiler::compile;
pub use qvm::QVM;

#[cfg(test)]
mod tests {
    // use super::tree::TreeBuilder;
    // use super::bytecode::Bytecode;

    #[test]
    fn test_tree() {
        assert_eq!(1, 1);
    }
}
