// mod optimizer;
mod fmt;

mod leaf;
mod tree;
mod matmul;
mod transpose;
mod outer;
mod trace;
mod hadamard;

pub use leaf::LeafNode;
pub use matmul::MatMulNode;
pub use transpose::TransposeNode;
pub use outer::OuterProductNode;
pub use trace::TraceNode;
pub use hadamard::HadamardProductNode;

// pub use optimizer::TreeOptimizer;
pub use tree::TTGTNode;
pub use tree::TTGTTree;

