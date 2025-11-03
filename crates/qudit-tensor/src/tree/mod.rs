// mod optimizer;
mod fmt;

mod hadamard;
mod leaf;
mod matmul;
mod outer;
mod trace;
mod transpose;
mod tree;

pub use hadamard::HadamardProductNode;
pub use leaf::LeafNode;
pub use trace::TraceNode;
pub use transpose::TransposeNode;

// pub use optimizer::TreeOptimizer;
pub use tree::TTGTNode;
pub use tree::TTGTTree;
