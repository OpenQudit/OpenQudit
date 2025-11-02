type TensorId = usize;

mod index;
mod tensor;
mod path;
mod network;
mod builder;

pub use tensor::QuditTensor;
pub use network::QuditTensorNetwork;
pub use builder::QuditCircuitTensorNetworkBuilder;

