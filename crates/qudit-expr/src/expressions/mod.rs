mod base;
mod bra;
mod brasys;
mod complex;
mod generator;
mod isometry;
mod ket;
mod ketsys;
mod kraus;
mod named;
mod tensor;
mod unitary;
mod utrysys;

#[allow(dead_code)]
pub trait JittableExpression: Into<NamedExpression> {
    fn generation_shape(&self) -> crate::GenerationShape;
}

pub use base::Constant;
pub use base::Expression;
pub use bra::BraExpression;
pub use brasys::BraSystemExpression;
pub use complex::ComplexExpression;
pub use generator::ExpressionGenerator;
pub use isometry::IsometryExpression;
pub use ket::KetExpression;
pub use ketsys::KetSystemExpression;
pub use kraus::KrausOperatorsExpression;
pub use named::ExpressionBody;
pub use named::NamedExpression;
pub use tensor::TensorExpression;
pub use unitary::UnitaryExpression;
pub use utrysys::UnitarySystemExpression;
