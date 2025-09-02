use super::NamedExpression;
use qudit_core::QuditRadices;

pub struct IsometryExpression {
    inner: NamedExpression,
    input_radices: QuditRadices,
    output_radices: QuditRadices,
}

impl From<IsometryExpression> for NamedExpression {
    fn from(value: IsometryExpression) -> Self {
        value.inner
    }
}

