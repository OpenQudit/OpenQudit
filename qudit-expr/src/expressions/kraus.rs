use super::NamedExpression;
use qudit_core::QuditRadices;

pub struct KrausOperatorsExpression {
    inner: NamedExpression,
    input_radices: QuditRadices,
    output_radices: QuditRadices,
    num_operators: usize,
}

impl From<KrausOperatorsExpression> for NamedExpression {
    fn from(value: KrausOperatorsExpression) -> Self {
        value.inner
    }
}

