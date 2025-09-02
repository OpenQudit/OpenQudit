use super::NamedExpression;
use qudit_core::QuditRadices;

pub struct BraSystemExpression {
    inner: NamedExpression,
    radices: QuditRadices,
    num_states: usize,
}

impl From<BraSystemExpression> for NamedExpression {
    fn from(value: BraSystemExpression) -> Self {
        value.inner
    }
}

