use super::NamedExpression;
use qudit_core::QuditRadices;

pub struct KetSystemExpression {
    inner: NamedExpression,
    radices: QuditRadices,
    num_states: usize,
}

impl From<KetSystemExpression> for NamedExpression {
    fn from(value: KetSystemExpression) -> Self {
        value.inner
    }
}

