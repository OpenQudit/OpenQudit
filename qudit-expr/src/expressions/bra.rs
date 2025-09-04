use std::ops::Deref;
use std::ops::DerefMut;

use crate::expressions::JittableExpression;
use crate::index::IndexDirection;
use crate::index::TensorIndex;
use crate::GenerationShape;
use crate::TensorExpression;

use super::NamedExpression;
use qudit_core::QuditRadices;
use qudit_core::QuditSystem;

#[derive(PartialEq, Eq, Debug, Clone)]
pub struct BraExpression {
    inner: NamedExpression,
    radices: QuditRadices,
}

impl JittableExpression for BraExpression {
    fn generation_shape(&self) -> GenerationShape {
        GenerationShape::Vector(self.radices.dimension())
    }
}

impl AsRef<NamedExpression> for BraExpression {
    fn as_ref(&self) -> &NamedExpression {
        &self.inner
    }
}

impl From<BraExpression> for NamedExpression {
    fn from(value: BraExpression) -> Self {
        value.inner
    }
}

impl Deref for BraExpression {
    type Target = NamedExpression;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl DerefMut for BraExpression {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.inner
    }
}

impl From<BraExpression> for TensorExpression {
    fn from(value: BraExpression) -> Self {
        let BraExpression { inner, radices } = value;
        // TODO: add a proper implementation of into_iter for QuditRadices
        let indices = radices.into_iter()
            .enumerate()
            .map(|(i, r)| TensorIndex::new(IndexDirection::Input, i, *r as usize))
            .collect();
        TensorExpression::from_raw(indices, inner)
    }
}

impl TryFrom<TensorExpression> for BraExpression {
    // TODO: Come up with proper error handling
    type Error = String;

    fn try_from(value: TensorExpression) -> Result<Self, Self::Error> {
        if value.indices().iter().any(|idx| idx.direction() != IndexDirection::Input) {
            return Err(String::from("Cannot convert a tensor with non-input indices to a bra."));
        }
        let radices = QuditRadices::from_iter(value.indices().iter().map(|idx| idx.index_size()));
        Ok(BraExpression {
            inner: value.into(),
            radices,
        })
    }
}
