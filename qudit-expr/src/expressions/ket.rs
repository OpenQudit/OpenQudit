use std::ops::{Deref, DerefMut};

use crate::{index::{IndexDirection, TensorIndex}, TensorExpression};

use super::NamedExpression;
use qudit_core::QuditRadices;

pub struct KetExpression {
    inner: NamedExpression,
    radices: QuditRadices,
}

impl AsRef<NamedExpression> for KetExpression {
    fn as_ref(&self) -> &NamedExpression {
        &self.inner
    }
}

impl From<KetExpression> for NamedExpression {
    fn from(value: KetExpression) -> Self {
        value.inner
    }
}

impl Deref for KetExpression {
    type Target = NamedExpression;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl DerefMut for KetExpression {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.inner
    }
}

impl From<KetExpression> for TensorExpression {
    fn from(value: KetExpression) -> Self {
        let KetExpression { inner, radices } = value;
        // TODO: add a proper implementation of into_iter for QuditRadices
        let indices = radices.into_iter()
            .enumerate()
            .map(|(i, r)| TensorIndex::new(IndexDirection::Output, i, *r as usize))
            .collect();
        TensorExpression::from_raw(indices, inner);
    }
}

impl TryFrom<TensorExpression> for KetExpression {
    // TODO: Come up with proper error handling
    type Error = String;

    fn try_from(value: TensorExpression) -> Result<Self, Self::Error> {
        if value.indices().iter().map(|idx| idx.direction() != IndexDirection::Output).any() {
            return Err("Cannot convert a tensor with non-output indices to a ket.");
        }
        let radices = QuditRadices::from_iter(value.indices().iter().map(|idx| idx.index_size()));
        KetExpression {
            inner: value.into(),
            radices,
        }
    }
}
