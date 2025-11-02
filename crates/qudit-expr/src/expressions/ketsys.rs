use std::ops::{Deref, DerefMut};

use crate::{expressions::JittableExpression, index::{IndexDirection, TensorIndex}, GenerationShape, TensorExpression};

use super::NamedExpression;
use qudit_core::Radices;
use qudit_core::QuditSystem;

#[derive(PartialEq, Eq, Debug, Clone)]
pub struct KetSystemExpression {
    inner: NamedExpression,
    radices: Radices,
    num_states: usize,
}

impl JittableExpression for KetSystemExpression {
    fn generation_shape(&self) -> GenerationShape {
        GenerationShape::Tensor3D(self.num_states, self.radices.dimension(), 1)
    }
}

impl AsRef<NamedExpression> for KetSystemExpression {
    fn as_ref(&self) -> &NamedExpression {
        &self.inner
    }
}

impl From<KetSystemExpression> for NamedExpression {
    fn from(value: KetSystemExpression) -> Self {
        value.inner
    }
}

impl Deref for KetSystemExpression {
    type Target = NamedExpression;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl DerefMut for KetSystemExpression {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.inner
    }
}

impl From<KetSystemExpression> for TensorExpression {
    fn from(value: KetSystemExpression) -> Self {
        let KetSystemExpression { inner, radices, num_states } = value;
        // TODO: add a proper implementation of into_iter for QuditRadices
        let indices = [num_states].into_iter()
            .map(|r| (IndexDirection::Batch, r))
            .chain(radices.into_iter().map(|r| (IndexDirection::Output, usize::from(*r))))
            .enumerate()
            .map(|(i, (d, r))| TensorIndex::new(d, i, r))
            .collect();
        TensorExpression::from_raw(indices, inner)
    }
}

impl TryFrom<TensorExpression> for KetSystemExpression {
    // TODO: Come up with proper error handling
    type Error = String;

    fn try_from(value: TensorExpression) -> Result<Self, Self::Error> {
        let mut num_states = None;
        let mut radices = vec![];
        for idx in value.indices() {
            match idx.direction() {
                IndexDirection::Batch => {
                    match num_states {
                        Some(n) => num_states = Some(n * idx.index_size()),
                        None => num_states = Some(idx.index_size()),
                    }
                }
                IndexDirection::Output => { radices.push(idx.index_size()); }
                _ => { return Err(String::from("Cannot convert a tensor with non-output or batch indices to a ket system.")); }
            }
        }
        
        Ok(KetSystemExpression {
            inner: value.into(),
            radices: radices.into(),
            num_states: num_states.unwrap_or(1),
        })
    }
}
