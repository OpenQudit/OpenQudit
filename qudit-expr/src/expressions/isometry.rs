use std::ops::{Deref, DerefMut};

use crate::{expressions::JittableExpression, index::{IndexDirection, TensorIndex}, GenerationShape, TensorExpression};

use super::NamedExpression;
use qudit_core::QuditRadices;
use qudit_core::QuditSystem;

#[derive(PartialEq, Eq, Debug, Clone)]
pub struct IsometryExpression {
    inner: NamedExpression,
    input_radices: QuditRadices,
    output_radices: QuditRadices,
}

impl JittableExpression for IsometryExpression {
    fn generation_shape(&self) -> GenerationShape {
        GenerationShape::Matrix(self.output_radices.dimension(), self.input_radices.dimension())
    }
}

impl AsRef<NamedExpression> for IsometryExpression {
    fn as_ref(&self) -> &NamedExpression {
        &self.inner
    }
}

impl From<IsometryExpression> for NamedExpression {
    fn from(value: IsometryExpression) -> Self {
        value.inner
    }
}

impl Deref for IsometryExpression {
    type Target = NamedExpression;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl DerefMut for IsometryExpression {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.inner
    }
}

impl From<IsometryExpression> for TensorExpression {
    fn from(value: IsometryExpression) -> Self {
        let IsometryExpression { inner, input_radices, output_radices } = value;
        // TODO: add a proper implementation of into_iter for QuditRadices
        let indices = output_radices.into_iter()
            .map(|r| (IndexDirection::Output, *r as usize))
            .chain(input_radices.into_iter().map(|r| (IndexDirection::Input, *r as usize)))
            .enumerate()
            .map(|(i, (d, r))| TensorIndex::new(d, i, r))
            .collect();
        TensorExpression::from_raw(indices, inner)
    }
}

impl TryFrom<TensorExpression> for IsometryExpression {
    // TODO: Come up with proper error handling
    type Error = String;

    fn try_from(value: TensorExpression) -> Result<Self, Self::Error> {
        let mut input_radices = vec![];
        let mut output_radices = vec![];
        for idx in value.indices() {
            match idx.direction() {
                IndexDirection::Input => { input_radices.push(idx.index_size()); }
                IndexDirection::Output => { output_radices.push(idx.index_size()); }
                _ => { return Err(String::from("Cannot convert a tensor with non-input, non-output indices to an isometry.")); }
            }
        }
        
        Ok(IsometryExpression {
            inner: value.into(),
            input_radices: input_radices.into(),
            output_radices: output_radices.into(),
        })
    }
}
