use std::ops::{Deref, DerefMut};

use crate::{expressions::JittableExpression, index::{IndexDirection, TensorIndex}, GenerationShape, TensorExpression};

use super::NamedExpression;
use qudit_core::QuditRadices;
use qudit_core::QuditSystem;

#[derive(PartialEq, Eq, Debug, Clone)]
pub struct KrausOperatorsExpression {
    inner: NamedExpression,
    input_radices: QuditRadices,
    output_radices: QuditRadices,
    num_operators: usize,
}

impl JittableExpression for KrausOperatorsExpression {
    fn generation_shape(&self) -> GenerationShape {
        GenerationShape::Tensor3D(
            self.num_operators,
            self.output_radices.dimension(),
            self.input_radices.dimension()
        )
    }
}

impl AsRef<NamedExpression> for KrausOperatorsExpression {
    fn as_ref(&self) -> &NamedExpression {
        &self.inner
    }
}

impl From<KrausOperatorsExpression> for NamedExpression {
    fn from(value: KrausOperatorsExpression) -> Self {
        value.inner
    }
}

impl Deref for KrausOperatorsExpression {
    type Target = NamedExpression;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl DerefMut for KrausOperatorsExpression {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.inner
    }
}

impl From<KrausOperatorsExpression> for TensorExpression {
    fn from(value: KrausOperatorsExpression) -> Self {
        let KrausOperatorsExpression { inner, input_radices, output_radices, num_operators } = value;
        // TODO: add a proper implementation of into_iter for QuditRadices
        let indices = [num_operators].into_iter()
            .map(|r| (IndexDirection::Batch, r))
            .chain(output_radices.into_iter().map(|r| (IndexDirection::Output, *r as usize)))
            .chain(input_radices.into_iter().map(|r| (IndexDirection::Input, *r as usize)))
            .enumerate()
            .map(|(i, (d, r))| TensorIndex::new(d, i, r))
            .collect();
        TensorExpression::from_raw(indices, inner)
    }
}

impl TryFrom<TensorExpression> for KrausOperatorsExpression {
    // TODO: Come up with proper error handling
    type Error = String;

    fn try_from(value: TensorExpression) -> Result<Self, Self::Error> {
        let mut num_operators = None;
        let mut input_radices = vec![];
        let mut output_radices = vec![];
        for idx in value.indices() {
            match idx.direction() {
                IndexDirection::Batch => {
                    match num_operators {
                        Some(n) => return Err(String::from("More than one batch index in kraus operator conversion.")),
                        None => num_operators = Some(idx.index_size()),
                    }
                }
                IndexDirection::Input => { input_radices.push(idx.index_size()); }
                IndexDirection::Output => { output_radices.push(idx.index_size()); }
                _ => unreachable!()
            }
        }
        
        Ok(KrausOperatorsExpression {
            inner: value.into(),
            input_radices: input_radices.into(),
            output_radices: output_radices.into(),
            num_operators: num_operators.unwrap_or(1),
        })
    }
}
