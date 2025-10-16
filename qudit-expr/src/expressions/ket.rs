use std::ops::{Deref, DerefMut};

use crate::{expressions::JittableExpression, index::{IndexDirection, TensorIndex}, GenerationShape, TensorExpression};

use super::NamedExpression;
use super::ComplexExpression;
use qudit_core::QuditRadices;
use qudit_core::QuditSystem;

#[derive(PartialEq, Eq, Debug, Clone)]
pub struct KetExpression {
    inner: NamedExpression,
    radices: QuditRadices,
}

impl KetExpression {
    pub fn zero<R: Into<QuditRadices>>(radices: R) -> Self {
        let name = "zero";
        let radices = radices.into();
        let mut body = vec![ComplexExpression::zero(); radices.dimension()];
        body[0] = ComplexExpression::one();
        let variables = vec![];
        let inner = NamedExpression::new(name, variables, body);
        KetExpression {
            inner,
            radices,
        }
    }

    pub fn num_qudits(&self) -> usize {
        self.radices.num_qudits()
    }
}

impl JittableExpression for KetExpression {
    fn generation_shape(&self) -> GenerationShape {
        GenerationShape::Matrix(self.radices.dimension(), 1)
    }
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
        TensorExpression::from_raw(indices, inner)
    }
}

impl TryFrom<TensorExpression> for KetExpression {
    // TODO: Come up with proper error handling
    type Error = String;

    fn try_from(value: TensorExpression) -> Result<Self, Self::Error> {
        if value.indices().iter().any(|idx| idx.direction() != IndexDirection::Output) {
            return Err(String::from("Cannot convert a tensor with non-output indices to a ket."));
        }
        let radices = QuditRadices::from_iter(value.indices().iter().map(|idx| idx.index_size()));
        Ok(KetExpression {
            inner: value.into(),
            radices,
        })
    }
}
