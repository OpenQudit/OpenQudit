use std::ops::{Deref, DerefMut};

use crate::{expressions::JittableExpression, index::{IndexDirection, TensorIndex}, GenerationShape, TensorExpression};

use super::NamedExpression;
use qudit_core::QuditRadices;
use qudit_core::QuditSystem;

#[derive(PartialEq, Eq, Debug, Clone)]
pub struct BraSystemExpression {
    inner: NamedExpression,
    radices: QuditRadices,
    num_states: usize,
}

impl BraSystemExpression {
    pub fn new<T: AsRef<str>>(input: T) -> Self {
        TensorExpression::new(input).try_into().unwrap()
    }
}

impl JittableExpression for BraSystemExpression {
    fn generation_shape(&self) -> GenerationShape {
        GenerationShape::Tensor3D(self.num_states, 1, self.radices.dimension())
    }
}

impl AsRef<NamedExpression> for BraSystemExpression {
    fn as_ref(&self) -> &NamedExpression {
        &self.inner
    }
}

impl From<BraSystemExpression> for NamedExpression {
    fn from(value: BraSystemExpression) -> Self {
        value.inner
    }
}

impl Deref for BraSystemExpression {
    type Target = NamedExpression;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl DerefMut for BraSystemExpression {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.inner
    }
}


// TODO: replace individual From<X> for TensorExpression impls with blanket one
// pub trait HasIndices {
//     fn indices(&self) -> &[TensorIndex];
// }

// impl<T: HasIndices + Into<NamedExpression>> From<T> for TensorExpression {
//     fn from(value: T) -> Self {
//         let indices = value.indices().iter().cloned().collect();
//         let inner = value.into();
//         TensorExpression::from_raw(indices, inner)
//     }
// }


impl From<BraSystemExpression> for TensorExpression {
    fn from(value: BraSystemExpression) -> Self {
        let BraSystemExpression { inner, radices, num_states } = value;
        // TODO: add a proper implementation of into_iter for QuditRadices
        let indices = [num_states].into_iter()
            .map(|r| (IndexDirection::Batch, r))
            .chain(radices.into_iter().map(|r| (IndexDirection::Input, *r as usize)))
            .enumerate()
            .map(|(i, (d, r))| TensorIndex::new(d, i, r))
            .collect();
        TensorExpression::from_raw(indices, inner)
    }
}

impl TryFrom<TensorExpression> for BraSystemExpression {
    // TODO: Come up with proper error handling
    type Error = String;

    fn try_from(value: TensorExpression) -> Result<Self, Self::Error> {
        let mut num_states = None;
        let mut radices = vec![];
        for idx in value.indices() {
            match idx.direction() {
                IndexDirection::Batch => {
                    match num_states {
                        Some(n) => return Err(String::from("More than one batch index in bra system conversion.")),
                        None => num_states = Some(idx.index_size()),
                    }
                }
                IndexDirection::Input => { radices.push(idx.index_size()); }
                _ => { return Err(String::from("Cannot convert a tensor with non-input or batch indices to a bra system.")); }
            }
        }
        
        Ok(BraSystemExpression {
            inner: value.into(),
            radices: radices.into(),
            num_states: num_states.unwrap_or(1),
        })
    }
}
