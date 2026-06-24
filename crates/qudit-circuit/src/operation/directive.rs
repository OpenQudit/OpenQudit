use qudit_core::{ParamIndices, Radices};

use crate::Result;
use crate::param::{IntoArgumentList, ParameterVector};
use crate::{OpCode, circuit::InternableOperation, operation::OperationSet};

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq, PartialOrd, Ord)]
#[repr(u64)]
pub enum DirectiveOperation {
    Barrier = 0,
}
impl DirectiveOperation {
    pub fn specialize(self, _args: crate::ArgumentList) -> Result<DirectiveOperation> {
        Ok(self)
    }
}

impl InternableOperation for DirectiveOperation {
    fn intern_operation(
        self,
        operation_set: &mut OperationSet,
        _parameter_vector: &mut ParameterVector,
        _args: impl IntoArgumentList,
        _qudit_radices: Radices,
        _dit_radices: Radices,
    ) -> Result<(OpCode, ParamIndices)> {
        let op_code = operation_set.convert_directive(self);
        match self {
            DirectiveOperation::Barrier => Ok((op_code, ParamIndices::empty())),
        }
    }
}

impl TryFrom<u64> for DirectiveOperation {
    type Error = crate::Error;

    fn try_from(value: u64) -> Result<Self> {
        match value {
            0 => Ok(DirectiveOperation::Barrier),
            _ => Err(crate::Error::GenericError(String::from(
                "Invalid directive operation discriminant provided.",
            ))),
        }
    }
}

impl TryFrom<usize> for DirectiveOperation {
    type Error = crate::Error;

    fn try_from(value: usize) -> Result<Self> {
        Self::try_from(value as u64)
    }
}
