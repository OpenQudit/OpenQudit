use qudit_core::HasParams;
use qudit_core::ParamIndices;
use qudit_core::QuditSystem;
use qudit_core::Radices;
use qudit_expr::BraSystemExpression;
use qudit_expr::KetExpression;
use qudit_expr::KrausOperatorsExpression;
use qudit_expr::NamedExpression;
use qudit_expr::TensorExpression;
use qudit_expr::UnitaryExpression;
use qudit_expr::UnitarySystemExpression;
use qudit_expr::index::IndexDirection;
use qudit_expr::index::TensorIndex;
use serde::Deserialize;
use serde::Serialize;

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq, PartialOrd, Ord, Deserialize, Serialize)]
pub enum ExpressionOpKind {
    UnitaryGate,
    KrausOperators,
    TerminatingMeasurement,
    ClassicallyControlledUnitary,
    QuditInitialization,
}

impl ExpressionOpKind {
    pub fn cast<T>(self, expr: T) -> crate::Result<ExpressionOperation>
    where
        T: Into<TensorExpression>,
    {
        let expr = expr.into();

        Ok(match self {
            Self::UnitaryGate => {
                ExpressionOperation::UnitaryGate(UnitaryExpression::try_from(expr)?)
            }
            Self::KrausOperators => {
                ExpressionOperation::KrausOperators(KrausOperatorsExpression::try_from(expr)?)
            }
            Self::TerminatingMeasurement => {
                ExpressionOperation::TerminatingMeasurement(BraSystemExpression::try_from(expr)?)
            }
            Self::ClassicallyControlledUnitary => {
                ExpressionOperation::ClassicallyControlledUnitary(
                    UnitarySystemExpression::try_from(expr)?,
                )
            }
            Self::QuditInitialization => {
                ExpressionOperation::QuditInitialization(KetExpression::try_from(expr)?)
            }
        })
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ExpressionOperation {
    UnitaryGate(UnitaryExpression),           // No batch, input = output
    KrausOperators(KrausOperatorsExpression), // batch, input, output
    TerminatingMeasurement(BraSystemExpression), // batch, input, no output
    ClassicallyControlledUnitary(UnitarySystemExpression), // batch, input, output
    QuditInitialization(KetExpression),       // no batch, no input, only output
}

impl ExpressionOperation {
    pub fn num_qudits(&self) -> usize {
        match self {
            ExpressionOperation::UnitaryGate(e) => e.num_qudits(),
            ExpressionOperation::KrausOperators(e) => e.num_qudits(),
            ExpressionOperation::TerminatingMeasurement(e) => e.num_qudits(),
            ExpressionOperation::ClassicallyControlledUnitary(e) => e.num_qudits(),
            ExpressionOperation::QuditInitialization(e) => e.num_qudits(),
        }
    }

    pub fn expr_type(&self) -> ExpressionOpKind {
        match self {
            ExpressionOperation::UnitaryGate(_) => ExpressionOpKind::UnitaryGate,
            ExpressionOperation::KrausOperators(_) => ExpressionOpKind::KrausOperators,
            ExpressionOperation::TerminatingMeasurement(_) => {
                ExpressionOpKind::TerminatingMeasurement
            }
            ExpressionOperation::ClassicallyControlledUnitary(_) => {
                ExpressionOpKind::ClassicallyControlledUnitary
            }
            ExpressionOperation::QuditInitialization(_) => ExpressionOpKind::QuditInitialization,
        }
    }

    pub fn specialize(self, args: crate::ArgumentList) -> Result<ExpressionOperation> {
        let new_variables = args.variables();
        let expressions = args.expressions();

        // Substitute params into self
        let subbed_op = match self {
            ExpressionOperation::UnitaryGate(e) => {
                let e: TensorExpression = e.into();
                let subbed_expr: UnitaryExpression = e
                    .substitute_parameters(&new_variables, &expressions)
                    .try_into()
                    .unwrap();
                ExpressionOperation::UnitaryGate(subbed_expr)
            }
            ExpressionOperation::KrausOperators(e) => {
                let e: TensorExpression = e.into();
                let subbed_expr: KrausOperatorsExpression = e
                    .substitute_parameters(&new_variables, &expressions)
                    .try_into()
                    .unwrap();
                ExpressionOperation::KrausOperators(subbed_expr)
            }
            ExpressionOperation::TerminatingMeasurement(e) => {
                let e: TensorExpression = e.into();
                let subbed_expr: BraSystemExpression = e
                    .substitute_parameters(&new_variables, &expressions)
                    .try_into()
                    .unwrap();
                ExpressionOperation::TerminatingMeasurement(subbed_expr)
            }
            ExpressionOperation::ClassicallyControlledUnitary(e) => {
                let e: TensorExpression = e.into();
                let subbed_expr: UnitarySystemExpression = e
                    .substitute_parameters(&new_variables, &expressions)
                    .try_into()
                    .unwrap();
                ExpressionOperation::ClassicallyControlledUnitary(subbed_expr)
            }
            ExpressionOperation::QuditInitialization(e) => {
                let e: TensorExpression = e.into();
                let subbed_expr: KetExpression = e
                    .substitute_parameters(&new_variables, &expressions)
                    .try_into()
                    .unwrap();
                ExpressionOperation::QuditInitialization(subbed_expr)
            }
        };

        Ok(subbed_op)
    }
}

impl AsRef<NamedExpression> for ExpressionOperation {
    fn as_ref(&self) -> &NamedExpression {
        match self {
            ExpressionOperation::UnitaryGate(e) => e.as_ref(),
            ExpressionOperation::KrausOperators(e) => e.as_ref(),
            ExpressionOperation::TerminatingMeasurement(e) => e.as_ref(),
            ExpressionOperation::ClassicallyControlledUnitary(e) => e.as_ref(),
            ExpressionOperation::QuditInitialization(e) => e.as_ref(),
        }
    }
}

impl From<ExpressionOperation> for TensorExpression {
    fn from(value: ExpressionOperation) -> Self {
        match value {
            ExpressionOperation::UnitaryGate(e) => e.into(),
            ExpressionOperation::KrausOperators(e) => e.into(),
            ExpressionOperation::TerminatingMeasurement(e) => e.into(),
            ExpressionOperation::ClassicallyControlledUnitary(e) => e.into(),
            ExpressionOperation::QuditInitialization(e) => e.into(),
        }
    }
}

impl HasParams for ExpressionOperation {
    fn num_params(&self) -> usize {
        match self {
            ExpressionOperation::UnitaryGate(e) => e.num_params(),
            ExpressionOperation::KrausOperators(e) => e.num_params(),
            ExpressionOperation::TerminatingMeasurement(e) => e.num_params(),
            ExpressionOperation::ClassicallyControlledUnitary(e) => e.num_params(),
            ExpressionOperation::QuditInitialization(e) => e.num_params(),
        }
    }
}

impl From<UnitaryExpression> for ExpressionOperation {
    fn from(value: UnitaryExpression) -> Self {
        ExpressionOperation::UnitaryGate(value)
    }
}

impl From<KrausOperatorsExpression> for ExpressionOperation {
    fn from(value: KrausOperatorsExpression) -> Self {
        ExpressionOperation::KrausOperators(value)
    }
}

impl From<BraSystemExpression> for ExpressionOperation {
    fn from(value: BraSystemExpression) -> Self {
        ExpressionOperation::TerminatingMeasurement(value)
    }
}

impl From<UnitarySystemExpression> for ExpressionOperation {
    fn from(value: UnitarySystemExpression) -> Self {
        ExpressionOperation::ClassicallyControlledUnitary(value)
    }
}

impl From<KetExpression> for ExpressionOperation {
    fn from(value: KetExpression) -> Self {
        ExpressionOperation::QuditInitialization(value)
    }
}

use crate::OpCode;
use crate::Result;
use crate::circuit::InternableOperation;
use crate::operation::OperationSet;
use crate::param::IntoArgumentList;
use crate::param::ParameterVector;

impl<E: Into<ExpressionOperation>> InternableOperation for E {
    fn intern_operation(
        self,
        operation_set: &mut OperationSet,
        parameter_vector: &mut ParameterVector,
        args: impl IntoArgumentList,
        _qudit_radices: Radices,
        dit_radices: Radices,
    ) -> Result<(OpCode, ParamIndices)> {
        let op: ExpressionOperation = self.into();
        let args: crate::param::ArgumentList = args.into_args(op.num_params())?;
        let param_ids = parameter_vector.parse(&args); // persistent ids; not indices

        let subbed_op = op.specialize(args)?;

        let subbed_op = if !dit_radices.is_empty() {
            let expression_type = subbed_op.expr_type();
            let mut tensor_expr: TensorExpression = subbed_op.into();

            // Reindex the expression's batch dimensions to match dits
            let batch = dit_radices
                .iter()
                .map(|r| (IndexDirection::Batch, usize::from(*r)));
            let outs = tensor_expr
                .indices()
                .iter()
                .filter(|idx| idx.direction() == IndexDirection::Output)
                .map(|idx| (idx.direction(), idx.index_size()));
            let ins = tensor_expr
                .indices()
                .iter()
                .filter(|idx| idx.direction() == IndexDirection::Input)
                .map(|idx| (idx.direction(), idx.index_size()));
            let new_indices = batch
                .chain(outs)
                .chain(ins)
                .enumerate()
                .map(|(i, (d, s))| TensorIndex::new(d, i, s))
                .collect();
            tensor_expr.reindex(new_indices);

            expression_type.cast(tensor_expr)?
        } else {
            subbed_op
        };

        let op_code = operation_set.insert_expression(subbed_op)?;
        operation_set.increment(op_code);

        Ok((op_code, param_ids))
    }
}

#[cfg(feature = "python")]
mod python {
    use super::*;
    use pyo3::{exceptions::PyTypeError, prelude::*};

    impl<'a, 'py> FromPyObject<'a, 'py> for ExpressionOperation {
        type Error = PyErr;

        fn extract(obj: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
            if let Ok(expr) = obj.extract::<UnitaryExpression>() {
                Ok(ExpressionOperation::UnitaryGate(expr))
            } else if let Ok(expr) = obj.extract::<KrausOperatorsExpression>() {
                Ok(ExpressionOperation::KrausOperators(expr))
            } else if let Ok(expr) = obj.extract::<BraSystemExpression>() {
                Ok(ExpressionOperation::TerminatingMeasurement(expr))
            } else if let Ok(expr) = obj.extract::<UnitarySystemExpression>() {
                Ok(ExpressionOperation::ClassicallyControlledUnitary(expr))
            } else if let Ok(expr) = obj.extract::<KetExpression>() {
                Ok(ExpressionOperation::QuditInitialization(expr))
            } else {
                Err(PyTypeError::new_err("Unrecognized operation type."))
            }
        }
    }
}
