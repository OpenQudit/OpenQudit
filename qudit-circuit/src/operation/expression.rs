use qudit_core::HasParams;
use qudit_core::QuditSystem;
use qudit_expr::NamedExpression;
use qudit_expr::UnitaryExpression;
use qudit_expr::KrausOperatorsExpression;
use qudit_expr::BraSystemExpression;
use qudit_expr::TensorExpression;
use qudit_expr::KetExpression;
use qudit_expr::ExpressionGenerator;
use qudit_expr::UnitarySystemExpression;

#[derive(Clone, Debug, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub enum ExpressionOpKind {
    UnitaryGate,
    KrausOperators,
    TerminatingMeasurement,
    ClassicallyControlledUnitary,
    QuditInitialization,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ExpressionOperation { 
    UnitaryGate(UnitaryExpression), // No batch, input = output
    KrausOperators(KrausOperatorsExpression), // batch, input, output
    TerminatingMeasurement(BraSystemExpression), // batch, input, no output
    ClassicallyControlledUnitary(UnitarySystemExpression), // batch, input, output
    QuditInitialization(KetExpression), // no batch, no input, only output
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
            ExpressionOperation::TerminatingMeasurement(_) => ExpressionOpKind::TerminatingMeasurement,
            ExpressionOperation::ClassicallyControlledUnitary(_) => ExpressionOpKind::ClassicallyControlledUnitary,
            ExpressionOperation::QuditInitialization(_) => ExpressionOpKind::QuditInitialization,
        }
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

impl From<qudit_gates::Gate> for ExpressionOperation {
    fn from(value: qudit_gates::Gate) -> Self {
        ExpressionOperation::UnitaryGate(value.generate_expression())
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

#[cfg(feature = "python")]
mod python {
    use super::*;
    use pyo3::{exceptions::PyTypeError, prelude::*};

    impl FromPyObject<'_> for ExpressionOperation {
        fn extract_bound(ob: &Bound<'_, PyAny>) -> PyResult<Self> {
            if let Ok(expr) = ob.extract::<UnitaryExpression>() {
                Ok(ExpressionOperation::UnitaryGate(expr))
            } else if let Ok(expr) = ob.extract::<KrausOperatorsExpression>() {
                Ok(ExpressionOperation::KrausOperators(expr))
            } else if let Ok(expr) = ob.extract::<BraSystemExpression>() {
                Ok(ExpressionOperation::TerminatingMeasurement(expr))
            } else if let Ok(expr) = ob.extract::<UnitarySystemExpression>() {
                Ok(ExpressionOperation::ClassicallyControlledUnitary(expr))
            } else if let Ok(expr) = ob.extract::<KetExpression>() {
                Ok(ExpressionOperation::QuditInitialization(expr))
            } else {
                return Err(PyTypeError::new_err("Unrecognized operation type."));
            }
        }
    }
}
