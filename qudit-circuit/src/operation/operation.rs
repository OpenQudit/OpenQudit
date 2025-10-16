use qudit_core::HasParams;
use qudit_core::QuditSystem;

use crate::operation::directive::DirectiveOperation;
use crate::operation::expression::ExpressionOperation;
use crate::operation::subcircuit::CircuitOperation;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Operation {
    Expression(ExpressionOperation),
    Subcircuit(CircuitOperation),
    Directive(DirectiveOperation),
}

impl Operation {
    pub fn num_qudits(&self) -> Option<usize> {
        match self {
            Operation::Expression(e) => Some(e.num_qudits()),
            Operation::Subcircuit(c) => Some(c.num_qudits()),
            Operation::Directive(d) => None,
        }
    }
}

impl<T: Into<ExpressionOperation>> From<T> for Operation {
    fn from(value: T) -> Self {
        Operation::Expression(value.into())
    }
}

impl HasParams for Operation {
    fn num_params(&self) -> usize {
        match self {
            Operation::Expression(e) => e.num_params(),
            Operation::Subcircuit(c) => c.num_params(),
            Operation::Directive(_) => 0,
        }
    }
}

