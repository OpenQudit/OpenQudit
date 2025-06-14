use qudit_circuit::QuditCircuit;
use qudit_core::ComplexScalar;
use qudit_core::unitary::UnitaryMatrix;
use qudit_tensor::QVM;

use crate::function::CostFunction;
use crate::function::Hessian;
use crate::function::Jacobian;
use crate::function::Gradient;
use crate::function::Function;
use crate::function::ResidualFunction;

pub enum InstantiationTarget<C: ComplexScalar> {
    // StateVector
    // StateSystem
    // MixedState (Future, but worth thinking about in API)
    // MixedStateSystem (Future, but worth thinking about in API)
    UnitaryMatrix(UnitaryMatrix<C>),
    // Kraus Operators (Future, but worth thinking about in API)
    // SuperOperator? (Future, but worth thinking about in API)
    // CostFunctionGen(Box<dyn CostFunctionGenerator<C>>),
}

pub struct StandardCost<C: ComplexScalar> {
    qvm: QVM<C>
}

impl<C: ComplexScalar> InstantiationTarget<C> {
    pub fn to_function(&self, circuit: &QuditCircuit<C>) -> StandardCost<C> {
        todo!()
    }
}
