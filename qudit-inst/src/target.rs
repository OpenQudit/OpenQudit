use qudit_core::ComplexScalar;
use qudit_core::unitary::UnitaryMatrix;

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
