use crate::math::unitary::UnitaryMatrix;
use crate::math::ComplexScalar;
use crate::QuditCircuit;

use super::cost::{BoxedCostFunction, CostFunctionGenerator};

pub enum InstantiationTarget<C: ComplexScalar> {
    UnitaryMatrix(UnitaryMatrix<C>),
    CostFunctionGen(Box<dyn CostFunctionGenerator<C>>),
}

impl<C: ComplexScalar> InstantiationTarget<C> {
    fn as_cost_fn_gen(&self) -> Box<dyn CostFunctionGenerator<C>> {
        todo!()
    }
}

impl<C: ComplexScalar> CostFunctionGenerator<C> for InstantiationTarget<C> {
    fn gen_cost(&self, circuit: &QuditCircuit<C>) -> BoxedCostFunction<C> {
        match self {
            Self::UnitaryMatrix(matrix) => {
                todo!()
            },
            Self::CostFunctionGen(cost_fn_gen) => cost_fn_gen.gen_cost(circuit),
        }
    }
}
