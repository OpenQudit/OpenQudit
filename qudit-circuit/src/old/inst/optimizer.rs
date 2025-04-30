use crate::{math::ComplexScalar, QuditCircuit};

use super::cost::BoxedCostFunction;

pub trait StartingPointGenerator<C: ComplexScalar> {
    fn gen_starting_point(&self, circuit: &QuditCircuit<C>) -> Vec<C::Re>;
}

pub trait Optimizer<C: ComplexScalar> {
    fn optimize(
        &self,
        cost: &BoxedCostFunction<C>,
        x0: Vec<C::Re>,
    ) -> Vec<C::Re>;
}
