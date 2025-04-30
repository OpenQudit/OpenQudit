use super::cost::{BoxedCostFunction, CostFunction, CostFunctionGenerator};
use crate::math::Matrix;
use crate::{
    math::{unitary::UnitaryMatrix, ComplexScalar, Function},
    sim::QVM,
    QuditCircuit,
};
use faer_core::{unzipped, zipped};
use num_traits::Float;

pub struct HilbertSchmidtCostFunctionGenerator<C: ComplexScalar> {
    target: UnitaryMatrix<C>,
}

impl<C: ComplexScalar> HilbertSchmidtCostFunctionGenerator<C> {
    pub fn new(target: UnitaryMatrix<C>) -> Self {
        Self { target }
    }
}

impl<C: ComplexScalar> CostFunctionGenerator<C>
    for HilbertSchmidtCostFunctionGenerator<C>
{
    fn gen_cost(&self, circuit: &QuditCircuit<C>) -> BoxedCostFunction<C> {
        // TODO: should gen_cost also take an instantiation target?
        todo!()
    }
}

pub struct HilbertSchmidtCostFunction<C: ComplexScalar> {
    target: UnitaryMatrix<C>,
    qvm: QVM<C>,
    dem: C::Re,

    circuit_unitary_buffer: Matrix<C>,
}

impl<C: ComplexScalar> HilbertSchmidtCostFunction<C> {
    pub fn new(target: UnitaryMatrix<C>) -> Self {
        Self { target }
    }
}

impl<C: ComplexScalar> Function for HilbertSchmidtCostFunction<C> {
    fn get_num_params(&self) -> usize {
        self.qvm.get_num_params()
    }
}

impl<C: ComplexScalar> CostFunction<C> for HilbertSchmidtCostFunction<C> {
    fn get_cost(&self, params: &[C::Re]) -> C::Re {
        self.qvm
            .write_unitary(params, self.circuit_unitary_buffer.as_mut());
        let mut acm = C::zero();
        zipped!(self.circuit_unitary_buffer.as_ref(), self.target.as_ref())
            .for_each(|unzipped!(c, t)| {
                acm += t.read().conj() * c.read();
            });
        (C::real(1.0) - (acm.abs() / self.dem).powi(2)).sqrt()
    }
}
