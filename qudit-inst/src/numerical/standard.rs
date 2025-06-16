use qudit_core::ComplexScalar;
use crate::InstantiationTarget;
use crate::numerical::function::Function;
use crate::numerical::function::CostFunction;
use crate::numerical::function::Gradient;
use crate::numerical::problem::Problem;
use crate::numerical::problem::ProvidesCostFunction;
use crate::numerical::problem::ProvidesGradient;

pub struct QuantumProblem<'a, C: ComplexScalar> {
    pub circuit: &'a qudit_circuit::QuditCircuit<C>,
    pub target: &'a InstantiationTarget<C>,
}

impl<'a, C: ComplexScalar> Problem for QuantumProblem<'a, C> {
    fn num_params(&self) -> usize {
        self.circuit.num_params()
    }
}

impl<'a, C: ComplexScalar> ProvidesCostFunction<C> for QuantumProblem<'a, C> {
    type CostFunction = QuantumCostFunction;
    fn build_cost_function(&self) -> Self::CostFunction {
        // convert circuit and target to network
        // build qvm with network
        // instantiate QCF with nem QVM
        QuantumCostFunction
    }
}

impl<'a, C: ComplexScalar> ProvidesGradient<C> for QuantumProblem<'a, C> {
    type Gradient = QuantumCostFunction;
    fn build_gradient(&self) -> Self::Gradient {
        QuantumCostFunction
    }
}

pub struct QuantumCostFunction;

impl<C: ComplexScalar> Function<C> for QuantumCostFunction {
    fn num_params(&self) -> usize {
        0
    }
}

impl<C: ComplexScalar> CostFunction<C> for QuantumCostFunction {
    fn cost(&self, params: &[<C as ComplexScalar>::R]) -> <C as ComplexScalar>::R {
        todo!()
    }
}

impl<C: ComplexScalar> Gradient<C> for QuantumCostFunction {
    fn gradient(&self, params: &[<C as ComplexScalar>::R]) -> &[<C as ComplexScalar>::R] {
        todo!()
    }
}

