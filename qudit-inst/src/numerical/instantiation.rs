use qudit_circuit::QuditCircuit;
use qudit_core::RealScalar;

use crate::DataMap;
use crate::Instantiater;
use crate::InstantiationResult;
use crate::InstantiationTarget;

pub trait InstantiationProblem<'a, R: RealScalar> {
    fn from_instantiation(
        circuit: &'a QuditCircuit<R::C>,
        target: &'a InstantiationTarget<R::C>,
        data: &'a DataMap,
    ) -> Self;
}

pub struct MinimizingInstantiater<R, P> {
    pub runner: R,
    _phantom: std::marker::PhantomData<P>,
}

// impl<C, R> Instantiater<C> for MinimizingInstantiater<R, C>
// where
//     C: ComplexScalar,
//     R: for<'b> Runner<C, QuantumProblem<'b, C>, QuantumCostFunction<C, 0>>,
// {
//     fn instantiate(
//         &self,
//         circuit: &qudit_circuit::QuditCircuit<C>,
//         target: &InstantiationTarget<C>,
//         data: &std::collections::HashMap<String, Box<dyn std::any::Any>>,
//     ) -> InstantiationResult<C> {
//         let problem = QuantumProblem { circuit, target };
//         let result = self.runner.run(problem);
//         // 2. Implement the `CostFunction` (and `Gradient`, etc.) traits for that struct.
//         // 3. Pass that function object to the runner.
//         // 4. Convert the `MinimizationResult` to an `InstantiationResult`.
//         // let func = target.to_function(circuit);
//         // self.runner.run(func)
//         todo!()
//     }
// }

// #[cfg(test)]
// mod tests {

//     use qudit_core::c64;
//     use qudit_core::QuditRadices;

//     #[test]
//     fn can_build_minimizing_instantiater() {
//         let runner = MultiStartRunner { minimizer: LBFGS, guess_generator: Zeros::default(), num_starts: 1, _phantom: std::marker::PhantomData::<c64> };
//         let instantiater = MinimizingInstantiater { runner, _phantom: std::marker::PhantomData::<c64> };

//         let circuit = qudit_circuit::QuditCircuit::new(QuditRadices::new(&[2, 2]), 4);
//         let target = InstantiationTarget::UnitaryMatrix(UnitaryMatrix::identity(&[2, 2]));
//         let data = std::collections::HashMap::new();

//         let result = instantiater.instantiate(&circuit, &target, &data);
//     }
// }
