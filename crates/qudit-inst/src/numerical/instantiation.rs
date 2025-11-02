use std::sync::Arc;

use qudit_circuit::QuditCircuit;
use qudit_core::ComplexScalar;
use qudit_core::RealScalar;

use super::Runner;
use super::Problem;
use crate::DataMap;
use crate::Instantiater;
use crate::InstantiationResult;
use crate::InstantiationTarget;

pub trait InstantiationProblem<R: RealScalar>: Problem {
    fn from_instantiation(
        circuit: Arc<QuditCircuit>,
        target: Arc<InstantiationTarget<R::C>>,
        data: Arc<DataMap>,
    ) -> Self;
}

#[derive(Clone)]
pub struct MinimizingInstantiater<R, P> {
    runner: R,
    _phantom: std::marker::PhantomData<P>,
}

impl<R, P> MinimizingInstantiater<R, P> {
    pub fn new(runner: R) -> Self {
        Self {
            runner,
            _phantom: std::marker::PhantomData::<P>,
        }
    }
}

impl<C, R, P> Instantiater<C> for MinimizingInstantiater<R, P>
where
    C: ComplexScalar,
    P: InstantiationProblem<C::R>,
    R: Runner<C::R, P>,
{
    fn instantiate(
        &self,
        circuit: Arc<QuditCircuit>,
        target: Arc<InstantiationTarget<C>>,
        data: Arc<DataMap>,
    ) -> InstantiationResult<C> {
        let problem = P::from_instantiation(circuit, target, data);
        self.runner.run(problem).to_instantiation()
    }
}

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
//

#[cfg(feature = "python")]
mod python {
    use crate::{instantiater::python::{BoxedInstantiater, InstantiaterWrapper}, numerical::{functions::HSProblem, initializers::Uniform, minimizers::LM, runners::MultiStartRunner}, python::PyInstantiationRegistrar};

    use super::*;
    use pyo3::prelude::*;

    impl InstantiaterWrapper for MinimizingInstantiater<MultiStartRunner<LM<f64>, Uniform<f64>>, HSProblem<f64>> {}
    
    #[pyfunction]
    fn DefaultInstantiater() -> BoxedInstantiater {
        let minimizer = LM::default();
        let initializer = Uniform::default();
        let runner = MultiStartRunner { minimizer, guess_generator: initializer, num_starts: 16 };
        let instantiater = MinimizingInstantiater::<_, HSProblem<f64>>::new(runner);
        BoxedInstantiater { inner: Box::new(instantiater) }
    }

    /// Registers the Instantiaters with the Python module.
    fn register(parent_module: &Bound<'_, PyModule>) -> PyResult<()> {
        parent_module.add_function(wrap_pyfunction!(DefaultInstantiater, parent_module)?)?;
        Ok(())
    }
    inventory::submit!(PyInstantiationRegistrar { func: register }); 
}
