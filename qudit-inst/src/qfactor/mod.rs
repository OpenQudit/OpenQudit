use qudit_core::ComplexScalar;

use crate::InstantiationResult;
use crate::InstantiationTarget;
use crate::Instantiater;

struct QfactorInstantiater {
}

impl<C: ComplexScalar> Instantiater<C> for QfactorInstantiater {
    fn instantiate(
        &self,
        circuit: &qudit_circuit::QuditCircuit<C>,
        target: &InstantiationTarget<C>,
        data: &std::collections::HashMap<String, Box<dyn std::any::Any>>,
    ) -> InstantiationResult<C> {
        todo!()
    }
}

// TODO: QfactorInstantiater
// TODO: QfactorSampleInstantiater
// TODO: QfactorGPUInstantiater
// TODO: QfactorSampleGPUInstantiater
