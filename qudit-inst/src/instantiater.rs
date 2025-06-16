use std::any::Any;
use std::collections::HashMap;

use qudit_core::ComplexScalar;
use qudit_circuit::QuditCircuit;

use crate::InstantiationResult;
use crate::InstantiationTarget;

pub trait Instantiater<C: ComplexScalar> {
    fn instantiate(
        &self,
        circuit: &QuditCircuit<C>,
        target: &InstantiationTarget<C>,
        data: &HashMap<String, Box<dyn Any>>,
    ) -> InstantiationResult<C>;


    fn batched_instantiate(
        &self,
        circuit: &QuditCircuit<C>,
        targets: &[&InstantiationTarget<C>],
        data: &HashMap<String, Box<dyn Any>>,
    ) -> Vec<InstantiationResult<C>> {
        targets.iter().map(|t| self.instantiate(circuit, t, data)).collect()
    }
}
    // fn instantiate_in_place(
    //     &self,
    //     circuit: &mut QuditCircuit<C>,
    //     target: &InstantiationTarget<C>,
    //     data: HashMap<String, Box<dyn Any>>,
    // ) -> InstantiationResult<()> {
    //     let result = self.instantiate(circuit, target, data)?;
    //     circuit.update_params(result.unpack());
    //     Ok(())
    // }

// TODO: QfactorInstantiater
// TODO: QfactorSampleInstantiater
// TODO: QfactorGPUInstantiater
// TODO: QfactorSampleGPUInstantiater
// TODO: MinimizingInstantiater<R: MinimizationRunner>
