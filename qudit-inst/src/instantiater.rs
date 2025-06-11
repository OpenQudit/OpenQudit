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
        data: HashMap<String, Box<dyn Any>>,
    ) -> InstantiationResult<Vec<C::R>>;

    fn instantiate_in_place(
        &self,
        circuit: &mut QuditCircuit<C>,
        target: &InstantiationTarget<C>,
        data: HashMap<String, Box<dyn Any>>,
    ) -> InstantiationResult<()> {
        let result = self.instantiate(circuit, target, data)?;
        circuit.update_params(result.unpack());
        Ok(())
    }

    fn batched_instantiate(
        &self,
        circuit: &QuditCircuit<C>,
        targets: &[&InstantiationTarget<C>]
        data: HashMap<String, Box<dyn Any>>,
    ) -> InstantiationResult<Vec<Vec<C::R>>> {
        targets.iter().map(|t| self.instantiate(circuit, t)).collect()
    }
}
