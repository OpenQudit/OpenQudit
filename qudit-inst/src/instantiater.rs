use std::any::Any;
use std::collections::HashMap;

use qudit_core::ComplexScalar;
use qudit_circuit::QuditCircuit;

use crate::InstantiationResult;
use crate::InstantiationTarget;

pub type DataMap = HashMap<String, Box<dyn Any>>;

pub trait Instantiater<C: ComplexScalar> {
    fn instantiate(
        &self,
        circuit: &QuditCircuit<C>,
        target: &InstantiationTarget<C>,
        data: &DataMap,
    ) -> InstantiationResult<C>;


    fn batched_instantiate(
        &self,
        circuit: &QuditCircuit<C>,
        targets: &[&InstantiationTarget<C>],
        data: &DataMap,
    ) -> Vec<InstantiationResult<C>> {
        targets.iter().map(|t| self.instantiate(circuit, t, data)).collect()
    }
}

