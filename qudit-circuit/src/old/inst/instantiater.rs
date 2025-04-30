use std::any::Any;
use std::collections::HashMap;

use super::target::InstantiationTarget;
use crate::math::ComplexScalar;
use crate::QuditCircuit;

pub struct InstantiationResult {
    pub params: Vec<f64>,
    pub data: HashMap<String, Box<dyn Any>>,
}

pub trait Instantiater<C: ComplexScalar> {
    fn instantiate(
        &self,
        circuit: &QuditCircuit<C>,
        target: &InstantiationTarget<C>,
    ) -> InstantiationResult;
}

// pub struct Preconditioner {}

// impl<C: ComplexScalar> Instantiater<C> for Preconditioner {
//     fn check(
//         &self,
//         _circuit: &QuditCircuit<C>,
//         _target: &InstantiationTarget<C>,
//     ) -> Result<(), &'static str> {
//         Ok(())
//     }

//     fn instantiate(
//         &self,
//         circuit: &QuditCircuit<C>,
//         target: &InstantiationTarget<C>,
//     ) -> (C::Re, Vec<C::Re>) {
//         // let _cost = target.gen_cost(circuit);

//         todo!()
//     }
// }
