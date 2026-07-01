use super::*;
use qudit_core::Radices;
use qudit_core::{ClassicalSystem, HasParams, HybridSystem, QuditSystem};

impl QuditSystem for QuditCircuit {
    #[inline(always)]
    fn num_qudits(&self) -> usize {
        self.qudit_radices.num_qudits()
    }

    #[inline(always)]
    fn dimension(&self) -> usize {
        self.qudit_radices.dimension()
    }

    fn radices(&self) -> Radices {
        self.qudit_radices.clone()
    }
}

impl HasParams for QuditCircuit {
    #[inline(always)]
    fn num_params(&self) -> usize {
        // self.params.len()
        self.params.num_unassigned()
    }
}

impl ClassicalSystem for QuditCircuit {
    fn radices(&self) -> Radices {
        self.dit_radices.clone()
    }

    #[inline(always)]
    fn num_dits(&self) -> usize {
        self.dit_radices.num_qudits()
    }
}

impl HybridSystem for QuditCircuit {}
