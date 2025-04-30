use crate::QuditRadices;

pub trait QuditSystem {
    fn get_radices(&self) -> QuditRadices;

    fn get_num_qudits(&self) -> usize {
        self.get_radices().get_num_qudits()
    }

    fn get_dimension(&self) -> usize {
        self.get_radices().get_dimension()
    }

    fn is_qubit_only(&self) -> bool {
        self.get_radices().is_qubit_only()
    }

    fn is_qutrit_only(&self) -> bool {
        self.get_radices().is_qutrit_only()
    }

    fn is_qudit_only(&self, radix: usize) -> bool {
        self.get_radices().is_qudit_only(radix)
    }

    fn is_homogenous(&self) -> bool {
        self.get_radices().is_homogenous()
    }
}

pub trait ClassicalSystem {
    fn get_num_clbits(&self) -> usize;
}

pub trait HybridSystem: QuditSystem + ClassicalSystem {
    fn is_pure_quantum(&self) -> bool {
        self.get_num_clbits() == 0
    }

    fn is_pure_classical(&self) -> bool {
        self.get_num_qudits() == 0
    }

    fn is_hybrid(&self) -> bool {
        self.get_num_qudits() != 0 && self.get_num_clbits() != 0
    }
}
