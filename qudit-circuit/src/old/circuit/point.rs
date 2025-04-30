#[derive(Hash, PartialEq, Eq, Clone, Debug, Copy)]
pub enum DitOrBit {
    Qudit(usize),
    Clbit(usize),
}

#[derive(Hash, PartialEq, Eq, Clone, Debug, Copy)]
pub struct CircuitPoint {
    pub cycle: usize,
    pub dit_or_bit: DitOrBit,
}

#[macro_export]
macro_rules! point {
    ($cycle:expr, $dit_or_bit:expr) => {
        CircuitPoint {
            cycle: $cycle,
            dit_or_bit: DitOrBit::Qudit($dit_or_bit),
        }
    };
    ($cycle:expr; $dit_or_bit:expr) => {
        CircuitPoint {
            cycle: $cycle,
            dit_or_bit: DitOrBit::Clbit($dit_or_bit),
        }
    };
}
