//! This module contains the definition of the `CircuitPoint` struct and the `DitOrBit` enum.

/// The `DitOrBit` enum represents a quantum dit or classical bit.
///
/// While specifying locations in a quantum circuit, this represents the
/// y-axis index or the row from the perspective of the circuit diagram.
///
/// # Variants
/// - `Qudit(usize)`: Represents a quantum dit at the given index.
/// - `Clbit(usize)`: Represents a classical bit at the given index.
///
/// # Examples
///
/// ```
/// use qudit_circuit::DitOrBit;
/// let qudit = DitOrBit::Qudit(0);
/// let clbit = DitOrBit::Clbit(0);
/// ```
///
/// # See Also
///
/// - TODO
#[derive(Hash, PartialEq, Eq, Clone, Debug, Copy, PartialOrd, Ord)]
pub enum DitOrBit {

    /// Represents a quantum dit at the given index.
    Qudit(usize),

    /// Represents a classical bit at the given index.
    Clbit(usize),
}

impl std::fmt::Display for DitOrBit {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DitOrBit::Qudit(index) => write!(f, "Q{}", index),
            DitOrBit::Clbit(index) => write!(f, "C{}", index),
        }
    }
}

/// The `CircuitPoint` struct represents a point in a quantum circuit.
///
/// It is defined by the cycle number and the `DitOrBit` index. The cycle
/// number represents the x-axis index or the column from the perspective of
/// the circuit diagram. In many quantum computing frameworks, this is
/// referred to as the time step or moment.
///
/// # Examples
///
/// TODO
#[derive(Hash, PartialEq, Eq, Clone, Debug, Copy)]
pub struct CircuitPoint {

    /// The cycle number or the x-axis index.
    pub cycle: usize,

    /// The `DitOrBit` index or the y-axis index.
    pub dit_or_bit: DitOrBit,
}

/// Shortcut macro for defining a point in the quantum portion of a circuit.
///
/// # Examples
///
/// ```
/// use qudit_circuit::qpoint;
/// use qudit_circuit::DitOrBit;
/// use qudit_circuit::CircuitPoint;
/// let point = qpoint!(0, 0);
/// ```
///
/// # See Also
///
/// - `cpoint!` A point for the classical portion of the circuit.
#[macro_export]
macro_rules! qpoint {
    ($cycle:expr, $dit_or_bit:expr) => {
        CircuitPoint {
            cycle: $cycle,
            dit_or_bit: DitOrBit::Qudit($dit_or_bit),
        }
    };
}

/// Shortcut macro for defining a point in the classical portion of a circuit.
///
/// # Examples
///
/// ```
/// use qudit_circuit::cpoint;
/// use qudit_circuit::DitOrBit;
/// use qudit_circuit::CircuitPoint;
/// let point = cpoint!(0, 0);
/// ```
///
/// # See Also
///
/// - `qpoint!` A point for the quantum portion of the circuit.
#[macro_export]
macro_rules! cpoint {
    ($cycle:expr, $dit_or_bit:expr) => {
        CircuitPoint {
            cycle: $cycle,
            dit_or_bit: DitOrBit::Clbit($dit_or_bit),
        }
    };
}
//
//#[macro_export]
//macro_rules! point {
//    ($cycle:expr, $dit_or_bit:expr) => {
//        CircuitPoint {
//            cycle: $cycle,
//            dit_or_bit: DitOrBit::Qudit($dit_or_bit),
//        }
//    };
//    ($cycle:expr; $dit_or_bit:expr) => {
//        CircuitPoint {
//            cycle: $cycle,
//            dit_or_bit: DitOrBit::Clbit($dit_or_bit),
//        }
//    };
//}
