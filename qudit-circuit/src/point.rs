//! This module contains the definition of the `CircuitPoint` struct and the `DitOrBit` enum.

/// The `CircuitDitId` enum separately identifies quantum from classical dits.
///
/// While specifying locations in a quantum circuit, this represents the
/// y-axis index or the wire/row from the perspective of the circuit diagram.
///
/// # Examples
///
/// ```
/// use qudit_circuit::CircuitDitId;
/// let zeroth_qudit = CircuitDitId::Quantum(0);
/// let fifth_classical_dit = CircuitDitId::Classical(5);
/// ```
///
/// # See Also
///
/// - `[CircuitPoint]` A fully specified position in a quantum circuit.
/// - `[CircuitLocation]` A register of qudits and/or dits.
#[derive(Hash, PartialEq, Eq, Clone, Debug, Copy, PartialOrd, Ord)]
pub enum CircuitDitId {

    /// Identifies a quantum dit at the given index.
    Quantum(usize),

    /// Identifies a classical dit at the given index.
    Classical(usize),
}

impl std::fmt::Display for CircuitDitId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CircuitDitId::Quantum(index) => write!(f, "q[{}]", index),
            CircuitDitId::Classical(index) => write!(f, "c[{}]", index),
        }
    }
}

impl CircuitDitId {
    /// A constructor for a new qudit Id.
    ///
    /// # Arguments
    /// * `id` - The unique integer identifier for the quantum dit.
    pub fn quantum(id: usize) -> Self {
        Self::Quantum(id)
    }

    /// A constructor for a new classical dit Id.
    ///
    /// # Arguments
    /// * `id` - The unique integer identifier for the classical dit.
    pub fn classical(id: usize) -> Self {
        Self::Classical(id)
    }

    /// Returns the raw Id for the dit, regardless of its type.
    pub fn id(&self) -> usize {
        match self {
            CircuitDitId::Quantum(id) => *id,
            CircuitDitId::Classical(id) => *id,
        }
    }

    /// Returns true if the dit is quantum.
    pub fn is_quantum(&self) -> bool {
        matches!(self, CircuitDitId::Quantum(_))
    }

    /// Returns true if the dit is classical.
    pub fn is_classical(&self) -> bool {
        matches!(self, CircuitDitId::Classical(_))
    }
}

impl From<usize> for CircuitDitId {
    /// Always converts non-negative numbers to quantum numbers.
    fn from(value: usize) -> Self {
        CircuitDitId::Quantum(value)
    }
}

impl From<i32> for CircuitDitId {
    /// Converts negative numbers to classical indices and non-negative to quantum.
    ///
    /// To represent the zeroth classical dit, you must use the full constructor.
    fn from(value: i32) -> Self {
        if value < 0 {
            CircuitDitId::Classical(-value as usize)
        } else {
            CircuitDitId::Quantum(value as usize)
        }
    }
}

impl From<isize> for CircuitDitId {
    /// Converts negative numbers to classical indices and non-negative to quantum.
    ///
    /// To represent the zeroth classical dit, you must use the full constructor.
    fn from(value: isize) -> Self {
        if value < 0 {
            CircuitDitId::Classical(-value as usize)
        } else {
            CircuitDitId::Quantum(value as usize)
        }
    }
}

/// The `CircuitPoint` struct represents a point in a quantum circuit.
///
/// It is defined by the cycle number and a `CircuitDitId`. The cycle
/// number represents the x-axis index or the column from the perspective of
/// the circuit diagram. This is the "when" an operation will be executed.
/// In many quantum computing frameworks, this is referred to as the time
/// step or moment.
///
/// # See Also
///
/// - `[CircuitDitId]` An identifier for a qudit or classical dit in a circuit.
#[derive(Hash, PartialEq, Eq, Clone, Debug, Copy, Ord, PartialOrd)]
pub struct CircuitPoint {
    /// The cycle number or the x-axis index.
    pub cycle: usize,

    /// The `DitOrBit` index or the y-axis index.
    pub dit_id: CircuitDitId,
}

impl std::fmt::Display for CircuitPoint {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "({}, {})", self.cycle, self.dit_id)
    }
}

impl CircuitPoint {
    /// Creates a new `CircuitPoint`.
    ///
    /// # Arguments
    /// * `cycle` - The cycle number (x-axis index).
    /// * `dit_id` - The `CircuitDitId` (y-axis index).
    pub fn new<I: Into<CircuitDitId>>(cycle: usize, dit_id: I) -> Self {
        Self { cycle, dit_id: dit_id.into() }
    }

    /// Creates a new `CircuitPoint` for a quantum dit.
    ///
    /// # Arguments
    /// * `cycle` - The cycle number (x-axis index).
    /// * `quantum_id` - The unique integer identifier for the quantum dit.
    pub fn quantum(cycle: usize, quantum_id: usize) -> Self {
        Self {
            cycle,
            dit_id: CircuitDitId::Quantum(quantum_id),
        }
    }

    /// Creates a new `CircuitPoint` for a classical dit.
    ///
    /// # Arguments
    /// * `cycle` - The cycle number (x-axis index).
    /// * `classical_id` - The unique integer identifier for the classical dit.
    pub fn classical(cycle: usize, classical_id: usize) -> Self {
        Self {
            cycle,
            dit_id: CircuitDitId::Classical(classical_id),
        }
    }

    /// Returns true if the `CircuitPoint` refers to a quantum dit.
    pub fn is_quantum(&self) -> bool {
        self.dit_id.is_quantum()
    }

    /// Returns true if the `CircuitPoint` refers to a classical dit.
    pub fn is_classical(&self) -> bool {
        self.dit_id.is_classical()
    }
}

impl From<(usize, usize)> for CircuitPoint {
    /// Converts a tuple `(cycle, quantum_dit_id)` into a `CircuitPoint`
    /// representing a quantum dit.
    fn from(value: (usize, usize)) -> Self {
        CircuitPoint::quantum(value.0, value.1)
    }
}

impl From<(usize, isize)> for CircuitPoint {
    /// Converts a tuple `(cycle, dit_id)` into a `CircuitPoint`.
    ///
    /// The `dit_id` (isize) is converted to `CircuitDitId`
    /// where negative values represent classical dits and non-negative
    /// values represent quantum dits.
    fn from(value: (usize, isize)) -> Self {
        CircuitPoint::new(value.0, value.1)
    }
}


impl From<(usize, i32)> for CircuitPoint {
    /// Converts a tuple `(cycle, dit_id)` into a `CircuitPoint`.
    ///
    /// The `dit_id` (i32) is converted to `CircuitDitId`
    /// where negative values represent classical dits and non-negative
    /// values represent quantum dits.
    fn from(value: (usize, i32)) -> Self {
        CircuitPoint::new(value.0, value.1)
    }
}

impl From<(i32, i32)> for CircuitPoint {
    /// Converts a tuple `(cycle, dit_id)` into a `CircuitPoint`.
    ///
    /// The `dit_id` (i32) is converted to `CircuitDitId`
    /// where negative values represent classical dits and non-negative
    /// values represent quantum dits.
    fn from(value: (i32, i32)) -> Self {
        if value.0 < 0 {
            panic!("Invalid cycle index {}.", value.0);
        }
        CircuitPoint::new(value.0 as usize, value.1)
    }
}
