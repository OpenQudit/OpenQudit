use std::fmt;
use std::cmp::Ordering;

use qudit_core::CompactStorage;

/// An index for a quantum or classical wire in a quantum circuit.
///
/// Wire uses a signed integer representation where:
/// - Positive values (1, 2, 3, ...) represent quantum wires with indices (0, 1, 2, ...)
/// - Negative values (-1, -2, -3, ...) represent classical wires with indices (0, 1, 2, ...)
/// - Zero represents a null/invalid wire
///
/// # Notes
///
/// - The internal representation imposes limits on wire indices:
///     - Maximum quantum wire index: `(isize::MAX - 1) as usize`
///     - Maximum classical wire index: `(-(isize::MIN + 1)) as usize`
///
/// - This is not a persistent identifier, and may change if wires are inserted
/// or removed in the circuit.
///
/// # Examples
///
/// ```rust
/// # use qudit_circuit::Wire;
///
/// // Create quantum wires
/// let q0 = Wire::quantum(0);  // First quantum wire
/// let q1 = Wire::quantum(1);  // Second quantum wire
/// assert!(q0.is_quantum());
/// assert_eq!(q0.index(), 0);
/// assert_eq!(format!("{}", q0), "q0");
///
/// // Create classical wires
/// let c0 = Wire::classical(0);  // First classical wire
/// let c1 = Wire::classical(1);  // Second classical wire
/// assert!(c0.is_classical());
/// assert_eq!(c0.index(), 0);
/// assert_eq!(format!("{}", c0), "c0");
///
/// // Create from raw values
/// let null_wire = Wire::from_raw(0);
/// assert!(null_wire.is_null());
/// assert!(!null_wire.is_valid());
///
/// // Wire ordering: null < quantum(by index) < classical(by index)
/// assert!(null_wire < q0);
/// assert!(q0 < q1);
/// assert!(q1 < c0);
/// assert!(c0 < c1);
/// ```
#[derive(Hash, PartialEq, Eq, Clone, Debug, Copy)]
#[repr(transparent)]
pub struct Wire(isize);

/// Maximum valid index for quantum wires
pub const MAX_QUANTUM_INDEX: usize = (isize::MAX - 1) as usize;

/// Maximum valid index for classical wires
pub const MAX_CLASSICAL_INDEX: usize = isize::MIN.unsigned_abs() - 2;

impl Wire {
    /// Creates a quantum wire with the given index.
    ///
    /// # Arguments
    /// * `idx` - The zero-based index of the quantum wire
    ///
    /// # Returns
    /// A `Wire` representing a quantum wire at the specified index
    ///
    /// # Panics
    /// Panics if the index would cause an overflow in the internal representation
    ///
    /// # Examples
    /// ```rust
    /// # use qudit_circuit::Wire;
    /// let q0 = Wire::quantum(0);  // raw value: 1
    /// let q5 = Wire::quantum(5);  // raw value: 6
    /// assert!(q0.is_quantum());
    /// assert_eq!(q0.index(), 0);
    /// assert_eq!(q5.index(), 5);
    /// ```
    #[inline]
    pub fn quantum(idx: usize) -> Self {
        if idx > MAX_QUANTUM_INDEX {
            panic!("Quantum wire overflow.");
        }
        Wire((idx + 1) as isize)
    }

    /// Creates a classical wire with the given index.
    ///
    /// # Arguments
    /// * `idx` - The zero-based index of the classical wire
    ///
    /// # Returns
    /// A `Wire` representing a classical wire at the specified index
    ///
    /// # Panics
    /// Panics if the index would cause an overflow in the internal representation
    ///
    /// # Examples
    /// ```rust
    /// # use qudit_circuit::Wire;
    /// let c0 = Wire::classical(0);  // raw value: -1
    /// let c3 = Wire::classical(3);  // raw value: -4
    /// assert!(c0.is_classical());
    /// assert_eq!(c0.index(), 0);
    /// assert_eq!(c3.index(), 3);
    /// ```
    #[inline]
    pub fn classical(idx: usize) -> Self {
        if idx > MAX_CLASSICAL_INDEX {
            panic!("Classical wire overflow.");
        }
        Wire(-((idx + 1) as isize))
    }

    /// Creates a wire from a raw isize value.
    ///
    /// # Arguments
    /// * `val` - Any value that can be converted to `isize`
    ///
    /// # Returns
    /// A `Wire` with the specified raw value
    ///
    /// # Examples
    /// ```rust
    /// # use qudit_circuit::Wire;
    /// let quantum = Wire::from_raw(1);    // Quantum wire at index 0
    /// let classical = Wire::from_raw(-1); // Classical wire at index 0
    /// let null = Wire::from_raw(0);       // Null wire
    /// ```
    #[inline(always)]
    pub const fn from_raw(val: isize) -> Self {
        Wire(val)
    }

    /// Returns `true` if this wire represents a quantum wire.
    ///
    /// Quantum wires have positive internal values.
    ///
    /// # Examples
    /// ```rust
    /// # use qudit_circuit::Wire;
    /// assert!(Wire::quantum(0).is_quantum());
    /// assert!(!Wire::classical(0).is_quantum());
    /// assert!(!Wire::from_raw(0).is_quantum());
    /// ```
    #[inline(always)]
    pub const fn is_quantum(self) -> bool {
        self.0 > 0
    }

    /// Returns `true` if this wire represents a classical wire.
    ///
    /// Classical wires have negative internal values.
    ///
    /// # Examples
    /// ```rust
    /// # use qudit_circuit::Wire;
    /// assert!(Wire::classical(0).is_classical());
    /// assert!(!Wire::quantum(0).is_classical());
    /// assert!(!Wire::from_raw(0).is_classical());
    /// ```
    #[inline(always)]
    pub const fn is_classical(self) -> bool {
        self.0 < 0
    }

    /// Returns `true` if this wire is null/invalid.
    ///
    /// Null wires have an internal value of zero.
    ///
    /// # Examples
    /// ```rust
    /// # use qudit_circuit::Wire;
    /// assert!(Wire::from_raw(0).is_null());
    /// assert!(!Wire::quantum(0).is_null());
    /// assert!(!Wire::classical(0).is_null());
    /// ```
    #[inline(always)]
    pub const fn is_null(self) -> bool {
        self.0 == 0
    }

    /// Returns `true` if this wire is valid (not null).
    ///
    /// Valid wires are either quantum or classical wires.
    ///
    /// # Examples
    /// ```rust
    /// # use qudit_circuit::Wire;
    /// assert!(Wire::quantum(0).is_valid());
    /// assert!(Wire::classical(0).is_valid());
    /// assert!(!Wire::from_raw(0).is_valid());
    /// ```
    #[inline(always)]
    pub const fn is_valid(self) -> bool {
        !self.is_null()
    }

    /// Returns the zero-based index of the wire.
    ///
    /// This converts the internal representation back to the original index
    /// that was passed to `quantum()` or `classical()`.
    ///
    /// # Examples
    /// ```rust
    /// # use qudit_circuit::Wire;
    /// assert_eq!(Wire::quantum(5).index(), 5);
    /// assert_eq!(Wire::classical(3).index(), 3);
    /// assert_eq!(Wire::from_raw(0).index(), usize::MAX); // Null wire wraps around
    /// ```
    #[inline]
    pub const fn index(self) -> usize {
        // Note: null wire (raw value 0) returns usize::MAX due to underflow
        self.0.unsigned_abs().wrapping_sub(1)
    }

    /// Returns the raw internal value of the wire.
    ///
    /// # Examples
    /// ```rust
    /// # use qudit_circuit::Wire;
    /// assert_eq!(Wire::quantum(0).raw_value(), 1);
    /// assert_eq!(Wire::classical(0).raw_value(), -1);
    /// assert_eq!(Wire::from_raw(0).raw_value(), 0);
    /// ```
    #[inline(always)]
    pub const fn raw_value(self) -> isize {
        self.0
    }
}

impl PartialOrd for Wire {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Wire {
    /// Custom ordering: null < quantum(by index) < classical(by index)
    ///
    /// This ordering places wires in the following sequence:
    /// 1. Null wire first
    /// 2. Quantum wires ordered by their index (q0 < q1 < q2 < ...)
    /// 3. Classical wires ordered by their index (c0 < c1 < c2 < ...)
    #[inline]
    fn cmp(&self, other: &Self) -> Ordering {
        match (self.is_null(), other.is_null()) {
            (true, true) => Ordering::Equal,
            (true, false) => Ordering::Less,    // null comes first
            (false, true) => Ordering::Greater, // null comes first
            (false, false) => {
                match (self.is_quantum(), other.is_quantum()) {
                    (true, true) => self.index().cmp(&other.index()), // both quantum: compare by index
                    (false, false) => self.index().cmp(&other.index()), // both classical: compare by index
                    (true, false) => Ordering::Less,    // quantum comes before classical
                    (false, true) => Ordering::Greater, // quantum comes before classical
                }
            }
        }
    }
}

impl From<i32> for Wire {
    /// Creates a wire from an i32 value.
    ///
    /// This is equivalent to `Wire::from_raw(val as isize)`.
    ///
    /// # Arguments
    /// * `val` - Any value that can be converted to `i32`
    ///
    /// # Returns
    /// A `Wire` with the specified raw value
    ///
    /// # Examples
    /// ```rust
    /// # use qudit_circuit::Wire;
    /// let quantum: Wire = 1i32.into();    // Quantum wire at index 0
    /// let classical: Wire = (-1i32).into(); // Classical wire at index 0
    /// let null: Wire = 0i32.into();       // Null wire
    /// ```
    #[inline(always)]
    fn from(val: i32) -> Self {
        Wire::from_raw(val as isize)
    }
}

impl From<isize> for Wire {
    /// Creates a wire from an isize value.
    ///
    /// This is equivalent to `Wire::from_raw(val)`.
    ///
    /// # Examples
    /// ```rust
    /// # use qudit_circuit::Wire;
    /// let quantum: Wire = 1.into();    // Quantum wire at index 0
    /// let classical: Wire = (-1).into(); // Classical wire at index 0
    /// let null: Wire = 0.into();       // Null wire
    /// ```
    #[inline(always)]
    fn from(val: isize) -> Self {
        Wire(val)
    }
}

impl fmt::Display for Wire {
    /// Formats the wire for display.
    ///
    /// # Format
    /// - Quantum wires: `q{index}` (e.g., `q0`, `q5`)
    /// - Classical wires: `c{index}` (e.g., `c0`, `c3`)
    /// - Null wire: `null`
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_quantum() {
            write!(f, "q{}", self.index())
        } else if self.is_classical() {
            write!(f, "c{}", self.index())
        } else {
            write!(f, "null")
        }
    }
}

impl CompactStorage for Wire {
    type InlineType = i8;
    const CONVERSION_INFALLIBLE: bool = false;

    #[inline]
    fn to_inline(value: Self) -> Result<Self::InlineType, Self> {
        let raw = value.raw_value();
        if raw >= i8::MIN as isize && raw <= i8::MAX as isize {
            Ok(raw as i8)
        } else {
            Err(value)
        }
    }

    #[inline(always)]
    fn from_inline(value: Self::InlineType) -> Self { Self::from_raw(value as isize) }

    #[inline(always)]
    fn to_inline_unchecked(value: Self) -> Self::InlineType { value.raw_value() as i8 }
}       

#[cfg(feature = "python")]
mod python {
    use super::Wire;
    use crate::python::PyCircuitRegistrar;
    use pyo3::prelude::*;

    /// Python wrapper for the Wire struct.
    ///
    /// Provides access to quantum circuit wire functionality from Python.
    #[pyclass(name = "Wire", frozen, hash, eq, ord)]
    #[derive(Clone, Debug, Hash, PartialEq, Eq, PartialOrd, Ord)]
    pub struct PyWire {
        inner: Wire,
    }

    #[pymethods]
    impl PyWire {
        /// Create a quantum wire with the given index.
        #[staticmethod]
        fn quantum(idx: usize) -> PyResult<Self> {
            if idx > super::MAX_QUANTUM_INDEX {
                return Err(pyo3::exceptions::PyOverflowError::new_err(
                    "Quantum wire index overflow"
                ));
            }
            Ok(PyWire {
                inner: Wire::quantum(idx),
            })
        }

        /// Create a classical wire with the given index.
        #[staticmethod]
        fn classical(idx: usize) -> PyResult<Self> {
            if idx > super::MAX_CLASSICAL_INDEX {
                return Err(pyo3::exceptions::PyOverflowError::new_err(
                    "Classical wire index overflow"
                ));
            }
            Ok(PyWire {
                inner: Wire::classical(idx),
            })
        }

        /// Create a wire from a raw isize value.
        #[staticmethod]
        fn from_raw(val: isize) -> Self {
            PyWire {
                inner: Wire::from_raw(val),
            }
        }

        /// Check if this wire represents a quantum wire.
        #[getter]
        fn is_quantum(&self) -> bool {
            self.inner.is_quantum()
        }

        /// Check if this wire represents a classical wire.
        #[getter]
        fn is_classical(&self) -> bool {
            self.inner.is_classical()
        }

        /// Check if this wire is null/invalid.
        #[getter]
        fn is_null(&self) -> bool {
            self.inner.is_null()
        }

        /// Check if this wire is valid (not null).
        #[getter]
        fn is_valid(&self) -> bool {
            self.inner.is_valid()
        }

        /// Get the zero-based index of the wire.
        #[getter]
        fn index(&self) -> usize {
            self.inner.index()
        }

        /// Get the raw internal value of the wire.
        #[getter]
        fn raw_value(&self) -> isize {
            self.inner.raw_value()
        }

        /// String representation of the wire.
        fn __str__(&self) -> String {
            format!("{}", self.inner)
        }

        /// Representation string for the wire.
        fn __repr__(&self) -> String {
            if self.inner.is_quantum() {
                format!("Wire.quantum({})", self.inner.index())
            } else if self.inner.is_classical() {
                format!("Wire.classical({})", self.inner.index())
            } else {
                format!("Wire.from_raw({})", self.inner.raw_value())
            }
        }
    }

    impl From<Wire> for PyWire {
        fn from(wire: Wire) -> Self {
            PyWire { inner: wire }
        }
    }

    impl From<PyWire> for Wire {
        fn from(py_wire: PyWire) -> Self {
            py_wire.inner
        }
    }

    impl<'py> IntoPyObject<'py> for Wire {
        type Target = <PyWire as IntoPyObject<'py>>::Target;
        type Output = <PyWire as IntoPyObject<'py>>::Output;
        type Error = <PyWire as IntoPyObject<'py>>::Error;

        fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
            PyWire::from(self).into_pyobject(py)
        }
    }

    impl<'py> FromPyObject<'py> for Wire {
        fn extract_bound(ob: &Bound<'py, PyAny>) -> PyResult<Self> {
            let py_wire: PyWire = ob.extract()?;
            Ok(py_wire.inner)
        }
    }

    /// Registers the Wire class with the Python module.
    fn register(parent_module: &Bound<'_, PyModule>) -> PyResult<()> {
        parent_module.add_class::<PyWire>()?;
        parent_module.add("MAX_QUANTUM_INDEX", super::MAX_QUANTUM_INDEX)?;
        parent_module.add("MAX_CLASSICAL_INDEX", super::MAX_CLASSICAL_INDEX)?;
        Ok(())
    }
    inventory::submit!(PyCircuitRegistrar { func: register });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantum_wire_creation() {
        let q0 = Wire::quantum(0);
        let q1 = Wire::quantum(1);
        let q10 = Wire::quantum(10);

        assert_eq!(q0.raw_value(), 1);
        assert_eq!(q1.raw_value(), 2);
        assert_eq!(q10.raw_value(), 11);
    }

    #[test]
    fn test_classical_wire_creation() {
        let c0 = Wire::classical(0);
        let c1 = Wire::classical(1);
        let c10 = Wire::classical(10);

        assert_eq!(c0.raw_value(), -1);
        assert_eq!(c1.raw_value(), -2);
        assert_eq!(c10.raw_value(), -11);
    }

    #[test]
    fn test_wire_creation_from_raw_values() {
        let positive = Wire::from_raw(5);
        let negative = Wire::from_raw(-3);
        let zero = Wire::from_raw(0);

        assert_eq!(positive.raw_value(), 5);
        assert_eq!(negative.raw_value(), -3);
        assert_eq!(zero.raw_value(), 0);
    }

    #[test]
    fn test_wire_creation_from_isize() {
        let quantum: Wire = 1.into();
        let classical: Wire = (-1).into();
        let null: Wire = 0.into();

        assert!(quantum.is_quantum());
        assert!(classical.is_classical());
        assert!(null.is_null());
    }

    #[test]
    fn test_quantum_wire_identification() {
        let quantum_wire = Wire::quantum(0);
        let classical_wire = Wire::classical(0);
        let null_wire = Wire::from_raw(0);

        assert!(quantum_wire.is_quantum());
        assert!(!classical_wire.is_quantum());
        assert!(!null_wire.is_quantum());
    }

    #[test]
    fn test_classical_wire_identification() {
        let quantum_wire = Wire::quantum(0);
        let classical_wire = Wire::classical(0);
        let null_wire = Wire::from_raw(0);

        assert!(!quantum_wire.is_classical());
        assert!(classical_wire.is_classical());
        assert!(!null_wire.is_classical());
    }

    #[test]
    fn test_null_wire_identification() {
        let quantum_wire = Wire::quantum(0);
        let classical_wire = Wire::classical(0);
        let null_wire = Wire::from_raw(0);

        assert!(!quantum_wire.is_null());
        assert!(!classical_wire.is_null());
        assert!(null_wire.is_null());
    }

    #[test]
    fn test_valid_wire_identification() {
        let quantum_wire = Wire::quantum(0);
        let classical_wire = Wire::classical(0);
        let null_wire = Wire::from_raw(0);

        assert!(quantum_wire.is_valid());
        assert!(classical_wire.is_valid());
        assert!(!null_wire.is_valid());
    }

    #[test]
    fn test_quantum_wire_index_extraction() {
        let q0 = Wire::quantum(0);
        let q5 = Wire::quantum(5);
        let q100 = Wire::quantum(100);

        assert_eq!(q0.index(), 0);
        assert_eq!(q5.index(), 5);
        assert_eq!(q100.index(), 100);
    }

    #[test]
    fn test_classical_wire_index_extraction() {
        let c0 = Wire::classical(0);
        let c5 = Wire::classical(5);
        let c100 = Wire::classical(100);

        assert_eq!(c0.index(), 0);
        assert_eq!(c5.index(), 5);
        assert_eq!(c100.index(), 100);
    }

    #[test]
    fn test_wire_display_formatting() {
        let quantum = Wire::quantum(42);
        let classical = Wire::classical(7);
        let null = Wire::from_raw(0);

        assert_eq!(format!("{}", quantum), "q42");
        assert_eq!(format!("{}", classical), "c7");
        assert_eq!(format!("{}", null), "null");
    }

    #[test]
    fn test_raw_value_retrieval() {
        let quantum = Wire::quantum(42);
        let classical = Wire::classical(42);
        let from_raw = Wire::from_raw(999);

        assert_eq!(quantum.raw_value(), 43);  // 42 + 1
        assert_eq!(classical.raw_value(), -43); // -(42 + 1)
        assert_eq!(from_raw.raw_value(), 999);
    }

    #[test]
    fn test_wire_ordering_and_comparison() {
        let null = Wire::from_raw(0); 
        let q0 = Wire::quantum(0);    
        let q1 = Wire::quantum(1);    
        let q2 = Wire::quantum(2);
        let c0 = Wire::classical(0);  
        let c1 = Wire::classical(1);  
        let c2 = Wire::classical(2);

        // Test custom ordering: null < quantum(by index) < classical(by index)
        assert!(null < q0);
        assert!(q0 < q1);
        assert!(q1 < q2);
        assert!(q2 < c0);
        assert!(c0 < c1);
        assert!(c1 < c2);
        
        // Test quantum ordering
        assert!(q0 < q1);
        assert!(q1 < q2);
        
        // Test classical ordering
        assert!(c0 < c1);
        assert!(c1 < c2);
        
        // Test cross-category ordering
        assert!(null < q0);
        assert!(q1 < c0);
    }

    #[test]
    fn test_wire_equality() {
        let q0_first = Wire::quantum(0);
        let q0_second = Wire::quantum(0);
        let q1 = Wire::quantum(1);

        assert_eq!(q0_first, q0_second);
        assert_ne!(q0_first, q1);
    }

    #[test]
    fn test_wire_cloning_and_copying() {
        let original = Wire::quantum(5);
        let cloned = original.clone();
        let copied = original;

        assert_eq!(original, cloned);
        assert_eq!(original, copied);
    }

    #[test]
    #[should_panic(expected = "Quantum wire overflow")]
    fn test_quantum_wire_overflow_protection() {
        Wire::quantum((std::isize::MAX) as usize);
    }

    #[test]
    #[should_panic(expected = "Classical wire overflow")]
    fn test_classical_wire_overflow_protection() {
        Wire::classical(MAX_CLASSICAL_INDEX + 1);
    }

    #[test]
    fn test_maximum_valid_quantum_wire_creation() {
        let max_idx = MAX_QUANTUM_INDEX;
        let max_wire = Wire::quantum(max_idx);
        
        assert!(max_wire.is_quantum());
        assert_eq!(max_wire.index(), max_idx);
        assert_eq!(max_wire.raw_value(), std::isize::MAX);
    }

    #[test]
    fn test_maximum_valid_classical_wire_creation() {
        let max_idx = MAX_CLASSICAL_INDEX;
        let max_wire = Wire::classical(max_idx);
        
        assert!(max_wire.is_classical());
        assert_eq!(max_wire.index(), max_idx);
        assert_eq!(max_wire.raw_value(), std::isize::MIN + 1);
    }
}

