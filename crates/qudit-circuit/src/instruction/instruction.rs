use crate::{OpCode, Wire, WireList};
use qudit_core::{CompactVec, LimitedSizeVec};
use qudit_core::{HasParams, ParamIndices};

/// Compact storage for parameter indices within instruction memory layout.
#[derive(Clone, Copy, Debug)]
pub enum CompactParamIndices {
    /// Up to 3 parameter indices stored as u8 values.
    ///
    /// The first element is the length (0-3), and the array contains
    /// the actual parameter indices. Only the first `length` elements
    /// of the array are valid.
    Array(u8, [u8; 3]),
    /// Contiguous parameter range stored as start and length.
    ///
    /// Represents parameters `start..start+length`. The NonZero enables
    /// null pointer optimization on the enum size.
    Range(u32, std::num::NonZero<u32>),
}

impl CompactParamIndices {
    /// Creates an empty parameter indices representation.
    ///
    /// This is used when an instruction has no parameters, storing
    /// the information in the most compact way possible.
    fn empty() -> Self {
        CompactParamIndices::Array(0, [0u8; 3])
    }

    /// Returns the number of parameter indices.
    fn len(&self) -> usize {
        match self {
            CompactParamIndices::Array(len, _) => *len as usize,
            CompactParamIndices::Range(_, len) => len.get() as usize,
        }
    }
}

impl PartialEq for CompactParamIndices {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (CompactParamIndices::Array(len1, data1), CompactParamIndices::Array(len2, data2)) => {
                len1 == len2 && data1[0..(*len1 as usize)] == data2[0..(*len2 as usize)]
            }
            (
                CompactParamIndices::Range(start1, len1),
                CompactParamIndices::Range(start2, len2),
            ) => start1 == start2 && len1 == len2,

            // Cross-variant comparison requires logical comparison
            (CompactParamIndices::Array(len1, _), CompactParamIndices::Range(_, len2)) => {
                if *len1 as u32 != len2.get() {
                    return false;
                }

                let params1: ParamIndices = (*self).into();
                let params2: ParamIndices = (*other).into();
                params1 == params2
            }
            (CompactParamIndices::Range(_, len1), CompactParamIndices::Array(len2, _)) => {
                if len1.get() != *len2 as u32 {
                    return false;
                }

                let params1: ParamIndices = (*self).into();
                let params2: ParamIndices = (*other).into();
                params1 == params2
            }
        }
    }
}

impl Eq for CompactParamIndices {}

impl std::hash::Hash for CompactParamIndices {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        match self {
            CompactParamIndices::Range(start, len) => {
                for i in *start..(*start + len.get()) {
                    i.hash(state);
                }
            }
            CompactParamIndices::Array(len, data) => {
                for i in 0..*len as usize {
                    (data[i] as u32).hash(state)
                }
            }
        }
    }
}

impl From<CompactParamIndices> for ParamIndices {
    /// Converts compact parameter storage back to full ParamIndices representation.
    ///
    /// This conversion is used when extracting parameter information from
    /// stored instructions. The compact format saves memory but needs to be
    /// expanded for use with the rest of the system.
    #[inline]
    fn from(value: CompactParamIndices) -> Self {
        match value {
            CompactParamIndices::Array(length, data) => {
                if length == 0 {
                    ParamIndices::Joint(0, 0)
                } else {
                    ParamIndices::Disjoint(
                        data.into_iter()
                            .take(length as usize)
                            .map(|idx| idx as usize)
                            .collect(),
                    )
                }
            }

            CompactParamIndices::Range(start, length) => {
                ParamIndices::Joint(start as usize, length.get() as usize)
            }
        }
    }
}

/// A quantum instruction combining operation, target wires, and parameter references.
///
/// Instructions are stored inline within 32 bytes when possible, automatically falling back
/// to heap allocation for complex cases. Each instruction references parameters by index
/// in the owner's parameter vector rather than storing values directly.
#[derive(Clone, Debug)]
pub enum Instruction {
    /// Inline storage for small instructions (up to 7 wires, limited parameters).
    ///
    /// Fields:
    /// - `OpCode`: The quantum operation to perform
    /// - `[i8; 7]`: Wire data stored directly (classical wires are negative)
    /// - `u8`: Number of valid wires in the array
    /// - `CompactParamIndices`: Parameter indices in compact form
    Inline(OpCode, [i8; 7], u8, CompactParamIndices),

    /// Heap storage for large or complex instructions.
    ///
    /// Fields:
    /// - `OpCode`: The quantum operation to perform
    /// - `LimitedSizeVec<usize>`: Combined wire and parameter data
    /// - `u64`: Split point between wires and parameters in the vector
    Heap(OpCode, LimitedSizeVec<usize>, u64),
}

impl Instruction {
    /// Safely converts a Wire vector to a usize vector for heap storage.
    #[inline]
    fn wire_vec_to_usize_vec(vec: LimitedSizeVec<Wire>) -> LimitedSizeVec<usize> {
        // Safety: Compile-time assertions guarantee Wire and usize have same layout
        unsafe { std::mem::transmute(vec) }
    }

    /// Extracts wire data from heap storage as a Wire slice.
    #[inline]
    fn extract_wires_from_heap(data: &[usize], split: u64) -> &[Wire] {
        let wire_data = data.split_at(split as usize).0;
        // Safety: Wire has same layout as usize, verified by compile-time assertions
        unsafe { std::mem::transmute(wire_data) }
    }

    /// Creates a WireList from inline wire data.
    #[inline]
    fn create_inline_wirelist(wire_data: [i8; 7], wire_len: u8) -> WireList {
        let vec = CompactVec::<Wire>::Inline(wire_data, wire_len);
        // Safety: Instruction maintains valid wire lists
        unsafe { WireList::from_raw_inner(vec) }
    }

    /// Creates a WireList from heap wire data.
    #[inline]
    fn create_heap_wirelist(wire_slice: &[Wire]) -> WireList {
        let vec = CompactVec::from(wire_slice);
        // Safety: Instruction maintains valid wire lists
        unsafe { WireList::from_raw_inner(vec) }
    }

    #[cold]
    fn new_heap(op: OpCode, wire_data: [i8; 7], wire_len: u8, param_indices: ParamIndices) -> Self {
        // Use heap variant - exceeds inline limits
        let mut data = LimitedSizeVec::new();

        // Add wires first (as usize for heap storage)
        for i in 0..wire_len as usize {
            data.push(wire_data[i] as usize);
        }

        let split_point = data.len() as u64;

        // Add parameters after split point
        for param in param_indices.iter() {
            data.push(param);
        }

        Instruction::Heap(op, data, split_point)
    }

    /// Creates a new instruction from operation code, target wires, and parameter indices.
    ///
    /// This constructor automatically chooses between inline and heap storage based on
    /// the size and complexity of the instruction data. Small instructions with few
    /// wires and simple parameter patterns are stored inline for better performance.
    ///
    /// # Arguments
    ///
    /// * `op` - The quantum operation code
    /// * `wires` - Target wires for the operation  
    /// * `param_indices` - Indices referencing parameters (by id) in the owner's parameter vector
    pub fn new(op: OpCode, wires: WireList, param_indices: ParamIndices) -> Self {
        let wires: CompactVec<Wire> = wires.into();

        match wires {
            // If the wires are inlined, then we can almost certainly just move the data
            CompactVec::Inline(wire_data, wire_len) => {
                match param_indices {
                    ParamIndices::Joint(start, length) => {
                        if length == 0 {
                            Instruction::Inline(
                                op,
                                wire_data,
                                wire_len,
                                CompactParamIndices::empty(),
                            )
                        } else if start < u32::MAX as usize && length < u32::MAX as usize {
                            // Safety: length has just been checked to not be zero.
                            let compact_params = CompactParamIndices::Range(start as u32, unsafe {
                                std::num::NonZero::new_unchecked(length as u32)
                            });
                            Instruction::Inline(op, wire_data, wire_len, compact_params)
                        } else {
                            Instruction::new_heap(
                                op,
                                wire_data,
                                wire_len,
                                ParamIndices::Joint(start, length),
                            )
                        }
                    }
                    ParamIndices::Disjoint(vec) => {
                        if vec.is_empty() {
                            Instruction::Inline(
                                op,
                                wire_data,
                                wire_len,
                                CompactParamIndices::empty(),
                            )
                        } else if vec.len() <= 3 {
                            if vec.iter().all(|&idx| idx < u8::MAX as usize) {
                                let mut array_params = [0u8; 3];
                                for (i, &idx) in vec.iter().enumerate() {
                                    array_params[i] = idx as u8;
                                }
                                Instruction::Inline(
                                    op,
                                    wire_data,
                                    wire_len,
                                    CompactParamIndices::Array(vec.len() as u8, array_params),
                                )
                            } else {
                                Instruction::new_heap(
                                    op,
                                    wire_data,
                                    wire_len,
                                    ParamIndices::Disjoint(vec),
                                )
                            }
                        } else {
                            Instruction::new_heap(
                                op,
                                wire_data,
                                wire_len,
                                ParamIndices::Disjoint(vec),
                            )
                        }
                    }
                }
            }

            // The the wires are not inlined, then we need to create a heap instruction and can
            // reuse the wires vector, to avoid unnecessary allocations.
            CompactVec::Heap(vec) => {
                let mut data = Self::wire_vec_to_usize_vec(vec);
                let split = data.len() as u64;
                for param in param_indices.iter() {
                    data.push(param);
                }
                Instruction::Heap(op, data, split)
            }
        }
    }

    /// Returns the operation code for this instruction.
    #[inline]
    pub fn op_code(&self) -> OpCode {
        match self {
            Instruction::Inline(op_code, ..) => *op_code,
            Instruction::Heap(op_code, ..) => *op_code,
        }
    }

    /// Returns the target wires for this instruction.
    #[inline]
    pub fn wires(&self) -> WireList {
        match self {
            Instruction::Inline(_, wire_data, wire_len, _) => {
                Self::create_inline_wirelist(*wire_data, *wire_len)
            }
            Instruction::Heap(_, data, split) => {
                let wire_slice = Self::extract_wires_from_heap(data.as_slice(), *split);
                Self::create_heap_wirelist(wire_slice)
            }
        }
    }

    /// Returns the parameter indices for this instruction.
    ///
    /// Note: these are persistent identifiers that do not correspond to direct
    /// parameter vector positions. One must consult the parameter vector directly
    /// to translate between ids and indices.
    #[inline]
    pub fn params(&self) -> ParamIndices {
        match self {
            Instruction::Inline(_, _, _, params) => (*params).into(),
            Instruction::Heap(_, data, split) => {
                let param_data = data.as_slice().split_at(*split as usize).1;
                ParamIndices::Disjoint(param_data.to_owned())
            }
        }
    }
}

impl HasParams for Instruction {
    fn num_params(&self) -> usize {
        match self {
            Instruction::Inline(.., params) => params.len(),
            Instruction::Heap(_, data, split) => data.len() - *split as usize,
        }
    }
}

impl PartialEq for Instruction {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            // Same-variant comparisons can compare storage directly
            (
                Instruction::Inline(op1, wires1, len1, params1),
                Instruction::Inline(op2, wires2, len2, params2),
            ) => op1 == op2 && len1 == len2 && wires1 == wires2 && params1 == params2,

            (Instruction::Heap(op1, data1, split1), Instruction::Heap(op2, data2, split2)) => {
                op1 == op2 && split1 == split2 && data1 == data2
            }

            // Cross-variant comparison requires logical comparison
            _ => {
                self.op_code() == other.op_code()
                    && self.wires() == other.wires()
                    && self.params() == other.params()
            }
        }
    }
}

impl Eq for Instruction {}

impl std::hash::Hash for Instruction {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        // Hash based on logical content to maintain consistency with PartialEq
        // Since cross-variant instructions can be equal, we must hash their logical representation
        self.op_code().hash(state);
        self.wires().hash(state);
        self.params().hash(state);
    }
}

const _: () = assert!(std::mem::size_of::<Wire>() == std::mem::size_of::<usize>());
const _: () = assert!(std::mem::align_of::<Wire>() == std::mem::align_of::<usize>());

#[cfg(feature = "python")]
mod python {
    use super::*;
    use pyo3::prelude::*;

    /// Python wrapper for quantum instructions.
    ///
    /// This provides a read-only view of instructions from Python, with
    /// access to operation codes, target wires, and parameter indices.
    #[pyclass(name = "Instruction", frozen, eq, hash)]
    #[derive(Clone, Debug, Hash, PartialEq, Eq)]
    pub struct PyInstruction {
        inner: Instruction,
    }

    #[pymethods]
    impl PyInstruction {
        /// Returns the operation code for this instruction.
        #[getter]
        fn op_code(&self) -> OpCode {
            self.inner.op_code()
        }

        /// Returns the target wires for this instruction.
        #[getter]
        fn wires(&self) -> WireList {
            self.inner.wires()
        }

        /// Returns the parameter indices for this instruction.
        #[getter]
        fn params(&self) -> ParamIndices {
            self.inner.params()
        }

        /// Returns the number of parameters for this instruction.
        #[getter]
        fn num_params(&self) -> usize {
            self.inner.num_params()
        }

        fn __repr__(&self) -> String {
            format!(
                "Instruction(op_code={:?}, wires={:?}, num_params={})",
                self.inner.op_code(),
                self.inner.wires(),
                self.inner.num_params()
            )
        }
    }

    impl From<Instruction> for PyInstruction {
        fn from(instruction: Instruction) -> Self {
            PyInstruction { inner: instruction }
        }
    }

    impl From<PyInstruction> for Instruction {
        fn from(py_instruction: PyInstruction) -> Self {
            py_instruction.inner
        }
    }

    impl<'py> IntoPyObject<'py> for Instruction {
        type Target = PyInstruction;
        type Output = Bound<'py, Self::Target>;
        type Error = PyErr;

        fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
            let py_instruction = PyInstruction::from(self);
            Bound::new(py, py_instruction)
        }
    }

    impl<'a, 'py> FromPyObject<'a, 'py> for Instruction {
        type Error = PyErr;

        fn extract(obj: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
            let py_instruction: PyInstruction = obj.extract()?;
            Ok(py_instruction.into())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::operation::OpKind;
    use qudit_core::ParamIndices;
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    #[test]
    fn test_inline_instruction_no_params() {
        let op = OpCode::new(OpKind::Expression, 0);
        let wires = WireList::from_wires(&[Wire::quantum(0), Wire::quantum(1)]);
        let params = ParamIndices::Joint(0, 0);

        let instruction = Instruction::new(op, wires.clone(), params);

        assert_eq!(instruction.op_code(), op);
        assert_eq!(instruction.wires(), wires);
        assert_eq!(instruction.params(), ParamIndices::Joint(0, 0));

        // Should be inline variant
        assert!(matches!(instruction, Instruction::Inline(..)));
    }

    #[test]
    fn test_inline_instruction_joint_params() {
        let op = OpCode::new(OpKind::Expression, 1);
        let wires = WireList::from_wires(&[Wire::quantum(0)]);
        let params = ParamIndices::Joint(5, 2);

        let instruction = Instruction::new(op, wires.clone(), params.clone());

        assert_eq!(instruction.op_code(), op);
        assert_eq!(instruction.wires(), wires);
        assert_eq!(instruction.params(), params);

        assert!(matches!(instruction, Instruction::Inline(..)));
    }

    #[test]
    fn test_inline_instruction_small_disjoint_params() {
        let op = OpCode::new(OpKind::Expression, 2);
        let wires = WireList::from_wires(&[Wire::quantum(0)]);
        let params = ParamIndices::Disjoint(vec![10, 20, 30]);

        let instruction = Instruction::new(op, wires.clone(), params.clone());

        assert_eq!(instruction.op_code(), op);
        assert_eq!(instruction.wires(), wires);
        assert_eq!(instruction.params(), params);

        assert!(matches!(instruction, Instruction::Inline(..)));
    }

    #[test]
    fn test_heap_instruction_large_joint_params() {
        let op = OpCode::new(OpKind::Expression, 1);
        let wires = WireList::from_wires(&[Wire::quantum(0)]);
        // Use parameters that exceed u32::MAX to force heap allocation
        let params = ParamIndices::Joint(std::u32::MAX as usize + 1, 1);

        let instruction = Instruction::new(op, wires.clone(), params.clone());

        assert_eq!(instruction.op_code(), op);
        assert_eq!(instruction.wires(), wires);
        assert_eq!(instruction.params(), params);

        assert!(matches!(instruction, Instruction::Heap(..)));
    }

    #[test]
    fn test_heap_instruction_large_disjoint_params() {
        let op = OpCode::new(OpKind::Expression, 2);
        let wires = WireList::from_wires(&[Wire::quantum(0), Wire::quantum(1)]);
        // Use more than 3 parameters to force heap allocation
        let params = ParamIndices::Disjoint(vec![10, 20, 30, 40, 50]);

        let instruction = Instruction::new(op, wires.clone(), params.clone());

        assert_eq!(instruction.op_code(), op);
        assert_eq!(instruction.wires(), wires);
        assert_eq!(instruction.params(), params);

        assert!(matches!(instruction, Instruction::Heap(..)));
    }

    #[test]
    fn test_heap_instruction_large_param_indices() {
        let op = OpCode::new(OpKind::Expression, 2);
        let wires = WireList::from_wires(&[Wire::quantum(0)]);
        // Use parameter indices that exceed u8::MAX to force heap allocation
        let params = ParamIndices::Disjoint(vec![300, 400]);

        let instruction = Instruction::new(op, wires.clone(), params.clone());

        assert_eq!(instruction.op_code(), op);
        assert_eq!(instruction.wires(), wires);
        assert_eq!(instruction.params(), params);

        assert!(matches!(instruction, Instruction::Heap(..)));
    }

    #[test]
    fn test_heap_instruction_many_wires() {
        let op = OpCode::new(OpKind::Expression, 3);
        // Create enough wires to force heap allocation
        let wire_vec: Vec<Wire> = (0..20).map(Wire::quantum).collect();
        let wires = WireList::from_wires(&wire_vec);
        let params = ParamIndices::Joint(0, 0);

        let instruction = Instruction::new(op, wires.clone(), params.clone());

        assert_eq!(instruction.op_code(), op);
        assert_eq!(instruction.wires(), wires);
        assert_eq!(instruction.params(), params);

        assert!(matches!(instruction, Instruction::Heap(..)));
    }

    #[test]
    fn test_classical_and_quantum_wires() {
        let op = OpCode::new(OpKind::Expression, 4);
        let wires = WireList::from_wires(&[
            Wire::classical(0),
            Wire::quantum(1),
            Wire::classical(2),
            Wire::quantum(3),
        ]);
        let params = ParamIndices::Joint(0, 0);

        let instruction = Instruction::new(op, wires.clone(), params);

        assert_eq!(instruction.op_code(), op);
        assert_eq!(instruction.wires(), wires);
    }

    #[test]
    fn test_empty_disjoint_params() {
        let op = OpCode::new(OpKind::Expression, 0);
        let wires = WireList::from_wires(&[Wire::quantum(0)]);
        let params = ParamIndices::Disjoint(vec![]);

        let instruction = Instruction::new(op, wires.clone(), params);

        assert_eq!(instruction.op_code(), op);
        assert_eq!(instruction.wires(), wires);
        assert_eq!(instruction.params(), ParamIndices::Joint(0, 0));

        assert!(matches!(instruction, Instruction::Inline(..)));
    }

    #[test]
    fn test_compact_param_indices_conversion() {
        // Test Array variant
        let array_params = CompactParamIndices::Array(2, [10, 20, 0]);
        let param_indices: ParamIndices = array_params.into();
        assert_eq!(param_indices, ParamIndices::Disjoint(vec![10, 20]));

        // Test Range variant
        let range_params = CompactParamIndices::Range(5, std::num::NonZero::new(3).unwrap());
        let param_indices: ParamIndices = range_params.into();
        assert_eq!(param_indices, ParamIndices::Joint(5, 3));

        // Test empty
        let empty_params = CompactParamIndices::empty();
        let param_indices: ParamIndices = empty_params.into();
        assert_eq!(param_indices, ParamIndices::Joint(0, 0));
    }

    #[test]
    fn test_compact_param_indices_equality_same_variant() {
        // Array variant equality
        let array1 = CompactParamIndices::Array(2, [10, 20, 0]);
        let array2 = CompactParamIndices::Array(2, [10, 20, 0]);
        let array3 = CompactParamIndices::Array(2, [10, 21, 0]);
        let array4 = CompactParamIndices::Array(1, [10, 20, 0]); // different length

        assert_eq!(array1, array2);
        assert_ne!(array1, array3); // different values
        assert_ne!(array1, array4); // different length

        // Range variant equality
        let range1 = CompactParamIndices::Range(5, std::num::NonZero::new(3).unwrap());
        let range2 = CompactParamIndices::Range(5, std::num::NonZero::new(3).unwrap());
        let range3 = CompactParamIndices::Range(6, std::num::NonZero::new(3).unwrap());
        let range4 = CompactParamIndices::Range(5, std::num::NonZero::new(2).unwrap());

        assert_eq!(range1, range2);
        assert_ne!(range1, range3); // different start
        assert_ne!(range1, range4); // different length
    }

    #[test]
    fn test_compact_param_indices_cross_variant_equality() {
        // Array [5, 6, 7] should equal Range(5, 3)
        let array = CompactParamIndices::Array(3, [5, 6, 7]);
        let range = CompactParamIndices::Range(5, std::num::NonZero::new(3).unwrap());

        assert_eq!(array, range);
        assert_eq!(range, array); // symmetry

        // Array [5, 7, 9] should NOT equal Range(5, 3) [5, 6, 7]
        let array_non_contiguous = CompactParamIndices::Array(3, [5, 7, 9]);
        assert_ne!(array_non_contiguous, range);
        assert_ne!(range, array_non_contiguous);

        // Different lengths should be caught early
        let array_short = CompactParamIndices::Array(2, [5, 6, 0]);
        assert_ne!(array_short, range);
        assert_ne!(range, array_short);
    }

    #[test]
    fn test_compact_param_indices_empty_cases() {
        let empty1 = CompactParamIndices::empty();
        let empty2 = CompactParamIndices::Array(0, [0, 0, 0]);
        let empty3 = CompactParamIndices::Array(0, [1, 2, 3]); // unused data shouldn't matter

        assert_eq!(empty1, empty2);
        assert_eq!(empty1, empty3);
        assert_eq!(empty2, empty3);
    }

    #[test]
    fn test_compact_param_indices_hash_consistency() {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        fn hash_value<T: Hash>(value: &T) -> u64 {
            let mut hasher = DefaultHasher::new();
            value.hash(&mut hasher);
            hasher.finish()
        }

        // Equal values should have equal hashes
        let array1 = CompactParamIndices::Array(2, [10, 20, 0]);
        let array2 = CompactParamIndices::Array(2, [10, 20, 99]); // unused data
        assert_eq!(hash_value(&array1), hash_value(&array2));

        let range1 = CompactParamIndices::Range(5, std::num::NonZero::new(3).unwrap());
        let range2 = CompactParamIndices::Range(5, std::num::NonZero::new(3).unwrap());
        assert_eq!(hash_value(&range1), hash_value(&range2));

        // Cross-variant equal values should have equal hashes
        let array = CompactParamIndices::Array(3, [5, 6, 7]);
        let range = CompactParamIndices::Range(5, std::num::NonZero::new(3).unwrap());
        assert_eq!(array, range); // Verify they're equal first
        assert_eq!(hash_value(&array), hash_value(&range));

        // Different values should (usually) have different hashes
        let array_diff = CompactParamIndices::Array(3, [5, 7, 9]);
        assert_ne!(hash_value(&array), hash_value(&array_diff));
    }

    #[test]
    fn test_instruction_equality_same_variant() {
        let op = OpCode::new(OpKind::Expression, 0);
        let wires = WireList::from_wires(&[Wire::quantum(0), Wire::quantum(1)]);
        let params = ParamIndices::Joint(5, 2);

        let inst1 = Instruction::new(op, wires.clone(), params.clone());
        let inst2 = Instruction::new(op, wires.clone(), params.clone());
        let inst3 = Instruction::new(
            OpCode::new(OpKind::Directive, 0),
            wires.clone(),
            params.clone(),
        );

        assert_eq!(inst1, inst2);
        assert_ne!(inst1, inst3); // different opcode
    }

    #[test]
    fn test_instruction_cross_variant_equality() {
        let op = OpCode::new(OpKind::Expression, 0);

        // Create one instruction that will be inline
        let small_wires = WireList::from_wires(&[Wire::quantum(0)]);
        let small_params = ParamIndices::Joint(0, 0);
        let inline_inst = Instruction::new(op, small_wires.clone(), small_params.clone());

        // Create one instruction that will be heap (many wires)
        let large_wire_vec: Vec<Wire> = (0..20).map(Wire::quantum).collect();
        let large_wires = WireList::from_wires(&large_wire_vec);
        let heap_inst = Instruction::new(op, large_wires, small_params.clone());

        // They should not be equal (different wires)
        assert_ne!(inline_inst, heap_inst);

        // But instructions with same logical content should be equal regardless of storage
        let another_inline = Instruction::new(op, small_wires, small_params);
        assert_eq!(inline_inst, another_inline);

        // Verify storage types as expected
        assert!(matches!(inline_inst, Instruction::Inline(..)));
        assert!(matches!(heap_inst, Instruction::Heap(..)));
    }

    fn hash_value<T: Hash>(value: &T) -> u64 {
        let mut hasher = DefaultHasher::new();
        value.hash(&mut hasher);
        hasher.finish()
    }

    #[test]
    fn test_instruction_hash_consistency() {
        let op = OpCode::new(OpKind::Expression, 0);
        let wires = WireList::from_wires(&[Wire::quantum(0), Wire::quantum(1)]);
        let params = ParamIndices::Joint(5, 2);

        let inst1 = Instruction::new(op, wires.clone(), params.clone());
        let inst2 = Instruction::new(op, wires.clone(), params.clone());

        // Equal instructions should have equal hashes
        assert_eq!(inst1, inst2);
        assert_eq!(hash_value(&inst1), hash_value(&inst2));

        // Different instructions should (usually) have different hashes
        let inst3 = Instruction::new(OpCode::new(OpKind::Directive, 0), wires, params);
        assert_ne!(hash_value(&inst1), hash_value(&inst3));
    }

    #[test]
    fn test_instruction_equality_with_different_param_representations() {
        let op = OpCode::new(OpKind::Expression, 0);
        let wires = WireList::from_wires(&[Wire::quantum(0)]);

        // These should create the same logical parameters but might use different storage
        let joint_params = ParamIndices::Joint(10, 3); // [10, 11, 12]
        let disjoint_params = ParamIndices::Disjoint(vec![10, 11, 12]);

        let inst1 = Instruction::new(op, wires.clone(), joint_params);
        let inst2 = Instruction::new(op, wires, disjoint_params);

        // Should be logically equal
        assert_eq!(inst1, inst2);
        assert_eq!(hash_value(&inst1), hash_value(&inst2));
    }

    #[test]
    fn test_wire_roundtrip() {
        let original_wires = [Wire::quantum(5), Wire::classical(0), Wire::quantum(100)];
        let wire_list = WireList::from_wires(&original_wires);

        let instruction = Instruction::new(
            OpCode::new(OpKind::Expression, 0),
            wire_list.clone(),
            ParamIndices::Joint(0, 0),
        );
        let recovered_wires = instruction.wires();

        assert_eq!(recovered_wires, wire_list);
    }
}
