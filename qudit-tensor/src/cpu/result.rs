use qudit_core::{memory::MemoryBuffer, ComplexScalar};
use qudit_expr::GenerationShape;
use super::buffer::SizedTensorBuffer;

pub enum TNVMReturnType<'a, C: ComplexScalar> {
    Scalar(&'a C),
    Vector(qudit_core::matrix::RowRef<'a, C>),
    Matrix(qudit_core::matrix::MatRef<'a, C>),
    Tensor3D(qudit_core::array::TensorRef<'a, C, 3>),
    Tensor4D(qudit_core::array::TensorRef<'a, C, 4>),
    SymSqMatrix(qudit_core::array::SymSqTensorRef<'a, C, 2>),
    SymSqTensor3D(qudit_core::array::SymSqTensorRef<'a, C, 3>),
    SymSqTensor4D(qudit_core::array::SymSqTensorRef<'a, C, 4>),
    SymSqTensor5D(qudit_core::array::SymSqTensorRef<'a, C, 5>),
}

impl<'a, C: ComplexScalar> TNVMReturnType<'a, C> {
    pub fn unpack_scalar(self) -> &'a C {
        match self {
            TNVMReturnType::Scalar(s) => s,
            _ => panic!("cannot unpack a non-scalar type as a scalar"),
        }
    }

    pub fn unpack_vector(self) -> qudit_core::matrix::RowRef<'a, C> {
        match self {
            TNVMReturnType::Vector(v) => v,
            _ => panic!("cannot unpack a non-row-vector type as a row-vector"),
        }
    }

    // TODO: mutate the row into a col
    // pub fn unpack_col_vector(self) -> qudit_core::matrix::ColRef<'a, C> {
    //     match self {
    //         TNVMReturnType::ColVector(v) => v,
    //         _ => panic!("cannot unpack a non-col-vector type as a col-vector"),
    //     }
    // }

    pub fn unpack_matrix(self) -> qudit_core::matrix::MatRef<'a, C> {
        match self {
            TNVMReturnType::Matrix(m) => m,
            _ => panic!("cannot unpack a non-matrix type as a matrix"),
        }
    }

    pub fn unpack_tensor3d(self) -> qudit_core::array::TensorRef<'a, C, 3> {
        match self {
            TNVMReturnType::Tensor3D(m) => m,
            _ => panic!("cannot unpack a non-tensor3d type as a tensor3d"),
        }
    }

    pub fn unpack_tensor4d(self) -> qudit_core::array::TensorRef<'a, C, 4> {
        match self {
            TNVMReturnType::Tensor4D(t) => t,
            _ => panic!("cannot unpack a non-tensor4d type as a tensor4d"),
        }
    }

    pub fn unpack_symsq_matrix(self) -> qudit_core::array::SymSqTensorRef<'a, C, 2> {
        match self {
            TNVMReturnType::SymSqMatrix(t) => t,
            _ => panic!("cannot unpack a non-symsq-matrix type as a symsq-matrix"),
        }
    }

    pub fn unpack_symsq_tensor3d(self) -> qudit_core::array::SymSqTensorRef<'a, C, 3> {
        match self {
            TNVMReturnType::SymSqTensor3D(t) => t,
            _ => panic!("cannot unpack a non-symsq-tensor3d type as a symsq-tensor3d"),
        }
    }

    pub fn unpack_symsq_tensor4d(self) -> qudit_core::array::SymSqTensorRef<'a, C, 4> {
        match self {
            TNVMReturnType::SymSqTensor4D(t) => t,
            _ => panic!("cannot unpack a non-symsq-tensor4d type as a symsq-tensor4d"),
        }
    }

    pub fn unpack_symsq_tensor5d(self) -> qudit_core::array::SymSqTensorRef<'a, C, 5> {
        match self {
            TNVMReturnType::SymSqTensor5D(t) => t,
            _ => panic!("cannot unpack a non-symsq-tensor5d type as a symsq-tensor5d"),
        }
    }

    // TODO: Decide, do I want unpack_symsq_tensor3D or do I want unpack_tensor3d to
    // un-symsq it? ... or both, why not?
}

pub struct TNVMResult<'a, C: ComplexScalar> {
    memory: &'a MemoryBuffer<C>,
    buffer: &'a SizedTensorBuffer<C>,
}

impl<'a, C: ComplexScalar> TNVMResult<'a, C> {
    pub fn new(memory: &'a MemoryBuffer<C>, buffer: &'a SizedTensorBuffer<C>) -> Self {
        TNVMResult { memory, buffer }
    }

    pub fn get_fn_result(&self) -> TNVMReturnType<'a, C> {
        match self.buffer.shape() {
            // Safety: TNVM told me this output buffer is mine
            GenerationShape::Scalar => {
                TNVMReturnType::Scalar(unsafe { self.buffer.as_scalar_ref(self.memory) })
            }
            GenerationShape::Vector(_) => {
                TNVMReturnType::Vector(unsafe { self.buffer.as_vector_ref(self.memory) })
            }
            GenerationShape::Matrix(_, _) => {
                TNVMReturnType::Matrix(unsafe { self.buffer.as_matrix_ref(self.memory) })
            }
            GenerationShape::Tensor3D(_, _, _) => {
                TNVMReturnType::Tensor3D(unsafe { self.buffer.as_tensor3d_ref(self.memory) })
            }
            _ => panic!("No Tensor4D should be exposed a function value output."),
        }
    }

    // TODO: this needs to be made more safe by gating it behind const DifferentiationLevel generic
    // impls
    pub fn get_grad_result(&self) -> TNVMReturnType<'a, C> {
        match self.buffer.shape() {
            // Safety: TNVM told me this output buffer is mine
            GenerationShape::Scalar => {
                TNVMReturnType::Vector(unsafe { self.buffer.grad_as_vector_ref(self.memory) })
            }
            GenerationShape::Vector(_) => {
                TNVMReturnType::Matrix(unsafe { self.buffer.grad_as_matrix_ref(self.memory) })
            }
            GenerationShape::Matrix(_, _) => {
                TNVMReturnType::Tensor3D(unsafe { self.buffer.grad_as_tensor3d_ref(self.memory) })
            }
            GenerationShape::Tensor3D(_, _, _) => {
                TNVMReturnType::Tensor4D(unsafe { self.buffer.grad_as_tensor4d_ref(self.memory) })
            }
            _ => panic!("No Tensor4D should be exposed a function value output."),
        }
    }

    pub fn get_hess_result(&self) -> TNVMReturnType<'a, C> {
        match self.buffer.shape() {
            // Safety: TNVM told me this output buffer is mine
            GenerationShape::Scalar => {
                TNVMReturnType::SymSqMatrix(unsafe { self.buffer.hess_as_symsq_matrix_ref(self.memory) })
            }
            GenerationShape::Vector(_) => {
                TNVMReturnType::SymSqTensor3D(unsafe { self.buffer.hess_as_symsq_tensor3d_ref(self.memory) })
            }
            GenerationShape::Matrix(_, _) => {
                TNVMReturnType::SymSqTensor4D(unsafe { self.buffer.hess_as_symsq_tensor4d_ref(self.memory) })
            }
            GenerationShape::Tensor3D(_, _, _) => {
                TNVMReturnType::SymSqTensor5D(unsafe { self.buffer.hess_as_symsq_tensor5d_ref(self.memory) })
            }
            _ => panic!("No Tensor4D should be exposed a function value output."),
        }
    }
}

