use qudit_core::{memory::MemoryBuffer, ComplexScalar};
use super::buffer::SizedTensorBuffer;

pub enum TNVMReturnType<'a, C: ComplexScalar> {
    Scalar(C),
    RowVector(qudit_core::matrix::RowRef<'a, C>),
    ColVector(qudit_core::matrix::ColRef<'a, C>),
    Matrix(qudit_core::matrix::MatRef<'a, C>),
    MatVec(qudit_core::matrix::MatVecRef<'a, C>),
    Tensor4D(qudit_core::array::TensorRef<'a, C, 4>),
}

impl<'a, C: ComplexScalar> TNVMReturnType<'a, C> {
    pub fn unpack_scalar(self) -> C {
        match self {
            TNVMReturnType::Scalar(s) => s,
            _ => panic!("cannot unpack a non-scalar type as a scalar"),
        }
    }

    pub fn unpack_row_vector(self) -> qudit_core::matrix::RowRef<'a, C> {
        match self {
            TNVMReturnType::RowVector(v) => v,
            _ => panic!("cannot unpack a non-row-vector type as a row-vector"),
        }
    }

    pub fn unpack_col_vector(self) -> qudit_core::matrix::ColRef<'a, C> {
        match self {
            TNVMReturnType::ColVector(v) => v,
            _ => panic!("cannot unpack a non-col-vector type as a col-vector"),
        }
    }

    pub fn unpack_matrix(self) -> qudit_core::matrix::MatRef<'a, C> {
        match self {
            TNVMReturnType::Matrix(m) => m,
            _ => panic!("cannot unpack a non-matrix type as a matrix"),
        }
    }

    pub fn unpack_matvec(self) -> qudit_core::matrix::MatVecRef<'a, C> {
        match self {
            TNVMReturnType::MatVec(m) => m,
            _ => panic!("cannot unpack a non-matvec type as a matvec"),
        }
    }

    pub fn unpack_tensor4d(self) -> qudit_core::array::TensorRef<'a, C, 4> {
        match self {
            TNVMReturnType::Tensor4D(t) => t,
            _ => panic!("cannot unpack a non-tensor4d type as a tensor4d"),
        }
    }
}

pub struct TNVMResult<'a, C: ComplexScalar> {
    memory: &'a MemoryBuffer<C>,
    buffer: SizedTensorBuffer<C>,
}

impl<'a, C: ComplexScalar> TNVMResult<'a, C> {
    pub fn get_fn_result(&self) -> TNVMReturnType<'a, C> {
        todo!()
        // match self.buffer.shape() {
        //     GenerationShape::Scalar => {
        //         todo!()
        //     }
        //     GenerationShape::Vector(len) => {
        //         todo!()
        //     }
        //     GenerationShape::Matrix(rows, cols) => {
        //         TNVMReturnType::Matrix(self.buffer.as_matref(self.memory))
        //     }
        //     GenerationShape::Tensor3D(mats, rows, cols) => {
        //         println!("mats: {}, rows: {}, cols: {}", mats, rows, cols);
        //         TNVMReturnType::MatVec(self.buffer.as_matvecref_non_gradient(self.memory))
        //     }
        //     _ => panic!("Dynamic tensor shape unsupported"),
        // }
    }

    pub fn get_grad_result(&self) -> TNVMReturnType<'a, C> {
        todo!()
        // match self.buffer.shape() {
        //     GenerationShape::Scalar => {
        //         todo!()
        //     }
        //     GenerationShape::Vector(_len) => {
        //         todo!()
        //     }
        //     GenerationShape::Matrix(_rows, _cols) => {
        //         TNVMReturnType::MatVec(self.buffer.as_matvecref(self.memory))
        //     }
        //     GenerationShape::Tensor3D(_mats, _rows, _cols) => {
        //         TNVMReturnType::Tensor4D(self.buffer.as_tensor4d(self.memory))
        //         // // For a tensor (MatVec) function, the gradient would be a higher-order tensor
        //         // // This is typically represented as a MatVec where each original matrix's gradient is included.
        //         // TNVMReturnType::MatVec(self.buffer.as_matvecref_gradient(self.memory))
        //     }
        //     _ => panic!("Dynamic tensor shape unsupported"),
        // }
    }

    pub fn get_hess_result(&self) -> TNVMReturnType<'a, C> {
        todo!()
    }
}

