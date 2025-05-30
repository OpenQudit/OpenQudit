use qudit_core::matrix::{MatMut, MatRef};
use qudit_core::matrix::{SymSqMatMatMut, SymSqMatMatRef};
use qudit_core::matrix::{MatVecMut, MatVecRef};
use qudit_core::ComplexScalar;
use qudit_expr::TensorGenerationShape;
use crate::bytecode::buffer::SizedTensorBuffer;
use qudit_core::memory::MemoryBuffer;

pub struct ReshapeStruct {
    pub input: SizedTensorBuffer,
    pub out: SizedTensorBuffer,
}

impl ReshapeStruct {
    pub fn new(
        input: SizedTensorBuffer,
        out: SizedTensorBuffer,
    ) -> Self {
        Self {
            input,
            out,
        }
    }

    #[inline(always)]
    pub fn evaluate<C: ComplexScalar>(&self, memory: &mut MemoryBuffer<C>) {
        match (self.input.shape(), self.out.shape()) {
            (TensorGenerationShape::Scalar, TensorGenerationShape::Scalar) => {}
            (TensorGenerationShape::Scalar, TensorGenerationShape::Vector(len)) => {} 
            (TensorGenerationShape::Scalar, TensorGenerationShape::Matrix(a, b)) => {} 
            (TensorGenerationShape::Scalar, TensorGenerationShape::Tensor(a, b, c)) => {} 
            
            (TensorGenerationShape::Vector(len), TensorGenerationShape::Scalar) => {}
            (TensorGenerationShape::Vector(len), TensorGenerationShape::Vector(len_b)) => {}
            (TensorGenerationShape::Vector(len), TensorGenerationShape::Matrix(a, b)) => {
                todo!()
            }
            (TensorGenerationShape::Vector(len), TensorGenerationShape::Tensor(a, b, c)) => {
                todo!()
            }

            (TensorGenerationShape::Matrix(a, b), TensorGenerationShape::Scalar) => {}
            (TensorGenerationShape::Matrix(a, b), TensorGenerationShape::Vector(len)) => {
                todo!()
            }
            (TensorGenerationShape::Matrix(a, b), TensorGenerationShape::Matrix(c, d)) => {
                todo!()
            }
            (TensorGenerationShape::Matrix(a, b), TensorGenerationShape::Tensor(c, d, e)) => {
                for row_index in 0..a {
                    for col_index in 0..b {
                        let linear_index = row_index * b + col_index;
                        let out_mat_index = linear_index / (d*e);
                        let out_row_index = (linear_index % (d*e)) / d;
                        let out_col_index = linear_index % d;
                        
                        let in_buffer_index = self.input.offset + self.input.col_stride * col_index + self.input.row_stride * row_index;
                        let out_buffer_index = self.out.offset + self.out.col_stride * out_col_index + self.out.row_stride * out_row_index + self.out.mat_stride * out_mat_index;

                        memory[out_buffer_index] = memory[in_buffer_index];
                    }
                }
            }

            (TensorGenerationShape::Tensor(a, b, c), TensorGenerationShape::Scalar) => {}
            (TensorGenerationShape::Tensor(a, b, c), TensorGenerationShape::Vector(len)) => {
                todo!()
            }
            (TensorGenerationShape::Tensor(a, b, c), TensorGenerationShape::Matrix(d, e)) => {
                todo!()
            }
            (TensorGenerationShape::Tensor(a, b, c), TensorGenerationShape::Tensor(d, e, f)) => {
                todo!()
            }
        }
    }

    #[inline(always)]
    pub fn evaluate_gradient<C: ComplexScalar>(&self, memory: &mut MemoryBuffer<C>) {
        match (self.input.shape(), self.out.shape()) {
            (TensorGenerationShape::Scalar, TensorGenerationShape::Scalar) => {}
            (TensorGenerationShape::Scalar, TensorGenerationShape::Vector(len)) => {} 
            (TensorGenerationShape::Scalar, TensorGenerationShape::Matrix(a, b)) => {} 
            (TensorGenerationShape::Scalar, TensorGenerationShape::Tensor(a, b, c)) => {} 
            
            (TensorGenerationShape::Vector(len), TensorGenerationShape::Scalar) => {}
            (TensorGenerationShape::Vector(len), TensorGenerationShape::Vector(len_b)) => {}
            (TensorGenerationShape::Vector(len), TensorGenerationShape::Matrix(a, b)) => {
                todo!()
            }
            (TensorGenerationShape::Vector(len), TensorGenerationShape::Tensor(a, b, c)) => {
                todo!()
            }

            (TensorGenerationShape::Matrix(a, b), TensorGenerationShape::Scalar) => {}
            (TensorGenerationShape::Matrix(a, b), TensorGenerationShape::Vector(len)) => {
                todo!()
            }
            (TensorGenerationShape::Matrix(a, b), TensorGenerationShape::Matrix(c, d)) => {
                todo!()
            }
            (TensorGenerationShape::Matrix(a, b), TensorGenerationShape::Tensor(c, d, e)) => {
                for row_index in 0..a {
                    for col_index in 0..b {
                        let linear_index = row_index * b + col_index;
                        let out_mat_index = linear_index / (d*e);
                        let out_row_index = (linear_index % (d*e)) / d;
                        let out_col_index = linear_index % d;
                        
                        let in_buffer_index = self.input.offset + self.input.col_stride * col_index + self.input.row_stride * row_index;
                        let out_buffer_index = self.out.offset + self.out.col_stride * out_col_index + self.out.row_stride * out_row_index + self.out.mat_stride * out_mat_index;

                        memory[out_buffer_index] = memory[in_buffer_index];
                    }
                }
                for param_index in 0..self.input.num_params {
                    for row_index in 0..a {
                        for col_index in 0..b {
                            let linear_index = row_index * b + col_index;
                            let out_mat_index = linear_index / (d*e);
                            let out_row_index = (linear_index % (d*e)) / d;
                            let out_col_index = linear_index % d;
                            
                            let in_buffer_index = self.input.offset + self.input.col_stride * col_index + self.input.row_stride * row_index + (param_index + 1) * self.input.unit_size();
                            let out_buffer_index = self.out.offset + self.out.col_stride * out_col_index + self.out.row_stride * out_row_index + self.out.mat_stride * out_mat_index + (param_index + 1) * self.input.unit_size();

                            memory[out_buffer_index] = memory[in_buffer_index];
                        }
                    }
                }
            }

            (TensorGenerationShape::Tensor(a, b, c), TensorGenerationShape::Scalar) => {}
            (TensorGenerationShape::Tensor(a, b, c), TensorGenerationShape::Vector(len)) => {
                todo!()
            }
            (TensorGenerationShape::Tensor(a, b, c), TensorGenerationShape::Matrix(d, e)) => {
                todo!()
            }
            (TensorGenerationShape::Tensor(a, b, c), TensorGenerationShape::Tensor(d, e, f)) => {
                todo!()
            }
        }
    }

    #[inline(always)]
    pub fn evaluate_hessian<C: ComplexScalar>(&self, memory: &mut MemoryBuffer<C>) {
        match (self.input.shape(), self.out.shape()) {
            (TensorGenerationShape::Scalar, TensorGenerationShape::Scalar) => {}
            (TensorGenerationShape::Scalar, TensorGenerationShape::Vector(len)) => {} 
            (TensorGenerationShape::Scalar, TensorGenerationShape::Matrix(a, b)) => {} 
            (TensorGenerationShape::Scalar, TensorGenerationShape::Tensor(a, b, c)) => {} 
            
            (TensorGenerationShape::Vector(len), TensorGenerationShape::Scalar) => {}
            (TensorGenerationShape::Vector(len), TensorGenerationShape::Vector(len_b)) => {}
            (TensorGenerationShape::Vector(len), TensorGenerationShape::Matrix(a, b)) => {
                todo!()
            }
            (TensorGenerationShape::Vector(len), TensorGenerationShape::Tensor(a, b, c)) => {
                todo!()
            }

            (TensorGenerationShape::Matrix(a, b), TensorGenerationShape::Scalar) => {}
            (TensorGenerationShape::Matrix(a, b), TensorGenerationShape::Vector(len)) => {
                todo!()
            }
            (TensorGenerationShape::Matrix(a, b), TensorGenerationShape::Matrix(c, d)) => {
                todo!()
            }
            (TensorGenerationShape::Matrix(a, b), TensorGenerationShape::Tensor(c, d, e)) => {
                todo!()
            }

            (TensorGenerationShape::Tensor(a, b, c), TensorGenerationShape::Scalar) => {}
            (TensorGenerationShape::Tensor(a, b, c), TensorGenerationShape::Vector(len)) => {
                todo!()
            }
            (TensorGenerationShape::Tensor(a, b, c), TensorGenerationShape::Matrix(d, e)) => {
                todo!()
            }
            (TensorGenerationShape::Tensor(a, b, c), TensorGenerationShape::Tensor(d, e, f)) => {
                todo!()
            }
        }
    }


    // #[inline(always)]
    // pub fn execute_unitary<C: ComplexScalar>(&self, memory: &mut MemoryBuffer<C>) {
    //     let input_matref = self.input.as_matref::<C>(memory);
    //     let out_matmut = self.out.as_matmut::<C>(memory);
    //     self.calculate_unitary(input_matref, out_matmut);
    // }

    // #[inline(always)]
    // pub fn execute_unitary_and_gradient<C: ComplexScalar>(
    //     &self,
    //     memory: &mut MemoryBuffer<C>,
    // ) {
    //     let input_matref = self.input.as_matref::<C>(memory);
    //     let input_gradref = self.input.as_matvecref::<C>(memory);
    //     let out_matmut = self.out.as_matmut::<C>(memory);
    //     let out_gradmut = self.out.as_matvecmut::<C>(memory);
    //     self.calculate_unitary(input_matref, out_matmut);
    //     self.calculate_gradient(input_gradref, out_gradmut);
    // }

    // #[inline(always)]
    // pub fn execute_unitary_gradient_and_hessian<C: ComplexScalar>(
    //     &self,
    //     memory: &mut MemoryBuffer<C>,
    // ) {
    //     let input_matref = self.input.as_matref::<C>(memory);
    //     let input_gradref = self.input.as_matvecref::<C>(memory);
    //     let input_hessref = self.input.as_symsqmatref::<C>(memory);
    //     let out_matmut = self.out.as_matmut::<C>(memory);
    //     let out_gradmut = self.out.as_matvecmut::<C>(memory);
    //     let out_hessmut = self.out.as_symsqmatmut::<C>(memory);
    //     self.calculate_unitary(input_matref, out_matmut);
    //     self.calculate_gradient(input_gradref, out_gradmut);
    //     self.calculate_hessian(input_hessref, out_hessmut);
    // }

    // #[inline(always)]
    // pub fn execute_unitary_into<C: ComplexScalar>(
    //     &self,
    //     memory: &mut MemoryBuffer<C>,
    //     out: MatMut<C>,
    // ) {
    //     let input_matref = self.input.as_matref::<C>(memory);
    //     self.calculate_unitary(input_matref, out);
    // }

    // #[inline(always)]
    // pub fn execute_unitary_and_gradient_into<C: ComplexScalar>(
    //     &self,
    //     memory: &mut MemoryBuffer<C>,
    //     out: MatMut<C>,
    //     out_grad: MatVecMut<C>,
    // ) {
    //     let input_matref = self.input.as_matref::<C>(memory);
    //     let input_gradref = self.input.as_matvecref::<C>(memory);
    //     self.calculate_unitary(input_matref, out);
    //     self.calculate_gradient(input_gradref, out_grad);
    // }

    // #[inline(always)]
    // pub fn execute_unitary_gradient_and_hessian_into<C: ComplexScalar>(
    //     &self,
    //     memory: &mut MemoryBuffer<C>,
    //     out: MatMut<C>,
    //     out_grad: MatVecMut<C>,
    //     out_hess: SymSqMatMatMut<C>,
    // ) {
    //     let input_matref = self.input.as_matref::<C>(memory);
    //     let input_gradref = self.input.as_matvecref::<C>(memory);
    //     let input_hessref = self.input.as_symsqmatref::<C>(memory);
    //     self.calculate_unitary(input_matref, out);
    //     self.calculate_gradient(input_gradref, out_grad);
    //     self.calculate_hessian(input_hessref, out_hess);
    // }
}
