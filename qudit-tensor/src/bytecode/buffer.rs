use qudit_core::array::Tensor;
use qudit_core::array::TensorRef;
use qudit_core::matrix::MatMut;
use qudit_core::matrix::MatRef;
use qudit_core::matrix::MatVecMut;
use qudit_core::matrix::MatVecRef;
use qudit_core::matrix::SymSqMatMatMut;
use qudit_core::matrix::SymSqMatMatRef;
use qudit_core::memory::MemoryBuffer;
use qudit_core::ComplexScalar;
use qudit_expr::DifferentiationLevel;
use qudit_core::TensorShape;
use qudit_expr::UnitaryExpression;
use qudit_core::QuditSystem;
use qudit_core::HasParams;

// #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
// pub struct MatrixBuffer {
//     pub nrows: usize,
//     pub ncols: usize,
//     pub num_params: usize,
// }

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct TensorBuffer {
    pub shape: TensorShape,
    pub num_params: usize,
}

impl TensorBuffer {
    pub fn nrows(&self) -> usize {
        match self.shape {
            TensorShape::Scalar => 1,
            TensorShape::Vector(len) => len,
            TensorShape::Matrix(a, _) => a,
            TensorShape::Tensor3D(_, a, _) => a,
            _ => panic!("Dynamic tensor shape unsupported"),
        }
    }

    pub fn ncols(&self) -> usize {
        match self.shape {
            TensorShape::Scalar => 1,
            TensorShape::Vector(_) => 1,
            TensorShape::Matrix(_, b) => b,
            TensorShape::Tensor3D(_, _, b) => b,
            _ => panic!("Dynamic tensor shape unsupported"),
        }
    }

    pub fn nmats(&self) -> usize {
        match self.shape {
            TensorShape::Scalar => 1,
            TensorShape::Vector(_) => 1,
            TensorShape::Matrix(_, _) => 1,
            TensorShape::Tensor3D(c, _, _) => c,
            _ => panic!("Dynamic tensor shape unsupported"),
        }
    }

    pub fn specialize<C: ComplexScalar>(&self, offset: usize) -> SizedTensorBuffer {
        let col_stride = qudit_core::memory::calc_col_stride::<C>(self.nrows(), self.ncols());
        let mat_stride = qudit_core::memory::calc_mat_stride::<C>(self.nrows(), self.ncols(), col_stride);
        SizedTensorBuffer {
            offset,
            shape: self.shape.clone(),
            row_stride: 1,
            col_stride,
            mat_stride,
            num_params: self.num_params,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct SizedTensorBuffer {
    pub offset: usize,
    pub shape: TensorShape,
    pub row_stride: usize,
    pub col_stride: usize,
    pub mat_stride: usize,
    pub num_params: usize,
}

impl SizedTensorBuffer {
    pub fn shape(&self) -> TensorShape {
        self.shape.clone()
    }

    pub fn is_matrix(&self) -> bool {
        self.shape.is_matrix()
    }

    pub fn unit_size(&self) -> usize {
        match self.shape {
            TensorShape::Scalar => 1,
            TensorShape::Vector(len) => len,
            TensorShape::Matrix(_, _) => self.mat_stride,
            TensorShape::Tensor3D(c, _, _) => self.mat_stride * c,
            _ => panic!("Dynamic tensor shape unsupported"),
        }
    }

    pub fn size(&self, diff_lvl: DifferentiationLevel) -> usize {
        match diff_lvl {
            DifferentiationLevel::None => self.unit_size(),
            DifferentiationLevel::Gradient => self.unit_size() * (self.num_params + 1),
            DifferentiationLevel::Hessian => {
                self.unit_size() * (1 + self.num_params + (self.num_params * (self.num_params + 1)) / 2)
            }
        }
    }

    pub fn nrows(&self) -> usize {
        match self.shape {
            TensorShape::Scalar => 1,
            TensorShape::Vector(len) => len,
            TensorShape::Matrix(a, _) => a,
            TensorShape::Tensor3D(_, a, _) => a,
            _ => panic!("Dynamic tensor shape unsupported"),
        }
    }

    pub fn ncols(&self) -> usize {
        match self.shape {
            TensorShape::Scalar => 1,
            TensorShape::Vector(_) => 1,
            TensorShape::Matrix(_, b) => b,
            TensorShape::Tensor3D(_, _, b) => b,
            _ => panic!("Dynamic tensor shape unsupported"),
        }
    }

    pub fn nmats(&self) -> usize {
        match self.shape {
            TensorShape::Scalar => 1,
            TensorShape::Vector(_) => 1,
            TensorShape::Matrix(_, _) => 1,
            TensorShape::Tensor3D(c, _, _) => c,
            _ => panic!("Dynamic tensor shape unsupported"),
        }
    }

    pub fn as_matmut<'a, C: ComplexScalar>(
        &self,
        memory: &mut MemoryBuffer<C>,
    ) -> MatMut<'a, C> {
        unsafe {
            qudit_core::matrix::MatMut::from_raw_parts_mut(
                memory.as_mut_ptr().offset(self.offset as isize),
                self.nrows(),
                self.ncols(),
                self.row_stride as isize,
                self.col_stride as isize,
            )
        }
    }

    pub fn as_matref<'a, C: ComplexScalar>(
        &self,
        memory: &MemoryBuffer<C>,
    ) -> MatRef<'a, C> {
        unsafe {
            faer::MatRef::from_raw_parts(
                memory.as_ptr().offset(self.offset as isize),
                self.nrows(),
                self.ncols(),
                self.row_stride as isize,
                self.col_stride as isize,
            )
        }
    }

    pub fn as_matvecmut<'a, C: ComplexScalar>(
        &self,
        memory: &mut MemoryBuffer<C>,
    ) -> MatVecMut<'a, C> {
        let mat_size = self.col_stride * self.ncols();
        unsafe {
            MatVecMut::from_raw_parts(
                memory.as_mut_ptr().offset((self.offset + mat_size) as isize),
                self.nrows(),
                self.ncols(),
                self.num_params,
                self.col_stride,
                self.mat_stride,
            )
        }
    }

    pub fn as_matvecref_for_write<'a, C: ComplexScalar>(
        &self,
        memory: &mut MemoryBuffer<C>,
    ) -> MatVecRef<'a, C> {
        unsafe {
            MatVecRef::from_raw_parts(
                memory.as_ptr().offset((self.offset) as isize),
                self.nrows(),
                self.ncols(),
                self.num_params + 1,
                self.col_stride,
                self.mat_stride,
            )
        }
    }

    pub fn as_matvecmut_for_write<'a, C: ComplexScalar>(
        &self,
        memory: &mut MemoryBuffer<C>,
    ) -> MatVecMut<'a, C> {
        let mat_size = self.col_stride * self.ncols();
        unsafe {
            MatVecMut::from_raw_parts(
                memory.as_mut_ptr().offset((self.offset) as isize),
                self.nrows(),
                self.ncols(),
                self.num_params + 1,
                self.col_stride,
                self.mat_stride,
            )
        }
    }

    pub fn as_matvecref_non_gradient<'a, C: ComplexScalar>(
        &self,
        memory: &MemoryBuffer<C>,
    ) -> MatVecRef<'a, C> {
        unsafe {
            MatVecRef::from_raw_parts(
                memory.as_ptr().offset((self.offset) as isize),
                self.nrows(),
                self.ncols(),
                self.nmats(),
                self.col_stride,
                self.mat_stride,
            )
        }
    }

    pub fn as_matvecref<'a, C: ComplexScalar>(
        &self,
        memory: &MemoryBuffer<C>,
    ) -> MatVecRef<'a, C> {
        unsafe {
            MatVecRef::from_raw_parts(
                memory.as_ptr().offset((self.offset + self.mat_stride) as isize),
                self.nrows(),
                self.ncols(),
                self.num_params,
                self.col_stride,
                self.mat_stride,
            )
        }
    }

    pub fn as_tensor4d<'a, C: ComplexScalar>(&self, memory: &MemoryBuffer<C>) -> TensorRef<'a, C, 4> {
        let matvec_size = self.mat_stride * self.nmats();
        unsafe {
            TensorRef::from_raw_parts(
                memory.as_ptr().offset((self.offset + matvec_size) as isize),
                [self.num_params, self.nmats(), self.nrows(), self.ncols()],
                [matvec_size, self.mat_stride, 1, self.col_stride],
            )
        }
    }

    pub fn as_symsqmatmut<'a, C: ComplexScalar>(
        &self,
        memory: &mut MemoryBuffer<C>,
    ) -> SymSqMatMatMut<'a, C> {
        let mat_size = self.col_stride * self.ncols();
        let grad_size = mat_size * self.num_params;
        unsafe {
            SymSqMatMatMut::from_raw_parts(
                memory.as_mut_ptr().offset((self.offset + mat_size + grad_size) as isize),
                self.nrows(),
                self.ncols(),
                self.num_params,
                self.col_stride,
                self.mat_stride,
            )
        }
    }

    pub fn as_symsqmatref<'a, C: ComplexScalar>(
        &self,
        memory: &MemoryBuffer<C>,
    ) -> SymSqMatMatRef<'a, C> {
        let mat_size = self.col_stride * self.ncols();
        let grad_size = mat_size * self.num_params;
        unsafe {
            SymSqMatMatRef::from_raw_parts(
                memory.as_ptr().offset((self.offset + mat_size + grad_size) as isize),
                self.nrows(),
                self.ncols(),
                self.num_params,
                self.col_stride,
                self.mat_stride,
            )
        }
    }
}

// impl MatrixBuffer {
//     pub fn size(&self) -> usize {
//         self.nrows * self.ncols * self.num_params
//     }
// }

// impl From<UnitaryExpression> for MatrixBuffer {
//     fn from(expr: UnitaryExpression) -> Self {
//         Self {
//             nrows: expr.dimension(),
//             ncols: expr.dimension(),
//             num_params: expr.num_params(),
//         }
//     }
// }

// impl From<&UnitaryExpression> for MatrixBuffer {
//     fn from(expr: &UnitaryExpression) -> Self {
//         Self {
//             nrows: expr.dimension(),
//             ncols: expr.dimension(),
//             num_params: expr.num_params(),
//         }
//     }
// }

// #[derive(Clone, Debug)]
// pub struct SizedMatrixBuffer {
//     pub offset: usize,
//     pub nrows: usize,
//     pub ncols: usize,
//     pub col_stride: isize,
//     pub mat_stride: isize,
//     pub num_params: usize,
// }

// impl SizedMatrixBuffer {
//     pub fn as_matmut<'a, C: ComplexScalar>(
//         &self,
//         memory: &mut MemoryBuffer<C>,
//     ) -> MatMut<'a, C> {
//         unsafe {
//             faer::MatMut::from_raw_parts_mut(
//                 memory.as_mut_ptr().offset(self.offset as isize),
//                 self.nrows,
//                 self.ncols,
//                 1,
//                 self.col_stride.try_into().unwrap(),
//             )
//         }
//     }

//     pub fn as_matref<'a, C: ComplexScalar>(
//         &self,
//         memory: &MemoryBuffer<C>,
//     ) -> MatRef<'a, C> {
//         unsafe {
//             faer::MatRef::from_raw_parts(
//                 memory.as_ptr().offset(self.offset as isize),
//                 self.nrows,
//                 self.ncols,
//                 1,
//                 self.col_stride,
//             )
//         }
//     }

//     pub fn as_matvecmut<'a, C: ComplexScalar>(
//         &self,
//         memory: &mut MemoryBuffer<C>,
//     ) -> MatVecMut<'a, C> {
//         let mat_size = self.col_stride * self.ncols as isize;
//         unsafe {
//             MatVecMut::from_raw_parts(
//                 memory.as_mut_ptr().offset(self.offset as isize + mat_size),
//                 self.nrows,
//                 self.ncols,
//                 self.num_params,
//                 self.col_stride as usize,
//                 self.mat_stride as usize,
//             )
//         }
//     }

//     pub fn as_matvecref<'a, C: ComplexScalar>(
//         &self,
//         memory: &MemoryBuffer<C>,
//     ) -> MatVecRef<'a, C> {
//         let mat_size = self.col_stride * self.ncols as isize;
//         unsafe {
//             MatVecRef::from_raw_parts(
//                 memory.as_ptr().offset(self.offset as isize + mat_size),
//                 self.nrows,
//                 self.ncols,
//                 self.num_params,
//                 self.col_stride as usize,
//                 self.mat_stride as usize,
//             )
//         }
//     }

//     pub fn as_symsqmatmut<'a, C: ComplexScalar>(
//         &self,
//         memory: &mut MemoryBuffer<C>,
//     ) -> SymSqMatMatMut<'a, C> {
//         let mat_size = self.col_stride * self.ncols as isize;
//         let grad_size = mat_size * self.num_params as isize;
//         unsafe {
//             SymSqMatMatMut::from_raw_parts(
//                 memory.as_mut_ptr().offset(self.offset as isize + mat_size + grad_size),
//                 self.nrows,
//                 self.ncols,
//                 self.num_params,
//                 self.col_stride as usize,
//                 self.mat_stride as usize,
//             )
//         }
//     }

//     pub fn as_symsqmatref<'a, C: ComplexScalar>(
//         &self,
//         memory: &MemoryBuffer<C>,
//     ) -> SymSqMatMatRef<'a, C> {
//         let mat_size = self.col_stride * self.ncols as isize;
//         let grad_size = mat_size * self.num_params as isize;
//         unsafe {
//             SymSqMatMatRef::from_raw_parts(
//                 memory.as_ptr().offset(self.offset as isize + mat_size + grad_size),
//                 self.nrows,
//                 self.ncols,
//                 self.num_params,
//                 self.col_stride as usize,
//                 self.mat_stride as usize,
//             )
//         }
//     }
// }
