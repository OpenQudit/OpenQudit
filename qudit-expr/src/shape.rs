use crate::index::{IndexDirection, TensorIndex};
use std::ops::Add;

/// Represents the shape of a tensor as it will be generated.
///
/// While tensors can conceptually have rank larger than four, even infinite,
/// tensors in the OpenQudit Expression library are generated into a buffer
/// indexed by 0, 1, 2, 3, or 4 physical dimensions.
#[derive(Copy, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum GenerationShape {
    /// A 0-dimensional tensor (a single value).
    Scalar,

    /// A 1-dimensional tensor with `nelems` elements.
    Vector(usize),

    /// A 2-dimensional tensor (matrix) with `nrows` rows and `ncols` columns.
    Matrix(usize, usize),

    /// A 3-dimensional tensor with `nmats` matrices, each of `nrows` rows and `ncols` columns.
    Tensor3D(usize, usize, usize),

    /// A 4-dimensional tensor usually for derivatives (ntens, nmats, nrows, ncols)
    Tensor4D(usize, usize, usize, usize)
}

impl std::fmt::Debug for GenerationShape {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GenerationShape::Scalar => write!(f, "Scalar"),
            GenerationShape::Vector(nelems) => write!(f, "Vector({})", nelems),
            GenerationShape::Matrix(nrows, ncols) => write!(f, "Matrix({}, {})", nrows, ncols),
            GenerationShape::Tensor3D(nmats, nrows, ncols) => write!(f, "Tensor3D({}, {}, {})", nmats, nrows, ncols),
            GenerationShape::Tensor4D(ntens, nmats, nrows, ncols) => write!(f, "Tensor4D({}, {}, {}, {})", ntens, nmats, nrows, ncols),
        }
    }
}

impl GenerationShape {
    /// Calculates the total number of elements in a tensor with this shape.
    ///
    /// # Returns
    /// The total number of elements as `usize`.
    pub fn num_elements(&self) -> usize {
        match self {
            GenerationShape::Scalar => 1,
            GenerationShape::Vector(nelems) => *nelems,
            GenerationShape::Matrix(nrows, ncols) => nrows * ncols,
            GenerationShape::Tensor3D(nmats, nrows, ncols) => nmats * nrows * ncols,
            GenerationShape::Tensor4D(ntens, nmats, nrows, ncols) => ntens * nmats * nrows * ncols,
        }
    }

    /// Determines the shape of the derivative of a tensor with respect to `num_params` parameters.
    ///
    /// This method effectively prepends `num_params` to the current tensor's dimensions.
    /// For example, the derivative of a `Scalar` with respect to `num_params` becomes a `Vector(num_params)`.
    /// The derivative of a `Matrix(R, C)` becomes a `Tensor3D(num_params, R, C)`.
    ///
    /// # Arguments
    /// * `num_params` - The number of parameters in the gradient.
    ///
    /// # Returns
    /// A new `GenerationShape` representing the shape of the gradient.
    ///
    /// # See Also
    /// - `[hessian_shape]` For the shape of a hessian tensor.
    pub fn gradient_shape(&self, num_params: usize) -> Self {
        match self {
            GenerationShape::Scalar => GenerationShape::Vector(num_params),
            GenerationShape::Vector(nelems) => GenerationShape::Matrix(num_params, *nelems),
            GenerationShape::Matrix(nrows, ncols) => GenerationShape::Tensor3D(num_params, *nrows, *ncols),
            GenerationShape::Tensor3D(nmats, nrows, ncols) => GenerationShape::Tensor4D(num_params, *nmats, *nrows, *ncols),
            GenerationShape::Tensor4D(ntens, nmats, nrows, ncols) => panic!("Unable to find shape for gradient of 4D tensor."),
        }
    }

    /// Determine the hessian shape of a tensor with this shape that has `num_params` parameters.
    ///
    /// # Arguments
    /// * `num_params` - The number of parameters in the hessian.
    ///
    /// # Returns
    /// A new `GenerationShape` representing the shape of the hessian.
    ///
    /// # See Also
    /// - `[gradient_shape]` For the shape of a gradient tensor.
    pub fn hessian_shape(&self, num_params: usize) -> Self {
        let sym_sq_size = num_params * (num_params - 1) / 2; // TODO: Is it +1 or -1
        match self {
            GenerationShape::Scalar => GenerationShape::Vector(sym_sq_size),
            GenerationShape::Vector(nelems) => GenerationShape::Matrix(sym_sq_size, *nelems),
            GenerationShape::Matrix(nrows, ncols) => GenerationShape::Tensor3D(sym_sq_size, *nrows, *ncols),
            GenerationShape::Tensor3D(nmats, nrows, ncols) => GenerationShape::Tensor4D(sym_sq_size, *nmats, *nrows, *ncols),
            GenerationShape::Tensor4D(ntens, nmats, nrows, ncols) => panic!("Unable to find shape for Hessian of 4D tensor."),
        }
    }

    /// Converts the tensor shape object to a vector of integers.
    /// 
    /// # Returns
    /// 
    /// A `Vec<usize>` containing the dimensions of the shape.
    /// 
    /// # Examples
    /// ```
    /// use qudit_expr::GenerationShape;
    /// 
    /// let scalar_shape = GenerationShape::Scalar;
    /// assert_eq!(scalar_shape.to_vec(), vec![]);
    /// 
    /// let vector_shape = GenerationShape::Vector(5);
    /// assert_eq!(vector_shape.to_vec(), vec![5]);
    /// 
    /// let matrix_shape = GenerationShape::Matrix(2, 3);
    /// assert_eq!(matrix_shape.to_vec(), vec![2, 3]);
    /// ```
    pub fn to_vec(&self) -> Vec<usize> {
        match self {
            GenerationShape::Scalar => vec![],
            GenerationShape::Vector(nelems) => vec![*nelems],
            GenerationShape::Matrix(nrows, ncols) => vec![*nrows, *ncols],
            GenerationShape::Tensor3D(nmats, nrows, ncols) => vec![*nmats, *nrows, *ncols],
            GenerationShape::Tensor4D(ntens, nmats, nrows, ncols) => vec![*ntens, *nmats, *nrows, *ncols],
        }
    }

    /// Checks if the current `GenerationShape` is strictly a scalar variant.
    pub fn is_scalar(&self) -> bool {
        matches!(self, GenerationShape::Scalar)
    }

    /// Checks if the current `GenerationShape` is strictly a vector variant.
    pub fn is_vector(&self) -> bool {
        matches!(self, GenerationShape::Vector(_))
    }

    /// Checks if the current `GenerationShape` is strictly a matrix variant.
    pub fn is_matrix(&self) -> bool {
        matches!(self, GenerationShape::Matrix(_, _))
    }

    /// Checks if the current `GenerationShape` is strictly a tensor3D variant.
    pub fn is_tensor3d(&self) -> bool {
        matches!(self, GenerationShape::Tensor3D(_, _, _))
    }

    /// Checks if the current `GenerationShape` is strictly a tensor4D variant.
    pub fn is_tensor4d(&self) -> bool {
        matches!(self, GenerationShape::Tensor4D(_, _, _, _))
    }

    /// Check if there is only one element.
    /// 
    /// # Returns
    /// `true` if the shape can be treated as a scalar, `false` otherwise.
    /// 
    /// # Examples
    /// ```
    /// use qudit_expr::GenerationShape;
    /// 
    /// let test_scalar = GenerationShape::Scalar;
    /// let test_vector = GenerationShape::Vector(1);
    /// let test_matrix = GenerationShape::Matrix(1, 1);
    /// let test_tensor3d = GenerationShape::Tensor3D(1, 1, 1);
    /// 
    /// let test_vector_2 = GenerationShape::Vector(9);
    /// let test_matrix_2 = GenerationShape::Matrix(9, 9);
    /// let test_tensor3d_2 = GenerationShape::Tensor3D(1, 9, 9);
    ///
    /// assert!(test_scalar.is_0d());
    /// assert!(test_vector.is_0d());
    /// assert!(test_matrix.is_0d());
    /// assert!(test_tensor3d.is_0d());
    /// 
    /// assert_eq!(test_vector_2.is_0d(), false);
    /// assert_eq!(test_matrix_2.is_0d(), false);
    /// assert_eq!(test_tensor3d_2.is_0d(), false);
    /// ```
    pub fn is_0d(&self) -> bool {
        self.num_elements() == 1
    }

    /// Check if the shape can be conceptually treated as a 1d tensor.
    ///
    /// # Returns
    /// `true` if the shape has exactly one dimension with 1 or more elements.
    /// 
    /// # Examples
    /// ```
    /// use qudit_expr::GenerationShape;
    /// 
    /// let test_vector = GenerationShape::Vector(9);
    /// let test_matrix = GenerationShape::Matrix(1, 9);
    /// let test_tensor3d = GenerationShape::Tensor3D(1, 1, 9);
    /// 
    /// let test_scalar = GenerationShape::Scalar;
    /// let test_matrix_2 = GenerationShape::Matrix(9, 9);
    /// let test_tensor3d_2 = GenerationShape::Tensor3D(1, 9, 9);
    ///
    /// assert!(test_vector.is_1d());
    /// assert!(test_matrix.is_1d());
    /// assert!(test_tensor3d.is_1d());
    /// assert!(test_tensor_nd.is_1d());
    /// 
    /// assert_eq!(test_scalar.is_1d(), false);
    /// assert_eq!(test_matrix_2.is_1d(), false);
    /// assert_eq!(test_tensor3d_2.is_1d(), false);
    /// ```
    pub fn is_1d(&self) -> bool {
        match self {
            GenerationShape::Scalar => false,
            GenerationShape::Vector(_) => true,
            GenerationShape::Matrix(nrows, ncols) => *nrows == 1 || *ncols == 1,
            GenerationShape::Tensor3D(nmats, nrows, ncols) => {
                let non_one_count = [*nmats, *nrows, *ncols].iter().filter(|&&d| d > 1).count();
                non_one_count == 1
            }
            GenerationShape::Tensor4D(ntens, nmats, nrows, ncols) => {
                let non_one_count = [*ntens, *nmats, *nrows, *ncols].iter().filter(|&&d| d > 1).count();
                non_one_count == 1
            }
        }
    }

    /// Checks if the current `GenerationShape` can be conceptually treated as a 2-dimensional matrix.
    /// This is true for `GenerationShape` variants with a dimensionality of at least 2, with
    /// any additional dimensions having size 1.
    ///
    /// # Returns
    /// `true` if the shape can be treated as a matrix, `false` otherwise.
    /// 
    /// # Examples
    /// ```
    /// use qudit_expr::GenerationShape;
    /// 
    /// let test_scalar = GenerationShape::Scalar;
    /// let test_vector = GenerationShape::Vector(1);
    /// let test_tensor3d_2 = GenerationShape::Tensor3D(9, 9, 9);
    /// let test_tensor_nd_2 = GenerationShape::TensorND(vec![1, 1, 9, 9, 9]);
    /// 
    /// let test_matrix = GenerationShape::Matrix(9, 9);
    /// let test_tensor3d = GenerationShape::Tensor3D(1, 9, 9);
    /// let test_tensor_nd = GenerationShape::TensorND(vec![1, 1, 1, 9, 9]);
    /// 
    /// assert_eq!(test_scalar.is_2d(), false);
    /// assert_eq!(test_vector.is_2d(), false);
    /// assert_eq!(test_tensor3d_2.is_2d(), false);
    /// assert_eq!(test_tensor_nd_2.is_2d(), false);
    /// 
    /// assert_eq!(test_matrix.is_2d(), true);
    /// assert_eq!(test_tensor3d.is_2d(), true);
    /// assert_eq!(test_tensor_nd.is_2d(), true);
    /// ```
    pub fn is_2d(&self) -> bool {
        match self {
            GenerationShape::Scalar => false,
            GenerationShape::Vector(_) => false,
            GenerationShape::Matrix(_, _) => true,
            // A Tensor3D can be seen as a matrix if it's essentially a stack of column vectors,
            // or perhaps if it represents a single matrix (nmats=1).
            // The current implementation checks if ncols is 1, implying it's a stack of column vectors.
            GenerationShape::Tensor3D(nmats, _, _) => *nmats == 1,
            GenerationShape::Tensor4D(ntens, nmats, _, _) => *ntens == 1 && *nmats == 1,
        }
    }

    /// Checks if the current `GenerationShape` can be conceptually treated as a 3D tensor.
    /// This is true for `GenerationShape` variants with a dimensionality of at least 3, with
    /// any additional dimensions having size 1.
    ///
    /// # Returns
    /// `true` if the shape can be treated as a 3D tensor, `false` otherwise.
    /// 
    /// # Examples
    /// ```
    /// use qudit_expr::GenerationShape;
    /// 
    /// let test_scalar = GenerationShape::Scalar;
    /// let test_vector = GenerationShape::Vector(1);
    /// let test_matrix = GenerationShape::Matrix(1, 1);
    /// let test_tensor_nd_2 = GenerationShape::TensorND(vec![1, 9, 9, 9, 9]);
    /// 
    /// let test_tensor3d = GenerationShape::Tensor3D(9, 9, 9);
    /// let test_tensor_nd = GenerationShape::TensorND(vec![1, 1, 9, 9, 9]);
    ///
    /// assert_eq!(test_scalar.is_3d(), false);
    /// assert_eq!(test_vector.is_3d(), false);
    /// assert_eq!(test_matrix.is_3d(), false);
    /// assert_eq!(test_tensor_nd_2.is_3d(), false);
    /// 
    /// assert_eq!(test_tensor3d.is_3d(), true);
    /// assert_eq!(test_tensor_nd.is_3d(), true);
    /// ```
    pub fn is_3d(&self) -> bool {
        match self {
            GenerationShape::Scalar => false,
            GenerationShape::Vector(_) => false,
            GenerationShape::Matrix(_, _) => false,
            GenerationShape::Tensor3D(_, _, _) => true,
            GenerationShape::Tensor4D(ntens, nmats, _, _) => *ntens == 1 && *nmats > 1,
        }
    }

    /// Checks if the current `GenerationShape` can be conceptually treated as a 4D tensor.
    /// This is true for `GenerationShape` variants with a dimensionality of at least 4, with
    /// any additional dimensions having size 1.
    ///
    /// # Returns
    /// `true` if the shape can be treated as a 4D tensor, `false` otherwise.
    /// 
    /// # Examples
    /// ```
    /// use qudit_expr::GenerationShape;
    /// 
    /// let test_scalar = GenerationShape::Scalar;
    /// let test_vector = GenerationShape::Vector(1);
    /// let test_matrix = GenerationShape::Matrix(1, 1);
    /// let test_tensor3d = GenerationShape::Tensor3D(1, 1, 1);
    /// 
    /// let test_tensor4d = GenerationShape::Tensor4D(9, 9, 9, 9);
    /// let test_tensor_nd = GenerationShape::TensorND(vec![1, 9, 9, 9, 9]);
    ///
    /// assert_eq!(test_scalar.is_4d(), false);
    /// assert_eq!(test_vector.is_4d(), false);
    /// assert_eq!(test_matrix.is_4d(), false);
    /// assert_eq!(test_tensor3d.is_4d(), false);
    /// 
    /// assert_eq!(test_tensor4d.is_4d(), true);
    /// assert_eq!(test_tensor_nd.is_4d(), true);
    /// ```
    pub fn is_4d(&self) -> bool {
        match self {
            GenerationShape::Scalar => false,
            GenerationShape::Vector(_) => false,
            GenerationShape::Matrix(_, _) => false,
            GenerationShape::Tensor3D(_, _, _) => false,
            GenerationShape::Tensor4D(_, _, _, _) => true,
        }
    }

    /// Returns the number of columns for the current shape.
    ///
    /// # Returns
    /// The number of columns.
    ///
    /// # Examples
    /// ```
    /// use qudit_expr::GenerationShape;
    /// let matrix_shape = GenerationShape::Matrix(2, 3);
    /// assert_eq!(matrix_shape.ncols(), 3);
    /// ```
    pub fn ncols(&self) -> usize {
        match self {
            GenerationShape::Scalar => 1,
            GenerationShape::Vector(ncols) => *ncols,
            GenerationShape::Matrix(_, ncols) => *ncols,
            GenerationShape::Tensor3D(_, _, ncols) => *ncols,
            GenerationShape::Tensor4D(_, _, _, ncols) => *ncols,
        }
    }

    /// Returns the number of rows for the current shape.
    ///
    /// # Returns
    /// The number of rows.
    ///
    /// # Examples
    /// ```
    /// use qudit_expr::GenerationShape;
    /// let matrix_shape = GenerationShape::Matrix(2, 3);
    /// assert_eq!(matrix_shape.nrows(), 2);
    /// ```
    pub fn nrows(&self) -> usize {
        match self {
            GenerationShape::Scalar => 1,
            GenerationShape::Vector(_) => 1,
            GenerationShape::Matrix(nrows, _) => *nrows,
            GenerationShape::Tensor3D(_, nrows, _) => *nrows,
            GenerationShape::Tensor4D(_, _, nrows, _) => *nrows,
        }
    }

    /// Returns the number of matrices for the current shape.
    ///
    /// # Returns
    /// The number of matrices.
    ///
    /// # Examples
    /// ```
    /// use qudit_expr::GenerationShape;
    /// let tensor3d_shape = GenerationShape::Tensor3D(5, 2, 3);
    /// assert_eq!(tensor3d_shape.nmats(), 5);
    /// ```
    pub fn nmats(&self) -> usize {
        match self {
            GenerationShape::Scalar => 1,
            GenerationShape::Vector(_) => 1,
            GenerationShape::Matrix(_, _) => 1,
            GenerationShape::Tensor3D(nmats, _, _) => *nmats,
            GenerationShape::Tensor4D(_, nmats, _, _) => *nmats,
        }
    }

    /// Returns the number of tensors (in the first dimension) for the current shape.
    ///
    /// # Returns
    /// The number of tensors.
    ///
    /// # Examples
    /// ```
    /// use qudit_expr::GenerationShape;
    /// let tensor4d_shape = GenerationShape::Tensor4D(7, 5, 2, 3);
    /// assert_eq!(tensor4d_shape.ntens(), 7);
    /// ```
    pub fn ntens(&self) -> usize {
        match self {
            GenerationShape::Scalar => 1,
            GenerationShape::Vector(_) => 1,
            GenerationShape::Matrix(_, _) => 1,
            GenerationShape::Tensor3D(_, _, _) => 1,
            GenerationShape::Tensor4D(ntens, _, _, _) => *ntens,
        }
    }

    pub fn calculate_directions(&self, index_sizes: &[usize]) -> Vec<IndexDirection> {
        match self {
            GenerationShape::Scalar => vec![],
            GenerationShape::Vector(nelems) => vec![IndexDirection::Input; index_sizes.len()],
            GenerationShape::Matrix(nrows, _) => {
                let mut index_size_acm = 1usize;
                let mut index_iter = 0;
                let mut index_directions = vec![];
                while index_size_acm < *nrows {
                    index_size_acm *= index_sizes[index_iter]; 
                    index_directions.push(IndexDirection::Output);
                    index_iter += 1;
                }
                while index_iter != index_sizes.len() {
                    index_directions.push(IndexDirection::Input);
                    index_iter += 1;
                }
                index_directions
            }
            GenerationShape::Tensor3D(nmats, nrows, _) => {
                let mut index_size_acm = 1usize;
                let mut index_iter = 0;
                let mut index_directions = vec![];
                while index_size_acm < *nmats {
                    index_size_acm *= index_sizes[index_iter]; 
                    index_directions.push(IndexDirection::Batch);
                    index_iter += 1;
                }
                while index_size_acm < *nrows {
                    index_size_acm *= index_sizes[index_iter]; 
                    index_directions.push(IndexDirection::Output);
                    index_iter += 1;
                }
                while index_iter != index_sizes.len() {
                    index_directions.push(IndexDirection::Input);
                    index_iter += 1;
                }
                index_directions
            }
            GenerationShape::Tensor4D(_, _, _, _) => {
                todo!()
            }
        }
    }
}

// impl From<Vec<TensorIndex>> for GenerationShape {
//     fn from(indices: Vec<TensorIndex>) -> Self {        
//         GenerationShape::from(indices.as_slice())
//     }
// }

impl<I: AsRef<[TensorIndex]>> From<I> for GenerationShape {
    fn from(indices: I) -> Self {        
        let indices = indices.as_ref();
        let mut dimensions = [1, 1, 1, 1];
        for index in indices.iter() {
            match index.direction() {
                IndexDirection::Derivative => dimensions[0] *= index.index_size(),
                IndexDirection::Batch => dimensions[1] *= index.index_size(),
                IndexDirection::Output => dimensions[2] *= index.index_size(),
                IndexDirection::Input => dimensions[3] *= index.index_size(),
            }
        }

        match dimensions {
            [1, 1, 1, 1] => GenerationShape::Scalar,
            [1, 1, 1, nelems] => GenerationShape::Vector(nelems),
            [1, 1, nrows, ncols] => GenerationShape::Matrix(nrows, ncols),
            [1, nmats, nrows, ncols] => GenerationShape::Tensor3D(nmats, nrows, ncols),
            [ntens, nmats, nrows, ncols] => GenerationShape::Tensor4D(ntens, nmats, nrows, ncols),
        }
    }
}

impl Add for GenerationShape {
    type Output = Self;

    fn add(self, other: Self) -> Self::Output {
        match (self, other) {
            // TODO: re-evaluate this...
            (GenerationShape::Scalar, other_shape) => other_shape,
            (other_shape, GenerationShape::Scalar) => other_shape,  
            (GenerationShape::Vector(s_nelems), GenerationShape::Vector(o_nelems)) => {
                GenerationShape::Vector(s_nelems + o_nelems)
            },
            (GenerationShape::Matrix(s_nrows, s_ncols), GenerationShape::Matrix(o_nrows, o_ncols)) => {
                GenerationShape::Matrix(s_nrows + o_nrows, s_ncols + o_ncols)
            },
            (GenerationShape::Tensor3D(s_nmats, s_nrows, s_ncols), GenerationShape::Tensor3D(o_nmats, o_nrows, o_ncols)) => {
                GenerationShape::Tensor3D(s_nmats + o_nmats, s_nrows + o_nrows, s_ncols + o_ncols)
            },
            (GenerationShape::Tensor4D(s_ntens, s_nmats, s_nrows, s_ncols), GenerationShape::Tensor4D(o_ntens, o_nmats, o_nrows, o_ncols)) => {
                GenerationShape::Tensor4D(s_ntens + o_ntens, s_nmats + o_nmats, s_nrows + o_nrows, s_ncols + o_ncols)
            },
            _ => panic!("Cannot add tensors of different fundamental shapes or incompatible ranks."),
        }
    }
}
