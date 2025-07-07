/// Represents the shape of a tensor, supporting various dimensions from scalar to N-dimensional.
///
/// This enum defines common tensor shapes like Scalar, Vector, Matrix, and 3D Tensors,
/// along with a general N-dimensional tensor variant.
#[derive(Clone, PartialEq, Eq, Hash)]
pub enum TensorShape {
    /// A 0-dimensional tensor (a single value).
    Scalar,
    /// A 1-dimensional tensor with `nelems` elements.
    Vector(usize),
    /// A 2-dimensional tensor (matrix) with `nrows` rows and `ncols` columns.
    Matrix(usize, usize),
    /// A 3-dimensional tensor with `nmats` matrices, each of `nrows` rows and `ncols` columns.
    Tensor3D(usize, usize, usize),
    /// An N-dimensional tensor where `v` is a vector of dimensions.
    TensorND(Vec<usize>),
}

impl std::fmt::Debug for TensorShape {
    /// Implements the `Debug` trait for `TensorShape`, providing a human-readable
    /// representation of the tensor's dimensions.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TensorShape::Scalar => write!(f, "Scalar"),
            TensorShape::Vector(nelems) => write!(f, "Vector({})", nelems),
            TensorShape::Matrix(nrows, ncols) => write!(f, "Matrix({}, {})", nrows, ncols),
            TensorShape::Tensor3D(nmats, nrows, ncols) => write!(f, "Tensor3D({}, {}, {})", nmats, nrows, ncols),
            TensorShape::TensorND(v) => {
                write!(f, "TensorND(");
                for (i, x) in v.iter().enumerate() {
                    write!(f, "{}", x)?;
                    if i < v.len() - 1 {
                        write!(f, ", ")?;
                    }
                }
                write!(f, ")")
            }
        }
    }
}

impl TensorShape {
    /// Calculates the total number of elements in a tensor of this shape.
    ///
    /// # Returns
    /// The total number of elements as `usize`.
    pub fn num_elements(&self) -> usize {
        match self {
            TensorShape::Scalar => 1,
            TensorShape::Vector(nelems) => *nelems,
            TensorShape::Matrix(nrows, ncols) => nrows * ncols,
            TensorShape::Tensor3D(nmats, nrows, ncols) => nmats * nrows * ncols,
            TensorShape::TensorND(v) => v.iter().product(),
        }
    }

    /// Determines the shape of the derivative of a tensor with respect to `num_params` parameters.
    ///
    /// This method effectively prepends `num_params` to the current tensor's dimensions.
    /// For example, the derivative of a `Scalar` with respect to `num_params` becomes a `Vector(num_params)`.
    /// The derivative of a `Matrix(R, C)` becomes a `Tensor3D(num_params, R, C)`.
    ///
    /// # Arguments
    /// * `num_params` - The number of parameters with respect to which the derivative is taken.
    ///
    /// # Returns
    /// A new `TensorShape` representing the shape of the derivative.
    pub fn derivative_shape(&self, num_params: usize) -> Self {
        match self {
            TensorShape::Scalar => TensorShape::Vector(num_params),
            TensorShape::Vector(nelems) => TensorShape::Matrix(*nelems, num_params),
            TensorShape::Matrix(nrows, ncols) => TensorShape::Tensor3D(num_params, *nrows, *ncols),
            TensorShape::Tensor3D(nmats, nrows, ncols) => TensorShape::TensorND(vec![num_params, *nmats, *nrows, *ncols]),
            TensorShape::TensorND(v) => {
                let mut new_v = vec![num_params];
                new_v.extend_from_slice(v);
                TensorShape::TensorND(new_v)
            }
        }
    }

    /// Checks if the current `TensorShape` can be conceptually treated as a 2-dimensional matrix.
    ///
    /// This is true for:
    /// - `TensorShape::Matrix`
    /// - `TensorShape::Tensor3D` where the last dimension (`ncols`) is 1 (representing a stack of column vectors).
    ///   Note: This interpretation might be specific to the context. A 3D tensor often isn't a "matrix" by itself.
    /// - `TensorShape::TensorND` where the dimensions beyond the first two are all 1.
    ///
    /// # Returns
    /// `true` if the shape can be treated as a matrix, `false` otherwise.
    pub fn is_matrix(&self) -> bool {
        match self {
            TensorShape::Scalar => false,
            TensorShape::Vector(_) => false,
            TensorShape::Matrix(_, _) => true,
            // A Tensor3D can be seen as a matrix if it's essentially a stack of column vectors,
            // or perhaps if it represents a single matrix (nmats=1).
            // The current implementation checks if ncols is 1, implying it's a stack of column vectors.
            TensorShape::Tensor3D(nmats, _, _) => *nmats == 1,
            TensorShape::TensorND(v) => v.len() >= 2 && v.iter().rev().skip(2).all( |&x| x == 1),
        }
    }

    /// Checks if the current `TensorShape` can be conceptually treated as a scalar.
    ///
    /// This is true for:
    /// - `TensorShape::Scalar`.
    /// - `TensorShape::Vector` where the number of elements is 1.
    /// - `TensorShape::Matrix` where the number of columns and rows are 1.
    /// - `TensorShape::Tensor3D` where the number of matrices, rows, and columns are all 1.
    /// - `TensorShape::TensorND` where the size of all dimensions is 1.
    ///
    /// # Returns
    /// `true` if the shape can be treated as a scalar, `false` otherwise.
    /// 
    /// # Examples
    /// ```
    /// use qudit_core::TensorShape;
    /// 
    /// let test_scalar = TensorShape::Scalar;
    /// let test_vector = TensorShape::Vector(1);
    /// let test_matrix = TensorShape::Matrix(1, 1);
    /// let test_tensor3d = TensorShape::Tensor3D(1, 1, 1);
    /// let test_tensor_nd = TensorShape::TensorND(vec![1, 1, 1, 1]);
    ///
    /// assert!(test_scalar.is_scalar());
    /// assert!(test_vector.is_scalar());
    /// assert!(test_matrix.is_scalar());
    /// assert!(test_tensor3d.is_scalar());
    /// assert!(test_tensor_nd.is_scalar());
    /// ```
    pub fn is_scalar(&self) -> bool {
        match self {
            TensorShape::Scalar => true,
            TensorShape::Vector(nelems) => *nelems == 1,
            TensorShape::Matrix(nrows, ncols) => *nrows == 1 && *ncols == 1,
            TensorShape::Tensor3D(nmats, nrows, ncols) => *nmats == 1 && *nrows == 1 && *ncols == 1,
            TensorShape::TensorND(v) => v.iter().all( |&x| x == 1),
        }
    }

    /// Checks if the current `TensorShape` can be conceptually treated as a 1-dimensional vector.
    ///
    /// This is true for:
    /// - `TensorShape::Vector`.
    /// - `TensorShape::Matrix` where one of its dimensions is 1 (e.g., a row or column vector).
    /// - `TensorShape::Tensor3D` where at most one of its dimensions is greater than 1 (meaning it's
    ///   either a scalar-like 3D tensor or a 3D tensor with a single non-unit dimension).
    /// - `TensorShape::TensorND` where, after collapsing dimensions of size 1, at most one
    ///   dimension remains.
    ///
    /// # Returns
    /// `true` if the shape can be treated as a vector, `false` otherwise.
    pub fn is_vector(&self) -> bool {
        match self {
            TensorShape::Scalar => false,
            TensorShape::Vector(_) => true,
            TensorShape::Matrix(nrows, ncols) => *nrows == 1 || *ncols == 1,
            TensorShape::Tensor3D(nmats, nrows, ncols) => {
                let non_one_count = [*nmats, *nrows, *ncols].iter().filter(|&&d| d > 1).count();
                non_one_count == 1
            }
            TensorShape::TensorND(v) => {
                let non_one_count = v.iter().filter(|&&d| d > 1).count();
                non_one_count == 1
            }
        }
    }

    /// Checks if the current `TensorShape` can be conceptually treated as a 3D tensor.
    ///
    /// This is true for:
    /// - `TensorShape::Tensor3D`.
    /// - `TensorShape::TensorND` where the dimensionality is at least 3 and all axes
    ///     outside the first 3 axes have size 1.
    ///
    /// # Returns
    /// `true` if the shape can be treated as a 3D tensor, `false` otherwise.
    /// 
    /// # Examples
    /// ```
    /// use qudit_core::TensorShape;
    /// 
    /// let test_scalar = TensorShape::Scalar;
    /// let test_vector = TensorShape::Vector(1);
    /// let test_matrix = TensorShape::Matrix(1, 1);
    /// let test_tensor3d = TensorShape::Tensor3D(9, 9, 9);
    /// let test_tensor_nd = TensorShape::TensorND(vec![1, 1, 9, 9, 9]);
    ///
    /// assert_eq!(test_scalar.is_tensor3d(), false);
    /// assert_eq!(test_vector.is_tensor3d(), false);
    /// assert_eq!(test_matrix.is_tensor3d(), false);
    /// assert_eq!(test_tensor3d.is_tensor3d(), true);
    /// assert_eq!(test_tensor_nd.is_tensor3d(), true);
    /// ```
    pub fn is_tensor3d(&self) -> bool {
        match self {
            TensorShape::Scalar => false,
            TensorShape::Vector(_) => false,
            TensorShape::Matrix(_, _) => false,
            TensorShape::Tensor3D(_, _, _) => true,
            TensorShape::TensorND(v) => v.len() >= 3 && v.iter().rev().skip(3).all( |&x| x == 1),
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
    /// use qudit_core::shape::TensorShape;
    /// 
    /// let scalar_shape = TensorShape::Scalar;
    /// assert_eq!(scalar_shape.to_vec(), vec![]);
    /// 
    /// let vector_shape = TensorShape::Vector(5);
    /// assert_eq!(vector_shape.to_vec(), vec![5]);
    /// 
    /// let matrix_shape = TensorShape::Matrix(2, 3);
    /// assert_eq!(matrix_shape.to_vec(), vec![2, 3]);
    /// 
    /// let tensor_shape = TensorShape::TensorND(vec![2, 3, 4, 5]);
    /// assert_eq!(tensor_shape.to_vec(), vec![2, 3, 4, 5]);
    /// ```
    pub fn to_vec(&self) -> Vec<usize> {
        match self {
            TensorShape::Scalar => vec![],
            TensorShape::Vector(nelems) => vec![*nelems],
            TensorShape::Matrix(nrows, ncols) => vec![*nrows, *ncols],
            TensorShape::Tensor3D(nmats, nrows, ncols) => vec![*nmats, *nrows, *ncols],
            TensorShape::TensorND(v) => v.clone(),
        }
    }
}

