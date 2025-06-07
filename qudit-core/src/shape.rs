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
            TensorShape::Tensor3D(_, _, ncols) => *ncols == 1,
            TensorShape::TensorND(v) => v.len() >= 2 && v.iter().skip(2).all( |&x| x == 1),
        }
    }
}

