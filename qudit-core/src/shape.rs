
#[derive(Clone, PartialEq, Eq, Hash)]
pub enum TensorShape {
    Scalar,
    Vector(usize),
    Matrix(usize, usize),
    Tensor3D(usize, usize, usize),
    TensorND(Vec<usize>),
}

impl std::fmt::Debug for TensorShape {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TensorShape::Scalar => write!(f, "Scalar"),
            TensorShape::Vector(n) => write!(f, "Vector({})", n),
            TensorShape::Matrix(m, n) => write!(f, "Matrix({}, {})", m, n),
            TensorShape::Tensor3D(m, n, p) => write!(f, "Tensor3D({}, {}, {})", m, n, p),
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
    pub fn num_elements(&self) -> usize {
        match self {
            TensorShape::Scalar => 1,
            TensorShape::Vector(n) => *n,
            TensorShape::Matrix(m, n) => m * n,
            TensorShape::Tensor3D(m, n, p) => m * n * p,
            TensorShape::TensorND(v) => v.iter().product(),
        }
    }

    pub fn derivative_shape(&self, num_params: usize) -> Self {
        match self {
            TensorShape::Scalar => TensorShape::Vector(num_params),
            TensorShape::Vector(n) => TensorShape::Matrix(*n, num_params),
            TensorShape::Matrix(m, n) => TensorShape::Tensor3D(num_params, *m, *n),
            TensorShape::Tensor3D(m, n, p) => TensorShape::TensorND(*m * num_params, *n, *p),
        }
    }

    pub fn is_matrix(&self) -> bool {
        match self {
            TensorShape::Scalar => false,
            TensorShape::Vector(_) => false,
            TensorShape::Matrix(_, _) => true,
            TensorShape::Tensor(_, _, c) => *c == 1,
        }
    }
}
