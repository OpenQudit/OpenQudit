use qudit_expr::GenerationShape;

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct TensorBuffer {
    shape: GenerationShape,
    num_params: usize,
}

impl TensorBuffer {
    pub fn new(shape: GenerationShape, num_params: usize) -> Self {
        TensorBuffer { shape, num_params }
    }
    pub fn ncols(&self) -> usize {
        self.shape.ncols()
    }

    pub fn nrows(&self) -> usize {
        self.shape.nrows()
    }

    pub fn nmats(&self) -> usize {
        self.shape.nmats()
    }

    pub fn num_params(&self) -> usize {
        self.num_params
    }

    pub fn shape(&self) -> GenerationShape {
        self.shape
    }
}
