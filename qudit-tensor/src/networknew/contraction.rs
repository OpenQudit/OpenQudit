#[derive(Debug, Clone)]
pub struct QuditContraction {
    pub left_id: usize,
    pub right_id: usize,
    pub left_indices: Vec<usize>,
    pub right_indices: Vec<usize>,
    pub total_dimension: usize,
}

impl QuditContraction {
    /// Creates a new Contraction operation.
    pub fn new() -> Self {
        todo!()
    }

    /// Returns the indices involved in the contraction.
    pub fn contraction_indices(&self) -> Vec<usize> {
        todo!()
    }

    /// Predicts the shape of the resulting tensor after contraction.
    pub fn output_shape(&self, tensor1_shape: &[usize], tensor2_shape: &[usize]) -> Vec<usize> {
        todo!()
    }
}
