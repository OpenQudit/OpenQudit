#[derive(Debug, Clone)]
pub struct QuditContraction {
    pub left_id: usize,
    pub right_id: usize,
    pub left_indices: Vec<usize>,
    pub right_indices: Vec<usize>,
    pub total_dimension: usize,
}
