pub use qudit_expr::index::IndexSize;
pub use qudit_expr::index::IndexId;
pub use qudit_expr::index::IndexDirection;
pub use qudit_expr::index::TensorIndex;
pub use qudit_expr::index::WeightedIndex;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ContractionIndex {
    pub left_id: usize,
    pub right_id: usize,
    pub total_dimension: IndexSize,
}

impl ContractionIndex {
    pub fn index_size(&self) -> IndexSize {
        self.total_dimension
    }
}

/// Network indices are either a contraction between two tensors
/// or an open (output) edge of the network.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum NetworkIndex {
    Output(TensorIndex),
    Contracted(ContractionIndex),
}

impl NetworkIndex {
    pub fn is_output(&self) -> bool {
        match self {
            &NetworkIndex::Output(_) => true,
            &NetworkIndex::Contracted(_) => false,
        }
    }

    pub fn is_contracted(&self) -> bool {
        match self {
            &NetworkIndex::Output(_) => false,
            &NetworkIndex::Contracted(_) => true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_index_direction_ordering() {
        assert!(IndexDirection::Batch < IndexDirection::Output);
        assert!(IndexDirection::Output < IndexDirection::Input);
        assert!(IndexDirection::Batch < IndexDirection::Input);

        assert!(IndexDirection::Input > IndexDirection::Output);
        assert!(IndexDirection::Output > IndexDirection::Batch);
        assert!(IndexDirection::Input > IndexDirection::Batch);

        assert_eq!(IndexDirection::Batch, IndexDirection::Batch);
        assert_ne!(IndexDirection::Batch, IndexDirection::Output);
    }

    #[test]
    fn test_tensor_index_ordering() {
        let ti1 = TensorIndex::new(IndexDirection::Batch, 0, 2);
        let ti2 = TensorIndex::new(IndexDirection::Batch, 1, 2);
        let ti3 = TensorIndex::new(IndexDirection::Output, 0, 2);
        let ti4 = TensorIndex::new(IndexDirection::Output, 1, 2);
        let ti5 = TensorIndex::new(IndexDirection::Input, 0, 2);
        let ti6 = TensorIndex::new(IndexDirection::Input, 1, 2);

        // Test sorting by direction first
        assert!(ti1 < ti3); // Batch < Output
        assert!(ti3 < ti5); // Output < Input
        assert!(ti1 < ti5); // Batch < Input

        // Test sorting by index_id second (for same direction)
        assert!(ti1 < ti2); // Batch, id 0 < Batch, id 1
        assert!(ti3 < ti4); // Output, id 0 < Output, id 1
        assert!(ti5 < ti6); // Input, id 0 < Input, id 1

        // Test equality
        assert_eq!(ti1, TensorIndex::new(IndexDirection::Batch, 0, 2));
        assert_ne!(ti1, ti3);
        assert_ne!(ti1, ti2);
    }
}
