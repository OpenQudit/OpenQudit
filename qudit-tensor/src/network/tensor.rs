use qudit_core::unitary::UnitaryMatrix;
use qudit_core::ComplexScalar;
use qudit_core::ParamIndices;
use qudit_core::ParamInfo;
use qudit_core::TensorShape;
use qudit_expr::index::IndexSize;
use qudit_expr::ExpressionId;
use qudit_expr::GenerationShape;
use qudit_expr::TensorExpression;

use qudit_expr::index::TensorIndex;
use qudit_expr::index::IndexDirection;
use qudit_expr::UnitaryExpression;

#[derive(Debug, Clone)]
pub struct QuditTensor {
    pub indices: Vec<TensorIndex>,
    pub expression: ExpressionId,
    pub param_info: ParamInfo,
}

impl QuditTensor {
    /// Construct a new tensor object from a tensor expression and param indices.
    pub fn new(indices: Vec<TensorIndex>, expression: ExpressionId, param_info: ParamInfo) -> Self {
        QuditTensor {
            indices,
            expression,
            param_info,
        }
    }

    pub fn num_indices(&self) -> usize {
        self.indices.len()
    }

    /// Returns a vector of index IDs for all batch legs of the tensor.
    pub fn batch_indices(&self) -> Vec<usize> {
        self.indices.iter()
            .filter_map(|index| match index.direction() {
                IndexDirection::Batch => Some(index.index_id()),
                _ => None,
            })
            .collect()
    }

    /// Returns a vector of index IDs for all output legs of the tensor.
    pub fn output_indices(&self) -> Vec<usize> {
        self.indices.iter()
            .filter_map(|index| match index.direction() {
                IndexDirection::Output => Some(index.index_id()),
                _ => None,
            })
            .collect()
    }

    /// Returns a vector of index IDs for all input legs of the tensor.
    pub fn input_indices(&self) -> Vec<usize> {
        self.indices.iter()
            .filter_map(|index| match index.direction() {
                IndexDirection::Input => Some(index.index_id()),
                _ => None,
            })
            .collect()
    }

    /// Returns a vector of sizes for all batch legs of the tensor.
    pub fn batch_sizes(&self) -> Vec<usize> {
        self.indices.iter()
            .filter_map(|index| match index.direction() {
                IndexDirection::Batch => Some(index.index_size()),
                _ => None,
            })
            .collect()
    }

    /// Returns a vector of sizes for all output legs of the tensor.
    pub fn output_sizes(&self) -> Vec<usize> {
        self.indices.iter()
            .filter_map(|index| match index.direction() {
                IndexDirection::Output => Some(index.index_size()),
                _ => None,
            })
            .collect()
    }

    /// Returns a vector of sizes for all input legs of the tensor.
    pub fn input_sizes(&self) -> Vec<usize> {
        self.indices.iter()
            .filter_map(|index| match index.direction() {
                IndexDirection::Input => Some(index.index_size()),
                _ => None,
            })
            .collect()
    }

    /// Returns the rank of the tensor.
    pub fn rank(&self) -> usize {
        return self.indices.len()
    }
}

