use qudit_core::ParamIndices;
use qudit_core::TensorShape;
use qudit_expr::TensorExpression;

use super::index::TensorLeg;
use super::index::IndexDirection;

#[derive(Debug, Clone)]
pub struct QuditTensor {
    pub indices: Vec<TensorLeg>,
    pub expression: TensorExpression,
    pub param_indices: ParamIndices,
}

impl QuditTensor {
    /// Construct a new tensor object from a tensor expression and param indices.
    pub fn new(expression: TensorExpression, param_indices: ParamIndices) -> Self {
        let (batch_dims, output_dims, input_dims) = expression.split_dimensions();
        let mut id_counter = 0;
        let mut indices = vec![];
        for batch_dim in batch_dims {
            indices.push(TensorLeg::new(IndexDirection::Batch, id_counter, batch_dim));
        }
        for output_dim in output_dims {
            indices.push(TensorLeg::new(IndexDirection::Output, id_counter, output_dim));
        }
        for input_dim in input_dims {
            indices.push(TensorLeg::new(IndexDirection::Input, id_counter, input_dim));
        }
        QuditTensor {
            indices,
            expression,
            param_indices,
        }
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

    /// Returns the shape of the tensor's data as it is generated.
    pub fn shape(&self) -> TensorShape {
        self.expression.generation_shape()
    }

    /// Permutes the axes of the tensor according to the given permutation.
    pub fn permute(&self, permutation: &[usize]) -> Self {
        todo!()
    }

    /// Reshapes the tensor to the given new shape.
    pub fn reshape(&self, new_shape: &[usize]) -> Self {
        todo!()
    }
}
