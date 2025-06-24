use std::collections::HashMap;

use qudit_core::ParamIndices;
use qudit_core::QuditRadices;
use qudit_expr::TensorExpression;

use super::LocalTensorIndex;
use super::NetworkIndex;
use super::IndexDirection;

#[derive(Debug, Clone)]
pub struct QuditTensor {
    pub indices: Vec<LocalTensorIndex>,
    pub local_to_global_index_map: HashMap<usize, NetworkIndex>,
    pub expression: TensorExpression,
    pub param_indices: ParamIndices,
}

impl QuditTensor {
    pub fn new(
        expression: TensorExpression,
        param_indices: ParamIndices,
    ) -> Self {
        let gen_shape = expression.generation_shape();
        let (right_indices_total, left_indices_total, up_indices_total) = match gen_shape {
            qudit_core::TensorShape::Scalar => (0, 0, 0),
            qudit_core::TensorShape::Vector(a) => (a, 0, 0),
            qudit_core::TensorShape::Matrix(a, b) => (a, b, 0),
            qudit_core::TensorShape::Tensor3D(a, b, c) => (b, c, a),
            _ => panic!("Dynamic tensor shape unsupport"),
        };

        let mut indices = Vec::new();
        let mut id_counter = 0;
        let dims = expression.dimensions();
        let mut up_indices_total_factor = 1;

        while up_indices_total_factor < up_indices_total {
            indices.push(
                LocalTensorIndex {
                    direction: IndexDirection::Up,
                    index_id: id_counter,
                    index_size: dims[id_counter] as usize,
                }
            );
            up_indices_total_factor *= dims[id_counter] as usize;
            id_counter += 1;
        }

        let mut right_indices_total_factor = 1;

        while right_indices_total_factor < right_indices_total {
            indices.push(
                LocalTensorIndex {
                    direction: IndexDirection::Output,
                    index_id: id_counter,
                    index_size: dims[id_counter] as usize,
                }
            );
            right_indices_total_factor *= dims[id_counter] as usize;
            id_counter += 1;
        }

        let mut left_indices_total_factor = 1;

        while left_indices_total_factor < left_indices_total {
            indices.push(
                LocalTensorIndex {
                    direction: IndexDirection::Input,
                    index_id: id_counter,
                    index_size: dims[id_counter] as usize,
                }
            );
            left_indices_total_factor *= dims[id_counter] as usize;
            id_counter += 1;
        }
        QuditTensor {
            indices,
            local_to_global_index_map: HashMap::new(),
            expression,
            param_indices,
        }
    }

    pub fn network_indices(&self) -> Vec<NetworkIndex> {
        let mut indices = Vec::new();
        for i in 0..self.indices.len() {
            if let Some(global_index) = self.local_to_global_index_map.get(&i) {
                indices.push(global_index.clone());
            } else {
                panic!("Index {} not found in local_to_global_index_map", i);
            }
        }
        indices
    }

    pub fn output_indices(&self) -> Vec<usize> {
        self.indices.iter()
            .filter_map(|index| match index.direction {
                IndexDirection::Output => Some(index.index_id),
                _ => None,
            })
            .collect()
    }

    pub fn output_radices(&self) -> QuditRadices {
        self.indices.iter()
            .filter_map(|index| match index.direction {
                IndexDirection::Output => Some(index.index_size as u8),
                _ => None,
            })
            .collect()
    }

    pub fn input_indices(&self) -> Vec<usize> {
        self.indices.iter()
            .filter_map(|index| match index.direction {
                IndexDirection::Input => Some(index.index_id),
                _ => None,
            })
            .collect()
    }

    pub fn input_radices(&self) -> QuditRadices {
        self.indices.iter()
            .filter_map(|index| match index.direction {
                IndexDirection::Input => Some(index.index_size as u8),
                _ => None,
            })
            .collect()
    }

    pub fn up_indices(&self) -> Vec<usize> {
        self.indices.iter()
            .filter_map(|index| match index.direction {
                IndexDirection::Up => Some(index.index_id),
                _ => None,
            })
            .collect()
    }
    
    pub fn up_radices(&self) -> QuditRadices {
        self.indices.iter()
            .filter_map(|index| match index.direction {
                IndexDirection::Up => Some(index.index_size as u8),
                _ => None,
            })
            .collect()
    }
}
