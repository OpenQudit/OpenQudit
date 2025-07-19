use std::collections::HashMap;
use std::ops::Range;

use itertools::Itertools;
use qudit_core::matrix::Mat;
use qudit_core::matrix::MatMut;
use qudit_core::matrix::MatVecMut;
use qudit_core::unitary::DifferentiableUnitaryFn;
use qudit_core::unitary::UnitaryFn;
use qudit_core::unitary::UnitaryMatrix;
use qudit_core::state::StateVector;
use qudit_core::ComplexScalar;
use qudit_core::HasParams;
use qudit_core::HasPeriods;
use qudit_core::QuditPermutation;
use qudit_core::QuditRadices;
use qudit_core::QuditSystem;
use qudit_core::RealScalar;
use qudit_core::ToRadices;
use faer::reborrow::ReborrowMut;

use crate::analysis::simplify_expressions;
use crate::analysis::simplify_matrix;
use crate::complex::ComplexExpression;
use crate::expression::Expression;
use crate::qgl::parse_qobj;
// use crate::qgl::parse_unitary;
use crate::qgl::Expression as CiscExpression;
use crate::DifferentiationLevel;
use crate::StateExpression;
use crate::StateSystemExpression;
use crate::UnitaryExpression;
use qudit_core::TensorShape;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TensorExpression {
    pub name: String,
    pub shape: TensorShape,
    pub variables: Vec<String>,
    pub body: Vec<ComplexExpression>,
    pub dimensions: Vec<usize>,
}

impl TensorExpression {
    pub fn new<T: AsRef<str>>(input: T) -> Self {
        let qdef = match parse_qobj(input.as_ref()) {
            Ok(qdef) => qdef,
            Err(e) => panic!("Parsing Error: {}", e),
        };

        let radices = qdef.get_radices();
        let name = qdef.name;
        let variables = qdef.variables;
        let element_wise = qdef.body.into_element_wise();
        let shape = element_wise.gen_shape();
        let body: Vec<ComplexExpression> = match element_wise {
            CiscExpression::Vector(vec) => vec.into_iter().map(|expr| ComplexExpression::new(expr)).collect(),
            CiscExpression::Matrix(mat) => mat
                .into_iter()
                .flat_map(|row| {
                    row.into_iter()
                        .map(|expr| ComplexExpression::new(expr))
                        .collect::<Vec<_>>()
                })
                .collect(),
            CiscExpression::Tensor(tensor) => tensor
                .into_iter()
                .flat_map(|row| {
                    row.into_iter()
                        .flat_map(|col| {
                            col.into_iter()
                                .map(|expr| ComplexExpression::new(expr))
                                .collect::<Vec<_>>()
                        })
                        .collect::<Vec<_>>()
                })
                .collect(),
            _ => panic!("Tensor body must be a vector"),
        };

        TensorExpression {
            name,
            shape,
            dimensions: radices,
            variables,
            body,
        }
    }

    /// Splits the tensor dimensions into groups based on generation shape.
    ///
    /// TODO: Change return type to `Vec<Vec<usize>>` and handle TensorND.
    pub fn split_dimensions(&self) -> (Vec<usize>, Vec<usize>, Vec<usize>) {
        let (batch_dim_total, output_dim_total, input_dim_total) = match self.generation_shape() {
            qudit_core::TensorShape::Scalar => (0, 0, 0),
            qudit_core::TensorShape::Vector(a) => (0, 0, a),
            qudit_core::TensorShape::Matrix(a, b) => (0, a, b),
            qudit_core::TensorShape::Tensor3D(a, b, c) => (a, b, c),
            _ => panic!("Dynamic tensor shape unsupport"),
        };

        let mut batch_dims = vec![];
        let mut output_dims = vec![];
        let mut input_dims = vec![];

        let mut batch_dim_acm = 1;
        let mut output_dim_acm = 1;
        let mut input_dim_acm = 1;

        let dims = &self.dimensions;
        let mut id_counter = 0;

        while batch_dim_acm < batch_dim_total {
            batch_dims.push(dims[id_counter]);
            batch_dim_acm *= dims[id_counter];
            id_counter += 1;
        }

        while output_dim_acm < output_dim_total {
            output_dims.push(dims[id_counter]);
            output_dim_acm *= dims[id_counter];
            id_counter += 1;
        }

        while input_dim_acm < input_dim_total {
            input_dims.push(dims[id_counter]);
            input_dim_acm *= dims[id_counter];
            id_counter += 1;
        }

        (batch_dims, output_dims, input_dims)
    }

    pub fn to_unitary_expression(&self) -> UnitaryExpression {
        match self.shape {
            TensorShape::Matrix(nrows, ncols) => {
                assert_eq!(nrows, ncols);
                let mut body = Vec::with_capacity(nrows);
                for i in 0..nrows {
                    let start = i * ncols;
                    let end = start + ncols;
                    let row = self.body[start..end].to_vec();
                    body.push(row);
                }
                let radices = self.dimensions[0..(self.dimensions.len()/2)].to_vec().into();
                UnitaryExpression {
                    name: self.name.clone(),
                    radices,
                    variables: self.variables.clone(),
                    body,
                }
            }
            _ => panic!("TensorExpression shape must be a matrix to convert to UnitaryExpression"),
        }
    }

    pub fn to_state_expression(&self) -> StateExpression {
        match self.shape {
            TensorShape::Vector(_) => {
                StateExpression {
                    name: self.name.clone(),
                    radices: QuditRadices::new(&self.dimensions),
                    variables: self.variables.clone(),
                    body: self.body.clone(),
                }
            }
            _ => panic!("TensorExpression shape must be a vector to convert to StateExpression"),
        }
    }

    pub fn to_state_system_expression(&self) -> StateSystemExpression {
        match self.shape {
            TensorShape::Matrix(nrows, ncols) => {
                let mut body = Vec::with_capacity(nrows);
                for i in 0..nrows {
                    let start = i * ncols;
                    let end = start + ncols;
                    let row = self.body[start..end].to_vec();
                    body.push(row);
                }
                StateSystemExpression {
                    name: self.name.clone(),
                    radices: QuditRadices::new(&self.dimensions),
                    variables: self.variables.clone(),
                    body,
                }
            }
            _ => panic!("TensorExpression shape must be a matrix to convert to StateSystemExpression"),
        }
    }

    /// Calculates the strides for each dimension of the tensor.
    ///
    /// The stride for a dimension is the number of elements one must skip in the
    /// flattened array to move to the next element along that dimension.
    /// The strides are calculated such that the last dimension has a stride of 1,
    /// the second to last dimension has a stride equal to the size of the last dimension,
    /// and so on.
    ///
    /// # Examples
    ///
    /// ```
    /// use qudit_expr::TensorExpression;
    ///
    /// let mut tensor = TensorExpression::new("example() {[
    ///         [
    ///             [ 1, 0, 0, 0 ],
    ///             [ 0, 1, 0, 0 ],
    ///             [ 0, 0, 1, 0 ],
    ///             [ 0, 0, 0, 1 ],
    ///         ], [
    ///             [ 1, 0, 0, 0 ],
    ///             [ 0, 1, 0, 0 ],
    ///             [ 0, 0, 0, 1 ],
    ///             [ 0, 0, 1, 0 ],
    ///         ]
    ///     ]}");
    ///
    /// assert_eq!(tensor.tensor_strides(), vec![16, 8, 4, 2, 1]);
    /// ```
    pub fn tensor_strides(&self) -> Vec<usize> {
        let mut strides = Vec::with_capacity(self.dimensions.len());
        let mut current_stride = 1;
        for &dim in self.dimensions.iter().rev() {
            strides.push(current_stride);
            current_stride *= dim;
        }
        strides.reverse();
        strides
    }
    
    pub fn num_params(&self) -> usize {
        self.variables.len()
    }
    
    pub fn dimensions(&self) -> Vec<usize> {
        self.dimensions.clone()
    }

    pub fn generation_shape(&self) -> TensorShape {
        self.shape.clone()
    }
    
    pub fn name(&self) -> &str {
        self.name.as_str()
    }

    pub fn conjugate(&self) -> Self {
        let mut out_body = Vec::new();
        for expr in &self.body {
            out_body.push(expr.conjugate());
        }
        TensorExpression {
            name: format!("{}^_", self.name),
            shape: self.generation_shape(),
            variables: self.variables.clone(),
            body: out_body,
            dimensions: self.dimensions()
        }
    }

    pub fn redimension(&mut self, new_dimensions: Vec<usize>) -> &mut Self {
        assert_eq!(new_dimensions.iter().product::<usize>(), self.body.len(), "Product of new dimensions must match the total number of elements in the tensor body.");
        self.dimensions = new_dimensions;
        self
    }

    pub fn reshape(&mut self, new_shape: TensorShape) -> &mut Self {
        assert_eq!(new_shape.num_elements(), self.body.len()); 
        self.shape = new_shape;
        self
    }

    pub fn permute(&mut self, perm: &[usize]) -> &mut Self {
        assert_eq!(perm.len(), self.dimensions.len());

        let in_dims = self.dimensions.clone();
        let in_strides: Vec<usize> = {
            let mut strides = Vec::with_capacity(in_dims.len() + 1);
            let mut dim_factor = 1;
            for &dim in in_dims.iter() {
                strides.push(dim_factor);
                dim_factor *= dim as usize;
            }
            strides.push(dim_factor);
            strides
        };

        let out_strides: Vec<usize> = {
            let mut strides = Vec::with_capacity(in_dims.len() + 1);
            let mut dim_factor = 1;
            for &pd in perm {
                let dim = in_dims[pd];
                strides.push(dim_factor);
                dim_factor *= dim as usize;
            }
            strides.push(dim_factor);
            strides.reverse();
            strides
        };

        let mut index_perm: Vec<usize> = Vec::with_capacity(self.body.len());
        for i in 0..self.body.len() {
            let coordinate: Vec<usize> = (0..in_dims.len())
                .into_iter()
                .map(|d| ((i % in_strides[d+1]) / in_strides[d]))
                .rev()
                .collect();
            // println!("index: {}, coord: {:?}", i, coordinate);

            let perm_index: usize = perm.iter()
                .enumerate()
                .map(|(d, pd)| coordinate[*pd] * out_strides[d])
                .sum();

            index_perm.push(perm_index);
        }

        let mut swap_vec = vec![];
        std::mem::swap(&mut swap_vec, &mut self.body);
        self.body = swap_vec.into_iter().enumerate().sorted_by(|a, b| index_perm[a.0].cmp(&index_perm[b.0])).map(|(_, expr)| expr).collect();
        self
    }

    pub fn stack_with_identity(&self, positions: &[usize], new_dim: usize) -> TensorExpression {
        // Assertions for input validity
        let (nrows, ncols) = match self.shape {
            TensorShape::Matrix(r, c) => (r, c),
            _ => panic!("TensorExpression must be a square matrix to use stack_with_identity, got {:?}", self.shape),
        };
        assert_eq!(nrows, ncols, "TensorExpression must be a square matrix for stack_with_identity");
        // assert_eq!(nrows, self.dimensions.dimension(), "Matrix dimension must match qudit dimensions for stack_with_identity");
        assert!(positions.len() <= new_dim, "Cannot place tensor in more locations than length of new dimension.");

        // Ensure positions are unique
        let mut sorted_positions = positions.to_vec();
        sorted_positions.sort_unstable();
        assert!(sorted_positions.iter().dedup().count() == sorted_positions.len(), "Positions must be unique");

        // Construct identity expression
        let mut identity = Vec::with_capacity(nrows * ncols);
        for i in 0..nrows {
            for j in 0..ncols {
                if i == j {
                    identity.push(ComplexExpression::one());
                } else {
                    identity.push(ComplexExpression::zero());
                }
            }
        }

        // construct larger tensor
        let mut expressions = Vec::with_capacity(nrows * ncols * new_dim);
        for i in 0..new_dim {
            if positions.contains(&i) {
                expressions.extend(self.body.iter().cloned());
            } else {
                expressions.extend(identity.iter().cloned());
            }
        }

        let new_shape = TensorShape::Tensor3D(new_dim, nrows, ncols);
        let new_radices = [[new_dim].as_slice(), self.dimensions.as_slice()].concat();

        TensorExpression {
            name: format!("Stacked_{}", self.name),
            shape: new_shape,
            variables: self.variables.clone(),
            body: expressions,
            dimensions: new_radices,
        }
    }

    pub fn partial_trace(&self, dimension_pairs: &[(usize, usize)]) -> TensorExpression {
        let in_dims = self.dimensions.clone();
        let num_dims = in_dims.len();

        // 1. Validate dimension_pairs and identify dimensions to keep/trace
        let mut traced_dim_indices = std::collections::HashSet::new();
        for &(d1, d2) in dimension_pairs {
            if d1 >= num_dims || d2 >= num_dims {
                panic!("Dimension index out of bounds: ({}, {}) for dimensions {:?}", d1, d2, in_dims);
            }
            if in_dims[d1] != in_dims[d2] {
                panic!("Dimensions being traced must have the same size: D{} ({}) != D{} ({})", d1, in_dims[d1], d2, in_dims[d2]);
            }
            if !traced_dim_indices.insert(d1) {
                panic!("Dimension {} appears more than once as a trace source.", d1);
            }
            if !traced_dim_indices.insert(d2) {
                panic!("Dimension {} appears more than once as a trace target.", d2);
            }
        }

        let mut remaining_dims_indices: Vec<usize> = (0..num_dims)
            .filter(|&i| !traced_dim_indices.contains(&i))
            .collect();

        let out_dims: Vec<usize> = remaining_dims_indices
            .iter()
            .map(|&i| in_dims[i])
            .collect();

        // If all dimensions are traced, the result is a scalar
        if out_dims.is_empty() {
            if self.body.is_empty() && num_dims == 0 { // Case of tracing an empty/scalar tensor
                 return TensorExpression {
                    name: format!("PartialTraced_{}", self.name),
                    shape: TensorShape::Scalar,
                    variables: self.variables.clone(),
                    body: vec![ComplexExpression::zero()],
                    dimensions: vec![],
                };
            }
            
            // Calculate the trace of the entire tensor
            let mut total_trace_sum = ComplexExpression::zero();
            let in_strides: Vec<usize> = {
                let mut strides = Vec::with_capacity(in_dims.len() + 1);
                let mut dim_factor = 1;
                for &dim in in_dims.iter() {
                    strides.push(dim_factor);
                    dim_factor *= dim as usize;
                }
                strides.push(dim_factor);
                strides
            };

            for i in 0..self.body.len() {
                let original_coordinate: Vec<usize> = (0..in_dims.len())
                    .into_iter()
                    .map(|d| ((i % in_strides[d+1]) / in_strides[d]))
                    .rev()
                    .collect();

                let mut is_trace_element = true;
                for &(d1, d2) in dimension_pairs {
                    if original_coordinate[d1] != original_coordinate[d2] {
                        is_trace_element = false;
                        break;
                    }
                }
                if is_trace_element {
                    total_trace_sum = total_trace_sum + self.body[i].clone();
                }
            }

            return TensorExpression {
                name: format!("PartialTraced_{}", self.name),
                shape: TensorShape::Scalar,
                variables: self.variables.clone(),
                body: vec![total_trace_sum],
                dimensions: vec![], // Scalar has no dimensions
            };
        }

        let new_body_len = out_dims.iter().product::<usize>();
        let mut new_body = vec![ComplexExpression::zero(); new_body_len];

        let in_strides = self.tensor_strides();

        // Calculate output strides for new linear index conversion (row-major)
        let out_strides: Vec<usize> = {
            let mut strides = vec![0; out_dims.len()];
            strides[out_dims.len() - 1] = 1; // Stride for the last dimension is 1
            for i in (0..out_dims.len() - 1).rev() {
                strides[i] = strides[i + 1] * out_dims[i + 1];
            }
            strides
        };

        // 3. Iterate through original tensor elements
        for (i, expr) in self.body.iter().enumerate() {
            let original_coordinate: Vec<usize> = (0..in_dims.len())
                .into_iter()
                .map(|d| ((i % in_strides[d+1]) / in_strides[d]))
                .rev()
                .collect();

            if dimension_pairs.iter().all(|(d1, d2)| original_coordinate[*d1] == original_coordinate[*d2]) {
                // Construct new_coordinate
                let mut new_coordinate: Vec<usize> = Vec::with_capacity(remaining_dims_indices.len());
                for &idx in &remaining_dims_indices {
                    new_coordinate.push(original_coordinate[idx]);
                }

                // Convert new_coordinate to new_linear_idx
                let mut new_linear_idx = 0;
                for (dim_coord, &stride) in new_coordinate.iter().zip(&out_strides) {
                    new_linear_idx += dim_coord * stride;
                }

                new_body[new_linear_idx] += expr;
            }
        }

        // 4. Determine new shape
        let new_shape = match out_dims.len() {
            0 => TensorShape::Scalar,
            1 => TensorShape::Vector(out_dims[0]),
            2 => TensorShape::Matrix(out_dims[0], out_dims[1]),
            3 => TensorShape::Tensor3D(out_dims[0], out_dims[1], out_dims[2]),
            _ => TensorShape::TensorND(out_dims.clone()), // Use TensorND for >3 dimensions
        }; // TODO: Not right, need to identity which ones were removed (batch, output, input?)

        TensorExpression {
            name: format!("PartialTraced_{}", self.name),
            shape: new_shape,
            variables: self.variables.clone(),
            body: new_body,
            dimensions: out_dims,
        }
    }
}
