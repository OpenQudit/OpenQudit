use std::collections::HashMap;
use std::ops::Deref;
use std::ops::DerefMut;

use qudit_core::unitary::UnitaryMatrix;
use qudit_core::ComplexScalar;
use super::ComplexExpression;
use crate::expressions::NamedExpression;
use crate::index::IndexSize;
use crate::qgl::parse_qobj;
use crate::qgl::Expression as CiscExpression;
use crate::Expression;
use crate::GenerationShape;
use crate::index::TensorIndex;
use crate::index::IndexDirection;
use crate::UnitaryExpression;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorExpression {
    indices: Vec<TensorIndex>,
    inner: NamedExpression,
}

impl AsRef<NamedExpression> for TensorExpression {
    fn as_ref(&self) -> &NamedExpression {
        &self.inner
    }
}

impl Deref for TensorExpression {
    type Target = NamedExpression;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl DerefMut for TensorExpression {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.inner
    }
}

impl TensorExpression {
    pub fn new<T: AsRef<str>>(input: T) -> Self {
        let qdef = match parse_qobj(input.as_ref()) {
            Ok(qdef) => qdef,
            Err(e) => panic!("Parsing Error: {}", e),
        };

        let indices = qdef.get_tensor_indices();
        let name = qdef.name;
        let variables = qdef.variables;
        let element_wise = qdef.body.into_element_wise();
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
            indices,
            inner: NamedExpression::new(name, variables, body),
        }
    }

    pub fn from_raw(indices: Vec<TensorIndex>, inner: NamedExpression) -> Self {
        TensorExpression {
            indices,
            inner,
        }
    }
 
    pub fn indices(&self) -> &[TensorIndex] {
        &self.indices
    }

    pub fn dimensions(&self) -> Vec<IndexSize> {
        self.indices.iter().map(|idx| idx.index_size()).collect()
    }

    pub fn rank(&self) -> usize {
        self.indices.len()
    }

    pub fn generation_shape(&self) -> GenerationShape {
        self.indices().into()
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
        let mut strides = Vec::with_capacity(self.indices.len());
        let mut current_stride = 1;
        for &index in self.indices.iter().rev() {
            strides.push(current_stride);
            current_stride *= index.index_size();
        }
        strides.reverse();
        strides
    }

    pub fn reindex(&mut self, new_indices: Vec<TensorIndex>) -> &mut Self {
        assert_eq!(new_indices.iter().map(|idx| idx.index_size()).product::<usize>(), self.num_elements(), "Product of new dimensions must match the total number of elements in the tensor body.");
        
        // Assert that all indices are lined up correctly (Derv | Batch | Output | Input)
        let mut last_direction = IndexDirection::Derivative;
        for index in &new_indices {
            let current_direction = index.direction();
            if current_direction < last_direction {
                panic!("New indices are not ordered correctly. Expected order: Derv, Batch, Output, Input.");
            }
            last_direction = current_direction;
        }

        self.indices = new_indices;
        self
    }

    // Really a fused reshape and permutation
    pub fn permute(&mut self, perm: &[usize], redirection: Vec<IndexDirection>) -> &mut Self {
        assert_eq!(perm.len(), self.rank());

        // Store original strides and dimensions for body permutation before `self.indices` is modified
        let original_strides = self.tensor_strides();
        let original_dimensions = self.dimensions();

        //Reorder the TensorIndex objects based on `perm`
        let reordered_indices: Vec<TensorIndex> = perm.iter()
            .enumerate()
            .map(|(id, &p_id)| TensorIndex::new(redirection[id], self.indices[p_id].index_id(), self.indices[p_id].index_size()))
            .collect();

        // Update self.indices with the newly reordered and direction-assigned indices
        self.reindex(reordered_indices);

        // Get the new strides based on the updated `self.indices`
        let new_strides = self.tensor_strides();

        // Permute elements in the body based on the new index order.
        let mut elem_perm: Vec<usize> = Vec::with_capacity(self.num_elements());
        for i in 0..self.num_elements() {
            let mut original_coordinate: Vec<usize> = Vec::with_capacity(self.rank());
            let mut temp_i = i;
            for d_idx in 0..self.rank() {
                original_coordinate.push((temp_i / original_strides[d_idx]) % original_dimensions[d_idx]);
                temp_i %= original_strides[d_idx]; // Update temp_i for next dimension
            }

            // Map original coordinate components to their new positions according to `perm`.
            // If `perm[j]` is `k`, it means the `j`-th dimension in the new order
            // corresponds to the `k`-th dimension in the original order.
            let mut permuted_coordinate: Vec<usize> = vec![0; self.rank()];
            for j in 0..self.rank() {
                permuted_coordinate[j] = original_coordinate[perm[j]];
            }

            // Calculate new linear index using the permuted coordinate and new strides
            let mut new_linear_idx = 0;
            for d_idx in 0..self.rank() {
                new_linear_idx += permuted_coordinate[d_idx] * new_strides[d_idx];
            }
            elem_perm.push(new_linear_idx);
        }

        self.body.apply_element_permutation(&elem_perm);
        self
    }

    pub fn stack_with_identity(&self, positions: &[usize], new_dim: usize) -> TensorExpression {
        // Assertions for input validity
        let (nrows, ncols) = match self.generation_shape() {
            GenerationShape::Matrix(r, c) => (r, c),
            _ => panic!("TensorExpression must be a square matrix to use stack_with_identity, got {:?}", self.generation_shape()),
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

        let new_indices = [TensorIndex::new(IndexDirection::Batch, 0, new_dim)].into_iter()
            .chain(self.indices().iter().map(|idx| TensorIndex::new(idx.direction(), idx.index_id() + 1, idx.index_size())))
            .collect();

        TensorExpression {
            indices: new_indices,
            inner: NamedExpression::new(format!("Stacked_{}", self.name), self.variables().to_owned(), expressions),
        }
    }

    pub fn partial_trace(&self, dimension_pairs: &[(usize, usize)]) -> TensorExpression {
        let in_dims = self.indices();
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

        let remaining_dims_indices: Vec<usize> = (0..num_dims)
            .filter(|&i| !traced_dim_indices.contains(&i))
            .collect();

        let out_dims: Vec<usize> = remaining_dims_indices
            .iter()
            .map(|&i| in_dims[i].index_size())
            .collect();

        let new_body_len = out_dims.iter().product::<usize>();
        let mut new_body = vec![ComplexExpression::zero(); new_body_len];

        let mut in_strides = self.tensor_strides();
        in_strides.insert(0, self.num_elements());

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
                let mut new_linear_idx = 0;
                for (&idx, stride) in remaining_dims_indices.iter().zip(&out_strides) {
                    new_linear_idx += original_coordinate[idx] * stride;
                }

                new_body[new_linear_idx] += expr;
            }
        }

        let new_indices = remaining_dims_indices.iter().map(|x| in_dims[*x]).collect();

        TensorExpression {
            indices: new_indices,
            inner: NamedExpression::new(format!("PartialTraced_{}", self.name), self.variables().to_owned(), new_body),
        }
    }

    /// replace all variables with values, new variables given in variables input
    pub fn substitute_parameters<S: AsRef<str>, C: AsRef<Expression>>(&self, variables: &[S], values: &[C]) -> Self {
        let sub_map: HashMap<_, _> = self.variables.iter().zip(values.iter()).map(|(k, v)| (Expression::Variable(k.to_string()), v.as_ref())).collect();

        let mut new_body = vec![];
        for expr in self.body.iter() {
            let mut new_expr = None; 
            for (var, value) in &sub_map {
                match new_expr {
                    None => {
                        new_expr = Some(expr.substitute(var, value))
                    }
                    Some(ref mut e) => *e = e.substitute(var, value),
                }
            } 
            match new_expr {
                None => new_body.push(expr.clone()),
                Some(e) => new_body.push(e),
            }
        }

        let new_variables = variables.iter().map(|s| s.as_ref().to_string()).collect();

        TensorExpression {
            indices: self.indices.clone(),
            inner: NamedExpression::new(format!("{}_subbed", self.name), new_variables, new_body),
        }
    }

    pub fn destruct(self) -> (String, Vec<String>, Vec<ComplexExpression>, Vec<TensorIndex>) {
        let Self { inner, indices } = self;
        let (name, variables, body) = inner.destruct();
        (name, variables, body, indices)
    }
}

impl<C: ComplexScalar> From<UnitaryMatrix<C>> for TensorExpression {
    fn from(utry: UnitaryMatrix<C>) -> Self {
        UnitaryExpression::from(utry).to_tensor_expression()
    }
}

impl From<TensorExpression> for NamedExpression {
    fn from(value: TensorExpression) -> Self {
        value.inner
    }
}
