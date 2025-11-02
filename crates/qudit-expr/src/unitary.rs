
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct StateExpression {
    pub name: String,
    pub radices: QuditRadices,
    pub variables: Vec<String>,
    pub body: Vec<ComplexExpression>,
}

impl StateExpression {
    /// Creates a new `StateExpression` from a QGL string representation.
    ///
    /// This function parses the input string as a QGL object, then converts it
    /// into a `TensorExpression` and subsequently into a `StateExpression`.
    ///
    /// # Arguments
    ///
    /// * `input` - A type that can be converted to a string reference,
    ///             representing the QGL definition of the state expression.
    ///
    /// # Returns
    ///
    /// A new `StateExpression` instance.
    pub fn new<T: AsRef<str>>(input: T) -> Self {
        TensorExpression::new(input).into()
    }

    pub fn zero<T: ToRadices>(radices: T) -> Self {
        let radices = radices.to_radices();
        let mut body = vec![ComplexExpression::zero(); radices.dimension()];
        body[0] = ComplexExpression::one();
        Self {
            name: "Zero".into(),
            radices,
            variables: vec![],
            body,
        }
    }

    pub fn to_tensor_expression(self) -> TensorExpression {
        // TODO: What about distinguishing between bras and kets??!?
        let indices = self.radices.iter().enumerate().map(|(id, r)| TensorIndex::new(IndexDirection::Output, id, *r as IndexSize)).collect();
        TensorExpression::from_raw(indices, NamedExpression::new(self.name, self.variables, self.body))
    }

    /// Evaluates the state expression with the given arguments and returns a `StateVector`.
    ///
    /// This function substitutes the provided real scalar arguments into the complex
    /// expressions that define the state vector's body, and then constructs a
    /// `StateVector` from the evaluated complex values.
    ///
    /// # Type Parameters
    ///
    /// * `C`: A type that implements `ComplexScalar`, representing the complex number
    ///        type for the state vector elements.
    ///
    /// # Arguments
    ///
    /// * `args` - A slice of real scalar values (`C::R`) to substitute for the
    ///            variables in the expression. The order of arguments must match
    ///            the order of `self.variables`.
    ///
    /// # Returns
    ///
    /// A `StateVector<C>` containing the evaluated complex elements of the state.
    pub fn eval<C: ComplexScalar>(&self, args: &[C::R]) -> StateVector<C> {
        let arg_map = self.variables.iter().zip(args.iter()).map(|(a, b)| (a.as_str(), *b)).collect();
        let evaluated_body: Vec<C> = self.body.iter().map(|expr| expr.eval(&arg_map)).collect();
        StateVector::new(self.radices.clone(), evaluated_body)
    }

    pub fn conjugate(&self) -> Self {
        let new_body = self.body.iter().map(|expr| expr.conjugate()).collect();
        StateExpression {
            name: format!("{}^_", self.name),
            radices: self.radices.clone(),
            variables: self.variables.clone(),
            body: new_body,
        }
    }

    pub fn name(&self) -> String {
        self.name.clone()
    }
}



// #[cfg(test)]
// mod tests {
//     use crate::{Module, ModuleBuilder};

//     use qudit_core::{c64, matrix::MatVec};
//     use super::*;

//     #[test]
//     fn test_cnot_reshape2() {
//         let mut cnot = TensorExpression::new("CNOT() {
//             [
//                 [1, 0, 0, 0],
//                 [0, 1, 0, 0],
//                 [0, 0, 0, 1],
//                 [0, 0, 1, 0],
//             ]
//         }");

//         let name = cnot.name().to_owned();
//         let reshaped = cnot.reshape(TensorShape::Matrix(2, 8)).permute(&vec![2, 0, 1, 3]);
//         // println!("{:?}", reshaped);

//         let module: Module<c64> = ModuleBuilder::new("test", DifferentiationLevel::None)
//             .add_tensor_expression(reshaped.clone())
//             .build();

//         // let mut out_tensor: MatVec<c64> = MatVec::zeros(4, 4, 2);
//         let col_stride = qudit_core::memory::calc_col_stride::<c64>(2, 8);
//         let mut memory = qudit_core::memory::alloc_zeroed_memory::<c64>(col_stride*8);
        
//         let matmut = unsafe {
//             qudit_core::matrix::MatMut::from_raw_parts_mut(memory.as_mut_ptr(), 2, 8, 1, col_stride as isize)
//         };
//         // let mut out_tensor: Mat<c64> = Mat::zeros(2, 8);
//         // let mut out_ptr: *mut f64 = out_tensor.as_ptr_mut() as *mut f64;
//         let mut out_ptr: *mut f64 = memory.as_mut_ptr() as *mut f64;
//         let func = module.get_function(&name).unwrap();

//         let null_ptr = std::ptr::null() as *const f64;

//         unsafe { func.call(null_ptr, out_ptr); }

//         println!("{:?}", matmut);
//         // for r in 0..matmut.nrows() {
//         //     for c in 0..matmut.ncols() {
//         //         println!("({}, {}): {}", r, c, matmut.get(r, c));
//         //     }
//         // }
//     }

//     #[test]
//     fn test_cnot_reshape() {
//         let mut cnot = TensorExpression::new("CNOT() {
//             [
//                 [1, 0, 0, 0],
//                 [0, 1, 0, 0],
//                 [0, 0, 0, 1],
//                 [0, 0, 1, 0],
//             ]
//         }");

//         let name = cnot.name().to_owned();
//         let reshaped = cnot.reshape(TensorShape::Matrix(8, 2)).permute(&vec![0, 1, 3, 2]);

//         let module: Module<c64> = ModuleBuilder::new("test", DifferentiationLevel::None)
//             .add_tensor_expression(reshaped.clone())
//             .build();

//         // let mut out_tensor: MatVec<c64> = MatVec::zeros(4, 4, 2);
//         let mut out_tensor: Mat<c64> = Mat::zeros(8, 2);
//         let mut out_ptr: *mut f64 = out_tensor.as_ptr_mut() as *mut f64;
//         let func = module.get_function(&name).unwrap();

//         let null_ptr = std::ptr::null() as *const f64;

//         unsafe { func.call(null_ptr, out_ptr); }

//         println!("{:?}", out_tensor);

//     }


//     #[test]
//     fn test_tensor_gen() {
//         let expr = TensorExpression::new("ZZParity() {
//             [
//                 [
//                     [ 1, 0, 0, 0 ], 
//                     [ 0, 0, 0, 0 ],
//                     [ 0, 0, 0, 0 ],
//                     [ 0, 0, 0, 1 ],
//                 ],
//                 [
//                     [ 0, 0, 0, 0 ], 
//                     [ 0, 1, 0, 0 ],
//                     [ 0, 0, 1, 0 ],
//                     [ 0, 0, 0, 0 ],
//                 ],
//             ]
//         }");

//         // for elem in expr.body.iter() {
//         //     println!("{:?}", elem);
//         // }

//         let name = expr.name().to_owned();

//         let module: Module<c64> = ModuleBuilder::new("test", DifferentiationLevel::None)
//             .add_tensor_expression(expr)
//             .build();

//         let mut out_tensor: MatVec<c64> = MatVec::zeros(4, 4, 2);
//         let mut out_ptr: *mut f64 = out_tensor.as_mut_ptr() as *mut f64;
//         let func = module.get_function(&name).unwrap();

//         let null_ptr = std::ptr::null() as *const f64;

//         unsafe { func.call(null_ptr, out_ptr); }

//         println!("{:?}", out_tensor);

// //         let params = vec![1.7, 2.3, 3.1];
// //         let mut out_utry: Mat<c64> = Mat::zeros(2, 2);
// //         let mut out_grad: MatVec<c64> = MatVec::zeros(2, 2, 3);

// //         let module: Module<c64> = ModuleBuilder::new("test", DifferentiationLevel::Gradient)
// //             .add_expression_with_stride(expr, out_utry.col_stride().try_into().unwrap())
// //             .build();

// //         println!("{}", module);

// //         let u3_grad_combo_func = module.get_function_and_gradient("U3").unwrap();
// //         let out_ptr = out_utry.as_mut().as_ptr_mut() as *mut f64;
// //         let out_grad_ptr = out_grad.as_mut().as_mut_ptr().as_ptr() as *mut f64;

// //         let start = std::time::Instant::now();
// //         for _ in 0..1000000 {
// //             unsafe { u3_grad_combo_func.call(params.as_ptr(), out_ptr, out_grad_ptr); }
// //         }
// //         let duration = start.elapsed();
// //         println!("Time elapsed in expensive_function() is: {:?}", duration);
// //         println!("Average time: {:?}", duration / 1000000);

// //         println!("{:?}", out_utry);
// //         println!("{:?}", out_grad);
//     }
// }

impl<C: ComplexScalar> From<UnitaryMatrix<C>> for UnitaryExpression {
    fn from(utry: UnitaryMatrix<C>) -> Self {
        let mut body = vec![Vec::with_capacity(utry.ncols()); utry.nrows()];
        for col in utry.col_iter() {
            for (row_id, elem) in col.iter().enumerate() {
                body[row_id].push(ComplexExpression::from(*elem));
            }
        }

        UnitaryExpression {
            name: "Constant".into(),
            radices: utry.radices(),
            variables: vec![],
            body,
        }
    }
}

impl From<TensorExpression> for UnitaryExpression {
    fn from(value: TensorExpression) -> Self {    
        match value.generation_shape() {
            GenerationShape::Matrix(nrows, ncols) => {
                assert_eq!(nrows, ncols);
                let mut body = Vec::with_capacity(nrows);
                for i in 0..nrows {
                    let start = i * ncols;
                    let end = start + ncols;
                    let row = value[start..end].to_vec();
                    body.push(row);
                }
                let radices = QuditRadices::from_iter(value.indices().iter().filter(|&i| i.direction().is_input()).map(|i| i.index_size()));
                // TODO: use destruct to avoid allocations
                UnitaryExpression {
                    name: value.name().to_owned(),
                    radices,
                    variables: value.variables().to_owned(),
                    body,
                }
            }
            // TODO: Should be done with TryFrom
            _ => panic!("TensorExpression shape must be a matrix to convert to UnitaryExpression"),
        }
    }
}

impl From<TensorExpression> for StateExpression {
    fn from(value: TensorExpression) -> Self {    
        match value.generation_shape() {
            GenerationShape::Vector(_) => {
                let radices = QuditRadices::from_iter(value.indices().iter().map(|i| i.index_size()));
                // TODO: use destruct to avoid allocations
                StateExpression {
                    name: value.name().to_owned(),
                    radices,
                    variables: value.variables().to_owned(),
                    body: value.elements().to_owned(),
                }
            }
            // TODO: Should be done with TryFrom
            _ => panic!("TensorExpression shape must be a vector to convert to StateExpression"),
        }
    }
}

impl From<TensorExpression> for StateSystemExpression {
    fn from(value: TensorExpression) -> Self {    
        match value.generation_shape() {
            GenerationShape::Tensor3D(ntens, nrows, ncols) => {
                let body = if nrows == 1 {
                    let mut body = Vec::with_capacity(ntens);
                    for i in 0..ntens {
                        let start = i * ncols;
                        let end = start + ncols;
                        let row = value[start..end].to_vec();
                        body.push(row);
                    }
                    body
                } else if ncols == 1 {
                    let mut body = Vec::with_capacity(ntens);
                    for i in 0..ntens {
                        let start = i * nrows;
                        let end = start + nrows;
                        let row = value[start..end].to_vec();
                        body.push(row);
                    }
                    body
                } else {
                    panic!("Wrong (TODO: better message).")
                };

                let radices = QuditRadices::from_iter(value.indices().iter().filter(|&i| !i.direction().is_batch()).map(|i| i.index_size()));

                StateSystemExpression {
                    name: value.name().to_owned(),
                    radices,
                    variables: value.variables().to_owned(),
                    body,
                }
            }
            _ => panic!("TensorExpression shape must be a Tensor3D to convert to StateSystemExpression"),
        }
    }
}

impl From<UnitaryExpression> for TensorExpression {
    fn from(value: UnitaryExpression) -> Self {
        let flattened_body: Vec<ComplexExpression> = value.body.clone().into_iter().flat_map(|row| row.into_iter()).collect();
        let indices = value.radices.iter()
            .map(|&r| (IndexDirection::Output, r))
            .chain(value.radices.iter().map(|&r| (IndexDirection::Input, r)))
            .enumerate()
            .map(|(id, (dir, size))| TensorIndex::new(dir, id, size as usize))
            .collect();
        TensorExpression::from_raw(indices, NamedExpression::new(value.name.clone(), value.variables.clone(), flattened_body))
    }
}
