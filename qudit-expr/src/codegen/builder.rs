use std::collections::HashMap;

use super::{codegen::CodeGenerator, module::Module};
use qudit_core::{ComplexScalar, ParamIndices, QuditSystem};

use crate::{analysis::{simplify_expressions, simplify_matrix_and_matvec}, expression::Expression, unitary::{MatVecExpression, TensorExpression, TensorGenerationShape, UnitaryExpression}, DerivedExpression};


#[derive(Default, Clone)]
struct IdxMapBuilder {
    nrows: Option<usize>,
    ncols: Option<usize>,
    nmats: Option<usize>,
    row_stride: Option<usize>,
    col_stride: Option<usize>,
    mat_stride: Option<usize>,
    dims: Option<Vec<usize>>,
    perm: Option<Vec<usize>>,
}

impl IdxMapBuilder {
    pub fn new() -> Self {
        Self::default() 
    }

    pub fn nrows(mut self, nrows: usize) -> Self {
        self.nrows = Some(nrows);
        self
    }

    pub fn ncols(mut self, ncols: usize) -> Self {
        self.ncols = Some(ncols);
        self
    }

    pub fn nmats(mut self, nmats: usize) -> Self {
        self.nmats = Some(nmats);
        self
    }

    pub fn row_stride(mut self, row_stride: usize) -> Self {
        self.row_stride = Some(row_stride);
        self
    }

    pub fn col_stride(mut self, col_stride: usize) -> Self {
        self.col_stride = Some(col_stride);
        self
    }

    pub fn mat_stride(mut self, mat_stride: usize) -> Self {
        self.mat_stride = Some(mat_stride);
        self
    }

    pub fn dims(mut self, dims: Vec<usize>) -> Self {
        self.dims = Some(dims);
        self
    }

    pub fn perm(mut self, perm: Vec<usize>) -> Self {
        self.perm = Some(perm);
        self
    }

    pub fn build(self) -> Box<dyn Fn(usize) -> usize> {
        let nrows = self.nrows.expect("Nrows must be set");
        let ncols = self.ncols.expect("Ncols must be set");
        let nmats = self.nmats.expect("Nmats must be set");
        let row_stride = self.row_stride.unwrap_or(1);
        let col_stride = self.col_stride.unwrap_or(nrows);
        let mat_stride = self.mat_stride.unwrap_or(nrows * ncols);
        // let dims = self.dims.unwrap_or(vec![nrows, ncols, nmats]);
        // let perm = self.perm.unwrap_or(vec![0, 1, 2]);

        Box::new(move |idx: usize| -> usize {
            let mat = (idx/2) / (nrows * ncols);
            let row = ((idx/2) % (nrows * ncols)) / ncols;
            let col = (idx/2) % ncols;
            let imag_offset = idx % 2;
            // println!("idx: {} || mat: {}, row: {}, col: {}, imag_offset: {}, nrows: {}, ncols: {}", idx, mat, row, col, imag_offset, nrows, ncols);
            2 * (mat * mat_stride + row * row_stride + col * col_stride) + imag_offset
        })
    }
}


struct CompilableUnit {
    pub fn_name: String,
    pub exprs: Vec<Expression>,
    /// Lookup table for the parameter pointer offset for each variable
    pub variable_table: HashMap<String, usize>,
    pub expr_idx_to_offset_map: Box<dyn Fn(usize) -> usize>,
}

impl CompilableUnit {
    pub fn new(name: &str, exprs: Vec<Expression>, variable_list: Vec<String>) -> Self {
        let expr_idx_to_offset_map = move |idx: usize| -> usize {
            idx
        };
        let variable_table: HashMap<String, usize> = variable_list.into_iter().enumerate().map(|(i, v)| (v, i)).collect();
        CompilableUnit {
            fn_name: name.to_string(),
            exprs,
            variable_table,
            expr_idx_to_offset_map: Box::new(expr_idx_to_offset_map),
        }
    }
    
    pub fn add_to_module<C: ComplexScalar>(&self, module: &Module<C>) {
        // println!("Adding fn_name: {} to module.", self.fn_name);
        // for expr in &self.exprs {
        //     println!("{:?}", expr);
        // }
        let mut codegen = CodeGenerator::new(&module);
        codegen.gen_func(&self.fn_name, &self.exprs, &self.variable_table, &self.expr_idx_to_offset_map).expect("Error generating function.");
    }

    pub fn new_with_matrix_out_buffer(name: &str, exprs: Vec<Expression>, variable_list: Vec<String>, nrows: usize, ncols: usize, col_stride: usize) -> Self {
        let expr_idx_to_offset_map = move |idx: usize| -> usize {
            let row = (idx/2) / ncols;
            let col = (idx/2) % ncols;
            2 * (col * col_stride + row)
        };
        let variable_table: HashMap<String, usize> = variable_list.into_iter().enumerate().map(|(i, v)| (v, i)).collect();
        CompilableUnit {
            fn_name: name.to_string(),
            exprs,
            variable_table,
            expr_idx_to_offset_map: Box::new(expr_idx_to_offset_map),
        }
    }

    pub fn new_with_matvec_out_buffer(name: &str, exprs: Vec<Expression>, variable_list: Vec<String>, nrows: usize, ncols: usize, col_stride: usize, mat_stride: usize) -> Self {
        let expr_idx_to_offset_map = move |idx: usize| -> usize {
            let mat = (idx/2) / (nrows * ncols);
            let row = (idx/2) / nrows;
            let col = (idx/2) % nrows;
            2 * (mat * mat_stride + col * col_stride + row)
        };
        let variable_table: HashMap<String, usize> = variable_list.into_iter().enumerate().map(|(i, v)| (v, i)).collect();
        CompilableUnit {
            fn_name: name.to_string(),
            exprs,
            variable_table,
            expr_idx_to_offset_map: Box::new(expr_idx_to_offset_map),
        }
    }
}

#[derive(Default, Clone)]
struct CompilableUnitBuilder {
    name: Option<String>,
    exprs: Option<Vec<Expression>>,
    variables: Option<Vec<String>>,
    indices: Option<ParamIndices>,
    gen_shape: Option<TensorGenerationShape>,
}

impl CompilableUnitBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn name(mut self, name: &str) -> Self {
        self.name = Some(name.to_string());
        self
    }

    pub fn exprs(mut self, exprs: Vec<Expression>) -> Self {
        self.exprs = Some(exprs);
        self
    }

    pub fn variable_list(mut self, variables: Vec<String>) -> Self {
        self.variables = Some(variables);
        self
    }

    pub fn indices(mut self, indices: ParamIndices) -> Self {
        self.indices = Some(indices);
        self
    }

    pub fn gen_shape(mut self, gen_shape: TensorGenerationShape) -> Self {
        self.gen_shape = Some(gen_shape);
        self
    }

    pub fn build<C: ComplexScalar>(self) -> CompilableUnit {
        let fn_name = self.name.expect("Name must be set");
        let exprs = self.exprs.expect("Exprs must be set");
        let variables = self.variables.expect("Variables must be set");

        let variable_table = if let Some(indices) = self.indices {
            let mut table: HashMap<String, usize> = HashMap::new();
            for (i, idx) in indices.iter().enumerate() {
                table.insert(variables[i].clone(), idx);
            }
            table
        } else {
            variables.into_iter().enumerate().map(|(i, v)| (v, i)).collect()
        };

        let gen_shape = self.gen_shape.expect("Gen shape must be set");

        println!("Shape: {:?}, Expr length: {}", gen_shape, exprs.len());
        assert!(gen_shape.num_elements()*2 == exprs.len());
        
        let idx_map = match gen_shape {
            TensorGenerationShape::Scalar => {
                IdxMapBuilder::new()
                    .ncols(1)
                    .nrows(0)
                    .nmats(0)
                    .build()
            }
            TensorGenerationShape::Vector(length) => {
                IdxMapBuilder::new()
                    .ncols(length)
                    .nrows(1)
                    .nmats(1)
                    .row_stride(1)
                    .build()
            }
            TensorGenerationShape::Matrix(nrows, ncols) => {
                let col_stride = qudit_core::memory::calc_col_stride::<C>(nrows, ncols);
                IdxMapBuilder::new()
                    .ncols(ncols)
                    .nrows(nrows)
                    .nmats(1)
                    .row_stride(1)
                    .col_stride(col_stride)
                    .build()
            }
            TensorGenerationShape::Tensor(nmats, nrows, ncols) => {
                let col_stride = qudit_core::memory::calc_col_stride::<C>(nrows, ncols);
                let mat_stride = qudit_core::memory::calc_mat_stride::<C>(nrows, ncols, col_stride);
                IdxMapBuilder::new()
                    .ncols(ncols)
                    .nrows(nrows)
                    .nmats(nmats)
                    .row_stride(1)
                    .col_stride(col_stride)
                    .mat_stride(mat_stride)
                    .build()
            }
        };

        CompilableUnit { fn_name, exprs, variable_table, expr_idx_to_offset_map: idx_map }
    }
}

#[derive(Debug, PartialEq, Clone, Copy, Eq, Hash, PartialOrd, Ord)]
pub enum DifferentiationLevel {
    None,
    Gradient,
    Hessian,
}

impl DifferentiationLevel {
    pub fn gradient_capable(&self) -> bool {
        match self {
            DifferentiationLevel::None => false,
            DifferentiationLevel::Gradient => true,
            DifferentiationLevel::Hessian => true,
        }
    }

    pub fn hessian_capable(&self) -> bool {
        match self {
            DifferentiationLevel::None => false,
            DifferentiationLevel::Gradient => false,
            DifferentiationLevel::Hessian => true,
        }
    }
}

pub struct ModuleBuilder<C: ComplexScalar> {
    name: String,
    exprs: Vec<CompilableUnit>,
    diff_lvl: DifferentiationLevel,
    phantom: std::marker::PhantomData<C>,
}

impl<C: ComplexScalar> ModuleBuilder<C> {
    pub fn new(name: &str, diff_lvl: DifferentiationLevel) -> Self {
        ModuleBuilder {
            name: name.to_string(),
            exprs: Vec::new(),
            diff_lvl,
            phantom: std::marker::PhantomData,
        }
    }

    pub fn add_unit(mut self, unit: CompilableUnit) -> Self {
        self.exprs.push(unit);
        self
    }

    pub fn add_expression_with_stride(mut self, expr: UnitaryExpression, col_stride: usize) -> Self {
        // let grad_col_stride = qudit_core::memory::calc_col_stride::<C>(expr.dimension(), expr.dimension());
        // let mat_stride = qudit_core::memory::calc_mat_stride::<C>(expr.dimension(), expr.dimension(), grad_col_stride);
        // let compilable_unitary_unit = CompilableUnitaryUnit::new_with_grad(expr, col_stride, grad_col_stride, mat_stride);
        // self.exprs.push(compilable_unitary_unit);
        // self
        todo!()
    }

    pub fn add_expression(mut self, expr: UnitaryExpression) -> Self {
        let col_stride = qudit_core::memory::calc_col_stride::<C>(expr.dimension(), expr.dimension());
        let nrows = expr.dimension();
        let ncols = expr.dimension();
        let UnitaryExpression { name, radices, variables, body } = expr;
        let exprs: Vec<Expression> = body.into_iter().flatten().map(|c| vec![c.real, c.imag]).flatten().collect();
        let compilable_unit = CompilableUnit::new_with_matrix_out_buffer(&name, exprs.clone(), variables.clone(), nrows, ncols, col_stride);
        self.exprs.push(compilable_unit);

        if !self.diff_lvl.gradient_capable() {
            return self;
        }

        let mut grad_exprs = vec![];
        for variable in &variables {
            for expr in &exprs {
                let grad_expr = expr.differentiate(&variable);
                grad_exprs.push(grad_expr);
            }
        }

        let simplified_exprs = simplify_expressions(exprs.into_iter().chain(grad_exprs.into_iter()).collect());
        let mat_stride = qudit_core::memory::calc_mat_stride::<C>(nrows, ncols, col_stride);
        let compilable_unit = CompilableUnit::new_with_matvec_out_buffer(&(name + "_grad"), simplified_exprs, variables, nrows, ncols, col_stride, mat_stride);
        self.exprs.push(compilable_unit);
        self
    }

    pub fn add_tensor_expression(mut self, expr: TensorExpression) -> Self {
        let TensorExpression { name, shape, variables, body, dimensions } = expr;
        let exprs: Vec<Expression> = body.into_iter().map(|c| vec![c.real, c.imag]).flatten().collect();
        let unit = CompilableUnitBuilder::new()
            .name(&name)
            .exprs(exprs.clone())
            .variable_list(variables.clone())
            .gen_shape(shape.clone())
            .build::<C>();
        self.exprs.push(unit);

        if self.diff_lvl.gradient_capable() {
            let mut grad_exprs = vec![];
            for variable in &variables {
                for expr in &exprs {
                    let grad_expr = expr.differentiate(&variable);
                    grad_exprs.push(grad_expr);
                }
            }

            let simplified_exprs = simplify_expressions(exprs.into_iter().chain(grad_exprs.into_iter()).collect());
            let unit = CompilableUnitBuilder::new()
                .name(&(name.clone() + "_grad"))
                .exprs(simplified_exprs)
                .variable_list(variables.clone())
                .gen_shape(shape.derivative_shape(1 + variables.len()))
                .build::<C>();
            self.exprs.push(unit);
        }

        self
    }

    pub fn add_tensor_expression_with_param_indices(mut self, expr: TensorExpression, indices: ParamIndices) -> Self {
        let TensorExpression { name, shape, variables, body, dimensions } = expr;
        let exprs: Vec<Expression> = body.into_iter().map(|c| vec![c.real, c.imag]).flatten().collect();
        let unit = CompilableUnitBuilder::new()
            .name(&name)
            .exprs(exprs.clone())
            .variable_list(variables.clone())
            .indices(indices.clone())
            .gen_shape(shape.clone())
            .build::<C>();
        self.exprs.push(unit);

        if self.diff_lvl.gradient_capable() {
            let mut grad_exprs = vec![];
            for variable in &variables {
                for expr in &exprs {
                    let grad_expr = expr.differentiate(&variable);
                    grad_exprs.push(grad_expr);
                }
            }

            let simplified_exprs = simplify_expressions(exprs.into_iter().chain(grad_exprs.into_iter()).collect());
            let unit = CompilableUnitBuilder::new()
                .name(&(name.clone() + "_grad"))
                .exprs(simplified_exprs)
                .variable_list(variables.clone())
                .indices(indices)
                .gen_shape(shape.derivative_shape(1 + variables.len()))
                .build::<C>();
            self.exprs.push(unit);
        }

        self


        // match shape {
        //     TensorGenerationShape::Scalar => {
        //         let compilable_unit = CompilableUnit::new(&name, exprs, variables);
        //         self.exprs.push(compilable_unit);
        //     }
        //     TensorGenerationShape::Vector(length) => {
        //         let compilable_unit = CompilableUnit::new(&name, exprs, variables);
        //         self.exprs.push(compilable_unit);
        //     }
        //     TensorGenerationShape::Matrix(nrows, ncols) => {
        //         let col_stride = qudit_core::memory::calc_col_stride::<C>(nrows, ncols);
        //         let compilable_unit = CompilableUnit::new_with_matrix_out_buffer(&name, exprs, variables, nrows, ncols, col_stride);
        //         self.exprs.push(compilable_unit);
        //     }
        //     TensorGenerationShape::Tensor(nrows, ncols, nmats) => {
        //         let col_stride = qudit_core::memory::calc_col_stride::<C>(nrows, ncols);
        //         let mat_stride = qudit_core::memory::calc_mat_stride::<C>(nrows, ncols, col_stride);
        //         let compilable_unit = CompilableUnit::new_with_matvec_out_buffer(&name, exprs, variables, nrows, ncols, col_stride, mat_stride);
        //         self.exprs.push(compilable_unit);
        //     }
        // }

        // if !self.diff_lvl.gradient_capable() {
        //     return self;
        // }

        // todo!()
        // self
    }

    // pub fn add_tensor_expression_with_param_indices(mut self, expr: TensorExpression, param_indices: ParamIndices) -> Self {
    //     let TensorExpression { name, shape, variables, body, dimensions } = expr;
    //     let exprs = body.into_iter().map(|c| vec![c.real, c.imag]).flatten().collect();
    //     match shape {
    //         TensorGenerationShape::Scalar => {
    //             let compilable_unit = CompilableUnit::new(&name, exprs, variables);
    //             self.exprs.push(compilable_unit);
    //         }
    //         TensorGenerationShape::Vector(length) => {
    //             let compilable_unit = CompilableUnit::new(&name, exprs, variables);
    //             self.exprs.push(compilable_unit);
    //         }
    //         TensorGenerationShape::Matrix(nrows, ncols) => {
    //             let col_stride = qudit_core::memory::calc_col_stride::<C>(nrows, ncols);
    //             let compilable_unit = CompilableUnit::new_with_matrix_out_buffer(&name, exprs, variables, nrows, ncols, col_stride);
    //             self.exprs.push(compilable_unit);
    //         }
    //         TensorGenerationShape::Tensor(nrows, ncols, nmats) => {
    //             let col_stride = qudit_core::memory::calc_col_stride::<C>(nrows, ncols);
    //             let mat_stride = qudit_core::memory::calc_mat_stride::<C>(nrows, ncols, col_stride);
    //             let compilable_unit = CompilableUnit::new_with_matvec_out_buffer(&name, exprs, variables, nrows, ncols, col_stride, mat_stride);
    //             self.exprs.push(compilable_unit);
    //         }
    //     }

    //     if !self.diff_lvl.gradient_capable() {
    //         return self;
    //     }

    //     todo!()
    //     // self
    // }

    pub fn build(self) -> Module<C> {
        let module = Module::new(&self.name, self.diff_lvl);
        for expr in &self.exprs {
            expr.add_to_module(&module);
        }
        module
    }
}

// #[cfg(test)]
// mod tests {
//     use super::*;
//     use qudit_core::c64;
//     use qudit_core::matrix::{Mat, MatVec};

//     struct U3Gate;
//     impl U3Gate {
//         pub fn gen_expr(&self) -> UnitaryExpression {
//             UnitaryExpression::new(
//                 String::from("
//                 utry U3(f1, f2, f3) {
//                     [
//                         [ cos(f1/2), ~e^(i*f3)*sin(f1/2) ],
//                         [ e^(i*f2)*sin(f1/2), e^(i*(f2+f3))*cos(f1/2) ]
//                     ]
//                 }
//             "),
//             )
//         }
//     }

//     #[test]
//     fn test_skeleton() {
//         let u3_gate = U3Gate;
//         let expr = u3_gate.gen_expr();

//         for row in &expr.body {
//             for expr in row {
//                 println!("{:?}", expr);
//             }
//         }

//         let params = vec![1.7, 2.3, 3.1];
//         let mut out_utry: Mat<c64> = Mat::zeros(2, 2);
//         let mut out_grad: MatVec<c64> = MatVec::zeros(2, 2, 3);

//         let module: Module<c64> = ModuleBuilder::new("test", DifferentiationLevel::Gradient)
//             .add_expression_with_stride(expr, out_utry.col_stride().try_into().unwrap())
//             .build();

//         println!("{}", module);

//         let u3_grad_combo_func = module.get_function_and_gradient("U3").unwrap();
//         let out_ptr = out_utry.as_mut().as_ptr_mut() as *mut f64;
//         let out_grad_ptr = out_grad.as_mut().as_mut_ptr().as_ptr() as *mut f64;

//         let start = std::time::Instant::now();
//         for _ in 0..1000000 {
//             unsafe { u3_grad_combo_func.call(params.as_ptr(), out_ptr, out_grad_ptr); }
//         }
//         let duration = start.elapsed();
//         println!("Time elapsed in expensive_function() is: {:?}", duration);
//         println!("Average time: {:?}", duration / 1000000);

//         println!("{:?}", out_utry);
//         println!("{:?}", out_grad);
//     }

//     // #[test]
//     // fn test_time() {
//     //     let u3_gate = U3Gate;
//     //     let expr = u3_gate.gen_expr();
//     //     let mut out_utry: Mat<c64> = Mat::zeros(2, 2);
//     //     let mut out_grad: MatVec<c64> = MatVec::zeros(2, 2, 3);

//     //     let module = Module::new("test", vec![expr.clone()], vec![out_utry.col_stride().try_into().unwrap()], qudit_expr::DifferentiationLevel::Gradient, Some(vec![out_grad.col_stride().try_into().unwrap()]), Some(vec![out_grad.mat_stride().try_into().unwrap()]), None, None);
//     //     let context = module.jit::<c64>();
//     //     let utry_and_grad_func = context.get_utry_and_grad_func(&expr.expr.name);
    
//     //     let out_ptr = unsafe { qudit_core::matrix::matmut_to_ptr(out_utry.as_mut()) };
//     //     let out_grad_ptr = unsafe { qudit_core::matrix::matvecmut_to_ptr(out_grad.as_mut()) };
//     //     for i in 0..100 {
//     //         unsafe { utry_and_grad_func.call(params[i].as_ptr(), out_ptr, out_grad_ptr); }
//     //     }
//     // }
// }

// // impl QGLModule {
// //     pub fn new(
// //         name: &str,
// //         exprs: Vec<UnitaryExpression>,
// //         col_strides: Vec<usize>,
// //         diff_lvl: DifferentiationLevel,
// //         grad_col_strides: Option<Vec<usize>>,
// //         grad_mat_strides: Option<Vec<usize>>,
// //         hess_col_strides: Option<Vec<usize>>,
// //         hess_mat_strides: Option<Vec<usize>>,
// //     ) -> Self {
// //         let context = Context::create();
// //         let mut grad_exprs = None;
// //         let mut hess_exprs = None;
// //         match diff_lvl {
// //             DifferentiationLevel::None => (),
// //             DifferentiationLevel::Gradient => {
// //                 grad_exprs = Some(exprs.iter().map(|expr| auto_diff(&expr.expr)).collect());
//             },
//             DifferentiationLevel::Hessian => {
//                 grad_exprs = Some(exprs.iter().map(|expr| auto_diff(&expr.expr)).collect());
//                 let grad_exprs: Vec<Vec<TypedUnitaryDefinition>> = grad_exprs.clone().unwrap();
//                 let mut hess_exprs_builder = Vec::new();
//                 for grad_expr in grad_exprs {
//                     let variables = grad_expr[0].variables.clone();
//                     hess_exprs_builder.push(auto_diff_symsq(&variables, grad_expr));
//                 }
//                 hess_exprs = Some(hess_exprs_builder);
//             },
//         }

//         QGLModule {
//             name: name.to_string(),
//             exprs,
//             grad_exprs,
//             hess_exprs,
//             col_strides,
//             grad_col_strides,
//             grad_mat_strides,
//             hess_col_strides,
//             hess_mat_strides,
//             context,
//         }
//     }

//     pub fn jit<'a, C: ComplexScalar>(&'a self) -> JITContext<'a, C> {
//         let context = JITContext::create(
//             &self.name,
//             OptimizationLevel::Aggressive,
//             &self.context,
//         );

//         let mut codegen: CodeGenerator<C> = CodeGenerator::new(context);

//         // for (expr, col_stride) in self.exprs.iter().zip(&self.col_strides) {
//         for i in 0..self.exprs.len() {
//             let expr = &self.exprs[i];
//             let col_stride = &self.col_strides[i];
//             match codegen.gen_utry_func(&expr.expr, *col_stride) {
//                 Ok(_) => (),
//                 Err(e) => {
//                     println!("Error: {:?}", e);
//                     panic!("Error generating function.");
//                 } 
//             }
//             if self.grad_exprs.is_some() {
//                 let name = expr.expr.name.clone() + "_grad";
//                 let grad_expr = &self.grad_exprs.as_ref().unwrap()[i];
//                 let grad_col_stride = self.grad_col_strides.as_ref().unwrap()[i];
//                 let grad_mat_stride = self.grad_mat_strides.as_ref().unwrap()[i];
//                 match codegen.gen_grad_func(&name, grad_expr, grad_col_stride, grad_mat_stride) {
//                     Ok(_) => (),
//                     Err(e) => {
//                         println!("Error: {:?}", e);
//                         panic!("Error generating gradient function.");
//                     }
//                 }

//                 let name = expr.expr.name.clone() + "_grad_combo";
//                 match codegen.gen_utry_and_grad_func(&name, &expr.expr, grad_expr, *col_stride, grad_col_stride, grad_mat_stride) {
//                     Ok(_) => (),
//                     Err(e) => {
//                         println!("Error: {:?}", e);
//                         panic!("Error generating function.");
//                     }
//                 }

//                 if self.hess_exprs.is_some() {
//                     let name = expr.expr.name.clone() + "_hess";
//                     let hess_expr = &self.hess_exprs.as_ref().unwrap()[i];
//                     let hess_col_stride = self.hess_col_strides.as_ref().unwrap()[i];
//                     let hess_mat_stride = self.hess_mat_strides.as_ref().unwrap()[i];
//                     match codegen.gen_grad_func(&name, hess_expr, hess_col_stride, hess_mat_stride) {
//                         Ok(_) => (),
//                         Err(e) => {
//                             println!("Error: {:?}", e);
//                             panic!("Error generating Hessian function.");
//                         }
//                     }
//                 }
//             }
//         }

//         codegen.build()
//     }
// }
