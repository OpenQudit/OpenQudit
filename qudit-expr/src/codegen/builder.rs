use super::{codegen::CodeGenerator, module::Module};
use qudit_core::{ComplexScalar, QuditSystem};

use crate::{analysis::simplify_matrix_and_matvec, unitary::{MatVecExpression, UnitaryExpression}};

struct CompilableUnitaryUnit {
    pub expr: UnitaryExpression,
    pub mat_idx_to_offset_map: Box<dyn Fn(usize, usize) -> usize>,
    pub grad_idx_to_offset_map: Option<Box<dyn Fn(usize, usize, usize) -> usize>>,
}

impl CompilableUnitaryUnit {
    #[allow(dead_code)]
    pub fn new(expr: UnitaryExpression, col_stride: usize) -> Self {
        let mat_idx_to_offset_map = move |row: usize, col: usize| -> usize {
            2 * (col * col_stride + row)
        };
        CompilableUnitaryUnit {
            expr,
            mat_idx_to_offset_map: Box::new(mat_idx_to_offset_map),
            grad_idx_to_offset_map: None,
        }
    }

    pub fn new_with_grad(expr: UnitaryExpression, col_stride: usize, grad_col_stride: usize, mat_stride: usize) -> Self {
        let mat_idx_to_offset_map = move |row: usize, col: usize| -> usize {
            2 * (col * col_stride + row)
        };
        let grad_idx_to_offset_map = move |grad_idx: usize, row: usize, col: usize| -> usize {
            2 * (grad_idx * mat_stride + col * grad_col_stride + row)
        };
        CompilableUnitaryUnit {
            expr,
            mat_idx_to_offset_map: Box::new(mat_idx_to_offset_map),
            grad_idx_to_offset_map: Some(Box::new(grad_idx_to_offset_map)),
        }
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
    exprs: Vec<CompilableUnitaryUnit>,
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

    pub fn add_expression_with_stride(mut self, expr: UnitaryExpression, col_stride: usize) -> Self {
        let grad_col_stride = qudit_core::memory::calc_col_stride::<C>(expr.dimension(), expr.dimension());
        let mat_stride = qudit_core::memory::calc_mat_stride::<C>(expr.dimension(), expr.dimension(), grad_col_stride);
        let compilable_unitary_unit = CompilableUnitaryUnit::new_with_grad(expr, col_stride, grad_col_stride, mat_stride);
        self.exprs.push(compilable_unitary_unit);
        self
    }

    pub fn add_expression(mut self, expr: UnitaryExpression) -> Self {
        let col_stride = qudit_core::memory::calc_col_stride::<C>(expr.dimension(), expr.dimension());
        let mat_stride = qudit_core::memory::calc_mat_stride::<C>(expr.dimension(), expr.dimension(), col_stride);
        let compilable_unitary_unit = CompilableUnitaryUnit::new_with_grad(expr, col_stride, col_stride, mat_stride);
        self.exprs.push(compilable_unitary_unit);
        self
    }

    pub fn build(self) -> Module<C> {
        let module = Module::new(&self.name, self.diff_lvl);
        for expr in &self.exprs {
            let mut codegen = CodeGenerator::new(&module);
            codegen.gen_utry_func(&expr.expr, &expr.mat_idx_to_offset_map).expect("Error generating function.");

            if self.diff_lvl == DifferentiationLevel::Gradient {
                let grad_expr = expr.expr.differentiate();
                let (simple_expr_body, simple_grad_body) = simplify_matrix_and_matvec(&expr.expr.body, &grad_expr.body);

                let utry_expr_simple = UnitaryExpression {
                    name: expr.expr.name.clone(),
                    radices: expr.expr.radices.clone(),
                    variables: expr.expr.variables.clone(),
                    body: simple_expr_body,
                };

                let grad_expr_simple = MatVecExpression {
                    name: grad_expr.name.clone(),
                    variables: grad_expr.variables.clone(),
                    body: simple_grad_body,
                };

                if let Some(grad_idx_to_offset_map) = &expr.grad_idx_to_offset_map {
                    codegen.gen_utry_and_grad_func(&utry_expr_simple, &grad_expr_simple, &expr.mat_idx_to_offset_map, grad_idx_to_offset_map).expect("Error generating gradient function.");
                }
                // if let Some(grad_idx_to_offset_map) = &expr.grad_idx_to_offset_map {
                //     codegen.gen_utry_and_grad_func(&expr.expr, &grad_expr, &expr.mat_idx_to_offset_map, grad_idx_to_offset_map).expect("Error generating gradient function.");
                // }
            }
        }
        module
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use qudit_core::c64;
    use qudit_core::matrix::{Mat, MatVec};

    struct U3Gate;
    impl U3Gate {
        pub fn gen_expr(&self) -> UnitaryExpression {
            UnitaryExpression::new(
                String::from("
                utry U3(f1, f2, f3) {
                    [
                        [ cos(f1/2), ~e^(i*f3)*sin(f1/2) ],
                        [ e^(i*f2)*sin(f1/2), e^(i*(f2+f3))*cos(f1/2) ]
                    ]
                }
            "),
            )
        }
    }

    #[test]
    fn test_skeleton() {
        let u3_gate = U3Gate;
        let expr = u3_gate.gen_expr();

        for row in &expr.body {
            for expr in row {
                println!("{:?}", expr);
            }
        }

        let params = vec![1.7, 2.3, 3.1];
        let mut out_utry: Mat<c64> = Mat::zeros(2, 2);
        let mut out_grad: MatVec<c64> = MatVec::zeros(2, 2, 3);

        let module: Module<c64> = ModuleBuilder::new("test", DifferentiationLevel::Gradient)
            .add_expression_with_stride(expr, out_utry.col_stride().try_into().unwrap())
            .build();

        println!("{}", module);

        let u3_grad_combo_func = module.get_function_and_gradient("U3").unwrap();
        let out_ptr = out_utry.as_mut().as_ptr_mut() as *mut f64;
        let out_grad_ptr = out_grad.as_mut().as_mut_ptr().as_ptr() as *mut f64;

        let start = std::time::Instant::now();
        for _ in 0..1000000 {
            unsafe { u3_grad_combo_func.call(params.as_ptr(), out_ptr, out_grad_ptr); }
        }
        let duration = start.elapsed();
        println!("Time elapsed in expensive_function() is: {:?}", duration);
        println!("Average time: {:?}", duration / 1000000);

        println!("{:?}", out_utry);
        println!("{:?}", out_grad);
    }

    // #[test]
    // fn test_time() {
    //     let u3_gate = U3Gate;
    //     let expr = u3_gate.gen_expr();
    //     let mut out_utry: Mat<c64> = Mat::zeros(2, 2);
    //     let mut out_grad: MatVec<c64> = MatVec::zeros(2, 2, 3);

    //     let module = Module::new("test", vec![expr.clone()], vec![out_utry.col_stride().try_into().unwrap()], qudit_expr::DifferentiationLevel::Gradient, Some(vec![out_grad.col_stride().try_into().unwrap()]), Some(vec![out_grad.mat_stride().try_into().unwrap()]), None, None);
    //     let context = module.jit::<c64>();
    //     let utry_and_grad_func = context.get_utry_and_grad_func(&expr.expr.name);
    
    //     let out_ptr = unsafe { qudit_core::matrix::matmut_to_ptr(out_utry.as_mut()) };
    //     let out_grad_ptr = unsafe { qudit_core::matrix::matvecmut_to_ptr(out_grad.as_mut()) };
    //     for i in 0..100 {
    //         unsafe { utry_and_grad_func.call(params[i].as_ptr(), out_ptr, out_grad_ptr); }
    //     }
    // }
}

// impl QGLModule {
//     pub fn new(
//         name: &str,
//         exprs: Vec<UnitaryExpression>,
//         col_strides: Vec<usize>,
//         diff_lvl: DifferentiationLevel,
//         grad_col_strides: Option<Vec<usize>>,
//         grad_mat_strides: Option<Vec<usize>>,
//         hess_col_strides: Option<Vec<usize>>,
//         hess_mat_strides: Option<Vec<usize>>,
//     ) -> Self {
//         let context = Context::create();
//         let mut grad_exprs = None;
//         let mut hess_exprs = None;
//         match diff_lvl {
//             DifferentiationLevel::None => (),
//             DifferentiationLevel::Gradient => {
//                 grad_exprs = Some(exprs.iter().map(|expr| auto_diff(&expr.expr)).collect());
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
