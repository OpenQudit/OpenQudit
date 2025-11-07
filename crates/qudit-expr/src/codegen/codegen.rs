use inkwell::AddressSpace;
use inkwell::builder::Builder;
use inkwell::values::FloatValue;
use inkwell::values::FunctionValue;

use coe::is_same;
use qudit_core::RealScalar;
use std::collections::HashMap;

use crate::ComplexExpression;
use crate::Expression;

use super::builtins::Builtins;
use super::module::Module;

#[derive(Debug)]
pub struct CodeGenError {
    pub message: String,
}

impl CodeGenError {
    pub fn new(message: &str) -> Self {
        CodeGenError {
            message: message.to_string(),
        }
    }
}

type CodeGenResult<T> = Result<T, CodeGenError>;

#[derive(Debug)]
pub struct CodeGenerator<'ctx, R: RealScalar> {
    pub context: &'ctx Module<R>,
    pub builder: Builder<'ctx>,

    variables: HashMap<String, FloatValue<'ctx>>,
    expressions: HashMap<String, FloatValue<'ctx>>,
    functions: HashMap<String, FunctionValue<'ctx>>,
    fn_value_opt: Option<FunctionValue<'ctx>>,
    output_ptr_idx: Option<u32>,
}

impl<'ctx, R: RealScalar> CodeGenerator<'ctx, R> {
    pub fn new(context: &'ctx Module<R>) -> Self {
        let builder = context.context().create_builder();
        CodeGenerator {
            context,
            builder,
            variables: HashMap::new(),
            functions: HashMap::new(),
            expressions: HashMap::new(),
            fn_value_opt: None,
            output_ptr_idx: None,
        }
    }

    fn int_type(&self) -> inkwell::types::IntType<'ctx> {
        if is_same::<R, f32>() {
            self.context.context().i32_type()
        } else if is_same::<R, f64>() {
            self.context.context().i64_type()
        } else {
            panic!("Unknown bit width");
        }
    }

    fn float_type(&self) -> inkwell::types::FloatType<'ctx> {
        if is_same::<R, f32>() {
            self.context.context().f32_type()
        } else if is_same::<R, f64>() {
            self.context.context().f64_type()
        } else {
            panic!("Unknown bit width");
        }
    }

    fn gen_write_func_proto(&self, name: &str) -> CodeGenResult<FunctionValue<'ctx>> {
        let ret_type = self.context.context().void_type();
        let ptr_type = self.context.context().ptr_type(AddressSpace::default());
        // Match Rust WriteFunc<R> signature: (*const R, *mut R, *const u64, *const u64, u64, *const bool)
        let param_types = vec![
            ptr_type.into(),                          // *const R
            ptr_type.into(),                          // *mut R
            ptr_type.into(),                          // *const u64
            ptr_type.into(),                          // *const u64
            self.context.context().i64_type().into(), // u64
            ptr_type.into(),                          // *const bool
        ];
        let func_type = ret_type.fn_type(&param_types, false);
        let func = self
            .context
            .with_module(|module| module.add_function(name, func_type, None));
        Ok(func)
    }

    fn build_expression(&mut self, expr: &Expression) -> CodeGenResult<FloatValue<'ctx>> {
        let expr_str = expr.to_string();
        let cached = self.expressions.get(&expr_str);
        if let Some(c) = cached {
            return Ok(*c);
        }

        let val = match expr {
            Expression::Pi => Ok(self.float_type().const_float(std::f64::consts::PI)),
            Expression::Constant(_) => Ok(self.float_type().const_float(expr.to_float())),
            Expression::Variable(name) => self
                .variables
                .get(name)
                .ok_or(CodeGenError::new(&format!("Variable {} not found", name)))
                .copied(),
            Expression::Neg(expr) => {
                let val = self.build_expression(expr)?;
                Ok(self.builder.build_float_neg(val, "tmp").unwrap())
            }
            Expression::Add(lhs, rhs) => {
                let lhs_val = self.build_expression(lhs)?;
                let rhs_val = self.build_expression(rhs)?;
                Ok(self
                    .builder
                    .build_float_add(lhs_val, rhs_val, "tmp")
                    .unwrap())
            }
            Expression::Sub(lhs, rhs) => {
                let lhs_val = self.build_expression(lhs)?;
                let rhs_val = self.build_expression(rhs)?;
                Ok(self
                    .builder
                    .build_float_sub(lhs_val, rhs_val, "tmp")
                    .unwrap())
            }
            Expression::Mul(lhs, rhs) => {
                let lhs_val = self.build_expression(lhs)?;
                let rhs_val = self.build_expression(rhs)?;
                Ok(self
                    .builder
                    .build_float_mul(lhs_val, rhs_val, "tmp")
                    .unwrap())
            }
            Expression::Div(lhs, rhs) => {
                let lhs_val = self.build_expression(lhs)?;
                let rhs_val = self.build_expression(rhs)?;
                Ok(self
                    .builder
                    .build_float_div(lhs_val, rhs_val, "tmp")
                    .unwrap())
            }
            Expression::Pow(base, exponent) => {
                let base_val = self.build_expression(base)?;
                let exponent_val = self.build_expression(exponent)?;
                let pow = self.get_builtin("pow");
                let args = [base_val.into(), exponent_val.into()];
                let val = self
                    .builder
                    .build_call(pow, &args, "tmp")
                    .unwrap()
                    .try_as_basic_value()
                    .left()
                    .unwrap()
                    .into_float_value();
                Ok(val)
            }
            Expression::Sqrt(expr) => {
                let arg = self.build_expression(expr)?;
                let sqrt = self.get_builtin("sqrt");
                let val = self
                    .builder
                    .build_call(sqrt, &[arg.into()], "tmp")
                    .unwrap()
                    .try_as_basic_value()
                    .left()
                    .unwrap()
                    .into_float_value();
                Ok(val)
            }
            Expression::Sin(expr) => {
                let arg = self.build_expression(expr)?;
                let sin = self.get_builtin("sin");
                let val = self
                    .builder
                    .build_call(sin, &[arg.into()], "tmp")
                    .unwrap()
                    .try_as_basic_value()
                    .left()
                    .unwrap()
                    .into_float_value();
                Ok(val)
            }
            Expression::Cos(expr) => {
                let arg = self.build_expression(expr)?;
                let cos = self.get_builtin("cos");
                let val = self
                    .builder
                    .build_call(cos, &[arg.into()], "tmp")
                    .unwrap()
                    .try_as_basic_value()
                    .left()
                    .unwrap()
                    .into_float_value();
                Ok(val)
            }
        };

        if let Ok(val) = val {
            self.expressions.insert(expr_str, val);
        }
        val
    }

    pub fn compile_expr(
        &mut self,
        expr: &ComplexExpression,
        re_offset: usize,
        need_to_write_real_zero: bool,
    ) -> CodeGenResult<()> {
        let re_offset: u64 = re_offset as u64;
        let ptr_idx = self
            .output_ptr_idx
            .to_owned()
            .expect("Output pointer index not set");
        let ptr = self
            .fn_value_opt
            .unwrap()
            .get_nth_param(ptr_idx)
            .unwrap()
            .into_pointer_value();

        if need_to_write_real_zero || !expr.real.is_zero_fast() {
            let val = self.build_expression(&expr.real)?;
            let offset = self.int_type().const_int(re_offset, false);
            let offset_ptr = unsafe {
                self.builder
                    .build_gep(self.float_type(), ptr, &[offset], "offset_ptr")
                    .unwrap()
            };

            match self.builder.build_store(offset_ptr, val) {
                Ok(_) => {}
                Err(e) => {
                    return Err(CodeGenError::new(&format!("Error storing value: {}", e)));
                }
            };
        }

        if !expr.imag.is_zero_fast() {
            let val = self.build_expression(&expr.imag)?;
            let offset = self.int_type().const_int(re_offset + 1, false);
            let offset_ptr = unsafe {
                self.builder
                    .build_gep(self.float_type(), ptr, &[offset], "offset_ptr")
                    .unwrap()
            };

            match self.builder.build_store(offset_ptr, val) {
                Ok(_) => {}
                Err(e) => {
                    return Err(CodeGenError::new(&format!("Error storing value: {}", e)));
                }
            };
        }

        Ok(())
    }

    fn build_var_table(&mut self, variables: &[String]) {
        self.variables.clear();
        let params_ptr = self
            .fn_value_opt
            .unwrap()
            .get_nth_param(0)
            .unwrap()
            .into_pointer_value();
        let param_offset_ptr = self
            .fn_value_opt
            .unwrap()
            .get_nth_param(2)
            .unwrap()
            .into_pointer_value();

        for (map_idx, var_name) in variables.iter().enumerate() {
            // Load the actual offset from param_offset_ptr using map_idx
            // param_offset_ptr points to u64 values according to Rust signature
            let map_idx_val = self
                .context
                .context()
                .i64_type()
                .const_int(map_idx as u64, false);
            let actual_offset_ptr = unsafe {
                self.builder
                    .build_gep(
                        self.context.context().i64_type(),
                        param_offset_ptr,
                        &[map_idx_val],
                        "actual_offset_ptr_gep",
                    )
                    .unwrap()
            };
            let actual_offset_val = self
                .builder
                .build_load(
                    self.context.context().i64_type(),
                    actual_offset_ptr,
                    "actual_offset_val",
                )
                .unwrap()
                .into_int_value();

            // Use the actual_offset_val to index into params_ptr
            let var_ptr = unsafe {
                self.builder
                    .build_gep(
                        self.float_type(),
                        params_ptr,
                        &[actual_offset_val],
                        "var_ptr_gep",
                    )
                    .unwrap()
            };

            let val = self
                .builder
                .build_load(self.float_type(), var_ptr, var_name)
                .unwrap()
                .into_float_value();
            self.variables.insert(var_name.to_owned(), val);
        }
    }

    fn get_builtin(&mut self, name: &str) -> FunctionValue<'ctx> {
        if let Some(f) = self.functions.get(name) {
            return *f;
        }

        let b = match Builtins::from_str(name) {
            Some(b) => b,
            None => {
                panic!("Unsupported builtin function: {}", name);
            }
        };

        let intr = match b.intrinsic() {
            Some(i) => i,
            None => {
                panic!("Unsupported builtin function: {}", name);
            }
        };

        let decl = self
            .context
            .with_module(|module| intr.get_declaration(&module, &[self.float_type().into()]));

        let fn_value = match decl {
            Some(f) => f,
            None => {
                panic!("Unsupported builtin function: {}", name);
            }
        };

        self.functions.insert(name.to_string(), fn_value);
        fn_value
    }

    // fn get_expression(&mut self, name: &str) -> Option<FloatValue<'ctx>> {
    //     if let Some(c) = self.expressions.get(name) {
    //         return Some(c.clone());
    //     }

    //     let c = match name {
    //         "pi" => Some(self.float_type().const_float(std::f64::consts::PI)),
    //         "π" => Some(self.float_type().const_float(std::f64::consts::PI)),
    //         "e" => Some(self.float_type().const_float(std::f64::consts::E)),
    //         _ => None
    //     };

    //     if let Some(c) = c {
    //         self.expressions.insert(name.to_string(), c);
    //         return Some(c);
    //     }

    //     None
    // }

    pub fn gen_func(
        &mut self,
        fn_name: &str,
        fn_expr: &[Expression],
        var_table: &[String],
        fn_len: usize,
    ) -> CodeGenResult<()> {
        self.expressions.clear();
        let func = self.gen_write_func_proto(fn_name)?;
        let entry = self.context.context().append_basic_block(func, "entry");
        self.builder.position_at_end(entry);
        self.fn_value_opt = Some(func);
        // println!("name: {:?}, var_table: {:?}", fn_name, var_table);
        self.build_var_table(var_table);

        let output_ptr = self
            .fn_value_opt
            .unwrap()
            .get_nth_param(1)
            .unwrap()
            .into_pointer_value();
        let output_map_ptr = self
            .fn_value_opt
            .unwrap()
            .get_nth_param(3)
            .unwrap()
            .into_pointer_value();
        let fn_unit_offset = self
            .fn_value_opt
            .unwrap()
            .get_nth_param(4)
            .unwrap()
            .into_int_value();
        let const_param_ptr = self
            .fn_value_opt
            .unwrap()
            .get_nth_param(5)
            .unwrap()
            .into_pointer_value();
        let int_increment = self.context.context().i64_type().const_int(1u64, false);
        let mut dyn_fn_unit_idx = self.context.context().i64_type().const_int(0u64, false);

        for (fn_unit_idx, fn_unit_exprs) in fn_expr.chunks(fn_len).enumerate() {
            let current_fn = self.fn_value_opt.unwrap();
            let compute_block = self
                .context
                .context()
                .append_basic_block(current_fn, &format!("compute_unit_{}", fn_unit_idx));
            let skip_block = self
                .context
                .context()
                .append_basic_block(current_fn, &format!("skip_unit_{}", fn_unit_idx));

            if fn_unit_idx == 0 {
                // First function unit is the function, always compute.
                self.builder
                    .build_unconditional_branch(compute_block)
                    .unwrap();
            } else if fn_unit_idx <= var_table.len() {
                // Next var_table.len() function units are the partials in the gradient.
                // Check if the corresponding parameter is constant.
                let param_idx_for_const_check = self
                    .context
                    .context()
                    .i64_type()
                    .const_int((fn_unit_idx - 1) as u64, false);
                let const_param_elem_ptr = unsafe {
                    self.builder
                        .build_gep(
                            self.context.context().i8_type(),
                            const_param_ptr,
                            &[param_idx_for_const_check],
                            "const_param_elem_ptr",
                        )
                        .unwrap()
                };
                let is_constant_val = self
                    .builder
                    .build_load(
                        self.context.context().i8_type(),
                        const_param_elem_ptr,
                        "is_constant",
                    )
                    .unwrap()
                    .into_int_value();

                let is_constant_true = self
                    .builder
                    .build_int_compare(
                        inkwell::IntPredicate::EQ,
                        is_constant_val,
                        self.context.context().i8_type().const_int(1, false),
                        "is_constant_true",
                    )
                    .unwrap();

                // If is_constant_true, branch to skip_block. Else, branch to compute_block.
                self.builder
                    .build_conditional_branch(is_constant_true, skip_block, compute_block)
                    .unwrap();
            } else {
                // Last are the hessian units

                let index = fn_unit_idx - var_table.len() - 1;
                let param_j = ((((8 * index + 1) as f64).sqrt().floor() as usize) - 1) / 2;
                let param_i = index - param_j * (param_j + 1) / 2;

                // Check if the corresponding parameter is constant.
                let param_idx_i_for_const_check = self
                    .context
                    .context()
                    .i64_type()
                    .const_int((param_i - 1) as u64, false);
                let const_param_elem_ptr = unsafe {
                    self.builder
                        .build_gep(
                            self.context.context().i8_type(),
                            const_param_ptr,
                            &[param_idx_i_for_const_check],
                            "const_param_i_elem_ptr",
                        )
                        .unwrap()
                };
                let is_i_constant_val = self
                    .builder
                    .build_load(
                        self.context.context().i8_type(),
                        const_param_elem_ptr,
                        "is_i_constant",
                    )
                    .unwrap()
                    .into_int_value();

                let is_i_constant_true = self
                    .builder
                    .build_int_compare(
                        inkwell::IntPredicate::EQ,
                        is_i_constant_val,
                        self.context.context().i8_type().const_int(1, false),
                        "is_i_constant_true",
                    )
                    .unwrap();

                // Check if the corresponding parameter is constant.
                let param_idx_j_for_const_check = self
                    .context
                    .context()
                    .i64_type()
                    .const_int((param_j - 1) as u64, false);
                let const_param_elem_ptr = unsafe {
                    self.builder
                        .build_gep(
                            self.context.context().i8_type(),
                            const_param_ptr,
                            &[param_idx_j_for_const_check],
                            "const_param_j_elem_ptr",
                        )
                        .unwrap()
                };
                let is_j_constant_val = self
                    .builder
                    .build_load(
                        self.context.context().i8_type(),
                        const_param_elem_ptr,
                        "is_j_constant",
                    )
                    .unwrap()
                    .into_int_value();

                let is_j_constant_true = self
                    .builder
                    .build_int_compare(
                        inkwell::IntPredicate::EQ,
                        is_j_constant_val,
                        self.context.context().i8_type().const_int(1, false),
                        "is_j_constant_true",
                    )
                    .unwrap();

                let is_either_constant = self
                    .builder
                    .build_or(is_i_constant_true, is_j_constant_true, "is_either_constant")
                    .unwrap();

                // If is_constant_true, branch to skip_block. Else, branch to compute_block.
                self.builder
                    .build_conditional_branch(is_either_constant, skip_block, compute_block)
                    .unwrap();
            }

            // Position builder at the compute block to emit the actual computation logic
            self.builder.position_at_end(compute_block);

            let this_unit_offset = self
                .builder
                .build_int_mul(fn_unit_offset, dyn_fn_unit_idx, "this_unit_offset")
                .unwrap();
            dyn_fn_unit_idx = self
                .builder
                .build_int_add(dyn_fn_unit_idx, int_increment, "dyn_fn_unit_idx")
                .unwrap();

            for (i, expr) in fn_unit_exprs.iter().enumerate() {
                if expr.is_zero_fast() {
                    continue;
                }

                let val = self.build_expression(expr)?;
                let offset_ptr = unsafe {
                    let output_idx = self.context.context().i64_type().const_int(i as u64, false);
                    let output_map_elem_ptr = self
                        .builder
                        .build_gep(
                            self.context.context().i64_type(),
                            output_map_ptr,
                            &[output_idx],
                            "output_map_elem_ptr",
                        )
                        .unwrap();
                    let output_map_offset = self
                        .builder
                        .build_load(
                            self.context.context().i64_type(),
                            output_map_elem_ptr,
                            "output_map_offset",
                        )
                        .unwrap()
                        .into_int_value();
                    let combined_offset = self
                        .builder
                        .build_int_add(output_map_offset, this_unit_offset, "combined_offset")
                        .unwrap();
                    self.builder
                        .build_gep(
                            self.float_type(),
                            output_ptr,
                            &[combined_offset],
                            "offset_ptr",
                        )
                        .unwrap()
                };

                match self.builder.build_store(offset_ptr, val) {
                    Ok(_) => {}
                    Err(e) => {
                        return Err(CodeGenError::new(&format!("Error storing value: {}", e)));
                    }
                };
            }

            // After computation (if compute_block was taken), branch to the skip_block
            // to ensure control flow continues to the next iteration of the loop.
            self.builder.build_unconditional_branch(skip_block).unwrap();

            // Position builder at the skip block for the next iteration.
            self.builder.position_at_end(skip_block);
        }

        match self.builder.build_return(None) {
            Ok(_) => Ok(()),
            Err(e) => Err(CodeGenError::new(&format!("Error building return: {}", e))),
        }
    }
}
