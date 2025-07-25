use inkwell::builder::Builder;
use inkwell::values::FloatValue;
use inkwell::values::FunctionValue;
use inkwell::AddressSpace;

use coe::is_same;
use qudit_core::c32;
use qudit_core::c64;
use std::collections::HashMap;
use qudit_core::ComplexScalar;

use crate::codegen::process_name_for_gen;
use crate::complex::ComplexExpression;
use crate::expression::Expression;
use crate::unitary::MatVecExpression;
use crate::UnitaryExpression;

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
pub struct CodeGenerator<'ctx, C: ComplexScalar> {
    pub context: &'ctx Module<C>,
    pub builder: Builder<'ctx>,

    variables: HashMap<String, FloatValue<'ctx>>,
    expressions: HashMap<String, FloatValue<'ctx>>,
    functions: HashMap<String, FunctionValue<'ctx>>,
    fn_value_opt: Option<FunctionValue<'ctx>>,
    output_ptr_idx: Option<u32>,
    phantom: std::marker::PhantomData<C>,
}

impl<'ctx, C: ComplexScalar> CodeGenerator<'ctx, C> {
    pub fn new(context: &'ctx Module<C>) -> Self {
        let builder = context.context().create_builder();
        CodeGenerator {
            context,
            builder,
            variables: HashMap::new(),
            functions: HashMap::new(),
            expressions: HashMap::new(),
            fn_value_opt: None,
            output_ptr_idx: None,
            phantom: std::marker::PhantomData,
        }
    }

    fn int_type(&self) -> inkwell::types::IntType<'ctx> { 
        if is_same::<C, c32>() {
            self.context.context().i32_type()
        } else if is_same::<C, c64>() {
            self.context.context().i64_type()
        } else {
            panic!("Unknown bit width");
        }
    }

    fn float_type(&self) -> inkwell::types::FloatType<'ctx> {
        if is_same::<C, c32>() {
            self.context.context().f32_type()
        } else if is_same::<C, c64>() {
            self.context.context().f64_type()
        } else {
            panic!("Unknown bit width");
        }
    }

    fn gen_utry_func_proto(&self, name: &str) -> CodeGenResult<FunctionValue<'ctx>> {
        let ret_type = self.context.context().void_type();
        let ptr_type = self.context.context().ptr_type(AddressSpace::default());
        let param_types = vec![ptr_type.into(), ptr_type.into()];
        let func_type = ret_type.fn_type(&param_types, false);
        let func = self.context.module().add_function(name, func_type, None);
        Ok(func)
    }

    fn gen_utry_and_grad_func_proto(&self, name: &str) -> CodeGenResult<FunctionValue<'ctx>> {
        let ret_type = self.context.context().void_type();
        let ptr_type = self.context.context().ptr_type(AddressSpace::default());
        let param_types = vec![ptr_type.into(), ptr_type.into(), ptr_type.into()];
        let func_type = ret_type.fn_type(&param_types, false);
        let func = self.context.module().add_function(name, func_type, None);
        Ok(func)
    }

    fn build_expression(&mut self, expr: &Expression) -> CodeGenResult<FloatValue<'ctx>> {
        let expr_str = expr.to_string();
        let cached = self.expressions.get(&expr_str);
        if let Some(c) = cached {
            return Ok(c.clone());
        }

        let val = match expr {
            Expression::Pi => Ok(self.float_type().const_float(std::f64::consts::PI)),
            Expression::Constant(_) => Ok(self.float_type().const_float(expr.to_float())),
            Expression::Variable(name) => {
                self.variables.get(name).ok_or(CodeGenError::new(&format!("Variable {} not found", name))).copied()
            },
            Expression::Neg(expr) => {
                let val = self.build_expression(expr)?;
                Ok(self.builder.build_float_neg(val, "tmp").unwrap())
            },
            Expression::Add(lhs, rhs) => {
                let lhs_val = self.build_expression(lhs)?;
                let rhs_val = self.build_expression(rhs)?;
                Ok(self.builder.build_float_add(lhs_val, rhs_val, "tmp").unwrap())
            },
            Expression::Sub(lhs, rhs) => {
                let lhs_val = self.build_expression(lhs)?;
                let rhs_val = self.build_expression(rhs)?;
                Ok(self.builder.build_float_sub(lhs_val, rhs_val, "tmp").unwrap())
            },
            Expression::Mul(lhs, rhs) => {
                let lhs_val = self.build_expression(lhs)?;
                let rhs_val = self.build_expression(rhs)?;
                Ok(self.builder.build_float_mul(lhs_val, rhs_val, "tmp").unwrap())
            },
            Expression::Div(lhs, rhs) => {
                let lhs_val = self.build_expression(lhs)?;
                let rhs_val = self.build_expression(rhs)?;
                Ok(self.builder.build_float_div(lhs_val, rhs_val, "tmp").unwrap())
            },
            Expression::Pow(base, exponent) => {
                let base_val = self.build_expression(base)?;
                let exponent_val = self.build_expression(exponent)?;
                let pow = self.get_builtin("pow");
                let args = [base_val.into(), exponent_val.into()];
                let val = self.builder.build_call(pow, &args, "tmp").unwrap().try_as_basic_value().left().unwrap().into_float_value();
                Ok(val)
            },
            Expression::Sqrt(expr) => {
                let arg = self.build_expression(expr)?;
                let sqrt = self.get_builtin("sqrt");
                let val = self.builder.build_call(sqrt, &[arg.into()], "tmp").unwrap().try_as_basic_value().left().unwrap().into_float_value();
                Ok(val)
            },
            Expression::Sin(expr) => {
                let arg = self.build_expression(expr)?;
                let sin = self.get_builtin("sin");
                let val = self.builder.build_call(sin, &[arg.into()], "tmp").unwrap().try_as_basic_value().left().unwrap().into_float_value();
                Ok(val)
            },
            Expression::Cos(expr) => {
                let arg = self.build_expression(expr)?;
                let cos = self.get_builtin("cos");
                let val = self.builder.build_call(cos, &[arg.into()], "tmp").unwrap().try_as_basic_value().left().unwrap().into_float_value();
                Ok(val)
            },
        };
        
        if let Ok(val) = val {
            self.expressions.insert(expr_str, val);
        }
        val
    }

    pub fn compile_expr(&mut self, expr: &ComplexExpression, re_offset: usize, need_to_write_real_zero: bool) -> CodeGenResult<()> {
        let re_offset: u64 = re_offset as u64;
        let ptr_idx = self.output_ptr_idx.to_owned().expect("Output pointer index not set");
        let ptr = self.fn_value_opt.unwrap().get_nth_param(ptr_idx).unwrap().into_pointer_value();

        if need_to_write_real_zero || !expr.real.is_zero_fast() {
            let val = self.build_expression(&expr.real)?;
            let offset = self.int_type().const_int(re_offset, false);
            let offset_ptr = unsafe {
                self.builder.build_gep(
                    self.float_type(),
                    ptr,
                    &[offset],
                    "offset_ptr"
                ).unwrap()
            };

            match self.builder.build_store(offset_ptr, val) {
                Ok(_) => {},
                Err(e) => { return Err(CodeGenError::new(&format!("Error storing value: {}", e))); }
            };
        }

        if !expr.imag.is_zero_fast() {
            let val = self.build_expression(&expr.imag)?;
            let offset = self.int_type().const_int(re_offset + 1, false);
            let offset_ptr = unsafe {
                self.builder.build_gep(
                    self.float_type(),
                    ptr,
                    &[offset],
                    "offset_ptr"
                ).unwrap()
            };

            match self.builder.build_store(offset_ptr, val) {
                Ok(_) => {},
                Err(e) => { return Err(CodeGenError::new(&format!("Error storing value: {}", e))); }
            };
        }

        Ok(())
    }

    fn build_var_table(&mut self, variables: &HashMap<String, usize>) {
        self.variables.clear();

        for (var, offset) in variables.iter() {
            let ptr = self.fn_value_opt.unwrap().get_nth_param(0).unwrap().into_pointer_value();
            let offset_ptr = match offset {
                0 => ptr,
                _ => unsafe {
                    self.builder.build_gep(
                        self.float_type(),
                        ptr,
                        &[self.int_type().const_int(*offset as u64, false)],
                        "offset_ptr"
                    ).unwrap()
                }
            };

            let val = self.builder.build_load(self.float_type(), offset_ptr, "tmp").unwrap().into_float_value();
            self.variables.insert(var.to_string(), val);
        }
    }

    fn build_var_map(&mut self, variables: &[String]) {
        self.variables.clear();

        for (offset, var) in variables.iter().enumerate() {
            let ptr = self.fn_value_opt.unwrap().get_nth_param(0).unwrap().into_pointer_value();
            let offset_ptr = match offset {
                0 => ptr,
                _ => unsafe {
                    self.builder.build_gep(
                        self.float_type(),
                        ptr,
                        &[self.int_type().const_int(offset as u64, false)],
                        "offset_ptr"
                    ).unwrap()
                }
            };

            let val = self.builder.build_load(self.float_type(), offset_ptr, "tmp").unwrap().into_float_value();
            self.variables.insert(var.to_string(), val);
        }
    }

    fn get_builtin(&mut self, name: &str) -> FunctionValue<'ctx> {
        if let Some(f) = self.functions.get(name) {
            return f.clone();
        }

        let b = match Builtins::from_str(name) {
            Some(b) => b,
            None => { panic!("Unsupported builtin function: {}", name); }
        };

        let intr = match b.intrinsic() {
            Some(i) => i,
            None => { panic!("Unsupported builtin function: {}", name); }
        };

        let decl = intr.get_declaration(&self.context.module(), &[self.float_type().into()]);

        let fn_value = match decl {
            Some(f) => f,
            None => { panic!("Unsupported builtin function: {}", name); }
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


    pub fn gen_utry_func(
        &mut self,
        utry: &UnitaryExpression,
        mat_idx_to_offset_map: &Box<dyn Fn(usize, usize) -> usize>
    ) -> CodeGenResult<()> {
        let name = process_name_for_gen(&utry.name);
        self.expressions.clear();  
        let func = self.gen_utry_func_proto(&name)?;
        let entry = self.context.context().append_basic_block(func, "entry");
        self.builder.position_at_end(entry);
        self.fn_value_opt = Some(func);
        self.output_ptr_idx = Some(1);
        self.build_var_map(&utry.variables);

        for (i, row) in utry.body.iter().enumerate()
        {
            for (j, elem) in row.iter().enumerate()
            {
                if i == j && elem.is_one_fast() {
                    continue;
                }
                if i != j && elem.is_zero_fast() {
                    continue;
                }
                self.compile_expr(elem, mat_idx_to_offset_map(i, j), i == j)?;
            }
        }

        match self.builder.build_return(None) {
            Ok(_) => Ok(()),
            Err(e) => Err(CodeGenError::new(&format!("Error building return: {}", e))),
        }
    }

    pub fn gen_func(
        &mut self,
        fn_name: &str,
        fn_expr: &[Expression],
        var_table: &HashMap<String, usize>,
        expr_idx_to_offset_map: &Box<dyn Fn(usize) -> usize>,
    ) -> CodeGenResult<()> {
        self.expressions.clear();
        let func = self.gen_utry_func_proto(fn_name)?;
        let entry = self.context.context().append_basic_block(func, "entry");
        self.builder.position_at_end(entry);
        self.fn_value_opt = Some(func);
        // println!("name: {:?}, var_table: {:?}", fn_name, var_table);
        self.build_var_table(var_table);

        let output_ptr = self.fn_value_opt.unwrap().get_nth_param(1).unwrap().into_pointer_value();

        for (i, expr) in fn_expr.iter().enumerate() {
            if expr.is_zero_fast() {
                continue;
            }

            let val = self.build_expression(&expr)?;
            let offset = self.int_type().const_int(expr_idx_to_offset_map(i) as u64, false);
            let offset_ptr = unsafe {
                self.builder.build_gep(
                    self.float_type(),
                    output_ptr,
                    &[offset],
                    "offset_ptr"
                ).unwrap()
            };

            match self.builder.build_store(offset_ptr, val) {
                Ok(_) => {},
                Err(e) => { return Err(CodeGenError::new(&format!("Error storing value: {}", e))); }
            };
        }

        match self.builder.build_return(None) {
            Ok(_) => Ok(()),
            Err(e) => Err(CodeGenError::new(&format!("Error building return: {}", e))),
        }
    }

    pub fn gen_utry_and_grad_func(
        &mut self,
        utry: &UnitaryExpression,
        grad: &MatVecExpression,
        mat_idx_to_offset_map: &Box<dyn Fn(usize, usize) -> usize>,
        grad_idx_to_offset_map: &Box<dyn Fn(usize, usize, usize) -> usize>
    ) -> CodeGenResult<()> {
        let name = process_name_for_gen(&utry.name) + "_grad";
        self.expressions.clear();
        let func = self.gen_utry_and_grad_func_proto(&name)?; 
        let entry = self.context.context().append_basic_block(func, "entry");
        self.builder.position_at_end(entry);
        self.fn_value_opt = Some(func);
        self.output_ptr_idx = Some(1);
        self.build_var_map(&utry.variables);

        
        for (i, row) in utry.body.iter().enumerate()
        {
            for (j, elem) in row.iter().enumerate()
            {
                if i == j && elem.is_one_fast() {
                    continue;
                }
                if i != j && elem.is_zero_fast() {
                    continue;
                }
                self.compile_expr(elem, mat_idx_to_offset_map(i, j), i == j)?;
            }
        }

        self.output_ptr_idx = Some(2);

        for (m, partial) in grad.body.iter().enumerate()
        {
            for (i, row) in partial.iter().enumerate()
            {
                for (j, elem) in row.iter().enumerate()
                {
                    if elem.is_zero_fast() {
                        continue;
                    }
                    self.compile_expr(elem, grad_idx_to_offset_map(m, i, j), false)?;
                }
            }
        }

        match self.builder.build_return(None) {
            Ok(_) => Ok(()),
            Err(e) => Err(CodeGenError::new(&format!("Error building return: {}", e))),
        }
    }
}
