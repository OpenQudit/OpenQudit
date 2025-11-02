use super::{codegen::CodeGenerator, module::Module};
use qudit_core::RealScalar;

use crate::Expression;

pub struct CompilableUnit<'a> {
    pub fn_name: String,
    pub exprs: &'a [Expression],
    pub variables: Vec<String>,
    pub unit_size: usize,
}

impl<'a> CompilableUnit<'a> {
    pub fn new(
        name: &str,
        exprs: &'a [Expression],
        variables: Vec<String>,
        unit_size: usize,
    ) -> Self {
        CompilableUnit {
            fn_name: name.to_string(),
            exprs,
            variables,
            unit_size,
        }
    }

    pub fn add_to_module<R: RealScalar>(&self, module: &Module<R>) {
        // println!("Adding fn_name: {} to module.", self.fn_name);
        // for expr in &self.exprs {
        //     println!("{:?}", expr);
        // }
        let mut codegen = CodeGenerator::new(&module);
        codegen
            .gen_func(&self.fn_name, &self.exprs, &self.variables, self.unit_size)
            .expect("Error generating function.");
    }
}

pub type DifferentiationLevel = usize;
pub const FUNCTION: DifferentiationLevel = 1;
pub const GRADIENT: DifferentiationLevel = 2;
pub const HESSIAN: DifferentiationLevel = 3;

// pub trait DifferentiationLevel {}

// struct Function {}
// impl DifferentiationLevel for Function {}
// struct Gradient {}
// impl DifferentiationLevel for Gradient {}
// struct Hessian {}
// impl DifferentiationLevel for Hessian {}

// impl DifferentiationLevel {
//     pub fn gradient_capable(&self) -> bool {
//         match self {
//             DifferentiationLevel::None => false,
//             DifferentiationLevel::Gradient => true,
//             DifferentiationLevel::Hessian => true,
//         }
//     }

//     pub fn hessian_capable(&self) -> bool {
//         match self {
//             DifferentiationLevel::None => false,
//             DifferentiationLevel::Gradient => false,
//             DifferentiationLevel::Hessian => true,
//         }
//     }
// }

pub struct ModuleBuilder<'a, R: RealScalar> {
    name: String,
    exprs: Vec<CompilableUnit<'a>>,
    _phantom_c: std::marker::PhantomData<R>,
}

impl<'a, R: RealScalar> ModuleBuilder<'a, R> {
    pub fn new(name: &str) -> Self {
        ModuleBuilder {
            name: name.to_string(),
            exprs: Vec::new(),
            _phantom_c: std::marker::PhantomData,
        }
    }

    pub fn add_unit(mut self, unit: CompilableUnit<'a>) -> Self {
        self.exprs.push(unit);
        self
    }

    pub fn build(self) -> Module<R> {
        let module = Module::new(&self.name);
        for expr in &self.exprs {
            expr.add_to_module(&module);
        }
        module
    }
}
