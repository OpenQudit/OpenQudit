use inkwell::execution_engine::JitFunction;
use inkwell::{context::Context, execution_engine::ExecutionEngine};
use qudit_core::ComplexScalar;

use std::borrow::Cow;
use std::ffi::{CStr, CString};
use std::mem::{ManuallyDrop, MaybeUninit};
use std::rc::Rc;

use inkwell::targets::{InitializationConfig, Target};
use llvm_sys::core::{LLVMContextCreate, LLVMModuleCreateWithNameInContext};
use llvm_sys::execution_engine::{LLVMCreateJITCompilerForModule, LLVMDisposeExecutionEngine, LLVMExecutionEngineRef};
use llvm_sys::prelude::LLVMModuleRef;

use inkwell::module::Module as InkwellModule;

use super::builder::DifferentiationLevel;
use super::{process_name_for_gen, UtryGradFunc, UtryFunc};

pub(crate) fn to_c_str(mut s: &str) -> Cow<'_, CStr> {
    if s.is_empty() {
        s = "\0";
    }

    // Start from the end of the string as it's the most likely place to find a null byte
    if !s.chars().rev().any(|ch| ch == '\0') {
        return Cow::from(CString::new(s).expect("unreachable since null bytes are checked"));
    }

    unsafe { Cow::from(CStr::from_ptr(s.as_ptr() as *const _)) }
}

fn convert_c_string(c_str: *mut i8) -> String {
    // Safety: Ensure that c_str is not null and points to a valid null-terminated string.
    assert!(!c_str.is_null());

    // Convert the raw pointer to a CStr, which will handle the null termination.
    let c_str = unsafe { CStr::from_ptr(c_str) };

    // Convert CStr to String
    c_str.to_string_lossy().into_owned()
}

#[derive(Debug)]
pub struct Module<C: ComplexScalar> {
    engine: Rc<LLVMExecutionEngineRef>,
    module: LLVMModuleRef,
    context: Context,
    phantom: std::marker::PhantomData<C>,
    _diff_lvl: DifferentiationLevel,
}

impl<C: ComplexScalar> Module<C> {
    pub fn new(module_name: &str, _diff_lvl: DifferentiationLevel) -> Self {
        unsafe {
            let core_context = LLVMContextCreate();
            
            let c_string = to_c_str(module_name);
            let core_module = LLVMModuleCreateWithNameInContext(c_string.as_ptr(), core_context);

            match Target::initialize_native(&InitializationConfig::default()) {
                Ok(_) => {},
                Err(string) => panic!("Error initializing native target: {:?}", string),
            }

            let mut execution_engine = MaybeUninit::uninit();
            let mut err_string = MaybeUninit::uninit();

            let code = LLVMCreateJITCompilerForModule(
                execution_engine.as_mut_ptr(),
                core_module,
                3,
                err_string.as_mut_ptr(),
            );

            if code == 1 {
                panic!("Error creating JIT compiler: {:?}", convert_c_string(err_string.assume_init()));
            }

            let execution_engine = Rc::new(execution_engine.assume_init());

            Module {
                context: Context::new(core_context),
                module: core_module,
                engine: execution_engine,
                phantom: std::marker::PhantomData,
                _diff_lvl,
            }
        }
    }

    pub fn module<'a>(&'a self) -> ManuallyDrop<InkwellModule<'a>> {
        unsafe { ManuallyDrop::new(InkwellModule::new(self.module)) }
    }

    pub fn engine<'a>(&'a self) -> ExecutionEngine<'a> {
        unsafe { ExecutionEngine::new(self.engine.clone(), true) }
    }

    pub fn context(&self) -> &Context {
        &self.context
    }

    pub fn get_function<'a>(&'a self, name: &str) -> Option<JitFunction<'a, UtryFunc<C>>> {
        let name = process_name_for_gen(name);
        unsafe { self.engine().get_function(&name).ok() }
    }

    pub unsafe fn get_function_raw(&self, name: &str) -> UtryFunc<C> {
        let name = process_name_for_gen(name);
        self.engine().get_function(&name).unwrap().as_raw()
    }

    pub fn get_function_and_gradient<'a>(&'a self, name: &str) -> Option<JitFunction<'a, UtryGradFunc<C>>> {
        let name = process_name_for_gen(name) + "_grad";
        unsafe { self.engine().get_function(&name).ok() }
    }

    pub unsafe fn get_function_and_gradient_raw(&self, name: &str) -> UtryGradFunc<C> {
        let name = process_name_for_gen(name) + "_grad";
        self.engine().get_function(&name).unwrap().as_raw()
    }
}

impl<C: ComplexScalar> Drop for Module<C> {
    fn drop(&mut self) {
        assert_eq!(Rc::strong_count(&self.engine), 1);
        unsafe { 
            LLVMDisposeExecutionEngine(*self.engine);
        }
    }
}

impl<C: ComplexScalar> std::fmt::Display for Module<C> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.module().print_to_string().to_string().fmt(f)
    }
}
