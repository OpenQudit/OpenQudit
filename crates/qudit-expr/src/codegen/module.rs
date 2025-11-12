use inkwell::context::Context;
use qudit_core::RealScalar;

use std::borrow::Cow;
use std::ffi::{CStr, CString};
use std::mem::{ManuallyDrop, MaybeUninit};
use std::sync::Mutex;

use inkwell::targets::{InitializationConfig, Target};
use llvm_sys::core::{LLVMContextCreate, LLVMModuleCreateWithNameInContext};
use llvm_sys::execution_engine::{
    LLVMCreateJITCompilerForModule, LLVMDisposeExecutionEngine, LLVMExecutionEngineRef,
    LLVMGetFunctionAddress, LLVMLinkInMCJIT
};
use llvm_sys::prelude::LLVMModuleRef;

use inkwell::module::Module as InkwellModule;

use crate::WriteFunc;

use super::process_name_for_gen;

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
pub struct Module<R: RealScalar> {
    engine: Mutex<LLVMExecutionEngineRef>,
    module: Mutex<LLVMModuleRef>,
    context: Context,
    phantom: std::marker::PhantomData<R>,
}

impl<R: RealScalar> Module<R> {
    pub fn new(module_name: &str) -> Self {
        unsafe {
            let core_context = LLVMContextCreate();

            let c_string = to_c_str(module_name);
            let core_module = LLVMModuleCreateWithNameInContext(c_string.as_ptr(), core_context);
            // LLVMLinkInMCJIT();
            match Target::initialize_native(&InitializationConfig::default()) {
                Ok(_) => {}
                Err(string) => panic!("Error initializing native target: {:?}", string),
            }

            let mut execution_engine = MaybeUninit::uninit();
            let mut err_string = MaybeUninit::uninit();
            LLVMLinkInMCJIT();

            let code = LLVMCreateJITCompilerForModule(
                execution_engine.as_mut_ptr(),
                core_module,
                3,
                err_string.as_mut_ptr(),
            );

            if code == 1 {
                panic!(
                    "Error creating JIT compiler: {:?}",
                    convert_c_string(err_string.assume_init())
                );
            }

            let execution_engine = execution_engine.assume_init();

            Module {
                context: Context::new(core_context),
                module: core_module.into(),
                engine: execution_engine.into(),
                phantom: std::marker::PhantomData,
            }
        }
    }

    pub fn with_module<'a, F, G>(&self, f: F) -> G
    where
        F: FnOnce(ManuallyDrop<InkwellModule<'a>>) -> G,
    {
        let module_ref = self.module.lock().unwrap();
        let module = unsafe { ManuallyDrop::new(InkwellModule::new(*module_ref)) };
        f(module)
    }

    pub fn context(&self) -> &Context {
        &self.context
    }

    // pub fn get_function<'a>(&'a self, name: &str) -> Option<WriteFuncWithLifeTime<'a, R>> {
    pub fn get_function(&self, name: &str) -> Option<WriteFunc<R>> {
        let name = process_name_for_gen(name);
        let engine_ref = self.engine.lock().unwrap();

        assert!(!(*engine_ref).is_null());

        let address = {
            let c_string = to_c_str(&name);
            let address = unsafe { LLVMGetFunctionAddress(*engine_ref, c_string.as_ptr()) };
            if address == 0 {
                return None;
            }
            address as usize
        };

        Some(unsafe { std::mem::transmute_copy(&address) })
    }
}

impl<R: RealScalar> Drop for Module<R> {
    fn drop(&mut self) {
        let engine_ref = self.engine.lock().unwrap();
        unsafe {
            LLVMDisposeExecutionEngine(*engine_ref);
        }
    }
}

impl<R: RealScalar> std::fmt::Display for Module<R> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.with_module(|module| module.print_to_string().to_string().fmt(f))
    }
}

unsafe impl<R: RealScalar> Send for Module<R> {}
