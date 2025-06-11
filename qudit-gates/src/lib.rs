//use qudit_core::matrix::MatMut;
//use qudit_core::matrix::MatVecMut;
use qudit_core::radices;
//use qudit_core::unitary::DifferentiableUnitaryFn;
//use qudit_core::unitary::UnitaryFn;
//use qudit_core::ComplexScalar;
use qudit_core::HasParams;
use qudit_core::QuditRadices;
use qudit_expr::{UnitaryExpression, UnitaryExpressionGenerator};

pub mod constant {
    pub mod h;
    pub mod i;
    pub mod swap;
    pub mod x;
}
pub mod parameterized {
    pub mod p;
    pub mod u3;
}

pub mod composed {
    pub mod control;
    pub mod dagger;
}

pub use constant::i::IGate;
pub use constant::h::HGate;
pub use constant::x::XGate;
pub use parameterized::p::PGate;
pub use parameterized::u3::U3Gate;
pub use composed::control::ControlledGate;

#[derive(Hash, PartialEq, Eq, Clone, Debug)]
pub enum Gate {
    HGate(HGate),
    PGate(PGate),
    XGate(XGate),
    U3Gate(U3Gate),
    Controlled(ControlledGate),
    Expression(UnitaryExpression),
}

impl Gate {
    #[allow(non_snake_case)]
    pub fn H(radix: usize) -> Self {
        Gate::HGate(HGate::new(radix))
    }
    
    #[allow(non_snake_case)]
    pub fn P(radix: usize) -> Self {
        Gate::PGate(PGate::new(radix))
    }
    
    #[allow(non_snake_case)]
    pub fn X(radix: usize) -> Self {
        Gate::XGate(XGate::new(radix))
    }

    #[allow(non_snake_case)]
    pub fn CP() -> Self {
        Gate::Controlled(ControlledGate::new(PGate::new(2), radices![2], vec![vec![1]]))
    }

    #[allow(non_snake_case)]
    pub fn CX() -> Self {
        Gate::Controlled(ControlledGate::new(XGate::new(2), radices![2], vec![vec![1]]))
    }

    #[allow(non_snake_case)]
    pub fn U3() -> Self {
        Gate::U3Gate(U3Gate)
    }
}

impl UnitaryExpressionGenerator for Gate {
    fn gen_expr(&self) -> UnitaryExpression {
        match self {
            Gate::HGate(gate) => gate.gen_expr(),
            Gate::PGate(gate) => gate.gen_expr(),
            Gate::XGate(gate) => gate.gen_expr(),
            Gate::U3Gate(gate) => gate.gen_expr(),
            Gate::Controlled(gate) => gate.gen_expr(),
            Gate::Expression(expr) => expr.clone(),
        }
    }
}

impl HasParams for Gate {
    fn num_params(&self) -> usize {
        match self {
            Gate::HGate(_gate) => 0,
            Gate::XGate(_gate) => 0,
            Gate::U3Gate(_gate) => 3,
            Gate::PGate(gate) => gate.radix - 1,
            Gate::Controlled(gate) => gate.num_params(),
            Gate::Expression(expr) => expr.num_params(),
        }
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use qudit_core::matrix::MatVec;
    use qudit_core::unitary::{UnitaryFn, DifferentiableUnitaryFn, UnitaryMatrix};
    use qudit_core::c64;
    //use qudit_expr::{DifferentiationLevel, Module, ModuleBuilder};

    #[test]
    fn test_h_gate() {
        let h_gate = HGate::new(2);
        let expr = h_gate.gen_expr();
        println!("{:?}", expr);

        let params = vec![];
        let utry: UnitaryMatrix<c64> = expr.get_unitary(&params);
        println!("{:?}", utry);
        // let mut mat: qudit_core::matrix::Mat<c64> = qudit_core::matrix::Mat::zeros(2, 2);
        // unsafe {
        //     let ptr = qudit_core::matrix::matmut_to_ptr(mat.as_mut());
        //     *ptr = 2.3f64;
        // }
        // println!("{:?}", mat);
    }

    #[test]
    fn test_p_gate() {
        let p_gate = PGate::new(2);
        let expr = p_gate.gen_expr();
        println!("{:?}", expr);

        let params = vec![1.7];
        let utry: UnitaryMatrix<c64> = expr.get_unitary(&params);
        println!("{:?}", utry);

        let grad: MatVec<c64> = expr.get_gradient(&params);
        println!("{:?}", grad);
    }

    #[test]
    fn test_u3_gate() {
        let u3_gate = U3Gate;
        let expr = u3_gate.gen_expr();
        println!("{:?}", expr);

        let params = vec![1.7, 2.3, 3.1];
        let utry: UnitaryMatrix<c64> = expr.get_unitary(&params);
        println!("{:?}", utry);

        let grad: MatVec<c64> = expr.get_gradient(&params);
        println!("{:?}", grad);
    }

    // #[test]
    // fn test_time_u3() {
    //     let u3_gate = U3Gate;
    //     let expr = u3_gate.gen_expr();

    //     let mut out_utry: Mat<c64> = Mat::zeros(2, 2);
    //     let mut out_grad: MatVec<c64> = MatVec::zeros(2, 2, 3);

    //     let name = expr.name();
    //     let module: Module<c64> = ModuleBuilder::new("test", DifferentiationLevel::Gradient)
    //         .add_expression_with_stride(expr, out_utry.col_stride().try_into().unwrap())
    //         .build();

    //     let utry_and_grad_func = module.get_function_and_gradient(&name).unwrap();
    //     let out_ptr = unsafe { qudit_core::matrix::matmut_to_ptr(out_utry.as_mut()) };
    //     let out_grad_ptr = unsafe { qudit_core::matrix::matvecmut_to_ptr(out_grad.as_mut()) };
    
    //     let n = 1000000;
    //     let params = vec![vec![rand::random::<f64>(), rand::random::<f64>(), rand::random::<f64>()]; n];
    //     let start = std::time::Instant::now();
    //     for i in 0..n {
    //         unsafe { utry_and_grad_func.call(params[i].as_ptr(), out_ptr, out_grad_ptr); }
    //     }
    //     let duration = start.elapsed();
    //     println!("U3, Time elapsed: {:?}", duration);
    //     println!("U3, Time per call: {:?}", duration / n as u32);

    //     let u3_gate = U3GateNative;
    //     let start = std::time::Instant::now();
    //     for i in 0..n {
    //         u3_gate.write_unitary_and_gradient(&params[i], out_utry.as_mut(), out_grad.as_mut());
    //     }
    //     let duration = start.elapsed();
    //     println!("Native, Time elapsed: {:?}", duration);
    //     println!("Native, Time per call: {:?}", duration / n as u32);
    // }

    // #[test]
    // fn test_time_fusion() {
    //     let u3_gate = U3Gate;
    //     let cnot_gate = CNOTGate;
    //     let u3_expr = u3_gate.gen_expr();
    //     let cnot_expr = cnot_gate.gen_expr();
    //     let u3u3 = u3_expr.otimes(&u3_expr);
    //     let cnotu3u3 = u3u3.dot(&cnot_expr);

    //     let mut out_utry: Mat<c64> = Mat::zeros(4, 4);
    //     let mut out_grad: MatVec<c64> = MatVec::zeros(4, 4, 6);

    //     let name = cnotu3u3.name();
    //     let module: Module<c64> = ModuleBuilder::new("test", DifferentiationLevel::Gradient)
    //         .add_expression_with_stride(cnotu3u3, out_utry.col_stride().try_into().unwrap())
    //         .build();
    //     // println!("{}", module);

    //     let utry_and_grad_func = module.get_function_and_gradient(&name).unwrap();
    //     let out_ptr = unsafe { qudit_core::matrix::matmut_to_ptr(out_utry.as_mut()) };
    //     let out_grad_ptr = unsafe { qudit_core::matrix::matvecmut_to_ptr(out_grad.as_mut()) };
    
    //     let n = 1000000;
    //     let params = vec![vec![rand::random::<f64>(); 6]; n];
    //     let start = std::time::Instant::now();
    //     for i in 0..n {
    //         unsafe { utry_and_grad_func.call(params[i].as_ptr(), out_ptr, out_grad_ptr); }
    //     }
    //     let duration = start.elapsed();
    //     println!("Time elapsed: {:?}", duration);
    //     println!("Time per call: {:?}", duration / n as u32);


    //     let u3_gate = U3GateNative;
    //     let cnot_gate = CNOTGateNative;

    //     let mut u31_out_utry: Mat<c64> = Mat::zeros(2, 2);
    //     let mut u31_out_grad: MatVec<c64> = MatVec::zeros(2, 2, 3);
    //     let mut u32_out_utry: Mat<c64> = Mat::zeros(2, 2);
    //     let mut u32_out_grad: MatVec<c64> = MatVec::zeros(2, 2, 3);
    //     let mut cnot_out_utry: Mat<c64> = Mat::zeros(4, 4);
    //     let mut cnot_out_grad: MatVec<c64> = MatVec::zeros(4, 4, 0);
    //     let mut kron_out_utry: Mat<c64> = Mat::zeros(4, 4);
    //     let mut kron_out_grad: MatVec<c64> = MatVec::zeros(4, 4, 6);
    //     let mut mul_out_utry: Mat<c64> = Mat::zeros(4, 4);
    //     let mut mul_out_grad: MatVec<c64> = MatVec::zeros(4, 4, 6);


    //     let start = std::time::Instant::now();
    //     for i in 0..n {
    //         u3_gate.write_unitary_and_gradient(&params[i][0..3], u31_out_utry.as_mut(), u31_out_grad.as_mut());
    //         u3_gate.write_unitary_and_gradient(&params[i][3..6], u32_out_utry.as_mut(), u32_out_grad.as_mut());
    //         cnot_gate.write_unitary_and_gradient(&[], cnot_out_utry.as_mut(), cnot_out_grad.as_mut());
            
    //         unsafe {
    //             qudit_core::accel::kron_sq_unchecked(kron_out_utry.as_mut(), u31_out_utry.as_ref(), u32_out_utry.as_ref());
    //             qudit_core::accel::kron_sq_unchecked(kron_out_grad.mat_mut(0), u31_out_grad.mat_ref(0), u32_out_utry.as_ref());
    //             qudit_core::accel::kron_sq_unchecked(kron_out_grad.mat_mut(1), u31_out_grad.mat_ref(1), u32_out_utry.as_ref());
    //             qudit_core::accel::kron_sq_unchecked(kron_out_grad.mat_mut(2), u31_out_grad.mat_ref(2), u32_out_utry.as_ref());
    //             qudit_core::accel::kron_sq_unchecked(kron_out_grad.mat_mut(3), u31_out_utry.as_ref(), u32_out_grad.mat_ref(0));
    //             qudit_core::accel::kron_sq_unchecked(kron_out_grad.mat_mut(4), u31_out_utry.as_ref(), u32_out_grad.mat_ref(1));
    //             qudit_core::accel::kron_sq_unchecked(kron_out_grad.mat_mut(5), u31_out_utry.as_ref(), u32_out_grad.mat_ref(2));

    //             qudit_core::accel::matmul_unchecked(kron_out_utry.as_ref(), cnot_out_utry.as_ref(), mul_out_utry.as_mut());
    //             qudit_core::accel::matmul_unchecked(kron_out_grad.mat_ref(0), cnot_out_utry.as_ref(), mul_out_grad.mat_mut(0));
    //             qudit_core::accel::matmul_unchecked(kron_out_grad.mat_ref(1), cnot_out_utry.as_ref(), mul_out_grad.mat_mut(1));
    //             qudit_core::accel::matmul_unchecked(kron_out_grad.mat_ref(2), cnot_out_utry.as_ref(), mul_out_grad.mat_mut(2));
    //             qudit_core::accel::matmul_unchecked(kron_out_grad.mat_ref(3), cnot_out_utry.as_ref(), mul_out_grad.mat_mut(3));
    //             qudit_core::accel::matmul_unchecked(kron_out_grad.mat_ref(4), cnot_out_utry.as_ref(), mul_out_grad.mat_mut(4));
    //             qudit_core::accel::matmul_unchecked(kron_out_grad.mat_ref(5), cnot_out_utry.as_ref(), mul_out_grad.mat_mut(5));
    //         }
    //     }
    //     let duration = start.elapsed();
    //     println!("Native, Time elapsed: {:?}", duration);
    //     println!("Native, Time per call: {:?}", duration / n as u32);
    // }

    // #[test]
    // fn test_time_fusion3() {
    //     let u3_gate = U3Gate;
    //     let cnot_gate = CNOTGate;
    //     let u3_expr = u3_gate.gen_expr();
    //     let cnot_expr = cnot_gate.gen_expr();
    //     let u3u3 = u3_expr.otimes(&u3_expr);
    //     let cnotu3u3 = u3u3.dot(&cnot_expr);
    //     // let full_expr = cnotu3u3.dot(&cnotu3u3).dot(&cnotu3u3).dot(&u3u3);

    //     let mut out_utry: Mat<c64> = Mat::zeros(4, 4);
    //     let mut out_grad: MatVec<c64> = MatVec::zeros(4, 4, 24);
    //     // let params = vec![1.7, 2.3, 3.1];

    //     // let name = full_expr.name();
    //     // let module: Module<c64> = ModuleBuilder::new("test", DifferentiationLevel::Gradient)
    //     //     .add_expression_with_stride(full_expr, out_utry.col_stride().try_into().unwrap())
    //     //     .build();
    //     // println!("{}", module);

    //     // let utry_and_grad_func = module.get_function_and_gradient(&name).unwrap();
    //     // let out_ptr = unsafe { qudit_core::matrix::matmut_to_ptr(out_utry.as_mut()) };
    //     // let out_grad_ptr = unsafe { qudit_core::matrix::matvecmut_to_ptr(out_grad.as_mut()) };
    
    //     let n = 1000000;
    //     let params = vec![vec![rand::random::<f64>(); 24]; n];
    //     // let start = std::time::Instant::now();
    //     // for i in 0..n {
    //     //     unsafe { utry_and_grad_func.call(params[i].as_ptr(), out_ptr, out_grad_ptr); }
    //     // }
    //     // let duration = start.elapsed();
    //     // println!("Time elapsed: {:?}", duration);
    //     // println!("Time per call: {:?}", duration / n as u32);

    //     let mut kron_out_utry: Mat<c64> = Mat::zeros(4, 4);
    //     let mut kron_out_grad: MatVec<c64> = MatVec::zeros(4, 4, 6);
    //     let mut mul_out_utry: Mat<c64> = Mat::zeros(4, 4);
    //     let mut mul_out_grad: MatVec<c64> = MatVec::zeros(4, 4, 6);
    //     let mut mul2_out_utry: Mat<c64> = Mat::zeros(4, 4);
    //     let mut mul2_out_grad: MatVec<c64> = MatVec::zeros(4, 4, 6);
    //     let mut mul3_out_utry: Mat<c64> = Mat::zeros(4, 4);
    //     let mut mul3_out_grad: MatVec<c64> = MatVec::zeros(4, 4, 6);
    //     let mut mul4_out_utry: Mat<c64> = Mat::zeros(4, 4);
    //     let mut mul4_out_grad: MatVec<c64> = MatVec::zeros(4, 4, 12);
    //     let mut mul5_out_utry: Mat<c64> = Mat::zeros(4, 4);
    //     let mut mul5_out_grad: MatVec<c64> = MatVec::zeros(4, 4, 18);
    //     let mut mul6_out_utry: Mat<c64> = Mat::zeros(4, 4);
    //     let mut mul6_out_grad: MatVec<c64> = MatVec::zeros(4, 4, 24);


    //     let kron_name = u3u3.name();
    //     let mul_name = cnotu3u3.name();
    //     let module: Module<c64> = ModuleBuilder::new("test", DifferentiationLevel::Gradient)
    //         .add_expression_with_stride(u3u3, out_utry.col_stride().try_into().unwrap())
    //         .add_expression_with_stride(cnotu3u3, out_utry.col_stride().try_into().unwrap())
    //         .build();

    //     let kron_utry_and_grad_func = module.get_function_and_gradient(&kron_name).unwrap();
    //     let mul_utry_and_grad_func = module.get_function_and_gradient(&mul_name).unwrap();

    //     let kron1_out_ptr = unsafe { qudit_core::matrix::matmut_to_ptr(kron_out_utry.as_mut()) };
    //     let kron1_out_grad_ptr = unsafe { qudit_core::matrix::matvecmut_to_ptr(kron_out_grad.as_mut()) };
    //     let mul1_out_ptr = unsafe { qudit_core::matrix::matmut_to_ptr(mul_out_utry.as_mut()) };
    //     let mul1_out_grad_ptr = unsafe { qudit_core::matrix::matvecmut_to_ptr(mul_out_grad.as_mut()) };
    //     let mul2_out_ptr = unsafe { qudit_core::matrix::matmut_to_ptr(mul2_out_utry.as_mut()) };
    //     let mul2_out_grad_ptr = unsafe { qudit_core::matrix::matvecmut_to_ptr(mul2_out_grad.as_mut()) };
    //     let mul3_out_ptr = unsafe { qudit_core::matrix::matmut_to_ptr(mul3_out_utry.as_mut()) };
    //     let mul3_out_grad_ptr = unsafe { qudit_core::matrix::matvecmut_to_ptr(mul3_out_grad.as_mut()) };
    //     // let mul4_out_ptr = unsafe { qudit_core::matrix::matmut_to_ptr(mul4_out_utry.as_mut()) };
    //     // let mul4_out_grad_ptr = unsafe { qudit_core::matrix::matvecmut_to_ptr(mul4_out_grad.as_mut()) };
    //     // let mul5_out_ptr = unsafe { qudit_core::matrix::matmut_to_ptr(mul5_out_utry.as_mut()) };
    //     // let mul5_out_grad_ptr = unsafe { qudit_core::matrix::matvecmut_to_ptr(mul5_out_grad.as_mut()) };
    //     // let mul6_out_ptr = unsafe { qudit_core::matrix::matmut_to_ptr(mul6_out_utry.as_mut()) };
    //     // let mul6_out_grad_ptr = unsafe { qudit_core::matrix::matvecmut_to_ptr(mul6_out_grad.as_mut()) };



    //     let start = std::time::Instant::now();
    //     for i in 0..n {
    //         unsafe {
    //             mul_utry_and_grad_func.call(params[i][0..6].as_ptr(), mul1_out_ptr, mul1_out_grad_ptr);
    //             mul_utry_and_grad_func.call(params[i][6..12].as_ptr(), mul2_out_ptr, mul2_out_grad_ptr);
    //             mul_utry_and_grad_func.call(params[i][12..18].as_ptr(), mul3_out_ptr, mul3_out_grad_ptr);
    //             kron_utry_and_grad_func.call(params[i][18..24].as_ptr(), kron1_out_ptr, kron1_out_grad_ptr);
                
    //             qudit_core::accel::matmul_unchecked(mul_out_utry.as_ref(), kron_out_utry.as_ref(), mul4_out_utry.as_mut());
    //             qudit_core::accel::matmul_unchecked(mul_out_grad.mat_ref(0), kron_out_utry.as_ref(), mul4_out_grad.mat_mut(0));
    //             qudit_core::accel::matmul_unchecked(mul_out_grad.mat_ref(1), kron_out_utry.as_ref(), mul4_out_grad.mat_mut(1));
    //             qudit_core::accel::matmul_unchecked(mul_out_grad.mat_ref(2), kron_out_utry.as_ref(), mul4_out_grad.mat_mut(2));
    //             qudit_core::accel::matmul_unchecked(mul_out_grad.mat_ref(3), kron_out_utry.as_ref(), mul4_out_grad.mat_mut(3));
    //             qudit_core::accel::matmul_unchecked(mul_out_grad.mat_ref(4), kron_out_utry.as_ref(), mul4_out_grad.mat_mut(4));
    //             qudit_core::accel::matmul_unchecked(mul_out_grad.mat_ref(5), kron_out_utry.as_ref(), mul4_out_grad.mat_mut(5));
    //             qudit_core::accel::matmul_unchecked(mul_out_utry.as_ref(), kron_out_grad.mat_ref(0), mul4_out_grad.mat_mut(6));
    //             qudit_core::accel::matmul_unchecked(mul_out_utry.as_ref(), kron_out_grad.mat_ref(1), mul4_out_grad.mat_mut(7));
    //             qudit_core::accel::matmul_unchecked(mul_out_utry.as_ref(), kron_out_grad.mat_ref(2), mul4_out_grad.mat_mut(8));
    //             qudit_core::accel::matmul_unchecked(mul_out_utry.as_ref(), kron_out_grad.mat_ref(3), mul4_out_grad.mat_mut(9));
    //             qudit_core::accel::matmul_unchecked(mul_out_utry.as_ref(), kron_out_grad.mat_ref(4), mul4_out_grad.mat_mut(10));
    //             qudit_core::accel::matmul_unchecked(mul_out_utry.as_ref(), kron_out_grad.mat_ref(5), mul4_out_grad.mat_mut(11));

    //             qudit_core::accel::matmul_unchecked(mul2_out_utry.as_ref(), mul4_out_utry.as_ref(), mul5_out_utry.as_mut());
    //             qudit_core::accel::matmul_unchecked(mul2_out_grad.mat_ref(0), mul4_out_utry.as_ref(), mul5_out_grad.mat_mut(0));
    //             qudit_core::accel::matmul_unchecked(mul2_out_grad.mat_ref(1), mul4_out_utry.as_ref(), mul5_out_grad.mat_mut(1));
    //             qudit_core::accel::matmul_unchecked(mul2_out_grad.mat_ref(2), mul4_out_utry.as_ref(), mul5_out_grad.mat_mut(2));
    //             qudit_core::accel::matmul_unchecked(mul2_out_grad.mat_ref(3), mul4_out_utry.as_ref(), mul5_out_grad.mat_mut(3));
    //             qudit_core::accel::matmul_unchecked(mul2_out_grad.mat_ref(4), mul4_out_utry.as_ref(), mul5_out_grad.mat_mut(4));
    //             qudit_core::accel::matmul_unchecked(mul2_out_grad.mat_ref(5), mul4_out_utry.as_ref(), mul5_out_grad.mat_mut(5));
    //             qudit_core::accel::matmul_unchecked(mul2_out_utry.as_ref(), mul4_out_grad.mat_ref(0), mul5_out_grad.mat_mut(6));
    //             qudit_core::accel::matmul_unchecked(mul2_out_utry.as_ref(), mul4_out_grad.mat_ref(1), mul5_out_grad.mat_mut(7));
    //             qudit_core::accel::matmul_unchecked(mul2_out_utry.as_ref(), mul4_out_grad.mat_ref(2), mul5_out_grad.mat_mut(8));
    //             qudit_core::accel::matmul_unchecked(mul2_out_utry.as_ref(), mul4_out_grad.mat_ref(3), mul5_out_grad.mat_mut(9));
    //             qudit_core::accel::matmul_unchecked(mul2_out_utry.as_ref(), mul4_out_grad.mat_ref(4), mul5_out_grad.mat_mut(10));
    //             qudit_core::accel::matmul_unchecked(mul2_out_utry.as_ref(), mul4_out_grad.mat_ref(5), mul5_out_grad.mat_mut(11));
    //             qudit_core::accel::matmul_unchecked(mul2_out_utry.as_ref(), mul4_out_grad.mat_ref(6), mul5_out_grad.mat_mut(12));
    //             qudit_core::accel::matmul_unchecked(mul2_out_utry.as_ref(), mul4_out_grad.mat_ref(7), mul5_out_grad.mat_mut(13));
    //             qudit_core::accel::matmul_unchecked(mul2_out_utry.as_ref(), mul4_out_grad.mat_ref(8), mul5_out_grad.mat_mut(14));
    //             qudit_core::accel::matmul_unchecked(mul2_out_utry.as_ref(), mul4_out_grad.mat_ref(9), mul5_out_grad.mat_mut(15));
    //             qudit_core::accel::matmul_unchecked(mul2_out_utry.as_ref(), mul4_out_grad.mat_ref(10), mul5_out_grad.mat_mut(16));
    //             qudit_core::accel::matmul_unchecked(mul2_out_utry.as_ref(), mul4_out_grad.mat_ref(11), mul5_out_grad.mat_mut(17));

    //             qudit_core::accel::matmul_unchecked(mul3_out_utry.as_ref(), mul5_out_utry.as_ref(), mul6_out_utry.as_mut());
    //             qudit_core::accel::matmul_unchecked(mul3_out_grad.mat_ref(0), mul5_out_utry.as_ref(), mul6_out_grad.mat_mut(0));
    //             qudit_core::accel::matmul_unchecked(mul3_out_grad.mat_ref(1), mul5_out_utry.as_ref(), mul6_out_grad.mat_mut(1));
    //             qudit_core::accel::matmul_unchecked(mul3_out_grad.mat_ref(2), mul5_out_utry.as_ref(), mul6_out_grad.mat_mut(2));
    //             qudit_core::accel::matmul_unchecked(mul3_out_grad.mat_ref(3), mul5_out_utry.as_ref(), mul6_out_grad.mat_mut(3));
    //             qudit_core::accel::matmul_unchecked(mul3_out_grad.mat_ref(4), mul5_out_utry.as_ref(), mul6_out_grad.mat_mut(4));
    //             qudit_core::accel::matmul_unchecked(mul3_out_grad.mat_ref(5), mul5_out_utry.as_ref(), mul6_out_grad.mat_mut(5));
    //             qudit_core::accel::matmul_unchecked(mul3_out_utry.as_ref(), mul5_out_grad.mat_ref(0), mul6_out_grad.mat_mut(6));
    //             qudit_core::accel::matmul_unchecked(mul3_out_utry.as_ref(), mul5_out_grad.mat_ref(1), mul6_out_grad.mat_mut(7));
    //             qudit_core::accel::matmul_unchecked(mul3_out_utry.as_ref(), mul5_out_grad.mat_ref(2), mul6_out_grad.mat_mut(8));
    //             qudit_core::accel::matmul_unchecked(mul3_out_utry.as_ref(), mul5_out_grad.mat_ref(3), mul6_out_grad.mat_mut(9));
    //             qudit_core::accel::matmul_unchecked(mul3_out_utry.as_ref(), mul5_out_grad.mat_ref(4), mul6_out_grad.mat_mut(10));
    //             qudit_core::accel::matmul_unchecked(mul3_out_utry.as_ref(), mul5_out_grad.mat_ref(5), mul6_out_grad.mat_mut(11));
    //             qudit_core::accel::matmul_unchecked(mul3_out_utry.as_ref(), mul5_out_grad.mat_ref(6), mul6_out_grad.mat_mut(12));
    //             qudit_core::accel::matmul_unchecked(mul3_out_utry.as_ref(), mul5_out_grad.mat_ref(7), mul6_out_grad.mat_mut(13));
    //             qudit_core::accel::matmul_unchecked(mul3_out_utry.as_ref(), mul5_out_grad.mat_ref(8), mul6_out_grad.mat_mut(14));
    //             qudit_core::accel::matmul_unchecked(mul3_out_utry.as_ref(), mul5_out_grad.mat_ref(9), mul6_out_grad.mat_mut(15));
    //             qudit_core::accel::matmul_unchecked(mul3_out_utry.as_ref(), mul5_out_grad.mat_ref(10), mul6_out_grad.mat_mut(16));
    //             qudit_core::accel::matmul_unchecked(mul3_out_utry.as_ref(), mul5_out_grad.mat_ref(11), mul6_out_grad.mat_mut(17));
    //             qudit_core::accel::matmul_unchecked(mul3_out_utry.as_ref(), mul5_out_grad.mat_ref(12), mul6_out_grad.mat_mut(18));
    //             qudit_core::accel::matmul_unchecked(mul3_out_utry.as_ref(), mul5_out_grad.mat_ref(13), mul6_out_grad.mat_mut(19));
    //             qudit_core::accel::matmul_unchecked(mul3_out_utry.as_ref(), mul5_out_grad.mat_ref(14), mul6_out_grad.mat_mut(20));
    //             qudit_core::accel::matmul_unchecked(mul3_out_utry.as_ref(), mul5_out_grad.mat_ref(15), mul6_out_grad.mat_mut(21));
    //             qudit_core::accel::matmul_unchecked(mul3_out_utry.as_ref(), mul5_out_grad.mat_ref(16), mul6_out_grad.mat_mut(22));
    //             qudit_core::accel::matmul_unchecked(mul3_out_utry.as_ref(), mul5_out_grad.mat_ref(17), mul6_out_grad.mat_mut(23));
    //         }
    //     }
    //     let duration = start.elapsed();
    //     println!("Mix, Time elapsed: {:?}", duration);
    //     println!("Mix, Time per call: {:?}", duration / n as u32);


    //     let u3_gate = U3GateNative;
    //     let cnot_gate = CNOTGateNative;

    //     let mut u31_out_utry: Mat<c64> = Mat::zeros(2, 2);
    //     let mut u31_out_grad: MatVec<c64> = MatVec::zeros(2, 2, 3);
    //     let mut u32_out_utry: Mat<c64> = Mat::zeros(2, 2);
    //     let mut u32_out_grad: MatVec<c64> = MatVec::zeros(2, 2, 3);
    //     let mut cnot_out_utry: Mat<c64> = Mat::zeros(4, 4);
    //     let mut cnot_out_grad: MatVec<c64> = MatVec::zeros(4, 4, 0);
    //     let mut kron_out_utry: Mat<c64> = Mat::zeros(4, 4);
    //     let mut kron_out_grad: MatVec<c64> = MatVec::zeros(4, 4, 6);
    //     let mut mul1_out_utry: Mat<c64> = Mat::zeros(4, 4);
    //     let mut mul1_out_grad: MatVec<c64> = MatVec::zeros(4, 4, 6);
    //     let mut mul2_out_utry: Mat<c64> = Mat::zeros(4, 4);
    //     let mut mul2_out_grad: MatVec<c64> = MatVec::zeros(4, 4, 6);
    //     let mut mul3_out_utry: Mat<c64> = Mat::zeros(4, 4);
    //     let mut mul3_out_grad: MatVec<c64> = MatVec::zeros(4, 4, 6);


    //     let mut mul4_out_utry: Mat<c64> = Mat::zeros(4, 4);
    //     let mut mul4_out_grad: MatVec<c64> = MatVec::zeros(4, 4, 12);

    //     let mut mul5_out_utry: Mat<c64> = Mat::zeros(4, 4);
    //     let mut mul5_out_grad: MatVec<c64> = MatVec::zeros(4, 4, 18);

    //     let mut mul6_out_utry: Mat<c64> = Mat::zeros(4, 4);
    //     let mut mul6_out_grad: MatVec<c64> = MatVec::zeros(4, 4, 24);


    //     let start = std::time::Instant::now();
    //     for i in 0..n {
    //         u3_gate.write_unitary_and_gradient(&params[i][0..3], u31_out_utry.as_mut(), u31_out_grad.as_mut());
    //         u3_gate.write_unitary_and_gradient(&params[i][3..6], u32_out_utry.as_mut(), u32_out_grad.as_mut());
    //         cnot_gate.write_unitary_and_gradient(&[], cnot_out_utry.as_mut(), cnot_out_grad.as_mut());
            
    //         unsafe {
    //             qudit_core::accel::kron_sq_unchecked(kron_out_utry.as_mut(), u31_out_utry.as_ref(), u32_out_utry.as_ref());
    //             qudit_core::accel::kron_sq_unchecked(kron_out_grad.mat_mut(0), u31_out_grad.mat_ref(0), u32_out_utry.as_ref());
    //             qudit_core::accel::kron_sq_unchecked(kron_out_grad.mat_mut(1), u31_out_grad.mat_ref(1), u32_out_utry.as_ref());
    //             qudit_core::accel::kron_sq_unchecked(kron_out_grad.mat_mut(2), u31_out_grad.mat_ref(2), u32_out_utry.as_ref());
    //             qudit_core::accel::kron_sq_unchecked(kron_out_grad.mat_mut(3), u31_out_utry.as_ref(), u32_out_grad.mat_ref(0));
    //             qudit_core::accel::kron_sq_unchecked(kron_out_grad.mat_mut(4), u31_out_utry.as_ref(), u32_out_grad.mat_ref(1));
    //             qudit_core::accel::kron_sq_unchecked(kron_out_grad.mat_mut(5), u31_out_utry.as_ref(), u32_out_grad.mat_ref(2));

    //             qudit_core::accel::matmul_unchecked(kron_out_utry.as_ref(), cnot_out_utry.as_ref(), mul1_out_utry.as_mut());
    //             qudit_core::accel::matmul_unchecked(kron_out_grad.mat_ref(0), cnot_out_utry.as_ref(), mul1_out_grad.mat_mut(0));
    //             qudit_core::accel::matmul_unchecked(kron_out_grad.mat_ref(1), cnot_out_utry.as_ref(), mul1_out_grad.mat_mut(1));
    //             qudit_core::accel::matmul_unchecked(kron_out_grad.mat_ref(2), cnot_out_utry.as_ref(), mul1_out_grad.mat_mut(2));
    //             qudit_core::accel::matmul_unchecked(kron_out_grad.mat_ref(3), cnot_out_utry.as_ref(), mul1_out_grad.mat_mut(3));
    //             qudit_core::accel::matmul_unchecked(kron_out_grad.mat_ref(4), cnot_out_utry.as_ref(), mul1_out_grad.mat_mut(4));
    //             qudit_core::accel::matmul_unchecked(kron_out_grad.mat_ref(5), cnot_out_utry.as_ref(), mul1_out_grad.mat_mut(5));
    //         }

    //         u3_gate.write_unitary_and_gradient(&params[i][6..9], u31_out_utry.as_mut(), u31_out_grad.as_mut());
    //         u3_gate.write_unitary_and_gradient(&params[i][9..12], u32_out_utry.as_mut(), u32_out_grad.as_mut());
    //         cnot_gate.write_unitary_and_gradient(&[], cnot_out_utry.as_mut(), cnot_out_grad.as_mut());
            
    //         unsafe {
    //             qudit_core::accel::kron_sq_unchecked(kron_out_utry.as_mut(), u31_out_utry.as_ref(), u32_out_utry.as_ref());
    //             qudit_core::accel::kron_sq_unchecked(kron_out_grad.mat_mut(0), u31_out_grad.mat_ref(0), u32_out_utry.as_ref());
    //             qudit_core::accel::kron_sq_unchecked(kron_out_grad.mat_mut(1), u31_out_grad.mat_ref(1), u32_out_utry.as_ref());
    //             qudit_core::accel::kron_sq_unchecked(kron_out_grad.mat_mut(2), u31_out_grad.mat_ref(2), u32_out_utry.as_ref());
    //             qudit_core::accel::kron_sq_unchecked(kron_out_grad.mat_mut(3), u31_out_utry.as_ref(), u32_out_grad.mat_ref(0));
    //             qudit_core::accel::kron_sq_unchecked(kron_out_grad.mat_mut(4), u31_out_utry.as_ref(), u32_out_grad.mat_ref(1));
    //             qudit_core::accel::kron_sq_unchecked(kron_out_grad.mat_mut(5), u31_out_utry.as_ref(), u32_out_grad.mat_ref(2));

    //             qudit_core::accel::matmul_unchecked(kron_out_utry.as_ref(), cnot_out_utry.as_ref(), mul2_out_utry.as_mut());
    //             qudit_core::accel::matmul_unchecked(kron_out_grad.mat_ref(0), cnot_out_utry.as_ref(), mul2_out_grad.mat_mut(0));
    //             qudit_core::accel::matmul_unchecked(kron_out_grad.mat_ref(1), cnot_out_utry.as_ref(), mul2_out_grad.mat_mut(1));
    //             qudit_core::accel::matmul_unchecked(kron_out_grad.mat_ref(2), cnot_out_utry.as_ref(), mul2_out_grad.mat_mut(2));
    //             qudit_core::accel::matmul_unchecked(kron_out_grad.mat_ref(3), cnot_out_utry.as_ref(), mul2_out_grad.mat_mut(3));
    //             qudit_core::accel::matmul_unchecked(kron_out_grad.mat_ref(4), cnot_out_utry.as_ref(), mul2_out_grad.mat_mut(4));
    //             qudit_core::accel::matmul_unchecked(kron_out_grad.mat_ref(5), cnot_out_utry.as_ref(), mul2_out_grad.mat_mut(5));
    //         }

    //         u3_gate.write_unitary_and_gradient(&params[i][12..15], u31_out_utry.as_mut(), u31_out_grad.as_mut());
    //         u3_gate.write_unitary_and_gradient(&params[i][15..18], u32_out_utry.as_mut(), u32_out_grad.as_mut());

    //         unsafe {
    //             qudit_core::accel::kron_sq_unchecked(kron_out_utry.as_mut(), u31_out_utry.as_ref(), u32_out_utry.as_ref());
    //             qudit_core::accel::kron_sq_unchecked(kron_out_grad.mat_mut(0), u31_out_grad.mat_ref(0), u32_out_utry.as_ref());
    //             qudit_core::accel::kron_sq_unchecked(kron_out_grad.mat_mut(1), u31_out_grad.mat_ref(1), u32_out_utry.as_ref());
    //             qudit_core::accel::kron_sq_unchecked(kron_out_grad.mat_mut(2), u31_out_grad.mat_ref(2), u32_out_utry.as_ref());
    //             qudit_core::accel::kron_sq_unchecked(kron_out_grad.mat_mut(3), u31_out_utry.as_ref(), u32_out_grad.mat_ref(0));
    //             qudit_core::accel::kron_sq_unchecked(kron_out_grad.mat_mut(4), u31_out_utry.as_ref(), u32_out_grad.mat_ref(1));
    //             qudit_core::accel::kron_sq_unchecked(kron_out_grad.mat_mut(5), u31_out_utry.as_ref(), u32_out_grad.mat_ref(2));

    //             qudit_core::accel::matmul_unchecked(kron_out_utry.as_ref(), cnot_out_utry.as_ref(), mul3_out_utry.as_mut());
    //             qudit_core::accel::matmul_unchecked(kron_out_grad.mat_ref(0), cnot_out_utry.as_ref(), mul3_out_grad.mat_mut(0));
    //             qudit_core::accel::matmul_unchecked(kron_out_grad.mat_ref(1), cnot_out_utry.as_ref(), mul3_out_grad.mat_mut(1));
    //             qudit_core::accel::matmul_unchecked(kron_out_grad.mat_ref(2), cnot_out_utry.as_ref(), mul3_out_grad.mat_mut(2));
    //             qudit_core::accel::matmul_unchecked(kron_out_grad.mat_ref(3), cnot_out_utry.as_ref(), mul3_out_grad.mat_mut(3));
    //             qudit_core::accel::matmul_unchecked(kron_out_grad.mat_ref(4), cnot_out_utry.as_ref(), mul3_out_grad.mat_mut(4));
    //             qudit_core::accel::matmul_unchecked(kron_out_grad.mat_ref(5), cnot_out_utry.as_ref(), mul3_out_grad.mat_mut(5));
    //         }

    //         u3_gate.write_unitary_and_gradient(&params[i][18..21], u31_out_utry.as_mut(), u31_out_grad.as_mut());
    //         u3_gate.write_unitary_and_gradient(&params[i][21..24], u32_out_utry.as_mut(), u32_out_grad.as_mut());

    //         unsafe {
    //             qudit_core::accel::kron_sq_unchecked(kron_out_utry.as_mut(), u31_out_utry.as_ref(), u32_out_utry.as_ref());
    //             qudit_core::accel::kron_sq_unchecked(kron_out_grad.mat_mut(0), u31_out_grad.mat_ref(0), u32_out_utry.as_ref());
    //             qudit_core::accel::kron_sq_unchecked(kron_out_grad.mat_mut(1), u31_out_grad.mat_ref(1), u32_out_utry.as_ref());
    //             qudit_core::accel::kron_sq_unchecked(kron_out_grad.mat_mut(2), u31_out_grad.mat_ref(2), u32_out_utry.as_ref());
    //             qudit_core::accel::kron_sq_unchecked(kron_out_grad.mat_mut(3), u31_out_utry.as_ref(), u32_out_grad.mat_ref(0));
    //             qudit_core::accel::kron_sq_unchecked(kron_out_grad.mat_mut(4), u31_out_utry.as_ref(), u32_out_grad.mat_ref(1));
    //             qudit_core::accel::kron_sq_unchecked(kron_out_grad.mat_mut(5), u31_out_utry.as_ref(), u32_out_grad.mat_ref(2));
    //         }

    //         unsafe {
    //             qudit_core::accel::matmul_unchecked(mul1_out_utry.as_ref(), kron_out_utry.as_ref(), mul4_out_utry.as_mut());
    //             qudit_core::accel::matmul_unchecked(mul1_out_grad.mat_ref(0), kron_out_utry.as_ref(), mul4_out_grad.mat_mut(0));
    //             qudit_core::accel::matmul_unchecked(mul1_out_grad.mat_ref(1), kron_out_utry.as_ref(), mul4_out_grad.mat_mut(1));
    //             qudit_core::accel::matmul_unchecked(mul1_out_grad.mat_ref(2), kron_out_utry.as_ref(), mul4_out_grad.mat_mut(2));
    //             qudit_core::accel::matmul_unchecked(mul1_out_grad.mat_ref(3), kron_out_utry.as_ref(), mul4_out_grad.mat_mut(3));
    //             qudit_core::accel::matmul_unchecked(mul1_out_grad.mat_ref(4), kron_out_utry.as_ref(), mul4_out_grad.mat_mut(4));
    //             qudit_core::accel::matmul_unchecked(mul1_out_grad.mat_ref(5), kron_out_utry.as_ref(), mul4_out_grad.mat_mut(5));
    //             qudit_core::accel::matmul_unchecked(mul1_out_utry.as_ref(), kron_out_grad.mat_ref(0), mul4_out_grad.mat_mut(6));
    //             qudit_core::accel::matmul_unchecked(mul1_out_utry.as_ref(), kron_out_grad.mat_ref(1), mul4_out_grad.mat_mut(7));
    //             qudit_core::accel::matmul_unchecked(mul1_out_utry.as_ref(), kron_out_grad.mat_ref(2), mul4_out_grad.mat_mut(8));
    //             qudit_core::accel::matmul_unchecked(mul1_out_utry.as_ref(), kron_out_grad.mat_ref(3), mul4_out_grad.mat_mut(9));
    //             qudit_core::accel::matmul_unchecked(mul1_out_utry.as_ref(), kron_out_grad.mat_ref(4), mul4_out_grad.mat_mut(10));
    //             qudit_core::accel::matmul_unchecked(mul1_out_utry.as_ref(), kron_out_grad.mat_ref(5), mul4_out_grad.mat_mut(11));

    //             qudit_core::accel::matmul_unchecked(mul2_out_utry.as_ref(), mul4_out_utry.as_ref(), mul5_out_utry.as_mut());
    //             qudit_core::accel::matmul_unchecked(mul2_out_grad.mat_ref(0), mul4_out_utry.as_ref(), mul5_out_grad.mat_mut(0));
    //             qudit_core::accel::matmul_unchecked(mul2_out_grad.mat_ref(1), mul4_out_utry.as_ref(), mul5_out_grad.mat_mut(1));
    //             qudit_core::accel::matmul_unchecked(mul2_out_grad.mat_ref(2), mul4_out_utry.as_ref(), mul5_out_grad.mat_mut(2));
    //             qudit_core::accel::matmul_unchecked(mul2_out_grad.mat_ref(3), mul4_out_utry.as_ref(), mul5_out_grad.mat_mut(3));
    //             qudit_core::accel::matmul_unchecked(mul2_out_grad.mat_ref(4), mul4_out_utry.as_ref(), mul5_out_grad.mat_mut(4));
    //             qudit_core::accel::matmul_unchecked(mul2_out_grad.mat_ref(5), mul4_out_utry.as_ref(), mul5_out_grad.mat_mut(5));
    //             qudit_core::accel::matmul_unchecked(mul2_out_utry.as_ref(), mul4_out_grad.mat_ref(0), mul5_out_grad.mat_mut(6));
    //             qudit_core::accel::matmul_unchecked(mul2_out_utry.as_ref(), mul4_out_grad.mat_ref(1), mul5_out_grad.mat_mut(7));
    //             qudit_core::accel::matmul_unchecked(mul2_out_utry.as_ref(), mul4_out_grad.mat_ref(2), mul5_out_grad.mat_mut(8));
    //             qudit_core::accel::matmul_unchecked(mul2_out_utry.as_ref(), mul4_out_grad.mat_ref(3), mul5_out_grad.mat_mut(9));
    //             qudit_core::accel::matmul_unchecked(mul2_out_utry.as_ref(), mul4_out_grad.mat_ref(4), mul5_out_grad.mat_mut(10));
    //             qudit_core::accel::matmul_unchecked(mul2_out_utry.as_ref(), mul4_out_grad.mat_ref(5), mul5_out_grad.mat_mut(11));
    //             qudit_core::accel::matmul_unchecked(mul2_out_utry.as_ref(), mul4_out_grad.mat_ref(6), mul5_out_grad.mat_mut(12));
    //             qudit_core::accel::matmul_unchecked(mul2_out_utry.as_ref(), mul4_out_grad.mat_ref(7), mul5_out_grad.mat_mut(13));
    //             qudit_core::accel::matmul_unchecked(mul2_out_utry.as_ref(), mul4_out_grad.mat_ref(8), mul5_out_grad.mat_mut(14));
    //             qudit_core::accel::matmul_unchecked(mul2_out_utry.as_ref(), mul4_out_grad.mat_ref(9), mul5_out_grad.mat_mut(15));
    //             qudit_core::accel::matmul_unchecked(mul2_out_utry.as_ref(), mul4_out_grad.mat_ref(10), mul5_out_grad.mat_mut(16));
    //             qudit_core::accel::matmul_unchecked(mul2_out_utry.as_ref(), mul4_out_grad.mat_ref(11), mul5_out_grad.mat_mut(17));

    //             qudit_core::accel::matmul_unchecked(mul3_out_utry.as_ref(), mul5_out_utry.as_ref(), mul6_out_utry.as_mut());
    //             qudit_core::accel::matmul_unchecked(mul3_out_grad.mat_ref(0), mul5_out_utry.as_ref(), mul6_out_grad.mat_mut(0));
    //             qudit_core::accel::matmul_unchecked(mul3_out_grad.mat_ref(1), mul5_out_utry.as_ref(), mul6_out_grad.mat_mut(1));
    //             qudit_core::accel::matmul_unchecked(mul3_out_grad.mat_ref(2), mul5_out_utry.as_ref(), mul6_out_grad.mat_mut(2));
    //             qudit_core::accel::matmul_unchecked(mul3_out_grad.mat_ref(3), mul5_out_utry.as_ref(), mul6_out_grad.mat_mut(3));
    //             qudit_core::accel::matmul_unchecked(mul3_out_grad.mat_ref(4), mul5_out_utry.as_ref(), mul6_out_grad.mat_mut(4));
    //             qudit_core::accel::matmul_unchecked(mul3_out_grad.mat_ref(5), mul5_out_utry.as_ref(), mul6_out_grad.mat_mut(5));
    //             qudit_core::accel::matmul_unchecked(mul3_out_utry.as_ref(), mul5_out_grad.mat_ref(0), mul6_out_grad.mat_mut(6));
    //             qudit_core::accel::matmul_unchecked(mul3_out_utry.as_ref(), mul5_out_grad.mat_ref(1), mul6_out_grad.mat_mut(7));
    //             qudit_core::accel::matmul_unchecked(mul3_out_utry.as_ref(), mul5_out_grad.mat_ref(2), mul6_out_grad.mat_mut(8));
    //             qudit_core::accel::matmul_unchecked(mul3_out_utry.as_ref(), mul5_out_grad.mat_ref(3), mul6_out_grad.mat_mut(9));
    //             qudit_core::accel::matmul_unchecked(mul3_out_utry.as_ref(), mul5_out_grad.mat_ref(4), mul6_out_grad.mat_mut(10));
    //             qudit_core::accel::matmul_unchecked(mul3_out_utry.as_ref(), mul5_out_grad.mat_ref(5), mul6_out_grad.mat_mut(11));
    //             qudit_core::accel::matmul_unchecked(mul3_out_utry.as_ref(), mul5_out_grad.mat_ref(6), mul6_out_grad.mat_mut(12));
    //             qudit_core::accel::matmul_unchecked(mul3_out_utry.as_ref(), mul5_out_grad.mat_ref(7), mul6_out_grad.mat_mut(13));
    //             qudit_core::accel::matmul_unchecked(mul3_out_utry.as_ref(), mul5_out_grad.mat_ref(8), mul6_out_grad.mat_mut(14));
    //             qudit_core::accel::matmul_unchecked(mul3_out_utry.as_ref(), mul5_out_grad.mat_ref(9), mul6_out_grad.mat_mut(15));
    //             qudit_core::accel::matmul_unchecked(mul3_out_utry.as_ref(), mul5_out_grad.mat_ref(10), mul6_out_grad.mat_mut(16));
    //             qudit_core::accel::matmul_unchecked(mul3_out_utry.as_ref(), mul5_out_grad.mat_ref(11), mul6_out_grad.mat_mut(17));
    //             qudit_core::accel::matmul_unchecked(mul3_out_utry.as_ref(), mul5_out_grad.mat_ref(12), mul6_out_grad.mat_mut(18));
    //             qudit_core::accel::matmul_unchecked(mul3_out_utry.as_ref(), mul5_out_grad.mat_ref(13), mul6_out_grad.mat_mut(19));
    //             qudit_core::accel::matmul_unchecked(mul3_out_utry.as_ref(), mul5_out_grad.mat_ref(14), mul6_out_grad.mat_mut(20));
    //             qudit_core::accel::matmul_unchecked(mul3_out_utry.as_ref(), mul5_out_grad.mat_ref(15), mul6_out_grad.mat_mut(21));
    //             qudit_core::accel::matmul_unchecked(mul3_out_utry.as_ref(), mul5_out_grad.mat_ref(16), mul6_out_grad.mat_mut(22));
    //             qudit_core::accel::matmul_unchecked(mul3_out_utry.as_ref(), mul5_out_grad.mat_ref(17), mul6_out_grad.mat_mut(23));
    //         }
    //     }
    //     let duration = start.elapsed();
    //     println!("Native, Time elapsed: {:?}", duration);
    //     println!("Native, Time per call: {:?}", duration / n as u32);
    // }
}
