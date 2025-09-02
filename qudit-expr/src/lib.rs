mod qgl;
mod expressions;
mod codegen;
mod analysis;
mod shape;
mod cache;
pub mod index;

pub use expressions::Constant;
pub use expressions::Expression;
pub use expressions::ComplexExpression;
pub use expressions::UnitaryExpression;
pub use expressions::TensorExpression;
pub use expressions::ExpressionGenerator;
pub use codegen::DifferentiationLevel;
pub use codegen::{FUNCTION, GRADIENT, HESSIAN};
pub use codegen::ModuleBuilder;
pub use codegen::Module;
pub use codegen::WriteFunc;
pub use codegen::UtryFunc;
pub use codegen::UtryGradFunc;
pub use codegen::CodeGenerator;
pub use analysis::simplify_matrix_and_matvec;
pub use shape::GenerationShape;
pub use cache::ExpressionCache;
pub use cache::ExpressionId;


// use analysis::{check_equality, extract_best_sine};

// #[allow(dead_code)]
// fn formal_equivalence_check(guess: &UnitaryExpression, target: &UnitaryExpression) -> bool {
//     let mut lhs_vec = Vec::new();
//     let mut rhs_vec = Vec::new();
//     for i in 0..guess.body.len() {
//         for j in 0..guess.body[i].len() {
//             lhs_vec.push(&guess.body[i][j].real);
//             lhs_vec.push(&guess.body[i][j].imag);
//             rhs_vec.push(&target.body[i][j].real);
//             rhs_vec.push(&target.body[i][j].imag);
//         }
//     }
//     check_many_equality(&lhs_vec, &rhs_vec)
// }

// #[allow(dead_code)]
// fn formal_equivalence_check_with_phase(guess: &UnitaryExpression, target: &UnitaryExpression, phase: &Expression) -> bool {
//     let cis = ComplexExpression { real: Expression::Cos(Box::new(phase.clone())), imag: Expression::Sin(Box::new(phase.clone())) };
//     let mut products = Vec::new();
//     let mut lhs_vec = Vec::new();
//     let mut rhs_vec = Vec::new();
//     for i in 0..guess.body.len() {
//         for j in 0..guess.body[i].len() {
//             let product = &target.body[i][j] * &cis;
//             products.push(product);
//         }
//     }

//     for i in 0..guess.body.len() {
//         for j in 0..guess.body[i].len() {
//             lhs_vec.push(&products[products.len()-1].real);
//             lhs_vec.push(&products[products.len()-1].imag);
//             rhs_vec.push(&guess.body[i][j].real);
//             rhs_vec.push(&guess.body[i][j].imag);
//         }
//     }

//     check_many_equality(&lhs_vec, &rhs_vec)
// }

// /// if congruent, guess = e^(i*phase)*target
// #[allow(dead_code)]
// fn formal_congruence_check_with_phase(guess: &UnitaryExpression, target: &UnitaryExpression) -> Option<Expression> {
//     let mut non_zero_row_index = None;
//     let mut non_zero_col_index = None;
//     for i in 0..guess.body.len() {
//         for j in 0..guess.body.len() {
//             if !guess.body[i][j].is_zero_fast() && !target.body[i][j].is_zero_fast() {
//                 non_zero_row_index = Some(i);
//                 non_zero_col_index = Some(j);
//                 break;
//             }
//         }
//     }
//     let non_zero_row_index = match non_zero_row_index {
//         Some(i) => i,
//         None => return None,
//     };
//     let non_zero_col_index = match non_zero_col_index {
//         Some(j) => j,
//         None => return None,
//     };

//     // println!("Finding phase difference");
//     let quotient = &guess.body[non_zero_row_index][non_zero_col_index] / &target.body[non_zero_row_index][non_zero_col_index];
//     // println!("Quotient: {:?}", quotient.simplify());
//     let phase = if quotient.real == Expression::from_int(1) {
//         // println!("Phase is 0");
//         Box::new(Expression::from_int(0))
//     } else if quotient.real == Expression::from_int(-1) {
//         // println!("Phase is pi");
//         Box::new(Expression::Pi)
//     } else {
//         // println!("Finding sine");
//         let sine = extract_best_sine(quotient.imag.clone());
//         if sine.is_none() {
//             // println!("Quotient: {:?}", quotient);
//             // println!("No sine found; system failure, re-evaluate life-choices");
//             // panic!("No sine found");
//             return None;
//         }
//         let sine = sine.unwrap();
//         match sine {
//             Expression::Sin(phase) => phase,
//             _ => panic!("Expected sine expression"),
//         }
//     };

//     // println!("Checking unitaries");
//     let cis = ComplexExpression { real: Expression::Cos(phase.clone()), imag: Expression::Sin(phase.clone()) };
//     for i in 0..guess.body.len() {
//         for j in 0..guess.body[i].len() {
//             let product = &target.body[i][j] * &cis;
//             if !check_equality(&product.real, &guess.body[i][j].real) || !check_equality(&product.imag, &guess.body[i][j].imag) {
//                 return None;
//             }
//         }
//     }

//     Some(*phase)
// }

// #[allow(dead_code)]
// fn formal_congruence_check(guess: &UnitaryExpression, target: &UnitaryExpression) -> bool {
//     formal_congruence_check_ratio(guess, target)
//     // formal_congruence_check_determinant(guess, target)
// }

// #[allow(dead_code)]
// fn formal_congruence_check_ratio(guess: &UnitaryExpression, target: &UnitaryExpression) -> bool {
//     // find first pair of non-zero elements
//     // println!("Finding 1st non-zero pair");
//     let mut non_zero_row_index = None;
//     let mut non_zero_col_index = None;
//     for i in 0..guess.body.len() {
//         for j in 0..guess.body.len() {
//             if !guess.body[i][j].is_zero_fast() && !target.body[i][j].is_zero_fast() {
//                 non_zero_row_index = Some(i);
//                 non_zero_col_index = Some(j);
//                 break;
//             }
//         }
//     }
//     let non_zero_row_index = match non_zero_row_index {
//         Some(i) => i,
//         None => return false,
//     };
//     let non_zero_col_index = match non_zero_col_index {
//         Some(j) => j,
//         None => return false,
//     };

//     // println!("Finding phase difference");
//     let quotient = &guess.body[non_zero_row_index][non_zero_col_index] / &target.body[non_zero_row_index][non_zero_col_index];
//     // println!("Quotient: {:?}", quotient.simplify());
//     let phase = if quotient.real == Expression::from_int(1) {
//         // println!("Phase is 0");
//         Box::new(Expression::from_int(0))
//     } else if quotient.real == Expression::from_int(-1) {
//         // println!("Phase is pi");
//         Box::new(Expression::Pi)
//     } else {
//         // println!("Finding sine");
//         let sine = extract_best_sine(quotient.imag.clone());
//         if sine.is_none() {
//             // println!("Quotient: {:?}", quotient);
//             // println!("No sine found; system failure, re-evaluate life-choices");
//             // panic!("No sine found");
//             return false;
//         }
//         let sine = sine.unwrap();
//         match sine {
//             Expression::Sin(phase) => phase,
//             _ => panic!("Expected sine expression"),
//         }
//     };

//     // println!("Checking unitaries");
//     let cis = ComplexExpression { real: Expression::Cos(phase.clone()), imag: Expression::Sin(phase.clone()) };
//     for i in 0..guess.body.len() {
//         for j in 0..guess.body[i].len() {
//             let product = &target.body[i][j] * &cis;
//             if !check_equality(&product.real, &guess.body[i][j].real) || !check_equality(&product.imag, &guess.body[i][j].imag) {
//                 return false;
//             }
//         }
//     }

//     true
//     // let mut products = Vec::new();
//     // let mut lhs_vec = Vec::new();
//     // let mut rhs_vec = Vec::new();
//     // for i in 0..guess.body.len() {
//     //     for j in 0..guess.body[i].len() {
//     //         let product = &target.body[i][j] * &cis;
//     //         products.push(product);
//     //     }
//     // }

//     // for i in 0..guess.body.len() {
//     //     for j in 0..guess.body[i].len() {
//     //         lhs_vec.push(&products[products.len()-1].real);
//     //         lhs_vec.push(&products[products.len()-1].imag);
//     //         rhs_vec.push(&guess.body[i][j].real);
//     //         rhs_vec.push(&guess.body[i][j].imag);
//     //     }
//     // }

//     // check_many_equality(&lhs_vec, &rhs_vec)
// }

// #[allow(dead_code)]
// fn formal_congruence_check_determinant(guess: &UnitaryExpression, target: &UnitaryExpression) -> bool {
//     let det_guess = guess.determinant().simplify();
//     let phase_guess = if det_guess.real.is_one() {
//         Box::new(Expression::from_int(0))
//     } else if Expression::Neg(Box::new(det_guess.real.clone())).is_one() {
//         Box::new(Expression::Pi)
//     } else {
//         let sine = extract_best_sine(det_guess.imag.clone());
//         if sine.is_none() {
//             println!("Guess determinant: {:?}", det_guess);
//             println!("No sine found; system failure, re-evaluate life-choices");
//             return false;
//         }
//         let sine = sine.unwrap();
//         match sine {
//             Expression::Sin(phase) => phase,
//             _ => panic!("Expected sine expression"),
//         }
//     };

//     let det_target = target.determinant().simplify();
//     let phase_target = if det_target.real.is_one() {
//         Box::new(Expression::from_int(0))
//     } else if Expression::Neg(Box::new(det_target.real.clone())).is_one() {
//         Box::new(Expression::Pi)
//     } else {
//         let sine = extract_best_sine(det_target.imag.clone());
//         if sine.is_none() {
//             println!("Target determinant: {:?}", det_target);
//             println!("No sine found; system failure, re-evaluate life-choices");
//             return false;
//         }
//         let sine = sine.unwrap();
//         match sine {
//             Expression::Sin(phase) => phase,
//             _ => panic!("Expected sine expression"),
//         }
//     };

//     let phase_diff = Expression::Div(Box::new(*phase_guess - *phase_target), Box::new(Expression::from_int(guess.dimension() as i64)));
//     let phase_diff = phase_diff.simplify();
//     let cis = ComplexExpression { real: Expression::Cos(Box::new(phase_diff.clone())), imag: Expression::Sin(Box::new(phase_diff.clone())) };

//     // Check if all elements differ by the same global phase
//     for i in 0..guess.body.len() {
//         for j in 0..guess.body[i].len() {
//             let product = &target.body[i][j] * &cis;
//             if !check_equality(&product.real, &guess.body[i][j].real) || !check_equality(&product.imag, &guess.body[i][j].imag) {
//                 return false;
//             }
//         }
//     }

//     true
// }

// #[cfg(test)]

// mod tests {
//     use super::*;

//     #[test]
//     fn global_phase_det() {
//         let u1 = UnitaryExpression::new(String::from("
//             utry U1(λ) {
//                 [
//                     [ 1, 0 ],
//                     [ 0, e^(i * λ) ]
//                 ]
//             }
//         "));
//         let rz = UnitaryExpression::new(String::from("
//             utry RZ(λ) {
//                 [
//                     [ e^(~i * λ / 2), 0 ],
//                     [ 0, e^(i * λ / 2) ]
//                 ]
//             }
//         "));
//         let u1 = u1.alpha_rename(0);
//         let rz = rz.alpha_rename(0);
//         let det_u1 = u1.determinant().simplify();
//         // let det_rz = rz.determinant().simplify();
        
//         let sine = extract_best_sine(det_u1.imag.clone());
//         assert!(sine.is_some());
//         let sine = sine.unwrap();
//         let u1_phase = match sine {
//             Expression::Sin(ref phase) => phase,
//             _ => panic!("Expected sine expression"),
//         };
//         let u1_phase = Expression::Div(u1_phase.clone(), Box::new(Expression::from_int(2)));
//         println!("{:?}", u1_phase);

//         // if 1 then 0, if -1 then π
//         // let sine = extract_best_sine(det_rz.imag.clone());
//         // assert!(sine.is_some());
//         // let sine = sine.unwrap();
//         // let rz_phase = match sine {
//         //     Expression::Sin(ref phase) => phase,
//         //     _ => panic!("Expected sine expression"),
//         // };
//         // println!("{:?}", rz_phase);
        
//         let cis = ComplexExpression { real: Expression::Cos(Box::new(u1_phase.clone())), imag: Expression::Sin(Box::new(u1_phase.clone())) };
//         for i in 0..u1.body.len() {
//             for j in 0..u1.body[i].len() {
//                 let product = &rz.body[i][j] * &cis;
//                 assert!(check_equality(&product.real, &u1.body[i][j].real));
//                 assert!(check_equality(&product.imag, &u1.body[i][j].imag));
//             }
//         }
//     }

//     #[test]
//     fn global_phase_congruence() {
//         // u1(λ) = e^(i * λ / 2)*RZ(λ)
//         let u1 = UnitaryExpression::new(String::from("
//             utry U1(λ) {
//                 [
//                     [ 1, 0 ],
//                     [ 0, e^(i * λ) ]
//                 ]
//             }
//         "));
//         let rz = UnitaryExpression::new(String::from("
//             utry RZ(λ) {
//                 [
//                     [ e^(~i * λ / 2), 0 ],
//                     [ 0, e^(i * λ / 2) ]
//                 ]
//             }
//         "));

//         // Check num of variables match, then normalize variable names
//         // use first pair of non-zero elements to find potential global phase
//         //     - if no pair on non-zero elements: not global phase congruent
//         //     - if pair exists: but unable to find global phase: not global phase congruent
//         // println!("{:?}", &u1.body[0][0]/&rz.body[0][0]);
//         let quotient = &u1.body[0][0]/&rz.body[0][0];
//         let sine = extract_best_sine(quotient.imag.clone());
//         assert!(sine.is_some());
//         let sine = sine.unwrap();
//         let phase = match sine {
//             Expression::Sin(ref phase) => phase,
//             _ => panic!("Expected sine expression"),
//         };
//         println!("{:?}", phase);
//         let cosine = Expression::Cos(phase.clone());
//         assert!(check_equality(&quotient.real, &cosine));
//         let cis = ComplexExpression { real: cosine, imag: sine };

//         // Check if all elements differ by the same global phase
//         for i in 0..u1.body.len() {
//             for j in 0..u1.body[i].len() {
//                 println!("{:?}", &rz.body[i][j]);
//                 println!("{:?}", &u1.body[i][j]);
//                 let product = &rz.body[i][j] * &cis;
//                 assert!(check_equality(&product.real, &u1.body[i][j].real));
//                 assert!(check_equality(&product.imag, &u1.body[i][j].imag));
//             }
//         }
//     }

//     #[test]
//     fn parameter_function_finding() {
//         let u1 = UnitaryExpression::new(String::from("
//             utry U1(λ) {
//                 [
//                     [ 1, 0 ],
//                     [ 0, e^(i * λ) ]
//                 ]
//             }
//         "));
//         let u1d2 = UnitaryExpression::new(String::from("
//             utry U1(λ) {
//                 [
//                     [ 1, 0 ],
//                     [ 0, e^(i * λ / 2) ]
//                 ]
//             }
//         "));
//         // Note representing the complex expression component wise,
//         // makes it easier to find meaningful GCAs, since if we represent this as complex numbers
//         // internal, the GCA of U1 wrt to λ would be e^(i * λ), but we can't substitute this with
//         // another variable since that would make the entire body non-unitary. With components,

//         // this is accurately figured out, since cos and sin components are not common.        
//         // for i in 0..u1.body.len() {
//         //     for j in 0..u1.body[i].len() {
//         //         // println!("{:?}", &u1.body[i][j]);
//         //         println!("{:?}", &u1d2.body[i][j]);
//         //     }
//         // }
//         // println!("{:?}", u1d2.greatest_common_ancestors());

//         // Have to go a different approach because there may be no common ancestor among elements
//         // but still exist a substitution that exists. For example diag(e^(2iθ), e^(iθ)) and diag(e^(iθ), e^(iθ/2))
//         // no common ancestor, but I can still sub θ for θ/2.
//         //
//         // Find first pair of elements that has the variable in question in common
//         // - if no pair exists: not parameter congruent
//         //
//         // Then find substitution by "unshelling" the expressions pairwise until one becomes the
//         // variable

//         let e1 = u1.body[1][1].real.clone();
//         let e2 = u1d2.body[1][1].real.clone();
//         // println!("{:?}", e1);
//         // println!("{:?}", e2);
//         // check that e1 discriminant is same as e2:
//         let (substitution, original) = match (&e1, &e2) {
//             (Expression::Cos(e1), Expression::Cos(e2)) => {
//                 // println!("cos");
//                 println!("{:?} = {:?}", e1, e2);
//                 (Some(e1.clone()), Some(e2.clone()))
//             },
//             _ => panic!("Expected cosine expression"),
//         };

//         let original = original.unwrap();
//         let substitution = substitution.unwrap();

//         e2.substitute(&original, &substitution);
//         println!("{:?}", e1);
//         println!("{:?}", e2);

//         let e1 = u1.body[1][1].imag.clone();
//         let e2 = u1d2.body[1][1].imag.clone();
//         e2.substitute(&original, &substitution);
//         println!("{:?}", e1);
//         println!("{:?}", e2);
//     }
// }
