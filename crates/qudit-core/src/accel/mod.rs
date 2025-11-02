//! Accelerated mathematical operations.

/// Creates match statements for each variable in the second input with patterns in the corresponding 
/// element in the third input. Once it finds a match for all variables in the second input 
/// (a single combination), it executes the first expression. 
/// 
/// # Arguments
/// 
/// * `{x}`: The expression to execute upon matching all elements in the second argument.
/// * `(first_expr, rest_expr)`: A tuple where `first_expr` is the first variable to match against,
///   and `rest_expr` is itself a tuple of remaining variables. We use this structure due to the
///   macro's recursive nature.
/// * `((first_pat, ...), rest_pat)`: A tuple where `first_pat` is a pattern to match against the first variable,
///   and `rest_pat` is a tuple with the remaining patterns stored in a similar recursive structure.
/// 
/// # Example
/// ```
/// // The following call
/// cartesian_match!(
///     {some_function()},
///     (A, (B, ())),
///     ((2, 3, _), ((2, 3, _), ()))
/// );
/// // is equivalent to:
/// match A {
///     2 => {
///         match B {
///             2 => some_function(),
///             3 => some_function(),
///             _ => some_function(),
///         }
///     }
///     3 => {
///         match B {
///             2 => some_function(),
///             3 => some_function(),
///             _ => some_function(),
///         }
///     }
///     _ => {
///         match B {
///             2 => some_function(),
///             3 => some_function(),
///             _ => some_function(),
///         }
///     }
/// }
/// ```
macro_rules! cartesian_match {
    ({ $x: expr }, (), ()) => {
        $x
    };
    (
        { $x: expr },
        ($first_expr: expr, $rest_expr: tt),
        (($($first_pat: pat),* $(,)?), $rest_pat:tt)
    ) => {
        // Creating multiple match arms for the first expression
        match $first_expr {
            $(
                $first_pat => {
                    // Due to this recursive call, `rest_expr` and `rest_pat` needs to be a tuple
                    // with the same structure as the second and third arguments, respectively.
                    cartesian_match!({ $x }, $rest_expr, $rest_pat);
                }
            )*
        }
    };
}

pub(in crate::accel) use cartesian_match;

mod kron;
pub use kron::kron;
pub use kron::kron_sq_unchecked;
pub use kron::kron_unchecked;
pub use kron::kron_add;
pub use kron::kron_kernel_raw;
pub use kron::kron_kernel_add_raw;

mod hadamard;
pub use hadamard::hadamard_kernel_raw;
pub use hadamard::hadamard_kernel_add_raw;

mod frpr;
pub use frpr::fused_reshape_permute_reshape_into;
pub use frpr::fused_reshape_permute_reshape_into_impl;
pub use frpr::fused_reshape_permute_reshape_into_prepare;
pub use frpr::tensor_fused_reshape_permute_reshape_into_prepare;

mod matmul;
pub use matmul::matmul_unchecked;
pub use matmul::MatMulPlan;
// pub use matmul::matmul;


#[cfg(test)]
mod macro_tests {
    use qudit_macro::{complex_tensor64, complex_tensor32};
    use crate::array::Tensor;
    use crate::{c64, c32};
    use std::slice::from_raw_parts;

    #[test]
    fn test_nd_tensor() {
        let attempt = complex_tensor64!([
            [
                [
                    [1.0 + 2.0j, 2.0 + 4.0j, 3.0 + 6.0j], 
                    [4.0 + 8.0j, 5.0 + 10.0j, 6.0 + 12.0j],
                    [7.0 + 14.0j, 8.0 + 16.0j, 9.0 + 18.0j]
                ],
                [
                    [10.0 + 20.0j, 11.0 + 22.0j, 12.0 + 24.0j], 
                    [13.0 + 26.0j, 14.0 + 28.0j, 15.0 + 30.0j],
                    [16.0 + 32.0j, 17.0 + 34.0j, 18.0 + 36.0j]
                ],
                [
                    [19.0 + 38.0j, 20.0 + 40.0j, 21.0 + 42.0j], 
                    [22.0 + 44.0j, 23.0 + 46.0j, 24.0 + 48.0j],
                    [25.0 + 50.0j, 26.0 + 52.0j, 27.0 + 54.0j]
                ]
            ],
            [
                [
                    [28.0 + 56.0j, 29.0 + 58.0j, 30.0 + 60.0j], 
                    [31.0 + 62.0j, 32.0 + 64.0j, 33.0 + 66.0j],
                    [34.0 + 68.0j, 35.0 + 70.0j, 36.0 + 72.0j]
                ],
                [
                    [37.0 + 74.0j, 38.0 + 76.0j, 39.0 + 78.0j], 
                    [40.0 + 80.0j, 41.0 + 82.0j, 42.0 + 84.0j],
                    [43.0 + 86.0j, 44.0 + 88.0j, 45.0 + 90.0j]
                ],
                [
                    [46.0 + 92.0j, 47.0 + 94.0j, 48.0 + 96.0j], 
                    [49.0 + 98.0j, 50.0 + 100.0j, 51.0 + 102.0j],
                    [52.0 + 104.0j, 53.0 + 106.0j, 54.0 + 108.0j]
                ]
            ]
        ]);
        let expected_data: Vec<c64> = (1..=2*3*3*3).map(|i| {
            let in2 = i as f64;
            c64::new(in2 * 1.0, in2 * 2.0)
        }).collect();
        let expected = Tensor::<c64, 4>::from_slice(&expected_data, [2, 3, 3, 3]);

        assert_eq!(attempt.dims(), expected.dims());
        unsafe{
            assert_eq!(from_raw_parts(attempt.as_ptr(), 2*3*3*3), from_raw_parts(expected.as_ptr(), 2*3*3*3));
        }
    }

    #[test]
    fn test_scalar_tensor() {
        let attempt = complex_tensor64!(1.0 - 1.0j + 5.7j + 832.5);
        let expected_data = vec![c64::new(833.5, 4.7)];
        let expected = Tensor::<c64, 0>::from_slice(&expected_data, []);

        assert_eq!(attempt.dims(), expected.dims());
        unsafe{
            assert_eq!(from_raw_parts(attempt.as_ptr(), 1), from_raw_parts(expected.as_ptr(), 1));
        }
    }

    #[test]
    fn test_tensor_with_parentheses() {
        let attempt = complex_tensor64!([(1.0 + 2.0j) * 3.0, -(2.0j)]);
        let expected_data = vec![c64::new(3.0, 6.0), c64::new(0.0, -2.0)];
        let expected = Tensor::<c64, 1>::from_slice(&expected_data, [2]);

        assert_eq!(attempt.dims(), expected.dims());
        unsafe{
            assert_eq!(from_raw_parts(attempt.as_ptr(), 2), from_raw_parts(expected.as_ptr(), 2));
        }
    }

    #[test]
    fn test_tensor_with_function_calls() {

        fn arbitrary_func(x: f64, y: f64) -> f64 {
            return x * y + 9.0 * x;
        }

        let attempt = complex_tensor64!([
            3.0 * arbitrary_func(1.5, 2.0)j + 4.5, 
            -(2.0j + arbitrary_func(5.5, 3.5))
        ]);
        let expected_data = vec![c64::new(4.5, 49.5), c64::new(-68.75, -2.0)];
        let expected = Tensor::<c64, 1>::from_slice(&expected_data, [2]);

        assert_eq!(attempt.dims(), expected.dims());
        unsafe{
            assert_eq!(from_raw_parts(attempt.as_ptr(), 2), from_raw_parts(expected.as_ptr(), 2));
        }
    }

    #[test]
    fn test_32_ver() {

        fn arbitrary_func(x: f32, y: f32) -> f32 {
            return x * y + 9.0 * x;
        }

        let attempt = complex_tensor32!([
            3.0 * arbitrary_func(1.5, 2.0)j + 4.5, 
            -(2.0j + arbitrary_func(5.5, 3.5))
        ]);
        let expected_data = vec![c32::new(4.5, 49.5), c32::new(-68.75, -2.0)];
        let expected = Tensor::<c32, 1>::from_slice(&expected_data, [2]);

        assert_eq!(attempt.dims(), expected.dims());
        unsafe{
            assert_eq!(from_raw_parts(attempt.as_ptr(), 2), from_raw_parts(expected.as_ptr(), 2));
        }
    }

}