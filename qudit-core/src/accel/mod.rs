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

mod frpr;
pub use frpr::fused_reshape_permute_reshape_into;
pub use frpr::fused_reshape_permute_reshape_into_impl;
pub use frpr::fused_reshape_permute_reshape_into_prepare;
pub use frpr::tensor_fused_reshape_permute_reshape_into_prepare;

mod matmul;
pub use matmul::matmul_unchecked;
pub use matmul::MatMulPlan;
// pub use matmul::matmul;
