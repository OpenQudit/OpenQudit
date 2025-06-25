//! Accelerated mathematical operations.

macro_rules! cartesian_match {
    ({ $x: expr }, (), ()) => {
        $x
    };
    (
        { $x: expr },
        ($first_expr: expr, $rest_expr: tt),
        (($($first_pat: pat),* $(,)?), $rest_pat:tt)
    ) => {
        match $first_expr {
            $(
                $first_pat => {
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

mod frpr;
pub use frpr::fused_reshape_permute_reshape_into;
pub use frpr::fused_reshape_permute_reshape_into_impl;
pub use frpr::fused_reshape_permute_reshape_into_prepare;
pub use frpr::tensor_fused_reshape_permute_reshape_into_prepare;

mod matmul;
pub use matmul::matmul_unchecked;
pub use matmul::MatMulPlan;
// pub use matmul::matmul;
