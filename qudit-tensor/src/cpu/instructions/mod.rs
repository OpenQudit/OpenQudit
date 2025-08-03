mod frpr;
mod hadamard;
mod kron;
mod matmul;
mod trace;
mod write_cs;
mod write_ss;

pub use frpr::FRPRStruct;
pub use hadamard::HadamardStruct;
pub use kron::KronStruct;
pub use matmul::MatmulStruct;
pub use trace::TraceStruct;
pub use write_cs::ConsecutiveParamSingleWriteStruct;
pub use write_ss::SplitParamSingleWriteStruct;


use super::buffer::SizedTensorBuffer;
use qudit_core::ParamIndices;
use qudit_core::ComplexScalar;
use std::collections::BTreeMap;

/// Captures the offsets into the TNVM memory for one gradient partial calculation.
///
/// Elements:
/// * usize: Offset for first left buffer
/// * usize: Offset for first right buffer
/// * usize: Offset for out buffer
/// * bool: Whether the product rule should be applied (and that there is more work)
/// * usize: Optional offset for second left buffer
/// * usize: Optional offset for second right buffer
///
/// Note: there is no second output buffer since its the same buffer.
type GradOffsetTuple = (usize, usize, usize, bool, usize, usize);

/// List of pointers necessary for a full gradient calculation.
type GradOffsetList = Vec<GradOffsetTuple>;

/// Captures the offsets into the TNVM memory for one hessian partial calculation.
///
/// Elements:
/// * usize: Offset for first left buffer
/// * usize: Offset for first right buffer
/// * usize: Offset for out buffer
/// * bool: Whether the product rule should be applied (and that there is more work)
/// * usize: Optional offset for second left buffer
/// * usize: Optional offset for second right buffer
/// * usize: Optional offset for third left buffer
/// * usize: Optional offset for third right buffer
/// * usize: Optional offset for fourth left buffer
/// * usize: Optional offset for fourth right buffer
///
/// Note: there is no second output buffer since its the same buffer.
type HessOffsetTuple = (usize, usize, usize, bool, usize, usize, usize, usize, usize, usize);

/// List of pointers necessary for a full hessian calculation.
type HessOffsetList = Vec<HessOffsetTuple>;

/// Cache offsets pointing to partials that will be operated on during gradient calculations.
///
/// This is dependent on the left, right, and output buffer sizes and the parameter
/// maps associated with left and right. Whether or not the product rule needs to
/// be applied is tracked as well. Output offsets are calculated such that
/// the output is sorted according to the resulting output parameter map.
fn cache_grad_offset_list<C: ComplexScalar>(
    left: &SizedTensorBuffer<C>,
    right: &SizedTensorBuffer<C>,
    out: &SizedTensorBuffer<C>,
    left_param_map: &ParamIndices,
    right_param_map: &ParamIndices,
) -> GradOffsetList {
    // Calculate grad offset map
    // We loop through all left and right parameters and record the ptrs of
    // matrices that need to be interacted and where they will be stored.
    let mut offset_map = BTreeMap::new();
    for (i, param) in left_param_map.sorted().iter().enumerate() {
        offset_map.insert(
            param,
            (
                // Location of left partial with respect to i. 
                left.offset() + left.unit_memory_size()*(i+1),
                right.offset(),
                // false => no need to apply product rule (at least not yet)
                false, 0, 0,
            )
        );
    }

    for (i, param) in right_param_map.sorted().iter().enumerate() {
        offset_map.entry(param).and_modify(|offs| {
            // This parameter is also in left, need to apply product rule
            offs.2 = true;
            offs.3 = left.offset();
            offs.4 = right.offset() + right.unit_memory_size()*(i+1);
        }).or_insert((
            left.offset(),
            right.offset() + right.unit_memory_size()*(i+1),
            false, 0, 0,
        ));
    }

    // Sort and organize for output
    let mut vec = offset_map.into_iter().collect::<Vec<_>>();
    vec.sort();
    vec.into_iter()
        .enumerate()
        .map(|(i, (_, (l_off, r_off, prod, l2_off, r2_off)))| {(
            l_off,
            r_off,
            // Out location is based off sorted order of parameter indices
            out.offset() + out.unit_memory_size()*(i+1),
            prod,
            l2_off,
            r2_off
        )}).collect::<Vec<_>>()
}

/// Cache offsets pointing to partials that will be operated on during hessian calculations.
///
/// See `[cache_grad_offset_list]` for more info.
fn cache_hess_offset_list<C: ComplexScalar>(
    left: &SizedTensorBuffer<C>,
    right: &SizedTensorBuffer<C>,
    out: &SizedTensorBuffer<C>,
    left_param_map: &ParamIndices,
    right_param_map: &ParamIndices,
) -> HessOffsetList {
    let mut offset_map = BTreeMap::new();
    for (i, param_i) in left_param_map.sorted().iter().enumerate() {
        for (j, param_j) in left_param_map.sorted().iter().enumerate() {
            // Only upper right triangle of hessian is stored since its a symmetric square
            if param_i > param_j {
                continue
            }
            let k = if i <= j { j * (j + 1) / 2 + i } else { i * (i + 1) / 2 + j };
            offset_map.insert(
                (param_i, param_j),
                (
                    // Location of left partial with respect to i then j. 
                    left.offset() + left.grad_memory_size() + left.unit_memory_size()*(k+1),
                    right.offset(),
                    false, 0, 0,
                    // Location of left partial with respect to i
                    left.offset() + left.unit_memory_size()*(i+1),
                    0,
                    // Location of left partial with respect to j
                    left.offset() + left.unit_memory_size()*(j+1),
                    0,
                )
            );
        }
    }

    for (i, param_i) in right_param_map.sorted().iter().enumerate() {
        for (j, param_j) in right_param_map.sorted().iter().enumerate() {
            if param_i > param_j {
                continue
            }
            let k = if i <= j { j * (j + 1) / 2 + i } else { i * (i + 1) / 2 + j };
            offset_map.entry((param_i, param_j))
                .and_modify(|offs| {
                    offs.2 = true;
                    offs.3 = left.offset();
                    offs.4 = right.offset() + right.grad_memory_size() + right.unit_memory_size()*(k+1);
                    offs.6 = right.offset() + right.unit_memory_size()*(j+1);
                    offs.8 = right.offset() + right.unit_memory_size()*(i+1);
                }).or_insert((
                    left.offset(),
                    right.offset() + right.grad_memory_size() + right.unit_memory_size()*(k+1),
                    false, 0, 0,
                    0,
                    right.offset() + right.unit_memory_size()*(j+1),
                    0,
                    right.offset() + right.unit_memory_size()*(i+1),
                ));
        }
    }

    // Hessian also includes double partials, where the first is taken with respect to a
    // parameter in left and the second in right. This is realized as a single partial of
    // left multiplied by a single partial of right.
    for (i, param_i) in left_param_map.sorted().iter().enumerate() {
        for (j, param_j) in right_param_map.sorted().iter().enumerate() {
            offset_map.entry((param_i, param_j))
                // If offset_map already contains this then param_i is in right and
                // param_j is in left. Since its shared, I don't do anything
                // here and just skip
                .or_insert((
                    left.offset() + left.unit_memory_size()*(i+1),
                    right.offset() + right.unit_memory_size()*(j+1),
                    false, 0, 0,
                    0, 0, 0, 0,
                ));
        }
    }

    // Sort and organize for output
    let mut vec = offset_map.into_iter().collect::<Vec<_>>();
    vec.sort();
    vec.into_iter()
        .enumerate()
        .map(|(k, (_, (l_off, r_off, prod, l2_off, r2_off, l3_off, r3_off, l4_off, r4_off)))| {
            (
                l_off,
                r_off,
                out.offset() + out.grad_memory_size() + out.unit_memory_size()*(k+1),
                prod,
                l2_off,
                r2_off,
                l3_off,
                r3_off,
                l4_off,
                r4_off,
            )
        }).collect::<Vec<_>>()
}

