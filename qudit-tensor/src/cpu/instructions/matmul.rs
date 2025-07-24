use std::collections::BTreeMap;

use qudit_core::matrix::{MatMut, MatRef};
use qudit_core::array::{TensorRef, TensorMut, SymSqTensorMut, SymSqTensorRef};
use qudit_core::{memory, ComplexScalar, ParamIndices};
use super::super::buffer::SizedTensorBuffer;
use qudit_expr::DifferentiationLevel;
use qudit_expr::{FUNCTION, GRADIENT, HESSIAN};
use qudit_core::memory::MemoryBuffer;
use qudit_core::accel::MatMulPlan;

pub struct MatmulStruct<C: ComplexScalar> {
    left: SizedTensorBuffer<C>,
    right: SizedTensorBuffer<C>,
    out: SizedTensorBuffer<C>,
    // left_offset, r_offset, out_off, dependent_variable, l2_off, r2_off
    grad_offset_map: Vec<(usize, usize, usize, bool, usize, usize)>,
    hess_offset_map: Vec<(usize, usize, usize, bool, usize, usize, usize, usize, usize, usize)>,
    plan: MatMulPlan<C>,
}

impl<C: ComplexScalar> MatmulStruct<C> {
    pub fn new(
        left: SizedTensorBuffer<C>,
        right: SizedTensorBuffer<C>,
        out: SizedTensorBuffer<C>,
        left_param_map: ParamIndices,
        right_param_map: ParamIndices,
    ) -> Self {

        // Calculate grad offset map
        // We loop through all left and right parameters and record the ptrs of
        // matrices that need to be multiplied and where they will be stored.
        let mut offset_map = BTreeMap::new();
        for (i, param) in left_param_map.iter().enumerate() {
            offset_map.insert(
                param,
                (
                    // Location of left partial with respect to i. 
                    left.offset() + left.unit_memory_size()*(i+1),
                    right.offset(),
                    // false => no need to apply product rule (atleast yet)
                    false, 0, 0,
                )
            );
        }
        for (i, param) in right_param_map.iter().enumerate() {
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
        let mut vec = offset_map.into_iter().collect::<Vec<_>>();
        vec.sort();
        let grad_offset_map = vec.into_iter()
            .enumerate()
            .map(|(i, (_, (l_off, r_off, prod, l2_off, r2_off)))| {(
                l_off,
                r_off,
                // Out location is based off sorted order of parameter indices
                out.offset() + out.unit_memory_size()*(i+1),
                prod,
                l2_off,
                r2_off
            )}).collect::<Vec<_>>();

        // Calculate hessian offset map (Same note as above, but more confusing)
        let mut offset_map = BTreeMap::new();
        for (i, param_i) in left_param_map.iter().enumerate() {
            for (j, param_j) in left_param_map.iter().enumerate() {
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

        for (i, param_i) in right_param_map.iter().enumerate() {
            for (j, param_j) in right_param_map.iter().enumerate() {
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

        // Hessian also includes double partials, where the first is take with respect to a
        // parameter in left and the second in right. This is realized as a single partial of
        // left multiplied by a single partial of right.
        for (i, param_i) in left_param_map.iter().enumerate() {
            for (j, param_j) in right_param_map.iter().enumerate() {
                offset_map.entry((param_i, param_j))
                    // If offset_map already contains this then param_i in right and param_j in left
                    // since its shared, I don't do anything here and just skip
                    .or_insert((
                        left.offset() + left.unit_memory_size()*(i+1),
                        right.offset() + right.unit_memory_size()*(j+1),
                        false, 0, 0,
                        0, 0, 0, 0,
                    ));
            }
        }
        let mut vec = offset_map.into_iter().collect::<Vec<_>>();
        vec.sort();
        let hess_offset_map = vec.into_iter()
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
            }).collect::<Vec<_>>();

        let plan = MatMulPlan::new(left.nrows(), right.ncols(), left.ncols());
        Self { left, right, out, grad_offset_map, hess_offset_map, plan }
    }

    #[inline(always)]
    pub fn get_output_buffer(&self) -> &SizedTensorBuffer<C> {
        &self.out
    }

    #[inline(always)]
    unsafe fn matmul(&self, left: *const C, right: *const C, out: *mut C) {
        self.plan.execute_raw_unchecked(
            left,
            right,
            out,
            self.out.row_stride() as isize,
            self.out.col_stride() as isize,
            self.left.row_stride() as isize,
            self.left.col_stride() as isize,
            self.right.row_stride() as isize,
            self.right.col_stride() as isize,
        );
    }

    #[inline(always)]
    unsafe fn batched_matmul(&self, mut left: *const C, mut right: *const C, mut out: *mut C) {
        for _ in 0..self.left.nmats() {
            self.plan.execute_raw_unchecked(
                left,
                right,
                out,
                self.out.row_stride() as isize,
                self.out.col_stride() as isize,
                self.left.row_stride() as isize,
                self.left.col_stride() as isize,
                self.right.row_stride() as isize,
                self.right.col_stride() as isize,
            );
            left = left.add(self.left.mat_stride());
            right = right.add(self.right.mat_stride());
            out = out.add(self.out.mat_stride());
        }
    }

    #[inline(always)]
    unsafe fn matmul_add(&self, left: *const C, right: *const C, out: *mut C) {
        self.plan.execute_add_raw_unchecked(
            left,
            right,
            out,
            self.out.row_stride() as isize,
            self.out.col_stride() as isize,
            self.left.row_stride() as isize,
            self.left.col_stride() as isize,
            self.right.row_stride() as isize,
            self.right.col_stride() as isize,
        );
    }

    #[inline(always)]
    unsafe fn batched_matmul_add(&self, mut left: *const C, mut right: *const C, mut out: *mut C) {
        for _ in 0..self.left.nmats() {
            self.plan.execute_add_raw_unchecked(
                left,
                right,
                out,
                self.out.row_stride() as isize,
                self.out.col_stride() as isize,
                self.left.row_stride() as isize,
                self.left.col_stride() as isize,
                self.right.row_stride() as isize,
                self.right.col_stride() as isize,
            );
            left = left.add(self.left.mat_stride());
            right = right.add(self.right.mat_stride());
            out = out.add(self.out.mat_stride());
        }
    }

    #[inline(always)]
    pub unsafe fn evaluate<const D: DifferentiationLevel>(&self, memory: &mut MemoryBuffer<C>) {
        let left = memory.as_ptr().add(self.left.offset());
        let right = memory.as_ptr().add(self.right.offset());
        let out = memory.as_mut_ptr().add(self.out.offset());
        self.matmul(left, right, out);

        if D >= GRADIENT {
            for (l_off, r_off, o_off, prod, l2_off, r2_off) in &self.grad_offset_map {
                let left = memory.as_ptr().add(*l_off);
                let right = memory.as_ptr().add(*r_off);
                let out = memory.as_mut_ptr().add(*o_off);
                self.matmul(left, right, out);

                if *prod {    
                    let left = memory.as_ptr().add(*l2_off);
                    let right = memory.as_ptr().add(*r2_off);
                    self.matmul_add(left, right, out);
                }
            }
        }

        if D >= HESSIAN {
            for (l_off, r_off, o_off, prod, l2_off, r2_off, l3_off, r3_off, l4_off, r4_off) in &self.hess_offset_map {
                let left = memory.as_ptr().add(*l_off);
                let right = memory.as_ptr().add(*r_off);
                let out = memory.as_mut_ptr().add(*o_off);
                self.matmul(left, right, out);

                if *prod {    
                    let left = memory.as_ptr().add(*l2_off);
                    let right = memory.as_ptr().add(*r2_off);
                    self.matmul_add(left, right, out);

                    let left = memory.as_ptr().add(*l3_off);
                    let right = memory.as_ptr().add(*r3_off);
                    self.matmul_add(left, right, out);

                    let left = memory.as_ptr().add(*l4_off);
                    let right = memory.as_ptr().add(*r4_off);
                    self.matmul_add(left, right, out);
                }
            }
        }
    }
    #[inline(always)]
    pub unsafe fn batched_evaluate<const D: DifferentiationLevel>(&self, memory: &mut MemoryBuffer<C>) {
        let left = memory.as_ptr().add(self.left.offset());
        let right = memory.as_ptr().add(self.right.offset());
        let out = memory.as_mut_ptr().add(self.out.offset());
        self.batched_matmul(left, right, out);

        if D >= GRADIENT {
            for (l_off, r_off, o_off, prod, l2_off, r2_off) in &self.grad_offset_map {
                let left = memory.as_ptr().add(*l_off);
                let right = memory.as_ptr().add(*r_off);
                let out = memory.as_mut_ptr().add(*o_off);
                self.batched_matmul(left, right, out);

                if *prod {    
                    let left = memory.as_ptr().add(*l2_off);
                    let right = memory.as_ptr().add(*r2_off);
                    self.batched_matmul_add(left, right, out);
                }
            }
        }

        if D >= HESSIAN {
            for (l_off, r_off, o_off, prod, l2_off, r2_off, l3_off, r3_off, l4_off, r4_off) in &self.hess_offset_map {
                let left = memory.as_ptr().add(*l_off);
                let right = memory.as_ptr().add(*r_off);
                let out = memory.as_mut_ptr().add(*o_off);
                self.batched_matmul(left, right, out);

                if *prod {    
                    let left = memory.as_ptr().add(*l2_off);
                    let right = memory.as_ptr().add(*r2_off);
                    self.batched_matmul_add(left, right, out);

                    let left = memory.as_ptr().add(*l3_off);
                    let right = memory.as_ptr().add(*r3_off);
                    self.batched_matmul_add(left, right, out);

                    let left = memory.as_ptr().add(*l4_off);
                    let right = memory.as_ptr().add(*r4_off);
                    self.batched_matmul_add(left, right, out);
                }
            }
        }
    }
}

