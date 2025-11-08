use super::{cache_grad_offset_list, cache_hess_offset_list, GradOffsetList, HessOffsetList};
use qudit_core::{ComplexScalar, ParamInfo};

use super::super::buffer::SizedTensorBuffer;
use qudit_core::accel::hadamard_kernel_add_raw;
use qudit_core::accel::hadamard_kernel_raw;
use qudit_core::memory::MemoryBuffer;
use qudit_expr::DifferentiationLevel;
use qudit_expr::{GRADIENT, HESSIAN};

pub struct HadamardStruct<C: ComplexScalar> {
    left: SizedTensorBuffer<C>,
    right: SizedTensorBuffer<C>,
    out: SizedTensorBuffer<C>,
    grad_offset_list: GradOffsetList,
    hess_offset_list: HessOffsetList,
}

impl<C: ComplexScalar> HadamardStruct<C> {
    pub fn new(
        left: SizedTensorBuffer<C>,
        right: SizedTensorBuffer<C>,
        out: SizedTensorBuffer<C>,
        left_param_info: ParamInfo,
        right_param_info: ParamInfo,
    ) -> Self {
        let grad_offset_list =
            cache_grad_offset_list(&left, &right, &out, &left_param_info, &right_param_info);
        let hess_offset_list =
            cache_hess_offset_list(&left, &right, &out, &left_param_info, &right_param_info);
        Self {
            left,
            right,
            out,
            grad_offset_list,
            hess_offset_list,
        }
    }

    #[inline(always)]
    unsafe fn hadamard(&self, left: *const C, right: *const C, out: *mut C) { unsafe {
        hadamard_kernel_raw(
            self.left.nrows(),
            self.left.ncols(),
            out,
            self.out.row_stride() as isize,
            self.out.col_stride() as isize,
            left,
            self.left.row_stride() as isize,
            self.left.col_stride() as isize,
            right,
            self.right.row_stride() as isize,
            self.right.col_stride() as isize,
        );
    }}

    #[inline(always)]
    unsafe fn batched_hadamard(&self, mut left: *const C, mut right: *const C, mut out: *mut C) { unsafe {
        for _ in 0..self.left.nmats() {
            self.hadamard(left, right, out);
            left = left.add(self.left.mat_stride());
            right = right.add(self.right.mat_stride());
            out = out.add(self.out.mat_stride());
        }
    }}

    #[inline(always)]
    unsafe fn hadamard_add(&self, left: *const C, right: *const C, out: *mut C) { unsafe {
        hadamard_kernel_add_raw(
            self.left.nrows(),
            self.left.ncols(),
            out,
            self.out.row_stride() as isize,
            self.out.col_stride() as isize,
            left,
            self.left.row_stride() as isize,
            self.left.col_stride() as isize,
            right,
            self.right.row_stride() as isize,
            self.right.col_stride() as isize,
        );
    }}

    #[inline(always)]
    unsafe fn batched_hadamard_add(
        &self,
        mut left: *const C,
        mut right: *const C,
        mut out: *mut C,
    ) { unsafe {
        for _ in 0..self.left.nmats() {
            self.hadamard_add(left, right, out);
            left = left.add(self.left.mat_stride());
            right = right.add(self.right.mat_stride());
            out = out.add(self.out.mat_stride());
        }
    }}

    #[inline(always)]
    pub unsafe fn evaluate<const D: DifferentiationLevel>(&self, memory: &mut MemoryBuffer<C>) { unsafe {
        let left = memory.as_ptr().add(self.left.offset());
        let right = memory.as_ptr().add(self.right.offset());
        let out = memory.as_mut_ptr().add(self.out.offset());
        self.hadamard(left, right, out);

        if D >= GRADIENT {
            for (l_off, r_off, o_off, prod, l2_off, r2_off) in &self.grad_offset_list {
                let left = memory.as_ptr().add(*l_off);
                let right = memory.as_ptr().add(*r_off);
                let out = memory.as_mut_ptr().add(*o_off);
                self.hadamard(left, right, out);

                if *prod {
                    let left = memory.as_ptr().add(*l2_off);
                    let right = memory.as_ptr().add(*r2_off);
                    self.hadamard_add(left, right, out);
                }
            }
        }

        if D >= HESSIAN {
            for (l_off, r_off, o_off, prod, l2_off, r2_off, l3_off, r3_off, l4_off, r4_off) in
                &self.hess_offset_list
            {
                let left = memory.as_ptr().add(*l_off);
                let right = memory.as_ptr().add(*r_off);
                let out = memory.as_mut_ptr().add(*o_off);
                self.hadamard(left, right, out);

                if *prod {
                    let left = memory.as_ptr().add(*l2_off);
                    let right = memory.as_ptr().add(*r2_off);
                    self.hadamard_add(left, right, out);

                    let left = memory.as_ptr().add(*l3_off);
                    let right = memory.as_ptr().add(*r3_off);
                    self.hadamard_add(left, right, out);

                    let left = memory.as_ptr().add(*l4_off);
                    let right = memory.as_ptr().add(*r4_off);
                    self.hadamard_add(left, right, out);
                }
            }
        }
    }}
    #[inline(always)]
    pub unsafe fn batched_evaluate<const D: DifferentiationLevel>(
        &self,
        memory: &mut MemoryBuffer<C>,
    ) { unsafe {
        let left = memory.as_ptr().add(self.left.offset());
        let right = memory.as_ptr().add(self.right.offset());
        let out = memory.as_mut_ptr().add(self.out.offset());
        self.batched_hadamard(left, right, out);

        if D >= GRADIENT {
            for (l_off, r_off, o_off, prod, l2_off, r2_off) in &self.grad_offset_list {
                let left = memory.as_ptr().add(*l_off);
                let right = memory.as_ptr().add(*r_off);
                let out = memory.as_mut_ptr().add(*o_off);
                self.batched_hadamard(left, right, out);

                if *prod {
                    let left = memory.as_ptr().add(*l2_off);
                    let right = memory.as_ptr().add(*r2_off);
                    self.batched_hadamard_add(left, right, out);
                }
            }
        }

        if D >= HESSIAN {
            for (l_off, r_off, o_off, prod, l2_off, r2_off, l3_off, r3_off, l4_off, r4_off) in
                &self.hess_offset_list
            {
                let left = memory.as_ptr().add(*l_off);
                let right = memory.as_ptr().add(*r_off);
                let out = memory.as_mut_ptr().add(*o_off);
                self.batched_hadamard(left, right, out);

                if *prod {
                    let left = memory.as_ptr().add(*l2_off);
                    let right = memory.as_ptr().add(*r2_off);
                    self.batched_hadamard_add(left, right, out);

                    let left = memory.as_ptr().add(*l3_off);
                    let right = memory.as_ptr().add(*r3_off);
                    self.batched_hadamard_add(left, right, out);

                    let left = memory.as_ptr().add(*l4_off);
                    let right = memory.as_ptr().add(*r4_off);
                    self.batched_hadamard_add(left, right, out);
                }
            }
        }
    }}
}
