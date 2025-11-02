//! Memory management types and helper functions for the Openqudit library.

use std::mem::size_of;

use aligned_vec::avec;
use aligned_vec::AVec;
use aligned_vec::CACHELINE_ALIGN;
use bytemuck::Zeroable;

/// A trait for types that can be stored in memory buffers.
pub trait Memorable: Sized + Zeroable + Copy {}

impl<T: Sized + Zeroable + Copy> Memorable for T {}

/// An aligned memory buffer.
///
/// The memory buffer is used to store the data for matrices, vectors,
/// and other data structures. This type aliases a group of aligned
/// vectors of the unit type of a given faer entity. For faer's native
/// complex numbers this simplfies to a single vector of complex numbers.
/// For num_complex complex numbers this because two vectors, one for the
/// real part and one for the imaginary part.
///
/// # Type Parameters
///
/// * `C`: The faer entity type for the memory buffer.
///
/// # See Also
///
/// - [alloc_zeroed_memory] function to allocate a new memory buffer.
/// - [faer_entity::Entity] trait for more information on faer entities.
/// - [faer_entity::GroupFor] trait for more information on faer groups.
#[allow(type_alias_bounds)]
pub type MemoryBuffer<C: Memorable> = AVec<C>;

/// A pointer to an element in a memory buffer.
///
/// The pointer is used to access elements in a memory buffer. This type
/// aliases a group of pointers to the unit type of a given faer entity.
/// For faer's native complex numbers this simplfies to a single pointer.
/// For num_complex complex numbers this becomes two pointers, one for the
/// real part and one for the imaginary part.
///
/// # Type Parameters
///
/// * `C`: The faer entity type for the memory buffer.
///
/// # See Also
///
/// - [Memory] type alias for more information on memory buffers.
/// - [faer_entity::Entity] trait for more information on faer entities.
/// - [faer_entity::GroupFor] trait for more information on faer groups.
/// - [PointerMut] type alias for more information on mutable pointers.
/// - [NonNullPointer] type alias for more information on non-null pointers.
#[allow(type_alias_bounds)]
pub type MemoryPointer<C: Memorable> = *const C;

/// A mutable pointer to an element in a memory buffer.
///
/// The mutable pointer is used to access or modify elements in a memory
/// buffer.
///
/// # Type Parameters
///
/// * `C`: The faer entity type for the memory buffer.
///
/// # See Also
///
/// - [Memory] type alias for more information on memory buffers.
/// - [faer_entity::Entity] trait for more information on faer entities.
/// - [faer_entity::GroupFor] trait for more information on faer groups.
/// - [Pointer] type alias for more information on const pointers.
/// - [NonNullPointer] type alias for more information on non-null pointers.
#[allow(type_alias_bounds)]
pub type MemoryPointerMut<C> = *mut C;

/// A const non-null pointer to an element in a memory buffer.
///
/// # Type Parameters
///
/// * `C`: The faer entity type for the memory buffer.
///
/// # See Also
///
/// - [Memory] type alias for more information on memory buffers.
/// - [faer_entity::Entity] trait for more information on faer entities.
/// - [faer_entity::GroupFor] trait for more information on faer groups.
/// - [Pointer] type alias for more information on const pointers.
/// - [PointerMut] type alias for more information on mutable pointers.
#[allow(type_alias_bounds)]
pub type MemoryPointerNonNull<C> = core::ptr::NonNull<C>;

/// Convert a mutable memory pointer to a non-null pointer without checking.
///
/// # Type Parameters
///
/// * `C`: The faer entity type for the memory buffer.
///
/// # Arguments
///
/// * `data` - The mutable memory pointer to convert to a non-null pointer.
///
/// # Returns
///
/// A non-null pointer to the memory buffer.
///
/// # Safety
///
/// The user must ensure that the data is not null.
///
/// # See Also
///
/// - [Memory] type alias for more information on memory buffers.
/// - [PointerMut] type alias for more information on mutable pointers.
/// - [NonNullPointer] type alias for more information on non-null pointers.
/// - [is_null_ptr] function to check if a pointer is null.
// #[inline(always)]
// #[track_caller]
// pub unsafe fn new_non_null_unchecked<C: Entity>(data: PointerMut<C>) -> NonNullPointer<C> {
//     C::faer_map(data,
//         #[inline(always)]
//         |ptr| { core::ptr::NonNull::new_unchecked(ptr) }
//     )
// }

/// Upgrade a memory pointer to a mutable memory pointer.
///
/// # Type Parameters
///
/// * `C`: The faer entity type for the memory buffer.
///
/// # Arguments
///
/// * `data` - The memory pointer to upgrade to a mutable memory pointer.
///
/// # Returns
///
/// A mutable memory pointer to the memory buffer.
///
/// # Safety
///
/// The user must ensure that the rust borrow checking rules are not violated.
///
/// # See Also
///
/// - [Pointer] type alias for more information on const pointers.
/// - [PointerMut] type alias for more information on mutable pointers.
/// - [NonNullPointer] type alias for more information on non-null pointers.
/// - [new_non_null_unchecked] function to convert a mutable pointer to a non-null pointer.
/// - [is_null_ptr] function to check if a pointer is null.
// #[inline(always)]
// #[track_caller]
// pub unsafe fn upgrade_pointer<C: Entity>(data: Pointer<C>) -> PointerMut<C> {
//     C::faer_map(data,
//         #[inline(always)]
//         |ptr| { ptr as *mut C::Unit }
//     )
// }

/// Check if a memory pointer is null.
///
/// # See Also
///
/// - [NonNullPointer] type alias for more information on non-null pointers.
/// - [new_non_null_unchecked] function to convert a mutable pointer to a non-null pointer.
/// - [is_null_ptr] function to check if a pointer is null.
// #[inline(always)]
// #[track_caller]
// pub fn is_null_ptr<C: Entity>(data: PointerMut<C>) -> bool {
//     let mut to_return = false;

//     C::faer_map(data,
//         #[inline(always)]
//         |ptr| { if ptr.is_null() { to_return = true; } },
//     );

//     to_return
// }

/// Allocate a new memory buffer with the given size.
///
/// The size is the number of elements to allocate in the memory buffer,
/// not the number of bytes. The memory buffer is aligned to cachelines,
/// and the elements will be zeroed.
///
/// # Type Parameters
///
/// * `C`: The faer entity type for the memory buffer.
///
/// # Arguments
///
/// * `size` - The number of elements to allocate in the memory buffer.
///
/// # Returns
///
/// A new memory buffer with the given size.
///
/// # Panics
///
/// This function will panic if the memory requirement overflows isize.
///
/// # Example
///
/// ```
/// use qudit_core::memory::alloc_zeroed_memory;
/// let size = 10;
/// let mem = alloc_zeroed_memory::<f32>(size);
/// assert_eq!(mem.len(), size);
/// ```
///
/// # See Also
///
/// - [Memory] type alias for more information on memory buffers.
/// - [faer_entity::Entity] trait for more information on faer entities.
pub fn alloc_zeroed_memory<C: Zeroable + Clone>(size: usize) -> MemoryBuffer<C> {
    let mem_size = size
        .checked_mul(size_of::<C>())
        .expect("Memory size overflows usize");

    if mem_size > isize::MAX as usize {
        panic!("Memory size overflows isize");
    }

    // TODO: This doesn't use calloc yet
    avec![<C as Zeroable>::zeroed(); size]
    // C::faer_map(C::UNIT, |()| avec![<C::Unit as Zeroable>::zeroed(); size])
}

/// Calculate column stride for a matrix with given rows and columns.
///
/// The column stride is the number of elements between the start of each column
/// in the matrix. Faer, the library qudit-core builds on, uses column-major
/// storage for matrices, so the column stride is the major stride for matrices.
/// Extra padding is added to columns to ensure that columns are aligned to
/// cachelines.
///
/// # Type Parameters
///
/// * `C`: The faer entity type for the matrix.
///
/// # Arguments
///
/// * `nrows` - The number of rows in the matrix.
///
/// * `ncols` - The number of columns in the matrix.
///
/// # Returns
///
/// The column stride for the matrix.
///
/// # Panics
///
/// This function will panic if any of the arithmetic operations overflow.
/// This should only happen for extremely large matrices.
///
/// # Example
///
/// ```
/// use qudit_core::memory::calc_col_stride;
/// let nrows = 3;
/// let ncols = 4;
/// let col_stride = calc_col_stride::<f32>(nrows, ncols);
/// assert_eq!(col_stride, 3);
///
/// let nrows = 14;
/// let ncols = 40;
/// let col_stride = calc_col_stride::<f32>(nrows, ncols);
/// assert_eq!(col_stride, 16);
/// ```
///
/// # See Also
///
/// - [calc_mat_stride] function to calculate the matrix stride.
/// - [faer_entity::Entity] trait for more information on faer entities.
///
/// # Notes
///
/// This function assumes that the first element starts at the beginning of
/// a cacheline but does not guarantee that matrices will end at a cacheline
/// boundary. If multiple matrices are stored in a single memory buffer, the
/// memory buffer should be aligned to cachelines, and you should calculate
/// a separate stride for the matrix dimension. See [calc_mat_stride] for
/// more information.
///
/// Additionally, this function returns a usize, while pointer arithmetic
/// is performed with isize types. Extra care should be taken to avoid
/// overflow when using the result of this function in pointer arithmetic.
///
/// This function always assumes the row stride is 1. If the row stride is
/// not 1, the column stride will be incorrect.
pub fn calc_col_stride<C>(nrows: usize, ncols: usize) -> usize {
    if nrows == 0 || ncols == 0 {
        return 0;
    }

    // let unit_size = size_of::<C::Unit>();
    let unit_size = size_of::<C>();

    if unit_size == 0 {
        return 0;
    }

    if nrows == 1 {
        return 1;
    }

    if ncols == 1 {
        return nrows;
    }

    if unit_size > CACHELINE_ALIGN {
        // This shouldn't happen for any reasonable type. If it does, we
        // can't do anything about it, since the following code won't work:
        // ```
        // let col_size = nrows * unit_size;
        // let remainder = CACHELINE_ALIGN - (col_size % CACHELINE_ALIGN);
        // return (col_size + remainder) / unit_size; // Due to this division
        // ```
        // It's guaranteed that col_size + remainder is not divisible by
        // unit_size, and appending empty cache lines seems extremely
        // wasteful (in impl effort and computation).
        return nrows;
    }

    let units_per_cache_line = CACHELINE_ALIGN / unit_size;
    let mat_size = nrows
        .checked_mul(ncols)
        .expect("Matrix size overflows usize");

    if mat_size <= units_per_cache_line {
        // TODO: Pad for SIMD? is it necessary? yes it is
        // If simd gets compiled into binary, then use a const with cfg
        // otherwise do runtime checks
        return nrows;
    }

    if nrows > units_per_cache_line {
        let remainder = units_per_cache_line - (nrows % units_per_cache_line);
        return nrows
            .checked_add(remainder)
            .expect("Column stride overflows usize");
    }

    // We now have the following:
    // - ncols > 1
    // - 1 < nrows < units_per_cache_line
    // - nrows * ncols > units_per_cache_line
    //
    // This means that we can potentially fit multiple columns in a single
    // cache line. We now want to find the number of columns that can fit
    // in a single cache line, and then pad the columns to ensure that they
    // are aligned to cachelines.
    //
    // We start with an initial packed guess:
    let mut ncols_per_line = units_per_cache_line / nrows;

    // We need the padding to be consistent across all columns, so we
    // continue to reduce ncols_per_line until the padding can be made
    // consistent. This happens when the leftover space in the cache line
    // (units_per_cache_line - cols_per_line * nrows) can be evenely split
    // into ncols_per_line pieces. In the worst case, we stop at
    // ncols_per_line = 1, which is the same as the faer implementation.
    let mut left_over = units_per_cache_line - (ncols_per_line * nrows);
    while left_over % ncols_per_line != 0 {
        ncols_per_line -= 1;
        left_over = units_per_cache_line - (ncols_per_line * nrows);
    }

    left_over / ncols_per_line + nrows
}

/// Calculate matrix stride for a tensor with given rows, columns, and column stride.
///
/// The matrix stride is the number of elements between the start of each matrix
/// in a tensor. Extra padding is added to matrices to ensure that matrices are
/// aligned to cachelines.
///
/// # Type Parameters
///
/// * `C`: The faer entity type for the matrix.
///
/// # Arguments
///
/// * `nrows` - The number of rows in the matrix.
///
/// * `ncols` - The number of columns in the matrix.
///
/// * `col_stride` - The column stride for the matrix.
///
/// # Returns
///
/// The matrix stride for the tensor.
///
/// # Example
///
/// ```
/// use qudit_core::memory::calc_mat_stride;
/// let nrows = 3;
/// let ncols = 4;
/// let col_stride = 4;
/// let mat_stride = calc_mat_stride::<f64>(nrows, ncols, col_stride);
/// assert_eq!(mat_stride, 16);
/// ```
///
/// # See Also
///
/// - [calc_col_stride] function to calculate the column stride.
/// - [faer_entity::Entity] trait for more information on faer entities.
/// - [Memory] type alias for more information on memory buffers.
/// - [MatVec] type for more information on three dimension tensors.
///
/// # Notes
///
/// This function assumes that the first element starts at the beginning
/// of a cacheline and will guarantee that matrices will end at a
/// cacheline boundary.
///
/// Additionally, this function returns a usize, while pointer arithmetic
/// is performed with isize types. Extra care should be taken to avoid
/// overflow when using the result of this function in pointer arithmetic.
///
/// This function always assumes the row stride is 1. If the row stride is
/// not 1, the column stride will be incorrect.
pub fn calc_mat_stride<C>(_nrows: usize, ncols: usize, col_stride: usize) -> usize {
    let packed_mat_size = ncols
        .checked_mul(col_stride)
        .expect("Matrix size overflows usize");

    // let unit_size = size_of::<C::Unit>();
    let unit_size = size_of::<C>();

    if unit_size == 0 {
        return 0;
    }

    // TODO: maybe check if all matrices can fit in a single cache line
    // A U3 of c32 has 4 matrices (one fn, 3 grad) and each one takes
    // 4 elements (32 bytes). If all packed that could nicely fit into
    // one 128 byte line or two 64 byte line rather than wasting half
    // or 3/4 of the space depending on line size. Especially since they
    // typically used together, performance should be speed up (speculation)

    if unit_size > CACHELINE_ALIGN {
        // See similar comment in [calc_col_stride].
        return packed_mat_size;
    }

    let units_per_cache_line = CACHELINE_ALIGN / unit_size;

    let remainder = units_per_cache_line - (packed_mat_size % units_per_cache_line);
    if remainder == units_per_cache_line {
        return packed_mat_size;
    }
    packed_mat_size + remainder
}

/// Calculates a subtensor's stride, given its size. We ensure the subtensor
/// is aligned to cachelines.
/// 
/// # Arguments
/// 
/// * `packed_subtensor_size` - The number of elements in the packed subtensor.
/// 
/// # Returns
/// 
/// * The subtensor stride, aligned to cachelines.
/// 
/// # Example
/// ```
/// use qudit_core::memory::calc_next_stride;
/// use qudit_core::c64;
///
/// let stride = calc_next_stride::<c64>(5);
/// let expected = 8;
/// assert_eq!(stride, expected);
/// ```
pub fn calc_next_stride<C>(packed_subtensor_size: usize) -> usize {
    let unit_size = size_of::<C>();

    if unit_size == 0 {
        return 0;
    }

    // See similar comment in [calc_col_stride].
    if unit_size > CACHELINE_ALIGN {
        return packed_subtensor_size;
    }

    let units_per_cache_line = CACHELINE_ALIGN / unit_size;
    let remainder = units_per_cache_line - (packed_subtensor_size % units_per_cache_line);
    
    // The case where the packed subtensor is already aligned to the cacheline
    if remainder == units_per_cache_line {
        return packed_subtensor_size;
    }

    // Adds padding `remainder` to align with the next cacheline.
    packed_subtensor_size + remainder
}

#[cfg(test)]
mod tests {
    use faer::c32;

    use super::*;

    // #[test]
    // fn test_alloc_zeroed_memory() {
    //     let size = 10;
    //     let mem = alloc_zeroed_memory::<f32>(size);
    //     assert_eq!(mem.len(), size);
    //     for i in 0..size {
    //         assert_eq!(mem[i], 0.0);
    //     }
    // }

    #[test]
    fn test_calc_col_stride() {
        let nrows = 3;
        let ncols = 4;
        let col_stride = calc_col_stride::<f32>(nrows, ncols);
        assert_eq!(col_stride, 3);

        let nrows = 14;
        let ncols = 40;
        let col_stride = calc_col_stride::<f32>(nrows, ncols);
        assert_eq!(col_stride, 16);

        let nrows = 4;
        let ncols = 4;
        let col_stride = calc_col_stride::<c32>(nrows, ncols);
        assert_eq!(col_stride, 4);
    }

    #[test]
    fn test_calc_mat_stride() {
        let nrows = 3;
        let ncols = 4;
        let col_stride = 4;
        let mat_stride = calc_mat_stride::<f64>(nrows, ncols, col_stride);
        assert_eq!(mat_stride, 16);
    }
}
