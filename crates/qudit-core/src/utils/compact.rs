use std::any::TypeId;
use std::hash::Hash;
use super::storage::CompactStorage;
use super::LimitedSizeVec;

const INLINE_CAPACITY: usize = 7;

/// A space-efficient vector that stores small collections inline and transitions to heap allocation when needed.
/// 
/// `CompactVec` optimizes memory usage by storing up to INLINE_CAPACITY elements directly inline without heap allocation,
/// automatically transitioning to heap storage when the capacity is exceeded or when inline storage
/// cannot represent certain values.
/// 
/// # Examples
/// 
/// ```
/// # use qudit_circuit::utils::CompactVec;
/// let mut vec: CompactVec<i8> = CompactVec::new();
/// vec.push(1);
/// vec.push(2);
/// vec.push(3);
/// assert_eq!(vec.len(), 3);
/// assert!(vec.is_inline());
/// ```
#[derive(Debug, Clone, Hash)] // TODO: The HASH derive here is invalid.
pub enum CompactVec<T: CompactStorage> {
    /// Inline storage for up to INLINE_CAPACITY elements.
    Inline([T::InlineType; INLINE_CAPACITY], u8),

    /// Heap storage for more than INLINE_CAPACITY elements.
    Heap(LimitedSizeVec<T>),
}

impl<T: CompactStorage> CompactVec<T> {
    /// Creates a new empty `CompactVec` using inline storage.
    /// 
    /// The vector starts with inline storage and can hold up to INLINE_CAPACITY elements
    /// before transitioning to heap allocation.
    /// 
    /// # Examples
    /// 
    /// ```
    /// # use qudit_circuit::utils::CompactVec;
    /// let vec: CompactVec<i8> = CompactVec::new();
    /// assert_eq!(vec.len(), 0);
    /// assert!(vec.is_inline());
    /// ```
    #[inline]
    pub fn new() -> Self {
        CompactVec::Inline([T::InlineType::default(); INLINE_CAPACITY], 0)
    }
    
    /// Transitions the vector from inline storage to heap storage.
    /// 
    /// This method converts all inline-stored elements to their full representation
    /// and moves them to a heap-allocated vector. It's called automatically when
    /// inline storage is insufficient (either due to capacity or representation limits).
    /// 
    /// Returns a mutable reference to the heap vector for immediate use.
    #[inline]
    fn transition_to_heap(&mut self) -> &mut LimitedSizeVec<T> {
        match self {
            CompactVec::Inline(storage, length) => {
                let mut heap_vec = LimitedSizeVec::new();
                for i in 0..*length {
                    heap_vec.push(T::from_inline(storage[i as usize]));
                }
                *self = CompactVec::Heap(heap_vec);
                match self {
                    CompactVec::Heap(vec) => vec,
                    _ => unreachable!(),
                }
            }
            CompactVec::Heap(vec) => vec,
        }
    }
    
    /// Fast push for types where conversion is infallible (i8, u8)
    /// Appends an element to the back of the collection without checking conversion validity.
    /// 
    /// This method provides optimal performance for types with infallible inline conversions
    /// (such as `i8` and `u8`) by skipping runtime conversion checks.
    /// 
    /// # Safety
    /// 
    /// This method should only be used with types where `T::CONVERSION_INFALLIBLE` is true.
    /// Using it with other types may cause debug assertions to fail.
    /// 
    /// # Examples
    /// 
    /// ```
    /// # use qudit_circuit::utils::CompactVec;
    /// let mut vec: CompactVec<i8> = CompactVec::new();
    /// vec.push_unchecked(42);
    /// vec.push_unchecked(-10);
    /// assert_eq!(vec.len(), 2);
    /// ```
    #[inline]
    pub fn push_unchecked(&mut self, value: T) {
        debug_assert!(T::CONVERSION_INFALLIBLE, "push_unchecked only valid for infallible types");
        
        match self {
            CompactVec::Inline(storage, length) => {
                if (*length as usize) < INLINE_CAPACITY {
                    storage[*length as usize] = T::to_inline_unchecked(value);
                    *length += 1;
                } else {
                    self.transition_to_heap().push(value);
                }
            }
            CompactVec::Heap(vec) => {
                vec.push(value);
            }
        }
    }
    
    /// Appends an element to the back of the collection.
    /// 
    /// If the vector is using inline storage and has capacity, the element is stored inline.
    /// If inline storage cannot represent the value or is full, the vector transitions to heap storage.
    /// 
    /// # Examples
    /// 
    /// ```
    /// # use qudit_circuit::utils::CompactVec;
    /// let mut vec: CompactVec<i8> = CompactVec::new();
    /// vec.push(1);
    /// vec.push(2);
    /// assert_eq!(vec.len(), 2);
    /// ```
    #[inline]
    pub fn push(&mut self, value: T) {
        // Fast path for infallible conversions
        if T::CONVERSION_INFALLIBLE {
            self.push_unchecked(value);
            return;
        }
        
        match self {
            CompactVec::Inline(storage, length) => {
                if (*length as usize) < INLINE_CAPACITY {
                    match T::to_inline(value) {
                        Ok(inline_value) => {
                            storage[*length as usize] = inline_value;
                            *length += 1;
                        }
                        Err(original_value) => {
                              self.transition_to_heap().push(original_value);
                        }
                    }
                } else {
                      self.transition_to_heap().push(value);
                }
            }
            CompactVec::Heap(vec) => {
                vec.push(value);
            }
        }
    }
    
    /// Removes the last element from the vector and returns it, or `None` if empty.
    /// 
    /// # Examples
    /// 
    /// ```
    /// # use qudit_circuit::utils::CompactVec;
    /// let mut vec: CompactVec<i8> = CompactVec::new();
    /// vec.push(1);
    /// vec.push(2);
    /// assert_eq!(vec.pop(), Some(2));
    /// assert_eq!(vec.pop(), Some(1));
    /// assert_eq!(vec.pop(), None);
    /// ```
    #[inline]
    pub fn pop(&mut self) -> Option<T> {
        match self {
            CompactVec::Inline(storage, length) => {
                if *length > 0 {
                    *length -= 1;
                    Some(T::from_inline(storage[*length as usize]))
                } else {
                    None
                }
            }
            CompactVec::Heap(vec) => vec.pop(),
        }
    }
    
    /// Returns the number of elements in the vector.
    /// 
    /// # Examples
    /// 
    /// ```
    /// # use qudit_circuit::utils::CompactVec;
    /// let mut vec: CompactVec<i8> = CompactVec::new();
    /// assert_eq!(vec.len(), 0);
    /// vec.push(1);
    /// assert_eq!(vec.len(), 1);
    /// ```
    #[inline]
    pub fn len(&self) -> usize {
        match self {
            CompactVec::Inline(_, length) => *length as usize,
            CompactVec::Heap(vec) => vec.len(),
        }
    }
    
    /// Returns `true` if the vector contains no elements.
    /// 
    /// # Examples
    /// 
    /// ```
    /// # use qudit_circuit::utils::CompactVec;
    /// let mut vec: CompactVec<i8> = CompactVec::new();
    /// assert!(vec.is_empty());
    /// vec.push(1);
    /// assert!(!vec.is_empty());
    /// ```
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    
    /// Returns a copy of the element at the given index, or `None` if the index is out of bounds.
    /// 
    /// # Examples
    /// 
    /// ```
    /// # use qudit_circuit::utils::CompactVec;
    /// let mut vec: CompactVec<i8> = CompactVec::new();
    /// vec.push(10);
    /// vec.push(20);
    /// assert_eq!(vec.get(0), Some(10));
    /// assert_eq!(vec.get(1), Some(20));
    /// assert_eq!(vec.get(2), None);
    /// ```
    #[inline]
    pub fn get(&self, index: usize) -> Option<T> {
        match self {
            CompactVec::Inline(storage, length) => {
                if index < *length as usize {
                    Some(T::from_inline(storage[index]))
                } else {
                    None
                }
            }
            CompactVec::Heap(vec) => vec.get(index).copied(),
        }
    }
    
    /// Returns a copy of the element at the given index without bounds checking.
    /// 
    /// This provides optimal performance when the caller can guarantee the index is valid.
    /// 
    /// # Safety
    /// 
    /// Calling this method with an out-of-bounds index is undefined behavior,
    /// even with a safe type `T`.
    /// 
    /// # Examples
    /// 
    /// ```
    /// # use qudit_circuit::utils::CompactVec;
    /// let mut vec: CompactVec<i8> = CompactVec::new();
    /// vec.push(42);
    /// unsafe {
    ///     assert_eq!(vec.get_unchecked(0), 42);
    /// }
    /// ```
    #[inline]
    pub unsafe fn get_unchecked(&self, index: usize) -> T {
        match self {
            CompactVec::Inline(storage, _) => {
                T::from_inline(*storage.get_unchecked(index))
            }
            CompactVec::Heap(vec) => *vec.get_unchecked(index),
        }
    }
    
    /// Returns `true` if the vector is currently using inline storage.
    /// 
    /// This can be useful for performance-sensitive code that needs to know
    /// whether operations will involve heap allocation or inline array access.
    /// 
    /// # Examples
    /// 
    /// ```
    /// # use qudit_circuit::utils::CompactVec;
    /// let mut vec: CompactVec<i8> = CompactVec::new();
    /// assert!(vec.is_inline());
    /// 
    /// // Still inline after adding elements
    /// for i in 0..7 {
    ///     vec.push(i);
    /// }
    /// assert!(vec.is_inline());
    /// 
    /// // Transitions to heap when capacity is exceeded
    /// vec.push(7);
    /// assert!(!vec.is_inline());
    /// ```
    #[inline]
    pub fn is_inline(&self) -> bool {
        matches!(self, CompactVec::Inline(_, _))
    }
    
    /// Returns the total capacity of the vector.
    /// 
    /// For inline storage, this is always INLINE_CAPACITY. For heap storage, this returns
    /// the current heap capacity which may be larger than the length.
    /// 
    /// # Examples
    /// 
    /// ```
    /// # use qudit_circuit::utils::CompactVec;
    /// let vec: CompactVec<i8> = CompactVec::new();
    /// assert_eq!(vec.capacity(), 7);
    /// 
    /// let mut vec: CompactVec<i8> = CompactVec::new();
    /// for i in 0..8 {  // Force transition to heap
    ///     vec.push(i);
    /// }
    /// assert!(vec.capacity() >= 8);  // Heap capacity may be larger
    /// ```
    #[inline]
    pub fn capacity(&self) -> usize {
        match self {
            CompactVec::Inline(_, _) => INLINE_CAPACITY,
            CompactVec::Heap(vec) => vec.capacity(),
        }
    }
    
    /// Returns the vector's contents as a slice.
    /// 
    /// This method provides zero-cost slice access for types where `T` and `T::InlineType`
    /// are the same (such as `i8` and `u8`). For other types, it panics since the
    /// inline storage cannot be directly interpreted as a slice of `T`.
    /// 
    /// # Panics
    /// 
    /// Panics if `T` and `T::InlineType` are different types, as conversion would be required.
    /// Use `iter()` instead for such types.
    /// 
    /// # Examples
    /// 
    /// ```
    /// # use qudit_circuit::utils::CompactVec;
    /// let mut vec: CompactVec<i8> = CompactVec::new();
    /// vec.push(1);
    /// vec.push(2);
    /// vec.push(3);
    /// let slice = vec.as_slice();
    /// assert_eq!(slice, &[1, 2, 3]);
    /// ```
    #[inline]
    pub fn as_slice(&self) -> &[T] {
        // Check if T and T::InlineType are the same type (zero-cost conversion)
        if TypeId::of::<T>() == TypeId::of::<T::InlineType>() {
            match self {
                CompactVec::Inline(storage, length) => {
                    unsafe {
                        // Safe because T == T::InlineType
                        let ptr = storage.as_ptr() as *const T;
                        std::slice::from_raw_parts(ptr, *length as usize)
                    }
                }
                CompactVec::Heap(vec) => vec.as_slice(),
            }
        } else {
            // TODO: transition to HEAP, then return slice
            panic!("Cannot get slice from inline storage for non-zero-cost types - use iter() instead")
        }
    }
    
    /// Removes and returns the element at position `index`, shifting all elements after it to the left.
    ///
    /// # Panics
    /// Panics if `index` is out of bounds.
    #[inline]
    pub fn remove(&mut self, index: usize) -> T {
        if index >= self.len() {
            panic!("removal index (is {}) should be < len (is {})", index, self.len());
        }
        
        match self {
            CompactVec::Inline(storage, length) => {
                let result = T::from_inline(storage[index]);
                // Shift elements left
                for i in index..(*length as usize - 1) {
                    storage[i] = storage[i + 1];
                }
                *length -= 1;
                result
            }
            CompactVec::Heap(vec) => vec.remove(index),
        }
    }
    
    /// Inserts an element at position `index`, shifting all elements after it to the right.
    ///
    /// # Panics
    /// Panics if `index > len`.
    #[inline]
    pub fn insert(&mut self, index: usize, element: T) {
        if index > self.len() {
            panic!("insertion index (is {}) should be <= len (is {})", index, self.len());
        }
        
        match self {
            CompactVec::Inline(storage, length) => {
                if (*length as usize) >= INLINE_CAPACITY || T::to_inline(element.clone()).is_err() {
                    self.transition_to_heap().insert(index, element);
                } else {
                    // Shift elements right
                    for i in (index..*length as usize).rev() {
                        storage[i + 1] = storage[i];
                    }
                    storage[index] = T::to_inline(element).unwrap_or_else(|_| unreachable!());
                    *length += 1;
                }
            }
            CompactVec::Heap(vec) => vec.insert(index, element),
        }
    }
    
    /// Shortens the vector, keeping the first `len` elements and dropping the rest.
    ///
    /// If `len` is greater than the vector's current length, this has no effect.
    #[inline]
    pub fn truncate(&mut self, len: usize) {
        match self {
            CompactVec::Inline(_, length) => {
                if len < *length as usize {
                    *length = len as u8;
                }
            }
            CompactVec::Heap(vec) => vec.truncate(len),
        }
    }
    
    /// Resizes the vector to the given length.
    ///
    /// If `new_len` is greater than `len`, the vector is extended with clones of `value`.
    /// If `new_len` is less than `len`, the vector is truncated.
    #[inline]
    pub fn resize(&mut self, new_len: usize, value: T) {
        let current_len = self.len();
        if new_len > current_len {
            for _ in current_len..new_len {
                self.push(value.clone());
            }
        } else {
            self.truncate(new_len);
        }
    }
    
    /// Extends the vector by cloning elements from a slice.
    /// 
    /// All elements from the slice are appended to the vector in order.
    /// If the vector is using inline storage and the additional elements
    /// cause it to exceed capacity or cannot be represented inline,
    /// it will transition to heap storage automatically.
    /// 
    /// # Examples
    /// 
    /// ```
    /// # use qudit_circuit::utils::CompactVec;
    /// let mut vec: CompactVec<i8> = CompactVec::new();
    /// vec.extend_from_slice(&[1, 2, 3]);
    /// assert_eq!(vec.len(), 3);
    /// 
    /// vec.extend_from_slice(&[4, 5]);
    /// assert_eq!(vec.len(), 5);
    /// ```
    #[inline]
    pub fn extend_from_slice(&mut self, other: &[T]) {
        for item in other {
            self.push(item.clone());
        }
    }
    
    /// Reserves capacity for at least `additional` more elements.
    ///
    /// For inline storage, this will transition to heap if the requested
    /// additional capacity would exceed the inline capacity of INLINE_CAPACITY elements.
    /// For heap storage, this forwards the call to the underlying vector's
    /// reserve method.
    /// 
    /// # Examples
    /// 
    /// ```
    /// # use qudit_circuit::utils::CompactVec;
    /// let mut vec: CompactVec<i8> = CompactVec::new();
    /// vec.reserve(10);  // This will transition to heap since 10 > INLINE_CAPACITY
    /// assert!(!vec.is_inline());
    /// assert!(vec.capacity() >= 10);
    /// ```
    #[inline]
    pub fn reserve(&mut self, additional: usize) {
        match self {
            CompactVec::Inline(_, _) => {
                let new_capacity = self.len() + additional;

                if self.capacity() >= new_capacity {
                    return;
                }

                if new_capacity > INLINE_CAPACITY {
                    self.transition_to_heap().reserve(additional);
                }
            }
            CompactVec::Heap(vec) => vec.reserve(additional),
        }
    }
    
    /// Clears the vector, removing all elements but keeping allocated capacity.
    /// 
    /// For inline storage, this resets the length to 0 without changing the storage mode.
    /// For heap storage, this forwards the call to the underlying vector, which clears
    /// all elements but preserves the allocated heap capacity.
    /// 
    /// # Examples
    /// 
    /// ```
    /// # use qudit_circuit::utils::CompactVec;
    /// let mut vec: CompactVec<i8> = CompactVec::new();
    /// vec.push(1);
    /// vec.push(2);
    /// vec.push(3);
    /// assert_eq!(vec.len(), 3);
    /// 
    /// vec.clear();
    /// assert_eq!(vec.len(), 0);
    /// assert!(vec.is_empty());
    /// ```
    #[inline]
    pub fn clear(&mut self) {
        match self {
            CompactVec::Inline(_, length) => *length = 0,
            CompactVec::Heap(vec) => vec.clear(),
        }
    }

    /// Returns `true` if the vector contains an element equal to `value`.
    ///
    /// # Examples
    ///
    /// ```
    /// # use qudit_circuit::utils::CompactVec;
    /// let mut vec: CompactVec<i8> = CompactVec::new();
    /// vec.push(1);
    /// vec.push(2);
    /// vec.push(3);
    ///
    /// assert!(vec.contains(&1));
    /// assert!(!vec.contains(&4));
    /// ```
    #[inline]
    pub fn contains(&self, value: &T) -> bool
    where
        T: PartialEq,
    {
        match self {
            CompactVec::Inline(storage, length) => {
                for i in 0..*length as usize {
                    if T::from_inline(storage[i]) == *value {
                        return true;
                    }
                }
                false
            }
            CompactVec::Heap(vec) => vec.contains(value),
        }
    }

    /// Sorts the vector in-place using the natural ordering of elements.
    ///
    /// This uses insertion sort for inline storage (which is efficient for small arrays)
    /// and delegates to the underlying vector's sort method for heap storage.
    ///
    /// # Examples
    ///
    /// ```
    /// # use qudit_circuit::utils::CompactVec;
    /// let mut vec: CompactVec<i8> = CompactVec::new();
    /// vec.extend_from_slice(&[3, 1, 4, 1, 5]);
    /// vec.sort();
    /// assert_eq!(vec.as_slice(), &[1, 1, 3, 4, 5]);
    /// ```
    #[inline]
    pub fn sort(&mut self)
    where
        T: Ord,
    {
        match self {
            CompactVec::Inline(storage, length) => {
                if *length <= 1 { return; }

                for i in 1..*length {
                    let mut j = i as usize;
                    while j > 0 && T::from_inline(storage[j - 1]) > T::from_inline(storage[j]) {
                        storage.swap(j - 1, j);
                        j -= 1;
                    }
                }
            }
            CompactVec::Heap(vec) => vec.sort(),
        }
    }
    
    /// Takes ownership of the vector's contents, leaving the original empty.
    /// 
    /// This is equivalent to `std::mem::replace(self, CompactVec::new())` but more explicit.
    /// The original vector is left in a new, empty state (using inline storage).
    /// 
    /// # Examples
    /// 
    /// ```
    /// # use qudit_circuit::utils::CompactVec;
    /// let mut vec: CompactVec<i8> = CompactVec::new();
    /// vec.extend_from_slice(&[1, 2, 3]);
    /// 
    /// let taken = vec.take();
    /// assert_eq!(taken.len(), 3);
    /// assert_eq!(vec.len(), 0);
    /// assert!(vec.is_empty());
    /// ```
    #[inline]
    pub fn take(&mut self) -> Self {
        std::mem::replace(self, Self::new())
    }
    
    /// Returns a mutable reference to the element at the given index, or `None` if out of bounds.
    /// 
    /// For inline storage with non-zero-cost type conversions, this method will transition
    /// the vector to heap storage to provide mutable access. For zero-cost conversions
    /// (where `T` and `T::InlineType` are the same), it provides direct mutable access
    /// to the inline storage.
    /// 
    /// # Examples
    /// 
    /// ```
    /// # use qudit_circuit::utils::CompactVec;
    /// let mut vec: CompactVec<i8> = CompactVec::new();
    /// vec.push(10);
    /// vec.push(20);
    /// 
    /// if let Some(elem) = vec.get_mut(1) {
    ///     *elem = 25;
    /// }
    /// assert_eq!(vec.get(1), Some(25));
    /// ```
    #[inline]
    pub fn get_mut(&mut self, index: usize) -> Option<&mut T> {
        // Check if T and T::InlineType are the same type (zero-cost conversion)
        match self {
            CompactVec::Inline(storage, length) => {
                if index < *length as usize {
                    if TypeId::of::<T>() == TypeId::of::<T::InlineType>() {
                        unsafe {
                            // Safe because T == T::InlineType
                            let ptr = storage.as_mut_ptr() as *mut T;
                            Some(&mut *ptr.add(index))
                        }
                    } else {
                        // We need to transition to heap for mutable access
                        self.transition_to_heap().get_mut(index)
                    }
                } else {
                    None
                }
            }
            CompactVec::Heap(vec) => vec.get_mut(index),
        }
    }
    
    /// Returns a mutable reference to an element without bounds checking.
    ///
    /// For inline storage with non-zero-cost type conversions, this method will transition
    /// the vector to heap storage to provide mutable access. For zero-cost conversions,
    /// it provides direct unsafe mutable access to the inline storage.
    ///
    /// # Safety
    /// 
    /// Calling this method with an out-of-bounds index is undefined behavior,
    /// even with a safe type `T`.
    /// 
    /// # Examples
    /// 
    /// ```
    /// # use qudit_circuit::utils::CompactVec;
    /// let mut vec: CompactVec<i8> = CompactVec::new();
    /// vec.push(42);
    /// unsafe {
    ///     *vec.get_mut_unchecked(0) = 100;
    ///     assert_eq!(vec.get(0), Some(100));
    /// }
    /// ```
    #[inline]
    pub unsafe fn get_mut_unchecked(&mut self, index: usize) -> &mut T {
        // Check if T and T::InlineType are the same type (zero-cost conversion)
        match self {
            CompactVec::Inline(storage, _) => {
                if TypeId::of::<T>() == TypeId::of::<T::InlineType>() {
                    // Safe because T == T::InlineType and caller guarantees bounds
                    let ptr = storage.as_mut_ptr() as *mut T;
                    &mut *ptr.add(index)
                } else {
                    self.transition_to_heap().get_mut_unchecked(index)
                }
            }
            CompactVec::Heap(vec) => vec.get_mut_unchecked(index),
        }
    }
    
    /// Returns the vector's contents as a mutable slice.
    /// 
    /// For inline storage with non-zero-cost type conversions, this method will transition
    /// the vector to heap storage to provide mutable slice access. For zero-cost conversions
    /// (where `T` and `T::InlineType` are the same), it provides direct mutable access
    /// to the inline storage as a slice.
    /// 
    /// # Examples
    /// 
    /// ```
    /// # use qudit_circuit::utils::CompactVec;
    /// let mut vec: CompactVec<i8> = CompactVec::new();
    /// vec.extend_from_slice(&[1, 2, 3]);
    /// 
    /// let slice = vec.as_slice_mut();
    /// slice[1] = 10;
    /// assert_eq!(vec.get(1), Some(10));
    /// ```
    #[inline]
    pub fn as_slice_mut(&mut self) -> &mut [T] {
        // Check if T and T::InlineType are the same type (zero-cost conversion)
        match self {
            CompactVec::Inline(storage, length) => {
                if TypeId::of::<T>() == TypeId::of::<T::InlineType>() {
                    unsafe {
                        // Safe because T == T::InlineType
                        let ptr = storage.as_mut_ptr() as *mut T;
                        std::slice::from_raw_parts_mut(ptr, *length as usize)
                    }
                } else {
                    self.transition_to_heap().as_mut()
                }
            }
            CompactVec::Heap(vec) => vec.as_mut(),
        }
    }
    
    /// Returns an iterator over the vector's elements.
    /// 
    /// This iterator is optimized for inline storage, avoiding heap allocations
    /// and providing efficient element access for small collections.
    /// 
    /// # Examples
    /// 
    /// ```
    /// # use qudit_circuit::utils::CompactVec;
    /// let mut vec: CompactVec<i8> = CompactVec::new();
    /// vec.extend_from_slice(&[1, 2, 3]);
    /// 
    /// let sum: i8 = vec.iter().sum();
    /// assert_eq!(sum, 6);
    /// 
    /// for (i, value) in vec.iter().enumerate() {
    ///     println!("Element {}: {}", i, value);
    /// }
    /// ```
    #[inline]
    pub fn iter(&self) -> CompactVecIter<T> {
        match self {
            CompactVec::Inline(storage, length) => {
                CompactVecIter::Inline(storage, 0, *length)
            }
            CompactVec::Heap(vec) => {
                CompactVecIter::Heap(vec.iter())
            }
        }
    }
}

impl<T: CompactStorage> Default for CompactVec<T> {
    /// Creates an empty `CompactVec<T>`.
    /// 
    /// This is equivalent to `CompactVec::new()`.
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

/// An iterator over the elements of a `CompactVec`.
/// 
/// This iterator is optimized to handle both inline and heap storage modes efficiently.
/// It's created by calling `iter()` on a `CompactVec`.
#[derive(Debug, Clone)]
pub enum CompactVecIter<'a, T: CompactStorage> {
    /// Iterator over inline storage elements.
    /// 
    /// Contains: (storage reference, current index, total length)
    Inline(&'a [T::InlineType; INLINE_CAPACITY], u8, u8),
    /// Iterator over heap storage elements.
    Heap(std::slice::Iter<'a, T>),
}

impl<'a, T: CompactStorage> Iterator for CompactVecIter<'a, T> {
    type Item = T;
    
    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        match self {
            CompactVecIter::Inline(storage, index, length) => {
                if *index < *length {
                    let result = T::from_inline(storage[*index as usize]);
                    *index += 1;
                    Some(result)
                } else {
                    None
                }
            }
            CompactVecIter::Heap(iter) => iter.next().copied(),
        }
    }
    
    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        match self {
            CompactVecIter::Inline(_, index, length) => {
                let remaining = (*length - *index) as usize;
                (remaining, Some(remaining))
            }
            CompactVecIter::Heap(iter) => iter.size_hint(),
        }
    }
}

impl<'a, T: CompactStorage> ExactSizeIterator for CompactVecIter<'a, T> {}

impl<T> AsMut<[T]> for CompactVec<T>
where
    T: CompactStorage,
{
    /// Returns a mutable slice of the vector's contents.
    /// 
    /// This delegates to `as_slice_mut()`.
    #[inline]
    fn as_mut(&mut self) -> &mut [T] {
        self.as_slice_mut()
    }
}

/// Enables converting a `Vec<T>` into a `CompactVec<T>`.
impl<T> From<Vec<T>> for CompactVec<T>
where
    T: CompactStorage,
{
    /// Creates a `CompactVec` from a standard `Vec`.
    /// 
    /// All elements are moved from the source vector. If the vector has INLINE_CAPACITY or fewer
    /// elements and they can all be represented inline, the result will use inline storage.
    /// Otherwise, it will use heap storage.
    /// 
    /// # Examples
    /// 
    /// ```
    /// # use qudit_circuit::utils::CompactVec;
    /// let vec = vec![1i8, 2, 3];
    /// let compact: CompactVec<i8> = vec.into();
    /// assert_eq!(compact.len(), 3);
    /// assert!(compact.is_inline());
    /// ```
    #[inline]
    fn from(vec: Vec<T>) -> Self {
        if vec.len() > INLINE_CAPACITY {
            Self::Heap(LimitedSizeVec::from(vec))
        } else {
            let mut compact_vec = Self::new();
            for item in vec {
                compact_vec.push(item);
            }
            compact_vec
        }
    }
}

/// Enables converting a fixed-size array `[T; N]` into a `CompactVec<T>`.
impl<T, const N: usize> From<[T; N]> for CompactVec<T>
where
    T: CompactStorage,
{
    /// Creates a `CompactVec` from a fixed-size array.
    /// 
    /// # Examples
    /// 
    /// ```
    /// # use qudit_circuit::utils::CompactVec;
    /// let array = [1i8, 2, 3, 4, 5];
    /// let compact: CompactVec<i8> = array.into();
    /// assert_eq!(compact.len(), 5);
    /// ```
    #[inline]
    fn from(array: [T; N]) -> Self {
        let mut compact_vec = Self::new();
        for item in array {
            compact_vec.push(item);
        }
        compact_vec
    }
}

/// Enables converting a fixed-size array `[T; N]` into a `CompactVec<T>`.
impl<T, const N: usize> From<&[T; N]> for CompactVec<T>
where
    T: CompactStorage,
{
    /// Creates a `CompactVec` from a fixed-size array.
    /// 
    /// # Examples
    /// 
    /// ```
    /// # use qudit_circuit::utils::CompactVec;
    /// let array = [1i8, 2, 3, 4, 5];
    /// let compact: CompactVec<i8> = array.into();
    /// assert_eq!(compact.len(), 5);
    /// ```
    #[inline]
    fn from(array: &[T; N]) -> Self {
        let mut compact_vec = Self::new();
        for item in array {
            compact_vec.push(item.clone());
        }
        compact_vec
    }
}

/// Enables converting a slice `&[T]` into a `CompactVec<T>`.
impl<T> From<&[T]> for CompactVec<T>
where
    T: CompactStorage + Clone,
{
    /// Creates a `CompactVec` from a slice by cloning all elements.
    /// 
    /// # Examples
    /// 
    /// ```
    /// # use qudit_circuit::utils::CompactVec;
    /// let slice = &[1i8, 2, 3];
    /// let compact: CompactVec<i8> = slice.into();
    /// assert_eq!(compact.len(), 3);
    /// ```
    #[inline]
    fn from(slice: &[T]) -> Self {
        let mut compact_vec = Self::new();
        for item in slice {
            compact_vec.push(item.clone());
        }
        compact_vec
    }
}

/// Enables converting a `&Vec<T>` into a `CompactVec<T>`.
impl<T> From<&Vec<T>> for CompactVec<T>
where
    T: CompactStorage + Clone,
{
    /// Creates a `CompactVec` from a vector reference by cloning all elements.
    /// 
    /// This delegates to the `&[T]` implementation.
    #[inline]
    fn from(vec: &Vec<T>) -> Self {
        Self::from(vec.as_slice())
    }
}

/// Enables converting a `CompactVec<T>` into a standard `Vec<T>`.
impl<T> From<CompactVec<T>> for Vec<T>
where
    T: CompactStorage,
{
    /// Creates a standard `Vec` from a `CompactVec`.
    /// 
    /// For inline storage, a new vector is allocated and all elements are copied.
    /// For heap storage, the underlying vector is returned directly when possible.
    /// 
    /// # Examples
    /// 
    /// ```
    /// # use qudit_circuit::utils::CompactVec;
    /// let mut compact: CompactVec<i8> = CompactVec::new();
    /// compact.extend_from_slice(&[1, 2, 3]);
    /// 
    /// let vec: Vec<i8> = compact.into();
    /// assert_eq!(vec, vec![1, 2, 3]);
    /// ```
    #[inline]
    fn from(compact_vec: CompactVec<T>) -> Self {
        match compact_vec {
            CompactVec::Inline(storage, length) => {
                let mut vec = Vec::with_capacity(length as usize);
                for i in 0..length {
                    vec.push(T::from_inline(storage[i as usize]));
                }
                vec
            }
            CompactVec::Heap(vec) => vec.into(),
        }
    }
}

impl<T: CompactStorage + PartialEq> PartialEq for CompactVec<T> {
    fn eq(&self, other: &Self) -> bool {
        if self.len() != other.len() {
            return false;
        }

        self.iter().zip(other.iter()).all(|(a, b)| a == b)
    }
}

impl<T: CompactStorage + Eq> Eq for CompactVec<T> {}

/// An owning iterator over the elements of a `CompactVec`.
///
/// This iterator moves elements out of the vector, consuming it in the process.
/// It's created by calling `into_iter()` on a `CompactVec` or by using a `for` loop.
/// Like `CompactVecIter`, it's optimized to handle both inline and heap storage modes.
pub enum CompactVecIntoIter<T: CompactStorage> {
    /// Iterator that moves out of inline storage elements.
    /// 
    /// Contains: (storage array, current index, total length)
    Inline([T::InlineType; INLINE_CAPACITY], u8, u8),
    /// Iterator that moves out of heap storage elements.
    Heap(<LimitedSizeVec<T> as IntoIterator>::IntoIter),
}

impl<T: CompactStorage> Iterator for CompactVecIntoIter<T> {
    type Item = T;
    
    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        match self {
            CompactVecIntoIter::Inline(storage, index, length) => {
                if *index < *length {
                    let result = T::from_inline(storage[*index as usize]);
                    *index += 1;
                    Some(result)
                } else {
                    None
                }
            }
            CompactVecIntoIter::Heap(iter) => iter.next(),
        }
    }
    
    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        match self {
            CompactVecIntoIter::Inline(_, index, length) => {
                let remaining = (*length - *index) as usize;
                (remaining, Some(remaining))
            }
            CompactVecIntoIter::Heap(iter) => iter.size_hint(),
        }
    }
}

impl<T: CompactStorage> ExactSizeIterator for CompactVecIntoIter<T> {}

impl<T: CompactStorage> FromIterator<T> for CompactVec<T> {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        let mut iter = iter.into_iter();
        let (min_len, max_len) = iter.size_hint();

        // If we know the exact size of the iterator
        if let Some(len) = max_len {
            if len <= INLINE_CAPACITY {
                // If the known size fits within inline capacity:
                // For types with infallible inline conversions (e.g., i8, u8),
                // we can directly populate the inline array for maximum performance.
                if T::CONVERSION_INFALLIBLE {
                    let mut storage = [T::InlineType::default(); INLINE_CAPACITY];
                    for i in 0..len {
                        storage[i] = T::to_inline_unchecked(iter.next().expect("iterator should have enough elements based on size hint"));
                    }
                    return CompactVec::Inline(storage, len as u8);
                }
                // For types with fallible inline conversions, even if the size fits,
                // we cannot pre-populate without checking each item. In this case,
                // we fall through to the general `push` loop, which handles conversions
                // and potential transitions to heap as needed.
            } else {
                // If the known size exceeds inline capacity, it will definitely be a heap-allocated vector.
                // We collect all elements directly into a LimitedSizeVec, leveraging its FromIterator
                // implementation which uses the size hint for efficient allocation.
                return CompactVec::Heap(iter.collect::<LimitedSizeVec<T>>());
            }
        } else {
            // If the exact size is unknown, but the lower bound suggests it's likely to be large,
            // we start with a heap-allocated vector to avoid multiple reallocations during inline-to-heap transition.
            if min_len > INLINE_CAPACITY {
                // Collect into a LimitedSizeVec. Its FromIterator will use min_len for initial capacity.
                return CompactVec::Heap(iter.collect::<LimitedSizeVec<T>>());
            }
        }

        // Default path: This covers cases where a specialized path wasn't taken,
        let mut compact_vec = Self::new();
        for item in iter {
            compact_vec.push(item);
        }
        compact_vec
    }
}

/// Enables using `CompactVec` as a mutable slice through the `AsMut` trait.
impl<T> IntoIterator for CompactVec<T>
where
    T: CompactStorage,
{
    type Item = T;
    type IntoIter = CompactVecIntoIter<T>;

    /// Creates an owning iterator that consumes the vector.
    /// 
    /// # Examples
    /// 
    /// ```
    /// # use qudit_circuit::utils::CompactVec;
    /// let mut vec: CompactVec<i8> = CompactVec::new();
    /// vec.extend_from_slice(&[1, 2, 3]);
    /// 
    /// let doubled: Vec<i8> = vec.into_iter().map(|x| x * 2).collect();
    /// assert_eq!(doubled, vec![2, 4, 6]);
    /// ```
    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        match self {
            CompactVec::Inline(storage, length) => {
                CompactVecIntoIter::Inline(storage, 0, length)
            }
            CompactVec::Heap(vec) => {
                CompactVecIntoIter::Heap(vec.into_iter())
            }
        }
    }
}

impl<'a, T> IntoIterator for &'a CompactVec<T>
where
    T: CompactStorage,
{
    type Item = T;
    type IntoIter = CompactVecIter<'a, T>;

    /// Creates an iterator over references to the vector's elements.
    /// 
    /// This delegates to `iter()`.
    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<'a, T> IntoIterator for &'a mut CompactVec<T>
where
    T: CompactStorage,
{
    type Item = &'a mut T;
    type IntoIter = std::slice::IterMut<'a, T>;

    /// Creates an iterator over mutable references to the vector's elements.
    /// 
    /// For inline storage with non-zero-cost conversions, this will transition
    /// the vector to heap storage.
    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.as_slice_mut().iter_mut()
    }
}

// Specialized implementations for zero-cost conversions (T == T::InlineType)
// These provide optimal performance for i8 and u8 types.
/// Enables treating `CompactVec<i8>` as a slice through the `Deref` trait.
/// 
/// This implementation is only available for `i8` because it has zero-cost
/// conversion with its inline type.
impl std::ops::Deref for CompactVec<i8> {
    type Target = [i8];

    /// Returns a slice view of the vector's contents.
    /// 
    /// This provides zero-cost access to the underlying storage.
    #[inline]
    fn deref(&self) -> &Self::Target {
        match self {
            CompactVec::Inline(storage, length) => {
                unsafe {
                    std::slice::from_raw_parts(storage.as_ptr(), *length as usize)
                }
            }
            CompactVec::Heap(vec) => vec.as_slice(),
        }
    }
}

/// Enables treating `CompactVec<u8>` as a slice through the `Deref` trait.
/// 
/// This implementation is only available for `u8` because it has zero-cost
/// conversion with its inline type.
impl std::ops::Deref for CompactVec<u8> {
    type Target = [u8];

    /// Returns a slice view of the vector's contents.
    /// 
    /// This provides zero-cost access to the underlying storage.
    #[inline]
    fn deref(&self) -> &Self::Target {
        match self {
            CompactVec::Inline(storage, length) => {
                unsafe {
                    std::slice::from_raw_parts(storage.as_ptr(), *length as usize)
                }
            }
            CompactVec::Heap(vec) => vec.as_slice(),
        }
    }
}

/// Enables indexing `CompactVec<i8>` with `[]` syntax.
impl std::ops::Index<usize> for CompactVec<i8> {
    type Output = i8;

    /// Returns a reference to the element at the given index.
    /// 
    /// # Panics
    /// 
    /// Panics if the index is out of bounds.
    #[inline]
    fn index(&self, index: usize) -> &Self::Output {
        match self {
            CompactVec::Inline(storage, length) => {
                if index >= *length as usize {
                    panic!("index out of bounds: the len is {} but the index is {}", *length, index);
                }
                &storage[index]
            }
            CompactVec::Heap(vec) => &vec[index],
        }
    }
}

/// Enables indexing `CompactVec<u8>` with `[]` syntax.
impl std::ops::Index<usize> for CompactVec<u8> {
    type Output = u8;

    /// Returns a reference to the element at the given index.
    /// 
    /// # Panics
    /// 
    /// Panics if the index is out of bounds.
    #[inline]
    fn index(&self, index: usize) -> &Self::Output {
        match self {
            CompactVec::Inline(storage, length) => {
                if index >= *length as usize {
                    panic!("index out of bounds: the len is {} but the index is {}", *length, index);
                }
                &storage[index]
            }
            CompactVec::Heap(vec) => &vec[index],
        }
    }
}

/// Enables using `CompactVec<i8>` where `&[i8]` is expected.
impl AsRef<[i8]> for CompactVec<i8> {
    /// Returns a slice view of the vector's contents.
    #[inline]
    fn as_ref(&self) -> &[i8] {
            self.as_slice()
    }
}

/// Enables using `CompactVec<u8>` where `&[u8]` is expected.
impl AsRef<[u8]> for CompactVec<u8> {
    /// Returns a slice view of the vector's contents.
    #[inline]
    fn as_ref(&self) -> &[u8] {
            self.as_slice()
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    // Helper function to create a test CompactVec with known elements
    fn test_vec_with_elements<T: CompactStorage + Clone>(elements: &[T]) -> CompactVec<T> {
        let mut vec: CompactVec<T> = CompactVec::new();
        for elem in elements {
            vec.push(elem.clone());
        }
        vec
    }

    #[test]
    fn test_new() {
        let vec: CompactVec<i8> = CompactVec::new();
        assert_eq!(vec.len(), 0);
        assert!(vec.is_empty());
        assert!(vec.is_inline());
        assert_eq!(vec.capacity(), INLINE_CAPACITY);
    }

    #[test]
    fn test_default() {
        let vec: CompactVec<i8> = CompactVec::default();
        assert_eq!(vec.len(), 0);
        assert!(vec.is_empty());
        assert!(vec.is_inline());
    }

    #[test]
    fn test_push_inline() {
        let mut vec: CompactVec<i8> = CompactVec::new();
        
        // Test pushing elements within inline capacity
        for i in 0..INLINE_CAPACITY as i8 {
            vec.push(i);
            assert_eq!(vec.len(), (i + 1) as usize);
            assert!(vec.is_inline());
            assert_eq!(vec.get(i as usize), Some(i));
        }
        
        assert_eq!(vec.len(), INLINE_CAPACITY);
        assert!(vec.is_inline());
    }

    #[test]
    fn test_push_transition_to_heap() {
        let mut vec: CompactVec<i8> = CompactVec::new();
        
        // Fill inline storage
        for i in 0..INLINE_CAPACITY as i8 {
            vec.push(i);
        }
        assert!(vec.is_inline());
        
        // This should trigger transition to heap
        vec.push(INLINE_CAPACITY as i8);
        assert!(!vec.is_inline());
        assert_eq!(vec.len(), INLINE_CAPACITY + 1);
        assert!(vec.capacity() >= INLINE_CAPACITY + 1);
        
        // Verify all elements are still accessible
        for i in 0..=INLINE_CAPACITY as i8 {
            assert_eq!(vec.get(i as usize), Some(i));
        }
    }

    #[test]
    fn test_push_unchecked() {
        let mut vec: CompactVec<i8> = CompactVec::new();
        
        vec.push_unchecked(42);
        vec.push_unchecked(-10);
        
        assert_eq!(vec.len(), 2);
        assert_eq!(vec.get(0), Some(42));
        assert_eq!(vec.get(1), Some(-10));
    }

    #[test]
    fn test_pop() {
        let mut vec: CompactVec<i8> = CompactVec::new();
        
        // Pop from empty vector
        assert_eq!(vec.pop(), None);
        
        // Push some elements and pop them
        vec.push(1);
        vec.push(2);
        vec.push(3);
        
        assert_eq!(vec.pop(), Some(3));
        assert_eq!(vec.len(), 2);
        assert_eq!(vec.pop(), Some(2));
        assert_eq!(vec.pop(), Some(1));
        assert_eq!(vec.len(), 0);
        assert_eq!(vec.pop(), None);
    }

    #[test]
    fn test_get_and_bounds() {
        let vec = test_vec_with_elements(&[1i8, 2, 3]);
        
        assert_eq!(vec.get(0), Some(1));
        assert_eq!(vec.get(1), Some(2));
        assert_eq!(vec.get(2), Some(3));
        assert_eq!(vec.get(3), None);
        assert_eq!(vec.get(100), None);
    }

    #[test]
    fn test_get_unchecked() {
        let vec = test_vec_with_elements(&[42i8, -10, 100]);
        
        unsafe {
            assert_eq!(vec.get_unchecked(0), 42);
            assert_eq!(vec.get_unchecked(1), -10);
            assert_eq!(vec.get_unchecked(2), 100);
        }
    }

    #[test]
    fn test_capacity() {
        let mut vec: CompactVec<i8> = CompactVec::new();
        assert_eq!(vec.capacity(), INLINE_CAPACITY);
        
        // Fill to capacity
        for i in 0..INLINE_CAPACITY as i8 {
            vec.push(i);
        }
        assert_eq!(vec.capacity(), INLINE_CAPACITY);
        
        // Transition to heap
        vec.push(INLINE_CAPACITY as i8);
        assert!(vec.capacity() >= INLINE_CAPACITY + 1);
    }

    #[test]
    fn test_remove() {
        let mut vec = test_vec_with_elements(&[1i8, 2, 3, 4, 5]);
        
        // Remove from middle
        assert_eq!(vec.remove(2), 3);
        assert_eq!(vec.len(), 4);
        assert_eq!(vec.get(2), Some(4));
        
        // Remove first element
        assert_eq!(vec.remove(0), 1);
        assert_eq!(vec.len(), 3);
        assert_eq!(vec.get(0), Some(2));
        
        // Remove last element
        assert_eq!(vec.remove(2), 5);
        assert_eq!(vec.len(), 2);
    }

    #[test]
    #[should_panic(expected = "removal index")]
    fn test_remove_out_of_bounds() {
        let mut vec = test_vec_with_elements(&[1i8, 2, 3]);
        vec.remove(3);
    }

    #[test]
    fn test_insert() {
        let mut vec = test_vec_with_elements(&[1i8, 3, 4]);
        
        // Insert at middle
        vec.insert(1, 2);
        assert_eq!(vec.len(), 4);
        assert_eq!(vec.get(0), Some(1));
        assert_eq!(vec.get(1), Some(2));
        assert_eq!(vec.get(2), Some(3));
        assert_eq!(vec.get(3), Some(4));
        
        // Insert at beginning
        vec.insert(0, 0);
        assert_eq!(vec.len(), 5);
        assert_eq!(vec.get(0), Some(0));
        
        // Insert at end
        vec.insert(5, 5);
        assert_eq!(vec.len(), 6);
        assert_eq!(vec.get(5), Some(5));
    }

    #[test]
    #[should_panic(expected = "insertion index")]
    fn test_insert_out_of_bounds() {
        let mut vec = test_vec_with_elements(&[1i8, 2, 3]);
        vec.insert(4, 42);
    }

    #[test]
    fn test_truncate() {
        let mut vec = test_vec_with_elements(&[1i8, 2, 3, 4, 5]);
        
        vec.truncate(3);
        assert_eq!(vec.len(), 3);
        assert_eq!(vec.get(2), Some(3));
        assert_eq!(vec.get(3), None);
        
        // Truncate to larger size should have no effect
        vec.truncate(10);
        assert_eq!(vec.len(), 3);
        
        // Truncate to 0
        vec.truncate(0);
        assert_eq!(vec.len(), 0);
        assert!(vec.is_empty());
    }

    #[test]
    fn test_resize() {
        let mut vec = test_vec_with_elements(&[1i8, 2, 3]);
        
        // Resize larger
        vec.resize(5, 42);
        assert_eq!(vec.len(), 5);
        assert_eq!(vec.get(3), Some(42));
        assert_eq!(vec.get(4), Some(42));
        
        // Resize smaller
        vec.resize(2, 99);
        assert_eq!(vec.len(), 2);
        assert_eq!(vec.get(1), Some(2));
        assert_eq!(vec.get(2), None);
    }

    #[test]
    fn test_extend_from_slice() {
        let mut vec = test_vec_with_elements(&[1i8, 2]);
        
        vec.extend_from_slice(&[3, 4, 5]);
        assert_eq!(vec.len(), 5);
        assert_eq!(vec.get(2), Some(3));
        assert_eq!(vec.get(3), Some(4));
        assert_eq!(vec.get(4), Some(5));
        
        // Extend to force heap transition
        vec.extend_from_slice(&[6, 7, 8]);
        assert_eq!(vec.len(), 8);
        assert!(!vec.is_inline());
    }

    #[test]
    fn test_reserve() {
        let mut vec: CompactVec<i8> = CompactVec::new();
        
        // Reserve within inline capacity
        vec.reserve(5);
        assert!(vec.is_inline());
        
        // Reserve beyond inline capacity
        vec.reserve(10);
        assert!(!vec.is_inline());
        assert!(vec.capacity() >= 10);
    }

    #[test]
    fn test_clear() {
        let mut vec = test_vec_with_elements(&[1i8, 2, 3, 4, 5]);
        let was_inline = vec.is_inline();
        
        vec.clear();
        assert_eq!(vec.len(), 0);
        assert!(vec.is_empty());
        assert_eq!(vec.is_inline(), was_inline); // Storage mode should remain the same
    }

    #[test]
    fn test_contains_method() {
        let mut vec: CompactVec<i8> = CompactVec::new();
        vec.push(1);
        vec.push(2);
        vec.push(3);

        assert!(vec.contains(&1));
        assert!(vec.contains(&2));
        assert!(vec.contains(&3));
        assert!(!vec.contains(&4));
        assert!(!vec.contains(&0));
    }

    #[test]
    fn test_sort() {
        let mut vec = test_vec_with_elements(&[3i8, 1, 4, 1, 5]);
        
        vec.sort();
        assert_eq!(vec.get(0), Some(1));
        assert_eq!(vec.get(1), Some(1));
        assert_eq!(vec.get(2), Some(3));
        assert_eq!(vec.get(3), Some(4));
        assert_eq!(vec.get(4), Some(5));
    }

    #[test]
    fn test_take() {
        let mut vec = test_vec_with_elements(&[1i8, 2, 3]);
        
        let taken = vec.take();
        assert_eq!(taken.len(), 3);
        assert_eq!(vec.len(), 0);
        assert!(vec.is_empty());
        assert!(vec.is_inline());
    }

    #[test]
    fn test_as_slice_i8() {
        let vec = test_vec_with_elements(&[1i8, 2, 3]);
        let slice = vec.as_slice();
        assert_eq!(slice, &[1, 2, 3]);
    }

    #[test]
    fn test_get_mut_i8() {
        let mut vec = test_vec_with_elements(&[10i8, 20, 30]);
        
        if let Some(elem) = vec.get_mut(1) {
            *elem = 25;
        }
        assert_eq!(vec.get(1), Some(25));
    }

    #[test]
    fn test_get_mut_unchecked_i8() {
        let mut vec = test_vec_with_elements(&[42i8]);
        
        unsafe {
            *vec.get_mut_unchecked(0) = 100;
        }
        assert_eq!(vec.get(0), Some(100));
    }

    #[test]
    fn test_as_slice_mut_i8() {
        let mut vec = test_vec_with_elements(&[1i8, 2, 3]);
        
        let slice = vec.as_slice_mut();
        slice[1] = 10;
        assert_eq!(vec.get(1), Some(10));
    }

    #[test]
    fn test_iter() {
        let vec = test_vec_with_elements(&[1i8, 2, 3, 4, 5]);
        
        let collected: Vec<i8> = vec.iter().collect();
        assert_eq!(collected, vec![1, 2, 3, 4, 5]);
        
        let sum: i8 = vec.iter().sum();
        assert_eq!(sum, 15);
    }

    #[test]
    fn test_iter_size_hint() {
        let vec = test_vec_with_elements(&[1i8, 2, 3]);
        let mut iter = vec.iter();
        
        assert_eq!(iter.size_hint(), (3, Some(3)));
        iter.next();
        assert_eq!(iter.size_hint(), (2, Some(2)));
    }

    #[test]
    fn test_into_iter() {
        let vec = test_vec_with_elements(&[1i8, 2, 3]);
        
        let collected: Vec<i8> = vec.into_iter().collect();
        assert_eq!(collected, vec![1, 2, 3]);
    }

    #[test]
    fn test_into_iter_ref() {
        let vec = test_vec_with_elements(&[1i8, 2, 3]);
        
        let collected: Vec<i8> = (&vec).into_iter().collect();
        assert_eq!(collected, vec![1, 2, 3]);
        assert_eq!(vec.len(), 3); // Original should still exist
    }

    #[test]
    fn test_into_iter_mut() {
        let mut vec = test_vec_with_elements(&[1i8, 2, 3]);
        
        for elem in &mut vec {
            *elem *= 2;
        }
        
        assert_eq!(vec.get(0), Some(2));
        assert_eq!(vec.get(1), Some(4));
        assert_eq!(vec.get(2), Some(6));
    }

    #[test]
    fn test_from_vec() {
        let vec = vec![1i8, 2, 3, 4, 5];
        let compact: CompactVec<i8> = vec.into();
        
        assert_eq!(compact.len(), 5);
        assert!(compact.is_inline());
        assert_eq!(compact.get(0), Some(1));
        assert_eq!(compact.get(4), Some(5));
    }

    #[test]
    fn test_from_array() {
        let array = [1i8, 2, 3, 4, 5];
        let compact: CompactVec<i8> = array.into();
        
        assert_eq!(compact.len(), 5);
        assert_eq!(compact.get(0), Some(1));
        assert_eq!(compact.get(4), Some(5));
    }

    #[test]
    fn test_from_array_ref() {
        let array = [1i8, 2, 3, 4, 5];
        let compact: CompactVec<i8> = (&array).into();
        
        assert_eq!(compact.len(), 5);
        assert_eq!(compact.get(0), Some(1));
        assert_eq!(compact.get(4), Some(5));
    }

    #[test]
    fn test_from_slice() {
        let slice = &[1i8, 2, 3];
        let compact: CompactVec<i8> = slice.into();
        
        assert_eq!(compact.len(), 3);
        assert_eq!(compact.get(0), Some(1));
        assert_eq!(compact.get(2), Some(3));
    }

    #[test]
    fn test_into_vec() {
        let compact = test_vec_with_elements(&[1i8, 2, 3]);
        let vec: Vec<i8> = compact.into();
        
        assert_eq!(vec, vec![1, 2, 3]);
    }

    #[test]
    fn test_deref_i8() {
        let vec = test_vec_with_elements(&[1i8, 2, 3]);
        let slice: &[i8] = &vec; // Uses Deref
        
        assert_eq!(slice, &[1, 2, 3]);
    }

    #[test]
    fn test_index_i8() {
        let vec = test_vec_with_elements(&[10i8, 20, 30]);
        
        assert_eq!(vec[0], 10);
        assert_eq!(vec[1], 20);
        assert_eq!(vec[2], 30);
    }

    #[test]
    #[should_panic(expected = "index out of bounds")]
    fn test_index_out_of_bounds() {
        let vec = test_vec_with_elements(&[1i8, 2, 3]);
        let _ = vec[3];
    }

    #[test]
    fn test_as_ref_i8() {
        let vec = test_vec_with_elements(&[1i8, 2, 3]);
        let slice: &[i8] = vec.as_ref();
        
        assert_eq!(slice, &[1, 2, 3]);
    }

    #[test]
    fn test_as_mut_trait() {
        let mut vec = test_vec_with_elements(&[1i8, 2, 3]);
        let slice: &mut [i8] = vec.as_mut();
        
        slice[1] = 42;
        assert_eq!(vec.get(1), Some(42));
    }

    #[test]
    fn test_heap_operations() {
        let mut vec: CompactVec<i8> = CompactVec::new();
        
        // Fill beyond inline capacity to force heap
        for i in 0..10 {
            vec.push(i);
        }
        
        assert!(!vec.is_inline());
        assert_eq!(vec.len(), 10);
        
        // Test operations on heap storage
        assert_eq!(vec.pop(), Some(9));
        assert_eq!(vec.get(5), Some(5));
        
        vec.insert(5, 42);
        assert_eq!(vec.get(5), Some(42));
        
        let removed = vec.remove(5);
        assert_eq!(removed, 42);
    }

    #[test]
    fn test_empty_operations() {
        let mut vec: CompactVec<i8> = CompactVec::new();
        
        assert_eq!(vec.pop(), None);
        assert_eq!(vec.get(0), None);
        assert!(vec.is_empty());
        assert_eq!(vec.len(), 0);
        
        let iter_count = vec.iter().count();
        assert_eq!(iter_count, 0);
    }

    #[test]
    fn test_clone() {
        let vec = test_vec_with_elements(&[1i8, 2, 3]);
        let cloned = vec.clone();
        
        assert_eq!(vec.len(), cloned.len());
        assert_eq!(vec.is_inline(), cloned.is_inline());
        
        for i in 0..vec.len() {
            assert_eq!(vec.get(i), cloned.get(i));
        }
    }

    #[test]
    fn test_u8_specialization() {
        let mut vec: CompactVec<u8> = CompactVec::new();
        vec.extend_from_slice(&[1u8, 2, 3]);
        
        // Test u8 specific deref
        let slice: &[u8] = &vec;
        assert_eq!(slice, &[1u8, 2, 3]);
        
        // Test u8 indexing
        assert_eq!(vec[1], 2u8);
        
        // Test u8 as_ref
        let slice: &[u8] = vec.as_ref();
        assert_eq!(slice, &[1u8, 2, 3]);
    }

    #[test]
    fn test_debug() {
        let vec = test_vec_with_elements(&[1i8, 2, 3]);
        let debug_str = format!("{:?}", vec);
        
        // Just ensure it doesn't panic and produces some output
        assert!(!debug_str.is_empty());
    }

    #[test]
    fn test_from_iterator() {
        let iter = 0..5;
        let vec: CompactVec<i8> = iter.collect();
        assert_eq!(vec.len(), 5);
        assert_eq!(vec.as_slice(), &[0, 1, 2, 3, 4]);

        let empty_iter: std::vec::IntoIter<i8> = Vec::new().into_iter();
        let empty_vec: CompactVec<i8> = empty_iter.collect();
        assert_eq!(empty_vec.len(), 0);
    }

    #[test]
    fn test_transition_preserves_order() {
        let mut vec: CompactVec<i8> = CompactVec::new();
        
        // Add elements that will fill inline storage
        for i in 0..INLINE_CAPACITY {
            vec.push((i * 10) as i8);
        }
        assert!(vec.is_inline());
        
        // Force transition to heap
        vec.push((INLINE_CAPACITY * 10) as i8);
        assert!(!vec.is_inline());
        
        // Verify all elements are in correct order
        for i in 0..=INLINE_CAPACITY {
            assert_eq!(vec.get(i), Some((i as i8) * 10));
        }
    }
}
