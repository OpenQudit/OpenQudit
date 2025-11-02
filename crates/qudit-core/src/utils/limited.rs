use std::alloc::{alloc, dealloc, Layout};
use std::fmt;
use std::hash::Hash;
use std::ptr::NonNull;

/// A vector-like container with 32-bit length and capacity limits.
///
/// `LimitedSizeVec` is a performance-optimized vector implementation that uses
/// `u32` for length and capacity instead of `usize`, limiting the maximum
/// number of elements to 2³² - 1 (approximately 4.3 billion elements).
/// This reduces memory overhead on 64-bit systems where the smaller size
/// representation can improve cache performance.
///
/// # Memory Layout
///
/// The struct uses manual memory management with raw pointers for optimal
/// performance. Memory is allocated using the global allocator and grows
/// by doubling capacity when needed.
///
/// # Safety
///
/// This implementation maintains memory safety through careful pointer
/// arithmetic and proper Drop implementation. All unsafe operations are
/// encapsulated within safe public methods.
///
/// # Limits
///
/// - Maximum capacity: `u32::MAX` (4,294,967,295 elements)
/// - Initial capacity: 8 elements (when using `new()`)
/// - Growth factor: 2x when capacity is exceeded
///
/// # Examples
///
/// ```rust
/// # use qudit_core::LimitedSizeVec;
///
/// // Create a new vector
/// let mut vec = LimitedSizeVec::new();
/// assert_eq!(vec.len(), 0);
/// assert_eq!(vec.capacity(), 8);
///
/// // Add elements
/// vec.push("hello");
/// vec.push("world");
/// assert_eq!(vec.len(), 2);
///
/// // Access elements
/// assert_eq!(vec[0], "hello");
/// assert_eq!(vec.get(1), Some(&"world"));
///
/// // Convert to slice
/// let slice: &[&str] = vec.as_slice();
/// assert_eq!(slice, ["hello", "world"]);
/// ```
#[repr(C)]
pub struct LimitedSizeVec<T> {
    data: NonNull<T>,
    len: u32,
    capacity: u32,
}

/// Default initial capacity for new vectors
const DEFAULT_CAPACITY: u32 = 8;

impl<T> LimitedSizeVec<T> {
    /// Creates a new empty vector with default capacity.
    ///
    /// The vector will initially allocate space for 8 elements.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use qudit_core::LimitedSizeVec;
    ///
    /// let vec: LimitedSizeVec<i32> = LimitedSizeVec::new();
    /// assert_eq!(vec.len(), 0);
    /// assert_eq!(vec.capacity(), 8);
    /// ```
    #[inline]
    pub fn new() -> Self {
        Self::new_with_capacity(DEFAULT_CAPACITY)
    }

    /// Creates a new empty vector with the specified capacity.
    ///
    /// # Arguments
    /// * `capacity` - The initial capacity (must be > 0)
    ///
    /// # Panics
    /// Panics if capacity is 0 or if memory allocation fails.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use qudit_core::LimitedSizeVec;
    ///
    /// let vec: LimitedSizeVec<i32> = LimitedSizeVec::new_with_capacity(100);
    /// assert_eq!(vec.len(), 0);
    /// assert_eq!(vec.capacity(), 100);
    /// ```
    #[inline]
    pub fn new_with_capacity(capacity: u32) -> Self {
        if capacity == 0 {
            Self::zero_capacity_panic();
        }

        let layout =
            Layout::array::<T>(capacity as usize).unwrap_or_else(|_| Self::layout_error_panic());

        let ptr = unsafe { alloc(layout) as *mut T };
        let non_null = NonNull::new(ptr).unwrap_or_else(|| Self::allocation_failed_panic());

        Self {
            data: non_null,
            len: 0,
            capacity,
        }
    }

    #[cold]
    #[inline(never)]
    fn zero_capacity_panic() -> ! {
        panic!("Cannot create vector with zero capacity");
    }

    #[cold]
    #[inline(never)]
    fn layout_error_panic() -> ! {
        panic!("Failed to create memory layout for vector");
    }

    #[cold]
    #[inline(never)]
    fn allocation_failed_panic() -> ! {
        panic!("Memory allocation failed");
    }

    /// Appends an element to the back of the vector.
    ///
    /// If the vector's capacity is exceeded, it will be reallocated with
    /// double the previous capacity.
    ///
    /// # Arguments
    /// * `value` - The element to append
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use qudit_core::LimitedSizeVec;
    ///
    /// let mut vec = LimitedSizeVec::new();
    /// vec.push(42);
    /// vec.push(100);
    /// assert_eq!(vec.len(), 2);
    /// assert_eq!(vec[0], 42);
    /// assert_eq!(vec[1], 100);
    /// ```
    #[inline]
    pub fn push(&mut self, value: T) {
        if self.len >= self.capacity {
            self.grow();
        }
        unsafe {
            self.data.as_ptr().add(self.len as usize).write(value);
        }
        self.len += 1;
    }

    /// Grows the vector's capacity.
    ///
    /// Doubles the current capacity, with a minimum of 8 elements.
    /// This method is called automatically when needed by `push()`.
    fn grow(&mut self) {
        let new_capacity = if self.capacity == 0 {
            DEFAULT_CAPACITY
        } else {
            self.capacity.saturating_mul(2)
        };

        unsafe {
            let new_layout = Layout::array::<T>(new_capacity as usize).unwrap();
            let new_ptr = alloc(new_layout) as *mut T;

            if new_ptr.is_null() {
                Self::allocation_failed_panic();
            }

            // Zero capacity only possible with dangling pointer; special case
            if self.capacity != 0 {
                let old_ptr = self.data.as_ptr();
                std::ptr::copy_nonoverlapping(old_ptr, new_ptr, self.len as usize);
                let old_layout = Layout::array::<T>(self.capacity as usize).unwrap();
                dealloc(old_ptr as *mut u8, old_layout);
            }

            self.data = NonNull::new(new_ptr).unwrap();
        }
        self.capacity = new_capacity;
    }

    /// Grows the vector's capacity to at least the specified capacity.
    fn grow_to_capacity(&mut self, new_capacity: u32) {
        if new_capacity <= self.capacity {
            return;
        }

        unsafe {
            let new_layout = Layout::array::<T>(new_capacity as usize).unwrap();
            let new_ptr = alloc(new_layout) as *mut T;

            if new_ptr.is_null() {
                Self::allocation_failed_panic();
            }

            // Copy existing elements if any
            if self.capacity != 0 {
                let old_ptr = self.data.as_ptr();
                std::ptr::copy_nonoverlapping(old_ptr, new_ptr, self.len as usize);
                let old_layout = Layout::array::<T>(self.capacity as usize).unwrap();
                dealloc(old_ptr as *mut u8, old_layout);
            }

            self.data = NonNull::new(new_ptr).unwrap();
        }
        self.capacity = new_capacity;
    }

    /// Returns the number of elements in the vector.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use qudit_core::LimitedSizeVec;
    ///
    /// let mut vec = LimitedSizeVec::new();
    /// assert_eq!(vec.len(), 0);
    ///
    /// vec.push("hello");
    /// assert_eq!(vec.len(), 1);
    /// ```
    #[inline(always)]
    pub fn len(&self) -> usize {
        self.len as usize
    }

    /// Returns the total capacity of the vector.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use qudit_core::LimitedSizeVec;
    ///
    /// let vec: LimitedSizeVec<i32> = LimitedSizeVec::new_with_capacity(50);
    /// assert_eq!(vec.capacity(), 50);
    /// ```
    #[inline(always)]
    pub fn capacity(&self) -> usize {
        self.capacity as usize
    }

    /// Returns `true` if the vector contains no elements.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use qudit_core::LimitedSizeVec;
    ///
    /// let mut vec = LimitedSizeVec::new();
    /// assert!(vec.is_empty());
    ///
    /// vec.push(42);
    /// assert!(!vec.is_empty());
    /// ```
    #[inline(always)]
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Returns a reference to the element at the given index, or `None` if out of bounds.
    ///
    /// # Arguments
    /// * `index` - The index to access
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use qudit_core::LimitedSizeVec;
    ///
    /// let mut vec = LimitedSizeVec::new();
    /// vec.push("hello");
    ///
    /// assert_eq!(vec.get(0), Some(&"hello"));
    /// assert_eq!(vec.get(1), None);
    /// ```
    #[inline]
    pub fn get(&self, index: usize) -> Option<&T> {
        if index < self.len as usize {
            unsafe { Some(&*self.data.as_ptr().add(index)) }
        } else {
            None
        }
    }

    /// Returns a mutable reference to the element at the given index, or `None` if out of bounds.
    ///
    /// # Arguments
    /// * `index` - The index to access
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use qudit_core::LimitedSizeVec;
    ///
    /// let mut vec = LimitedSizeVec::new();
    /// vec.push(String::from("hello"));
    ///
    /// if let Some(element) = vec.get_mut(0) {
    ///     element.push_str(" world");
    /// }
    /// assert_eq!(vec[0], "hello world");
    /// ```
    #[inline]
    pub fn get_mut(&mut self, index: usize) -> Option<&mut T> {
        if index < self.len as usize {
            unsafe { Some(&mut *self.data.as_ptr().add(index)) }
        } else {
            None
        }
    }

    /// Returns a reference to an element without bounds checking.
    ///
    /// # Safety
    /// Calling this method with an out-of-bounds index is undefined behavior.
    /// The caller must ensure that `index < self.len()`.
    ///
    /// # Arguments
    /// * `index` - The index to access
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use qudit_core::LimitedSizeVec;
    ///
    /// let mut vec = LimitedSizeVec::new();
    /// vec.push("hello");
    ///
    /// unsafe {
    ///     assert_eq!(vec.get_unchecked(0), &"hello");
    /// }
    /// ```
    #[inline]
    pub unsafe fn get_unchecked(&self, index: usize) -> &T {
        debug_assert!(
            index < self.len as usize,
            "index out of bounds: {} >= {}",
            index,
            self.len
        );
        &*self.data.as_ptr().add(index)
    }

    /// Returns a mutable reference to an element without bounds checking.
    ///
    /// # Safety
    /// Calling this method with an out-of-bounds index is undefined behavior.
    /// The caller must ensure that `index < self.len()`.
    ///
    /// # Arguments
    /// * `index` - The index to access
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use qudit_core::LimitedSizeVec;
    ///
    /// let mut vec = LimitedSizeVec::new();
    /// vec.push(String::from("hello"));
    ///
    /// unsafe {
    ///     vec.get_mut_unchecked(0).push_str(" world");
    /// }
    /// assert_eq!(vec[0], "hello world");
    /// ```
    #[inline]
    pub unsafe fn get_mut_unchecked(&mut self, index: usize) -> &mut T {
        debug_assert!(
            index < self.len as usize,
            "index out of bounds: {} >= {}",
            index,
            self.len
        );
        &mut *self.data.as_ptr().add(index)
    }

    /// Removes the last element from the vector and returns it, or `None` if empty.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use qudit_core::LimitedSizeVec;
    ///
    /// let mut vec = LimitedSizeVec::new();
    /// vec.push(1);
    /// vec.push(2);
    ///
    /// assert_eq!(vec.pop(), Some(2));
    /// assert_eq!(vec.pop(), Some(1));
    /// assert_eq!(vec.pop(), None);
    /// ```
    #[inline]
    pub fn pop(&mut self) -> Option<T> {
        if self.len > 0 {
            self.len -= 1;
            unsafe { Some(std::ptr::read(self.data.as_ptr().add(self.len as usize))) }
        } else {
            None
        }
    }

    /// Removes and returns the element at position `index`, shifting all elements after it to the left.
    ///
    /// # Panics
    /// Panics if `index` is out of bounds.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use qudit_core::LimitedSizeVec;
    ///
    /// let mut vec = LimitedSizeVec::new();
    /// vec.push("a");
    /// vec.push("b");
    /// vec.push("c");
    ///
    /// assert_eq!(vec.remove(1), "b");
    /// assert_eq!(vec.as_slice(), ["a", "c"]);
    /// ```
    #[inline]
    pub fn remove(&mut self, index: usize) -> T {
        if index >= self.len as usize {
            panic!(
                "removal index (is {}) should be < len (is {})",
                index, self.len
            );
        }

        unsafe {
            let ptr = self.data.as_ptr().add(index);
            let ret = std::ptr::read(ptr);

            // Shift elements left
            let remaining = self.len as usize - index - 1;
            if remaining > 0 {
                std::ptr::copy(ptr.add(1), ptr, remaining);
            }

            self.len -= 1;
            ret
        }
    }

    /// Inserts an element at position `index`, shifting all elements after it to the right.
    ///
    /// # Panics
    /// Panics if `index > len`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use qudit_core::LimitedSizeVec;
    ///
    /// let mut vec = LimitedSizeVec::new();
    /// vec.push("a");
    /// vec.push("c");
    ///
    /// vec.insert(1, "b");
    /// assert_eq!(vec.as_slice(), ["a", "b", "c"]);
    /// ```
    #[inline]
    pub fn insert(&mut self, index: usize, element: T) {
        if index > self.len as usize {
            panic!(
                "insertion index (is {}) should be <= len (is {})",
                index, self.len
            );
        }

        if self.len >= self.capacity {
            self.grow();
        }

        unsafe {
            let ptr = self.data.as_ptr().add(index);

            // Shift elements right
            let remaining = self.len as usize - index;
            if remaining > 0 {
                std::ptr::copy(ptr, ptr.add(1), remaining);
            }

            std::ptr::write(ptr, element);
            self.len += 1;
        }
    }

    /// Shortens the vector, keeping the first `len` elements and dropping the rest.
    ///
    /// If `len` is greater than the vector's current length, this has no effect.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use qudit_core::LimitedSizeVec;
    ///
    /// let mut vec = LimitedSizeVec::new();
    /// vec.push(1);
    /// vec.push(2);
    /// vec.push(3);
    /// vec.push(4);
    ///
    /// vec.truncate(2);
    /// assert_eq!(vec.as_slice(), [1, 2]);
    /// ```
    #[inline]
    pub fn truncate(&mut self, len: usize) {
        if len < self.len as usize {
            let new_len = len as u32;
            if std::mem::needs_drop::<T>() {
                unsafe {
                    for i in new_len..self.len {
                        std::ptr::drop_in_place(self.data.as_ptr().add(i as usize));
                    }
                }
            }
            self.len = new_len;
        }
    }

    /// Resizes the vector to the given length.
    ///
    /// If `new_len` is greater than `len`, the vector is extended with clones of `value`.
    /// If `new_len` is less than `len`, the vector is truncated.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use qudit_core::LimitedSizeVec;
    ///
    /// let mut vec = LimitedSizeVec::new();
    /// vec.push(1);
    ///
    /// vec.resize(3, 0);
    /// assert_eq!(vec.as_slice(), [1, 0, 0]);
    ///
    /// vec.resize(1, 2);
    /// assert_eq!(vec.as_slice(), [1]);
    /// ```
    #[inline]
    pub fn resize(&mut self, new_len: usize, value: T)
    where
        T: Clone,
    {
        let current_len = self.len as usize;
        if new_len > current_len {
            self.reserve(new_len - current_len);
            for _ in current_len..new_len {
                self.push(value.clone());
            }
        } else {
            self.truncate(new_len);
        }
    }

    /// Extends the vector by cloning elements from a slice.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use qudit_core::LimitedSizeVec;
    ///
    /// let mut vec = LimitedSizeVec::new();
    /// vec.push(1);
    ///
    /// vec.extend_from_slice(&[2, 3, 4]);
    /// assert_eq!(vec.as_slice(), [1, 2, 3, 4]);
    /// ```
    #[inline]
    pub fn extend_from_slice(&mut self, other: &[T])
    where
        T: Clone,
    {
        self.reserve(other.len());
        for item in other {
            self.push(item.clone());
        }
    }

    /// Reserves capacity for at least `additional` more elements.
    ///
    /// Does nothing if the capacity is already sufficient.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use qudit_core::LimitedSizeVec;
    ///
    /// let mut vec: LimitedSizeVec<usize> = LimitedSizeVec::new();
    /// vec.reserve(100);
    /// assert!(vec.capacity() >= 100);
    /// ```
    #[inline]
    pub fn reserve(&mut self, additional: usize) {
        let required = self.len as usize + additional;
        if required > self.capacity as usize {
            self.grow_to_capacity(required.max(self.capacity as usize * 2) as u32);
        }
    }

    /// Appends an element without checking capacity.
    ///
    /// # Safety
    /// The caller must ensure that `self.len < self.capacity`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use qudit_core::LimitedSizeVec;
    ///
    /// let mut vec = LimitedSizeVec::new_with_capacity(10);
    /// unsafe {
    ///     vec.push_unchecked(42);
    /// }
    /// assert_eq!(vec[0], 42);
    /// ```
    #[inline]
    pub unsafe fn push_unchecked(&mut self, value: T) {
        debug_assert!(
            self.len < self.capacity,
            "capacity exceeded: {} >= {}",
            self.len,
            self.capacity
        );
        self.data.as_ptr().add(self.len as usize).write(value);
        self.len += 1;
    }

    /// Clears the vector, removing all elements but keeping allocated capacity.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use qudit_core::LimitedSizeVec;
    ///
    /// let mut vec = LimitedSizeVec::new();
    /// vec.push(1);
    /// vec.push(2);
    ///
    /// let old_capacity = vec.capacity();
    /// vec.clear();
    ///
    /// assert_eq!(vec.len(), 0);
    /// assert_eq!(vec.capacity(), old_capacity);
    /// ```
    #[inline]
    pub fn clear(&mut self) {
        if self.len > 0 {
            if std::mem::needs_drop::<T>() {
                for i in 0..self.len {
                    unsafe {
                        std::ptr::drop_in_place(self.data.as_ptr().add(i as usize));
                    }
                }
            }
            self.len = 0;
        }
    }

    /// Sorts the vector in-place using the natural ordering of elements.
    ///
    /// This method requires that `T` implements `Ord`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use qudit_core::LimitedSizeVec;
    ///
    /// let mut vec = LimitedSizeVec::new();
    /// vec.push(3);
    /// vec.push(1);
    /// vec.push(2);
    ///
    /// vec.sort();
    /// assert_eq!(vec.as_slice(), [1, 2, 3]);
    /// ```
    #[inline]
    pub fn sort(&mut self)
    where
        T: Ord,
    {
        let slice = self.as_slice_mut();
        slice.sort();
    }

    /// Takes ownership of the vector's contents, leaving the original empty.
    ///
    /// This is a zero-cost operation that transfers ownership without cloning.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use qudit_core::LimitedSizeVec;
    ///
    /// let mut vec1 = LimitedSizeVec::new();
    /// vec1.push(1);
    /// vec1.push(2);
    ///
    /// let vec2 = vec1.take();
    /// assert_eq!(vec1.len(), 0);
    /// assert_eq!(vec2.len(), 2);
    /// ```
    #[inline]
    pub fn take(&mut self) -> Self {
        let cap = self.capacity;
        let len = self.len;
        let ptr = self.data;

        // Reset self to valid empty state
        self.data = NonNull::dangling();
        self.len = 0;
        self.capacity = 0;

        Self {
            data: ptr,
            len,
            capacity: cap,
        }
    }

    /// Creates a new vector by cloning this one.
    ///
    /// This method is equivalent to `clone()` but provides a more explicit name.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use qudit_core::LimitedSizeVec;
    ///
    /// let mut vec1 = LimitedSizeVec::new();
    /// vec1.push("hello".to_string());
    ///
    /// let vec2 = vec1.to_owned();
    /// assert_eq!(vec1.len(), 1);
    /// assert_eq!(vec2.len(), 1);
    /// ```
    #[inline]
    pub fn to_owned(&self) -> Self
    where
        T: Clone,
    {
        self.clone()
    }

    /// Returns the vector's contents as a slice.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use qudit_core::LimitedSizeVec;
    ///
    /// let mut vec = LimitedSizeVec::new();
    /// vec.push(1);
    /// vec.push(2);
    ///
    /// let slice = vec.as_slice();
    /// assert_eq!(slice, [1, 2]);
    /// ```
    #[inline(always)]
    pub fn as_slice(&self) -> &[T] {
        unsafe { std::slice::from_raw_parts(self.data.as_ptr(), self.len as usize) }
    }

    /// Returns the vector's contents as a mutable slice.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use qudit_core::LimitedSizeVec;
    ///
    /// let mut vec = LimitedSizeVec::new();
    /// vec.push(1);
    /// vec.push(2);
    ///
    /// let slice = vec.as_slice_mut();
    /// slice[0] = 10;
    /// assert_eq!(vec[0], 10);
    /// ```
    #[inline(always)]
    pub fn as_slice_mut(&mut self) -> &mut [T] {
        unsafe { std::slice::from_raw_parts_mut(self.data.as_ptr(), self.len as usize) }
    }
}

impl<T: Clone> Clone for LimitedSizeVec<T> {
    /// Creates a clone of the vector.
    ///
    /// All elements are cloned individually, and the new vector will have
    /// the same capacity as the original.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use qudit_core::LimitedSizeVec;
    ///
    /// let mut vec1 = LimitedSizeVec::new();
    /// vec1.push("hello".to_string());
    ///
    /// let vec2 = vec1.clone();
    /// assert_eq!(vec1.len(), vec2.len());
    /// assert_eq!(vec1[0], vec2[0]);
    /// ```
    fn clone(&self) -> Self {
        if self.len == 0 {
            return Self::new_with_capacity(self.capacity);
        }

        let mut new_vec = Self::new_with_capacity(self.capacity);

        // Clone each element individually for safety
        for i in 0..self.len {
            unsafe {
                let item = &*self.data.as_ptr().add(i as usize);
                new_vec.push(item.clone());
            }
        }

        new_vec
    }
}

impl<T> Default for LimitedSizeVec<T> {
    /// Creates an empty vector with default capacity.
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl<T: fmt::Debug> fmt::Debug for LimitedSizeVec<T> {
    /// Formats the vector for debugging output.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_list().entries(self.as_slice().iter()).finish()
    }
}

impl<T: PartialEq> PartialEq for LimitedSizeVec<T> {
    fn eq(&self, other: &Self) -> bool {
        if self.len != other.len {
            return false;
        }
        self.as_slice().eq(other.as_slice())
    }
}

impl<T: Eq> Eq for LimitedSizeVec<T> {}

impl<T: Hash> Hash for LimitedSizeVec<T> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.len.hash(state);
        self.as_slice().hash(state);
    }
}

impl<T> std::ops::Deref for LimitedSizeVec<T> {
    type Target = [T];

    #[inline(always)]
    fn deref(&self) -> &Self::Target {
        self.as_slice()
    }
}

impl<T> std::ops::DerefMut for LimitedSizeVec<T> {
    #[inline(always)]
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.as_slice_mut()
    }
}

impl<T> std::ops::Index<usize> for LimitedSizeVec<T> {
    type Output = T;

    #[inline(always)]
    fn index(&self, index: usize) -> &Self::Output {
        &self.as_slice()[index]
    }
}

impl<T> std::ops::IndexMut<usize> for LimitedSizeVec<T> {
    #[inline(always)]
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.as_slice_mut()[index]
    }
}

impl<T> AsRef<[T]> for LimitedSizeVec<T> {
    #[inline(always)]
    fn as_ref(&self) -> &[T] {
        self.as_slice()
    }
}

impl<T> AsMut<[T]> for LimitedSizeVec<T> {
    #[inline(always)]
    fn as_mut(&mut self) -> &mut [T] {
        self.as_slice_mut()
    }
}

impl<T> From<Vec<T>> for LimitedSizeVec<T> {
    /// Converts a `Vec<T>` into a `LimitedSizeVec<T>`.
    ///
    fn from(vec: Vec<T>) -> Self {
        let len = vec.len();
        let capacity = vec.capacity();

        if capacity > u32::MAX as usize {
            panic!("Vector capacity exceeds maximum allowed for LimitedSizeVec (u32::MAX)");
        }

        // Prevent `vec` from dropping its contents
        let mut temp_vec = std::mem::ManuallyDrop::new(vec);

        // Get raw parts
        let ptr = temp_vec.as_mut_ptr();
        let len_u32 = len as u32; // Assuming len fits, as capacity is checked
        let cap_u32 = capacity as u32;

        LimitedSizeVec {
            data: NonNull::new(ptr).unwrap_or_else(|| Self::allocation_failed_panic()),
            len: len_u32,
            capacity: cap_u32,
        }
    }
}

impl<T: Clone> From<&[T]> for LimitedSizeVec<T> {
    /// Converts a slice into a `LimitedSizeVec<T>` by cloning elements.
    fn from(slice: &[T]) -> Self {
        let mut limited_vec =
            Self::new_with_capacity(slice.len().max(DEFAULT_CAPACITY as usize) as u32);
        for item in slice {
            limited_vec.push(item.clone());
        }
        limited_vec
    }
}

impl<T: Clone> From<&Vec<T>> for LimitedSizeVec<T> {
    /// Converts a `&Vec<T>` into a `LimitedSizeVec<T>` by cloning elements.
    #[inline]
    fn from(vec: &Vec<T>) -> Self {
        Self::from(vec.as_slice())
    }
}

impl<T> FromIterator<T> for LimitedSizeVec<T> {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        let iter = iter.into_iter();
        let (lower_bound, _) = iter.size_hint();
        let mut vec =
            LimitedSizeVec::new_with_capacity(lower_bound.max(DEFAULT_CAPACITY as usize) as u32);
        for item in iter {
            vec.push(item);
        }
        vec
    }
}

impl<T> From<LimitedSizeVec<T>> for Vec<T> {
    /// Converts a `LimitedSizeVec<T>` into a `Vec<T>` without copying elements.
    ///
    /// This is a zero-cost conversion that transfers ownership of the underlying buffer.
    fn from(mut limited_vec: LimitedSizeVec<T>) -> Self {
        let len = limited_vec.len();
        let capacity = limited_vec.capacity();
        let ptr = limited_vec.data.as_ptr();

        // Prevent limited_vec from dropping the memory by resetting it
        limited_vec.data = NonNull::dangling();
        limited_vec.len = 0;
        limited_vec.capacity = 0;

        unsafe { Self::from_raw_parts(ptr, len, capacity) }
    }
}

/// An iterator that moves out of a `LimitedSizeVec`.
pub struct IntoIter<T> {
    data: NonNull<T>,
    start: *const T,
    end: *const T,
    capacity: u32,
}

impl<T> Iterator for IntoIter<T> {
    type Item = T;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.start == self.end {
            None
        } else {
            unsafe {
                let current = self.start;
                self.start = self.start.add(1);
                Some(std::ptr::read(current))
            }
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = unsafe { self.end.offset_from(self.start) } as usize;
        (len, Some(len))
    }
}

impl<T> DoubleEndedIterator for IntoIter<T> {
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.start == self.end {
            None
        } else {
            unsafe {
                self.end = self.end.sub(1);
                Some(std::ptr::read(self.end))
            }
        }
    }
}

impl<T> ExactSizeIterator for IntoIter<T> {}

impl<T> Drop for IntoIter<T> {
    fn drop(&mut self) {
        // Drop any remaining elements
        for _ in &mut *self {}

        // Deallocate memory if we have capacity
        if self.capacity > 0 {
            unsafe {
                dealloc(
                    self.data.as_ptr() as *mut u8,
                    Layout::array::<T>(self.capacity as usize).unwrap(),
                );
            }
        }
    }
}

impl<T> IntoIterator for LimitedSizeVec<T> {
    type Item = T;
    type IntoIter = IntoIter<T>;

    /// Creates a consuming iterator, that is, one that moves each value out of
    /// the vector (from start to end). The vector cannot be used after calling this.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use qudit_core::LimitedSizeVec;
    ///
    /// let mut vec = LimitedSizeVec::new();
    /// vec.push(1);
    /// vec.push(2);
    /// vec.push(3);
    ///
    /// let mut iterator = vec.into_iter();
    /// assert_eq!(iterator.next(), Some(1));
    /// assert_eq!(iterator.next(), Some(2));
    /// assert_eq!(iterator.next(), Some(3));
    /// assert_eq!(iterator.next(), None);
    /// ```
    #[inline]
    fn into_iter(mut self) -> Self::IntoIter {
        let ptr = self.data.as_ptr();
        let len = self.len as usize;
        let capacity = self.capacity;

        // Prevent the vector from being dropped
        self.data = NonNull::dangling();
        self.len = 0;
        self.capacity = 0;

        IntoIter {
            data: NonNull::new(ptr).unwrap(),
            start: ptr,
            end: unsafe { ptr.add(len) },
            capacity,
        }
    }
}

impl<'a, T> IntoIterator for &'a LimitedSizeVec<T> {
    type Item = &'a T;
    type IntoIter = std::slice::Iter<'a, T>;

    /// Creates an iterator over references to the elements of the vector.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use qudit_core::LimitedSizeVec;
    ///
    /// let mut vec = LimitedSizeVec::new();
    /// vec.push(1);
    /// vec.push(2);
    ///
    /// for item in &vec {
    ///     println!("{}", item);
    /// }
    /// ```
    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.as_slice().iter()
    }
}

impl<'a, T> IntoIterator for &'a mut LimitedSizeVec<T> {
    type Item = &'a mut T;
    type IntoIter = std::slice::IterMut<'a, T>;

    /// Creates an iterator over mutable references to the elements of the vector.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use qudit_core::LimitedSizeVec;
    ///
    /// let mut vec = LimitedSizeVec::new();
    /// vec.push(1);
    /// vec.push(2);
    ///
    /// for item in &mut vec {
    ///     *item *= 2;
    /// }
    /// assert_eq!(vec.as_slice(), [2, 4]);
    /// ```
    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.as_slice_mut().iter_mut()
    }
}
// Safety: LimitedSizeVec<T> is Send if T is Send
unsafe impl<T: Send> Send for LimitedSizeVec<T> {}

// Safety: LimitedSizeVec<T> is Sync if T is Sync
unsafe impl<T: Sync> Sync for LimitedSizeVec<T> {}

impl<T> Drop for LimitedSizeVec<T> {
    /// Cleans up the vector's memory and drops all elements.
    fn drop(&mut self) {
        if self.capacity > 0 {
            // Drop all elements
            for i in 0..self.len {
                unsafe {
                    std::ptr::drop_in_place(self.data.as_ptr().add(i as usize));
                }
            }

            // Deallocate memory
            unsafe {
                dealloc(
                    self.data.as_ptr() as *mut u8,
                    Layout::array::<T>(self.capacity as usize).unwrap(),
                );
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_vector_creation() {
        let vec: LimitedSizeVec<i32> = LimitedSizeVec::new();
        assert_eq!(vec.len(), 0);
        assert_eq!(vec.capacity(), DEFAULT_CAPACITY as usize);
        assert!(vec.is_empty());
    }

    #[test]
    fn test_vector_with_custom_capacity() {
        let vec: LimitedSizeVec<i32> = LimitedSizeVec::new_with_capacity(50);
        assert_eq!(vec.len(), 0);
        assert_eq!(vec.capacity(), 50);
        assert!(vec.is_empty());
    }

    #[test]
    #[should_panic(expected = "Cannot create vector with zero capacity")]
    fn test_zero_capacity_panics() {
        LimitedSizeVec::<i32>::new_with_capacity(0);
    }

    #[test]
    fn test_push_and_access_elements() {
        let mut vec = LimitedSizeVec::new();
        vec.push(1);
        vec.push(2);
        vec.push(3);

        assert_eq!(vec.len(), 3);
        assert!(!vec.is_empty());
        assert_eq!(vec[0], 1);
        assert_eq!(vec[1], 2);
        assert_eq!(vec[2], 3);
    }

    #[test]
    fn test_get_method_bounds_checking() {
        let mut vec = LimitedSizeVec::new();
        vec.push("hello");
        vec.push("world");

        assert_eq!(vec.get(0), Some(&"hello"));
        assert_eq!(vec.get(1), Some(&"world"));
        assert_eq!(vec.get(2), None);
        assert_eq!(vec.get(100), None);
    }

    #[test]
    fn test_get_mut_method() {
        let mut vec = LimitedSizeVec::new();
        vec.push(String::from("hello"));

        if let Some(element) = vec.get_mut(0) {
            element.push_str(" world");
        }

        assert_eq!(vec[0], "hello world");
        assert_eq!(vec.get_mut(1), None);
    }

    #[test]
    fn test_capacity_growth_on_push() {
        let mut vec = LimitedSizeVec::new_with_capacity(2);
        assert_eq!(vec.capacity(), 2);

        vec.push(1);
        vec.push(2);
        assert_eq!(vec.capacity(), 2);

        // This should trigger growth
        vec.push(3);
        assert_eq!(vec.capacity(), 4);
        assert_eq!(vec.len(), 3);
    }

    #[test]
    fn test_as_slice_operations() {
        let mut vec = LimitedSizeVec::new();
        vec.push(3);
        vec.push(1);
        vec.push(4);
        vec.push(1);
        vec.push(5);

        let slice = vec.as_slice();
        assert_eq!(slice, [3, 1, 4, 1, 5]);

        let mut_slice = vec.as_slice_mut();
        mut_slice[0] = 10;
        assert_eq!(vec[0], 10);
    }

    #[test]
    fn test_sort_method() {
        let mut vec = LimitedSizeVec::new();
        vec.push(3);
        vec.push(1);
        vec.push(4);
        vec.push(1);
        vec.push(5);

        vec.sort();
        assert_eq!(vec.as_slice(), [1, 1, 3, 4, 5]);
    }

    #[test]
    fn test_clear_method() {
        let mut vec = LimitedSizeVec::new();
        vec.push(1);
        vec.push(2);
        vec.push(3);

        let old_capacity = vec.capacity();
        vec.clear();

        assert_eq!(vec.len(), 0);
        assert!(vec.is_empty());
        assert_eq!(vec.capacity(), old_capacity); // Capacity preserved
    }

    #[test]
    fn test_take_method() {
        let mut vec1 = LimitedSizeVec::new();
        vec1.push("hello");
        vec1.push("world");

        let vec2 = vec1.take();

        assert_eq!(vec1.len(), 0);
        assert!(vec1.is_empty());
        assert_eq!(vec2.len(), 2);
        assert_eq!(vec2[0], "hello");
        assert_eq!(vec2[1], "world");
    }

    #[test]
    fn test_clone_implementation() {
        let mut vec1 = LimitedSizeVec::new();
        vec1.push(String::from("hello"));
        vec1.push(String::from("world"));

        let vec2 = vec1.clone();

        assert_eq!(vec1.len(), vec2.len());
        assert_eq!(vec1.capacity(), vec2.capacity());
        assert_eq!(vec1[0], vec2[0]);
        assert_eq!(vec1[1], vec2[1]);

        // Verify they're independent
        drop(vec1);
        assert_eq!(vec2.len(), 2);
    }

    #[test]
    fn test_to_owned_method() {
        let mut vec1 = LimitedSizeVec::new();
        vec1.push("test".to_string());

        let vec2 = vec1.to_owned();
        assert_eq!(vec1.len(), vec2.len());
        assert_eq!(vec1[0], vec2[0]);
    }

    #[test]
    fn test_deref_operations() {
        let mut vec = LimitedSizeVec::new();
        vec.push(1);
        vec.push(2);
        vec.push(3);

        // Test Deref to slice
        let slice: &[i32] = &vec;
        assert_eq!(slice.len(), 3);
        assert_eq!(slice[0], 1);

        // Test DerefMut to slice
        let slice_mut: &mut [i32] = &mut vec;
        slice_mut[0] = 10;
        assert_eq!(vec[0], 10);
    }

    #[test]
    fn test_from_vec_conversion() {
        let std_vec = vec![1, 2, 3, 4, 5];
        let limited_vec: LimitedSizeVec<i32> = LimitedSizeVec::from(std_vec);

        assert_eq!(limited_vec.len(), 5);
        assert_eq!(limited_vec.as_slice(), [1, 2, 3, 4, 5]);
        assert!(limited_vec.capacity() >= 5);
    }

    #[test]
    fn test_from_slice_conversion() {
        let slice = [1, 2, 3];
        let limited_vec: LimitedSizeVec<i32> = LimitedSizeVec::from(&slice[..]);

        assert_eq!(limited_vec.len(), 3);
        assert_eq!(limited_vec.as_slice(), [1, 2, 3]);
    }

    #[test]
    fn test_to_vec_conversion() {
        let mut limited_vec = LimitedSizeVec::new();
        limited_vec.push(1);
        limited_vec.push(2);
        limited_vec.push(3);

        let std_vec: Vec<i32> = Vec::from(limited_vec);
        assert_eq!(std_vec, [1, 2, 3]);
    }

    #[test]
    fn test_as_ref_and_as_mut() {
        let mut vec = LimitedSizeVec::new();
        vec.push("hello");
        vec.push("world");

        let slice_ref: &[&str] = vec.as_ref();
        assert_eq!(slice_ref, ["hello", "world"]);

        let slice_mut: &mut [&str] = vec.as_mut();
        slice_mut[0] = "goodbye";
        assert_eq!(vec[0], "goodbye");
    }

    #[test]
    fn test_debug_formatting() {
        let mut vec = LimitedSizeVec::new();
        vec.push(1);
        vec.push(2);

        let debug_str = format!("{:?}", vec);
        assert_eq!(debug_str, "[1, 2]");
    }

    #[test]
    fn test_empty_vector_operations() {
        let vec: LimitedSizeVec<i32> = LimitedSizeVec::new();
        assert_eq!(vec.len(), 0);
        assert!(vec.is_empty());
        assert_eq!(vec.get(0), None);
    }

    #[test]
    fn test_large_capacity_vector() {
        let vec: LimitedSizeVec<u8> = LimitedSizeVec::new_with_capacity(1000);
        assert_eq!(vec.capacity(), 1000);
        assert_eq!(vec.len(), 0);
    }

    #[test]
    fn test_drop_behavior() {
        use std::rc::Rc;

        let item = Rc::new("test");
        assert_eq!(Rc::strong_count(&item), 1);

        {
            let mut vec = LimitedSizeVec::new();
            vec.push(item.clone());
            assert_eq!(Rc::strong_count(&item), 2);
        } // vec is dropped here

        assert_eq!(Rc::strong_count(&item), 1);
    }

    #[test]
    fn test_pop_method() {
        let mut vec = LimitedSizeVec::new();
        assert_eq!(vec.pop(), None);

        vec.push(1);
        vec.push(2);
        vec.push(3);

        assert_eq!(vec.pop(), Some(3));
        assert_eq!(vec.len(), 2);
        assert_eq!(vec.pop(), Some(2));
        assert_eq!(vec.pop(), Some(1));
        assert_eq!(vec.pop(), None);
        assert_eq!(vec.len(), 0);
    }

    #[test]
    fn test_remove_method() {
        let mut vec = LimitedSizeVec::new();
        vec.push("a");
        vec.push("b");
        vec.push("c");
        vec.push("d");

        // Remove from middle
        assert_eq!(vec.remove(1), "b");
        assert_eq!(vec.as_slice(), ["a", "c", "d"]);

        // Remove from end
        assert_eq!(vec.remove(2), "d");
        assert_eq!(vec.as_slice(), ["a", "c"]);

        // Remove from start
        assert_eq!(vec.remove(0), "a");
        assert_eq!(vec.as_slice(), ["c"]);
    }

    #[test]
    #[should_panic(expected = "removal index (is 5) should be < len (is 2)")]
    fn test_remove_out_of_bounds() {
        let mut vec = LimitedSizeVec::new();
        vec.push(1);
        vec.push(2);
        vec.remove(5);
    }

    #[test]
    fn test_insert_method() {
        let mut vec = LimitedSizeVec::new();
        vec.push("a");
        vec.push("c");

        // Insert in middle
        vec.insert(1, "b");
        assert_eq!(vec.as_slice(), ["a", "b", "c"]);

        // Insert at beginning
        vec.insert(0, "start");
        assert_eq!(vec.as_slice(), ["start", "a", "b", "c"]);

        // Insert at end
        vec.insert(vec.len(), "end");
        assert_eq!(vec.as_slice(), ["start", "a", "b", "c", "end"]);
    }

    #[test]
    #[should_panic(expected = "insertion index (is 5) should be <= len (is 2)")]
    fn test_insert_out_of_bounds() {
        let mut vec = LimitedSizeVec::new();
        vec.push(1);
        vec.push(2);
        vec.insert(5, 3);
    }

    #[test]
    fn test_truncate_method() {
        let mut vec = LimitedSizeVec::new();
        vec.push(1);
        vec.push(2);
        vec.push(3);
        vec.push(4);
        vec.push(5);

        let old_capacity = vec.capacity();
        vec.truncate(3);

        assert_eq!(vec.len(), 3);
        assert_eq!(vec.as_slice(), [1, 2, 3]);
        assert_eq!(vec.capacity(), old_capacity); // Capacity preserved

        // Truncate to 0
        vec.truncate(0);
        assert_eq!(vec.len(), 0);
        assert!(vec.is_empty());

        // Truncate larger than length should do nothing
        vec.push(10);
        vec.truncate(10);
        assert_eq!(vec.len(), 1);
        assert_eq!(vec[0], 10);
    }

    #[test]
    fn test_resize_method() {
        let mut vec = LimitedSizeVec::new();
        vec.push(1);

        // Resize up
        vec.resize(4, 42);
        assert_eq!(vec.as_slice(), [1, 42, 42, 42]);

        // Resize down
        vec.resize(2, 0);
        assert_eq!(vec.as_slice(), [1, 42]);

        // Resize to same size
        vec.resize(2, 100);
        assert_eq!(vec.as_slice(), [1, 42]);
    }

    #[test]
    fn test_extend_from_slice() {
        let mut vec = LimitedSizeVec::new();
        vec.push(1);

        vec.extend_from_slice(&[2, 3, 4]);
        assert_eq!(vec.as_slice(), [1, 2, 3, 4]);

        vec.extend_from_slice(&[]);
        assert_eq!(vec.as_slice(), [1, 2, 3, 4]);

        vec.extend_from_slice(&[5, 6]);
        assert_eq!(vec.as_slice(), [1, 2, 3, 4, 5, 6]);
    }

    #[test]
    fn test_reserve_method() {
        let mut vec = LimitedSizeVec::new();
        assert_eq!(vec.capacity(), 8);

        vec.reserve(50);
        assert!(vec.capacity() >= 50);

        let old_capacity = vec.capacity();
        vec.reserve(10); // Less than current, should do nothing
        assert_eq!(vec.capacity(), old_capacity);

        // Test with existing elements
        vec.push(1);
        vec.push(2);
        vec.reserve(100);
        assert!(vec.capacity() >= 100);
        assert_eq!(vec.as_slice(), [1, 2]);
    }

    #[test]
    fn test_push_unchecked() {
        let mut vec = LimitedSizeVec::new_with_capacity(10);

        unsafe {
            vec.push_unchecked(1);
            vec.push_unchecked(2);
            vec.push_unchecked(3);
        }

        assert_eq!(vec.len(), 3);
        assert_eq!(vec.as_slice(), [1, 2, 3]);
    }

    #[test]
    fn test_get_unchecked_methods() {
        let mut vec = LimitedSizeVec::new();
        vec.push(String::from("hello"));
        vec.push(String::from("world"));

        unsafe {
            assert_eq!(vec.get_unchecked(0), &"hello");
            assert_eq!(vec.get_unchecked(1), &"world");

            vec.get_mut_unchecked(0).push_str(" there");
            assert_eq!(vec.get_unchecked(0), &"hello there");
        }
    }

    #[test]
    fn test_from_iterator() {
        let iter = 0..5;
        let vec: LimitedSizeVec<i32> = iter.collect();
        assert_eq!(vec.len(), 5);
        assert_eq!(vec.as_slice(), [0, 1, 2, 3, 4]);

        let empty_iter: std::vec::IntoIter<i32> = Vec::new().into_iter();
        let empty_vec: LimitedSizeVec<i32> = empty_iter.collect();
        assert_eq!(empty_vec.len(), 0);
    }

    #[test]
    fn test_from_iterator_with_strings() {
        let iter = vec!["a".to_string(), "b".to_string(), "c".to_string()].into_iter();
        let vec: LimitedSizeVec<String> = iter.collect();
        assert_eq!(vec.len(), 3);
        assert_eq!(vec.as_slice(), ["a", "b", "c"]);
    }

    #[test]
    fn test_into_iterator() {
        let mut vec = LimitedSizeVec::new();
        vec.push(1);
        vec.push(2);
        vec.push(3);

        let collected: Vec<i32> = vec.into_iter().collect();
        assert_eq!(collected, [1, 2, 3]);
    }

    #[test]
    fn test_iterator_double_ended() {
        let mut vec = LimitedSizeVec::new();
        vec.push(1);
        vec.push(2);
        vec.push(3);

        let mut iter = vec.into_iter();
        assert_eq!(iter.next(), Some(1));
        assert_eq!(iter.next_back(), Some(3));
        assert_eq!(iter.next(), Some(2));
        assert_eq!(iter.next(), None);
        assert_eq!(iter.next_back(), None);
    }

    #[test]
    fn test_borrowed_iterator() {
        let mut vec = LimitedSizeVec::new();
        vec.push(1);
        vec.push(2);
        vec.push(3);

        let sum: i32 = (&vec).into_iter().sum();
        assert_eq!(sum, 6);

        // Vector should still exist
        assert_eq!(vec.len(), 3);
    }

    #[test]
    fn test_mutable_borrowed_iterator() {
        let mut vec = LimitedSizeVec::new();
        vec.push(1);
        vec.push(2);
        vec.push(3);

        for item in &mut vec {
            *item *= 2;
        }

        assert_eq!(vec.as_slice(), [2, 4, 6]);
    }

    #[test]
    fn test_iterator_size_hint() {
        let mut vec = LimitedSizeVec::new();
        vec.push(1);
        vec.push(2);
        vec.push(3);

        let iter = vec.into_iter();
        assert_eq!(iter.size_hint(), (3, Some(3)));
    }

    #[test]
    fn test_contains_method() {
        let mut vec = LimitedSizeVec::new();
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
    fn test_clear_optimized_for_copy_types() {
        // Test with Copy type (no drops needed)
        let mut vec: LimitedSizeVec<i32> = LimitedSizeVec::new();
        vec.push(1);
        vec.push(2);
        vec.push(3);

        let old_capacity = vec.capacity();
        vec.clear();

        assert_eq!(vec.len(), 0);
        assert_eq!(vec.capacity(), old_capacity);

        // Test with non-Copy type (drops needed)
        let mut vec = LimitedSizeVec::new();
        vec.push(String::from("hello"));
        vec.push(String::from("world"));

        let old_capacity = vec.capacity();
        vec.clear();

        assert_eq!(vec.len(), 0);
        assert_eq!(vec.capacity(), old_capacity);
    }

    #[test]
    fn test_memory_efficiency_compared_to_std_vec() {
        // Test that our struct is smaller than Vec on 64-bit systems
        use std::mem::size_of;

        // LimitedSizeVec should be smaller due to u32 length/capacity
        let limited_size = size_of::<LimitedSizeVec<i32>>();
        let std_vec_size = size_of::<Vec<i32>>();

        #[cfg(target_pointer_width = "64")]
        {
            // On 64-bit systems, our vector should be smaller
            assert!(limited_size <= std_vec_size);
        }

        println!("LimitedSizeVec<i32> size: {} bytes", limited_size);
        println!("Vec<i32> size: {} bytes", std_vec_size);
    }
}
