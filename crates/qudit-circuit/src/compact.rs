use std::ptr::NonNull;

pub struct LimitedSizeVec<T> {
    data: NonNull<T>,
    len: u32,
    capacity: u32,
}

impl<T> LimitedSizeVec<T> {
    pub fn new() -> Self {
        Self::new_with_capacity(8)
    }

    #[inline]
    pub fn new_with_capacity(capacity: u32) -> Self {
        if capacity == 0 {
            panic!("Cannot reserve zero capacity vector.");
        }

        let ptr = unsafe {
            std::alloc::alloc(std::alloc::Layout::array::<T>(capacity as usize).unwrap()) as *mut T
        };

        Self {
            data: NonNull::new(ptr).unwrap(),
            len: 0,
            capacity,
        }
    }

    #[inline]
    pub fn push(&mut self, value: T) {
        if self.len >= self.capacity {
            self.grow();
        }
        unsafe { self.data.as_ptr().add(self.len as usize).write(value); }
        self.len += 1;
    }

    fn grow(&mut self) {
        let new_capacity = if self.capacity == 0 { 8 } else { self.capacity.saturating_mul(2) };
        unsafe {
            let new_ptr = std::alloc::alloc(std::alloc::Layout::array::<T>(new_capacity as usize).unwrap()) as *mut T;

            // Zero capacity only possible with dangling pointer; special case, don't copy/deallocate
            if self.capacity != 0 {
                let old_ptr = self.data.as_ptr();
                std::ptr::copy_nonoverlapping(old_ptr, new_ptr, self.len as usize);
                std::alloc::dealloc(old_ptr as *mut u8, std::alloc::Layout::array::<T>(self.capacity as usize).unwrap());
            }

            self.data = NonNull::new(new_ptr).unwrap();
        }
        self.capacity = new_capacity;
    }

    pub fn len(&self) -> usize {
        self.len as usize
    }

    pub fn capacity(&self) -> usize {
        self.capacity as usize
    }

    pub fn get(&self, index: usize) -> Option<&T> {
        if index < self.len as usize {
            unsafe { Some(&*self.data.as_ptr().add(index as usize)) }
        } else {
            None
        }
    }

    pub fn get_mut(&mut self, index: usize) -> Option<&mut T> {
        if index < self.len as usize {
            unsafe { Some(&mut *self.data.as_ptr().add(index)) }
        } else {
            None
        }
    }

    pub fn sort(&mut self)
    where
        T: Ord
    {
        let slice = self.as_slice_mut();
        slice.sort();
    }

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

    pub fn to_owned(&self) -> Self 
    where 
        T: Clone 
    {
        self.clone()
    }

    pub fn as_slice(&self) -> &[T] {
        unsafe {
            std::slice::from_raw_parts(self.data.as_ptr(), self.len as usize)
        }
    }

    pub fn as_slice_mut(&mut self) -> &mut [T] {
        unsafe {
            std::slice::from_raw_parts_mut(self.data.as_ptr(), self.len as usize)
        }
    }
}

impl<T: Clone> Clone for LimitedSizeVec<T> {
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
    fn default() -> Self {
        Self::new()
    }
}

impl<T: std::fmt::Debug> std::fmt::Debug for LimitedSizeVec<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_list()
            .entries(self.as_slice().iter())
            .finish()
    }
}

impl<T> std::ops::Deref for LimitedSizeVec<T> {
    type Target = [T];

    fn deref(&self) -> &Self::Target {
        self.as_slice()
    }
}

impl<T> std::ops::DerefMut for LimitedSizeVec<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.as_slice_mut()
    }
}

impl<T> std::ops::Index<usize> for LimitedSizeVec<T> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        &self.as_slice()[index]
    }
}

impl<T> std::ops::IndexMut<usize> for LimitedSizeVec<T> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.as_slice_mut()[index]
    }
}

impl<T> AsRef<[T]> for LimitedSizeVec<T> {
    fn as_ref(&self) -> &[T] {
        self.as_slice()
    }
}

impl<T> AsMut<[T]> for LimitedSizeVec<T> {
    fn as_mut(&mut self) -> &mut [T] {
        self.as_slice_mut()
    }
}

impl<T> From<Vec<T>> for LimitedSizeVec<T> {
    fn from(vec: Vec<T>) -> Self {
        let mut limited_vec = Self::new_with_capacity(vec.len().max(8) as u32);
        for item in vec {
            limited_vec.push(item);
        }
        limited_vec
    }
}

impl<T: Clone> From<&[T]> for LimitedSizeVec<T> {
    fn from(slice: &[T]) -> Self {
        let mut limited_vec = Self::new_with_capacity(slice.len().max(8) as u32);
        for item in slice {
            limited_vec.push(item.clone());
        }
        limited_vec
    }
}

impl<T: Clone> From<&Vec<T>> for LimitedSizeVec<T> {
    fn from(vec: &Vec<T>) -> Self {
        Self::from(vec.as_slice())
    }
}

impl<T> From<Box<[T]>> for LimitedSizeVec<T> {
    fn from(boxed_slice: Box<[T]>) -> Self {
        let vec = boxed_slice.into_vec();
        Self::from(vec)
    }
}

impl<T> From<LimitedSizeVec<T>> for Vec<T> {
    fn from(mut limited_vec: LimitedSizeVec<T>) -> Self {
        let len = limited_vec.len();
        let capacity = limited_vec.capacity();
        let ptr = limited_vec.data.as_ptr();
        
        // Prevent limited_vec from dropping the memory by resetting it
        limited_vec.data = std::ptr::NonNull::dangling();
        limited_vec.len = 0;
        limited_vec.capacity = 0;
        
        unsafe { 
            Self::from_raw_parts(ptr, len, capacity)
        }
    }
}

impl<T> Drop for LimitedSizeVec<T> {
    fn drop(&mut self) {
        if self.capacity > 0 {
            for i in 0..self.len {
                unsafe {
                    std::ptr::drop_in_place(self.data.as_ptr().add(i as usize));
                }
            }
            
            unsafe {
                std::alloc::dealloc(
                    self.data.as_ptr() as *mut u8, 
                    std::alloc::Layout::array::<T>(self.capacity as usize).unwrap()
                );
            }
        }
    }
}

pub enum CompactIntegerVector {
    Array(u8, [u8; 7]),
    Heap(LimitedSizeVec<usize>),
}

impl CompactIntegerVector {
    pub fn new() -> Self {
        Self::Array(0, [0; 7])
    }

    pub fn from_range(range: std::ops::Range<usize>) -> Self {
        let len = range.len();
        if len == 0 {
            return Self::new();
        }
        let max = range.end - 1;
        if max < u8::MAX as usize && len <= 7 {
            let mut array = [0; 7];
            for i in 0..len {
                array[i] = (range.start + i) as u8;
            }
            Self::Array(len as u8, array)
        } else {
            let mut vec = LimitedSizeVec::new();
            for i in range {
                vec.push(i);
            }
            Self::Heap(vec)
        }
    }

    pub fn push(&mut self, value: usize) {
        match self {
            Self::Array(len, array) => {
                if *len < 7 {
                    match <usize as TryInto<u8>>::try_into(value) {
                        Ok(value) => {
                            array[*len as usize] = value;
                            *len += 1;
                        }
                        Err(_) => {
                            let mut vec = LimitedSizeVec::new();
                            for i in 0..7 {
                                vec.push(array[i] as usize);
                            }
                            vec.push(value);
                            *self = Self::Heap(vec);
                        }
                    }
                } else {
                    let mut vec = LimitedSizeVec::new();
                    for i in 0..7 {
                        vec.push(array[i] as usize);
                    }
                    vec.push(value);
                    *self = Self::Heap(vec);
                }
            }
            Self::Heap(vec) => {
                vec.push(value);
            }
        }
    }

    pub fn iter(&self) -> CompactIntegerVectorIter {
        CompactIntegerVectorIter::new(self)
    }

    pub fn len(&self) -> usize {
        match self {
            Self::Array(len, _) => *len as usize,
            Self::Heap(vec) => vec.len() as usize,
        }
    }

    pub fn get(&self, index: usize) -> Option<usize> {
        match self {
            Self::Array(len, array) => {
                if index < *len as usize {
                    Some(array[index as usize] as usize)
                } else {
                    None
                }
            }
            Self::Heap(vec) => vec.get(index).copied(),
        }
    }

    pub fn contains(&self, value: usize) -> bool {
        for v in self.iter() {
            if v == value {
                return true;
            }
        }
        false
    }

    pub fn sort(&mut self) {
        match self {
            Self::Array(len, array) => {
                // insertion sort
                for i in 1..*len as usize {
                    let key = array[i];
                    let mut j = i;
                    while j > 0 && key < array[j - 1] {
                        array[j] = array[j - 1];
                        j -= 1;
                    }
                    array[j] = key;
                }
            }
            Self::Heap(vec) => {
                vec.sort()
            }
        }
    }

    // TODO: check if I need to mem forget or something with heap pointer?
    pub fn to_owned(&mut self) -> Self {
        match self {
            Self::Array(len, array) => {
                let out = Self::Array(*len, *array);
                *len = 0;
                out
            }
            Self::Heap(vec) => {
                Self::Heap(vec.take())
            }
        }
    }

    pub fn to_cloned(&self) -> Self {
        self.clone()
    }

    pub fn to_vec(&self) -> Vec<usize> {
        let mut vec = Vec::new();
        for value in self.iter() {
            vec.push(value);
        }
        vec
    }
}

impl<'a> IntoIterator for &'a CompactIntegerVector {
    type Item = usize;
    type IntoIter = CompactIntegerVectorIter<'a>;

    fn into_iter(self) -> Self::IntoIter {
        CompactIntegerVectorIter::new(self)
    }
}

// implement iterator for CompactIntegerVector without allocating a vector
pub struct CompactIntegerVectorIter<'a> {
    compact_integer_vector: &'a CompactIntegerVector,
    array_index: usize,
    heap_index: usize,
}

impl<'a> CompactIntegerVectorIter<'a> {
    pub fn new(compact_integer_vector: &'a CompactIntegerVector) -> Self {
        Self {
            compact_integer_vector,
            array_index: 0,
            heap_index: 0,
        }
    }
}

impl<'a> Iterator for CompactIntegerVectorIter<'a> {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        match self.compact_integer_vector {
            CompactIntegerVector::Array(len, array) => {
                if self.array_index < *len as usize {
                    let value = array[self.array_index];
                    self.array_index += 1;
                    Some(value as usize)
                } else {
                    None
                }
            }
            CompactIntegerVector::Heap(vec) => {
                if self.heap_index < vec.len() {
                    let value = vec.get(self.heap_index).unwrap();
                    self.heap_index += 1;
                    Some(*value)
                } else {
                    None
                }
            }
        }
    }
}

impl std::fmt::Debug for CompactIntegerVector {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("[")?;
        let mut iter = self.iter();
        if let Some(value) = iter.next() {
            write!(f, "{}", value)?;
        }
        for value in iter {
            write!(f, ", {}", value)?;
        }
        f.write_str("]")
    }
}

impl Clone for CompactIntegerVector {
    fn clone(&self) -> Self {
        let mut new = CompactIntegerVector::new();
        for value in self.iter() {
            new.push(value);
        }
        new
    }
}

impl PartialEq for CompactIntegerVector {
    fn eq(&self, other: &Self) -> bool {
        if self.len() != other.len() {
            return false;
        }
        for (a, b) in self.iter().zip(other.iter()) {
            if a != b {
                return false;
            }
        }
        true
    }
}

impl<T: AsRef<[usize]>> PartialEq<T> for CompactIntegerVector {
    fn eq(&self, other: &T) -> bool {
        if self.len() != other.as_ref().len() {
            return false;
        }
        for (a, b) in self.iter().zip(other.as_ref().iter()) {
            if a != *b {
                return false;
            }
        }
        true
    }
}

impl Eq for CompactIntegerVector {}

impl std::hash::Hash for CompactIntegerVector {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        for value in self.iter() {
            value.hash(state);
        }
    }
}

impl<T: AsRef<[usize]>> From<T> for CompactIntegerVector {
    fn from(vec: T) -> Self {
        let mut compact_integer_vector = CompactIntegerVector::new();
        for value in vec.as_ref() {
            compact_integer_vector.push(*value);
        }
        compact_integer_vector
    }
}

impl From<CompactIntegerVector> for Vec<usize> {
    fn from(compact_integer_vector: CompactIntegerVector) -> Self {
        let mut vec = Vec::new();
        for value in compact_integer_vector.iter() {
            vec.push(value);
        }
        vec
    }
}
