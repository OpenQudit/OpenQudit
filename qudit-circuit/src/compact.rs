use std::ptr::NonNull;
// TODO: This entire module is just a prototype for an api, a lot of inefficiencies and potential bugs

pub struct LimitedSizeVec {
    data: NonNull<usize>,
    len: u32,
    capacity: u32,
}

impl LimitedSizeVec {
    pub fn new() -> Self {
        Self::new_with_capacity(8)
    }

    #[inline]
    pub fn new_with_capacity(capacity: u32) -> Self {
        let ptr = unsafe {
            std::alloc::alloc(std::alloc::Layout::array::<usize>(capacity as usize).unwrap()) as *mut usize
        };
        Self {
            data: NonNull::new(ptr).unwrap(),
            len: 0,
            capacity,
        }
    }

    #[inline]
    pub fn push(&mut self, value: usize) {
        if self.len >= self.capacity {
            self.grow();
        }
        unsafe {
            self.data.as_ptr().add(self.len as usize).write(value);
        }
        self.len += 1;
    }

    fn grow(&mut self) {
        let new_capacity = self.capacity.saturating_mul(2);
        unsafe {
            let new_ptr = std::alloc::alloc(std::alloc::Layout::array::<usize>(new_capacity as usize).unwrap()) as *mut usize;
            let old_ptr = self.data.as_ptr();
            std::ptr::copy_nonoverlapping(old_ptr, new_ptr, self.len as usize);
            self.data = NonNull::new(new_ptr).unwrap();
            std::alloc::dealloc(old_ptr as *mut u8, std::alloc::Layout::array::<usize>(self.capacity as usize).unwrap());
        }
        self.capacity = new_capacity;
    }

    pub fn len(&self) -> usize {
        self.len as usize
    }

    pub fn capacity(&self) -> u32 {
        self.capacity
    }

    pub fn get(&self, index: usize) -> Option<&usize> {
        if index < self.len as usize {
            unsafe {
                Some(&*self.data.as_ptr().add(index as usize))
            }
        } else {
            None
        }
    }

    pub fn get_mut(&mut self, index: u32) -> Option<&mut usize> {
        if index < self.len {
            unsafe {
                Some(&mut *self.data.as_ptr().add(index as usize))
            }
        } else {
            None
        }
    }

    pub fn sort(&mut self) {
        // TODO: if self is large do something else
        // insertion sort
        for i in 1..self.len {
            let key = unsafe { self.data.as_ptr().add(i as usize).read() };
            let mut j = i;
            while j > 0 && key < unsafe { self.data.as_ptr().add(j as usize - 1).read() } {
                unsafe {
                    self.data.as_ptr().add(j as usize).write(self.data.as_ptr().add(j as usize - 1).read());
                }
                j -= 1;
            }
            unsafe {
                self.data.as_ptr().add(j as usize).write(key);
            }
        }
    }

    pub fn to_owned(&mut self) -> Self {
        let cap = self.capacity;
        let len = self.len;
        let ptr = self.data;
        self.data = NonNull::dangling();
        self.len = 0;
        self.capacity = 0;
        Self {
            data: ptr,
            len,
            capacity: cap,
        }
    }
}

impl Drop for LimitedSizeVec {
    fn drop(&mut self) {
        if self.capacity > 0 {
            unsafe {
                std::alloc::dealloc(self.data.as_ptr() as *mut u8, std::alloc::Layout::array::<usize>(self.capacity as usize).unwrap());
            }
        }
    }
}

pub enum CompactIntegerVector {
    Array(u8, [u8; 7]),
    Heap(LimitedSizeVec),
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
                Self::Heap(vec.to_owned())
            }
        }
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
