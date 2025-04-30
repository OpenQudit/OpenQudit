/// A primitive data type that can be converted to an unsigned, byte-sized radix.
pub trait ToRadix: Copy + Sized + core::fmt::Display {
    /// Convert the value to an unsigned, byte-sized radix.
    fn to_radix(self) -> u8;

    /// Check if the value is less than two.
    fn is_less_than_two(self) -> bool;
}

impl ToRadix for u8 {
    #[inline(always)]
    fn to_radix(self) -> u8 {
        self
    }

    #[inline(always)]
    fn is_less_than_two(self) -> bool {
        self < 2u8
    }
}

impl ToRadix for u16 {
    #[inline(always)]
    fn to_radix(self) -> u8 {
        debug_assert!(self <= 255, "Radix too large: {}", self);
        self as u8
    }

    #[inline(always)]
    fn is_less_than_two(self) -> bool {
        self < 2u16
    }
}

impl ToRadix for u32 {
    #[inline(always)]
    fn to_radix(self) -> u8 {
        debug_assert!(self <= 255, "Radix too large: {}", self);
        self as u8
    }

    #[inline(always)]
    fn is_less_than_two(self) -> bool {
        self < 2u32
    }
}

impl ToRadix for u64 {
    #[inline(always)]
    fn to_radix(self) -> u8 {
        debug_assert!(self <= 255, "Radix too large: {}", self);
        self as u8
    }

    #[inline(always)]
    fn is_less_than_two(self) -> bool {
        self < 2u64
    }
}

impl ToRadix for u128 {
    #[inline(always)]
    fn to_radix(self) -> u8 {
        debug_assert!(self <= 255, "Radix too large: {}", self);
        self as u8
    }

    #[inline(always)]
    fn is_less_than_two(self) -> bool {
        self < 2u128
    }
}

impl ToRadix for usize {
    #[inline(always)]
    fn to_radix(self) -> u8 {
        debug_assert!(self <= 255, "Radix too large: {}", self);
        self as u8
    }

    #[inline(always)]
    fn is_less_than_two(self) -> bool {
        self < 2usize
    }
}

impl ToRadix for i8 {
    #[inline(always)]
    fn to_radix(self) -> u8 {
        debug_assert!(self >= 0, "Negative radix: {}", self);
        self as u8
    }

    #[inline(always)]
    fn is_less_than_two(self) -> bool {
        self < 2i8
    }
}

impl ToRadix for i16 {
    #[inline(always)]
    fn to_radix(self) -> u8 {
        debug_assert!(self >= 0, "Negative radix: {}", self);
        debug_assert!(self <= 255, "Radix too large: {}", self);
        self as u8
    }

    #[inline(always)]
    fn is_less_than_two(self) -> bool {
        self < 2i16
    }
}

impl ToRadix for i32 {
    #[inline(always)]
    fn to_radix(self) -> u8 {
        debug_assert!(self >= 0, "Negative radix: {}", self);
        debug_assert!(self <= 255, "Radix too large: {}", self);
        self as u8
    }

    #[inline(always)]
    fn is_less_than_two(self) -> bool {
        self < 2i32
    }
}

impl ToRadix for i64 {
    #[inline(always)]
    fn to_radix(self) -> u8 {
        debug_assert!(self >= 0, "Negative radix: {}", self);
        debug_assert!(self <= 255, "Radix too large: {}", self);
        self as u8
    }

    #[inline(always)]
    fn is_less_than_two(self) -> bool {
        self < 2i64
    }
}

impl ToRadix for i128 {
    #[inline(always)]
    fn to_radix(self) -> u8 {
        debug_assert!(self >= 0, "Negative radix: {}", self);
        debug_assert!(self <= 255, "Radix too large: {}", self);
        self as u8
    }

    #[inline(always)]
    fn is_less_than_two(self) -> bool {
        self < 2i128
    }
}

impl ToRadix for isize {
    #[inline(always)]
    fn to_radix(self) -> u8 {
        debug_assert!(self >= 0, "Negative radix: {}", self);
        debug_assert!(self <= 255, "Radix too large: {}", self);
        self as u8
    }

    #[inline(always)]
    fn is_less_than_two(self) -> bool {
        self < 2isize
    }
}

impl<T: ToRadix> ToRadix for &T {
    #[inline(always)]
    fn to_radix(self) -> u8 {
        (*self).to_radix()
    }

    #[inline(always)]
    fn is_less_than_two(self) -> bool {
        (*self).is_less_than_two()
    }
}

impl<T: ToRadix> ToRadix for core::num::Wrapping<T> {
    #[inline(always)]
    fn to_radix(self) -> u8 {
        self.0.to_radix()
    }

    #[inline(always)]
    fn is_less_than_two(self) -> bool {
        self.0.is_less_than_two()
    }
}
