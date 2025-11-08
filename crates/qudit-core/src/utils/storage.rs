/// A trait for types that can be stored compactly by attempting to fit them into smaller storage types.
///
/// This trait is designed for scenarios where values are often, but not always, small enough to fit in a more compact
/// representation (e.g., storing i32 values that typically fit in i8 to save memory).
pub trait CompactStorage: Copy + 'static {
    /// The compact storage type used for inline representation
    type InlineType: Copy + Default + 'static;
    /// Whether conversion to inline type is guaranteed to never fail
    const CONVERSION_INFALLIBLE: bool;

    /// Attempts to convert the value to its compact inline representation.
    /// Returns `Err(value)` if the value cannot fit in the inline type.
    fn to_inline(value: Self) -> Result<Self::InlineType, Self>;
    /// Converts from the compact inline representation back to the original type.
    /// This conversion is always infallible as we're expanding to a larger type.
    fn from_inline(value: Self::InlineType) -> Self;

    /// Converts to inline representation without bounds checking.
    ///
    /// # Safety
    /// The caller must guarantee that the value fits within the bounds of `InlineType`.
    /// Violating this contract may result in data truncation or undefined behavior.
    fn to_inline_unchecked(value: Self) -> Self::InlineType;
}

// Trait implementations for signed integer types (use i8 storage)
impl CompactStorage for i8 {
    type InlineType = i8;
    const CONVERSION_INFALLIBLE: bool = true;

    #[inline(always)]
    fn to_inline(value: Self) -> Result<Self::InlineType, Self> {
        Ok(value)
    }

    #[inline(always)]
    fn from_inline(value: Self::InlineType) -> Self {
        value
    }

    #[inline(always)]
    fn to_inline_unchecked(value: Self) -> Self::InlineType {
        value
    }
}

impl CompactStorage for i16 {
    type InlineType = i8;
    const CONVERSION_INFALLIBLE: bool = false;

    #[inline]
    fn to_inline(value: Self) -> Result<Self::InlineType, Self> {
        if value >= i8::MIN as i16 && value <= i8::MAX as i16 {
            Ok(value as i8)
        } else {
            Err(value)
        }
    }

    #[inline(always)]
    fn from_inline(value: Self::InlineType) -> Self {
        value as i16
    }

    #[inline(always)]
    fn to_inline_unchecked(value: Self) -> Self::InlineType {
        value as i8
    }
}

impl CompactStorage for i32 {
    type InlineType = i8;
    const CONVERSION_INFALLIBLE: bool = false;

    #[inline]
    fn to_inline(value: Self) -> Result<Self::InlineType, Self> {
        if value >= i8::MIN as i32 && value <= i8::MAX as i32 {
            Ok(value as i8)
        } else {
            Err(value)
        }
    }
    #[inline(always)]
    fn from_inline(value: Self::InlineType) -> Self {
        value as i32
    }

    #[inline(always)]
    fn to_inline_unchecked(value: Self) -> Self::InlineType {
        value as i8
    }
}

impl CompactStorage for i64 {
    type InlineType = i8;
    const CONVERSION_INFALLIBLE: bool = false;

    #[inline]
    fn to_inline(value: Self) -> Result<Self::InlineType, Self> {
        if value >= i8::MIN as i64 && value <= i8::MAX as i64 {
            Ok(value as i8)
        } else {
            Err(value)
        }
    }

    #[inline(always)]
    fn from_inline(value: Self::InlineType) -> Self {
        value as i64
    }

    #[inline(always)]
    fn to_inline_unchecked(value: Self) -> Self::InlineType {
        value as i8
    }
}

impl CompactStorage for i128 {
    type InlineType = i8;
    const CONVERSION_INFALLIBLE: bool = false;

    #[inline]
    fn to_inline(value: Self) -> Result<Self::InlineType, Self> {
        if value >= i8::MIN as i128 && value <= i8::MAX as i128 {
            Ok(value as i8)
        } else {
            Err(value)
        }
    }

    #[inline(always)]
    fn from_inline(value: Self::InlineType) -> Self {
        value as i128
    }

    #[inline(always)]
    fn to_inline_unchecked(value: Self) -> Self::InlineType {
        value as i8
    }
}

impl CompactStorage for isize {
    type InlineType = i8;
    const CONVERSION_INFALLIBLE: bool = false;

    #[inline]
    fn to_inline(value: Self) -> Result<Self::InlineType, Self> {
        if value >= i8::MIN as isize && value <= i8::MAX as isize {
            Ok(value as i8)
        } else {
            Err(value)
        }
    }

    #[inline(always)]
    fn from_inline(value: Self::InlineType) -> Self {
        value as isize
    }

    #[inline(always)]
    fn to_inline_unchecked(value: Self) -> Self::InlineType {
        value as i8
    }
}

// Trait implementations for unsigned integer types (use u8 storage)
impl CompactStorage for u8 {
    type InlineType = u8;
    const CONVERSION_INFALLIBLE: bool = true;

    #[inline(always)]
    fn to_inline(value: Self) -> Result<Self::InlineType, Self> {
        Ok(value)
    }

    #[inline(always)]
    fn from_inline(value: Self::InlineType) -> Self {
        value
    }

    #[inline(always)]
    fn to_inline_unchecked(value: Self) -> Self::InlineType {
        value
    }
}

impl CompactStorage for u16 {
    type InlineType = u8;
    const CONVERSION_INFALLIBLE: bool = false;

    #[inline]
    fn to_inline(value: Self) -> Result<Self::InlineType, Self> {
        if value <= u8::MAX as u16 {
            Ok(value as u8)
        } else {
            Err(value)
        }
    }

    #[inline(always)]
    fn from_inline(value: Self::InlineType) -> Self {
        value as u16
    }

    #[inline(always)]
    fn to_inline_unchecked(value: Self) -> Self::InlineType {
        value as u8
    }
}

impl CompactStorage for u32 {
    type InlineType = u8;
    const CONVERSION_INFALLIBLE: bool = false;

    #[inline]
    fn to_inline(value: Self) -> Result<Self::InlineType, Self> {
        if value <= u8::MAX as u32 {
            Ok(value as u8)
        } else {
            Err(value)
        }
    }

    #[inline(always)]
    fn from_inline(value: Self::InlineType) -> Self {
        value as u32
    }

    #[inline(always)]
    fn to_inline_unchecked(value: Self) -> Self::InlineType {
        value as u8
    }
}

impl CompactStorage for u64 {
    type InlineType = u8;
    const CONVERSION_INFALLIBLE: bool = false;

    #[inline]
    fn to_inline(value: Self) -> Result<Self::InlineType, Self> {
        if value <= u8::MAX as u64 {
            Ok(value as u8)
        } else {
            Err(value)
        }
    }

    #[inline(always)]
    fn from_inline(value: Self::InlineType) -> Self {
        value as u64
    }

    #[inline(always)]
    fn to_inline_unchecked(value: Self) -> Self::InlineType {
        value as u8
    }
}

impl CompactStorage for u128 {
    type InlineType = u8;
    const CONVERSION_INFALLIBLE: bool = false;

    #[inline]
    fn to_inline(value: Self) -> Result<Self::InlineType, Self> {
        if value <= u8::MAX as u128 {
            Ok(value as u8)
        } else {
            Err(value)
        }
    }

    #[inline(always)]
    fn from_inline(value: Self::InlineType) -> Self {
        value as u128
    }

    #[inline(always)]
    fn to_inline_unchecked(value: Self) -> Self::InlineType {
        value as u8
    }
}

impl CompactStorage for usize {
    type InlineType = u8;
    const CONVERSION_INFALLIBLE: bool = false;

    #[inline]
    fn to_inline(value: Self) -> Result<Self::InlineType, Self> {
        if value <= u8::MAX as usize {
            Ok(value as u8)
        } else {
            Err(value)
        }
    }

    #[inline(always)]
    fn from_inline(value: Self::InlineType) -> Self {
        value as usize
    }

    #[inline(always)]
    fn to_inline_unchecked(value: Self) -> Self::InlineType {
        value as u8
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Test infallible types (i8, u8)
    #[test]
    fn test_infallible_conversions() {
        #![allow(clippy::assertions_on_constants)]
        assert!(i8::CONVERSION_INFALLIBLE);
        assert!(u8::CONVERSION_INFALLIBLE);
        assert_eq!(i8::to_inline(42), Ok(42));
        assert_eq!(u8::to_inline(42), Ok(42));
    }

    // Test fallible signed types
    #[test]
    fn test_signed_conversions() {
        // Test within range
        assert_eq!(i16::to_inline(42), Ok(42));
        assert_eq!(i32::to_inline(-42), Ok(-42));
        assert_eq!(i64::to_inline(127), Ok(127));
        assert_eq!(i128::to_inline(-128), Ok(-128));

        // Test out of range
        assert!(i16::to_inline(200).is_err());
        assert!(i32::to_inline(1000).is_err());
        assert!(i64::to_inline(i64::MAX).is_err());
        assert!(i128::to_inline(i128::MAX).is_err());
    }

    // Test fallible unsigned types
    #[test]
    fn test_unsigned_conversions() {
        // Test within range
        assert_eq!(u16::to_inline(42), Ok(42));
        assert_eq!(u32::to_inline(255), Ok(255));
        assert_eq!(u64::to_inline(100), Ok(100));
        assert_eq!(u128::to_inline(200), Ok(200));

        // Test out of range
        assert!(u16::to_inline(300).is_err());
        assert!(u32::to_inline(1000).is_err());
        assert!(u64::to_inline(u64::MAX).is_err());
        assert!(u128::to_inline(u128::MAX).is_err());
    }

    // Test round-trip conversions
    #[test]
    fn test_round_trip() {
        assert_eq!(i128::from_inline(42), 42i128);
        assert_eq!(u128::from_inline(255), 255u128);
    }
}
