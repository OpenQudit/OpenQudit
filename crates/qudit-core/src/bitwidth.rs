use crate::c32;
use crate::c64;

/// A trait for converting between different bit-width representations.
pub trait BitWidthConvertible {
    /// The 32-bit width representation of the type.
    type Width32;

    /// The 64-bit width representation of the type.
    type Width64;

    /// Converts the type into its 32-bit width representation.
    fn to32(self) -> Self::Width32;

    /// Converts the type into its 64-bit width representation.
    fn to64(self) -> Self::Width64;

    /// Constructs the type from a 32-bit width representation.
    fn from32(width32: Self::Width32) -> Self;

    /// Constructs the type from a 64-bit width representation.
    fn from64(width64: Self::Width64) -> Self;
}

impl BitWidthConvertible for f32 {
    type Width32 = f32;
    type Width64 = f64;

    #[inline(always)]
    fn to32(self) -> Self::Width32 {
        self
    }

    #[inline(always)]
    fn to64(self) -> Self::Width64 {
        self as f64
    }

    #[inline(always)]
    fn from32(width32: Self::Width32) -> Self {
        width32
    }

    #[inline(always)]
    fn from64(width64: Self::Width64) -> Self {
        width64 as f32
    }
}

impl BitWidthConvertible for f64 {
    type Width32 = f32;
    type Width64 = f64;

    #[inline(always)]
    fn to32(self) -> Self::Width32 {
        self as f32
    }

    #[inline(always)]
    fn to64(self) -> Self::Width64 {
        self
    }

    #[inline(always)]
    fn from32(width32: Self::Width32) -> Self {
        width32 as f64
    }

    #[inline(always)]
    fn from64(width64: Self::Width64) -> Self {
        width64
    }
}

impl BitWidthConvertible for c32 {
    type Width32 = c32;
    type Width64 = c64;

    #[inline(always)]
    fn to32(self) -> Self::Width32 {
        self
    }

    #[inline(always)]
    fn to64(self) -> Self::Width64 {
        c64::new(self.re as f64, self.im as f64)
    }

    #[inline(always)]
    fn from32(width32: Self::Width32) -> Self {
        width32
    }

    #[inline(always)]
    fn from64(width64: Self::Width64) -> Self {
        c32::new(width64.re as f32, width64.im as f32)
    }
}

impl BitWidthConvertible for c64 {
    type Width32 = c32;
    type Width64 = c64;

    #[inline(always)]
    fn to32(self) -> Self::Width32 {
        c32::new(self.re as f32, self.im as f32)
    }

    #[inline(always)]
    fn to64(self) -> Self::Width64 {
        self
    }

    #[inline(always)]
    fn from32(width32: Self::Width32) -> Self {
        c64::new(width32.re as f64, width32.im as f64)
    }

    #[inline(always)]
    fn from64(width64: Self::Width64) -> Self {
        width64
    }
}
