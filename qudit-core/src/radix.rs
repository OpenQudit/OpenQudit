use num_traits::AsPrimitive;

use crate::CompactStorage;

#[derive(Default, PartialEq, Eq, Hash, PartialOrd, Ord, Debug, Copy, Clone)]
#[repr(transparent)]
pub struct Radix(u8);

impl<T: AsPrimitive<u8> + From<u8> + PartialOrd> From<T> for Radix {
    fn from(value: T) -> Self {
        debug_assert!(value > 2u8.into() && value <= 255u8.into());
        Radix(value.as_())
    }
}

impl CompactStorage for Radix {
    type InlineType = Radix;
    const CONVERSION_INFALLIBLE: bool = true;
    
    #[inline(always)]
    fn to_inline(value: Self) -> Result<Self::InlineType, Self> { Ok(value) }

    #[inline(always)]
    fn from_inline(value: Self::InlineType) -> Self { value }

    #[inline(always)]
    fn to_inline_unchecked(value: Self) -> Self::InlineType { value }
}

impl From<Radix> for usize {
    #[inline(always)]
    fn from(value: Radix) -> Self {
        value.0 as usize
    }
}

impl From<Radix> for u64 {
    #[inline(always)]
    fn from(value: Radix) -> Self {
        value.0 as u64
    }
}

impl std::iter::Product<Radix> for usize {
    fn product<I: Iterator<Item = Radix>>(iter: I) -> usize {
        iter.into_iter().map(|r| r.0 as usize).product()
    }
}

impl<'a> std::iter::Product<&'a Radix> for usize {
    fn product<I: Iterator<Item = &'a Radix>>(iter: I) -> usize {
        iter.into_iter().map(|r| r.0 as usize).product()
    }
}

impl PartialEq<usize> for Radix {
    fn eq(&self, other: &usize) -> bool {
        (self.0 as usize) == *other
    }
}

impl std::fmt::Display for Radix {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl<T: Into<Radix>> std::ops::Add<T> for Radix {
    type Output = Self;

    fn add(self, rhs: T) -> Self::Output {
        Radix(self.0 + rhs.into().0)
    }
}

impl<T: Into<Radix>> std::ops::Sub<T> for Radix {
    type Output = Self;

    fn sub(self, rhs: T) -> Self::Output {
        Radix(self.0 - rhs.into().0)
    }
}

#[cfg(feature = "python")]
mod python {
    use super::*;
    use pyo3::prelude::*;

    impl<'py> IntoPyObject<'py> for Radix {
        type Target = <u8 as IntoPyObject<'py>>::Target;
        type Output = <u8 as IntoPyObject<'py>>::Output;
        type Error = <u8 as IntoPyObject<'py>>::Error;

        fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
            self.0.into_pyobject(py)
        }
    }

    impl<'py> FromPyObject<'py> for Radix {
        fn extract_bound(obj: &Bound<'py, PyAny>) -> PyResult<Self> {
            let value: u8 = obj.extract()?;
            Ok(Radix(value))
        }
    }
}
