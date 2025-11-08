//! Radix (number base) representation with compile-time validation.

use num_traits::AsPrimitive;

use crate::CompactStorage;

/// A validated radix (number base) value between 2 and 255 inclusive.
///
/// This type ensures that radix values are always valid for numeric base conversions.
///
/// # Examples
///
/// ```
/// # use qudit_core::Radix;
/// // Create from u8 or usize
/// let radix = Radix::from(10u8);
/// let radix = Radix::from(16usize);
///
/// // Convert to other integer types
/// let base = usize::from(radix);
/// let base: u64 = radix.into();
/// ```
#[derive(Default, PartialEq, Eq, Hash, PartialOrd, Ord, Debug, Copy, Clone)]
#[repr(transparent)]
pub struct Radix(u8);

impl<T: AsPrimitive<u8> + From<u8> + PartialOrd> From<T> for Radix {
    #[track_caller]
    fn from(value: T) -> Self {
        assert!(value >= 2u8.into(), "Radices cannot be less than 2.");
        assert!(value <= 255u8.into(), "Radix overflow.");
        Radix(value.as_())
    }
}

impl CompactStorage for Radix {
    type InlineType = Radix;
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

impl From<Radix> for u32 {
    #[inline(always)]
    fn from(value: Radix) -> Self {
        value.0 as u32
    }
}

impl From<Radix> for u16 {
    #[inline(always)]
    fn from(value: Radix) -> Self {
        value.0 as u16
    }
}

impl From<Radix> for u8 {
    #[inline(always)]
    fn from(value: Radix) -> Self {
        value.0
    }
}

impl From<Radix> for u128 {
    #[inline(always)]
    fn from(value: Radix) -> Self {
        value.0 as u128
    }
}

impl From<Radix> for i16 {
    #[inline(always)]
    fn from(value: Radix) -> Self {
        value.0 as i16
    }
}

impl From<Radix> for i32 {
    #[inline(always)]
    fn from(value: Radix) -> Self {
        value.0 as i32
    }
}

impl From<Radix> for i64 {
    #[inline(always)]
    fn from(value: Radix) -> Self {
        value.0 as i64
    }
}

impl From<Radix> for i128 {
    #[inline(always)]
    fn from(value: Radix) -> Self {
        value.0 as i128
    }
}

impl From<Radix> for f32 {
    #[inline(always)]
    fn from(value: Radix) -> Self {
        value.0 as f32
    }
}

impl From<Radix> for f64 {
    #[inline(always)]
    fn from(value: Radix) -> Self {
        value.0 as f64
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

    impl<'a, 'py> FromPyObject<'a, 'py> for Radix {
        type Error = PyErr;

        fn extract(obj: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
            let value: u8 = obj.extract()?;
            if value < 2 {
                Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "Radix value {} is invalid. Radices must be >= 2.",
                    value
                )))
            } else {
                Ok(Radix(value))
            }
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use pyo3::Python;

        #[test]
        fn test_into_pyobject() {
            Python::initialize();
            Python::attach(|py| {
                let radix = Radix::from(10u8);
                let py_obj = radix.into_pyobject(py).unwrap();
                let value: u8 = py_obj.extract().unwrap();
                assert_eq!(value, 10);
            });
        }

        #[test]
        fn test_from_pyobject() {
            Python::initialize();
            Python::attach(|py| {
                let py_int = 16u8.into_pyobject(py).unwrap();
                let radix: Radix = py_int.extract().unwrap();
                assert_eq!(usize::from(radix), 16);
            });
        }

        #[test]
        fn test_from_pyobject_invalid() {
            Python::initialize();
            Python::attach(|py| {
                let py_int = 1u8.into_pyobject(py).unwrap();
                let result: PyResult<Radix> = py_int.extract();
                assert!(result.is_err());
                let err = result.unwrap_err();
                assert!(err.to_string().contains("Radix value 1 is invalid"));
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn valid_radix_creation() {
        let r2 = Radix::from(2u8);
        let r10 = Radix::from(10u8);
        let r255 = Radix::from(255u8);

        assert_eq!(usize::from(r2), 2);
        assert_eq!(usize::from(r10), 10);
        assert_eq!(usize::from(r255), 255);
    }

    #[test]
    #[should_panic(expected = "Radices cannot be less than 2")]
    fn invalid_radix_too_small() {
        let _r = Radix::from(1u8);
    }

    #[test]
    fn conversions() {
        let r16 = Radix::from(16u8);
        assert_eq!(usize::from(r16), 16);
        assert_eq!(u64::from(r16), 16u64);
    }

    #[test]
    fn arithmetic() {
        let r10 = Radix::from(10u8);
        let r5 = Radix::from(5u8);

        assert_eq!(r10 + r5, Radix::from(15u8));
        assert_eq!(r10 - r5, Radix::from(5u8));
    }

    #[test]
    fn display() {
        let r16 = Radix::from(16u8);
        assert_eq!(format!("{}", r16), "16");
    }

    #[test]
    fn equality() {
        let r10 = Radix::from(10u8);
        assert_eq!(r10, 10usize);
    }

    #[test]
    fn product() {
        let radices = [Radix::from(2u8), Radix::from(3u8), Radix::from(4u8)];
        let product: usize = radices.iter().product();
        assert_eq!(product, 24);
    }
}
