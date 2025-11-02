use crate::c32;
use crate::c64;
use crate::memory::Memorable;

use faer_traits::ComplexField;
use faer_traits::RealField;
use num::bigint::BigInt;
use num::complex::ComplexFloat;
use num::rational::Ratio;
use num::FromPrimitive;
use num::Signed;
use num::ToPrimitive;
use num_traits::AsPrimitive;
use num_traits::ConstOne;
use num_traits::ConstZero;
use num_traits::Float;
use num_traits::FloatConst;
use num_traits::NumAssign;
use num_traits::NumAssignOps;
use num_traits::NumAssignRef;
use num_traits::NumOps;
use rand_distr::{Distribution, StandardNormal, Uniform};

use crate::bitwidth::BitWidthConvertible;

/// A generic real number within the OpenQudit library.
pub trait RealScalar:
    Clone
    + Copy
    + Sync
    + Send
    + Sized
    + std::any::Any
    + std::fmt::Debug
    + std::fmt::Display
    + std::iter::Sum
    + std::iter::Product
    + Float
    + FloatConst
    + FromPrimitive
    + ToPrimitive
    + AsPrimitive<f32>
    + AsPrimitive<f64>
    + NumAssign
    + NumAssignRef
    + Default
    + Signed
    + 'static
    + RealField
    + ConstOne
    + ConstZero
    + Memorable
    + BitWidthConvertible<Width32 = f32, Width64 = f64>
{
    /// The complex number type associated with this real number.
    type C: ComplexScalar<R = Self>;

    /// Convert this scalar to a rational number representation
    fn into_ratio(self) -> Option<Ratio<BigInt>>;
    
    /// Create a scalar from a rational number representation  
    fn from_ratio(ratio: Ratio<BigInt>) -> Option<Self>;
    
    /// Generate a random value from the standard normal distribution (mean=0, std=1)
    fn standard_random() -> Self;
    
    /// Generate a random value from a uniform distribution in the given range [min, max)
    fn uniform_random(min: Self, max: Self) -> Self;
    
    /// Check if two values are close using default tolerances
    fn is_close(self, other: impl RealScalar) -> bool;
    
    /// Check if two values are close with custom tolerances
    /// Uses the formula: abs(a - b) <= (atol + rtol * abs(b))
    fn is_close_with_tolerance(self, other: impl RealScalar, rtol: Self, atol: Self) -> bool;
}

impl RealScalar for f32 {
    type C = c32;

    fn into_ratio(self) -> Option<Ratio<BigInt>> {
        Ratio::from_float(self as f64)
    }
    
    fn from_ratio(ratio: Ratio<BigInt>) -> Option<Self> {
        ratio.to_f32()
    }
    
    fn standard_random() -> Self {
        let mut rng = rand::rng();
        StandardNormal.sample(&mut rng)
    }
    
    fn uniform_random(min: Self, max: Self) -> Self {
        let mut rng = rand::rng();
        Uniform::new(min, max).unwrap().sample(&mut rng)
    }
    
    fn is_close(self, other: impl RealScalar) -> bool {
        self.is_close_with_tolerance(other, 1e-5, 1e-8)
    }
    
    fn is_close_with_tolerance(self, other: impl RealScalar, rtol: Self, atol: Self) -> bool {
        (self - other.to32()).abs() <= (atol + rtol * other.to32().abs())
    }
}

impl RealScalar for f64 {
    type C = c64;

    fn into_ratio(self) -> Option<Ratio<BigInt>> {
        Ratio::from_float(self)
    }
    
    fn from_ratio(ratio: Ratio<BigInt>) -> Option<Self> {
        ratio.to_f64()
    }
    
    fn standard_random() -> Self {
        let mut rng = rand::rng();
        StandardNormal.sample(&mut rng)
    }
    
    fn uniform_random(min: Self, max: Self) -> Self {
        let mut rng = rand::rng();
        Uniform::new(min, max).unwrap().sample(&mut rng)
    }
    
    fn is_close(self, other: impl RealScalar) -> bool {
        self.is_close_with_tolerance(other, 1e-9, 1e-12)
    }
    
    fn is_close_with_tolerance(self, other: impl RealScalar, rtol: Self, atol: Self) -> bool {
        (self - other.to64()).abs() <= (atol + rtol * other.to64().abs())
    }
}

/// A generic complex number within the OpenQudit library.
pub trait ComplexScalar:
    Clone
    + Copy
    + Sync
    + Send
    + Sized
    + std::any::Any
    + std::fmt::Debug
    + std::fmt::Display
    + std::iter::Sum
    + std::iter::Product
    + NumAssign
    + NumAssignRef
    + NumOps<Self::R>
    + NumAssignOps<Self::R>
    + Default
    + 'static
    + ConstOne
    + ConstZero
    + ComplexField<Real = Self::R>
    + ComplexFloat<Real = Self::R>
    + Memorable
    + BitWidthConvertible<Width32 = c32, Width64 = c64>
{
    /// The real number type associated with this complex number.
    type R: RealScalar<C = Self>;
    
    /// Create a complex number from real and imaginary parts
    fn new(real: impl RealScalar, imag: impl RealScalar) -> Self;
    
    /// Create a complex number from just the real part (imaginary = 0)
    fn from_real(real: impl RealScalar) -> Self;

    /// The real component of the complex number.
    fn real(&self) -> Self::R;

    /// The imaginary component of the complex number.
    fn imag(&self) -> Self::R;
    
    /// Generate a random complex number with both real and imaginary parts from standard normal
    fn standard_random() -> Self;
    
    /// Generate a random complex number with both parts uniform in given ranges
    fn uniform_random(real_min: Self::R, real_max: Self::R, imag_min: Self::R, imag_max: Self::R) -> Self;

    /// Calculate the squared norm (|z|²) of the complex number
    #[inline(always)]
    fn norm_squared(self) -> Self::R {
        self.re() * self.re() + self.im() * self.im()
    }
}

impl ComplexScalar for c32 {
    type R = f32;
    
    fn new(real: impl RealScalar, imag: impl RealScalar) -> Self {
        c32::new(real.to32(), imag.to32())
    }
    
    fn from_real(real: impl RealScalar) -> Self {
        c32::new(real.to32(), 0.0)
    }

    #[inline(always)]
    fn real(&self) -> Self::R {
        self.re
    }

    #[inline(always)]
    fn imag(&self) -> Self::R {
        self.im
    }
    
    fn standard_random() -> Self {
        c32::new(f32::standard_random(), f32::standard_random())
    }
    
    fn uniform_random(real_min: Self::R, real_max: Self::R, imag_min: Self::R, imag_max: Self::R) -> Self {
        c32::new(
            f32::uniform_random(real_min, real_max),
            f32::uniform_random(imag_min, imag_max)
        )
    }
}

impl ComplexScalar for c64 {
    type R = f64;
    
    fn new(real: impl RealScalar, imag: impl RealScalar) -> Self {
        c64::new(real.to64(), imag.to64())
    }
    
    fn from_real(real: impl RealScalar) -> Self {
        c64::new(real.to64(), 0.0)
    }

    #[inline(always)]
    fn real(&self) -> Self::R {
        self.re
    }

    #[inline(always)]
    fn imag(&self) -> Self::R {
        self.im
    }
    
    fn standard_random() -> Self {
        c64::new(f64::standard_random(), f64::standard_random())
    }
    
    fn uniform_random(real_min: Self::R, real_max: Self::R, imag_min: Self::R, imag_max: Self::R) -> Self {
        c64::new(
            f64::uniform_random(real_min, real_max),
            f64::uniform_random(imag_min, imag_max)
        )
    }
}
