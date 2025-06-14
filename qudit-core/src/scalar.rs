use crate::c32;
use crate::c64;
use crate::memory::Memorable;
use std::ops::Neg;

use faer_traits::ComplexField;
use faer_traits::RealField;
use num::bigint::BigInt;
use num::complex::ComplexFloat;
use num::rational::Ratio;
use num::ToPrimitive;
use num::Zero;
use num_traits::Float;
use num_traits::FloatConst;
use num_traits::Inv;
use num_traits::NumAssign;
use num_traits::NumAssignOps;
use num_traits::NumOps;
use rand::Rng;

use crate::bitwidth::BitWidthConvertible;

/// A real number type that can be used as a scalar in the Qudit-Core library.
pub trait RealScalar:
    RealField
    + Copy
    + Sized
    + Float
    + FloatConst
    + NumAssign
    + Memorable
    + std::fmt::Debug
    + std::fmt::Display
    + BitWidthConvertible<Width32 = f32, Width64 = f64>
    + 'static
{
    /// Generate a random scalar from the standard normal distribution.
    fn standard_random() -> Self;

    /// Convert a rational number to the real number type.
    fn from_rational(r: &Ratio<BigInt>) -> Self;
}

/// A complex number type that can be used as a scalar in the Qudit-Core library.
pub trait ComplexScalar:
    ComplexField<Real = Self::R>
    + Copy
    + Sized
    + NumAssign
    + Memorable
    + Neg<Output = Self>
    + NumAssignOps<Self::R>
    + ComplexFloat
    // + NumAssignOps<Self::Conj>
    + NumOps<Self::R, Self>
    // + NumOps<Self::Conj, Self>
    + std::fmt::Debug
    + std::fmt::Display
    + BitWidthConvertible<Width32 = c32, Width64 = c64>
    + 'static
{
    /// Real number type that composes this complex number type.
    type R: RealScalar;

    /// Threshold for comparing two complex numbers.
    const THRESHOLD: Self::R;

    /// Threshold for gradient correctness evaluation.
    const GRAD_EPSILON: Self::R;

    /// Imaginary unit.
    const I: Self;

    /// Create a complex number equal to `e^(i * theta)`.
    fn cis(theta: Self::R) -> Self;

    /// Compute the sine of the complex number.
    fn sin(self) -> Self;

    /// Compute the cosine of the complex number.
    fn cos(self) -> Self;

    /// Compute the absolute value of the complex number.
    fn abs(self) -> Self::R;

    /// Raise the complex number to a non-negative integer power.
    fn powu(self, n: u32) -> Self;

    /// Raise the complex number to an integer power.
    fn powi(self, n: i32) -> Self;

    // /// Construct the complex conjugate of the complex number.
    // fn conj(self) -> Self;

    /// Compute the inverse of the complex number.
    fn inv(self) -> Self;

    // TODO: re-evaluate names: from_componenets, from_real?
    /// Construct a complex number from two real numbers.
    fn complex(re: impl RealScalar, im: impl RealScalar) -> Self;

    /// Convert a real number to the real part of the complex number.
    fn real(re: impl RealScalar) -> Self::R;

    /// Generate a random scalar from the standard normal distribution. 
    fn standard_random() -> Self;

    /// Construct a complex number from a float number.
    fn from_f32(re: f32) -> Self {
        Self::complex(re, Self::R::zero())
    }

    /// Construct a complex number from a integer number.
    fn from_i32(re: i32) -> Self {
        Self::complex(re as f32, Self::R::zero()) 
    }
}

impl ComplexScalar for c32 {
    type R = f32;

    const GRAD_EPSILON: Self::R = 1e-2;
    const THRESHOLD: Self::R = 1e-5;
    const I: Self = c32 { re: 0.0, im: 1.0 };

    #[inline(always)]
    fn cis(theta: Self::R) -> Self {
        c32::new(theta.cos(), theta.sin())
    }

    #[inline(always)]
    fn sin(self) -> Self {
        self.sin()
    }

    #[inline(always)]
    fn cos(self) -> Self {
        self.cos()
    }

    #[inline(always)]
    fn abs(self) -> Self::R {
        <c32 as ComplexFloat>::abs(self)
    }

    #[inline(always)]
    fn powu(self, n: u32) -> Self {
        c32::powu(&self, n)
    }

    #[inline(always)]
    fn powi(self, n: i32) -> Self {
        c32::powi(&self, n)
    }

    // #[inline(always)]
    // fn conj(mut self) -> Self {
    //     self.im *= -1.0;
    //     self
    // }

    #[inline(always)]
    fn inv(self) -> Self {
        <c32 as Inv>::inv(self)
    }

    #[inline(always)]
    fn complex(re: impl RealScalar, im: impl RealScalar) -> Self {
        c32::new(re.to32(), im.to32())
    }

    #[inline(always)]
    fn real(re: impl RealScalar) -> Self::R {
        re.to32()
    }

    #[inline(always)]
    fn standard_random() -> Self {
        c32::new(rand::rng().random(), rand::rng().random())
    }
}

impl ComplexScalar for c64 {
    type R = f64;

    const GRAD_EPSILON: Self::R = 1e-5;
    const THRESHOLD: Self::R = 1e-10;
    const I: Self = c64 { re: 0.0, im: 1.0 };

    #[inline(always)]
    fn cis(theta: Self::R) -> Self {
        c64::new(theta.cos(), theta.sin())
    }

    #[inline(always)]
    fn sin(self) -> Self {
        self.sin()
    }

    #[inline(always)]
    fn cos(self) -> Self {
        self.cos()
    }

    #[inline(always)]
    fn abs(self) -> Self::R {
        <c64 as ComplexFloat>::abs(self)
    }

    #[inline(always)]
    fn powu(self, n: u32) -> Self {
        c64::powu(&self, n)
    }

    #[inline(always)]
    fn powi(self, n: i32) -> Self {
        c64::powi(&self, n)
    }

    // #[inline(always)]
    // fn conj(mut self) -> Self {
    //     self.im *= -1.0;
    //     self
    // }

    #[inline(always)]
    fn inv(self) -> Self {
        <c64 as Inv>::inv(self)
    }

    #[inline(always)]
    fn complex(re: impl RealScalar, im: impl RealScalar) -> Self {
        c64::new(re.to64(), im.to64())
    }

    #[inline(always)]
    fn real(re: impl RealScalar) -> Self::R {
        re.to64()
    }

    #[inline(always)]
    fn standard_random() -> Self {
        c64::new(rand::rng().random(), rand::rng().random())
    }
}

impl RealScalar for f32 {
    #[inline(always)]
    fn standard_random() -> Self {
        rand::rng().random()
    }

    #[inline(always)]
    fn from_rational(r: &Ratio<BigInt>) -> Self {
        r.to_f32().expect("Failed to convert Rational to f32")
    }
}

impl RealScalar for f64 {
    #[inline(always)]
    fn standard_random() -> Self {
        rand::rng().random()
    }

    #[inline(always)]
    fn from_rational(r: &Ratio<BigInt>) -> Self {
        r.to_f64().expect("Failed to convert Rational to f64")
    }
}

