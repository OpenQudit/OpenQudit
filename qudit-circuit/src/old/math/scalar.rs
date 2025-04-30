use crate::math::c32;
use crate::math::c64;
use std::ops::Neg;

use faer_entity::ComplexField;
use faer_entity::RealField;
use num_traits::Float;
use num_traits::FloatConst;
use num_traits::NumAssign;
use num_traits::NumAssignOps;
use num_traits::NumOps;

pub trait RealScalar:
    RealField + Float + FloatConst + NumAssign + std::fmt::Debug
{
    fn to_f32(self) -> f32;

    fn to_f64(self) -> f64;

    fn from_f32(f: f32) -> Self;

    fn from_f64(f: f64) -> Self;
}

pub trait ComplexScalar:
    ComplexField<Real = Self::Re>
    + Copy
    + Sized
    + NumAssign
    + Neg<Output = Self>
    + NumAssignOps<Self::Re>
    + NumAssignOps<Self::Conj>
    + NumOps<Self::Re, Self>
    + NumOps<Self::Conj, Self>
    + std::fmt::Debug
{
    const THRESHOLD: Self::Re;

    const GRAD_EPSILON: Self::Re;

    type Re: RealScalar;

    fn cis(theta: Self::Re) -> Self;

    fn sin(self) -> Self;

    fn cos(self) -> Self;

    fn abs(self) -> Self::Re;

    fn powu(self, n: u32) -> Self;

    fn powi(self, n: i32) -> Self;

    fn conj(self) -> Self;

    fn from_c32(c: c32) -> Self;

    fn from_c64(c: c64) -> Self;

    fn complex(re: impl RealScalar, im: impl RealScalar) -> Self;

    fn real(re: impl RealScalar) -> Self::Re;
}

impl ComplexScalar for c32 {
    type Re = f32;

    const GRAD_EPSILON: Self::Re = 1e-2;
    const THRESHOLD: Self::Re = 1e-5;

    #[inline]
    fn cis(theta: Self::Re) -> Self {
        c32::new(theta.cos(), theta.sin())
    }

    #[inline]
    fn sin(self) -> Self {
        self.sin()
    }

    #[inline]
    fn cos(self) -> Self {
        self.cos()
    }

    #[inline]
    fn abs(self) -> Self::Re {
        self.abs()
    }

    #[inline]
    fn powu(self, n: u32) -> Self {
        self.powu(n)
    }

    #[inline]
    fn powi(self, n: i32) -> Self {
        self.powi(n)
    }

    #[inline]
    fn conj(self) -> Self {
        self.conj()
    }

    #[inline]
    fn from_c32(c: c32) -> Self {
        c
    }

    #[inline]
    fn from_c64(c: c64) -> Self {
        c32::new(c.re as f32, c.im as f32)
    }

    #[inline]
    fn complex(re: impl RealScalar, im: impl RealScalar) -> Self {
        c32::new(re.to_f32(), im.to_f32())
    }

    #[inline]
    fn real(re: impl RealScalar) -> Self::Re {
        re.to_f32()
    }
}

impl ComplexScalar for c64 {
    type Re = f64;

    const GRAD_EPSILON: Self::Re = 1e-5;
    const THRESHOLD: Self::Re = 1e-10;

    #[inline]
    fn cis(theta: Self::Re) -> Self {
        c64::new(theta.cos(), theta.sin())
    }

    #[inline]
    fn sin(self) -> Self {
        self.sin()
    }

    #[inline]
    fn cos(self) -> Self {
        self.cos()
    }

    #[inline]
    fn abs(self) -> Self::Re {
        self.abs()
    }

    #[inline]
    fn powu(self, n: u32) -> Self {
        self.powu(n)
    }

    #[inline]
    fn powi(self, n: i32) -> Self {
        self.powi(n)
    }

    #[inline]
    fn conj(self) -> Self {
        self.conj()
    }

    #[inline]
    fn from_c32(c: c32) -> Self {
        c64::new(c.re as f64, c.im as f64)
    }

    #[inline]
    fn from_c64(c: c64) -> Self {
        c64::new(c.re, c.im)
    }

    #[inline]
    fn complex(re: impl RealScalar, im: impl RealScalar) -> Self {
        c64::new(re.to_f64(), im.to_f64())
    }

    #[inline]
    fn real(re: impl RealScalar) -> Self::Re {
        re.to_f64()
    }
}

// impl ComplexScalar for Complex<f32> {
//     type Re = f32;
//
//     const GRAD_EPSILON: Self::Re = 1e-2;
//     const THRESHOLD: Self::Re = 1e-5;
//
//     #[inline]
//     fn cis(theta: Self::Re) -> Self { Complex::cis(theta) }
//
//     #[inline]
//     fn sin(self) -> Self { Complex::sin(self) }
//
//     #[inline]
//     fn cos(self) -> Self { Complex::cos(self) }
//
//     #[inline]
//     fn powu(self, n: u32) -> Self { Complex::powu(&self, n) }
//
//     #[inline]
//     fn powi(self, n: i32) -> Self { Complex::powi(&self, n) }
//
//     #[inline]
//     fn conj(self) -> Self { self.conj() }
//
//     #[inline]
//     fn from_c32(c: c32) -> Self { c }
//
//     #[inline]
//     fn from_c64(c: c64) -> Self { Complex::new(c.re as f32, c.im as f32) }
//
//     #[inline]
//     fn complex(re: impl RealScalar, im: impl RealScalar) -> Self {
//         Complex::new(re.to_f32(), im.to_f32())
//     }
//
//     #[inline]
//     fn real(re: impl RealScalar) -> Self::Re { re.to_f32() }
// }
//
// impl ComplexScalar for Complex<f64> {
//     type Re = f64;
//
//     const GRAD_EPSILON: Self::Re = 1e-5;
//     const THRESHOLD: Self::Re = 1e-10;
//
//     #[inline]
//     fn cis(theta: Self::Re) -> Self { Complex::cis(theta) }
//
//     #[inline]
//     fn sin(self) -> Self { Complex::sin(self) }
//
//     #[inline]
//     fn cos(self) -> Self { Complex::cos(self) }
//
//     #[inline]
//     fn powu(self, n: u32) -> Self { Complex::powu(&self, n) }
//
//     #[inline]
//     fn powi(self, n: i32) -> Self { Complex::powi(&self, n) }
//
//     #[inline]
//     fn conj(self) -> Self { self.conj() }
//
//     #[inline]
//     fn from_c32(c: num_complex::Complex32) -> Self {
//         Complex::new(c.re as f64, c.im as f64)
//     }
//
//     #[inline]
//     fn from_c64(c: num_complex::Complex64) -> Self { c }
//
//     #[inline]
//     fn complex(re: impl RealScalar, im: impl RealScalar) -> Self {
//         Complex::new(re.to_f64(), im.to_f64())
//     }
//
//     #[inline]
//     fn real(re: impl RealScalar) -> Self::Re { re.to_f64() }
// }

impl RealScalar for f32 {
    #[inline]
    fn to_f32(self) -> f32 {
        self
    }

    #[inline]
    fn to_f64(self) -> f64 {
        self as f64
    }

    #[inline]
    fn from_f32(f: f32) -> Self {
        f
    }

    #[inline]
    fn from_f64(f: f64) -> Self {
        f as f32
    }
}

impl RealScalar for f64 {
    #[inline]
    fn to_f32(self) -> f32 {
        self as f32
    }

    #[inline]
    fn to_f64(self) -> f64 {
        self
    }

    #[inline]
    fn from_f32(f: f32) -> Self {
        f as f64
    }

    #[inline]
    fn from_f64(f: f64) -> Self {
        f
    }
}
