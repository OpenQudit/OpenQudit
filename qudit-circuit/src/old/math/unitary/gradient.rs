use std::ops::Deref;
use std::ops::DerefMut;
use std::ops::Index;
use std::ops::IndexMut;
use std::ops::Mul;
use std::ops::Sub;

use faer_core::Mat;

use crate::math::ComplexScalar;
use crate::QuditRadices;

#[derive(Clone, Debug, PartialEq)]
pub struct UnitaryGradient<C: ComplexScalar> {
    partials: Vec<Mat<C>>,
    radices: QuditRadices,
}

impl<C: ComplexScalar> UnitaryGradient<C> {
    pub fn new(radices: QuditRadices, partials: Vec<Mat<C>>) -> Self {
        Self { partials, radices }
    }

    pub fn zeros(radices: QuditRadices, num_params: usize) -> Self {
        let dim = radices.get_dimension();
        let partials = (0..num_params).map(|_| Mat::zeros(dim, dim)).collect();
        Self::new(radices, partials)
    }
}

impl<C: ComplexScalar> Index<usize> for UnitaryGradient<C> {
    type Output = Mat<C>;

    fn index(&self, index: usize) -> &Self::Output {
        // TODO: check if index is in bounds
        &self.partials[index]
    }
}

impl<C: ComplexScalar> IndexMut<usize> for UnitaryGradient<C> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        // TODO: check if index is in bounds
        &mut self.partials[index]
    }
}

impl<C: ComplexScalar> Deref for UnitaryGradient<C> {
    type Target = [Mat<C>];

    fn deref(&self) -> &Self::Target {
        &self.partials
    }
}

impl<C: ComplexScalar> DerefMut for UnitaryGradient<C> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.partials
    }
}

// impl<C: ComplexScalar> Mul<C::Re> for UnitaryGradient<C> {
//     type Output = Self;

//     fn mul(self, rhs: C::Re) -> Self::Output {
//         let crhs = C::real(rhs);
//         let partials = self.partials.into_iter().map(|p| p * crhs).collect();
//         Self::new(self.radices, partials)
//     }
// }

// impl<C: ComplexScalar> Mul<UnitaryGradient<C>> for C::Re {
//     type Output = Self;

//     fn mul(self, rhs: C::Re) -> Self::Output {
//         C::real(rhs) * self
//     }
// }

impl<C: ComplexScalar> Mul<C> for UnitaryGradient<C> {
    type Output = UnitaryGradient<C>;

    fn mul(mut self, rhs: C) -> Self::Output {
        self.partials.iter_mut().for_each(|p| {
            for i in 0..p.nrows() {
                for j in 0..p.ncols() {
                    p.write(i, j, rhs * p.read(i, j));
                }
            }
        });
        Self::new(self.radices, self.partials)
    }
}

impl<C: ComplexScalar> Sub<UnitaryGradient<C>> for UnitaryGradient<C> {
    type Output = Self;

    fn sub(self, rhs: UnitaryGradient<C>) -> Self::Output {
        let partials = self
            .partials
            .into_iter()
            .zip(rhs.partials)
            .map(|(p1, p2)| p1 - p2)
            .collect();
        Self::new(self.radices, partials)
    }
}

// impl<C: ComplexScalar> Mul<UnitaryGradient<C>> for C {
//     type Output = UnitaryGradient<C>;

//     fn mul(self, rhs: UnitaryGradient<C>) -> Self::Output {
//         rhs * self
//     }
// }
