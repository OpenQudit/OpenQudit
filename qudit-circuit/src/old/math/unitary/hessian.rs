use std::ops::Mul;

use num_traits::ToPrimitive;

#[cfg(test)]
use super::UnitaryGradient;
use crate::math::matrix::MatHess;
use crate::math::ComplexScalar;
use crate::QuditRadices;

// Hessian is symmetric, this object saves on computation and space
// by only storing the upper triangle of the matrix

// #[derive(Clone, Debug, PartialEq)]
#[derive(PartialEq, Debug)]
pub struct UnitaryHessian<C: ComplexScalar> {
    radices: QuditRadices,
    stride: usize,

    // FLattenned, is symmetric so we save space
    // by virtualizing the the bottom left corner of the super matrix
    pub partials: MatHess<C>,
}

impl<C: ComplexScalar> UnitaryHessian<C> {
    pub fn new(
        radices: QuditRadices,
        stride: usize,
        partials: MatHess<C>,
    ) -> Self {
        Self {
            radices,
            stride,
            partials,
        }
    }

    pub fn zeros(radices: QuditRadices, num_params: usize) -> Self {
        let dim = radices.get_dimension();
        Self::new(radices, num_params, MatHess::zeros(num_params, dim, dim))
    }

    pub fn stride(&self) -> usize {
        self.stride
    }

    // Since the Hessian is stored in compact form as column-major,
    // we first find j by solving for the smallest N such that
    // N(N+1)/2 <= k. We can then just undo the equation for k in
    // coords to index.
    pub fn index_to_coords(&self, index: usize) -> (usize, usize) {
        // TODO: check if index is in bounds
        let j = (((8 * index + 1) as f64).sqrt().floor().to_usize().unwrap()
            - 1)
            / 2;
        let i = index - j * (j + 1) / 2;
        (i, j)
    }

    /// When storing the upper triangular part of a matrix (including the
    /// diagonal) into a compact vector, you essentially flatten the
    /// upper triangular part of the matrix column-wise into a one-dimensional
    /// array. Let's say you have an N*N matrix and a compact vector V of
    /// length N(N+1)/2 to store the upper triangular part of the matrix.
    /// For a matrix coordinate (i,j) in the upper triangular part
    /// where i<=j, the corresponding vector index k can be calculated
    /// using the formula:
    /// ```math
    ///     k = j * (j+1) / 2 + i
    /// ```
    pub fn coords_to_index(&self, coords: (usize, usize)) -> usize {
        // TODO: check if coords is in bounds
        let (i, j) = coords;
        if i <= j {
            j * (j + 1) / 2 + i
        } else {
            i * (i + 1) / 2 + j
        }
    }
}

// impl<C: ComplexScalar> Index<(usize, usize)> for UnitaryHessian<C> {
//     type Output = Mat<C>;

//     fn index(&self, index: (usize, usize)) -> &Self::Output {
//         &self.partials[self.coords_to_index(index)]
//     }
// }

// impl<C: ComplexScalar> IndexMut<(usize, usize)> for UnitaryHessian<C> {
//     fn index_mut(&mut self, index: (usize, usize)) -> &mut Self::Output {
//         let i = self.coords_to_index(index);
//         &mut self.partials[i]
//     }
// }

// impl<C: ComplexScalar> Deref for UnitaryHessian<C> {
//     type Target = [Mat<C>];

//     fn deref(&self) -> &Self::Target { &self.partials }
// }

// impl<C: ComplexScalar> DerefMut for UnitaryHessian<C> {
//     fn deref_mut(&mut self) -> &mut Self::Target { &mut self.partials }
// }

impl<C: ComplexScalar> Mul<C> for UnitaryHessian<C> {
    type Output = UnitaryHessian<C>;

    fn mul(mut self, rhs: C) -> Self::Output {
        for p1 in 0..self.partials.num_params() {
            for p2 in p1..self.partials.num_params() {
                for i in 0..self.partials.nrows() {
                    for j in 0..self.partials.ncols() {
                        self.partials.write(
                            p1,
                            p2,
                            i,
                            j,
                            rhs * self.partials.read(p1, p2, i, j),
                        );
                    }
                }
            }
        }
        self
    }
}

// impl<C: ComplexScalar, T: Mul<Mat<C>>> Mul<T> for UnitaryHessian<C> {
//     type Output = UnitaryHessian<C>;

//     fn mul(self, rhs: T) -> Self::Output {
//         let partials = self.partials.into_iter().map(|p| rhs * p).collect();
//         Self::new(self.radices, self.stride, partials)
//     }
// }

// impl<C: ComplexScalar> Mul<C::Re> for UnitaryHessian<C> {
//     type Output = Self;

//     fn mul(self, rhs: C::Re) -> Self::Output {
//         let crhs = C::real(rhs);
//         let partials = self.partials.into_iter().map(|p| p * crhs).collect();
//         Self::new(self.radices, self.stride, partials)
//     }
// }

// impl<C: ComplexScalar> Mul<UnitaryHessian<C>> for C::Re {
//     type Output = Self;

//     fn mul(self, rhs: C::Re) -> Self::Output {
//         C::real(rhs) * self
//     }
// }

// impl<C: ComplexScalar> Mul<C> for UnitaryHessian<C> {
//     type Output = Self;

//     fn mul(self, rhs: C) -> Self::Output {
//         let partials = self.partials.into_iter().map(|p| p * rhs).collect();
//         Self::new(self.radices, self.stride, partials)
//     }
// }

// impl<C: ComplexScalar> Mul<UnitaryHessian<C>> for C {
//     type Output = UnitaryHessian<C>;

//     fn mul(self, rhs: UnitaryHessian<C>) -> Self::Output {
//         rhs * self
//     }
// }

#[cfg(test)]
impl<C: ComplexScalar> UnitaryHessian<C> {
    pub fn get_row(&self, index: usize) -> UnitaryGradient<C> {
        // Copy for testing purposes
        // If this becomes used in production:
        // make a reference by defining UnitaryGradientRef or something
        // TODO: check if index is in bounds

        let mut partials = Vec::new();

        for j in 0..self.stride {
            partials.push(self.partials.get_matref(index, j).to_owned());
        }

        UnitaryGradient::new(self.radices.clone(), partials)
    }
}
