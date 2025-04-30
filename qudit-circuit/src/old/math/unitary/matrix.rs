use std::fmt::Debug;
use std::fmt::Formatter;
use std::ops::Deref;
use std::ops::DerefMut;
use std::ops::Mul;
use std::ops::Sub;

use faer_core::unzipped;
use faer_core::zipped;
use faer_core::AsMatMut;
use faer_core::AsMatRef;
use faer_core::Mat;
use faer_core::MatMut;
use faer_core::MatRef;
use num_traits::Float;

use crate::math::ComplexScalar;
use crate::QuditPermutation;
use crate::QuditRadices;
use crate::QuditSystem;

#[derive(Clone)]
pub struct UnitaryMatrix<C: ComplexScalar> {
    matrix: Mat<C>,
    radices: QuditRadices,
}

impl<C: ComplexScalar> UnitaryMatrix<C> {
    pub fn new(radices: QuditRadices, matrix: Mat<C>) -> Self {
        if cfg!(debug_assertions) {
            assert!(Self::is_unitary(&matrix));
        }

        Self { matrix, radices }
    }

    pub fn identity(radices: QuditRadices) -> Self {
        let dim = radices.get_dimension();
        Self::new(radices, Mat::identity(dim, dim))
    }

    /// Generate a random Unitary from the haar distribution.
    ///
    /// Reference:
    /// - <https://arxiv.org/pdf/math-ph/0609050v2.pdf>
    pub fn random(_radices: QuditRadices) -> Self {
        // Waiting on faer mat to implement random unitary
        todo!()
    }

    pub fn is_unitary(mat: impl AsMatRef<C>) -> bool {
        let mat_ref = mat.as_mat_ref();

        if mat_ref.nrows() != mat_ref.ncols() {
            return false;
        }

        let id: Mat<C> = Mat::identity(mat_ref.nrows(), mat_ref.ncols());
        let product = mat_ref * mat_ref.adjoint().to_owned();
        let error = product - id;
        error.norm_l2() < C::THRESHOLD
    }

    /// Global-phase-agnostic, psuedo-metric over the space of unitaries.
    /// This is based on the hilbert-schmidt inner product.
    /// It is defined as:
    ///
    /// $$
    /// \sqrt{1 - \big(\frac{|\text{tr}(A B^\dagger)|}{\text{dim}(A)}\big)^2}
    /// $$
    pub fn get_distance_from(&self, x: impl AsMatRef<C>) -> C::Re {
        let mat_ref = x.as_mat_ref().conjugate();

        if mat_ref.nrows() != self.nrows() || mat_ref.ncols() != self.ncols() {
            panic!("Unitary and matrix must have same shape.");
        }

        let mut acc = C::zero();
        zipped!(self.matrix.as_ref(), mat_ref).for_each(|unzipped!(a, b)| {
            acc += a.read() * b.read();
        });
        let num = acc.abs();
        let dem = C::real(self.get_dimension() as f64);
        if num > dem {
            // This shouldn't happen but can due to floating point errors.
            // If it does, we correct it to zero.
            C::real(0.0)
        } else {
            (C::real(1.0) - (num / dem).powi(2i32)).sqrt()
        }
    }

    pub fn permute(&self, perm: &QuditPermutation) -> UnitaryMatrix<C> {
        if cfg!(debug_assertions) {
            assert_eq!(perm.get_radices(), self.get_radices());
        }
        UnitaryMatrix::new(
            perm.get_permuted_radices(),
            perm.apply(&self.matrix),
        )
    }

    // TODO: Consider renaming to "From/Into"  style
    // pub fn to<C2: ComplexScalar>(self) -> UnitaryMatrix<C2> {
    //     if TypeId::of::<C>() == TypeId::of::<C2> {
    //         return self;
    //     }

    //     UnitaryMatrix::new(
    //         self.radices,
    //         self.matrix.map(|c| C2::complex(c.re(), c.im())),
    //     )
    // }

    pub fn conjugate(mut self) -> UnitaryMatrix<C> {
        zipped!(self.matrix.as_mut())
            .for_each(|unzipped!(mut e)| e.write(e.read().conj()));
        Self::new(self.radices, self.matrix)
    }

    pub fn transpose(self) -> UnitaryMatrix<C> {
        // TODO: redo when faer gets in-place transpose
        Self::new(self.radices, self.matrix.transpose().to_owned())
    }

    pub fn dagger(self) -> Self {
        Self::new(self.radices, self.matrix.adjoint().to_owned())
    }

    pub fn dot(self, rhs: impl AsMatRef<C>) -> Self {
        Self::new(self.radices.clone(), self.matrix * rhs.as_mat_ref())
    }
}

impl<C: ComplexScalar> QuditSystem for UnitaryMatrix<C> {
    fn get_radices(&self) -> QuditRadices {
        self.radices.clone()
    }

    fn get_dimension(&self) -> usize {
        self.matrix.nrows()
    }
}

impl<C: ComplexScalar> Deref for UnitaryMatrix<C> {
    type Target = Mat<C>;

    fn deref(&self) -> &Self::Target {
        &self.matrix
    }
}

impl<C: ComplexScalar> DerefMut for UnitaryMatrix<C> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.matrix
    }
}

impl<C: ComplexScalar> AsMatRef<C> for &UnitaryMatrix<C> {
    fn as_mat_ref(&self) -> MatRef<'_, C> {
        self.matrix.as_ref()
    }
}

impl<C: ComplexScalar> AsMatRef<C> for UnitaryMatrix<C> {
    fn as_mat_ref(&self) -> MatRef<'_, C> {
        self.matrix.as_ref()
    }
}

impl<C: ComplexScalar> AsMatMut<C> for UnitaryMatrix<C> {
    fn as_mat_mut(&mut self) -> MatMut<'_, C> {
        self.matrix.as_mut()
    }
}

impl<C: ComplexScalar> AsMatMut<C> for &mut UnitaryMatrix<C> {
    fn as_mat_mut(&mut self) -> MatMut<'_, C> {
        self.matrix.as_mut()
    }
}

impl<C: ComplexScalar> Sub<UnitaryMatrix<C>> for UnitaryMatrix<C> {
    type Output = Mat<C>;

    fn sub(self, rhs: Self) -> Self::Output {
        self.matrix - rhs.matrix
    }
}

impl<C: ComplexScalar> Mul<UnitaryMatrix<C>> for UnitaryMatrix<C> {
    type Output = UnitaryMatrix<C>;

    fn mul(self, rhs: Self) -> Self::Output {
        let output = Mat::from_fn(self.nrows(), self.ncols(), |i, j| {
            self.read(i, j) * rhs.read(i, j)
        });
        UnitaryMatrix::new(self.radices, output)
    }
}

impl<C: ComplexScalar> Mul<&UnitaryMatrix<C>> for Mat<C> {
    type Output = UnitaryMatrix<C>;

    fn mul(self, rhs: &UnitaryMatrix<C>) -> Self::Output {
        let output = Mat::from_fn(self.nrows(), self.ncols(), |i, j| {
            self.read(i, j) * rhs.read(i, j)
        });
        UnitaryMatrix::new(rhs.radices.clone(), output)
    }
}

impl<C: ComplexScalar> Mul<UnitaryMatrix<C>> for Mat<C> {
    type Output = UnitaryMatrix<C>;

    fn mul(self, rhs: UnitaryMatrix<C>) -> Self::Output {
        let output = Mat::from_fn(self.nrows(), self.ncols(), |i, j| {
            self.read(i, j) * rhs.read(i, j)
        });
        UnitaryMatrix::new(rhs.radices, output)
    }
}

impl<C: ComplexScalar> Mul<&UnitaryMatrix<C>> for &Mat<C> {
    type Output = UnitaryMatrix<C>;

    fn mul(self, rhs: &UnitaryMatrix<C>) -> Self::Output {
        let output = Mat::from_fn(self.nrows(), self.ncols(), |i, j| {
            self.read(i, j) * rhs.read(i, j)
        });
        UnitaryMatrix::new(rhs.radices.clone(), output)
    }
}

impl<C: ComplexScalar> Mul<UnitaryMatrix<C>> for &Mat<C> {
    type Output = UnitaryMatrix<C>;

    fn mul(self, rhs: UnitaryMatrix<C>) -> Self::Output {
        let output = Mat::from_fn(self.nrows(), self.ncols(), |i, j| {
            self.read(i, j) * rhs.read(i, j)
        });
        UnitaryMatrix::new(rhs.radices, output)
    }
}

// impl<C: ComplexScalar> Kronecker<UnitaryMatrix<C>> for UnitaryMatrix<C>
// {
//     type Output = UnitaryMatrix<C>;

//     fn kron(
//         &self,
//         rhs: &UnitaryMatrix<C>,
//     ) -> Self::Output
//     {
//         let radices = self.radices.clone();
//         let data = self.matrix.kron(&rhs.data);
//         UnitaryMatrix::new(radices, data)
//     }
// }

impl<C: ComplexScalar> Debug for UnitaryMatrix<C> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        // TODO: print radices and unitary's complex numbers more cleanly
        write!(f, "Unitary({:?})", self.matrix)
    }
}

impl<C: ComplexScalar> PartialEq<UnitaryMatrix<C>> for UnitaryMatrix<C> {
    fn eq(&self, other: &UnitaryMatrix<C>) -> bool {
        self.matrix == other.matrix
    }
}

impl<C: ComplexScalar> PartialEq<Mat<C>> for UnitaryMatrix<C> {
    fn eq(&self, other: &Mat<C>) -> bool {
        self.matrix == *other
    }
}

impl<C: ComplexScalar> Eq for UnitaryMatrix<C> {}

#[cfg(test)]
mod test {
    use super::*;

    impl<C: ComplexScalar> UnitaryMatrix<C> {
        pub fn assert_close_to(&self, x: impl AsMatRef<C>) {
            let dist = self.get_distance_from(x);
            assert!(
                dist < C::real(5e-7),
                "Distance between unitaries is {:?}",
                dist
            )
        }
    }
}
