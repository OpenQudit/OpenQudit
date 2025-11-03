//! Implements a struct for unitary matrices and associated methods for the Openqudit library.

use std::fmt::Debug;
use std::fmt::Formatter;
use std::ops::Deref;
use std::ops::Index;
use std::ops::Mul;
use std::ops::Sub;

use coe::{coerce_static, is_same};
use faer::mat::AsMatRef;
use faer::unzip;
use faer::zip;
use num_traits::Float;

use crate::bitwidth::BitWidthConvertible;
use crate::c32;
use crate::c64;
use crate::ComplexScalar;
use crate::QuditPermutation;
use crate::QuditSystem;
use crate::Radices;
use crate::RealScalar;
use faer::Mat;
use faer::MatMut;
use faer::MatRef;

/// A unitary matrix over a qudit system.
///
/// This is a thin wrapper around a matrix that ensures it is unitary.
#[derive(Clone)]
pub struct UnitaryMatrix<C: ComplexScalar> {
    radices: Radices,
    matrix: Mat<C>,
}

impl<C: ComplexScalar> UnitaryMatrix<C> {
    /// Create a new unitary matrix.
    ///
    /// # Arguments
    ///
    /// * `radices` - The radices of the qudit system.
    ///
    /// * `matrix` - The matrix to wrap.
    ///
    /// # Panics
    ///
    /// Panics if the matrix is not unitary.
    ///
    /// # Example
    ///
    /// ```
    /// use faer::Mat;
    /// use qudit_core::UnitaryMatrix;
    /// use qudit_core::c64;
    /// let unitary: UnitaryMatrix<c64> = UnitaryMatrix::new([2, 2], Mat::<c64>::identity(4, 4));
    /// ```
    ///
    /// # See Also
    ///
    /// * [UnitaryMatrix::is_unitary] - Check if a matrix is a unitary.
    /// * [UnitaryMatrix::new_unchecked] - Create a unitary without checking unitary conditions.
    /// * [UnitaryMatrix::identity] - Create a unitary identity matrix.
    /// * [UnitaryMatrix::random] - Create a random unitary matrix.
    #[inline(always)]
    #[track_caller]
    pub fn new<T: Into<Radices>>(radices: T, matrix: Mat<C>) -> Self {
        assert!(Self::is_unitary(&matrix));
        Self {
            matrix,
            radices: radices.into(),
        }
    }

    /// Create a new unitary matrix without checking if it is unitary.
    ///
    /// # Arguments
    ///
    /// * `radices` - The radices of the qudit system.
    ///
    /// * `matrix` - The matrix to wrap.
    ///
    /// # Safety
    ///
    /// The caller must ensure that the provided matrix is unitary.
    ///
    /// # Example
    ///
    /// ```
    /// use faer::Mat;
    /// use qudit_core::UnitaryMatrix;
    /// use qudit_core::c64;
    /// let unitary: UnitaryMatrix<c64> = UnitaryMatrix::new_unchecked([2, 2], Mat::identity(4, 4));
    /// ```
    ///
    /// # See Also
    ///
    /// * [UnitaryMatrix::new] - Create a unitary matrix.
    /// * [UnitaryMatrix::identity] - Create a unitary identity matrix.
    /// * [UnitaryMatrix::random] - Create a random unitary matrix.
    /// * [UnitaryMatrix::is_unitary] - Check if a matrix is a unitary.
    #[inline(always)]
    #[track_caller]
    pub fn new_unchecked<T: Into<Radices>>(radices: T, matrix: Mat<C>) -> Self {
        Self {
            matrix,
            radices: radices.into(),
        }
    }

    /// Create a new identity unitary matrix for a given qudit system.
    ///
    /// # Arguments
    ///
    /// * `radices` - The radices of the qudit system.
    ///
    /// # Returns
    ///
    /// A new unitary matrix that is the identity.
    ///
    /// # Example
    ///
    /// ```
    /// use faer::Mat;
    /// use qudit_core::UnitaryMatrix;
    /// use qudit_core::c64;
    /// let unitary: UnitaryMatrix<c64> = UnitaryMatrix::identity([2, 2]);
    /// assert_eq!(unitary, UnitaryMatrix::new([2, 2], Mat::identity(4, 4)));
    /// ```
    ///
    /// # See Also
    ///
    /// * [UnitaryMatrix::new] - Create a unitary matrix.
    /// * [UnitaryMatrix::random] - Create a random unitary matrix.
    pub fn identity<T: Into<Radices>>(radices: T) -> Self {
        let radices = radices.into();
        let dim = radices.dimension();
        Self::new(radices, Mat::identity(dim, dim))
    }

    /// Generate a random Unitary from the haar distribution.
    ///
    /// Reference:
    /// - <https://arxiv.org/pdf/math-ph/0609050v2.pdf>
    ///
    /// # Arguments
    ///
    /// * `radices` - The radices of the qudit system.
    ///
    /// # Returns
    ///
    /// A new unitary matrix that is random.
    ///
    /// # Example
    ///
    /// ```
    /// use qudit_core::c64;
    /// use qudit_core::UnitaryMatrix;
    /// let unitary: UnitaryMatrix<c64> = UnitaryMatrix::random([2, 2]);
    /// assert!(UnitaryMatrix::is_unitary(&unitary));
    /// ```
    ///
    /// # See Also
    ///
    /// * [UnitaryMatrix::new] - Create a unitary matrix.
    /// * [UnitaryMatrix::identity] - Create a unitary identity matrix.
    pub fn random<T: Into<Radices>>(radices: T) -> Self {
        let radices = radices.into();
        let n = radices.dimension();
        let standard: Mat<C> =
            Mat::from_fn(n, n, |_, _| C::standard_random() / C::from_real(2.0).sqrt());
        let qr = standard.qr();
        let r = qr.R();
        let mut q = qr.compute_Q();
        for j in 0..n {
            let r = r[(j, j)];
            let r = if r == C::zero() {
                C::one()
            } else {
                r / r.abs()
            };

            zip!(q.as_mut().col_mut(j)).for_each(|unzip!(q)| {
                *q *= r;
            });
        }
        UnitaryMatrix::new(radices, q)
    }

    /// Check if a matrix is unitary.
    ///
    /// A matrix is unitary if it satisfies the following condition:
    /// ```math
    /// U U^\dagger = U^\dagger U = I
    /// ```
    ///
    /// Where `U` is the matrix, `U^\dagger` is the dagger (conjugate-transpose)
    /// of `U`, and `I` is the identity matrix of the same size.
    ///
    /// # Arguments
    ///
    /// * `mat` - The matrix to check.
    ///
    /// # Returns
    ///
    /// `true` if the matrix is unitary, `false` otherwise.
    ///
    /// # Example
    ///
    /// ```
    /// use qudit_core::c64;
    /// use faer::mat;
    /// use faer::Mat;
    /// use qudit_core::UnitaryMatrix;
    /// let mat: Mat<c64> = Mat::identity(2, 2);
    /// assert!(UnitaryMatrix::is_unitary(&mat));
    ///
    /// let mat = Mat::from_fn(2, 2, |_, _| c64::new(1.0, 1.0));
    /// assert!(!UnitaryMatrix::is_unitary(&mat));
    /// ```
    ///
    /// # Notes
    ///
    /// The function checks the l2 norm or frobenius norm of the difference
    /// between the product of the matrix and its adjoint and the identity.
    /// Due to floating point errors, the norm is checked against a threshold
    /// defined by the `THRESHOLD` constant in the `ComplexScalar` trait.
    ///
    /// # See Also
    ///
    /// * [ComplexScalar] - The floating point number type used for the matrix.
    /// * [ComplexScalar::THRESHOLD] - The threshold used to check if a matrix is unitary.
    pub fn is_unitary(mat: impl AsMatRef<T = C, Rows = usize, Cols = usize>) -> bool {
        let mat_ref = mat.as_mat_ref();

        if mat_ref.nrows() != mat_ref.ncols() {
            return false;
        }

        let id: Mat<C> = Mat::identity(mat_ref.nrows(), mat_ref.ncols());
        let product = mat_ref * mat_ref.adjoint().to_owned();
        let error = product - id;
        C::R::is_close(error.norm_l2(), 0.0)
    }

    /// Global-phase-agnostic, psuedo-metric over the space of unitaries.
    ///
    /// This is based on the hilbert-schmidt inner product. It is defined as:
    ///
    /// ```math
    /// \sqrt{1 - \big(\frac{|\text{tr}(A B^\dagger)|}{\text{dim}(A)}\big)^2}
    /// ```
    ///
    /// Where `A` and `B` are the unitaries, `B^\dagger` is the conjugate transpose
    /// of `B`, `|\text{tr}(A B^\dagger)|` is the absolute value of the trace of the
    /// product of `A` and `B^\dagger`, and `dim(A)` is the dimension of `A`.
    ///
    /// # Arguments
    ///
    /// * `x` - The other unitary matrix.
    ///
    /// # Returns
    ///
    /// The distance between the two unitaries.
    ///
    /// # Panics
    ///
    /// Panics if the two unitaries have different dimensions.
    ///
    /// # Example
    ///
    /// ```
    /// use qudit_core::c64;
    /// use faer::mat;
    /// use faer::Mat;
    /// use qudit_core::UnitaryMatrix;
    /// use qudit_core::ComplexScalar;
    /// let u1: UnitaryMatrix<c64> = UnitaryMatrix::identity([2, 2]);
    /// let u2 = UnitaryMatrix::identity([2, 2]);
    /// let u3 = UnitaryMatrix::random([2, 2]);
    /// assert_eq!(u1.get_distance_from(&u2), 0.0);
    /// assert!(u1.get_distance_from(&u3) > 0.0);
    /// ```
    ///
    /// # See Also
    ///
    /// * [ComplexScalar::THRESHOLD] - The threshold used to check if a matrix is unitary.
    pub fn get_distance_from(&self, x: impl AsMatRef<T = C, Rows = usize, Cols = usize>) -> C::R {
        let mat_ref = x.as_mat_ref();

        if mat_ref.nrows() != self.nrows() || mat_ref.ncols() != self.ncols() {
            panic!("Unitary and matrix must have same shape.");
        }

        let mut acc = C::ZERO;
        zip!(self.matrix.as_ref(), mat_ref).for_each(|unzip!(a, b)| {
            acc += *a * b.conj();
        });
        let num = acc.abs();
        let dem = C::R::from64(self.dimension() as f64);
        if num > dem {
            // This shouldn't happen but can due to floating point errors.
            // If it does, we correct it to zero.
            C::R::from64(0.0)
        } else {
            (C::R::from64(1.0) - (num / dem).powi(2i32)).sqrt()
        }
    }

    /// Permute the unitary matrix according to a qudit system permutation.
    ///
    /// # Arguments
    ///
    /// * `perm` - The permutation to apply.
    ///
    /// # Returns
    ///
    /// A newly allocated unitary matrix that is the result of applying the
    /// permutation to the original unitary matrix.
    ///
    /// # Panics
    ///
    /// Panics if there is a radix mismatch between the unitary matrix and
    /// the permutation.
    ///
    /// # Example
    ///
    /// ```
    /// use qudit_core::c64;
    /// use faer::mat;
    /// use faer::Mat;
    /// use qudit_core::UnitaryMatrix;
    /// use qudit_core::QuditPermutation;
    /// use num_traits::{One, Zero};
    /// let unitary: UnitaryMatrix<c64> = UnitaryMatrix::identity([2, 2]);
    /// let perm = QuditPermutation::new([2, 2], &vec![1, 0]);
    /// let permuted = unitary.permute(&perm);
    /// let mat = mat![
    ///     [c64::one(), c64::zero(), c64::zero(), c64::zero()],
    ///     [c64::zero(), c64::zero(), c64::one(), c64::zero()],
    ///     [c64::zero(), c64::one(), c64::zero(), c64::zero()],
    ///     [c64::zero(), c64::zero(), c64::zero(), c64::one()],
    /// ];
    /// assert_eq!(permuted, UnitaryMatrix::new([2, 2], Mat::identity(4, 4)));
    /// ```
    pub fn permute(&self, perm: &QuditPermutation) -> UnitaryMatrix<C> {
        assert_eq!(perm.radices(), self.radices());
        UnitaryMatrix::new(perm.permuted_radices(), perm.apply(&self.matrix))
    }

    /// Conjugate the unitary matrix.
    ///
    /// # Returns
    ///
    /// A newly allocated unitary matrix that is the conjugate of the original
    /// unitary matrix.
    ///
    /// # Example
    /// ```
    /// use qudit_core::c64;
    /// use faer::mat;
    /// use faer::Mat;
    /// use qudit_core::UnitaryMatrix;
    /// use num_traits::Zero;
    /// let y_mat = mat![
    ///    [c64::zero(), c64::new(0.0, -1.0)],
    ///    [c64::new(0.0, 1.0), c64::zero()],
    /// ];
    /// let unitary = UnitaryMatrix::new([2], y_mat);
    /// let conjugate = unitary.conjugate();
    /// let y_mat_conjugate = mat![
    ///     [c64::zero(), c64::new(0.0, 1.0)],
    ///     [c64::new(0.0, -1.0), c64::zero()],
    /// ];
    /// assert_eq!(conjugate, UnitaryMatrix::new([2], y_mat_conjugate));
    /// ```
    pub fn conjugate(&self) -> UnitaryMatrix<C> {
        Self::new(self.radices.clone(), self.matrix.conjugate().to_owned())
    }

    /// Transpose the unitary matrix.
    ///
    /// # Returns
    ///
    /// A newly allocated unitary matrix that is the transpose of the original
    /// unitary matrix.
    ///
    /// # Example
    ///
    /// ```
    /// use qudit_core::c64;
    /// use faer::mat;
    /// use faer::Mat;
    /// use qudit_core::UnitaryMatrix;
    /// use num_traits::Zero;
    /// let y_mat = mat![
    ///   [c64::zero(), c64::new(0.0, -1.0)],
    ///   [c64::new(0.0, 1.0), c64::zero()],
    /// ];
    /// let unitary = UnitaryMatrix::new([2], y_mat);
    /// let transpose = unitary.transpose();
    /// let y_mat_transpose = mat![
    ///     [c64::zero(), c64::new(0.0, 1.0)],
    ///     [c64::new(0.0, -1.0), c64::zero()],
    /// ];
    /// assert_eq!(transpose, UnitaryMatrix::new([2], y_mat_transpose));
    /// ```
    pub fn transpose(&self) -> UnitaryMatrix<C> {
        Self::new(self.radices.clone(), self.matrix.transpose().to_owned())
    }

    /// Adjoint or dagger the unitary matrix.
    ///
    /// # Returns
    ///
    /// A newly allocated unitary matrix that is the adjoint of the original
    /// unitary matrix.
    ///
    /// # Example
    ///
    /// ```
    /// use qudit_core::c64;
    /// use faer::mat;
    /// use faer::Mat;
    /// use qudit_core::UnitaryMatrix;
    /// use num_traits::Zero;
    /// let y_mat: Mat<c64> = mat![
    ///     [c64::zero(), c64::new(0.0, -1.0)],
    ///     [c64::new(0.0, 1.0), c64::zero()],
    /// ];
    /// let unitary = UnitaryMatrix::new([2], y_mat);
    /// let dagger: UnitaryMatrix<c64> = unitary.dagger();
    /// let y_mat_adjoint = mat![
    ///     [c64::zero(), c64::new(0.0, -1.0)],
    ///     [c64::new(0.0, 1.0), c64::zero()],
    /// ];
    /// assert_eq!(dagger, UnitaryMatrix::new([2], y_mat_adjoint));
    /// assert_eq!(dagger.dagger(), unitary);
    /// assert_eq!(dagger.dot(&unitary), Mat::<c64>::identity(2, 2));
    /// assert_eq!(unitary.dot(&dagger), Mat::<c64>::identity(2, 2));
    /// ```
    pub fn dagger(&self) -> Self {
        Self::new(self.radices.clone(), self.matrix.adjoint().to_owned())
    }

    /// Adjoint or dagger the unitary matrix (Alias for [`dagger`]).
    pub fn adjoint(&self) -> Self {
        self.dagger()
    }

    /// Multiply the unitary matrix by another matrix.
    ///
    /// # Arguments
    ///
    /// * `rhs` - The matrix to multiply by.
    ///
    /// # Returns
    ///
    /// A newly allocated unitary matrix that is the result of multiplying the
    /// original unitary matrix by the other matrix.
    ///
    /// # Panics
    ///
    /// Panics if the two matrices have different dimensions.
    ///
    /// # Example
    ///
    /// ```
    /// use qudit_core::c64;
    /// use faer::mat;
    /// use faer::Mat;
    /// use qudit_core::UnitaryMatrix;
    /// use num_traits::Zero;
    /// let y_mat = mat![
    ///    [c64::zero(), c64::new(0.0, -1.0)],
    ///    [c64::new(0.0, 1.0), c64::zero()],
    /// ];
    /// let unitary = UnitaryMatrix::new([2], y_mat.clone());
    /// let result = unitary.dot(&unitary);
    /// assert_eq!(result, UnitaryMatrix::new([2], y_mat.clone() * y_mat));
    /// ```
    ///
    /// # See Also
    ///
    /// * [accel::matmul] - The accelerated version of the matrix multiplication.
    pub fn dot(&self, rhs: impl AsMatRef<T = C, Rows = usize, Cols = usize>) -> Self {
        Self::new(
            self.radices.clone(),
            self.matrix.as_ref() * rhs.as_mat_ref(),
        )
    }

    /// Kronecker product the unitary matrix with another matrix.
    ///
    /// # Arguments
    ///
    /// * `rhs` - The matrix to kronecker product with.
    ///
    /// # Returns
    ///
    /// A newly allocated unitary matrix that is the result of kronecker producting
    /// the original unitary matrix with the other matrix.
    ///
    /// # Example
    ///
    /// ```
    /// use qudit_core::c64;
    /// use faer::mat;
    /// use faer::Mat;
    /// use qudit_core::UnitaryMatrix;
    /// use num_traits::Zero;
    /// let y_mat = mat![
    ///   [c64::zero(), c64::new(0.0, -1.0)],
    ///   [c64::new(0.0, 1.0), c64::zero()],
    /// ];
    /// let unitary = UnitaryMatrix::new([2], y_mat.clone());
    /// let result = unitary.kron(&unitary);
    /// let y_mat_kron = mat![
    ///    [c64::zero(), c64::zero(), c64::zero(), c64::new(-1.0, 0.0)],
    ///    [c64::zero(), c64::zero(), c64::new(1.0, 0.0), c64::zero()],
    ///    [c64::zero(), c64::new(1.0, 0.0), c64::zero(), c64::zero()],
    ///    [c64::new(-1.0, 0.0), c64::zero(), c64::zero(), c64::zero()],
    /// ];
    /// assert_eq!(result, UnitaryMatrix::new([2, 2], y_mat_kron));
    /// ```
    ///
    /// # See Also
    ///
    /// * [Mat::kron] - The method used to perform the kronecker product.
    /// * [accel::kron] - The accelerated version of the kronecker product.
    pub fn kron(&self, rhs: &UnitaryMatrix<C>) -> Self {
        let mut dst = Mat::zeros(self.nrows() * rhs.nrows(), self.ncols() * rhs.ncols());
        faer::linalg::kron::kron(dst.as_mut(), self.matrix.as_ref(), rhs.as_mat_ref());
        Self::new(self.radices.concat(&rhs.radices()), dst)
    }
}

impl<C: ComplexScalar> QuditSystem for UnitaryMatrix<C> {
    #[inline(always)]
    fn radices(&self) -> Radices {
        self.radices.clone()
    }

    #[inline(always)]
    fn num_qudits(&self) -> usize {
        self.radices.len()
    }

    #[inline(always)]
    fn dimension(&self) -> usize {
        self.matrix.nrows()
    }
}

impl<C: ComplexScalar> Deref for UnitaryMatrix<C> {
    type Target = Mat<C>;

    #[inline(always)]
    fn deref(&self) -> &Self::Target {
        &self.matrix
    }
}

impl<C: ComplexScalar> AsMatRef for UnitaryMatrix<C> {
    type T = C;
    type Rows = usize;
    type Cols = usize;
    type Owned = Mat<C>;

    #[inline(always)]
    fn as_mat_ref(&self) -> MatRef<'_, C> {
        self.matrix.as_ref()
    }
}

impl<C: ComplexScalar> Index<(usize, usize)> for UnitaryMatrix<C> {
    type Output = C;

    #[inline(always)]
    fn index(&self, index: (usize, usize)) -> &Self::Output {
        &self.matrix[index]
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
            self[(i, j)] * rhs[(i, j)]
        });
        UnitaryMatrix::new(self.radices, output)
    }
}

impl<C: ComplexScalar> Mul<&UnitaryMatrix<C>> for Mat<C> {
    type Output = Mat<C>;

    fn mul(self, rhs: &UnitaryMatrix<C>) -> Self::Output {
        Mat::from_fn(self.nrows(), self.ncols(), |i, j| {
            self[(i, j)] * rhs[(i, j)]
        })
    }
}

impl<C: ComplexScalar> Mul<UnitaryMatrix<C>> for Mat<C> {
    type Output = Mat<C>;

    fn mul(self, rhs: UnitaryMatrix<C>) -> Self::Output {
        Mat::from_fn(self.nrows(), self.ncols(), |i, j| {
            self[(i, j)] * rhs[(i, j)]
        })
    }
}

impl<C: ComplexScalar> Mul<&UnitaryMatrix<C>> for &Mat<C> {
    type Output = Mat<C>;

    fn mul(self, rhs: &UnitaryMatrix<C>) -> Self::Output {
        Mat::from_fn(self.nrows(), self.ncols(), |i, j| {
            self[(i, j)] * rhs[(i, j)]
        })
    }
}

impl<C: ComplexScalar> Mul<UnitaryMatrix<C>> for &Mat<C> {
    type Output = Mat<C>;

    fn mul(self, rhs: UnitaryMatrix<C>) -> Self::Output {
        Mat::from_fn(self.nrows(), self.ncols(), |i, j| {
            self[(i, j)] * rhs[(i, j)]
        })
    }
}

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

impl<C: ComplexScalar> PartialEq<Mat<i32>> for UnitaryMatrix<C> {
    fn eq(&self, other: &Mat<i32>) -> bool {
        self.matrix.nrows() == other.nrows()
            && self.matrix.ncols() == other.ncols()
            && self.matrix.col_iter().zip(other.col_iter()).all(|(a, b)| {
                a.iter()
                    .zip(b.iter())
                    .all(|(a, b)| *a == C::from(*b).unwrap())
            })
    }
}

impl<C: ComplexScalar> Eq for UnitaryMatrix<C> {}

impl<C: ComplexScalar> From<UnitaryMatrix<C>> for Mat<C> {
    fn from(unitary: UnitaryMatrix<C>) -> Self {
        unitary.matrix
    }
}

impl<'a, C: ComplexScalar> From<&'a UnitaryMatrix<C>> for MatRef<'a, C> {
    fn from(unitary: &'a UnitaryMatrix<C>) -> Self {
        unitary.matrix.as_ref()
    }
}

impl<'a, C: ComplexScalar> From<&'a mut UnitaryMatrix<C>> for MatRef<'a, C> {
    fn from(unitary: &'a mut UnitaryMatrix<C>) -> Self {
        unitary.matrix.as_ref()
    }
}

impl<'a, C: ComplexScalar> From<&'a mut UnitaryMatrix<C>> for MatMut<'a, C> {
    fn from(unitary: &'a mut UnitaryMatrix<C>) -> Self {
        unitary.matrix.as_mut()
    }
}

impl<C: ComplexScalar> BitWidthConvertible for UnitaryMatrix<C> {
    type Width32 = UnitaryMatrix<c32>;
    type Width64 = UnitaryMatrix<c64>;

    fn to32(self) -> Self::Width32 {
        if is_same::<c32, C>() {
            coerce_static(self)
        } else {
            let matrix = Mat::from_fn(self.matrix.nrows(), self.matrix.ncols(), |i, j| {
                self.matrix[(i, j)].to32()
            });
            UnitaryMatrix::new(self.radices, matrix)
        }
    }

    fn to64(self) -> Self::Width64 {
        if is_same::<c64, C>() {
            coerce_static(self)
        } else {
            let matrix = Mat::from_fn(self.matrix.nrows(), self.matrix.ncols(), |i, j| {
                self.matrix[(i, j)].to64()
            });
            UnitaryMatrix::new(self.radices, matrix)
        }
    }

    fn from32(unitary: Self::Width32) -> Self {
        if is_same::<c32, C>() {
            coerce_static(unitary)
        } else {
            let matrix = Mat::from_fn(unitary.matrix.nrows(), unitary.matrix.ncols(), |i, j| {
                C::from32(unitary.matrix[(i, j)])
            });
            UnitaryMatrix::new(unitary.radices, matrix)
        }
    }

    fn from64(unitary: Self::Width64) -> Self {
        if is_same::<c64, C>() {
            coerce_static(unitary)
        } else {
            let matrix = Mat::from_fn(unitary.matrix.nrows(), unitary.matrix.ncols(), |i, j| {
                C::from64(unitary.matrix[(i, j)])
            });
            UnitaryMatrix::new(unitary.radices, matrix)
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;

    impl<C: ComplexScalar> UnitaryMatrix<C> {
        /// Checks if the unitary matrix is close to another unitary matrix.
        ///
        /// # Arguments
        ///
        /// * `x` - The other unitary matrix to compare to.
        ///
        /// # Notes
        ///
        /// This is a global-phase-agnostic check. It uses the
        /// `get_distance_from` method to calculate the distance
        /// between the two matrices.
        ///
        /// # See Also
        ///
        /// * [UnitaryMatrix::get_distance_from] - The method used to calculate the distance.
        pub fn assert_close_to(&self, x: impl AsMatRef<T = C, Rows = usize, Cols = usize>) {
            let dist = self.get_distance_from(x);
            assert!(
                C::R::is_close(dist, 0.0),
                "Distance between unitaries is {:?}",
                dist
            )
        }
    }
}

#[cfg(feature = "python")]
mod python {
    use super::*;
    use crate::PyRegistrar;
    use pyo3::{exceptions::PyTypeError, prelude::*};

    use numpy::{PyArray2, PyReadonlyArray2, PyUntypedArrayMethods};

    /// Python wrapper for UnitaryMatrix
    #[pyclass(name = "UnitaryMatrix")]
    #[derive(Clone)]
    pub struct PyUnitaryMatrix {
        inner: UnitaryMatrix<c64>,
    }

    #[pymethods]
    impl PyUnitaryMatrix {
        /// Create a new unitary matrix from a numpy array
        #[new]
        #[pyo3(signature = (radices, matrix))]
        fn new(radices: Radices, matrix: PyReadonlyArray2<c64>) -> PyResult<Self> {
            let array = matrix.as_array();
            let (nrows, ncols) = array.dim();

            let mat = Mat::from_fn(nrows, ncols, |i, j| array[[i, j]]);

            if !UnitaryMatrix::is_unitary(&mat) {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "Matrix is not unitary",
                ));
            }

            Ok(Self {
                inner: UnitaryMatrix::new(radices, mat),
            })
        }

        /// Convert to numpy array
        fn to_array<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<c64>> {
            let rows: Vec<Vec<c64>> = (0..self.inner.nrows())
                .map(|i| {
                    (0..self.inner.ncols())
                        .map(|j| self.inner[(i, j)])
                        .collect()
                })
                .collect();
            PyArray2::from_vec2(py, &rows).unwrap()
        }
    }

    // Conversion traits
    impl From<UnitaryMatrix<c64>> for PyUnitaryMatrix {
        fn from(inner: UnitaryMatrix<c64>) -> Self {
            Self { inner }
        }
    }

    impl From<PyUnitaryMatrix> for UnitaryMatrix<c64> {
        fn from(py_unitary: PyUnitaryMatrix) -> Self {
            py_unitary.inner
        }
    }

    impl<'py> IntoPyObject<'py> for UnitaryMatrix<c64> {
        type Target = PyUnitaryMatrix;
        type Output = Bound<'py, Self::Target>;
        type Error = PyErr;

        fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
            Bound::new(py, PyUnitaryMatrix::from(self))
        }
    }

    impl<'a, 'py> FromPyObject<'a, 'py> for UnitaryMatrix<c64> {
        type Error = PyErr;

        fn extract(obj: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
            if let Ok(py_unitary) = obj.extract::<PyRef<PyUnitaryMatrix>>() {
                Ok(py_unitary.inner.clone())
            } else if let Ok(py_unitary) = obj.extract::<PyReadonlyArray2<c64>>() {
                let shape = py_unitary.shape();
                let dimension = shape[0];
                let radices = Radices::guess(dimension);
                Ok(PyUnitaryMatrix::new(radices, py_unitary)
                    .unwrap()
                    .inner
                    .clone())
            } else {
                Err(PyTypeError::new_err(
                    "Invalid type for unitary matrix extraction.",
                ))
            }
        }
    }

    /// Registers the PyUnitaryMatrix with the Python Module.
    fn register(parent_module: &Bound<'_, PyModule>) -> PyResult<()> {
        parent_module.add_class::<PyUnitaryMatrix>()?;
        Ok(())
    }
    inventory::submit! { PyRegistrar { func: register } }
}
