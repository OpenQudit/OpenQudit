use faer_core::unzipped;
use faer_core::zipped;
use faer_core::MatMut;
use faer_core::MatRef;

use super::ComplexScalar;

// pub trait Conj
// {
//     type Output;

//     fn conj(&self) -> Self::Output;
// }

// impl<A, S, D> Conj for ArrayBase<S, D>
// where
//     A: Scalar,
//     S: Data<Elem = A>,
//     D: Dimension,
// {
//     type Output = ArrayBase<OwnedRepr<A>, D>;

//     fn conj(&self) -> ArrayBase<OwnedRepr<A>, D> { self.mapv(|i| i.conj()) }
// }

// pub fn trace<T: Scalar>(arr: ArrayView4<T>) -> Array2<T>
// {
//     let mut out = Array2::<T>::zeros((arr.shape()[2], arr.shape()[3]));
//     for i in 0..arr.shape()[2]
//     {
//         for j in 0..arr.shape()[3]
//         {
//             out[(i, j)] = arr
//                 .slice(s![.., .., i, ..])
//                 .slice(s![.., .., j])
//                 .into_dimensionality::<Ix2>()
//                 .unwrap()
//                 .into_owned()
//                 .diag()
//                 .sum();
//         }
//     }
//     out
// }

// /// Calculate the Kronecker product of two matrices.
// ///
// /// # Arguments
// ///
// /// * `dim` - The dimension of the `b` matrix.
// /// * `a` - The first matrix.
// /// * `b` - The second matrix.
// /// * `out` - The output matrix.
// ///
// /// # Examples
// ///
// /// ```
// /// use qudit_circuit::math::kron;
// ///
// /// let a = Array::from_shape_vec((2, 2), vec![1, 2, 3, 4]).unwrap();
// /// let b = Array::from_shape_vec((2, 2), vec![5, 6, 7, 8]).unwrap();
// /// let mut out = Array::zeros((4, 4));
// /// kron(2, &a, &b, &mut out);
// /// assert_eq!(out, Array2::from_shape_vec((4, 4), vec![
// ///     5, 6, 10, 12,
// ///     7, 8, 14, 16,
// ///     15, 18, 20, 24,
// ///     21, 24, 28, 32]
// /// ).unwrap());
// /// ```
// pub fn kron<A, S1, S2, S3>(
//     dim: usize,
//     a: &ArrayBase<S1, Ix2>,
//     b: &ArrayBase<S2, Ix2>,
//     out: &mut ArrayBase<S3, Ix2>,
// ) where
//     S1: Data<Elem = A>,
//     S2: Data<Elem = A>,
//     S3: DataMut<Elem = A>,
//     A: LinalgScalar,
// {
//     Zip::from(out.exact_chunks_mut((dim, dim)))
//         .and(a)
//         .for_each(|out, &a| {
//             Zip::from(out).and(b).for_each(|out, &b| {
//                 *out = a * b;
//             })
//         });
// }
// pub trait Kronecker<Rhs>
// {
//     type Output;

//     fn kron(
//         &self,
//         x: &Rhs,
//     ) -> Self::Output;
// }

// impl<T, S, S2> Kronecker<ArrayBase<S2, Ix2>> for ArrayBase<S, Ix2>
// where
//     S: Data<Elem = T>,
//     S2: Data<Elem = T>,
//     T: Scalar,
// {
//     type Output = ArrayBase<OwnedRepr<T>, Ix2>;

//     fn kron(
//         &self,
//         x: &ArrayBase<S2, Ix2>,
//     ) -> ArrayBase<OwnedRepr<T>, Ix2>
//     where
//         S2: Data<Elem = T>,
//     {
//         let dimar = self.shape()[0];
//         let dimac = self.shape()[1];
//         let dimbr = x.shape()[0];
//         let dimbc = x.shape()[1];
//         let mut out: Array2<MaybeUninit<T>> = Array2::uninit((dimar * dimbr,
// dimac * dimbc));         Zip::from(out.exact_chunks_mut((dimbr, dimbc)))
//             .and(self)
//             .for_each(|out, &a| {
//                 Zip::from(out).and(x).for_each(|out, &b| {
//                     *out = MaybeUninit::new(a * b);
//                 });
//             });
//         unsafe { out.assume_init() }
//     }
// }

// pub trait Norm
// {
//     type Output;

//     fn norm(&self) -> Self::Output;
// }

// impl<R, S, D> Norm for ArrayBase<S, D>
// where
//     R: Scalar,
//     S: Data<Elem = R>,
//     D: Dimension,
// {
//     type Output = R::Real;

//     fn norm(&self) -> R::Real { self.fold(R::real(0.0), |acc, &x| acc +
// x.square()).sqrt() } }

pub fn matrix_kron<C: ComplexScalar>(
    a: MatRef<C>,
    b: MatRef<C>,
    mut out: MatMut<C>,
) {
    for a_i in 0..a.nrows() {
        for a_j in 0..a.ncols() {
            // Safety: a_i and a_j are within bounds by definition.
            let a_e = unsafe { a.read_unchecked(a_i, a_j) };
            zipped!(
                out.as_mut().submatrix_mut(
                    a_i * b.nrows(),
                    a_j * b.nrows(),
                    b.nrows(),
                    b.nrows()
                ),
                b.as_ref()
            )
            .for_each(|unzipped!(mut out, b_e)| {
                out.write(a_e * b_e.read());
            });
        }
    }
}
