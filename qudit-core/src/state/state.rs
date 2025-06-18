use faer::col::AsColRef;

use crate::ComplexScalar;
use crate::RealScalar;
use crate::QuditRadices;
use crate::matrix::Col;
use crate::ToRadices;

pub trait ToColVector<C: ComplexScalar> {
    fn to_col_vector(self) -> Col<C>;
}

impl<C: ComplexScalar> ToColVector<C> for Col<C> {
    fn to_col_vector(self) -> Col<C> {
        self
    }
}

impl<C: ComplexScalar> ToColVector<C> for Vec<C> {
    fn to_col_vector(self) -> Col<C> {
        Col::from_fn(self.len(), |idx| self[idx])
    }
}

impl<'a, C: ComplexScalar> ToColVector<C> for &'a [C] {
    fn to_col_vector(self) -> Col<C> {
        Col::from_fn(self.len(), |idx| self[idx])
    }
}

/// A StateVector over a qudit system.
#[derive(Clone)]
pub struct StateVector<C: ComplexScalar> {
    radices: QuditRadices,
    vector: Col<C>,
}

impl<C: ComplexScalar> StateVector<C> {
    /// Create a new StateVector.
    ///
    /// # Arguments
    ///
    /// * `radices` - The radices of the qudit system.
    ///
    /// * `vector` - The vector to wrap, which can be any type that implements `ToColVector<C>`
    ///              (e.g., `Col<C>`, `Vec<C>`, `&[C]`).
    ///
    /// # Panics
    ///
    /// Panics if the vector is not a pure state.
    ///
    /// # Example
    ///
    /// ```
    /// use qudit_core::matrix::Col;
    /// use qudit_core::state::StateVector;
    /// use qudit_core::c64;
    /// let zero_state: StateVector<c64> = StateVector::new([2, 2], vec![
    ///     c64::one(), c64::zero(), c64::zero(), c64::zero()
    /// ]);
    /// ```
    ///
    /// # See Also
    ///
    #[inline(always)]
    #[track_caller]
    pub fn new<T: ToRadices, V: ToColVector<C>>(radices: T, vector: V) -> Self {
        let col_vector = vector.to_col_vector();
        assert!(Self::is_pure_state(&col_vector));
        Self {
            radices: radices.to_radices(),
            vector: col_vector,
        }
    }

    /// Check if a given vector represents a pure state.
    ///
    /// # Arguments
    ///
    /// * `vector` - The vector to check, implementing the `AsColRef` trait.
    ///
    /// # Returns
    ///
    /// A boolean indicating whether the vector is a pure state.
    ///
    /// # Example
    ///
    /// ```
    /// use qudit_core::matrix::Col;
    /// use qudit_core::state::StateVector;
    /// use qudit_core::c64;
    ///
    /// let vector = Col::from_fn(4, |x| if x == 0 { c64::one() } else { c64::zero() });
    /// let is_pure = StateVector::<c64>::is_pure_state(&vector);
    /// assert!(is_pure);
    /// ```
    ///
    /// # Note
    ///
    /// For a vector to represent a pure state, its norm must be exactly 1. This is verified by
    /// checking if the sum of the squares of the absolute values of its components equals 1
    /// within a defined threshold.
    pub fn is_pure_state<S: AsColRef<T = C>>(vector: S) -> bool {
        let col_ref = vector.as_col_ref();
        let norm: C::R = col_ref.iter().map(|c| c.abs().powi(2)).sum();
        (norm - C::R::new(1.0)).abs() < C::THRESHOLD
    }
}
