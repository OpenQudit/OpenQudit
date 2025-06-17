use crate::ComplexScalar;
use crate::QuditRadices;
use crate::matrix::Col;
use crate::ToRadices;

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
    /// * `vector` - The vector to wrap.
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
    /// let unitary: StateVector<c64> = StateVector::new([2, 2], Col::from_fn(4, (
    ///     |x| if x == 0 {
    ///         c64::one()
    ///     } else {
    ///         c64::zero()
    ///     }));
    /// ```
    ///
    /// # See Also
    ///
    #[inline(always)]
    #[track_caller]
    pub fn new<T: ToRadices>(radices: T, vector: Col<C>) -> Self {
        assert!(Self::is_pure_state(&vector));
        Self {
            radices: radices.to_radices(),
            vector,
        }
    }
}
