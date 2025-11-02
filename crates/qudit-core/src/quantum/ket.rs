use crate::ComplexScalar;
use crate::RealScalar;
use crate::Radices;
use crate::QuditSystem;
use faer::{Col, Row, Mat};
use std::ops::Index;
use num_traits::One;
use num_complex::ComplexFloat;

/// Represents a quantum state vector as a Ket.
pub struct Ket<R: RealScalar> {
    radices: Radices,
    vector: Col<R::C>,
}

impl<R: RealScalar> Ket<R> {
    /// Create a new Ket.
    ///
    /// # Arguments
    ///
    /// * `radices` - The radices of the qudit system.
    /// * `vector` - The vector to wrap.
    ///
    /// # Panics
    ///
    /// Panics if the vector is not a pure state.
    ///
    /// # Example
    ///
    /// ```
    /// use faer::Col;
    /// use qudit_core::Ket;
    /// use qudit_core::c64;
    /// let zero_state: Ket<f64> = Ket::new([2, 2], vec![
    ///     c64::ONE, c64::ZERO, c64::ZERO, c64::ZERO
    /// ]);
    /// ```
    ///
    /// # See Also
    ///
    #[inline(always)]
    #[track_caller]
    pub fn new<T: Into<Radices>, V: Into<Self>>(radices: T, vector: V) -> Self {
        let guessed_radices_ket = vector.into();
        let radices = radices.into();
        // assert!(Self::is_pure_state(&guessed_radices_ket));
        Self {
            radices,
            vector: guessed_radices_ket.vector,
        }
    }

    /// Create a zero state (all amplitudes zero).
    ///
    /// # Arguments
    ///
    /// * `radices` - The radices of the qudit system.
    ///
    /// # Example
    ///
    /// ```
    /// use qudit_core::Ket;
    /// let zero_state: Ket<f64> = Ket::zero([2, 2]);
    /// ```
    pub fn zero<T: Into<Radices>>(radices: T) -> Self {
        Self::basis(radices, 0)
    }

    /// Create a computational basis state |i⟩.
    ///
    /// # Arguments
    ///
    /// * `radices` - The radices of the qudit system.
    /// * `index` - The computational basis index.
    ///
    /// # Panics
    ///
    /// Panics if the index is out of bounds for the given radices.
    ///
    /// # Example
    ///
    /// ```
    /// use qudit_core::Ket;
    /// let state: Ket<f64> = Ket::basis([2, 2], 3); // |11⟩ state
    /// ```
    pub fn basis<T: Into<Radices>>(radices: T, index: usize) -> Self {
        let radices = radices.into();
        let dimension = radices.dimension();
        assert!(index < dimension, "Index {} out of bounds for dimension {}", index, dimension);
        
        let mut vector = Col::zeros(dimension);
        vector[index] = R::C::one();
        
        Self {
            radices,
            vector,
        }
    }

    /// Create a uniform superposition state (all amplitudes equal).
    ///
    /// # Arguments
    ///
    /// * `radices` - The radices of the qudit system.
    ///
    /// # Example
    ///
    /// ```
    /// use qudit_core::Ket;
    /// let uniform_state: Ket<f64> = Ket::uniform([2, 2]);
    /// ```
    pub fn uniform<T: Into<Radices>>(radices: T) -> Self {
        let radices = radices.into();
        let dimension = radices.dimension();
        let amplitude = R::C::from_real(R::one() / R::from(dimension).unwrap().sqrt());
        
        Self {
            radices,
            vector: Col::from_fn(dimension, |_| amplitude),
        }
    }

    /// Check if this is a valid normalized pure quantum state.
    ///
    /// A pure state must have norm² ≈ 1.0 (normalized).
    ///
    /// # Arguments
    ///
    /// * `tolerance` - Optional tolerance for normalization check (default: 1e-10)
    ///
    /// # Example
    ///
    /// ```
    /// use qudit_core::Ket;
    /// let basis_state: Ket<f64> = Ket::basis([2, 2], 0);
    /// assert!(basis_state.is_pure_state());
    /// ```
    pub fn is_pure_state(&self) -> bool {
        let norm_squared = self.vector.squared_norm_l2();
        R::ONE.is_close(norm_squared)
    }

    /// Get the distance from another quantum state.
    ///
    /// This computes 1 - |⟨ψ|φ⟩|² where |⟨ψ|φ⟩|² is the fidelity.
    /// Distance of 0 means identical states, distance of 1 means orthogonal states.
    ///
    /// # Arguments
    ///
    /// * `other` - The other quantum state to compare with
    ///
    /// # Panics
    ///
    /// Panics if the states have different dimensions.
    ///
    /// # Example
    ///
    /// ```
    /// use qudit_core::Ket;
    /// let state1: Ket<f64> = Ket::basis([2], 0);
    /// let state2: Ket<f64> = Ket::basis([2], 1);
    /// let distance = state1.get_distance_from(&state2);
    /// assert!((distance - 1.0).abs() < 1e-10); // Orthogonal states
    /// ```
    pub fn get_distance_from(&self, other: &Self) -> R {
        assert_eq!(self.vector.nrows(), other.vector.nrows(), 
                   "Cannot compute distance between states of different dimensions");
        
        let inner_product = self.inner_product(other);
        let fidelity = inner_product.norm_squared();
        R::one() - fidelity
    }

    // TODO: should not have explicit operation for inner product
    // rather: have a dagger that takes Ket -> Bra, and allow multiplication
    /// Compute the inner product ⟨self|other⟩.
    fn inner_product(&self, other: &Self) -> R::C {
        self.vector.iter()
            .zip(other.vector.iter())
            .map(|(&a, &b)| a.conj() * b)
            .sum()
    }

    /// Get the probabilities for all computational basis states.
    ///
    /// Returns a vector where each element is |amplitude|² for the corresponding basis state.
    ///
    /// # Example
    ///
    /// ```
    /// use qudit_core::Ket;
    /// let uniform_state: Ket<f64> = Ket::uniform([2, 2]);
    /// let probs = uniform_state.probabilities();
    /// // Each probability should be 0.25 for uniform superposition
    /// ```
    pub fn probabilities(&self) -> Vec<R> {
        self.vector.iter().map(|&z| z.norm_squared()).collect()
    }

    /// Get the probability of measuring the state in a specific computational basis state.
    ///
    /// # Arguments
    ///
    /// * `index` - The computational basis index
    ///
    /// # Panics
    ///
    /// Panics if the index is out of bounds.
    ///
    /// # Example
    ///
    /// ```
    /// use qudit_core::Ket;
    /// let basis_state: Ket<f64> = Ket::basis([2, 2], 1);
    /// assert!((basis_state.probability_at(1) - 1.0).abs() < 1e-10);
    /// assert!(basis_state.probability_at(0).abs() < 1e-10);
    /// ```
    pub fn probability_at(&self, index: usize) -> R {
        assert!(index < self.vector.nrows(), "Index {} out of bounds", index);
        self.vector[index].norm_squared()
    }

    /// Compute the tensor product with another quantum state.
    ///
    /// Creates a new state |ψ⟩ ⊗ |φ⟩ representing a composite quantum system.
    ///
    /// # Arguments
    ///
    /// * `other` - The other quantum state to tensor with
    ///
    /// # Example
    ///
    /// ```
    /// use qudit_core::Ket;
    /// let qubit1: Ket<f64> = Ket::basis([2], 0); // |0⟩
    /// let qubit2: Ket<f64> = Ket::basis([2], 1); // |1⟩
    /// let combined = qubit1.tensor_product(&qubit2); // |01⟩
    /// ```
    pub fn tensor_product(&self, other: &Self) -> Self {
        // Combine the radices
        let new_radices = self.radices.concat(&other.radices);
        
        // Compute Kronecker product of the vectors
        let self_dim = self.vector.nrows();
        let other_dim = other.vector.nrows();
        let new_dim = self_dim * other_dim;
        
        let new_vector = Col::from_fn(new_dim, |idx| {
            let i = idx / other_dim;
            let j = idx % other_dim;
            self.vector[i] * other.vector[j]
        });
        
        Self {
            radices: new_radices,
            vector: new_vector,
        }
    }
}

impl<R: RealScalar> QuditSystem for Ket<R> {
    fn radices(&self) -> Radices {
        self.radices.clone()
    }

    fn dimension(&self) -> usize {
        self.radices.dimension()
    }
}

// Index trait implementation for accessing amplitudes
impl<R: RealScalar> Index<usize> for Ket<R> {
    type Output = R::C;
    
    fn index(&self, index: usize) -> &Self::Output {
        &self.vector[index]
    }
}

// TODO:
// impl TryFrom<Tensor<R, 3>> for Ket<R>
// impl TryFrom<Tensor<C, 3>> for Ket<C::R>

impl<R: RealScalar, T: Into<R::C> + Copy> TryFrom<Mat<T>> for Ket<R> {
    type Error = String;
    
    fn try_from(value: Mat<T>) -> Result<Self, Self::Error> {
        if value.ncols() != 1 {
            return Err(format!(
                "Matrix has {} columns, expected 1 column for conversion to Ket", 
                value.ncols()
            ));
        }

        // TODO: If T == R::C then take ptr and manually drop value for zero-alloc
        
        let vec: Vec<R::C> = value.col(0).iter().copied().map(|x| x.into()).collect();
        Ok(vec.into())
    }
}

impl<C: ComplexScalar> From<Col<C>> for Ket<C::R> {
    fn from(value: Col<C>) -> Self {
        let radices = Radices::guess(value.nrows());
        Self {
            radices,
            vector: value
        }
    }
}

impl<R: RealScalar, T: Into<R::C> + Copy> From<Row<T>> for Ket<R> {
    fn from(value: Row<T>) -> Self {
        // TODO: If T == R::C then take ptr and manually drop value for zero-alloc
        // be extra safe here about strides and all that
        let vec: Vec<R::C> = value.iter().copied().map(|x| x.into()).collect();
        vec.into()
    }
}

impl<R: RealScalar, T: Into<R::C> + Copy> From<Vec<T>> for Ket<R> {
    fn from(value: Vec<T>) -> Self {
        let radices = Radices::guess(value.len());
        Self {
            radices,
            vector: Col::from_fn(value.len(), |i| value[i].into())
        }
    }
}

impl<R: RealScalar, T: Into<R::C>> FromIterator<T> for Ket<R> {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        let vec: Vec<R::C> = iter.into_iter().map(|x| x.into()).collect();
        vec.into()
    }
}

impl<R: RealScalar, T: Into<R::C> + Copy> From<&[T]> for Ket<R> {
    fn from(value: &[T]) -> Self {
        let radices = Radices::guess(value.len());
        Self {
            radices,
            vector: Col::from_fn(value.len(), |i| value[i].into())
        }
    }
}

impl<R: RealScalar, T: Into<R::C> + Copy, const N: usize> From<&[T; N]> for Ket<R> {
    fn from(value: &[T; N]) -> Self {
        let radices = Radices::guess(N);
        Self {
            radices,
            vector: Col::from_fn(N, |i| value[i].into())
        }
    }
}

impl<R: RealScalar, T: Into<R::C> + Copy, const N: usize> From<[T; N]> for Ket<R> {
    fn from(value: [T; N]) -> Self {
        let radices = Radices::guess(N);
        Self {
            radices,
            vector: Col::from_fn(N, |i| value[i].into())
        }
    }
}

