/// A data structure consisting of or operating on a specified number of finite-dimensional qudits.
///
/// This trait assumes that each qudit in the system has a well-defined radix, the number of basis
/// states or dimensions, and these radices are represented collectively by the [crate::QuditRadices]
/// object. Mixed-radix or heterogeneous quantum systems where different qudits may have differing
/// dimensions are allowed.
pub trait QuditSystem {
    /// Returns the radices of the qudits in the system.
    ///
    /// # Returns
    /// A `[QuditRadices]` instance representing the radices of the qudits.
    fn radices(&self) -> crate::QuditRadices;

    /// Returns the number of qudits in the system.
    ///
    /// # Returns
    /// A `usize` representing the number of qudits.
    #[inline(always)]
    fn num_qudits(&self) -> usize {
        self.radices().num_qudits()
    }

    /// Returns the total dimension of the quantum system.
    ///
    /// # Returns
    /// A `usize` representing the product of the radices of all qudits.
    #[inline(always)]
    fn dimension(&self) -> usize {
        self.radices().dimension()
    }

    /// Checks if the system consists only of qubits (2-level systems).
    ///
    /// # Returns
    /// `true` if all qudits are qubits, otherwise `false`.
    #[inline(always)]
    fn is_qubit_only(&self) -> bool {
        self.radices().is_qubit_only()
    }

    /// Checks if the system consists only of qutrits (3-level systems).
    ///
    /// # Returns
    /// `true` if all qudits are qutrits, otherwise `false`.
    #[inline(always)]
    fn is_qutrit_only(&self) -> bool {
        self.radices().is_qutrit_only()
    }

    /// Checks if the system consists only of qudits with a specified radix.
    ///
    /// # Parameters
    /// - `radix`: The radix to check against all qudits in the system.
    ///
    /// # Returns
    /// `true` if all qudits have the specified radix, otherwise `false`.
    #[inline(always)]
    fn is_qudit_only<T: crate::radix::ToRadix>(&self, radix: T) -> bool {
        self.radices().is_qudit_only(radix)
    }

    /// Checks if the system is homogenous, i.e., all qudits have the same radix.
    ///
    /// # Returns
    /// `true` if the system is homogenous, otherwise `false`.
    #[inline(always)]
    fn is_homogenous(&self) -> bool {
        self.radices().is_homogenous()
    }
}

/// A data structure consisting of or operating on a specified number of dits.
pub trait ClassicalSystem {
    /// Returns the radices of the dits in the system.
    ///
    /// # Returns
    /// A `[QuditRadices]` instance representing the radices of the dits.
    fn radices(&self) -> crate::QuditRadices;

    /// Returns the number of classical bits in the system.
    ///
    /// # Returns
    /// A `usize` representing the number of classical bits.
    fn num_dits(&self) -> usize {
        self.radices().len()
    }

    /// Returns the total dimension of the classical system.
    ///
    /// # Returns
    /// A `usize` representing the product of the radices of all dits.
    #[inline(always)]
    fn dimension(&self) -> usize {
        self.radices().dimension()
    }

    /// Checks if the system consists only of bits (2-level systems).
    ///
    /// # Returns
    /// `true` if all dits are bits, otherwise `false`.
    #[inline(always)]
    fn is_bit_only(&self) -> bool {
        self.radices().is_qubit_only() // TODO: seems hacky?
    }

    /// Checks if the system consists only of trits (3-level systems).
    ///
    /// # Returns
    /// `true` if all dits are trits, otherwise `false`.
    #[inline(always)]
    fn is_trit_only(&self) -> bool {
        self.radices().is_qutrit_only()
    }

    /// Checks if the system consists only of dits with a specified radix.
    ///
    /// # Parameters
    /// - `radix`: The radix to check against all dits in the system.
    ///
    /// # Returns
    /// `true` if all dits have the specified radix, otherwise `false`.
    #[inline(always)]
    fn is_dit_only<T: crate::radix::ToRadix>(&self, radix: T) -> bool {
        self.radices().is_qudit_only(radix)
    }

    /// Checks if the system is homogenous, i.e., all dits have the same radix.
    ///
    /// # Returns
    /// `true` if the system is homogenous, otherwise `false`.
    #[inline(always)]
    fn is_homogenous(&self) -> bool {
        self.radices().is_homogenous()
    }
}

/// A data structure that consists of or operates on a hybrid quantum-classical system.
pub trait HybridSystem: QuditSystem + ClassicalSystem { 
    /// Returns the radices of the qudits in the system.
    ///
    /// # Returns
    /// A `[QuditRadices]` instance representing the radices of the qudits.
    #[inline(always)]
    fn qudit_radices(&self) -> crate::QuditRadices {
        QuditSystem::radices(self)
    }

    /// Returns the radices of the dits in the system.
    ///
    /// # Returns
    /// A `[QuditRadices]` instance representing the radices of the dits.
    fn dit_radices(&self) -> crate::QuditRadices {
        ClassicalSystem::radices(self)
    }

    /// Checks if the system is purely quantum with no classical bits.
    ///
    /// # Returns
    /// `true` if there are no classical bits, otherwise `false`.
    #[inline(always)]
    fn is_pure_quantum(&self) -> bool {
        self.num_dits() == 0
    }

    /// Checks if the system is purely classical with no qudits.
    ///
    /// # Returns
    /// `true` if there are no qudits, otherwise `false`.
    #[inline(always)]
    fn is_classical_only(&self) -> bool {
        self.num_qudits() == 0
    }

    /// Checks if the system is hybrid with both qudits and classical bits.
    ///
    /// # Returns
    /// `true` if there are both qudits and classical bits, otherwise `false`.
    #[inline(always)]
    fn is_hybrid(&self) -> bool {
        !self.is_pure_quantum() && !self.is_classical_only()
    }
}
