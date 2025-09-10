use qudit_core::ComplexScalar;

pub struct InstantiationResult<C: ComplexScalar> {
    /// The instantiated solution.
    pub params: Option<Vec<C::R>>,

    /// Optional Function Evaluation
    pub fun: Option<C::R>,

    /// Termination status:
    /// - Zero means successful termination.
    /// - One means input cannot be handled by instantiater; see message.
    /// - Two+ is instantiater specific; see relevant documentation.
    pub status: usize,

    /// Optional Message
    pub message: Option<String>,
}

impl<C: ComplexScalar> InstantiationResult<C> {
    /// Creates a new `InstantiationResult`.
    pub fn new(
        params: Option<Vec<C::R>>,
        fun: Option<C::R>,
        status: usize,
        message: Option<String>,
    ) -> Self {
        Self {
            params,
            fun,
            status,
            message,
        }
    }
}
