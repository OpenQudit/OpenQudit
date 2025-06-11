use qudit_core::ComplexScalar;

pub struct InstantiationResult<C: ComplexScalar> {
    /// The instantiated solution.
    params: Vec<C::R>,

    /// Optional Function Evaluation
    fun: Option<C::R>,

    /// Termination status:
    /// - Zero means successful termination.
    /// - One means input cannot be handled by instantiater; see message.
    /// - Two+ is instantiater specific; see relevant documentation.
    status: usize,

    /// Optional Message
    message: Option<String>,
}
