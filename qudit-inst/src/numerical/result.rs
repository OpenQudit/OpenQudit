use qudit_core::RealScalar;

pub struct MinimizationResult<R: RealScalar> {
    pub params: Vec<R>,
    pub fun: R,
    pub status: usize,
    pub message: Option<String>,
}
