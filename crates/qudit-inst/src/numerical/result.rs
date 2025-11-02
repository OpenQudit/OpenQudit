use qudit_core::RealScalar;

use crate::InstantiationResult;

pub struct MinimizationResult<R: RealScalar> {
    pub params: Vec<R>,
    pub fun: R,
    pub status: usize,
    pub message: Option<String>,
}

impl<R: RealScalar> MinimizationResult<R> {
    pub fn simple_success(params: Vec<R>, fun: R) -> Self {
        Self {
            params,
            fun,
            status: 0,
            message: None,
        }
    }

    pub fn to_instantiation(self) -> InstantiationResult<R::C> {
        let Self { params, fun, status, message } = self;
        InstantiationResult::new(Some(params), Some(fun), status, message)
    }
}
