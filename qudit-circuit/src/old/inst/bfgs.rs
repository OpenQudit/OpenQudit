use crate::math::ComplexScalar;

use super::{cost::BoxedCostFunction, optimizer::Optimizer};

pub struct BFGSInstantiater {
    maximum_iterations: Option<usize>,
    parameter_tolerance: f64,
    function_tolerance: f64,
    gradient_tolerance: f64,
    // starting_point_gen: Box<dyn StartingPointGenerator<C>>,
}

impl BFGSInstantiater {
    pub fn new() -> Self {
        Self {
            maximum_iterations: None,
            parameter_tolerance: 0.0,
            function_tolerance: 0.0,
            gradient_tolerance: 1e-8,
        }
    }

    pub fn maximum_iterations(mut self, maximum_iterations: usize) -> Self {
        self.maximum_iterations = Some(maximum_iterations);
        self
    }

    pub fn parameter_tolerance(mut self, parameter_tolerance: f64) -> Self {
        self.parameter_tolerance = parameter_tolerance;
        self
    }

    pub fn function_tolerance(mut self, function_tolerance: f64) -> Self {
        self.function_tolerance = function_tolerance;
        self
    }

    pub fn gradient_tolerance(mut self, gradient_tolerance: f64) -> Self {
        self.gradient_tolerance = gradient_tolerance;
        self
    }
}

impl<C: ComplexScalar> Optimizer<C> for BFGSInstantiater {
    fn optimize(
        &self,
        cost: &BoxedCostFunction<C>,
        x0: Vec<C::Re>,
    ) -> Vec<C::Re> {
        todo!()
    }
}
