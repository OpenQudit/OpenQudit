use crate::{
    math::{ComplexScalar, Function, Matrix, Vector},
    QuditCircuit,
};

pub trait CostFunction<C: ComplexScalar>: Function {
    fn get_cost(&self, params: &[C::Re]) -> C::Re;
}

pub trait DifferentiableCostFunction<C: ComplexScalar>:
    CostFunction<C>
{
    fn get_gradient(&self, params: &[C::Re]) -> Vector<C::Re>;
    fn get_cost_and_gradient(&self, params: &[C::Re])
        -> (C::Re, Vector<C::Re>);
}

pub trait DoublyDifferentiableCostFunction<C: ComplexScalar>:
    DifferentiableCostFunction<C>
{
    fn get_hessian(&self, params: &[C::Re]) -> Matrix<C::Re>;
    fn get_cost_and_hessian(&self, params: &[C::Re]) -> (C::Re, Matrix<C::Re>);
    fn get_gradient_and_hessian(
        &self,
        params: &[C::Re],
    ) -> (Vector<C::Re>, Matrix<C::Re>);
    fn get_cost_gradient_and_hessian(
        &self,
        params: &[C::Re],
    ) -> (C::Re, Vector<C::Re>, Matrix<C::Re>);
}

pub enum BoxedCostFunction<C: ComplexScalar> {
    CostFunction(Box<dyn CostFunction<C>>),
    DifferentiableCostFunction(Box<dyn DifferentiableCostFunction<C>>),
    DoublyDifferentiableCostFunction(
        Box<dyn DoublyDifferentiableCostFunction<C>>,
    ),
}

impl<C: ComplexScalar> Function for BoxedCostFunction<C> {
    fn get_num_params(&self) -> usize {
        match self {
            Self::CostFunction(cost) => cost.get_num_params(),
            Self::DifferentiableCostFunction(cost) => cost.get_num_params(),
            Self::DoublyDifferentiableCostFunction(cost) => {
                cost.get_num_params()
            },
        }
    }
}

impl<C: ComplexScalar> CostFunction<C> for BoxedCostFunction<C> {
    fn get_cost(&self, params: &[C::Re]) -> C::Re {
        match self {
            Self::CostFunction(cost) => cost.get_cost(params),
            Self::DifferentiableCostFunction(cost) => cost.get_cost(params),
            Self::DoublyDifferentiableCostFunction(cost) => {
                cost.get_cost(params)
            },
        }
    }
}

impl<C: ComplexScalar> DifferentiableCostFunction<C> for BoxedCostFunction<C> {
    fn get_gradient(&self, params: &[C::Re]) -> Vector<C::Re> {
        match self {
            Self::CostFunction(cost) => {
                panic!("Cost function is not differentiable")
            },
            Self::DifferentiableCostFunction(cost) => cost.get_gradient(params),
            Self::DoublyDifferentiableCostFunction(cost) => {
                cost.get_gradient(params)
            },
        }
    }

    fn get_cost_and_gradient(
        &self,
        params: &[C::Re],
    ) -> (C::Re, Vector<C::Re>) {
        match self {
            Self::CostFunction(cost) => {
                panic!("Cost function is not differentiable")
            },
            Self::DifferentiableCostFunction(cost) => {
                cost.get_cost_and_gradient(params)
            },
            Self::DoublyDifferentiableCostFunction(cost) => {
                cost.get_cost_and_gradient(params)
            },
        }
    }
}

impl<C: ComplexScalar> DoublyDifferentiableCostFunction<C>
    for BoxedCostFunction<C>
{
    fn get_hessian(&self, params: &[C::Re]) -> Matrix<C::Re> {
        match self {
            Self::CostFunction(cost) => {
                panic!("Cost function is not doubly differentiable")
            },
            Self::DifferentiableCostFunction(cost) => {
                panic!("Cost function is not doubly differentiable")
            },
            Self::DoublyDifferentiableCostFunction(cost) => {
                cost.get_hessian(params)
            },
        }
    }

    fn get_cost_and_hessian(&self, params: &[C::Re]) -> (C::Re, Matrix<C::Re>) {
        match self {
            Self::CostFunction(cost) => {
                panic!("Cost function is not doubly differentiable")
            },
            Self::DifferentiableCostFunction(cost) => {
                panic!("Cost function is not doubly differentiable")
            },
            Self::DoublyDifferentiableCostFunction(cost) => {
                cost.get_cost_and_hessian(params)
            },
        }
    }

    fn get_gradient_and_hessian(
        &self,
        params: &[C::Re],
    ) -> (Vector<C::Re>, Matrix<C::Re>) {
        match self {
            Self::CostFunction(cost) => {
                panic!("Cost function is not doubly differentiable")
            },
            Self::DifferentiableCostFunction(cost) => {
                panic!("Cost function is not doubly differentiable")
            },
            Self::DoublyDifferentiableCostFunction(cost) => {
                cost.get_gradient_and_hessian(params)
            },
        }
    }

    fn get_cost_gradient_and_hessian(
        &self,
        params: &[C::Re],
    ) -> (C::Re, Vector<C::Re>, Matrix<C::Re>) {
        match self {
            Self::CostFunction(cost) => {
                panic!("Cost function is not doubly differentiable")
            },
            Self::DifferentiableCostFunction(cost) => {
                panic!("Cost function is not doubly differentiable")
            },
            Self::DoublyDifferentiableCostFunction(cost) => {
                cost.get_cost_gradient_and_hessian(params)
            },
        }
    }
}

pub trait CostFunctionGenerator<C: ComplexScalar> {
    fn gen_cost(&self, circuit: &QuditCircuit<C>) -> BoxedCostFunction<C>;
}
