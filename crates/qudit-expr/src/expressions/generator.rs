use super::TensorExpression;

pub trait ExpressionGenerator {
    type ExpressionType: Into<TensorExpression>;

    fn generate_expression(&self) -> Self::ExpressionType;
}
