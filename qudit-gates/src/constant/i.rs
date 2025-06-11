use qudit_expr::{TensorExpression, TensorExpressionGenerator};

/// The identity or no-op gate.
#[derive(Hash, PartialEq, Eq, Clone, Debug)]
pub struct IGate {
    pub radix: usize,
}

impl IGate {
    pub fn new(radix: usize) -> Self {
        Self { radix }
    }
}

impl TensorExpressionGenerator for IGate {
    fn gen_expr(&self) -> TensorExpression {
        let proto = format!("utry I<{}>()", self.radix);
        let mut body = "".to_string();
        body += "[";
        for i in 0..self.radix {
            body += "[";
            for j in 0..self.radix {
                if i == j {
                    body += "1,";
                } else {
                    body += "0,";
                }
            }
            body += "],";
        }
        body += "]";

        TensorExpression::new(proto + "{" + &body + "}")
    }
}
