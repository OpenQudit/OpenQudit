use super::passes::*;
use super::expr::TypedUnitaryDefinition;
use super::parser;

#[derive(Debug, Clone, PartialEq)]
pub struct CompilerError {
    pub error: String,
}

impl CompilerError {
    pub fn new(error: &str) -> Self {
        CompilerError { error: String::from(error), }
    }
}

pub type CompilerResult<T> = Result<T, CompilerError>;

pub fn compile(input: String) -> CompilerResult<TypedUnitaryDefinition> {
    let mut parser = parser::Parser::new(input);
    let udef_result = parser.parse();

    let udef = match udef_result {
        Ok(udef) => udef,
        Err(e) => return Err(CompilerError::new(&e.error)),
    };

    if udef.len() != 1 {
        let msg = format!("Expected exactly one unitary definition, found {}", udef.len());
        return Err(CompilerError::new(&msg));
    }
    let udef = udef.into_iter().next().unwrap();

    let udef = etoi_to_cis(udef);
    let udef = type_expr(udef);

    let dim = udef.body.dimension();
    assert!(dim != 0, "Unitary definition must have a non-zero dimension");

    let udef = order_real_then_complex(udef);
    let udef = make_element_wise(udef);
    let udef = simplify_powers_of_cis(udef);

    Ok(udef)
}

pub fn optimize(udef: TypedUnitaryDefinition) -> TypedUnitaryDefinition {
    let udef = handle_zeros_and_ones(udef);
    udef
}
