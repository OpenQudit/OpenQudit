use proc_macro2::{TokenStream, TokenTree, Delimiter, Span};
use syn::{Result, Error};
use syn::parse::{Parse, ParseStream};
use quote::quote;

// ###########################################################################################################

#[derive(PartialEq, Debug, Clone)]
enum TensorTokens {
    OpenBracket,
    ClosedBracket,
    
    OpenParenthesis,
    ClosedParenthesis,
    
    Comma,
    J,

    Plus,
    Minus,

    Number(f64),

    // Digit(u8),
    // Decimal
}

#[derive(PartialEq, Debug, Clone)]
struct ComplexElement {
    real: f64,
    imag: f64,
}

// struct ComplexTensor {
//     data: Vec<ComplexElement>,
//     shape: Vec<i32>,
//     strides: Vec<isize>
// }

#[derive(PartialEq, Debug, Clone)]
enum RecursiveTensor {
    Scalar(ComplexElement),
    SubTensor(Vec<RecursiveTensor>)
}

// ###########################################################################################################

fn tensor_lexer(token_stream: TokenStream) -> Result<Vec<TensorTokens>> {
    let mut processed_tokens = Vec::new();
    let mut token_iterator = token_stream.into_iter();
    let mut string_accumulator: String;

    while let Some(token) = token_iterator.next() {
        match token {
            
            TokenTree::Literal(literal) => {
                string_accumulator = literal.to_string();
                if let Some(literal_without_j) = string_accumulator.strip_suffix('j') {
                    match literal_without_j.parse() {
                        Ok(float) => processed_tokens.push(TensorTokens::Number(float)),
                        Err(_) => return Err(Error::new(literal.span(), "Literal is not a number"))
                    }
                    processed_tokens.push(TensorTokens::J);
                } else {
                    match string_accumulator.parse() {
                        Ok(float) => processed_tokens.push(TensorTokens::Number(float)),
                        Err(_) => return Err(Error::new(literal.span(), "Literal is not a number"))
                    }
                }
            }

            TokenTree::Ident(identifier) => match identifier.to_string().as_str() {
                "j" => processed_tokens.push(TensorTokens::J),
                _ => return Err(Error::new(identifier.span(), "Identifier is not j"))
            }

            TokenTree::Punct(punctuation) => match punctuation.as_char() {
                '+' => processed_tokens.push(TensorTokens::Plus),
                '-' => processed_tokens.push(TensorTokens::Minus),
                ',' => processed_tokens.push(TensorTokens::Comma),
                _ => return Err(Error::new(punctuation.span(), "Unexpected punctuation"))
            }

            TokenTree::Group(group) => match group.delimiter() {
                
                Delimiter::Bracket => {
                    processed_tokens.push(TensorTokens::OpenBracket);
                    processed_tokens.extend(tensor_lexer(group.stream())?);
                    processed_tokens.push(TensorTokens::ClosedBracket);
                },
                
                Delimiter::Parenthesis => {
                    processed_tokens.push(TensorTokens::OpenParenthesis);
                    processed_tokens.extend(tensor_lexer(group.stream())?);
                    processed_tokens.push(TensorTokens::ClosedParenthesis);
                }, 
                
                _ => return Err(Error::new(group.span(), "Unexpected brackets"))
            }

        }
    }
    return Ok(processed_tokens);
}

fn complex_number_parser(tokens: Vec<TensorTokens>) -> Result<(ComplexElement, usize)> {
    let mut real = 0.0;
    let mut imag = 0.0;
    let mut sign = 1.0;
    let mut delta_index = 0;

    loop {

        // The token might start with a sign
        match tokens.get(delta_index) {
            Some(TensorTokens::Plus) => {
                sign = 1.0;
                delta_index += 1;
            }
            
            Some(TensorTokens::Minus) => {
                sign = -1.0;
                delta_index += 1;
            }

            _ => {}
        }

        match tokens.get(delta_index) {
            Some(TensorTokens::Number(val)) => {
                // Checks if the number is imaginary
                if let Some(TensorTokens::J) = tokens.get(delta_index + 1) {
                    imag += sign * val;
                    delta_index += 2;
                } else {
                    real += sign * val;
                    delta_index += 1;
                }
            }
            // Edge case: `± j` = `± 1.0 j`.
            Some(TensorTokens::J) => {
                imag += sign * 1.0;
                delta_index += 1;
            }

            _ => return Err(Error::new(Span::call_site(), "Expected a number or j",))
        }

        // Continue parsing if we see an operator.
        match tokens.get(delta_index) {
            Some(TensorTokens::Plus) => {
                sign = 1.0;
                delta_index += 1;
            }
            Some(TensorTokens::Minus) => {
                sign = -1.0;
                delta_index += 1;
            }
            _ => break,
        }

    }

    return Ok((ComplexElement{real, imag}, delta_index));
}

impl Parse for ComplexElement {
    fn parse(input: ParseStream) -> Result<Self> {
        let token_stream = input.parse::<TokenStream>()?;
        let tokens = tensor_lexer(token_stream)?;
        let (element, tokens_consumed) = complex_number_parser(tokens.clone())?;

        if tokens_consumed < tokens.len() {
            return Err(Error::new(Span::call_site(), "Not a valid complex number"));
        }
        
        Ok(element)
    }
}

fn tensor_parser(tokens: &[TensorTokens]) -> Result<(RecursiveTensor, usize)> {
    let mut index = 0;

    // A tensor starts with [ or is a scalar.
    if let Some(TensorTokens::OpenBracket) = tokens.get(index) {
        index += 1;

        // Each recursive step is adding one dimension.
        let mut children = Vec::new();
        loop {
            if index > tokens.len() {
                return Err(Error::new(Span::call_site(), "Missing closing bracket"));
            }

            if tokens.get(index) == Some(&TensorTokens::ClosedBracket) {
                index += 1;
                return Ok((RecursiveTensor::SubTensor(children), index));
            }

            let (child, delta_index) = tensor_parser(&tokens[index..])?;
            children.push(child);
            index += delta_index;

            if let Some(TensorTokens::Comma) = tokens.get(index) {
                index += 1;
            }
        }
    } else {
        let (scalar, delta_index) = complex_number_parser(tokens.to_vec())?;
        return Ok((RecursiveTensor::Scalar(scalar), delta_index));
    }
}

// Helps us fit data into ComplexTensor. Returns elements of the RecursiveTensor and its shape.
fn flatten_tensor_data(input: &RecursiveTensor) -> (Vec<ComplexElement>, Vec<usize>) {
    match input {

        RecursiveTensor::Scalar(s) => (vec![s.clone()], vec![]),

        RecursiveTensor::SubTensor(subtensors) => {

            // Flattened data calculation
            let mut flat_data = Vec::new();
            for subtensor in subtensors {
                let (mut subtensor_data, _) = flatten_tensor_data(subtensor);
                flat_data.append(&mut subtensor_data);
            }

            // Shape calculation
            let (_, sub_shape) = flatten_tensor_data(&subtensors[0]);
            let mut final_shape = vec![subtensors.len()];
            final_shape.extend(sub_shape);

            return (flat_data, final_shape);
        }
    }
}

#[proc_macro]
pub fn complex_tensor(input: proc_macro::TokenStream) -> proc_macro::TokenStream {

    let tokens = match tensor_lexer(input.into()) {
        Ok(inner_val) => inner_val,
        Err(error) => return error.to_compile_error().into(),
    };

    let (recursive_tensor, _) = match tensor_parser(&tokens) {
        Ok(inner_val) => inner_val,
        Err(error) => return error.to_compile_error().into(),
    };

    let (flat_data, shape) = flatten_tensor_data(&recursive_tensor);

    let quoted_data = flat_data.iter().map(|c| {
        let real = c.real;
        let imag = c.imag;
        quote!{faer::c64::new(#real, #imag)}
    });
    
    let quoted_shape = quote!{[#(#shape),*]};

    let d = shape.len();
    quote!{{
            let data_vec: Vec<faer::c64> = vec![#(#quoted_data),*];
            Tensor::<faer::c64, #d>::from_slice(&data_vec, #quoted_shape)
    }}.into()
}
