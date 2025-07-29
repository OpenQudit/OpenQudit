use proc_macro::TokenStream;
use proc_macro2::{TokenStream as TokenStream2, TokenTree, Delimiter, Span, Spacing, Punct, Group};
use syn::{Result, Error};
use quote::quote;
use faer_traits::ComplexField;
use num_complex::ComplexFloat;
// use coe::is_same;

trait RealNumber {
    
}

trait ComplexNumber: ComplexField + ComplexFloat {
    type Real: RealNumber;
}

impl RealNumber for f64 {

}

impl RealNumber for f32 {
    
}

impl ComplexNumber for faer::c64 {
    type Real = f64;
}

impl ComplexNumber for faer::c32 {
    type Real = f32;
}

#[derive(Debug, Clone)]
enum TensorTokens {
    OpenBracket,
    ClosedBracket,
    
    //OpenParenthesis,
    //ClosedParenthesis,
    
    Comma,
    //J,

    // Plus,
    // Minus,
    // Multiply,
    // Divide,

    Number(Vec<TokenTree>),
}

#[derive(Debug, Clone)]
enum RecursiveTensor {
    Scalar(Vec<TokenTree>),
    SubTensor(Vec<RecursiveTensor>)
}

//`4j` -> `4.0 * j`
// `f(x)j` -> `f(x) * j`
fn j_processing32(input: TokenStream2) -> TokenStream2 {
    let tokens: Vec<TokenTree> = input.into_iter().collect();
    let mut new_stream = Vec::<TokenTree>::new();
    let mut stream_accumulator: Vec<TokenTree> ;

    let mut token: &TokenTree;
    let mut lit_str: String;
    let mut pass: bool;
    for index in 0..tokens.len() {
        stream_accumulator = Vec::<TokenTree>::new();
        pass = false;
        token = &tokens[index];

        // Replaces `j` with faer::c64::new(0.0, 1.0).
        match &token {

            TokenTree::Literal(literal) => {
                lit_str = literal.to_string();
                if let Some(num_part) = lit_str.strip_suffix('j') {
                    if let Ok(number_val) = num_part.parse::<f32>() {
                        stream_accumulator.extend(quote!{#number_val * faer::c32::new(0.0, 1.0)});
                        pass = true;
                    } else {
                        panic!("Not a valid number")
                    }
                } else if let Some(num_part) = lit_str.strip_prefix('j') {
                    if let Ok(number_val) = num_part.parse::<f32>() {
                        stream_accumulator.extend(quote!{#number_val * faer::c32::new(0.0, 1.0)});
                        pass = true;
                    } else {
                        panic!("Not a valid number")
                    }
                } 
            }

            TokenTree::Ident(identifier) => {
                if identifier.to_string().as_str() == "j" {
                    stream_accumulator.extend(quote!{faer::c32::new(0.0, 1.0)});
                    pass = true;
                } 
            }

            TokenTree::Group(group) => {
                let processed_inner_stream = j_processing32(group.stream());
                let new_group = Group::new(group.delimiter(), processed_inner_stream);
                new_stream.push(TokenTree::Group(new_group));
                continue;
            }

            _ => ()
        }

        if !pass {
            new_stream.push(token.clone());
            continue;
        }

        // Makes implicit multiplication explicit
        pass = false;
        if index > 0 {
            if let TokenTree::Punct(punct) = &tokens[index - 1] {
                match punct.as_char() {
                    ']' | ',' | '+' | '-' | '*' | '/' => pass = true,
                    _ => ()
                }
            }
            if !pass {
                new_stream.push(TokenTree::Punct(Punct::new('*', Spacing::Alone)));
            }
        }
        
        new_stream.extend(stream_accumulator);
        
        pass = false;
        if index < tokens.len() - 1 {
            if let TokenTree::Punct(punct) = &tokens[index + 1] {
                match punct.as_char() {
                    ']' | ',' | '+' | '-' | '*' | '/' => pass = true,
                    _ => ()
                }
            }
            if !pass {
                new_stream.push(TokenTree::Punct(Punct::new('*', Spacing::Alone)));
            }
        }
        
    }
    return TokenStream2::from_iter(new_stream);
}

//`4j` -> `4.0 * j`
// `f(x)j` -> `f(x) * j`
fn j_processing64(input: TokenStream2) -> TokenStream2 {
    let tokens: Vec<TokenTree> = input.into_iter().collect();
    let mut new_stream = Vec::<TokenTree>::new();
    let mut stream_accumulator: Vec<TokenTree> ;

    let mut token: &TokenTree;
    let mut lit_str: String;
    let mut pass: bool;
    for index in 0..tokens.len() {
        stream_accumulator = Vec::<TokenTree>::new();
        pass = false;
        token = &tokens[index];

        // Replaces `j` with faer::c64::new(0.0, 1.0).
        match &token {

            TokenTree::Literal(literal) => {
                lit_str = literal.to_string();
                if let Some(num_part) = lit_str.strip_suffix('j') {
                    if let Ok(number_val) = num_part.parse::<f64>() {
                        stream_accumulator.extend(quote!{#number_val * faer::c64::new(0.0, 1.0)});
                        pass = true;
                    } else {
                        panic!("Not a valid number")
                    }
                } else if let Some(num_part) = lit_str.strip_prefix('j') {
                    if let Ok(number_val) = num_part.parse::<f64>() {
                        stream_accumulator.extend(quote!{#number_val * faer::c64::new(0.0, 1.0)});
                        pass = true;
                    } else {
                        panic!("Not a valid number")
                    }
                } 
            }

            TokenTree::Ident(identifier) => {
                if identifier.to_string().as_str() == "j" {
                    stream_accumulator.extend(quote!{faer::c64::new(0.0, 1.0)});
                    pass = true;
                } 
            }

            TokenTree::Group(group) => {
                let processed_inner_stream = j_processing64(group.stream());
                let new_group = Group::new(group.delimiter(), processed_inner_stream);
                new_stream.push(TokenTree::Group(new_group));
                continue;
            }

            _ => ()
        }

        if !pass {
            new_stream.push(token.clone());
            continue;
        }

        // Makes implicit multiplication explicit
        pass = false;
        if index > 0 {
            if let TokenTree::Punct(punct) = &tokens[index - 1] {
                match punct.as_char() {
                    ']' | ',' | '+' | '-' | '*' | '/' => pass = true,
                    _ => ()
                }
            }
            if !pass {
                new_stream.push(TokenTree::Punct(Punct::new('*', Spacing::Alone)));
            }
        }
        
        new_stream.extend(stream_accumulator);
        
        pass = false;
        if index < tokens.len() - 1 {
            if let TokenTree::Punct(punct) = &tokens[index + 1] {
                match punct.as_char() {
                    ']' | ',' | '+' | '-' | '*' | '/' => pass = true,
                    _ => ()
                }
            }
            if !pass {
                new_stream.push(TokenTree::Punct(Punct::new('*', Spacing::Alone)));
            }
        }
        
    }
    return TokenStream2::from_iter(new_stream);
}

fn tensor_lexer(token_stream: TokenStream2) -> Result<Vec<TensorTokens>> {
    let mut processed_tokens = Vec::new();
    let mut token_iterator = token_stream.into_iter();

    let mut number_token_accumulator = Vec::new();

    while let Some(token) = token_iterator.next() {
        match &token {
            
            TokenTree::Literal(_literal) => {
                number_token_accumulator.push(token);
            }

            TokenTree::Ident(_identifier) => {
                number_token_accumulator.push(token);
            }

            TokenTree::Punct(punctuation) => match punctuation.as_char() {
                ',' => {
                    if !number_token_accumulator.is_empty() {
                        processed_tokens.push(TensorTokens::Number(number_token_accumulator));
                        number_token_accumulator = Vec::new();
                    }
                    
                    processed_tokens.push(TensorTokens::Comma)
                },
                _ => number_token_accumulator.push(token)
            }

            TokenTree::Group(group) => match group.delimiter() {
                Delimiter::Bracket => {
                    if !number_token_accumulator.is_empty() {
                        processed_tokens.push(TensorTokens::Number(number_token_accumulator));
                        number_token_accumulator = Vec::new();
                    }

                    processed_tokens.push(TensorTokens::OpenBracket);
                    processed_tokens.extend(tensor_lexer(group.stream())?);
                    processed_tokens.push(TensorTokens::ClosedBracket);
                },
                _ => number_token_accumulator.push(token)
            }

        }
    }
    if !number_token_accumulator.is_empty() {
        processed_tokens.push(TensorTokens::Number(number_token_accumulator));
    }
    return Ok(processed_tokens);
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

            if let Some(TensorTokens::ClosedBracket) = tokens.get(index) {
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
    } else if let Some(TensorTokens::Number(token_tree_vec)) = tokens.get(index) {
            return Ok((RecursiveTensor::Scalar(token_tree_vec.clone()), 1));
    } else {
        return Err(Error::new(Span::call_site(), "Not a valid tensor"));
    }
        
}

// Returns elements of the RecursiveTensor (in vector form) and its shape.
fn flatten_tensor_data(input: &RecursiveTensor) -> (Vec<TokenStream2>, Vec<usize>) {
    match input {

        RecursiveTensor::Scalar(token_vec) => {
            let stream = TokenStream2::from_iter(token_vec.clone());
            (vec![stream.clone()], vec![])
        }

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
pub fn complex_tensor64(input: TokenStream) -> TokenStream {

    let input_processed = j_processing64(input.into());

    let tokens = match tensor_lexer(input_processed) {
        Ok(inner_val) => inner_val,
        Err(error) => return error.to_compile_error().into(),
    };

    let (recursive_tensor, _) = match tensor_parser(&tokens) {
        Ok(inner_val) => inner_val,
        Err(error) => return error.to_compile_error().into(),
    };

    let (flat_data, shape) = flatten_tensor_data(&recursive_tensor);
    
    let quoted_shape = quote!{[#(#shape),*]};

    let d = shape.len();
    quote!{{
            let data_vec: Vec<faer::c64> = vec![#(#flat_data),*];
            Tensor::<faer::c64, #d>::from_slice(&data_vec, #quoted_shape)
    }}.into()
}

#[proc_macro]
pub fn complex_tensor32(input: TokenStream) -> TokenStream {

    let input_processed = j_processing32(input.into());

    let tokens = match tensor_lexer(input_processed) {
        Ok(inner_val) => inner_val,
        Err(error) => return error.to_compile_error().into(),
    };

    let (recursive_tensor, _) = match tensor_parser(&tokens) {
        Ok(inner_val) => inner_val,
        Err(error) => return error.to_compile_error().into(),
    };

    let (flat_data, shape) = flatten_tensor_data(&recursive_tensor);
    
    let quoted_shape = quote!{[#(#shape),*]};

    let d = shape.len();
    quote!{{
            let data_vec: Vec<faer::c32> = vec![#(#flat_data),*];
            Tensor::<faer::c32, #d>::from_slice(&data_vec, #quoted_shape)
    }}.into()
}
