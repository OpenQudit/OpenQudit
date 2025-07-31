use proc_macro::TokenStream;
use proc_macro2::{TokenStream as TokenStream2, TokenTree, Delimiter, Span, Spacing, Punct, Group};
use syn::{Result, Error};
use quote::quote;

#[derive(Debug, Clone)]
enum TensorTokens {
    OpenBracket,
    ClosedBracket,
    Comma,
    Number(Vec<TokenTree>)
}

#[derive(Debug, Clone)]
enum RecursiveTensor {
    Scalar(Vec<TokenTree>),
    SubTensor(Vec<RecursiveTensor>)
}

/// Replaces `j` with `faer::c32::new(0.0, 1.0)` in the input token stream.
/// Also makes implicit multiplication explicit. (e.g. `4j` becomes `4.0 * j`)
/// 
/// # Arguments
/// 
/// * `input` - A tokenstream containing the input tokens.
/// 
/// # Returns
/// 
/// * A tokenstream with the processed tokens.
/// 
/// # Panics
/// 
/// * If a literal with suffix or prefix `j` is not a valid number.
/// 
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

/// Replaces `j` with `faer::c64::new(0.0, 1.0)` in the input token stream.
/// Also makes implicit multiplication explicit. (e.g. `4j` becomes `4.0 * j`)
/// 
/// # Arguments
/// 
/// * `input` - A tokenstream containing the input tokens.
/// 
/// # Returns
/// 
/// * A tokenstream with the processed tokens.
/// 
/// # Panics
/// 
/// * If a literal with suffix or prefix `j` is not a valid number.
/// 
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

/// Categorizes the tokens in the input token stream to aid in parsing.
/// 
/// # Arguments
/// 
/// * `token_stream` - A tokenstream containing the input tokens.
/// 
/// # Returns
/// 
/// * A tokenstream with the processed tokens.
/// 
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

/// Organizes a series of custom tokens into a recursive tensor structure.
/// 
/// # Arguments
/// 
/// * `tokens` - A slice of `TensorTokens`, expected from `tensor_lexer`.
/// 
/// # Returns
/// 
/// * A recursive tensor storing the user's tokens.
/// * The number of tokens consumed from the input slice.
/// 
/// # Panics
/// 
/// * If there is a missing closing bracket.
/// * If the tensor does not start with an opening bracket or is not a scalar.
/// 
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

/// Flattens the recursive tensor structure into a single vector of tokens and calculates its shape.
/// 
/// # Arguments
/// 
/// * `input` - A reference to the recursive tensor structure.
/// 
/// # Returns
/// 
/// * A vector containing all elements of the input tensor.
/// * The shape of the input tensor.
/// 
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

/// Creates a 64-bit complex tensor from nested brackets. Complex numbers
/// can be created using `j`. (e.g. `4j`, `my_function()j`, or `some_variable * j`)
/// 
/// # Arguments
/// 
/// * `input` - The user's desired tensor written in simplified language.
/// 
/// # Returns
/// 
/// * A 64-bit complex tensor implementing the user's data.
/// 
/// # Panics
/// 
/// * If there is a missing closing bracket.
/// * If the tensor does not start with an opening bracket or is not a scalar.
/// * If a literal with suffix or prefix `j` is not a valid number.
/// 
/// # Example
/// ```
/// use qudit_macro::complex_tensor64;
/// use qudit_core::array::Tensor;
/// use qudit_core::c64;
/// use std::slice::from_raw_parts;
/// 
/// fn arbitrary_func(x: f64, y: f64) -> f64 {
///     return x * y + 9.0 * x;
/// }
/// 
/// let attempt = complex_tensor64!([
///    3.0 * arbitrary_func(1.5, 2.0)j + 4.5,
///   -(2.0j + arbitrary_func(5.5, 3.5))
/// ]);
/// let expected_data = vec![c64::new(4.5, 49.5), c64::new(-68.75, -2.0)];
/// let expected = Tensor::<c64, 1>::from_slice(&expected_data, [2]);
/// 
/// assert_eq!(attempt.dims(), expected.dims());
/// unsafe {
///    assert_eq!(from_raw_parts(attempt.as_ptr(), 2), from_raw_parts(expected.as_ptr(), 2));
/// }
/// ```
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

/// Creates a 32-bit complex tensor from nested brackets. Complex numbers
/// can be created using `j`. (e.g. `4j`, `my_function()j`, or `some_variable * j`)
/// 
/// # Arguments
/// 
/// * `input` - The user's desired tensor written in simplified language.
/// 
/// # Returns
/// 
/// * A 32-bit complex tensor implementing the user's data.
/// 
/// # Panics
/// 
/// * If there is a missing closing bracket.
/// * If the tensor does not start with an opening bracket or is not a scalar.
/// * If a literal with suffix or prefix `j` is not a valid number.
/// 
/// # Example
/// ```
/// use qudit_macro::complex_tensor32;
/// use qudit_core::array::Tensor;
/// use qudit_core::c32;
/// use std::slice::from_raw_parts;
/// 
/// fn arbitrary_func(x: f32, y: f32) -> f32 {
///     return x * y + 9.0 * x;
/// }
/// 
/// let attempt = complex_tensor32!([
///    3.0 * arbitrary_func(1.5, 2.0)j + 4.5,
///   -(2.0j + arbitrary_func(5.5, 3.5))
/// ]);
/// let expected_data = vec![c32::new(4.5, 49.5), c32::new(-68.75, -2.0)];
/// let expected = Tensor::<c32, 1>::from_slice(&expected_data, [2]);
/// 
/// assert_eq!(attempt.dims(), expected.dims());
/// unsafe {
///    assert_eq!(from_raw_parts(attempt.as_ptr(), 2), from_raw_parts(expected.as_ptr(), 2));
/// }
/// ```
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
