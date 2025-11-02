use proc_macro2::{Span, TokenStream};
use syn::{parse_macro_input, LitFloat, Result, Token, bracketed};
use syn::parse::{Parse, ParseStream};
use syn::punctuated::Punctuated;
use quote::quote;
//use faer::{c64, Mat};

// ###########################################################################################################
enum TensorTokens {
    J,
    OpenBracket,
    ClosedBracket,
    Comma,
    Number,
    Op,
    OpenParenthesis,
    ClosedParenthesis,
}

struct ComplexElement {
    real: LitFloat,
    imag: LitFloat,
}

mod keywords {
    syn::custom_keyword!(j);
}

impl Parse for ComplexElement {
    fn parse(elem: ParseStream) -> Result<Self> {
        let mut real_float = 0.0;
        let mut imag_float = 0.0;
        let mut number_accumulator: f64;
        let mut sign_accumulator: f64 = 1.0;
        let mut lit_float_accumulator: LitFloat;

        while !elem.is_empty() && !elem.peek(Token![,]) {
            // If we have a structure like `4.3j`, we need to detect `j` via .suffix().
            // If we have `4.3 j`, `4.3` and `j` are separate tokens and we detect `j` via `elem.peek(Token![,])`.

            lit_float_accumulator = elem.parse::<LitFloat>()?;
            number_accumulator = lit_float_accumulator.base10_parse::<f64>()?;
            
            if lit_float_accumulator.suffix() == "j" {
                // The case where we have `j` next to a float without whitespace
                imag_float += number_accumulator * sign_accumulator;
            } else if elem.peek(keywords::j) {
                // The case where we have `j` next to a float with whitespace
                imag_float += number_accumulator * sign_accumulator;
                elem.parse::<keywords::j>()?;
            } else {
                real_float += number_accumulator * sign_accumulator;
            }

            if elem.is_empty() || elem.peek(Token![,]) {
                break;
            } else if elem.peek(Token![+]) {
                sign_accumulator = 1.0;
                elem.parse::<Token![+]>()?;
            } else if elem.peek(Token![-]) {
                sign_accumulator = -1.0;
                elem.parse::<Token![-]>()?;
            } else {
                return Err(elem.error("Expected a sign but got something else"));
            }
        }
        let mut real_str = real_float.to_string();
        let mut imag_str = imag_float.to_string();

        // There's a bug in `.to_string()` where floats with no decimal part is converted 
        // to integer notation. We thus need to add the decimal back in.
        if !real_str.contains('.') {
            real_str.push_str(".0");
        }
        if !imag_str.contains('.') {
            imag_str.push_str(".0");
        }

        let real = LitFloat::new(&real_str, Span::call_site());
        let imag = LitFloat::new(&imag_str, Span::call_site());

        Ok(ComplexElement {real, imag})
    }
}

/// # Example
/// ```
/// use faer::c64;
/// use qudit_core_proc_macros::complex_elem;
/// 
/// let test1 = complex_elem!(3.4);
/// let test2 = complex_elem!(5.4 j);
/// let test3 = complex_elem!(3.4 + 5.4 j);
/// let test4 = complex_elem!(3.4 - 5.4 j);
/// let test5 = complex_elem!(3.4 j - 5.4);
/// let test6 = complex_elem!(-3.4 j - 5.4);
/// 
/// let expected1 = c64::(3.4, 0.0);
/// let expected2 = c64::(0.0, 5.4);
/// let expected3 = c64::(3.4, 5.4);
/// let expected4 = c64::(3.4, -5.4);
/// let expected5 = c64::(-5.4, 3.4);
/// let expected6 = c64::(-5.4, -3.4);
/// 
/// assert_eq!(test1, expected1);
/// assert_eq!(test2, expected2);
/// assert_eq!(test3, expected3);
/// assert_eq!(test4, expected4);
/// assert_eq!(test5, expected5);
/// assert_eq!(test6, expected6);
/// ```
#[proc_macro]
pub fn complex_elem(elem: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let parsed_elem = parse_macro_input!(elem as ComplexElement);

    let real_part = parsed_elem.real;
    let imag_part = parsed_elem.imag;

    quote!{c64::new(#real_part, #imag_part)}.into()
}

// ###########################################################################################################

struct ComplexMatrix {
    data: Vec<Vec<ComplexElement>>,
}

impl Parse for ComplexMatrix {
    fn parse(matrix: ParseStream) -> Result<Self> {
        let rows;
        bracketed!(rows in matrix);

        let mut data = Vec::new();
        let mut parsed_row_accumulator: Punctuated<ComplexElement, Token![,]>;

        while !rows.is_empty() {
            let row;
            bracketed!(row in rows);

            parsed_row_accumulator = Punctuated::parse_terminated(&row)?;
            
            data.push(
                parsed_row_accumulator.into_iter().collect()
            );

            if rows.peek(Token![,]) {
                rows.parse::<Token![,]>()?;
            }
        }
        Ok(ComplexMatrix{data})
    }
}

/// # Example
/// ```
/// use faer::{c64, Mat};
/// use qudit_core_proc_macros::complex_mat;
/// 
/// let answer = complex_mat!([
///     [1.0 + 1.0 j, 4.2 + 1.5 j],
///     [2.0 + 1.0 j, 4.9 + 1.5 j]
/// ]);
/// 
/// let expected_data = [
///     [c64::new(1.0, 1.0), c64::new(4.2, 1.5)],
///     [c64::new(2.0, 1.0), c64::new(4.9, 1.5)]
/// ];
/// let expected = Mat::from_fn(2, 2, |i, j| -> c64 {expected_data[i][j]});
/// 
/// assert_eq!(answer, expected);
/// ```
#[proc_macro]
pub fn complex_mat(raw_matrix: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let matrix = parse_macro_input!(raw_matrix as ComplexMatrix);

    let num_rows = matrix.data.len();

    let num_cols: usize;
    if num_rows == 0 {
        num_cols = 0;
    } else {
        num_cols = matrix.data[0].len();
    }

    let mut processed_matrix: Vec<TokenStream> = Vec::new();
    let mut row_accumulator: Vec<TokenStream>;
    let mut real: LitFloat;
    let mut imag: LitFloat;
    for row in matrix.data {
        row_accumulator = Vec::<TokenStream>::new();
        for elem in row {
            real = elem.real;
            imag = elem.imag;
            row_accumulator.push(
                quote!{faer::c64::new(#real, #imag)}.into()
            );
        }
        processed_matrix.push(
            quote! {
                [#(#row_accumulator),*]
            }.into()
        );
    }

    quote!{
        {
            let temp_array = [#(#processed_matrix),*];
            Mat::from_fn(#num_rows, #num_cols, |i, j| temp_array[i][j])
        }
    }.into()
}

// ###########################################################################################################