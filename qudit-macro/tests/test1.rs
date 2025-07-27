//use qudit_core_proc_macros::{complex_mat, complex_elem};
//use faer::{c64, Mat};

use qudit_core_proc_macros::complex_tensor;
use quote::quote;





//--------------------------------------------------------------------------------



//--------------------------------------------------------------------------------




// #[test]
// fn test_complex_mat() {
//     let answer = complex_mat!([
//         [1.0+1.0j , 4.2+1.5 j],
//         [2.0+1.0 j , 4.9+1.5j]
//     ]);

//     let expected_data = [
//         [c64::new(1.0, 1.0), c64::new(4.2, 1.5)],
//         [c64::new(2.0, 1.0), c64::new(4.9, 1.5)]
//     ];
//     let expected = Mat::from_fn(2, 2, |i, j| -> c64 {expected_data[i][j]});

//     assert_eq!(answer, expected);

//     let test1 = complex_mat!([
//         [1.0]
//     ]);
// }

// #[test]
// fn test_complex_elem() {
//     let test1 = complex_elem!(3.4);
//     let test2 = complex_elem!(5.4 j);
//     let test3 = complex_elem!(3.4 + 5.4 j);
//     let test4 = complex_elem!(3.4 - 5.4 j);
//     let test5 = complex_elem!(3.4 j - 5.4);
//     let test6 = complex_elem!(-3.4 j - 5.4);
//     let test7 = complex_elem!(-3.4j - 5.4);
    
//     let expected1 = c64::new(3.4, 0.0);
//     let expected2 = c64::new(0.0, 5.4);
//     let expected3 = c64::new(3.4, 5.4);
//     let expected4 = c64::new(3.4, -5.4);
//     let expected5 = c64::new(-5.4, 3.4);
//     let expected6 = c64::new(-5.4, -3.4);
//     let expected7 = c64::new(-5.4, -3.4);
    
//     assert_eq!(test1, expected1);
//     assert_eq!(test2, expected2);
//     assert_eq!(test3, expected3);
//     assert_eq!(test4, expected4);
//     assert_eq!(test5, expected5);
//     assert_eq!(test6, expected6);
//     assert_eq!(test7, expected7);
// }