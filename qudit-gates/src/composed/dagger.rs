use qudit_core::{HasParams, QuditRadices, QuditSystem};
use qudit_expr::{UnitaryExpression, UnitaryExpressionGenerator};

/// An arbitrary inverted gate.
///
/// Given any gate, DaggerGate takes the conjugate transpose of the input gate.
#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub struct DaggerGate {
    // The expression being inverted.
    expr: UnitaryExpression,
}

impl DaggerGate {
    /// Construct a DaggerGate.
    ///
    /// # Arguments
    ///
    /// * `gate` - The gate to invert.
    ///
    /// # Returns
    ///
    /// A new DaggerGate.
    ///
    /// # Examples
    ///
    /// // TODO: Come back to later
    pub fn new<E: UnitaryExpressionGenerator>(expr: E) -> Self {
        let gate_expr = expr.gen_expr();
        let expr = gate_expr.conjugate().transpose();
        DaggerGate { expr }
    }
}

impl HasParams for DaggerGate {
    #[inline]
    fn num_params(&self) -> usize {
        self.expr.num_params()
    }
}

impl QuditSystem for DaggerGate {
    #[inline]
    fn radices(&self) -> QuditRadices {
        self.expr.radices()
    }

    #[inline]
    fn num_qudits(&self) -> usize {
        self.expr.num_qudits()
    }

    #[inline]
    fn dimension(&self) -> usize {
        self.expr.dimension()
    }
}

impl UnitaryExpressionGenerator for DaggerGate {
    #[inline]
    fn gen_expr(&self) -> UnitaryExpression {
        self.expr.clone()
    }
}

#[cfg(test)]
pub mod strategies {
    // use proptest::prelude::*;
    // use proptest::strategy::BoxedStrategy;
    // use proptest::strategy::Strategy;

    // use super::*;
    // use crate::gates::strategies::ArbitraryGateWithRadices;

    // impl Arbitrary for DaggerGate {
    //     type Parameters = Option<Gate>;
    //     type Strategy = BoxedStrategy<Self>;

    //     fn arbitrary_with(args: Self::Parameters) -> Self::Strategy {
    //         let gate_strat = match args {
    //             Some(gate) => Just(gate).boxed(),
    //             None => any_with::<QuditRadices>((2, 4, 1, 1))
    //                 .prop_flat_map(|radices| {
    //                     Gate::arbitrary_with_radices_no_rec(radices).unwrap()
    //                 })
    //                 .boxed(),
    //         };

    //         gate_strat.prop_map(|gate| DaggerGate::new(gate)).boxed()
    //     }
    // }

    // impl ArbitraryGateWithRadices for DaggerGate {
    //     fn arbitrary_with_radices(
    //         radices: QuditRadices,
    //     ) -> Option<BoxedStrategy<Gate>> {
    //         if radices.get_num_qudits() != 1 {
    //             return None;
    //         }
    //         Some(
    //             Gate::arbitrary_with_radices_no_rec(radices)
    //                 .unwrap()
    //                 .prop_map(|g| Gate::from(DaggerGate::new(g)))
    //                 .boxed(),
    //         )
    //     }

    //     fn arbitrary_with_radices_no_rec(
    //         _radices: QuditRadices,
    //     ) -> Option<BoxedStrategy<Gate>> {
    //         None
    //     }
    // }
}

// #[cfg(test)]
// mod test {

//     use super::*;

//     #[test]
//     fn test_qubit_h_gate() {
//         let h_gate = HGate::new(2);
//         let unitary = h_gate.get_unitary(&[]);
//         let correct = array![
//             [
//                 c64::new(1.0 / 2.0_f64.sqrt(), 0.0),
//                 c64::new(1.0 / 2.0_f64.sqrt(), 0.0)
//             ],
//             [
//                 c64::new(1.0 / 2.0_f64.sqrt(), 0.0),
//                 c64::new(-1.0 / 2.0_f64.sqrt(), 0.0)
//             ]
//         ];
//         unitary.assert_close_to(&correct);
//     }
// }
