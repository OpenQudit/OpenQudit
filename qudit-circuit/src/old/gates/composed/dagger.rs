use std::hash::Hash;
use std::ops::Range;

use faer_core::unzipped;
use faer_core::zipped;
use faer_core::MatMut;

use super::super::NamedGate;
use crate::math::matrix::MatGradMut;
use crate::math::matrix::MatHessMut;
use crate::math::unitary::DifferentiableUnitaryFn;
use crate::math::unitary::DoublyDifferentiableUnitaryFn;
use crate::math::unitary::UnitaryFn;
use crate::math::unitary::UnitaryGradient;
use crate::math::unitary::UnitaryMatrix;
use crate::math::BoundedFn;
use crate::math::ComplexScalar;
use crate::math::Function;
use crate::Gate;
use crate::QuditRadices;
use crate::QuditSystem;

/// An arbitrary inverted gate.
///
/// Given any gate, DaggerGate takes the conjugate transpose of the input gate.
#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub struct DaggerGate {
    // The gate being inverted.
    gate: Box<Gate>,
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
    pub fn new(gate: Gate) -> Self {
        DaggerGate {
            gate: Box::new(gate.clone()),
        }
    }
}

impl NamedGate for DaggerGate {
    #[inline]
    fn get_name(&self) -> String {
        format!("Dagger({})", self.gate.get_name())
    }
}

impl Function for DaggerGate {
    #[inline]
    fn get_num_params(&self) -> usize {
        self.gate.get_num_params()
    }
}

impl BoundedFn for DaggerGate {
    #[inline]
    fn get_bounds(&self) -> Vec<Range<f64>> {
        self.gate.get_bounds()
    }
}

impl QuditSystem for DaggerGate {
    #[inline]
    fn get_radices(&self) -> QuditRadices {
        self.gate.get_radices().clone()
    }
}

impl<C: ComplexScalar> UnitaryFn<C> for DaggerGate {
    #[inline]
    fn write_unitary(&self, params: &[C::Re], utry: &mut MatMut<C>) {
        let mut transpose_utry = utry.as_mut().transpose_mut();
        self.gate.write_unitary(params, &mut transpose_utry);
        let mut utry = transpose_utry.transpose_mut();

        for r in 0..utry.nrows() {
            for c in 0..utry.ncols() {
                utry.write(r, c, utry.read(r, c).conj());
            }
        }
    }
}

impl<C: ComplexScalar> DifferentiableUnitaryFn<C> for DaggerGate {
    fn write_gradient(&self, params: &[C::Re], grad: &mut MatGradMut<C>) {
        let gate_grad: UnitaryGradient<C> = self.gate.get_gradient(params);
        for (i, g) in gate_grad.into_iter().enumerate() {
            zipped!(grad.get_matmut(i), g.transpose()).for_each(
                |unzipped!(mut out, g)| {
                    out.write(g.read().conj());
                },
            );
        }
    }

    fn write_unitary_and_gradient(
        &self,
        params: &[C::Re],
        utry: &mut MatMut<C>,
        grad: &mut MatGradMut<C>,
    ) {
        let (gate_unitary, gate_grad): (UnitaryMatrix<C>, UnitaryGradient<C>) =
            self.gate.get_unitary_and_gradient(params);
        zipped!(utry.as_mut(), gate_unitary.as_ref().transpose()).for_each(
            |unzipped!(mut out, u)| {
                out.write(u.read().conj());
            },
        );
        for (i, g) in gate_grad.into_iter().enumerate() {
            zipped!(grad.get_matmut(i), g.transpose()).for_each(
                |unzipped!(mut out, g)| {
                    out.write(g.read().conj());
                },
            );
        }
    }
}

impl<C: ComplexScalar> DoublyDifferentiableUnitaryFn<C> for DaggerGate {
    fn write_hessian(&self, params: &[C::Re], hess: &mut MatHessMut<C>) {
        let mut gate_hess = self.gate.get_hessian(params);
        let gate_hess: MatHessMut<C> = gate_hess.partials.as_mut();
        for p1 in 0..gate_hess.num_params() {
            for p2 in p1..gate_hess.num_params() {
                let g = gate_hess.get_matref(p1, p2);
                let mut o = hess.get_matmut(p1, p2);
                zipped!(o.as_mut(), g.transpose()).for_each(
                    |unzipped!(mut out, elem)| {
                        out.write(elem.read().conj());
                    },
                );
            }
        }
    }
}

#[cfg(test)]
pub mod strategies {
    use proptest::prelude::*;
    use proptest::strategy::BoxedStrategy;
    use proptest::strategy::Strategy;

    use super::*;
    use crate::gates::strategies::ArbitraryGateWithRadices;

    impl Arbitrary for DaggerGate {
        type Parameters = Option<Gate>;
        type Strategy = BoxedStrategy<Self>;

        fn arbitrary_with(args: Self::Parameters) -> Self::Strategy {
            let gate_strat = match args {
                Some(gate) => Just(gate).boxed(),
                None => any_with::<QuditRadices>((2, 4, 1, 1))
                    .prop_flat_map(|radices| {
                        Gate::arbitrary_with_radices_no_rec(radices).unwrap()
                    })
                    .boxed(),
            };

            gate_strat.prop_map(|gate| DaggerGate::new(gate)).boxed()
        }
    }

    impl ArbitraryGateWithRadices for DaggerGate {
        fn arbitrary_with_radices(
            radices: QuditRadices,
        ) -> Option<BoxedStrategy<Gate>> {
            if radices.get_num_qudits() != 1 {
                return None;
            }
            Some(
                Gate::arbitrary_with_radices_no_rec(radices)
                    .unwrap()
                    .prop_map(|g| Gate::from(DaggerGate::new(g)))
                    .boxed(),
            )
        }

        fn arbitrary_with_radices_no_rec(
            _radices: QuditRadices,
        ) -> Option<BoxedStrategy<Gate>> {
            None
        }
    }
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
