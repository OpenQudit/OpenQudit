use std::ops::Deref;

use qudit_core::{ParamIndices, RealScalar};
use qudit_expr::Constant;
use qudit_expr::ComplexExpression;
use qudit_expr::Expression;
use qudit_expr::TensorExpression;
use std::any::TypeId;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Parameter {
    /// Named dynamic variables that appear in the circuit
    Named(String),

    /// Unnamed dynamic variables that appear in the circuit
    Indexed,

    /// Statically assigned parameters that do not change
    Static(Constant),
}

pub struct ParamEntry {
    expression: Expression,
}

impl ParamEntry {
    pub fn new(input: impl AsRef<str>) -> ParamEntry {
        let parsed = ComplexExpression::from_string(input);

        if !parsed.is_real_fast() {
            panic!("Unable to handle complex parameter entries currently.");
        }

        ParamEntry {
            expression: parsed.real
        }
    }

    pub fn get_unique_variables(&self) -> Vec<String> {
        self.expression.get_unique_variables()
    }

    pub fn is_constant(&self) -> bool {
        !self.expression.is_parameterized()
    }

    // pub fn is_real(&self) -> bool {
    //     self.expression.is_real_fast()
    // }
    
    pub fn to_real_constant(&self) -> Constant {
        // self.expression.real.to_constant()
        self.expression.to_constant()
    }
}

impl From<f64> for ParamEntry {
    fn from(value: f64) -> Self {
        ParamEntry {
            expression: Expression::from_float_64(value)
        }
    }
}

impl From<f32> for ParamEntry {
    fn from(value: f32) -> Self {
        ParamEntry {
            expression: Expression::from_float_32(value)
        }
    }
}

impl From<String> for ParamEntry {
    fn from(value: String) -> Self {
        ParamEntry::new(value)
    }
}

impl From<&str> for ParamEntry {
    fn from(value: &str) -> Self {
        ParamEntry::new(value)
    }
}

pub struct ParamEntries {
    entries: Vec<ParamEntry>
}

impl ParamEntries {
    pub fn new(entries: Vec<ParamEntry>) -> Self {
        Self { entries }
    }

    pub fn to_expressions(self) -> Vec<Expression> {
        self.entries.into_iter().map(|e| e.expression).collect()
    }

    pub fn get_unique_variables(&self) -> Vec<String> {
        let mut unique_variables = vec![];
        for entry in self.entries.iter() {
            for var in entry.get_unique_variables() {
                if !unique_variables.contains(&var) {
                    unique_variables.push(var)
                }
            }
        }
        unique_variables
    }

    pub fn organize_parameter_vector(&self) -> Vec<Parameter> {
        let mut params = vec![];
        for entry in self.entries.iter() {
            if entry.is_constant() {
                // if !entry.is_real() {
                //     panic!("Cannot have imaginary or complex constant parameter entry");
                // }
                params.push(Parameter::Static(entry.to_real_constant()));
            } else {
                for var in entry.get_unique_variables() {
                    let param = Parameter::Named(var);
                    if !params.contains(&param) {
                        params.push(param);
                    }
                }
            }
        }
        params
    }
}

impl<E: Into<ParamEntry>> From<Vec<E>> for ParamEntries {
    fn from(value: Vec<E>) -> Self {
        Self::new(value.into_iter().map(|e| e.into()).collect())
    }
}

impl<E: Into<ParamEntry>, const N: usize> From<[E; N]> for ParamEntries {
    fn from(value: [E; N]) -> Self {
        Self::new(value.into_iter().map(|e| e.into()).collect())
    }
}

impl<E: Into<ParamEntry> + Clone, const N: usize> From<&[E; N]> for ParamEntries {
    fn from(value: &[E; N]) -> Self {
        Self::new(value.into_iter().map(|e| e.clone().into()).collect())
    }
}

impl Deref for ParamEntries {
    type Target = [ParamEntry];

    fn deref(&self) -> &Self::Target {
        &self.entries
    }
}


// pub enum ParamEntry<R: RealScalar> {
//     Uninitialized,
//     Static(R),
//     Dynamic(R),
//     Existing(usize),
//     // Parametric(String),
//     // Named(String),
//     // TODO: Parametric
// }

// impl <R: RealScalar> ParamEntry<R> {
//     fn is_static(&self) -> bool {
//         matches!(&self, ParamEntry::Static(_))
//     }

//     fn is_existing(&self) -> bool {
//         matches!(&self, ParamEntry::Existing(_))
//     }

//     fn is_new(&self) -> bool {
//         match self {
//             ParamEntry::Uninitialized => true,
//             ParamEntry::Static(_) => false,
//             ParamEntry::Dynamic(_) => true,
//             ParamEntry::Existing(_) => false,
//         }
//     }
// }

// impl<R: RealScalar> From<R> for ParamEntry<R> {
//     fn from(value: R) -> Self {
//         ParamEntry::Dynamic(value)
//     }
// }

// impl<R: RealScalar> From<Option<R>> for ParamEntry<R> {
//     fn from(value: Option<R>) -> Self {
//         match value {
//             Some(value) => ParamEntry::Dynamic(value),
//             None => ParamEntry::Uninitialized,
//         }
//     }
// }

// pub struct ParamList<R: RealScalar> {
//     list: Vec<ParamEntry<R>>,
// }

// impl<R: RealScalar> ParamList<R> {
//     pub fn new(list: Vec<ParamEntry<R>>) -> Self {
//         Self {
//             list
//         }
//     }

//     pub fn empty() -> Self {
//         Self {
//             list: vec![],
//         }
//     }

//     pub fn is_any_existing(&self) -> bool {
//         self.list.iter().any(|e| e.is_existing())
//     }

//     pub fn new_count(&self) -> usize {
//         self.list.iter().filter(|e| e.is_new()).count()
//     }

//     pub fn static_count(&self) -> usize {
//         self.list.iter().filter(|e| e.is_static()).count()
//     }

//     pub fn to_param_indices(&self, existing_length: usize) -> ParamIndices {
//         if !self.is_any_existing() {
//             ParamIndices::Joint(existing_length, self.len() - self.static_count())
//         } else {
//             let mut indices = vec![];
//             let mut non_existing_count = 0;
//             for entry in self.iter() {
//                 match entry {
//                     ParamEntry::Existing(id) => {
//                         assert!(*id < existing_length);
//                         indices.push(*id);
//                     }
//                     ParamEntry::Static(_) => {}
//                     _ => {
//                         indices.push(non_existing_count + existing_length);
//                         non_existing_count += 1;
//                     }
//                 }
//             }
//             ParamIndices::Disjoint(indices)
//         }
//     }

//     pub fn new_entries(&self) -> impl Iterator<Item = &ParamEntry<R>> {
//         self.list.iter().filter(|e| e.is_new())
//     }

//     pub fn new_entries_unwrapped(&self) -> impl Iterator<Item = R> + '_ {
//         self.list.iter().filter(|e| e.is_new()).map(|e| match e {
//             ParamEntry::Uninitialized => R::zero(),
//             ParamEntry::Dynamic(r) => *r,
//             _ => unreachable!(),
//         })
//     }
// }

// impl<R: RealScalar, E: Into<ParamEntry<R>>> From<Vec<E>> for ParamList<R> {
//     fn from(value: Vec<E>) -> Self {
//         Self::new(value.into_iter().map(|e| e.into()).collect())
//     }
// }

// impl<R: RealScalar, E: Into<ParamEntry<R>>, const N: usize> From<[E; N]> for ParamList<R> {
//     fn from(value: [E; N]) -> Self {
//         Self::new(value.into_iter().map(|e| e.into()).collect())
//     }
// }

// impl<R: RealScalar> Deref for ParamList<R> {
//     type Target = [ParamEntry<R>];

//     fn deref(&self) -> &Self::Target {
//         &self.list
//     }
// }
