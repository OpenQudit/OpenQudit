use std::ops::Deref;

use qudit_core::{ParamIndices, RealScalar};

pub enum ParamEntry<R: RealScalar> {
    Uninitialized,
    Static(R),
    Dynamic(R),
    Existing(usize),
    // Named(String),
    // TODO: Parametric
}

impl <R: RealScalar> ParamEntry<R> {
    fn is_static(&self) -> bool {
        matches!(&self, ParamEntry::Static(_))
    }

    fn is_existing(&self) -> bool {
        matches!(&self, ParamEntry::Existing(_))
    }

    fn is_new(&self) -> bool {
        match self {
            ParamEntry::Uninitialized => true,
            ParamEntry::Static(_) => false,
            ParamEntry::Dynamic(_) => true,
            ParamEntry::Existing(_) => false,
        }
    }
}

impl<R: RealScalar> From<R> for ParamEntry<R> {
    fn from(value: R) -> Self {
        ParamEntry::Dynamic(value)
    }
}

impl<R: RealScalar> From<Option<R>> for ParamEntry<R> {
    fn from(value: Option<R>) -> Self {
        match value {
            Some(value) => ParamEntry::Dynamic(value),
            None => ParamEntry::Uninitialized,
        }
    }
}

pub struct ParamList<R: RealScalar> {
    list: Vec<ParamEntry<R>>,
}

impl<R: RealScalar> ParamList<R> {
    pub fn new(list: Vec<ParamEntry<R>>) -> Self {
        Self {
            list
        }
    }

    pub fn empty() -> Self {
        Self {
            list: vec![],
        }
    }

    pub fn is_any_existing(&self) -> bool {
        self.list.iter().any(|e| e.is_existing())
    }

    pub fn new_count(&self) -> usize {
        self.list.iter().filter(|e| e.is_new()).count()
    }

    pub fn static_count(&self) -> usize {
        self.list.iter().filter(|e| e.is_static()).count()
    }

    pub fn to_param_indices(&self, existing_length: usize) -> ParamIndices {
        if !self.is_any_existing() {
            ParamIndices::Joint(existing_length, self.len() - self.static_count())
        } else {
            let mut indices = vec![];
            let mut non_existing_count = 0;
            for entry in self.iter() {
                match entry {
                    ParamEntry::Existing(id) => {
                        assert!(*id < existing_length);
                        indices.push(*id);
                    }
                    ParamEntry::Static(_) => {}
                    _ => {
                        indices.push(non_existing_count + existing_length);
                        non_existing_count += 1;
                    }
                }
            }
            ParamIndices::Disjoint(indices)
        }
    }

    pub fn new_entries(&self) -> impl Iterator<Item = &ParamEntry<R>> {
        self.list.iter().filter(|e| e.is_new())
    }

    pub fn new_entries_unwrapped(&self) -> impl Iterator<Item = R> + '_ {
        self.list.iter().filter(|e| e.is_new()).map(|e| match e {
            ParamEntry::Uninitialized => R::zero(),
            ParamEntry::Dynamic(r) => *r,
            _ => unreachable!(),
        })
    }
}

impl<R: RealScalar, E: Into<ParamEntry<R>>> From<Vec<E>> for ParamList<R> {
    fn from(value: Vec<E>) -> Self {
        Self::new(value.into_iter().map(|e| e.into()).collect())
    }
}

impl<R: RealScalar, E: Into<ParamEntry<R>>, const N: usize> From<[E; N]> for ParamList<R> {
    fn from(value: [E; N]) -> Self {
        Self::new(value.into_iter().map(|e| e.into()).collect())
    }
}

impl<R: RealScalar> Deref for ParamList<R> {
    type Target = [ParamEntry<R>];

    fn deref(&self) -> &Self::Target {
        &self.list
    }
}
