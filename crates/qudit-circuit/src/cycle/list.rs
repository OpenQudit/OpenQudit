use std::ops::{Index, IndexMut};

use rustc_hash::FxHashMap;

use super::QuditCycle;
use super::CycleId;
use super::CycleIndex;

/// A list of cycles in a circuit.
///
/// Provides two ways to access the cycles both in O(1) time.
/// 1. Access by cycle index: An index based on the order of the cycles, and
///    get shifted around when operations are removed. They can change.
/// 2. Access by cycle id: These are unique identifiers for each cycle.
///    They never change once assigned to a cycle.
#[derive(Clone)]
pub struct CycleList {
    /// The raw vector of cycles accessed by physical indices.
    cycles: Vec<QuditCycle>,

    id_to_index: FxHashMap<CycleId, CycleIndex>,

    id_counter: u64,
}

impl CycleList {
    pub fn with_capacity(capacity: usize) -> CycleList {
        CycleList {
            cycles: Vec::with_capacity(capacity),
            id_to_index: FxHashMap::default(),
            id_counter: 0,
        }
    }

    fn get_new_id(&mut self) -> CycleId {
        let out = self.id_counter;
        if out >= std::u64::MAX - 1 {
            panic!("Cycle identifier overflow.");
        }
        self.id_counter += 1;
        CycleId(out)
    }

    pub fn push(&mut self) -> CycleIndex {
        let new_id = self.get_new_id();
        self.id_to_index.insert(new_id, self.cycles.len().into());
        self.cycles.push(QuditCycle::new(new_id));
        (self.cycles.len() - 1).into()
    }

    pub fn len(&self) -> usize {
        self.cycles.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn id_to_index(&self, id: CycleId) -> Option<CycleIndex> {
        self.id_to_index.get(&id).copied()
    }

    pub fn index_to_id(&self, idx: CycleIndex) -> CycleId {
        self.cycles[usize::from(idx)].id()
    }

    pub fn iter(&self) -> <&Vec<QuditCycle> as IntoIterator>::IntoIter {
        self.cycles.iter()
    }

    pub fn remove_index(&mut self, idx: CycleIndex) {
        let idx = usize::from(idx);

        if idx >= self.len() {
            panic!("Index out of bounds."); // TODO: Error handling.
        }

        let id = self.cycles[idx].id();
        self.id_to_index.remove(&id);
        self.cycles.remove(idx);
    }

    pub fn remove_id(&mut self, id: CycleId) {
        match self.id_to_index(id) {
            None => {}, // TODO: log a warning? 
            Some(idx) => self.remove_index(idx),
        }
    }

    pub fn get_from_id(&self, id: CycleId) -> Option<&QuditCycle> {
        self.id_to_index(id).map(|idx| &self.cycles[idx.0 as usize])
    }

    pub fn get_mut_from_id(&mut self, id: CycleId) -> Option<&mut QuditCycle> {
        self.id_to_index(id).map(|idx| &mut self.cycles[idx.0 as usize])
    }

    pub fn is_id(&self, id: CycleId) -> bool {
        self.id_to_index.get(&id).is_some()
    }
}

impl Index<usize> for CycleList {
    type Output = QuditCycle;

    fn index(&self, idx: usize) -> &Self::Output {
        &self.cycles[idx]
    }
}

impl IndexMut<usize> for CycleList {
    fn index_mut(&mut self, idx: usize) -> &mut Self::Output {
        &mut self.cycles[idx]
    }
}

impl Index<CycleIndex> for CycleList {
    type Output = QuditCycle;

    fn index(&self, idx: CycleIndex) -> &Self::Output {
        &self.cycles[usize::from(idx)]
    }
}

impl IndexMut<CycleIndex> for CycleList {
    fn index_mut(&mut self, idx: CycleIndex) -> &mut Self::Output {
        &mut self.cycles[usize::from(idx)]
    }
}

impl Index<CycleId> for CycleList {
    type Output = QuditCycle;

    fn index(&self, id: CycleId) -> &Self::Output {
        let idx = self.id_to_index(id).expect("Cycle id does not exist.");
        &self.cycles[usize::from(idx)]
    }
}

impl IndexMut<CycleId> for CycleList {
    fn index_mut(&mut self, id: CycleId) -> &mut Self::Output {
        let idx = self.id_to_index(id).expect("Cycle id does not exist.");
        &mut self.cycles[usize::from(idx)]
    }
}

impl std::fmt::Debug for CycleList {
    fn fmt(&self, fmt: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut debug_struct_fmt = fmt.debug_struct("CycleList");
        for (i, cycle) in self.cycles.iter().enumerate() {
            debug_struct_fmt.field(&format!("Cycle {}", i), &cycle);
        }
        debug_struct_fmt.finish()
    }
}

impl<'a> IntoIterator for &'a CycleList {
    type IntoIter = <&'a Vec<QuditCycle> as IntoIterator>::IntoIter;
    type Item = &'a QuditCycle;

    fn into_iter(self) -> Self::IntoIter {
        self.cycles.iter()
    }
}
