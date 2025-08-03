use std::ops::{Index, IndexMut};

use crate::cycle::QuditCycle;

/// A list of cycles in a circuit.
///
/// Provides two ways to access the cycles both in O(1) time.
/// 1. Access by logical index: These are the indices that the user sees, and get
///    shifted around when operations are removed. They can change.
/// 2. Access by physical index: These are the indices that the circuit uses
///    internally to store DAG information, and do not change.
///
/// The cost of maintaining these O(1) lookups is an expensive remove operation.
#[derive(Clone)]
pub(super) struct CycleList {
    /// The number of logical active cycles in the list.
    num_cycles: usize,

    /// The raw vector of cycles accessed by physical indices.
    cycles: Vec<QuditCycle>,

    /// The list of physical indices are that are inactive.
    free: Vec<usize>,

    /// A map from logical cycle index to physical indices.
    logical_to_physical: Vec<usize>,
}

// TODO: Implement detection of good capacity for cycle's op vectors?

impl CycleList {
    pub(super) fn with_capacity(capacity: usize) -> CycleList {
        CycleList {
            num_cycles: 0,
            cycles: Vec::with_capacity(capacity),
            free: Vec::new(),
            logical_to_physical: Vec::with_capacity(capacity),
        }
    }

    pub(super) fn push(&mut self) -> usize {
        let logical_index = self.num_cycles;
        self.num_cycles += 1;

        if self.free.is_empty() {
            let physical_index = self.cycles.len();
            self.cycles.push(QuditCycle::new(logical_index));
            self.logical_to_physical.push(physical_index);
            physical_index
        } else {
            let physical_index = self.free.pop().unwrap();
            self.cycles[physical_index].zero();
            self.logical_to_physical.push(physical_index);
            physical_index
        }
    }

    pub(super) fn len(&self) -> usize {
        self.num_cycles
    }

    pub(super) fn is_empty(&self) -> bool {
        self.num_cycles == 0
    }

    pub(super) fn map_physical_to_logical_idx(&self, idx: usize) -> usize {
        self.cycles[idx].logical_index
    }

    pub(super) fn map_logical_to_physical_idx(&self, idx: usize) -> usize {
        self.logical_to_physical[idx]
    }

    // #[allow(dead_code)]
    // pub(super) fn get(&self, physical_index: usize) -> &QuditCycle {
    //     &self.cycles[physical_index]
    // }

    // #[allow(dead_code)]
    // pub(super) fn get_mut(
    //     &mut self,
    //     physical_index: usize,
    // ) -> &mut QuditCycle {
    //     &mut self.cycles[physical_index]
    // }

    pub(super) fn iter(&self) -> CycleListIter {
        CycleListIter::new(self)
    }

    pub(super) fn remove(&mut self, logical_index: usize) {
        if logical_index >= self.num_cycles {
            panic!("Index out of bounds."); // TODO: Error handling.
        }

        for other_logical_index in (logical_index + 1)..self.num_cycles {
            self[other_logical_index].logical_index -= 1;
        }

        let physical_index = self.logical_to_physical[logical_index];
        self.logical_to_physical.remove(logical_index);
        self.free.push(physical_index);
        self.num_cycles -= 1;
    }
}

impl Index<usize> for CycleList {
    type Output = QuditCycle;

    fn index(&self, logical_index: usize) -> &Self::Output {
        &self.cycles[self.logical_to_physical[logical_index]]
    }
}

impl IndexMut<usize> for CycleList {
    fn index_mut(&mut self, logical_index: usize) -> &mut Self::Output {
        &mut self.cycles[self.logical_to_physical[logical_index]]
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
    type IntoIter = CycleListIter<'a>;
    type Item = &'a QuditCycle;

    fn into_iter(self) -> Self::IntoIter {
        CycleListIter::new(self)
    }
}


pub(super) struct CycleListIter<'a> {
    list: &'a CycleList,
    next_index: usize,
}

impl<'a> CycleListIter<'a> {
    fn new(list: &'a CycleList) -> Self {
        Self {
            list,
            next_index: 0,
        }
    }
}

impl<'a> Iterator for CycleListIter<'a> {
    type Item = &'a QuditCycle;

    fn next(&mut self) -> Option<Self::Item> {
        if self.next_index >= self.list.num_cycles {
            return None;
        }

        let to_return = &self.list[self.next_index];
        self.next_index += 1;
        Some(to_return)
    }
}
