use std::collections::BTreeMap;
use std::collections::HashSet;
use std::fmt;
use std::ops::Index;
use std::ops::IndexMut;

use crate::math::ComplexScalar;

use super::node::OperationNode;
use super::point::DitOrBit;

#[derive(Clone)]
pub(super) struct QuditCycle<C: ComplexScalar> {
    pub num_ops: usize,
    pub nodes: Vec<OperationNode<C>>,
    pub qudit_map: BTreeMap<usize, usize>, // QuditIndex -> OperationIndex
    pub clbit_map: BTreeMap<usize, usize>, // ClbitIndex -> OperationIndex
    pub free: HashSet<usize>,              // OperationIndex
    pub logical_index: usize,
}

impl<C: ComplexScalar> QuditCycle<C> {
    pub(super) fn new(logical_index: usize) -> QuditCycle<C> {
        QuditCycle {
            num_ops: 0,
            nodes: Vec::new(),
            qudit_map: BTreeMap::new(),
            clbit_map: BTreeMap::new(),
            free: HashSet::new(), // TODO: benchmark BTreeSet instead
            logical_index: logical_index,
        }
    }

    pub(super) fn zero(&mut self) {
        self.num_ops = 0;
        self.nodes.clear();
        self.qudit_map.clear();
        self.clbit_map.clear();
        self.free.clear();
    }

    #[inline]
    fn get_physical_index(&self, dit_or_bit: DitOrBit) -> Option<usize> {
        match dit_or_bit {
            DitOrBit::Qudit(qudit_index) => self.qudit_map.get(&qudit_index),
            DitOrBit::Clbit(clbit_index) => self.clbit_map.get(&clbit_index),
        }
        .cloned()
    }

    pub(super) fn get(
        &self,
        dit_or_bit: DitOrBit,
    ) -> Option<&OperationNode<C>> {
        let physical_index = self.get_physical_index(dit_or_bit);

        match physical_index {
            Some(i) => Some(&self.nodes[i]),
            None => None,
        }
    }

    pub(super) fn get_mut(
        &mut self,
        dit_or_bit: DitOrBit,
    ) -> Option<&mut OperationNode<C>> {
        let physical_index = self.get_physical_index(dit_or_bit);

        match physical_index {
            Some(i) => Some(&mut self.nodes[i]),
            None => None,
        }
    }

    pub(super) fn get_all_indices(&self) -> Vec<DitOrBit> {
        let mut indices = Vec::with_capacity(self.num_ops);
        for (i, node) in self.nodes.iter().enumerate() {
            if self.free.contains(&i) {
                continue;
            }
            if node.op.location().get_num_qudits() == 0 {
                let clbit = node.op.location().clbits()[0];
                indices.push(DitOrBit::Clbit(clbit));
            } else {
                let qudit = node.op.location().qudits()[0];
                indices.push(DitOrBit::Qudit(qudit));
            }
        }
        indices
    }

    pub(super) fn push(&mut self, node: OperationNode<C>) -> usize {
        self.num_ops += 1;

        let physical_index = if self.free.is_empty() {
            self.nodes.push(node);
            self.nodes.len() - 1
        } else {
            let physical_index = self.free.iter().next().unwrap().clone();
            self.free.remove(&physical_index);
            self.nodes[physical_index] = node;
            physical_index
        };

        for qudit_index in self.nodes[self.nodes.len() - 1]
            .op
            .location()
            .qudits()
            .iter()
        {
            debug_assert!(self.qudit_map.get(qudit_index).is_none());
            self.qudit_map.insert(*qudit_index, physical_index);
        }

        for clbit_index in self.nodes[self.nodes.len() - 1]
            .op
            .location()
            .clbits()
            .iter()
        {
            debug_assert!(self.clbit_map.get(clbit_index).is_none());
            self.clbit_map.insert(*clbit_index, physical_index);
        }

        physical_index
    }

    pub(super) fn remove(&mut self, dit_or_bit: DitOrBit) -> OperationNode<C> {
        let physical_index = self.get_physical_index(dit_or_bit);

        let physical_index = match physical_index {
            Some(i) => i,
            None => panic!("Operation not found"),
        };

        let location = self.nodes[physical_index].op.location();
        for qudit_index in location.qudits().iter() {
            self.qudit_map.remove(qudit_index);
        }
        for clbit_index in location.clbits().iter() {
            self.clbit_map.remove(clbit_index);
        }

        self.num_ops -= 1;
        self.free.insert(physical_index);
        self.nodes[physical_index].clone()
    }

    pub(super) fn is_empty(&self) -> bool {
        self.num_ops == 0
    }

    // pub(super) fn iter(&self) -> CycleIter<C> { CycleIter::new(self) }
}

impl<C: ComplexScalar> fmt::Debug for QuditCycle<C> {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut debug_struct_fmt = fmt.debug_struct("QuditCycle");
        for node in &self.nodes {
            debug_struct_fmt.field("op", &node.op);
        }
        debug_struct_fmt.finish()
    }
}

/// A list of cycles in a circuit.
/// Provides two ways to access the cycles both in O(1) time.
/// 1. Access by logical index These are the indices that the user sees, and get
///    shifted around when operations are removed. They can change.
/// 2. Access by physical index These are the indices that the circuit uses
///    internally, and do not change when operations are removed.
#[derive(Clone)]
pub(super) struct CycleList<C: ComplexScalar> {
    num_cycles: usize,
    cycles: Vec<QuditCycle<C>>,
    free: HashSet<usize>, // TODO: benchmark BTreeSet instead
    logical_to_phsyical: Vec<usize>,
}

// TODO: Implement detection of good capacity for cycle's op vectors?

impl<C: ComplexScalar> CycleList<C> {
    pub(super) fn with_capacity(capacity: usize) -> CycleList<C> {
        CycleList {
            num_cycles: 0,
            cycles: Vec::with_capacity(capacity),
            free: HashSet::new(),
            logical_to_phsyical: Vec::with_capacity(capacity),
        }
    }

    pub(super) fn push(&mut self) -> usize {
        let logical_index = self.num_cycles;
        self.num_cycles += 1;

        if self.free.is_empty() {
            let physical_index = self.cycles.len();
            self.cycles.push(QuditCycle::new(logical_index));
            self.logical_to_phsyical.push(physical_index);
            physical_index
        } else {
            let physical_index = self.free.iter().next().unwrap().clone();
            self.free.remove(&physical_index);
            self.cycles[physical_index].zero();
            self.logical_to_phsyical.push(physical_index);
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
        self.logical_to_phsyical[idx]
    }

    pub(super) fn get(&self, physical_index: usize) -> &QuditCycle<C> {
        &self.cycles[physical_index]
    }

    pub(super) fn get_mut(
        &mut self,
        physical_index: usize,
    ) -> &mut QuditCycle<C> {
        &mut self.cycles[physical_index]
    }

    pub(super) fn iter(&self) -> CycleListIter<C> {
        CycleListIter::new(self)
    }

    pub(super) fn remove(&mut self, physical_index: usize) {
        if physical_index >= self.cycles.len() {
            panic!("Index out of bounds.");
        }

        let cycle = &mut self.cycles[physical_index];
        let logical_index = cycle.logical_index;

        match self.logical_to_phsyical.get(logical_index) {
            Some(&i) => {
                if i != physical_index {
                    panic!("Cycle doesn't exist.");
                }
            },
            None => panic!("Cycle doesn't exist."),
        }

        self.logical_to_phsyical.remove(logical_index);
        self.free.insert(physical_index);
        self.num_cycles -= 1;
        cycle.zero();
    }
}

impl<C: ComplexScalar> Index<usize> for CycleList<C> {
    type Output = QuditCycle<C>;

    fn index(&self, logical_index: usize) -> &Self::Output {
        &self.cycles[self.logical_to_phsyical[logical_index]]
    }
}

impl<C: ComplexScalar> IndexMut<usize> for CycleList<C> {
    fn index_mut(&mut self, logical_index: usize) -> &mut Self::Output {
        &mut self.cycles[self.logical_to_phsyical[logical_index]]
    }
}

impl<C: ComplexScalar> fmt::Debug for CycleList<C> {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut debug_struct_fmt = fmt.debug_struct("CycleList");
        for (i, cycle) in self.cycles.iter().enumerate() {
            debug_struct_fmt.field(&format!("Cycle {}", i), &cycle);
        }
        debug_struct_fmt.finish()
    }
}

impl<'a, C: ComplexScalar> IntoIterator for &'a CycleList<C> {
    type IntoIter = CycleListIter<'a, C>;
    type Item = &'a QuditCycle<C>;

    fn into_iter(self) -> Self::IntoIter {
        CycleListIter::new(self)
    }
}

impl<'a, C: ComplexScalar> IntoIterator for &'a QuditCycle<C> {
    type IntoIter = CycleIter<'a, C>;
    type Item = &'a OperationNode<C>;

    fn into_iter(self) -> Self::IntoIter {
        CycleIter::new(self)
    }
}

pub(super) struct CycleListIter<'a, C: ComplexScalar> {
    list: &'a CycleList<C>,
    next_index: usize,
}

impl<'a, C: ComplexScalar> CycleListIter<'a, C> {
    fn new(list: &'a CycleList<C>) -> Self {
        Self {
            list: list,
            next_index: 0,
        }
    }
}

impl<'a, C: ComplexScalar> Iterator for CycleListIter<'a, C> {
    type Item = &'a QuditCycle<C>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.next_index >= self.list.num_cycles {
            return None;
        }

        let to_return = &self.list[self.next_index];
        self.next_index += 1;
        Some(to_return)
    }
}

pub(super) struct CycleIter<'a, C: ComplexScalar> {
    cycle: &'a QuditCycle<C>,
    next_index: usize,
}

impl<'a, C: ComplexScalar> CycleIter<'a, C> {
    fn new(cycle: &'a QuditCycle<C>) -> Self {
        Self {
            cycle: cycle,
            next_index: 0,
        }
    }
}

impl<'a, C: ComplexScalar> Iterator for CycleIter<'a, C> {
    type Item = &'a OperationNode<C>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if !self.cycle.free.contains(&self.next_index) {
                break;
            }
            self.next_index += 1;
        }

        if self.next_index >= self.cycle.nodes.len() {
            return None;
        }

        let to_return = &self.cycle.nodes[self.next_index];
        self.next_index += 1;
        Some(to_return)
    }
}
