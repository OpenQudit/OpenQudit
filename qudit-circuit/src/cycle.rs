use std::collections::BTreeMap;
use std::fmt;

use crate::location::CircuitLocation;
use crate::instruction::InstructionReference;

use super::point::DitOrBit;

pub const INVALID_INDEX: usize = usize::MAX;

#[derive(Clone)]
pub(super) struct QuditCycle {
    pub insts: Vec<InstructionReference>,
    pub qudit_map: BTreeMap<usize, usize>, // QuditIndex -> OperationIndex
    pub clbit_map: BTreeMap<usize, usize>, // ClbitIndex -> OperationIndex
    pub dag_ptrs: BTreeMap<DitOrBit, (usize, usize)>,
    pub free: Vec<usize>,              // OperationIndex
    pub logical_index: usize,
}

impl QuditCycle {
    pub fn new(logical_index: usize) -> QuditCycle {
        QuditCycle {
            insts: Vec::new(),
            qudit_map: BTreeMap::new(),
            clbit_map: BTreeMap::new(),
            dag_ptrs: BTreeMap::new(),
            free: Vec::new(),
            logical_index,
        }
    }

    pub fn zero(&mut self) {
        self.insts.clear();
        self.qudit_map.clear();
        self.clbit_map.clear();
        self.dag_ptrs.clear();
        self.free.clear();
        self.logical_index = INVALID_INDEX;
    }

    #[inline]
    pub fn num_ops(&self) -> usize {
        self.insts.len() - self.free.len()
    }

    #[inline]
    fn get_physical_index(&self, dit_or_bit: DitOrBit) -> Option<usize> {
        match dit_or_bit {
            DitOrBit::Qudit(qudit_index) => self.qudit_map.get(&qudit_index).copied(),
            DitOrBit::Clbit(clbit_index) => self.clbit_map.get(&clbit_index).copied(),
        }
    }

    #[allow(dead_code)]
    pub fn get(&self, dit_or_bit: DitOrBit) -> Option<&InstructionReference> {
        let physical_index = self.get_physical_index(dit_or_bit);

        match physical_index {
            Some(i) => Some(&self.insts[i]),
            None => None,
        }
    }

    #[allow(dead_code)]
    pub fn get_mut(&mut self, dit_or_bit: DitOrBit) -> Option<&InstructionReference> {
        let physical_index = self.get_physical_index(dit_or_bit);

        match physical_index {
            Some(i) => Some(&mut self.insts[i]),
            None => None,
        }
    }

    pub fn push(&mut self, inst: InstructionReference) -> usize {
        let physical_index = if self.free.is_empty() {
            self.insts.push(inst);
            self.insts.len() - 1
        } else {
            let physical_index = self.free.pop().expect("Free list should not be empty");
            self.insts[physical_index] = inst;
            physical_index
        };
        let inst = &self.insts[physical_index];

        for qudit_index in inst
            .location
            .qudits()
            .iter()
        {
            debug_assert!(self.qudit_map.get(&qudit_index).is_none());
            self.qudit_map.insert(qudit_index, physical_index);
        }

        for clbit_index in inst
            .location
            .clbits()
            .iter()
        {
            debug_assert!(self.clbit_map.get(&clbit_index).is_none());
            self.clbit_map.insert(clbit_index, physical_index);
        }

        physical_index
    }

    pub fn remove(&mut self, dit_or_bit: DitOrBit) -> Option<InstructionReference> {
        let physical_index = self.get_physical_index(dit_or_bit);

        let physical_index = match physical_index {
            Some(i) => i,
            // None => panic!("Operation not found at {} in cycle {}", dit_or_bit, self.logical_index),
            None => return None,
        };

        let location = &self.insts[physical_index].location;
        for qudit_index in location.qudits().iter() {
            self.qudit_map.remove(&qudit_index);
        }
        for clbit_index in location.clbits().iter() {
            self.clbit_map.remove(&clbit_index);
        }

        self.free.push(physical_index);
        Some(self.insts[physical_index].to_owned())  // can probably be quicker with a directly memcpy and
        // capacity set to 0
    }

    pub fn get_location(&self, dit_or_bit: DitOrBit) -> Option<&CircuitLocation> {
        let physical_index = self.get_physical_index(dit_or_bit);

        match physical_index {
            Some(i) => Some(&self.insts[i].location),
            None => None,
        }
    }

    pub fn set_qnext(&mut self, qudit_index: usize, next_cycle: usize) {
        let key = DitOrBit::Qudit(qudit_index);
        match self.dag_ptrs.get_mut(&key) {
            Some(ptr) => {
                ptr.1 = next_cycle;
            },
            None => {
                self.dag_ptrs.insert(key, (INVALID_INDEX, next_cycle));
            },
        }
    }

    pub fn reset_qnext(&mut self, qudit_index: usize) {
        let key = DitOrBit::Qudit(qudit_index);
        match self.dag_ptrs.get_mut(&key) {
            Some(ptr) => {
                ptr.1 = INVALID_INDEX;
            },
            None => (),
        }
    }

    pub fn get_qnext(&self, qudit_index: usize) -> Option<usize> {
        let key = DitOrBit::Qudit(qudit_index);
        match self.dag_ptrs.get(&key) {
            Some(ptr) => {
                if ptr.1 == INVALID_INDEX {
                    None
                } else {
                    Some(ptr.1)
                }
            },
            None => {
                None
            },
        }
    }

    pub fn set_cnext(&mut self, clbit_index: usize, next_cycle: usize) {
        let key = DitOrBit::Clbit(clbit_index);
        match self.dag_ptrs.get_mut(&key) {
            Some(ptr) => {
                ptr.1 = next_cycle;
            },
            None => {
                self.dag_ptrs.insert(key, (INVALID_INDEX, next_cycle));
            },
        }
    }

    pub fn reset_cnext(&mut self, clbit_index: usize) {
        let key = DitOrBit::Clbit(clbit_index);
        match self.dag_ptrs.get_mut(&key) {
            Some(ptr) => {
                ptr.1 = INVALID_INDEX;
            },
            None => (),
        }
    }

    pub fn get_cnext(&self, clbit_index: usize) -> Option<usize> {
        let key = DitOrBit::Clbit(clbit_index);
        match self.dag_ptrs.get(&key) {
            Some(ptr) => {
                if ptr.1 == INVALID_INDEX {
                    None
                } else {
                    Some(ptr.1)
                }
            },
            None => {
                None
            },
        }
    }

    pub fn set_qprev(&mut self, qudit_index: usize, prev_cycle: usize) {
        let key = DitOrBit::Qudit(qudit_index);
        match self.dag_ptrs.get_mut(&key) {
            Some(ptr) => {
                ptr.0 = prev_cycle;
            },
            None => {
                self.dag_ptrs.insert(key, (prev_cycle, INVALID_INDEX));
            },
        }
    }

    pub fn reset_qprev(&mut self, qudit_index: usize) {
        let key = DitOrBit::Qudit(qudit_index);
        match self.dag_ptrs.get_mut(&key) {
            Some(ptr) => {
                ptr.0 = INVALID_INDEX;
            },
            None => (),
        }
    }

    pub fn get_qprev(&self, qudit_index: usize) -> Option<usize> {
        let key = DitOrBit::Qudit(qudit_index);
        match self.dag_ptrs.get(&key) {
            Some(ptr) => {
                if ptr.0 == INVALID_INDEX {
                    None
                } else {
                    Some(ptr.0)
                }
            },
            None => {
                None
            },
        }
    }

    pub fn set_cprev(&mut self, clbit_index: usize, prev_cycle: usize) {
        let key = DitOrBit::Clbit(clbit_index);
        match self.dag_ptrs.get_mut(&key) {
            Some(ptr) => {
                ptr.0 = prev_cycle;
            },
            None => {
                self.dag_ptrs.insert(key, (prev_cycle, INVALID_INDEX));
            },
        }
    }

    pub fn reset_cprev(&mut self, clbit_index: usize) {
        let key = DitOrBit::Clbit(clbit_index);
        match self.dag_ptrs.get_mut(&key) {
            Some(ptr) => {
                ptr.0 = INVALID_INDEX;
            },
            None => (),
        }
    }

    pub fn get_cprev(&self, clbit_index: usize) -> Option<usize> {
        let key = DitOrBit::Clbit(clbit_index);
        match self.dag_ptrs.get(&key) {
            Some(ptr) => {
                if ptr.0 == INVALID_INDEX {
                    None
                } else {
                    Some(ptr.0)
                }
            },
            None => {
                None
            },
        }
    }

    pub(super) fn is_empty(&self) -> bool {
        self.num_ops() == 0
    }

    #[allow(dead_code)]
    pub(super) fn iter(&self) -> CycleIter { CycleIter::new(self) }
}

impl fmt::Debug for QuditCycle {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut debug_struct_fmt = fmt.debug_struct("QuditCycle");
        for inst in &self.insts {
            debug_struct_fmt.field("inst", &inst);
        }
        debug_struct_fmt.finish()
    }
}

impl<'a> IntoIterator for &'a QuditCycle {
    type IntoIter = CycleIter<'a>;
    type Item = &'a InstructionReference;

    fn into_iter(self) -> Self::IntoIter {
        CycleIter::new(self)
    }
}

pub(super) struct CycleIter<'a> {
    cycle: &'a QuditCycle,
    next_index: usize,
}

impl<'a> CycleIter<'a> {
    fn new(cycle: &'a QuditCycle) -> Self {
        Self {
            cycle: cycle,
            next_index: 0,
        }
    }
}

impl<'a> Iterator for CycleIter<'a> {
    type Item = &'a InstructionReference;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if !self.cycle.free.contains(&self.next_index) {
                break;
            }
            self.next_index += 1;
        }

        if self.next_index >= self.cycle.insts.len() {
            return None;
        }

        let to_return = &self.cycle.insts[self.next_index];
        self.next_index += 1;
        Some(to_return)
    }
}
