use std::collections::BTreeMap;
use std::fmt;

use slotmap::SlotMap;

use super::INVALID_CYCLE_ID;
use super::InstId;
use super::CycleId;

use crate::instruction::Instruction;
use crate::Wire;
use crate::WireList;



#[derive(Clone)]
#[repr(align(128))]
pub struct QuditCycle {
    pub insts: SlotMap<InstId, Instruction>,
    pub reg_map: BTreeMap<Wire, InstId>,
    pub dag_ptrs: BTreeMap<Wire, (CycleId, CycleId)>,
    pub id: CycleId,
}

impl QuditCycle {
    pub fn new(id: CycleId) -> QuditCycle {
        QuditCycle {
            insts: SlotMap::with_key(),
            reg_map: BTreeMap::new(),
            dag_ptrs: BTreeMap::new(),
            id,
        }
    }

    #[inline]
    pub fn id(&self) -> CycleId {
        self.id
    }

    #[inline]
    pub fn num_ops(&self) -> usize {
        self.insts.len()
    }

    pub fn get(&self, wire: Wire) -> Option<&Instruction> {
        self.reg_map.get(&wire).map(|&id| self.insts.get(id).unwrap())
    }

    pub fn get_mut(&mut self, wire: Wire) -> Option<&mut Instruction> {
        self.reg_map.get(&wire).map(|&id| self.insts.get_mut(id).unwrap())
    }

    pub fn push(&mut self, inst: Instruction) -> InstId {
        let inst_id = self.insts.insert(inst.clone());

        for wire in inst.wires().wires() {
            debug_assert!(self.reg_map.get(&wire).is_none());
            self.reg_map.insert(wire, inst_id);
        }

        inst_id
    }

    pub fn remove(&mut self, wire: Wire) -> Option<Instruction> {
        let inst_id = match self.reg_map.get(&wire) {
            None => return None,
            Some(inst_id) => inst_id,
        };

        let inst = self.insts.remove(*inst_id);

        if let Some(inst_ref) = &inst {
            for wire in inst_ref.wires().wires() {
                self.reg_map.remove(&wire);
                self.dag_ptrs.remove(&wire);
            }
        }

        inst
    }

//     pub fn get_location(&self, wire: Wire) -> Option<&CircuitLocation> {
//         let inst_id = match self.reg_map.get(&wire) {
//             None => return None,
//             Some(inst_id) => inst_id,
//         };
        
//         self.get_location_from_id(*inst_id)
//     }

    // #[inline]
    // pub fn get_location_from_id(&self, inst_id: InstId) -> Option<&CircuitLocation> {
    //     self.insts.get(inst_id).map(|inst_ref| &inst_ref.location)
    // }

    pub fn set_next(&mut self, wire: Wire, next_cycle_id: CycleId) {
        self.dag_ptrs.entry(wire)
            .and_modify(|(_, n)| *n = next_cycle_id)
            .or_insert((INVALID_CYCLE_ID, next_cycle_id));
    }

    pub fn set_prev(&mut self, wire: Wire, prev_cycle_id: CycleId) {
        self.dag_ptrs.entry(wire)
            .and_modify(|(p, _)| *p = prev_cycle_id)
            .or_insert((prev_cycle_id, INVALID_CYCLE_ID));
    }

    pub fn get_next(&mut self, wire: Wire) -> Option<CycleId> {
        self.dag_ptrs.get(&wire).map(|(_, n)| *n)
    }

    pub fn get_prev(&mut self, wire: Wire) -> Option<CycleId> {
        self.dag_ptrs.get(&wire).map(|(p, _)| *p)
    }

    pub fn is_empty(&self) -> bool {
        self.num_ops() == 0
    }

    pub fn iter(&self) -> impl Iterator<Item = &Instruction> + '_ {
        self.insts.iter().map(|(_, inst)| inst)
    }

    pub fn iter_with_keys(&self) -> impl Iterator<Item = (InstId, &Instruction)> + '_ {
        self.insts.iter()
    }
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

