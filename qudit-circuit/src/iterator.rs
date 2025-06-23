use std::collections::HashSet;
use std::collections::VecDeque;

use itertools::Itertools;

use qudit_core::ComplexScalar;

use crate::instruction::Instruction;
use crate::instruction::InstructionReference;

use super::operation::Operation;
use super::point::CircuitPoint;
use super::point::DitOrBit;
use super::QuditCircuit;

// TODO: implement size_hints

pub struct QuditCircuitFastIterator<'a, C: ComplexScalar> {
    circuit: &'a QuditCircuit<C>,
    frontier: Vec<&'a InstructionReference>,
    next_cycle_index: usize,
}

impl<'a, C: ComplexScalar> QuditCircuitFastIterator<'a, C> {
    pub fn new(circuit: &'a QuditCircuit<C>) -> Self {
        Self {
            circuit: circuit,
            frontier: vec![],
            next_cycle_index: 0,
        }
    }
}

impl<'a, C: ComplexScalar> Iterator for QuditCircuitFastIterator<'a, C> {
    type Item = &'a InstructionReference;

    fn next(&mut self) -> Option<Self::Item> {
        if self.frontier.is_empty() {
            if self.next_cycle_index >= self.circuit.num_cycles() {
                return None;
            }
            let cycle = self.circuit.get_cycle(self.next_cycle_index);
            self.next_cycle_index += 1;

            for (i, inst) in cycle.insts.iter().enumerate() {
                if cycle.free.contains(&i) {
                    continue;
                }
                self.frontier.push(&inst);
            }
        }

        self.frontier.pop()
    }
}
// pub struct QuditCircuitFastIteratorWithCycles<'a, C: ComplexScalar> {
//     circuit: &'a QuditCircuit<C>,
//     frontier: Option<Vec<&'a Operation<C>>>,
//     next_cycle_index: usize,
// }

// impl<'a, C: ComplexScalar> QuditCircuitFastIteratorWithCycles<'a, C> {
//     pub fn new(circuit: &'a QuditCircuit<C>) -> Self {
//         Self {
//             circuit: circuit,
//             frontier: None,
//             next_cycle_index: 0,
//         }
//     }
// }

// impl<'a, C: ComplexScalar> Iterator
//     for QuditCircuitFastIteratorWithCycles<'a, C>
// {
//     type Item = (usize, &'a Operation<C>);

//     fn next(&mut self) -> Option<Self::Item> {
//         if self.frontier.is_none() {
//             if self.next_cycle_index >= self.circuit.num_cycles() {
//                 return None;
//             }
//             let mut frontier = Vec::new();
//             let cycle = self.circuit.get_cycle(self.next_cycle_index);
//             self.next_cycle_index += 1;

//             for (i, node) in cycle.nodes.iter().enumerate() {
//                 if cycle.free.contains(&i) {
//                     continue;
//                 }
//                 frontier.push(&node.op);
//             }

//             self.frontier = Some(frontier);
//         }

//         let frontier = self.frontier.as_mut().unwrap();

//         let to_return =
//             Some((self.next_cycle_index - 1, frontier.pop().unwrap()));

//         if frontier.is_empty() {
//             self.frontier = None;
//         } else {
//             self.frontier = Some(frontier.to_vec());
//         }

//         to_return
//     }
// }

pub struct QuditCircuitBFIterator<'a, C: ComplexScalar> {
    circuit: &'a QuditCircuit<C>,
    frontier: Option<Vec<&'a Operation>>,
    next_cycle_index: usize,
}

impl<'a, C: ComplexScalar> QuditCircuitBFIterator<'a, C> {
    pub fn new(circuit: &'a QuditCircuit<C>) -> Self {
        Self {
            circuit: circuit,
            frontier: None,
            next_cycle_index: 0,
        }
    }
}

impl<'a, C: ComplexScalar> Iterator for QuditCircuitBFIterator<'a, C> {
    type Item = &'a Operation;

    fn next(&mut self) -> Option<Self::Item> {
        todo!()
        // if self.frontier.is_none() {
        //     if self.next_cycle_index >= self.circuit.num_cycles() {
        //         return None;
        //     }

        //     let cycle = self.circuit.get_cycle(self.next_cycle_index);
        //     self.next_cycle_index += 1;

        //     let mut frontier = Vec::with_capacity(cycle.num_ops);
        //     let mut seen = HashSet::with_capacity(cycle.num_ops);

        //     cycle.qudit_map.iter().sorted().for_each(|(_, &i)| {
        //         if !seen.contains(&i) {
        //             seen.insert(i);
        //             frontier.push(&cycle.nodes[i].op);
        //         }
        //     });

        //     self.frontier = Some(frontier);
        // }

        // let frontier = self.frontier.as_mut().unwrap();

        // let to_return = frontier.pop();

        // if frontier.is_empty() {
        //     self.frontier = None;
        // } else {
        //     self.frontier = Some(frontier.to_vec());
        // }

        // to_return
    }
}

pub struct QuditCircuitDFIterator<'a, C: ComplexScalar> {
    circuit: &'a QuditCircuit<C>,
    frontier: VecDeque<CircuitPoint>,
    seen: HashSet<CircuitPoint>,
}

impl<'a, C: ComplexScalar> QuditCircuitDFIterator<'a, C> {
    pub fn new(circuit: &'a QuditCircuit<C>) -> Self {
        let mut iterator = Self {
            circuit: circuit,
            frontier: VecDeque::new(),
            seen: HashSet::new(),
        };

        for front_point in circuit.front().values() {
            iterator
                .frontier
                .push_back(iterator.standardize_point(*front_point));
        }

        iterator
    }

    pub fn standardize_point(&self, point: CircuitPoint) -> CircuitPoint {
        todo!()
        // let loc = self.circuit.get(point).location();
        // if loc.get_num_qudits() == 0 {
        //     CircuitPoint {
        //         cycle: point.cycle,
        //         dit_or_bit: DitOrBit::Clbit(loc.clbits()[0]),
        //     }
        // } else {
        //     CircuitPoint {
        //         cycle: point.cycle,
        //         dit_or_bit: DitOrBit::Qudit(loc.qudits()[0]),
        //     }
        // }
    }
}

impl<'a, C: ComplexScalar> Iterator for QuditCircuitDFIterator<'a, C> {
    type Item = &'a Operation;

    fn next(&mut self) -> Option<Self::Item> {
        todo!()
        // let point = match self.frontier.pop_front() {
        //     Some(p) => p,
        //     None => return None,
        // };

        // let nexts = QuditCircuit::next(self.circuit, point);
        // let nexts = nexts.values();

        // for next in nexts {
        //     let point = self.standardize_point(*next);
        //     if !self.seen.contains(&point) {
        //         self.seen.insert(point);
        //         self.frontier.push_front(point);
        //     }
        // }

        // Some(self.circuit.get(point))
    }
}
