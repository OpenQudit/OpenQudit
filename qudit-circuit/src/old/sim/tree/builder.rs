use std::collections::HashMap;
use std::collections::HashSet;
use std::collections::VecDeque;

use itertools::Itertools;

use super::contract::ContractNode;
use super::kron::KronNode;
use super::mul::MulNode;
use super::perm::PermNode;
use super::tree::ExpressionTree;
use crate::circuit::operation::Operation;
use crate::circuit::CircuitLocation;
use crate::math::ComplexScalar;
use crate::QuditPermutation;
use crate::QuditSystem;

/// A node in a DAG of comp tree nodes
#[derive(Debug)]
pub struct NodeDAGNode {
    pub node: ExpressionTree,
    pub location: CircuitLocation,
    pub next: Vec<Option<usize>>,
    pub prev: Vec<Option<usize>>,
}

/// A builder for a computation tree.
/// This builder is used to build a computation tree from a circuit.
#[derive(Debug)]
pub struct TreeBuilder {
    /// The number of qudits in the circuit.
    num_qudits: usize,

    /// Map from node index to node encoding a DAG of nodes.
    dag: HashMap<usize, NodeDAGNode>,

    /// The index of the next node to be added to the tree.
    index_counter: usize,
}

impl TreeBuilder {
    /// Create a new tree builder from a circuit.
    ///
    /// # Arguments
    ///
    /// * `circuit` - The circuit to build a tree from.
    ///
    /// # Returns
    ///
    /// A new tree builder.
    pub fn new<C: ComplexScalar>(
        num_qudits: usize,
        op_list: Vec<&Operation<C>>,
        next_list: Vec<Vec<Option<usize>>>,
        prev_list: Vec<Vec<Option<usize>>>,
    ) -> TreeBuilder {
        // TODO: From circuit to operator requires multiple passes over the
        // circuit before the op tree is even being constructed, we can
        // probably do better TODO: handle empty op_list
        let mut dag = HashMap::new();
        let num_ops = op_list.len();
        let next_prev_list = next_list
            .into_iter()
            .zip(prev_list.into_iter())
            .collect::<Vec<_>>();

        // Add all circuit operations to the DAG as leafs or permuted leafs
        // for (idx, (cycle, op)) in circuit.iter_with_cycles().enumerate() {
        for (idx, (op, (next_list, prev_list))) in op_list
            .into_iter()
            .zip(next_prev_list.into_iter())
            .enumerate()
        {
            let leaf =
                ExpressionTree::Leaf(op.instruction().get_gate().clone());
            let node = if op.location().is_sorted() {
                NodeDAGNode {
                    node: leaf,
                    location: op.location().clone(),
                    next: next_list,
                    prev: prev_list,
                }
            } else {
                let perm = QuditPermutation::locally_invert_location(
                    op.get_radices(),
                    &op.location(),
                );
                NodeDAGNode {
                    node: ExpressionTree::Perm(PermNode::new(leaf, perm)),
                    location: CircuitLocation::pure(
                        op.location()
                            .qudits()
                            .into_iter()
                            .sorted()
                            .map(|&f| f)
                            .collect(),
                    ),
                    next: next_list,
                    prev: prev_list,
                }
            };

            dag.insert(idx, node);
        }

        TreeBuilder {
            num_qudits: num_qudits,
            dag: dag,
            index_counter: num_ops,
        }
    }

    pub fn dag_ordered_iter(
        &self,
    ) -> impl Iterator<Item = (&usize, &NodeDAGNode)> {
        let mut dag_vec: Vec<(&usize, &NodeDAGNode)> =
            self.dag.iter().collect();
        dag_vec.sort_by(|(a_idx, a_node), (b_idx, b_node)| {
            // a_node.node.get_num_qudits().cmp(&b_node.node.get_num_qudits()).then_with(
            //     || a_idx.cmp(b_idx),
            // )
            a_idx.cmp(b_idx).then_with(|| {
                a_node
                    .node
                    .get_num_qudits()
                    .cmp(&b_node.node.get_num_qudits())
            })
        });
        dag_vec.into_iter()
    }

    /// Build the computation tree.
    pub fn build_tree(mut self) -> ExpressionTree {
        // First step is to multiply everything possible.
        // This while ensure there are no trivially combinable nodes.
        self.multiply_all_possible();

        // Sequence of n rounds
        // After round i, all nodes are joint-but-disjoint by at least i+1
        for disjoint_size in 1..=self.num_qudits {
            // Look for easy kron nodes that directly lead to multiplication.
            // Limit each of the nodes' size to be at most disjoint_size
            // to avoid degenerate cases.
            let kron_flag = self.pairwise_kron_towards_multiply(disjoint_size);

            // If we found a kron node, then we need to multiply again.
            if kron_flag {
                self.multiply_all_possible();
            }

            // Contract all nodes that are disjoint by at most disjoint_size.
            // After calling this function all nodes will with not be disjoint,
            // or be disjoint by at least disjoint_size + 1.
            self.contract_all(disjoint_size);

            // Multiply all nodes that can be multiplied.
            self.multiply_all_possible();
        }

        // If there are still disjoint graphs, then we need to handle them.
        // This can only arise with full separately systems of qudits.
        // Therefore, we kronecker all of them together.
        if self.dag.len() != 1 {
            self.kron_all_completely_disjoint();
            // TODO: enlarge
        }

        // Finally, we should have a single node left in the DAG.
        assert!(self.dag.len() == 1);

        for (_, v) in self.dag.drain().take(1) {
            return v.node;
        }

        panic!("Should never reach here");
    }

    /// Multiply all nodes that can be simply multiplied together.
    fn multiply_all_possible(&mut self) {
        loop {
            let num_nodes = self.dag.len();
            self.multiply_all_possible_single_step();
            if num_nodes == self.dag.len() {
                break;
            }
        }
    }

    fn multiply_all_possible_single_step(&mut self) {
        // Only need to check previous,
        // because one gate can multiply with its previous
        // iff that one can multiply with this one as its next
        let mut mul_pairs = Vec::new();
        let mut already_in_mul_this_round = HashSet::new();
        // Find all gates that can multiply with their previous
        for (idx, node) in self.dag_ordered_iter() {
            if already_in_mul_this_round.contains(idx) {
                continue;
            }
            // Can multiply with previous, if
            // 1. This gate only has one previous gate
            // 2. Both gates have the same location
            // 3. Previous gate is not already in a multiply pair
            let prevs: HashSet<usize> = node
                .prev
                .iter()
                .filter(|idx| idx.is_some())
                .map(|idx| idx.unwrap())
                .collect();

            if prevs.len() == 1 {
                let prev = prevs.iter().next().unwrap().clone();
                if node.location == self.dag[&prev].location {
                    if !already_in_mul_this_round.contains(&prev) {
                        already_in_mul_this_round.insert(*idx);
                        already_in_mul_this_round.insert(prev);
                        mul_pairs.push((prev, *idx));
                    }
                }
            }
        }

        // Update dag by removing old nodes and adding a mul node.
        for (idx_left, idx_right) in mul_pairs.iter() {
            let ndn_left = self.dag.remove(idx_left).unwrap();
            let ndn_right = self.dag.remove(idx_right).unwrap();
            let new_node_id = self.index_counter;
            self.index_counter += 1;

            // Update circuit-right gate's next gate's prev to be new mul node
            for (loc_idx, next) in ndn_right.next.iter().enumerate() {
                if let Some(next_idx) = next {
                    let qudit_index = ndn_right.location.qudits()[loc_idx];
                    let next_ndn = &self.dag[next_idx];
                    let next_loc_index = next_ndn
                        .location
                        .qudits()
                        .iter()
                        .position(|&i| i == qudit_index)
                        .unwrap();
                    self.dag.get_mut(next_idx).unwrap().prev[next_loc_index] =
                        Some(new_node_id);
                }
            }

            // Update circuit-left gate's prev gate's next to be new mul node
            for (loc_idx, prev) in ndn_left.prev.iter().enumerate() {
                if let Some(prev_idx) = prev {
                    let qudit_index = ndn_left.location.qudits()[loc_idx];
                    let prev_ndn = &self.dag[prev_idx];
                    let prev_loc_index = prev_ndn
                        .location
                        .qudits()
                        .iter()
                        .position(|&i| i == qudit_index)
                        .unwrap();
                    self.dag.get_mut(prev_idx).unwrap().next[prev_loc_index] =
                        Some(new_node_id);
                }
            }

            // Insert new node
            let new_ndn = NodeDAGNode {
                node: ExpressionTree::Mul(MulNode::new(
                    ndn_left.node,
                    ndn_right.node,
                )),
                location: ndn_left.location,
                next: ndn_right.next,
                prev: ndn_left.prev,
            };
            assert!(self.dag.insert(new_node_id, new_ndn).is_none());
        }
    }

    /// Choose two nodes to kronecker if it is helpful.
    fn pairwise_kron_towards_multiply(&mut self, node_size: usize) -> bool {
        let mut kron_pairs = Vec::new();
        let mut already_in_kron_this_round = HashSet::new();

        for (idx, node) in self.dag_ordered_iter() {
            if node.node.get_num_qudits() > node_size {
                continue;
            }
            if already_in_kron_this_round.contains(idx) {
                continue;
            }
            let min_loc = node.location.qudits().iter().min().unwrap();
            let max_loc = node.location.qudits().iter().max().unwrap();

            // if I can kron with one of prev's next that will work towards
            // multiplying with prev
            let prevs: Vec<usize> = node
                .prev
                .iter()
                .filter(|idx| idx.is_some())
                .map(|idx| idx.unwrap())
                .collect();
            if prevs.len() == 1 {
                let prev = &self.dag[&prevs[0]];
                let remaining = prev.location.difference(&node.location);
                let prev_nexts: Vec<_> = prev
                    .next
                    .iter()
                    .filter(|idx| idx.is_some())
                    .map(|idx| idx.unwrap())
                    .collect();

                let mut best_idx = None;
                let mut best_size = None;
                for prev_next_idx in prev_nexts {
                    if already_in_kron_this_round.contains(&prev_next_idx) {
                        continue;
                    }

                    let prev_next = &self.dag[&prev_next_idx];
                    if prev_next
                        .location
                        .qudits()
                        .iter()
                        .any(|q_idx| !remaining.qudits().contains(q_idx))
                    {
                        continue;
                    }

                    if self.has_non_direct_dependency(prev_next_idx, *idx) {
                        continue;
                    }

                    if self.has_non_direct_dependency(*idx, prev_next_idx) {
                        continue;
                    }

                    if prev_next
                        .location
                        .qudits()
                        .iter()
                        .all(|q_idx| q_idx < min_loc)
                        || prev_next
                            .location
                            .qudits()
                            .iter()
                            .all(|q_idx| q_idx > max_loc)
                    {
                        if best_size.is_none()
                            || best_size.unwrap() < prev_next.location.len()
                        {
                            best_size = Some(prev_next.location.len());
                            best_idx = Some(prev_next_idx);
                        }
                    }
                }

                if let Some(kron_idx) = best_idx {
                    already_in_kron_this_round.insert(*idx);
                    already_in_kron_this_round.insert(kron_idx);
                    // TODO: Change to explicit min or add test
                    // This works now since all qudits are either
                    // less than or greater than all node's qudits
                    // so we can just check the first one.
                    // This is a footgun though, if we change the
                    // behavior of kron later.
                    if self.dag[&kron_idx].location.qudits()[0] < *min_loc {
                        kron_pairs.push((kron_idx, *idx));
                    } else {
                        kron_pairs.push((*idx, kron_idx));
                    }
                    continue;
                }
            }

            // Otherwise try to kron with one of next's prev that will work
            // towards multiplying with next
            let nexts: Vec<usize> = node
                .next
                .iter()
                .filter(|idx| idx.is_some())
                .map(|idx| idx.unwrap())
                .collect();
            if nexts.len() == 1 {
                let next = &self.dag[&nexts[0]];
                let remaining = next.location.difference(&node.location);
                let next_prevs: Vec<_> = next
                    .prev
                    .iter()
                    .filter(|idx| idx.is_some())
                    .map(|idx| idx.unwrap())
                    .collect();

                let mut best_idx = None;
                let mut best_size = None;
                for next_prev_idx in next_prevs {
                    if already_in_kron_this_round.contains(&next_prev_idx) {
                        continue;
                    }

                    let next_prev = &self.dag[&next_prev_idx];
                    if next_prev
                        .location
                        .qudits()
                        .iter()
                        .any(|q_idx| !remaining.qudits().contains(q_idx))
                    {
                        continue;
                    }

                    if self.has_non_direct_dependency(*idx, next_prev_idx) {
                        continue;
                    }

                    if self.has_non_direct_dependency(next_prev_idx, *idx) {
                        continue;
                    }

                    if next_prev
                        .location
                        .qudits()
                        .iter()
                        .all(|q_idx| q_idx < min_loc)
                        || next_prev
                            .location
                            .qudits()
                            .iter()
                            .all(|q_idx| q_idx > max_loc)
                    {
                        if best_size.is_none()
                            || best_size.unwrap() < next_prev.location.len()
                        {
                            best_size = Some(next_prev.location.len());
                            best_idx = Some(next_prev_idx);
                        }
                    }
                }

                if let Some(kron_idx) = best_idx {
                    already_in_kron_this_round.insert(*idx);
                    already_in_kron_this_round.insert(kron_idx);
                    // TODO: Change to explicit min or add test
                    // This works now since all qudits are either
                    // less than or greater than all node's qudits
                    // so we can just check the first one.
                    // This is a footgun though, if we change the
                    // behavior of kron later.
                    if self.dag[&kron_idx].location.qudits()[0] < *min_loc {
                        kron_pairs.push((kron_idx, *idx));
                    } else {
                        kron_pairs.push((*idx, kron_idx));
                    }
                    continue;
                }
            }
        }

        // Update dag by removing old nodes and adding a kron node.
        // Left and right here are tensor ordering.
        // This means the left one is the one with the smaller indices.
        for (idx_left, idx_right) in kron_pairs.iter() {
            let ndn_left = self.dag.remove(idx_left).unwrap();
            let ndn_right = self.dag.remove(idx_right).unwrap();
            let new_node_id = self.index_counter;
            self.index_counter += 1;

            // Update both nodes nexts and prevs
            for ndn in vec![&ndn_left, &ndn_right] {
                for (loc_idx, next) in ndn.next.iter().enumerate() {
                    if let Some(next_idx) = next {
                        let qudit_index = ndn.location.qudits()[loc_idx];
                        let next_ndn = &self.dag[next_idx];
                        let next_loc_index = next_ndn
                            .location
                            .qudits()
                            .iter()
                            .position(|&i| i == qudit_index)
                            .unwrap();
                        self.dag.get_mut(next_idx).unwrap().prev
                            [next_loc_index] = Some(new_node_id)
                    }
                }
                for (loc_idx, prev) in ndn.prev.iter().enumerate() {
                    if let Some(prev_idx) = prev {
                        let qudit_index = ndn.location.qudits()[loc_idx];
                        let prev_ndn = &self.dag[prev_idx];
                        let prev_loc_index = prev_ndn
                            .location
                            .qudits()
                            .iter()
                            .position(|&i| i == qudit_index)
                            .unwrap();
                        self.dag.get_mut(prev_idx).unwrap().next
                            [prev_loc_index] = Some(new_node_id)
                    }
                }
            }
            // Insert new node
            let new_ndn = NodeDAGNode {
                node: ExpressionTree::Kron(KronNode::new(
                    ndn_left.node,
                    ndn_right.node,
                )),
                location: ndn_left.location.union(&ndn_right.location),
                next: ndn_left
                    .next
                    .iter()
                    .chain(ndn_right.next.iter())
                    .cloned()
                    .collect(),
                prev: ndn_left
                    .prev
                    .iter()
                    .chain(ndn_right.prev.iter())
                    .cloned()
                    .collect(),
            };
            assert!(self.dag.insert(new_node_id, new_ndn).is_none());
        }

        kron_pairs.len() > 0
    }

    /// Contract all pairs of gates with at most `disjoint_size` mismatched
    /// qudits.
    fn contract_all(&mut self, disjoint_size: usize) {
        loop {
            let num_nodes = self.dag.len();
            self.contract_all_single_step(disjoint_size);
            if num_nodes == self.dag.len() {
                break;
            }
        }
    }

    fn contract_all_single_step(&mut self, disjoint_size: usize) {
        let mut candidate_contract_pairs = Vec::new();

        // Find all gates that can contract with their previous
        for (idx, node) in self.dag_ordered_iter() {
            // if already_in_contract_this_round.contains(idx) {
            //     continue;
            // }

            // let mut best_is_prev = false;
            // let mut best_idx = None;
            // let mut best_size = None;

            let prevs: Vec<usize> = node
                .prev
                .iter()
                .filter(|idx| idx.is_some())
                .map(|idx| idx.unwrap())
                .collect();

            for prev in prevs {
                // if already_in_contract_this_round.contains(&prev) {
                //     continue;
                // }

                let prev_node = &self.dag[&prev];
                let union = node.location.union(&prev_node.location);
                let intersect = node.location.intersect(&prev_node.location);
                let disjoint = union.difference(&intersect);

                if disjoint.len() > disjoint_size {
                    continue;
                }

                if self.has_non_direct_dependency(prev, *idx) {
                    continue;
                }

                candidate_contract_pairs.push((union.len(), prev, *idx));

                // if best_idx.is_none() || best_size.unwrap() > union.len() {
                //     best_idx = Some(prev);
                //     best_size = Some(union.len());
                //     best_is_prev = true;
                // }
            }

            // let nexts: Vec<usize> = node
            //     .next
            //     .iter()
            //     .filter(|idx| idx.is_some())
            //     .map(|idx| idx.unwrap())
            //     .collect();

            // for next in nexts {
            //     if already_in_contract_this_round.contains(&next) {
            //         continue;
            //     }

            //     let next_node = &self.dag[&next];
            //     let union = node.location.union(&next_node.location);
            //     let intersect = node.location.intersect(&next_node.location);
            //     let disjoint = union.difference(&intersect);

            //     if disjoint.len() > disjoint_size {
            //         continue;
            //     }

            //     if self.has_non_direct_dependency(*idx, next) {
            //         continue;
            //     }

            //     if best_idx.is_none() || best_size.unwrap() > union.len() {
            //         best_idx = Some(next);
            //         best_size = Some(union.len());
            //         best_is_prev = false;
            //     }
            // }

            // if let Some(b_idx) = best_idx {
            //     already_in_contract_this_round.insert(*idx);
            //     already_in_contract_this_round.insert(b_idx);
            //     if best_is_prev {
            //         contract_pairs.push((b_idx, *idx));
            //     } else {
            //         contract_pairs.push((*idx, b_idx));
            //     }
            // }
        }

        let mut contract_pairs = Vec::new();
        let mut already_in_contract_this_round = HashSet::new();

        candidate_contract_pairs
            .sort_by(|(a_size, _, _), (b_size, _, _)| a_size.cmp(b_size));

        for (_, idx_left, idx_right) in candidate_contract_pairs.iter() {
            if already_in_contract_this_round.contains(idx_left) {
                continue;
            }
            if already_in_contract_this_round.contains(idx_right) {
                continue;
            }

            already_in_contract_this_round.insert(*idx_left);
            already_in_contract_this_round.insert(*idx_right);
            contract_pairs.push((*idx_left, *idx_right));
            // break;  // TODO: Evaluate this break
        }

        // Update dag by removing old nodes and adding a contract node.
        for (idx_left, idx_right) in contract_pairs.iter() {
            let ndn_left = self.dag.remove(idx_left).unwrap();
            let ndn_right = self.dag.remove(idx_right).unwrap();
            let new_node_id = self.index_counter;
            self.index_counter += 1;

            for ndn in vec![&ndn_left, &ndn_right] {
                // Update the node's next's prev to be new contract node
                for (loc_idx, next) in ndn.next.iter().enumerate() {
                    if let Some(next_idx) = next {
                        if next_idx == idx_left || next_idx == idx_right {
                            continue;
                        }

                        let qudit_index = ndn.location.qudits()[loc_idx];
                        let next_ndn = &self.dag[next_idx];
                        let next_loc_index = next_ndn
                            .location
                            .qudits()
                            .iter()
                            .position(|&i| i == qudit_index)
                            .unwrap();
                        self.dag.get_mut(next_idx).unwrap().prev
                            [next_loc_index] = Some(new_node_id);
                    }
                }

                // Update the node's prev's next to be new contract node
                for (loc_idx, prev) in ndn.prev.iter().enumerate() {
                    if let Some(prev_idx) = prev {
                        if prev_idx == idx_left || prev_idx == idx_right {
                            continue;
                        }

                        let qudit_index = ndn.location.qudits()[loc_idx];
                        let prev_ndn = &self.dag[prev_idx];
                        let prev_loc_index = prev_ndn
                            .location
                            .qudits()
                            .iter()
                            .position(|&i| i == qudit_index)
                            .unwrap();
                        self.dag.get_mut(prev_idx).unwrap().next
                            [prev_loc_index] = Some(new_node_id);
                    }
                }
            }

            let new_location =
                ndn_left.location.union(&ndn_right.location).to_sorted();

            let mut new_prev = Vec::new();
            let mut new_next = Vec::new();

            for qudit_index in new_location.qudits().iter() {
                let mut prev = None;
                let mut next = None;

                let left_contains =
                    ndn_left.location.qudits().contains(&qudit_index);
                let right_contains =
                    ndn_right.location.qudits().contains(&qudit_index);
                assert!(left_contains || right_contains);

                if left_contains {
                    let loc_idx = ndn_left
                        .location
                        .qudits()
                        .iter()
                        .position(|&i| i == *qudit_index)
                        .unwrap();
                    prev = ndn_left.prev[loc_idx];
                    if !right_contains {
                        next = ndn_left.next[loc_idx];
                    }
                }

                if ndn_right.location.qudits().contains(&qudit_index) {
                    let loc_idx = ndn_right
                        .location
                        .qudits()
                        .iter()
                        .position(|&i| i == *qudit_index)
                        .unwrap();
                    if !left_contains {
                        prev = ndn_right.prev[loc_idx];
                    }
                    next = ndn_right.next[loc_idx];
                }

                new_prev.push(prev);
                new_next.push(next);
            }

            // Insert new node
            let new_ndn = NodeDAGNode {
                node: ExpressionTree::Contract(ContractNode::new(
                    ndn_left.node,
                    ndn_right.node,
                    ndn_left.location.qudits().to_vec(),
                    ndn_right.location.qudits().to_vec(),
                )),
                location: new_location,
                next: new_next,
                prev: new_prev,
            };
            assert!(self.dag.insert(new_node_id, new_ndn).is_none());
        }
    }

    fn kron_all_completely_disjoint(&mut self) {
        // TODO: Also handle idle qubits
        // loop {
        //     // break if there are no completely disjoint pairs
        //dag_ordered_iter()
        //     for (idx1, node1) in self.dag.iter() {
        //         for (idx2, node2) in self.dag.iter() {
        //             if idx1 == idx2 {
        //                 continue;
        //             }

        //             if node1.location.intersect(&node2.location).len() == 0 {
        //                 assert!(node1.next.iter().all(|n| n.is_none()));
        //                 assert!(node1.prev.iter().all(|n| n.is_none()));
        //                 assert!(node2.next.iter().all(|n| n.is_none()));
        //                 assert!(node2.prev.iter().all(|n| n.is_none()));

        //                 // let node1 = self.dag.remove(idx_left).unwrap();
        //                 // let node2 = self.dag.remove(idx_right).unwrap();
        //                 // let new_node_id = self.index_counter;
        //                 // self.index_counter += 1;

        //                 // TODO: Need to add permutations here since the
        // disjoints doesn't                 // mean separable by an
        // integer index

        //                 break;
        //             }
        //         }
        //     }
        // }
    }

    /// Returns true if there is a non-direct dependency between the two nodes.
    /// This means that there is a path between the two nodes that goes
    /// through another node. This is directional, so this will only check
    /// for a path from `idx_left` to `idx_right`.
    fn has_non_direct_dependency(
        &self,
        idx_left: usize,
        idx_right: usize,
    ) -> bool {
        let right_projection = self.project_to_right(idx_right);
        let mut visited = HashSet::new();
        let mut queue = Vec::new();
        queue.push(idx_left);
        while let Some(idx) = queue.pop() {
            if idx == idx_right {
                return true;
            }

            if right_projection.contains(&idx) {
                // Once we hit the right_projection, we know we have past the
                // right node, so we can stop searching this path.
                // Note this is the right node's projection.
                continue;
            }

            if visited.contains(&idx) {
                continue;
            }
            visited.insert(idx);

            let node = &self.dag[&idx];
            for next in node.next.iter() {
                if let Some(next_idx) = next {
                    if idx == idx_left && next_idx == &idx_right {
                        // If we are looking at the direct dependency, then
                        // we can skip it. Remember that we are looking for
                        // non-direct dependencies.
                        continue;
                    }
                    queue.push(*next_idx);
                }
            }
        }

        false
    }

    /// Return the set of nodes seen first on each qudit in the circuit.
    /// This is calculated by breadth-first search to the right/next of
    /// the node pointed to by `idx`. When a node is traversed that touches
    /// a qudit not seen yet, that node is added to the set. Once we have
    /// seen all qudits or run out of nodes, we stop.
    fn project_to_right(&self, node_idx: usize) -> HashSet<usize> {
        let mut right_projection = HashSet::new();
        let mut seen_qudits: HashSet<usize> = HashSet::new();
        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();
        queue.push_back(node_idx);
        while let Some(idx) = queue.pop_front() {
            if visited.contains(&idx) {
                continue;
            }
            visited.insert(idx);

            let node = &self.dag[&idx];

            // Check if node touches new qudit
            for qudit in node.location.qudits().iter() {
                // right projection doesn't include starting node
                if idx == node_idx {
                    break;
                }

                if !seen_qudits.contains(qudit) {
                    seen_qudits.extend(node.location.qudits().iter());
                    right_projection.insert(idx);

                    if seen_qudits.len() == self.num_qudits {
                        return right_projection;
                    }
                }
            }

            for next in node.next.iter() {
                if let Some(next_idx) = next {
                    queue.push_back(*next_idx);
                }
            }
        }

        right_projection
    }
}

#[cfg(test)]
pub mod strategies {
    use crate::{Gate, QuditRadices};
    use proptest::prelude::*;

    use super::*;

    pub fn builder_from_locations(
        radices: QuditRadices,
        locations: Vec<Vec<usize>>,
    ) -> TreeBuilder {
        let mut dag = HashMap::new();
        let mut index_counter = 0;
        let mut frontier = vec![None; radices.len()];
        for location in locations {
            let prevs = location.iter().map(|&i| frontier[i].clone()).collect();
            let node = match location.len() {
                0 => panic!("Invalid location length"),
                1 => NodeDAGNode {
                    node: ExpressionTree::Leaf(Gate::P(radices[location[0]])),
                    location: CircuitLocation::pure(location),
                    next: vec![None],
                    prev: prevs,
                },
                _ => {
                    let base_gate = Gate::P(radices[location[0]]);
                    let control_radices = QuditRadices::new(
                        location[1..].iter().map(|&i| radices[i]).collect(),
                    );
                    let control_levels = location[1..]
                        .iter()
                        .map(|&i| vec![radices[i] - 1])
                        .collect();
                    let gate = Gate::Controlled(
                        base_gate,
                        control_radices,
                        control_levels,
                    );
                    NodeDAGNode {
                        node: ExpressionTree::Leaf(gate),
                        location: CircuitLocation::pure(location.clone()),
                        next: vec![None; location.len()],
                        prev: prevs,
                    }
                },
            };

            for (idx, prev) in node.prev.iter().enumerate() {
                let global_idx = node.location.qudits()[idx];
                if let Some(prev_idx) = prev {
                    let prev: &mut NodeDAGNode = dag.get_mut(prev_idx).unwrap();
                    let prev_loc_idx = prev
                        .location
                        .qudits()
                        .iter()
                        .position(|&i| i == global_idx)
                        .unwrap();
                    prev.next[prev_loc_idx] = Some(index_counter);
                }

                frontier[global_idx] = Some(index_counter);
            }

            dag.insert(index_counter, node);
            index_counter += 1;
        }

        TreeBuilder {
            num_qudits: radices.len(),
            dag,
            index_counter,
        }
    }

    impl Arbitrary for TreeBuilder {
        type Parameters = (usize, usize, usize, usize, usize, usize);
        type Strategy = BoxedStrategy<Self>;

        fn arbitrary() -> Self::Strategy {
            Self::arbitrary_with((2, 4, 1, 4, 0, 10))
        }

        fn arbitrary_with(args: Self::Parameters) -> Self::Strategy {
            let min_radix = args.0;
            let max_radix = args.1;
            let min_count = args.2;
            let max_count = args.3;
            let min_gates = args.4;
            let max_gates = args.5;
            prop::collection::vec(min_radix..=max_radix, min_count..=max_count)
                .prop_flat_map(move |v| {
                    (
                        Just(QuditRadices::new(v.clone())),
                        prop::collection::vec(
                            prop::collection::hash_set(0..v.len(), 1..=v.len()),
                            min_gates..=max_gates,
                        ),
                    )
                })
                .prop_map(|(radices, locations)| {
                    (
                        radices,
                        locations
                            .into_iter()
                            .map(|s| s.into_iter().collect())
                            .collect::<Vec<Vec<usize>>>(),
                    )
                })
                .prop_map(|(radices, locations)| {
                    builder_from_locations(radices, locations)
                })
                .boxed()
        }
    }
}

#[cfg(test)]
mod tests {

    use super::strategies::builder_from_locations;
    use super::*;
    use crate::radices;
    use crate::Gate;
    use crate::QuditRadices;
    use proptest::prelude::*;

    #[test]
    fn test_builder_from_locations() {
        let mut dag = HashMap::new();
        dag.insert(
            0,
            NodeDAGNode {
                node: ExpressionTree::Leaf(Gate::CX()),
                location: CircuitLocation::pure(vec![2, 3]),
                next: vec![Some(1), None],
                prev: vec![None, None],
            },
        );
        dag.insert(
            1,
            NodeDAGNode {
                node: ExpressionTree::Leaf(Gate::CX()),
                location: CircuitLocation::pure(vec![1, 2]),
                next: vec![Some(2), None],
                prev: vec![None, Some(0)],
            },
        );
        dag.insert(
            2,
            NodeDAGNode {
                node: ExpressionTree::Leaf(Gate::CX()),
                location: CircuitLocation::pure(vec![0, 1]),
                next: vec![None, None],
                prev: vec![None, Some(1)],
            },
        );

        let builder_1 = TreeBuilder {
            num_qudits: 4,
            dag,
            index_counter: 3,
        };

        let builder_2 = builder_from_locations(
            radices![2, 2, 2, 2],
            vec![vec![2, 3], vec![1, 2], vec![0, 1]],
        );

        assert_eq!(builder_1.index_counter, builder_2.index_counter);
        assert_eq!(builder_1.num_qudits, builder_2.num_qudits);
        assert_eq!(builder_1.dag.len(), builder_2.dag.len());
        for (idx, node) in builder_1.dag.iter() {
            assert_eq!(builder_2.dag[idx].location, node.location);
            assert_eq!(builder_2.dag[idx].next, node.next);
            assert_eq!(builder_2.dag[idx].prev, node.prev);
        }
    }

    #[test]
    fn test_has_non_direct_dependency_simple() {
        let builder = builder_from_locations(
            radices![2, 2],
            vec![vec![0, 1], vec![0, 1], vec![0, 1]],
        );
        assert!(!builder.has_non_direct_dependency(0, 1));
        assert!(builder.has_non_direct_dependency(0, 2));
        assert!(!builder.has_non_direct_dependency(1, 2));
    }

    #[test]
    fn test_has_non_direct_dependency_complex() {
        let builder = builder_from_locations(
            radices![2, 3, 2, 4],
            vec![vec![2, 3], vec![1, 2], vec![0, 1]],
        );
        assert!(!builder.has_non_direct_dependency(0, 1));
        assert!(builder.has_non_direct_dependency(0, 2));
        assert!(!builder.has_non_direct_dependency(1, 2));
    }

    /// Return true if you can find lidx from ridx by going backwards
    fn backward_search(tree: &TreeBuilder, lidx: usize, ridx: usize) -> bool {
        let mut visited = HashSet::new();
        let mut queue = Vec::new();
        queue.push(ridx);
        while let Some(idx) = queue.pop() {
            if idx == lidx {
                return true;
            }

            if visited.contains(&idx) {
                continue;
            }
            visited.insert(idx);

            let node = &tree.dag[&idx];
            for prev in node.prev.iter() {
                if let Some(prev_idx) = prev {
                    queue.push(*prev_idx);
                }
            }
        }

        false
    }

    proptest! {
        #[test]
        fn test_project_to_right_backward_search(tree in any::<TreeBuilder>()) {
            for (idx, _) in tree.dag.iter() {
                let right_projection = tree.project_to_right(*idx);
                // all nodes in right projection should have a path
                // to idx by going backwards
                for ridx in right_projection.iter() {
                    assert!(backward_search(&tree, *idx, *ridx));
                }
            }
        }
    }
}
