use std::collections::{HashMap, HashSet};

use itertools::Itertools;

use crate::Gate;

use super::{Bytecode, GeneralizedInstruction, MatrixBuffer};

pub fn remove_identity_frpr(code: Bytecode) -> Bytecode {
    let mut opt_code = Vec::new();
    let mut buffer_remap = HashMap::new();

    for mut inst in code.dynamic_code {
        match inst {
            GeneralizedInstruction::FRPR(
                in_buffer,
                ref _shape,
                ref perm,
                out_buffer,
            ) => {
                if code.matrix_buffers[in_buffer].nrows
                    == code.matrix_buffers[out_buffer].nrows
                {
                    if code.matrix_buffers[in_buffer].ncols
                        == code.matrix_buffers[out_buffer].ncols
                    {
                        if perm.iter().enumerate().all(|(i, &j)| i == j) {
                            buffer_remap.insert(out_buffer, in_buffer);
                            continue;
                        }
                    }
                }
                inst.replace_buffer_indices(&mut buffer_remap);
                opt_code.push(inst);
            },
            _ => {
                inst.replace_buffer_indices(&mut buffer_remap);
                opt_code.push(inst);
            },
        }
    }

    Bytecode {
        static_code: code.static_code,
        dynamic_code: opt_code,
        matrix_buffers: code.matrix_buffers,
        merged_buffers: code.merged_buffers,
    }
}

pub struct BufferOptimizer {
    in_use_buffers: HashSet<usize>,
    gate_buffers: HashMap<Gate, Vec<usize>>,
    clobber_buffers: HashMap<MatrixBuffer, Vec<usize>>,
    buffer_remapping: HashMap<usize, usize>,
    buffers: Vec<MatrixBuffer>,
    immortal_buffers: HashSet<usize>,
    old_buffers: Vec<MatrixBuffer>,
}

impl BufferOptimizer {
    pub fn new() -> Self {
        Self {
            in_use_buffers: HashSet::new(),
            gate_buffers: HashMap::new(),
            clobber_buffers: HashMap::new(),
            buffer_remapping: HashMap::new(),
            buffers: Vec::new(),
            immortal_buffers: HashSet::new(),
            old_buffers: Vec::new(),
        }
    }

    fn get_gate_buffer(&mut self, gate: Gate) -> usize {
        if let Some(buffer_list) = self.gate_buffers.get(&gate) {
            for buffer_index in buffer_list.iter() {
                if !self.in_use_buffers.contains(buffer_index) {
                    self.in_use_buffers.insert(*buffer_index);
                    return *buffer_index;
                }
            }
        }

        let out = self.buffers.len();
        self.buffers.push((&gate).into());
        self.in_use_buffers.insert(out);
        if self.gate_buffers.contains_key(&gate) {
            self.gate_buffers.get_mut(&gate).unwrap().push(out);
        } else {
            self.gate_buffers.insert(gate.clone(), vec![out]);
        }
        out
    }

    fn get_clobber_buffer(&mut self, buffer: MatrixBuffer) -> usize {
        if let Some(buffer_list) = self.clobber_buffers.get(&buffer) {
            for buffer_index in buffer_list.iter() {
                if !self.in_use_buffers.contains(buffer_index) {
                    self.in_use_buffers.insert(*buffer_index);
                    return *buffer_index;
                }
            }
        }

        let out = self.buffers.len();
        self.buffers.push(buffer.clone());
        self.in_use_buffers.insert(out);
        if self.clobber_buffers.contains_key(&buffer) {
            self.clobber_buffers.get_mut(&buffer).unwrap().push(out);
        } else {
            self.clobber_buffers.insert(buffer, vec![out]);
        }
        out
    }

    fn free_buffer(&mut self, index: usize) {
        if self.immortal_buffers.contains(&index) {
            return;
        }
        self.in_use_buffers.remove(&index);
    }

    fn immortalize_in_use_buffers(&mut self) {
        for &buffer_index in self.in_use_buffers.iter() {
            self.immortal_buffers.insert(buffer_index);
        }
    }

    fn optimize_region(
        &mut self,
        region: Vec<GeneralizedInstruction>,
    ) -> Vec<GeneralizedInstruction> {
        let mut opt_code = Vec::new();

        for inst in region {
            match inst {
                GeneralizedInstruction::Write(g, p, old_buffer) => {
                    let new_buffer = self.get_gate_buffer(g.clone());
                    opt_code
                        .push(GeneralizedInstruction::Write(g, p, new_buffer));
                    self.buffer_remapping.insert(old_buffer, new_buffer);
                },
                GeneralizedInstruction::Matmul(left, right, out) => {
                    let new_left = self.buffer_remapping[&left];
                    let new_right = self.buffer_remapping[&right];

                    let out_buffer = self.old_buffers[out];
                    let new_out = self.get_clobber_buffer(out_buffer);
                    opt_code.push(GeneralizedInstruction::Matmul(
                        new_left, new_right, new_out,
                    ));

                    self.free_buffer(new_left);
                    self.free_buffer(new_right);
                    self.buffer_remapping.insert(out, new_out);
                },
                GeneralizedInstruction::FRPR(old_in, shape, perm, old_out) => {
                    let new_in = self.buffer_remapping[&old_in];

                    let out_buffer = self.old_buffers[old_out];
                    let new_out = self.get_clobber_buffer(out_buffer);
                    opt_code.push(GeneralizedInstruction::FRPR(
                        new_in,
                        shape.clone(),
                        perm.clone(),
                        new_out,
                    ));

                    self.free_buffer(new_in);
                    self.buffer_remapping.insert(old_out, new_out);
                },
                GeneralizedInstruction::Kron(left, right, out) => {
                    let new_left = self.buffer_remapping[&left];
                    let new_right = self.buffer_remapping[&right];

                    let out_buffer = self.old_buffers[out];
                    let new_out = self.get_clobber_buffer(out_buffer);
                    opt_code.push(GeneralizedInstruction::Kron(
                        new_left, new_right, new_out,
                    ));

                    self.free_buffer(new_left);
                    self.free_buffer(new_right);
                    self.buffer_remapping.insert(out, new_out);
                },
            }
        }

        opt_code
    }

    pub fn optimize(mut self, code: Bytecode) -> Bytecode {
        self.old_buffers = code.matrix_buffers;
        let static_opt_code = self.optimize_region(code.static_code);
        self.immortalize_in_use_buffers();
        let dynamic_opt_code = self.optimize_region(code.dynamic_code);

        Bytecode {
            static_code: static_opt_code,
            dynamic_code: dynamic_opt_code,
            matrix_buffers: self.buffers,
            merged_buffers: code.merged_buffers,
        }
    }
}

pub struct BufferReuser {}

impl BufferReuser {
    pub fn new() -> Self {
        Self {}
    }

    pub fn check_lifespan_overlap(
        lifespan1: &Vec<(usize, usize)>,
        lifespan2: &Vec<(usize, usize)>,
    ) -> bool {
        // TODO: can be done way better
        for &(start, end) in lifespan1.iter() {
            for &(start2, end2) in lifespan2.iter() {
                if start2 <= end && start <= end2 {
                    return true;
                }
                if start <= end2 && start2 <= end {
                    return true;
                }
            }
        }
        false
    }

    pub fn get_mergeable_buffers(
        buffers: &Vec<MatrixBuffer>,
        buffer_lifespans: &HashMap<usize, Vec<(usize, usize)>>,
    ) -> HashMap<usize, Vec<usize>> {
        let mut mergeable_buffers: HashMap<usize, Vec<usize>> = HashMap::new();
        for (buffer1, lifespans1) in buffer_lifespans.iter() {
            for (buffer2, lifespans2) in buffer_lifespans.iter() {
                if buffer1 == buffer2 {
                    continue;
                }

                if Self::check_lifespan_overlap(lifespans1, lifespans2) {
                    continue;
                }

                if buffers[*buffer1].nrows > buffers[*buffer2].nrows {
                    continue;
                }

                if buffers[*buffer1].ncols > buffers[*buffer2].ncols {
                    continue;
                }

                if buffers[*buffer1].num_params > buffers[*buffer2].num_params {
                    continue;
                }

                if let Some(merge_list) = mergeable_buffers.get_mut(buffer1) {
                    merge_list.push(*buffer2);
                } else {
                    mergeable_buffers.insert(*buffer1, vec![*buffer2]);
                }
            }
        }
        mergeable_buffers
    }

    pub fn merge_one_buffer(
        matrix_buffers: &Vec<MatrixBuffer>,
        buffer_lifespans: &mut HashMap<usize, Vec<(usize, usize)>>,
        mergeable_buffers: &HashMap<usize, Vec<usize>>,
        buffer_remap: &mut HashMap<usize, usize>,
    ) {
        let mergeables_keys = mergeable_buffers.keys().collect::<Vec<&usize>>();
        mergeables_keys.iter().sorted_by(|&a, &b| {
            matrix_buffers[**a].size().cmp(&matrix_buffers[**b].size())
        });

        let mergeable = mergeables_keys.iter().rev().take(1).next().unwrap();
        let merge_list = mergeable_buffers.get(*mergeable).unwrap();
        let merge_vec = merge_list.iter().collect::<Vec<&usize>>();
        let biggest_mergeable = merge_vec
            .iter()
            .sorted_by(|&a, &b| {
                matrix_buffers[**a].size().cmp(&matrix_buffers[**b].size())
            })
            .rev()
            .take(1)
            .next()
            .unwrap();

        buffer_remap.insert(**mergeable, **biggest_mergeable);
        let old_lifespans = buffer_lifespans.remove(mergeable).unwrap();
        for old_lifespan in old_lifespans.iter() {
            buffer_lifespans
                .get_mut(biggest_mergeable)
                .unwrap()
                .push(*old_lifespan);
        }
    }

    pub fn reuse_buffers(mut self, code: Bytecode) -> Bytecode {
        let mut buffer_lifespans: HashMap<usize, Vec<(usize, usize)>> =
            HashMap::new();
        let mut active_buffers = HashMap::new();

        for (i, inst) in code.dynamic_code.iter().enumerate() {
            match inst {
                GeneralizedInstruction::Write(_g, _p, _buffer) => {
                    // active_buffers.insert(buffer, i);
                    // println!("{:?}", active_buffers);
                },
                GeneralizedInstruction::Matmul(left, right, out) => {
                    active_buffers.insert(out, i);
                    let start_inst = active_buffers.remove(&left);
                    if start_inst.is_some() {
                        if let Some(lifespans) = buffer_lifespans.get_mut(&left)
                        {
                            lifespans.push((start_inst.unwrap(), i));
                        } else {
                            buffer_lifespans
                                .insert(*left, vec![(start_inst.unwrap(), i)]);
                        }
                    }
                    let start_inst = active_buffers.remove(&right);
                    if start_inst.is_some() {
                        if let Some(lifespans) =
                            buffer_lifespans.get_mut(&right)
                        {
                            lifespans.push((start_inst.unwrap(), i));
                        } else {
                            buffer_lifespans
                                .insert(*right, vec![(start_inst.unwrap(), i)]);
                        }
                    }
                },
                GeneralizedInstruction::FRPR(
                    in_buffer,
                    ref _shape,
                    ref _perm,
                    out_buffer,
                ) => {
                    active_buffers.insert(out_buffer, i);
                    let start_inst = active_buffers.remove(in_buffer);
                    if start_inst.is_some() {
                        if let Some(lifespans) =
                            buffer_lifespans.get_mut(in_buffer)
                        {
                            lifespans.push((start_inst.unwrap(), i));
                        } else {
                            buffer_lifespans.insert(
                                *in_buffer,
                                vec![(start_inst.unwrap(), i)],
                            );
                        }
                    }
                },
                GeneralizedInstruction::Kron(left, right, out) => {
                    active_buffers.insert(out, i);
                    let start_inst = active_buffers.remove(&left);
                    if start_inst.is_some() {
                        if let Some(lifespans) = buffer_lifespans.get_mut(&left)
                        {
                            lifespans.push((start_inst.unwrap(), i));
                        } else {
                            buffer_lifespans
                                .insert(*left, vec![(start_inst.unwrap(), i)]);
                        }
                    }
                    let start_inst = active_buffers.remove(&right);
                    if start_inst.is_some() {
                        if let Some(lifespans) =
                            buffer_lifespans.get_mut(&right)
                        {
                            lifespans.push((start_inst.unwrap(), i));
                        } else {
                            buffer_lifespans
                                .insert(*right, vec![(start_inst.unwrap(), i)]);
                        }
                    }
                },
            }
        }
        let mut mergeable_buffers = Self::get_mergeable_buffers(
            &code.matrix_buffers,
            &buffer_lifespans,
        );

        let mut merged_buffers = HashMap::new();

        while !mergeable_buffers.is_empty() {
            Self::merge_one_buffer(
                &code.matrix_buffers,
                &mut buffer_lifespans,
                &mergeable_buffers,
                &mut merged_buffers,
            );
            mergeable_buffers = Self::get_mergeable_buffers(
                &code.matrix_buffers,
                &buffer_lifespans,
            );
        }

        Bytecode {
            static_code: code.static_code,
            dynamic_code: code.dynamic_code,
            matrix_buffers: code.matrix_buffers,
            merged_buffers,
        }
    }
}
