use std::f64::consts::PI;
use std::ffi::c_int;
use std::fs::File;
use std::path::Path;

use criterion::profiler::Profiler;
use pprof::ProfilerGuard;
use qudit_circuit::CircuitLocation;
use qudit_circuit::QuditCircuit;
use qudit_circuit::loc;
use qudit_core::radices;
use qudit_gates::Gate;
use qudit_core::QuditRadices;

/// Build a QFT circuit with `n` qubits.
///
/// This qft implementation does not consider numerical issues and does not
/// perform the final swap back step in the algorithm. It should only be
/// used for benchmarking purposes.
#[allow(dead_code)]
pub fn build_qft_circuit(n: usize) -> QuditCircuit {
    // TODO: Double check this is actually a QFT
    let mut circ = QuditCircuit::new(radices![2; n], 0);
    for i in 0..n {
        circ.append_gate(Gate::H(2), loc![i], vec![]);
        for j in (i + 1)..n {
            let p = PI * (2.0f64.powi((j - i) as i32));
            circ.append_gate(Gate::CP(), loc![i, j], vec![p]);
        }
    }
    circ
}

#[allow(dead_code)]
pub fn build_qsearch_thin_step_circuit(n: usize) -> QuditCircuit {
    let mut circ = QuditCircuit::new(radices![2; n], 0);
    for i in 0..n {
        circ.append_gate(Gate::U3(), loc![i], vec![]);
    }
    for _ in 0..n {
        for i in 0..(n - 1) {
            circ.append_gate(Gate::CX(), loc![i, i + 1], vec![]);
            circ.append_gate(Gate::U3(), loc![i], vec![]);
            circ.append_gate(Gate::U3(), loc![i + 1], vec![]);
        }
    }
    circ
}

#[allow(dead_code)]
pub fn build_qsearch_thick_step_circuit(n: usize) -> QuditCircuit {
    let mut circ = QuditCircuit::new(radices![2; n], 0);
    for i in 0..n {
        circ.append_gate(Gate::U3(), loc![i], vec![]);
    }
    for _ in 0..n {
        for i in 0..(n - 1) {
            for _j in 0..3 {
                circ.append_gate(Gate::CX(), loc![i, i + 1], vec![]);
                circ.append_gate(Gate::U3(), loc![i], vec![]);
                circ.append_gate(Gate::U3(), loc![i + 1], vec![]);
            }
        }
    }
    circ
}

/// Simple profiler that generates a flamegraph of the benchmark.
pub struct FlamegraphProfiler<'a> {
    frequency: c_int,
    active_profiler: Option<ProfilerGuard<'a>>,
}

impl<'a> FlamegraphProfiler<'a> {
    pub fn new(frequency: c_int) -> Self {
        FlamegraphProfiler {
            frequency,
            active_profiler: None,
        }
    }
}

impl<'a> Profiler for FlamegraphProfiler<'a> {
    fn start_profiling(&mut self, _benchmark_id: &str, _benchmark_dir: &Path) {
        self.active_profiler = Some(ProfilerGuard::new(self.frequency).unwrap());
    }

    fn stop_profiling(&mut self, _benchmark_id: &str, benchmark_dir: &Path) {
        std::fs::create_dir_all(benchmark_dir).unwrap();
        let flamegraph_path = benchmark_dir.join("flamegraph.svg");
        let flamegraph_file = File::create(&flamegraph_path)
            .expect("File system error while creating flamegraph.svg");
        if let Some(profiler) = self.active_profiler.take() {
            profiler
                .report()
                .build()
                .unwrap()
                .flamegraph(flamegraph_file)
                .expect("Error writing flamegraph");
        }
    }
}
