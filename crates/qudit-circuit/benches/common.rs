use std::f64::consts::PI;
use std::ffi::c_int;
use std::fs::File;
use std::path::Path;

use criterion::profiler::Profiler;
use pprof::ProfilerGuard;
use qudit_circuit::WireList;
use qudit_circuit::QuditCircuit;
use qudit_core::radices;
use qudit_expr::UnitaryExpression;
use qudit_gates::Gate;
use qudit_core::QuditRadices;
use rand::Rng;

/// Build a QFT circuit with `n` qubits.
///
/// This qft implementation does not consider numerical issues and does not
/// perform the final swap back step in the algorithm. It should only be
/// used for benchmarking purposes.
pub fn build_qft_circuit(n: usize) -> QuditCircuit {
    // TODO: Double check this is actually a QFT
    let mut circ = QuditCircuit::pure(radices![2; n]);
    let h_ref = circ.cache_operation(Gate::H(2));
    let cp_ref = circ.cache_operation(Gate::CP());
    for i in 0..n {
        circ.append_by_code::<_, [f64; 0]>(h_ref, [i], []);
        for j in (i + 1)..n {
            let p = PI * (2.0f64.powi((j - i) as i32));
            circ.append_by_code(cp_ref, [i, j], [p]);
        }
    }
    circ
}


pub fn build_dtc_circuit(n: usize) {
    let g = 0.95;

    let rx_expr = UnitaryExpression::new("RX(theta) {
        [
            [cos(theta/2), ~i*sin(theta/2)],
            [~i*sin(theta/2), cos(theta/2)],
        ]
    }");

    let rz_expr = UnitaryExpression::new("RZ(theta) {
        [
            [e^(~i*theta/2), 0],
            [0, e^(i*theta/2)],
        ]
    }");

    let rzz_expr = UnitaryExpression::new("RZZ(theta) {
        [
            [e^(~i*theta/2), 0, 0, 0],
            [0, e^(i*theta/2), 0, 0],
            [0, 0, e^(i*theta/2), 0],
            [0, 0, 0, e^(~i*theta/2)],
        ]
    }");

    let mut circ = QuditCircuit::pure(radices![2; n]);
    let rx_ref = circ.cache_operation(rx_expr);
    let rz_ref = circ.cache_operation(rz_expr);
    let rzz_ref = circ.cache_operation(rzz_expr);
    let mut rng = rand::thread_rng();

    for _ in 0..n {
        for i in 0..n {
            circ.append_by_code(rx_ref, i, [g * std::f64::consts::PI]);
        }

        for i in (0..(n - 1)).step_by(2) {
            let low = PI / 16.0;
            let high = 3.0 * PI / 16.0;
            let phi: f64 = rng.gen_range(low..high);  
            circ.append_by_code(rzz_ref, [i, i + 1], [phi]);
        }

        for i in (1..(n - 1)).step_by(2) {
            let low = PI / 16.0;
            let high = 3.0 * PI / 16.0;
            let phi: f64 = rng.gen_range(low..high);  
            circ.append_by_code(rzz_ref, [i, i + 1], [phi]);
        }

        for i in 0..n {
            let low = -PI;
            let high = PI;
            let phi: f64 = rng.gen_range(low..high);  
            circ.append_by_code(rz_ref, i, [phi]);
        }
    }
}

// #[allow(dead_code)]
// pub fn build_qsearch_thin_step_circuit(n: usize) -> QuditCircuit {
//     let mut circ = QuditCircuit::pure(radices![2; n]);
//     for i in 0..n {
//         circ.append_parameterized(Gate::U3(), [i]);
//     }
//     for _ in 0..n {
//         for i in 0..(n - 1) {
//             circ.append_parameterized(Gate::CX(), [i, i + 1]);
//             circ.append_parameterized(Gate::U3(), [i]);
//             circ.append_parameterized(Gate::U3(), [i + 1]);
//         }
//     }
//     circ
// }

// #[allow(dead_code)]
// pub fn build_qsearch_thick_step_circuit(n: usize) -> QuditCircuit {
//     let mut circ = QuditCircuit::pure(radices![2; n]);
//     for i in 0..n {
//         circ.append_parameterized(Gate::U3(), [i]);
//     }
//     for _ in 0..n {
//         for i in 0..(n - 1) {
//             for _j in 0..3 {
//                 circ.append_parameterized(Gate::CX(), [i, i + 1]);
//                 circ.append_parameterized(Gate::U3(), [i]);
//                 circ.append_parameterized(Gate::U3(), [i + 1]);
//             }
//         }
//     }
//     circ
// }

use pprof::criterion::{Output, PProfProfiler};
use pprof::flamegraph::Options;

pub struct FlamegraphProfiler<'a, 'b> {
    inner: PProfProfiler<'a, 'b>
}

impl<'a, 'b> FlamegraphProfiler<'a, 'b> {
    pub fn new(frequency: c_int) -> Self {
        let mut options = Options::default();
        options.image_width = Some(2560);
        options.hash = true;
        FlamegraphProfiler {
            inner: PProfProfiler::new(frequency, Output::Flamegraph(Some(options)))
        }
    }
}

impl<'a, 'b> Profiler for FlamegraphProfiler<'a, 'b> {
    fn start_profiling(&mut self, benchmark_id: &str, benchmark_dir: &Path) {
        self.inner.start_profiling(benchmark_id, benchmark_dir);
    }

    fn stop_profiling(&mut self, benchmark_id: &str, benchmark_dir: &Path) {
        self.inner.stop_profiling(benchmark_id, benchmark_dir);
    }
}
