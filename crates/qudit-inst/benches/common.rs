use std::ffi::c_int;
use std::fs::File;
use std::path::Path;

use criterion::profiler::Profiler;
use pprof::ProfilerGuard;
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

