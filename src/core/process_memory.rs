//! Live heap ownership in the native Python extension and process-scoped samples.
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use serde::Serialize;
use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::RefCell;
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};
use std::sync::{mpsc, Arc, Mutex, Weak};
use std::thread::{self, JoinHandle};
use std::time::Duration;

static RUST_BYTES: AtomicUsize = AtomicUsize::new(0);
static ARRAY_BYTES: AtomicUsize = AtomicUsize::new(0);

struct CountingAllocator;
#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

// Forward the original layouts unchanged to the system allocator. Counting
// requested payloads excludes allocator metadata, cached pages and foreign DLLs.
unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let ptr = unsafe { System.alloc(layout) };
        if !ptr.is_null() {
            RUST_BYTES.fetch_add(layout.size(), Relaxed);
        }
        ptr
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        let ptr = unsafe { System.alloc_zeroed(layout) };
        if !ptr.is_null() {
            RUST_BYTES.fetch_add(layout.size(), Relaxed);
        }
        ptr
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) };
        RUST_BYTES.fetch_sub(layout.size(), Relaxed);
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        let new = unsafe { System.realloc(ptr, layout, size) };
        if !new.is_null() {
            if size >= layout.size() {
                RUST_BYTES.fetch_add(size - layout.size(), Relaxed);
            } else {
                RUST_BYTES.fetch_sub(layout.size() - size, Relaxed);
            }
        }
        new
    }
}

#[pyclass(name = "_OwnedArrayBytes")]
pub(crate) struct OwnedArrayBytes(usize);
#[pymethods]
impl OwnedArrayBytes {
    #[new]
    fn new(bytes: usize) -> Self {
        ARRAY_BYTES.fetch_add(bytes, Relaxed);
        Self(bytes)
    }
}
impl Drop for OwnedArrayBytes {
    fn drop(&mut self) {
        ARRAY_BYTES.fetch_sub(self.0, Relaxed);
    }
}

#[cfg(windows)]
fn current_bytes() -> std::io::Result<u64> {
    use windows_sys::Win32::System::{
        ProcessStatus::{GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS},
        Threading::GetCurrentProcess,
    };
    let mut c = PROCESS_MEMORY_COUNTERS::default();
    let len = std::mem::size_of_val(&c) as u32;
    if unsafe { GetProcessMemoryInfo(GetCurrentProcess(), &mut c, len) } == 0 {
        return Err(std::io::Error::last_os_error());
    }
    Ok(c.PagefileUsage as u64)
}
#[cfg(target_os = "linux")]
fn current_bytes() -> std::io::Result<u64> {
    let status = std::fs::read_to_string("/proc/self/status")?;
    status
        .lines()
        .find_map(|s| s.strip_prefix("VmRSS:"))
        .and_then(|s| s.split_whitespace().next()?.parse::<u64>().ok())
        .map(|kb| kb * 1024)
        .ok_or_else(|| std::io::Error::other("VmRSS unavailable"))
}
#[cfg(target_os = "macos")]
fn current_bytes() -> std::io::Result<u64> {
    // Darwin proc_taskinfo starts with virtual_size and resident_size (uint64_t).
    #[link(name = "proc")]
    extern "C" {
        fn proc_pidinfo(
            pid: i32,
            flavor: i32,
            arg: u64,
            buf: *mut std::ffi::c_void,
            size: i32,
        ) -> i32;
    }
    let mut info = [0u64; 12];
    let n = unsafe {
        proc_pidinfo(
            std::process::id() as i32,
            4,
            0,
            info.as_mut_ptr().cast(),
            std::mem::size_of_val(&info) as i32,
        )
    };
    if n < 16 {
        return Err(std::io::Error::last_os_error());
    }
    Ok(info[1])
}
#[cfg(not(any(windows, target_os = "linux", target_os = "macos")))]
fn current_bytes() -> std::io::Result<u64> {
    Err(std::io::Error::other(
        "no current process footprint counter for this platform",
    ))
}

#[derive(Clone, Copy, Serialize)]
struct Sample {
    process: u64,
    rust_heap: usize,
    python_arrays: usize,
    host_visible: u64,
    mapped_buffers: u64,
    device_local: u64,
}
fn sample() -> std::io::Result<Sample> {
    let process = current_bytes()?;
    let (host_visible, device_local) = super::memory_tracker::global_tracker().ledger_totals();
    Ok(Sample {
        process,
        rust_heap: RUST_BYTES.load(Relaxed),
        python_arrays: ARRAY_BYTES.load(Relaxed),
        host_visible,
        mapped_buffers: super::resource_tracker::mapped_buffer_bytes(),
        device_local,
    })
}

#[derive(Serialize)]
struct Samples {
    baseline: Sample,
    at_peak: Sample,
    at_cpu_peak: Sample,
    count: usize,
    #[serde(skip)]
    previous: Sample,
    #[serde(skip)]
    phase: &'static str,
    #[serde(skip)]
    changes: BTreeMap<&'static str, i64>,
    residual_by_phase_at_peak: BTreeMap<&'static str, i64>,
}
impl Sample {
    fn tracked_cpu(self) -> usize {
        self.rust_heap + self.python_arrays
    }

    fn residual(self) -> i64 {
        self.process as i64
            - self.rust_heap as i64
            - self.python_arrays as i64
            - self.mapped_buffers as i64
    }
}
impl Samples {
    fn observe(&mut self) -> std::io::Result<()> {
        let row = sample()?;
        *self.changes.entry(self.phase).or_default() += row.residual() - self.previous.residual();
        self.previous = row;
        if row.tracked_cpu() > self.at_cpu_peak.tracked_cpu() {
            self.at_cpu_peak = row;
        }
        if row.process >= self.at_peak.process {
            self.at_peak = row;
            self.residual_by_phase_at_peak.clone_from(&self.changes);
        }
        self.count += 1;
        Ok(())
    }
}

thread_local! {
    static ACTIVE: RefCell<Vec<Weak<Mutex<Samples>>>> = const { RefCell::new(Vec::new()) };
}

/// Charge the interval ending here to the previous operation. This includes
/// driver activity and allocator retention during that interval.
pub(crate) fn render_phase(name: &'static str) {
    ACTIVE.with(|list| {
        for sampler in list.borrow().iter().filter_map(Weak::upgrade) {
            let mut state = sampler.lock().unwrap_or_else(|p| p.into_inner());
            if state.observe().is_ok() {
                state.phase = name;
            }
        }
    });
}

#[pyclass(name = "_ProcessMemoryCapture")]
pub(crate) struct ProcessMemoryCapture {
    samples: Arc<Mutex<Samples>>,
    stop: Option<mpsc::Sender<()>>,
    worker: Option<JoinHandle<std::io::Result<()>>>,
    previous_capture: u64,
    owner: thread::ThreadId,
}
impl ProcessMemoryCapture {
    fn stop_worker(&mut self) -> std::io::Result<()> {
        self.stop.take();
        if let Some(worker) = self.worker.take() {
            worker
                .join()
                .map_err(|_| std::io::Error::other("memory sampler panicked"))??;
        }
        Ok(())
    }
}
#[pymethods]
impl ProcessMemoryCapture {
    #[new]
    fn new(interval: f64) -> PyResult<Self> {
        let period = Duration::try_from_secs_f64(interval)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        if period.is_zero() {
            return Err(PyValueError::new_err("sample interval must be positive"));
        }
        let baseline = sample()?;
        let samples = Arc::new(Mutex::new(Samples {
            baseline,
            at_peak: baseline,
            at_cpu_peak: baseline,
            count: 1,
            previous: baseline,
            phase: "render setup and publication",
            changes: BTreeMap::new(),
            residual_by_phase_at_peak: BTreeMap::new(),
        }));
        let sink = samples.clone();
        let (stop, rx) = mpsc::channel();
        let worker = thread::Builder::new()
            .name("forge3d-memory".into())
            .spawn(move || {
                while rx.recv_timeout(period) == Err(mpsc::RecvTimeoutError::Timeout) {
                    sink.lock().unwrap_or_else(|p| p.into_inner()).observe()?;
                }
                Ok(())
            })?;
        ACTIVE.with(|list| {
            let mut list = list.borrow_mut();
            list.retain(|weak| weak.strong_count() != 0);
            list.push(Arc::downgrade(&samples));
        });
        Ok(Self {
            samples,
            stop: Some(stop),
            worker: Some(worker),
            previous_capture: super::certificate::capture_generation(),
            owner: thread::current().id(),
        })
    }
    fn finish(&mut self) -> PyResult<String> {
        self.stop_worker()?;
        let mut samples = self.samples.lock().unwrap_or_else(|p| p.into_inner());
        samples.observe()?;
        serde_json::to_string(&*samples).map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }
    fn scene_phase(&self, phase: &str) -> PyResult<()> {
        let name = match phase {
            "session initialization" => "session initialization",
            "terrain renderer construction" => "terrain renderer construction",
            "environment map initialization" => "environment map initialization",
            "render setup and publication" => "render setup and publication",
            _ => return Err(PyValueError::new_err("unknown memory observation phase")),
        };
        render_phase(name);
        Ok(())
    }
    fn attach(&self, json: String, flythrough: bool) -> PyResult<bool> {
        if thread::current().id() != self.owner {
            return Err(PyRuntimeError::new_err(
                "capture belongs to a different thread",
            ));
        }
        if !json.trim_start().starts_with('{') {
            return Err(PyValueError::new_err("expected a memory report object"));
        }
        let value = serde_json::value::RawValue::from_string(json)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(super::certificate::attach_job_memory(
            self.previous_capture,
            value,
            flythrough,
        ))
    }
}
impl Drop for ProcessMemoryCapture {
    fn drop(&mut self) {
        let _ = self.stop_worker();
    }
}
