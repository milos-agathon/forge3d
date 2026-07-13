use crate::core::error::RenderError;
use crate::core::memory_tracker::{calculate_texture_size, global_tracker, is_host_visible_usage};
use std::collections::{BTreeMap, HashMap};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};
use wgpu::util::DeviceExt;
use wgpu::{BufferDescriptor, BufferUsages, TextureDescriptor, TextureFormat};

/// Resource handle that automatically unregisters on drop
#[derive(Debug)]
pub enum ResourceHandle {
    Buffer {
        size: u64,
        is_host_visible: bool,
    },
    Texture {
        width: u32,
        height: u32,
        format: TextureFormat,
    },
}

impl Drop for ResourceHandle {
    fn drop(&mut self) {
        let tracker = global_tracker();
        match self {
            ResourceHandle::Buffer {
                size,
                is_host_visible,
            } => {
                tracker.free_buffer_allocation(*size, *is_host_visible);
            }
            ResourceHandle::Texture {
                width,
                height,
                format,
            } => {
                tracker.free_texture_allocation(*width, *height, *format);
            }
        }
    }
}

/// Register a buffer allocation and return a handle that will unregister on drop
pub fn register_buffer(size: u64, usage: BufferUsages) -> ResourceHandle {
    let is_host_visible = is_host_visible_usage(usage);
    let tracker = global_tracker();
    tracker.track_buffer_allocation(size, is_host_visible);
    ResourceHandle::Buffer {
        size,
        is_host_visible,
    }
}

/// Register a texture allocation and return a handle that will unregister on drop
pub fn register_texture(width: u32, height: u32, format: TextureFormat) -> ResourceHandle {
    let tracker = global_tracker();
    tracker.track_texture_allocation(width, height, format);
    ResourceHandle::Texture {
        width,
        height,
        format,
    }
}

/// Register a buffer allocation with explicit host-visible flag
pub fn register_buffer_explicit(size: u64, is_host_visible: bool) -> ResourceHandle {
    let tracker = global_tracker();
    tracker.track_buffer_allocation(size, is_host_visible);
    ResourceHandle::Buffer {
        size,
        is_host_visible,
    }
}

// ---------------------------------------------------------------------------
// CENSOR: total allocation ledger
// ---------------------------------------------------------------------------

/// Classification of a tracked allocation for the ledger.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LedgerCategory {
    Buffer,
    Texture,
}

#[derive(Clone, Debug)]
struct LedgerEntry {
    label: String,
    bytes: u64,
    host_visible: bool,
    #[allow(dead_code)]
    category: LedgerCategory,
    #[allow(dead_code)]
    call_site: String,
}

/// Global ledger recording every live tracked allocation.
///
/// Counters are only ever mutated while the `entries` mutex is held, so
/// [`AllocationLedger::snapshot`] can read a consistent view (and assert the
/// sum-of-entries == counters invariant in debug builds).
pub struct AllocationLedger {
    entries: Mutex<HashMap<u64, LedgerEntry>>,
    next_id: AtomicU64,
    current_host_visible: AtomicU64,
    current_device_local: AtomicU64,
    peak_host_visible: AtomicU64,
    peak_device_local: AtomicU64,
}

impl AllocationLedger {
    fn new() -> Self {
        Self {
            entries: Mutex::new(HashMap::new()),
            next_id: AtomicU64::new(1),
            current_host_visible: AtomicU64::new(0),
            current_device_local: AtomicU64::new(0),
            peak_host_visible: AtomicU64::new(0),
            peak_device_local: AtomicU64::new(0),
        }
    }

    fn insert(
        &self,
        label: String,
        bytes: u64,
        host_visible: bool,
        category: LedgerCategory,
        call_site: String,
    ) -> u64 {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let mut map = self.entries.lock().unwrap_or_else(|p| p.into_inner());
        map.insert(
            id,
            LedgerEntry {
                label,
                bytes,
                host_visible,
                category,
                call_site,
            },
        );
        if host_visible {
            let cur = self
                .current_host_visible
                .fetch_add(bytes, Ordering::Relaxed)
                + bytes;
            self.peak_host_visible.fetch_max(cur, Ordering::Relaxed);
        } else {
            let cur = self
                .current_device_local
                .fetch_add(bytes, Ordering::Relaxed)
                + bytes;
            self.peak_device_local.fetch_max(cur, Ordering::Relaxed);
        }
        id
    }

    fn remove(&self, id: u64) {
        let mut map = self.entries.lock().unwrap_or_else(|p| p.into_inner());
        if let Some(entry) = map.remove(&id) {
            if entry.host_visible {
                let _ = self.current_host_visible.fetch_update(
                    Ordering::Relaxed,
                    Ordering::Relaxed,
                    |current| Some(current.saturating_sub(entry.bytes)),
                );
            } else {
                let _ = self.current_device_local.fetch_update(
                    Ordering::Relaxed,
                    Ordering::Relaxed,
                    |current| Some(current.saturating_sub(entry.bytes)),
                );
            }
        }
    }

    fn snapshot(&self) -> LedgerReport {
        let map = self.entries.lock().unwrap_or_else(|p| p.into_inner());
        let mut by_label: BTreeMap<String, u64> = BTreeMap::new();
        let mut sum_host_visible = 0u64;
        let mut sum_device_local = 0u64;
        for entry in map.values() {
            *by_label.entry(entry.label.clone()).or_insert(0) += entry.bytes;
            if entry.host_visible {
                sum_host_visible += entry.bytes;
            } else {
                sum_device_local += entry.bytes;
            }
        }
        let current_host_visible_bytes = self.current_host_visible.load(Ordering::Relaxed);
        let current_device_local_bytes = self.current_device_local.load(Ordering::Relaxed);
        #[cfg(debug_assertions)]
        {
            debug_assert_eq!(
                sum_host_visible, current_host_visible_bytes,
                "ledger host-visible sum-of-entries must equal the running counter"
            );
            debug_assert_eq!(
                sum_device_local, current_device_local_bytes,
                "ledger device-local sum-of-entries must equal the running counter"
            );
        }
        let _ = (sum_host_visible, sum_device_local);
        LedgerReport {
            peak_host_visible_bytes: self.peak_host_visible.load(Ordering::Relaxed),
            peak_device_local_bytes: self.peak_device_local.load(Ordering::Relaxed),
            current_host_visible_bytes,
            current_device_local_bytes,
            by_label,
        }
    }
}

/// Immutable snapshot of the [`AllocationLedger`].
#[derive(Clone, Debug)]
pub struct LedgerReport {
    pub peak_host_visible_bytes: u64,
    pub peak_device_local_bytes: u64,
    pub current_host_visible_bytes: u64,
    pub current_device_local_bytes: u64,
    /// Sum of live allocation bytes per label.
    pub by_label: BTreeMap<String, u64>,
}

impl LedgerReport {
    /// The `n` labels consuming the most bytes, largest first (ties broken by label).
    pub fn top_consumers(&self, n: usize) -> Vec<(String, u64)> {
        let mut ranked: Vec<(String, u64)> = self
            .by_label
            .iter()
            .map(|(label, &bytes)| (label.clone(), bytes))
            .collect();
        ranked.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        ranked.truncate(n);
        ranked
    }
}

static LEDGER: OnceLock<AllocationLedger> = OnceLock::new();

/// Access the process-global allocation ledger.
pub fn ledger() -> &'static AllocationLedger {
    LEDGER.get_or_init(AllocationLedger::new)
}

/// Snapshot the global allocation ledger.
pub fn ledger_snapshot() -> LedgerReport {
    ledger().snapshot()
}

/// Render the top-`n` ledger consumers as a `"label=bytes, ..."` string for
/// budget-error messages. Returns `"(none)"` when the ledger is empty.
pub fn ledger_top_consumers_string(n: usize) -> String {
    let top = ledger_snapshot().top_consumers(n);
    if top.is_empty() {
        return "(none)".to_string();
    }
    top.iter()
        .map(|(label, bytes)| format!("{label}={bytes}"))
        .collect::<Vec<_>>()
        .join(", ")
}

// ---------------------------------------------------------------------------
// CENSOR: RAII tracked GPU resource wrappers
// ---------------------------------------------------------------------------

/// A `wgpu::Buffer` whose lifetime is tracked in the global registry + ledger.
///
/// `Deref`s to the inner buffer; dropping removes the ledger entry (the inner
/// `ResourceHandle` frees the registry accounting via its own `Drop`).
pub struct TrackedBuffer {
    inner: wgpu::Buffer,
    _registry: ResourceHandle,
    ledger_id: u64,
}

impl TrackedBuffer {
    /// Explicit accessor for the wrapped buffer (equivalent to `&*self`).
    pub fn inner(&self) -> &wgpu::Buffer {
        &self.inner
    }
}

impl std::ops::Deref for TrackedBuffer {
    type Target = wgpu::Buffer;
    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl Drop for TrackedBuffer {
    fn drop(&mut self) {
        ledger().remove(self.ledger_id);
    }
}

/// A `wgpu::Texture` whose lifetime is tracked in the global registry + ledger.
pub struct TrackedTexture {
    inner: wgpu::Texture,
    _registry: ResourceHandle,
    ledger_id: u64,
}

impl TrackedTexture {
    /// Explicit accessor for the wrapped texture (equivalent to `&*self`).
    pub fn inner(&self) -> &wgpu::Texture {
        &self.inner
    }
}

impl std::ops::Deref for TrackedTexture {
    type Target = wgpu::Texture;
    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl Drop for TrackedTexture {
    fn drop(&mut self) {
        ledger().remove(self.ledger_id);
    }
}

/// Format a `#[track_caller]` location as a cross-platform-stable `file:line`.
///
/// Must be called directly inside a `#[track_caller]` function (there is no
/// automatic propagation through an intermediate helper), so this is a macro.
macro_rules! caller_site {
    () => {{
        let loc = ::std::panic::Location::caller();
        format!("{}:{}", loc.file().replace('\\', "/"), loc.line())
    }};
}

/// Create a buffer, enforcing the host-visible budget policy and recording it
/// in the registry + allocation ledger.
#[track_caller]
pub fn tracked_create_buffer(
    device: &wgpu::Device,
    desc: &BufferDescriptor<'_>,
) -> Result<TrackedBuffer, RenderError> {
    let host_visible = is_host_visible_usage(desc.usage);
    let call_site = caller_site!();
    let label = desc
        .label
        .map(|s| s.to_string())
        .unwrap_or_else(|| call_site.clone());
    if host_visible {
        global_tracker().check_budget_labeled(desc.size, &label)?;
    }
    let buffer = device.create_buffer(desc);
    let registry = register_buffer_explicit(desc.size, host_visible);
    let ledger_id = ledger().insert(
        label,
        desc.size,
        host_visible,
        LedgerCategory::Buffer,
        call_site,
    );
    Ok(TrackedBuffer {
        inner: buffer,
        _registry: registry,
        ledger_id,
    })
}

/// Create a buffer with initial contents (like `DeviceExt::create_buffer_init`),
/// enforcing the host-visible budget policy and recording it in the ledger.
#[track_caller]
pub fn tracked_create_buffer_init(
    device: &wgpu::Device,
    desc: &wgpu::util::BufferInitDescriptor<'_>,
) -> Result<TrackedBuffer, RenderError> {
    let host_visible = is_host_visible_usage(desc.usage);
    let call_site = caller_site!();
    let label = desc
        .label
        .map(|s| s.to_string())
        .unwrap_or_else(|| call_site.clone());
    // Match wgpu's DeviceExt padding: round contents up to COPY_BUFFER_ALIGNMENT.
    let unpadded = desc.contents.len() as u64;
    let align_mask = wgpu::COPY_BUFFER_ALIGNMENT - 1;
    let size = ((unpadded + align_mask) & !align_mask).max(wgpu::COPY_BUFFER_ALIGNMENT);
    if host_visible {
        global_tracker().check_budget_labeled(size, &label)?;
    }
    let buffer = device.create_buffer_init(desc);
    let registry = register_buffer_explicit(size, host_visible);
    let ledger_id = ledger().insert(label, size, host_visible, LedgerCategory::Buffer, call_site);
    Ok(TrackedBuffer {
        inner: buffer,
        _registry: registry,
        ledger_id,
    })
}

/// Create a texture and record it in the registry + allocation ledger.
///
/// Textures are device-local; the 512 MiB host-visible budget does not apply,
/// so no budget check is performed (only registry/ledger accounting).
#[track_caller]
pub fn tracked_create_texture(
    device: &wgpu::Device,
    desc: &TextureDescriptor<'_>,
) -> Result<TrackedTexture, RenderError> {
    let call_site = caller_site!();
    let label = desc
        .label
        .map(|s| s.to_string())
        .unwrap_or_else(|| call_site.clone());
    let size = calculate_texture_size(desc.size.width, desc.size.height, desc.format);
    let texture = device.create_texture(desc);
    let registry = register_texture(desc.size.width, desc.size.height, desc.format);
    let ledger_id = ledger().insert(label, size, false, LedgerCategory::Texture, call_site);
    Ok(TrackedTexture {
        inner: texture,
        _registry: registry,
        ledger_id,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::memory_tracker::ResourceRegistry;

    #[test]
    fn test_resource_handle_cleanup() {
        // Test with isolated registry (can't easily test global one)
        let registry = ResourceRegistry::new();

        // Test buffer handle
        {
            let handle = ResourceHandle::Buffer {
                size: 1024,
                is_host_visible: true,
            };

            // Manually track allocation to simulate what register_buffer does
            registry.track_buffer_allocation(1024, true);

            let metrics = registry.get_metrics();
            assert_eq!(metrics.buffer_count, 1);
            assert_eq!(metrics.buffer_bytes, 1024);
            assert_eq!(metrics.host_visible_bytes, 1024);

            // Now drop the handle (but it will call global_tracker, not our local registry)
            drop(handle);
        }
    }

    #[test]
    fn test_register_buffer_helper() {
        let usage = BufferUsages::COPY_DST | BufferUsages::MAP_READ;
        let initial_metrics = global_tracker().get_metrics();

        {
            let _handle = register_buffer(2048, usage);
            let after_alloc_metrics = global_tracker().get_metrics();

            // Should have increased by our allocation
            assert_eq!(
                after_alloc_metrics.buffer_count,
                initial_metrics.buffer_count + 1
            );
            assert_eq!(
                after_alloc_metrics.buffer_bytes,
                initial_metrics.buffer_bytes + 2048
            );
            assert_eq!(
                after_alloc_metrics.host_visible_bytes,
                initial_metrics.host_visible_bytes + 2048
            );
        }

        // After handle drop, should return to initial state
        let final_metrics = global_tracker().get_metrics();
        assert_eq!(final_metrics.buffer_count, initial_metrics.buffer_count);
        assert_eq!(final_metrics.buffer_bytes, initial_metrics.buffer_bytes);
        assert_eq!(
            final_metrics.host_visible_bytes,
            initial_metrics.host_visible_bytes
        );
    }

    #[test]
    fn test_register_texture_helper() {
        let initial_metrics = global_tracker().get_metrics();

        {
            let _handle = register_texture(512, 512, TextureFormat::Rgba8Unorm);
            let after_alloc_metrics = global_tracker().get_metrics();

            // Should have increased by our allocation (512*512*4 = 1,048,576 bytes)
            assert_eq!(
                after_alloc_metrics.texture_count,
                initial_metrics.texture_count + 1
            );
            assert_eq!(
                after_alloc_metrics.texture_bytes,
                initial_metrics.texture_bytes + 1_048_576
            );
        }

        // After handle drop, should return to initial state
        let final_metrics = global_tracker().get_metrics();
        assert_eq!(final_metrics.texture_count, initial_metrics.texture_count);
        assert_eq!(final_metrics.texture_bytes, initial_metrics.texture_bytes);
    }

    /// Restore the budget policy at the end of a test that mutated it.
    struct PolicyGuard(&'static str);
    impl Drop for PolicyGuard {
        fn drop(&mut self) {
            let _ = global_tracker().set_budget_policy(self.0);
        }
    }
    fn save_policy() -> PolicyGuard {
        PolicyGuard(global_tracker().get_budget_policy())
    }

    #[test]
    fn test_enforce_error_names_label_and_top_consumers() {
        let _guard = save_policy();
        let _ = global_tracker().set_budget_policy("enforce");

        // A 600 MiB host-visible request exceeds the 512 MiB limit even with an
        // otherwise-empty budget, so this trips regardless of concurrent state.
        let bytes = 600u64 * 1024 * 1024;
        let err = global_tracker()
            .check_budget_labeled(bytes, "unit-test-blob")
            .expect_err("600 MiB host-visible request must exceed the budget under enforce");
        let msg = err.to_string();
        assert!(
            msg.contains("Memory budget exceeded"),
            "message keeps the legacy prefix: {msg}"
        );
        assert!(
            msg.contains("unit-test-blob"),
            "message names the offending label: {msg}"
        );
        assert!(
            msg.contains("top consumers"),
            "message names the top consumers: {msg}"
        );
    }

    #[test]
    fn test_drop_removes_ledger_entry_and_decrements_counters() {
        let device = match crate::core::gpu::create_device_for_test() {
            Some(d) => d,
            None => {
                eprintln!("skipping: no GPU adapter");
                return;
            }
        };

        let before = ledger_snapshot();
        let size = 4096u64;
        {
            let _buf = tracked_create_buffer(
                &device,
                &BufferDescriptor {
                    label: Some("drop-test-buffer"),
                    size,
                    usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
                    mapped_at_creation: false,
                },
            )
            .expect("small host-visible allocation must succeed");

            let during = ledger_snapshot();
            assert_eq!(
                during.current_host_visible_bytes,
                before.current_host_visible_bytes + size,
                "ledger host-visible counter increments while the buffer is live"
            );
            assert_eq!(
                during.by_label.get("drop-test-buffer").copied(),
                Some(size),
                "ledger records the labeled entry"
            );
        }

        let after = ledger_snapshot();
        assert_eq!(
            after.current_host_visible_bytes, before.current_host_visible_bytes,
            "dropping the TrackedBuffer decrements the ledger counter"
        );
        assert!(
            !after.by_label.contains_key("drop-test-buffer"),
            "dropping the TrackedBuffer removes the ledger entry"
        );

        device.poll(wgpu::Maintain::Wait);
        std::mem::forget(device);
    }

    #[test]
    fn test_invariant_holds_after_interleaved_alloc_free() {
        let device = match crate::core::gpu::create_device_for_test() {
            Some(d) => d,
            None => {
                eprintln!("skipping: no GPU adapter");
                return;
            }
        };

        let mk = |label: &'static str, size: u64| {
            tracked_create_buffer(
                &device,
                &BufferDescriptor {
                    label: Some(label),
                    size,
                    usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
                    mapped_at_creation: false,
                },
            )
            .expect("allocation must succeed")
        };

        let before = ledger_snapshot();
        let a = mk("inv-a", 1024);
        let b = mk("inv-b", 2048);
        // snapshot() debug-asserts the sum==counter invariant internally.
        let mid = ledger_snapshot();
        assert_eq!(
            mid.current_host_visible_bytes,
            before.current_host_visible_bytes + 1024 + 2048
        );
        drop(a);
        let c = mk("inv-c", 512);
        let _ = ledger_snapshot();
        drop(b);
        drop(c);

        let after = ledger_snapshot();
        assert_eq!(
            after.current_host_visible_bytes, before.current_host_visible_bytes,
            "counters return to baseline after all frees"
        );

        device.poll(wgpu::Maintain::Wait);
        std::mem::forget(device);
    }

    #[test]
    fn test_none_label_falls_back_to_call_site() {
        let device = match crate::core::gpu::create_device_for_test() {
            Some(d) => d,
            None => {
                eprintln!("skipping: no GPU adapter");
                return;
            }
        };

        let buf = tracked_create_buffer(
            &device,
            &BufferDescriptor {
                label: None,
                size: 256,
                usage: BufferUsages::COPY_DST,
                mapped_at_creation: false,
            },
        )
        .expect("device-local allocation must succeed");

        let snap = ledger_snapshot();
        // The fallback label is this file's path (forward-slashed) + a line number.
        let has_call_site = snap
            .by_label
            .keys()
            .any(|k| k.contains("resource_tracker.rs:") && !k.contains('\\'));
        assert!(
            has_call_site,
            "None-label allocation falls back to a normalized call-site label; labels: {:?}",
            snap.by_label.keys().collect::<Vec<_>>()
        );

        drop(buf);
        device.poll(wgpu::Maintain::Wait);
        std::mem::forget(device);
    }
}
