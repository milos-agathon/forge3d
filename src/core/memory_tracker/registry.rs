use super::helpers::calculate_texture_size;
use std::sync::atomic::{AtomicU32, AtomicU64, AtomicU8, Ordering};
use wgpu::TextureFormat;

const MEMORY_BUDGET_LIMIT: u64 = 512 * 1024 * 1024;
const BUDGET_POLICY_ENFORCE: u8 = 0;
const BUDGET_POLICY_WARN: u8 = 1;

/// Global memory tracking registry for GPU resources.
pub struct ResourceRegistry {
    pub(super) buffer_count: AtomicU32,
    pub(super) texture_count: AtomicU32,
    pub(super) buffer_bytes: AtomicU64,
    pub(super) texture_bytes: AtomicU64,
    pub(super) host_visible_bytes: AtomicU64,
    // Exact subset owned by resource_tracker::ResourceHandle. Unlike the
    // public memory totals, these exclude estimate-only legacy bookkeeping and
    // therefore must exactly equal the allocation ledger.
    pub(super) ledger_host_visible_bytes: AtomicU64,
    pub(super) ledger_device_local_bytes: AtomicU64,
    pub(super) peak_host_visible_bytes: AtomicU64,
    pub(super) peak_total_bytes: AtomicU64,
    pub(super) resident_tiles: AtomicU32,
    pub(super) resident_tile_bytes: AtomicU64,
    pub(super) staging_bytes_in_flight: AtomicU64,
    pub(super) staging_ring_count: AtomicU32,
    pub(super) staging_buffer_size: AtomicU64,
    pub(super) staging_buffer_stalls: AtomicU64,
    pub(super) budget_policy: AtomicU8,
    pub(super) budget_limit: u64,
}

impl ResourceRegistry {
    pub fn new() -> Self {
        Self {
            buffer_count: AtomicU32::new(0),
            texture_count: AtomicU32::new(0),
            buffer_bytes: AtomicU64::new(0),
            texture_bytes: AtomicU64::new(0),
            host_visible_bytes: AtomicU64::new(0),
            ledger_host_visible_bytes: AtomicU64::new(0),
            ledger_device_local_bytes: AtomicU64::new(0),
            peak_host_visible_bytes: AtomicU64::new(0),
            peak_total_bytes: AtomicU64::new(0),
            resident_tiles: AtomicU32::new(0),
            resident_tile_bytes: AtomicU64::new(0),
            staging_bytes_in_flight: AtomicU64::new(0),
            staging_ring_count: AtomicU32::new(0),
            staging_buffer_size: AtomicU64::new(0),
            staging_buffer_stalls: AtomicU64::new(0),
            budget_policy: AtomicU8::new(BUDGET_POLICY_ENFORCE),
            budget_limit: MEMORY_BUDGET_LIMIT,
        }
    }

    pub fn set_budget_policy(&self, policy: &str) -> Result<&'static str, String> {
        let normalized = match policy {
            "enforce" => {
                self.budget_policy
                    .store(BUDGET_POLICY_ENFORCE, Ordering::Relaxed);
                "enforce"
            }
            "warn" => {
                self.budget_policy
                    .store(BUDGET_POLICY_WARN, Ordering::Relaxed);
                "warn"
            }
            _ => {
                return Err(format!(
                    "Unknown memory budget policy {policy:?}; expected 'enforce' or 'warn'"
                ));
            }
        };
        Ok(normalized)
    }

    pub fn get_budget_policy(&self) -> &'static str {
        match self.budget_policy.load(Ordering::Relaxed) {
            BUDGET_POLICY_WARN => "warn",
            _ => "enforce",
        }
    }

    pub fn track_buffer_allocation(&self, size: u64, is_host_visible: bool) {
        self.buffer_count.fetch_add(1, Ordering::Relaxed);
        let buffer_bytes = self.buffer_bytes.fetch_add(size, Ordering::Relaxed) + size;
        let texture_bytes = self.texture_bytes.load(Ordering::Relaxed);
        self.record_peak_total(buffer_bytes.saturating_add(texture_bytes));

        if is_host_visible {
            let host_visible = self.host_visible_bytes.fetch_add(size, Ordering::Relaxed) + size;
            self.record_peak_host_visible(host_visible);
        }
    }

    pub fn free_buffer_allocation(&self, size: u64, is_host_visible: bool) {
        let _ = self
            .buffer_count
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |current| {
                Some(current.saturating_sub(1))
            });
        let _ = self
            .buffer_bytes
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |current| {
                Some(current.saturating_sub(size))
            });

        if is_host_visible {
            let _ = self.host_visible_bytes.fetch_update(
                Ordering::Relaxed,
                Ordering::Relaxed,
                |current| Some(current.saturating_sub(size)),
            );
        }
    }

    pub fn track_texture_allocation(&self, width: u32, height: u32, format: TextureFormat) {
        let size = calculate_texture_size(width, height, format);
        self.track_texture_allocation_bytes(size);
    }

    pub fn track_texture_allocation_bytes(&self, size: u64) {
        self.texture_count.fetch_add(1, Ordering::Relaxed);
        let texture_bytes = self.texture_bytes.fetch_add(size, Ordering::Relaxed) + size;
        let buffer_bytes = self.buffer_bytes.load(Ordering::Relaxed);
        self.record_peak_total(buffer_bytes.saturating_add(texture_bytes));
    }

    pub fn free_texture_allocation(&self, width: u32, height: u32, format: TextureFormat) {
        let size = calculate_texture_size(width, height, format);
        self.free_texture_allocation_bytes(size);
    }

    pub fn free_texture_allocation_bytes(&self, size: u64) {
        let _ = self
            .texture_count
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |current| {
                Some(current.saturating_sub(1))
            });
        let _ = self
            .texture_bytes
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |current| {
                Some(current.saturating_sub(size))
            });
    }

    /// Record one allocation that also owns an allocation-ledger entry.
    pub fn track_ledger_allocation(&self, size: u64, is_host_visible: bool) {
        let counter = if is_host_visible {
            &self.ledger_host_visible_bytes
        } else {
            &self.ledger_device_local_bytes
        };
        counter.fetch_add(size, Ordering::Relaxed);
    }

    /// Remove one allocation that also owned an allocation-ledger entry.
    pub fn free_ledger_allocation(&self, size: u64, is_host_visible: bool) {
        let counter = if is_host_visible {
            &self.ledger_host_visible_bytes
        } else {
            &self.ledger_device_local_bytes
        };
        let _ = counter.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |current| {
            Some(current.saturating_sub(size))
        });
    }

    /// Current registry totals for allocations that have ledger entries.
    pub fn ledger_totals(&self) -> (u64, u64) {
        (
            self.ledger_host_visible_bytes.load(Ordering::Relaxed),
            self.ledger_device_local_bytes.load(Ordering::Relaxed),
        )
    }

    pub fn set_resident_tiles(&self, count: u32, tile_bytes: u64) {
        self.resident_tiles.store(count, Ordering::Relaxed);
        self.resident_tile_bytes
            .store(tile_bytes, Ordering::Relaxed);
    }

    pub fn clear_resident_tiles(&self) {
        self.set_resident_tiles(0, 0);
    }

    pub fn set_staging_stats(
        &self,
        bytes_in_flight: u64,
        ring_count: usize,
        buffer_size: u64,
        stalls: u64,
    ) {
        self.staging_bytes_in_flight
            .store(bytes_in_flight, Ordering::Relaxed);
        self.staging_ring_count
            .store(ring_count as u32, Ordering::Relaxed);
        self.staging_buffer_size
            .store(buffer_size, Ordering::Relaxed);
        self.staging_buffer_stalls.store(stalls, Ordering::Relaxed);
    }

    pub fn clear_staging_stats(&self) {
        self.set_staging_stats(0, 0, 0, 0);
    }

    fn record_peak_total(&self, value: u64) {
        let _ =
            self.peak_total_bytes
                .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |current| {
                    Some(current.max(value))
                });
    }

    fn record_peak_host_visible(&self, value: u64) {
        let _ = self.peak_host_visible_bytes.fetch_update(
            Ordering::Relaxed,
            Ordering::Relaxed,
            |current| Some(current.max(value)),
        );
    }
}

static GLOBAL_REGISTRY: std::sync::OnceLock<ResourceRegistry> = std::sync::OnceLock::new();

pub fn global_tracker() -> &'static ResourceRegistry {
    GLOBAL_REGISTRY.get_or_init(ResourceRegistry::new)
}
