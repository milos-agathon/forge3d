use super::helpers::calculate_texture_size;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use wgpu::TextureFormat;

const MEMORY_BUDGET_LIMIT: u64 = 512 * 1024 * 1024;

/// Global memory tracking registry for GPU resources.
pub struct ResourceRegistry {
    pub(super) buffer_count: AtomicU32,
    pub(super) texture_count: AtomicU32,
    pub(super) buffer_bytes: AtomicU64,
    pub(super) texture_bytes: AtomicU64,
    pub(super) host_visible_bytes: AtomicU64,
    pub(super) resident_tiles: AtomicU32,
    pub(super) resident_tile_bytes: AtomicU64,
    pub(super) staging_bytes_in_flight: AtomicU64,
    pub(super) staging_ring_count: AtomicU32,
    pub(super) staging_buffer_size: AtomicU64,
    pub(super) staging_buffer_stalls: AtomicU64,
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
            resident_tiles: AtomicU32::new(0),
            resident_tile_bytes: AtomicU64::new(0),
            staging_bytes_in_flight: AtomicU64::new(0),
            staging_ring_count: AtomicU32::new(0),
            staging_buffer_size: AtomicU64::new(0),
            staging_buffer_stalls: AtomicU64::new(0),
            budget_limit: MEMORY_BUDGET_LIMIT,
        }
    }

    pub fn track_buffer_allocation(&self, size: u64, is_host_visible: bool) {
        self.buffer_count.fetch_add(1, Ordering::Relaxed);
        self.buffer_bytes.fetch_add(size, Ordering::Relaxed);

        if is_host_visible {
            self.host_visible_bytes.fetch_add(size, Ordering::Relaxed);
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
        self.texture_count.fetch_add(1, Ordering::Relaxed);
        self.texture_bytes.fetch_add(size, Ordering::Relaxed);
    }

    pub fn free_texture_allocation(&self, width: u32, height: u32, format: TextureFormat) {
        let size = calculate_texture_size(width, height, format);

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
}

static GLOBAL_REGISTRY: std::sync::OnceLock<ResourceRegistry> = std::sync::OnceLock::new();

pub fn global_tracker() -> &'static ResourceRegistry {
    GLOBAL_REGISTRY.get_or_init(ResourceRegistry::new)
}
