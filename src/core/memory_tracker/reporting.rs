use super::{registry::ResourceRegistry, types::MemoryMetrics};
use std::sync::atomic::Ordering;

impl ResourceRegistry {
    pub fn get_metrics(&self) -> MemoryMetrics {
        let buffer_count = self.buffer_count.load(Ordering::Relaxed);
        let texture_count = self.texture_count.load(Ordering::Relaxed);
        let buffer_bytes = self.buffer_bytes.load(Ordering::Relaxed);
        let texture_bytes = self.texture_bytes.load(Ordering::Relaxed);
        let host_visible_bytes = self.host_visible_bytes.load(Ordering::Relaxed);
        let total_bytes = buffer_bytes + texture_bytes;
        let within_budget = host_visible_bytes <= self.budget_limit;
        let utilization_ratio = host_visible_bytes as f64 / self.budget_limit as f64;
        let resident_tiles = self.resident_tiles.load(Ordering::Relaxed);
        let resident_tile_bytes = self.resident_tile_bytes.load(Ordering::Relaxed);
        let staging_bytes_in_flight = self.staging_bytes_in_flight.load(Ordering::Relaxed);
        let staging_ring_count = self.staging_ring_count.load(Ordering::Relaxed);
        let staging_buffer_size = self.staging_buffer_size.load(Ordering::Relaxed);
        let staging_buffer_stalls = self.staging_buffer_stalls.load(Ordering::Relaxed);

        MemoryMetrics {
            buffer_count,
            texture_count,
            buffer_bytes,
            texture_bytes,
            host_visible_bytes,
            total_bytes,
            limit_bytes: self.budget_limit,
            within_budget,
            utilization_ratio,
            resident_tiles,
            resident_tile_bytes,
            staging_bytes_in_flight,
            staging_ring_count,
            staging_buffer_size,
            staging_buffer_stalls,
        }
    }

    pub fn get_budget_limit(&self) -> u64 {
        self.budget_limit
    }

    pub fn check_budget(&self, additional_host_visible: u64) -> Result<(), String> {
        let current = self.host_visible_bytes.load(Ordering::Relaxed);
        if current.saturating_add(additional_host_visible) > self.budget_limit {
            return Err(format!(
                "Memory budget exceeded: current {} bytes + requested {} bytes would exceed limit of {} bytes",
                current, additional_host_visible, self.budget_limit
            ));
        }
        Ok(())
    }
}
