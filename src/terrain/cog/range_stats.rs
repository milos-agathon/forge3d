//! Aggregate request/cache statistics for [`super::range_reader::RangeReader`].

/// Statistics for range reader operations.
#[derive(Debug, Default)]
pub struct RangeReaderStats {
    pub requests: std::sync::atomic::AtomicU64,
    pub bytes_fetched: std::sync::atomic::AtomicU64,
    pub cache_hits: std::sync::atomic::AtomicU64,
    pub(crate) cached_bytes: std::sync::atomic::AtomicU64,
    pub(crate) disk_cached_bytes: std::sync::atomic::AtomicU64,
    pub total_latency_ms: std::sync::atomic::AtomicU64,
}

impl RangeReaderStats {
    pub fn requests(&self) -> u64 {
        self.requests.load(std::sync::atomic::Ordering::Relaxed)
    }

    pub fn bytes_fetched(&self) -> u64 {
        self.bytes_fetched
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    pub fn cache_hits(&self) -> u64 {
        self.cache_hits.load(std::sync::atomic::Ordering::Relaxed)
    }

    pub fn cached_bytes(&self) -> u64 {
        self.cached_bytes.load(std::sync::atomic::Ordering::Relaxed)
    }

    pub fn disk_cached_bytes(&self) -> u64 {
        self.disk_cached_bytes
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    pub fn avg_latency_ms(&self) -> f64 {
        let reqs = self.requests();
        if reqs == 0 {
            0.0
        } else {
            self.total_latency_ms
                .load(std::sync::atomic::Ordering::Relaxed) as f64
                / reqs as f64
        }
    }
}
