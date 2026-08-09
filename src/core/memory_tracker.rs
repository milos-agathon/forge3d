#[path = "memory_tracker/helpers.rs"]
mod helpers;
#[path = "memory_tracker/pool.rs"]
#[cfg(not(target_arch = "wasm32"))]
mod pool;
#[path = "memory_tracker/registry.rs"]
mod registry;
#[path = "memory_tracker/reporting.rs"]
mod reporting;
#[cfg(test)]
#[path = "memory_tracker/tests.rs"]
mod tests;
#[path = "memory_tracker/types.rs"]
mod types;

pub use helpers::{
    calculate_compressed_texture_size, calculate_texture_size, is_host_visible_usage,
};
#[cfg(not(target_arch = "wasm32"))]
pub use pool::{global_pools, init_global_pools, MemoryPoolManager, PoolBlock};
pub use registry::{global_tracker, ResourceRegistry};
pub use types::{DefragStats, MemoryMetrics, MemoryPoolStats};

pub(crate) const MEMORY_BUDGET_LIMIT: u64 = 512 * 1024 * 1024;
