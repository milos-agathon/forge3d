use serde::{Deserialize, Serialize};

/// Per-render cache activity. Labels remain in deterministic execution order.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct CacheReport {
    pub hits: Vec<String>,
    pub misses: Vec<String>,
    pub bytes_read: u64,
    pub bytes_written: u64,
    pub wall_ms_saved: f64,
}

impl CacheReport {
    pub fn hit_rate(&self) -> f64 {
        let total = self.hits.len() + self.misses.len();
        if total == 0 {
            0.0
        } else {
            self.hits.len() as f64 / total as f64
        }
    }
}
