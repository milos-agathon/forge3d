//! P3.3: COG tile cache with memory budget enforcement.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;

/// Cache key for COG tiles: (tile_x, tile_y, lod).
pub type CogTileCacheKey = (u32, u32, u32);

/// Statistics for COG tile cache.
#[derive(Debug, Clone, Default)]
pub struct CogCacheStats {
    pub hits: u64,
    pub misses: u64,
    pub evictions: u64,
    pub memory_used_bytes: u64,
    pub memory_budget_bytes: u64,
}

/// LRU entry for cache.
struct CacheEntry {
    data: Vec<f32>,
    memory_bytes: usize,
    last_access: u64,
}

/// COG tile cache with LRU eviction and memory budget.
pub struct CogTileCache {
    entries: Mutex<HashMap<CogTileCacheKey, CacheEntry>>,
    lru_order: Mutex<Vec<CogTileCacheKey>>,
    memory_budget_bytes: u64,
    current_memory: AtomicU64,
    access_counter: AtomicU64,
    hits: AtomicU64,
    misses: AtomicU64,
    evictions: AtomicU64,
}

impl CogTileCache {
    /// Create a new cache with the given memory budget in MB.
    pub fn new(budget_mb: u32) -> Self {
        let budget_bytes = (budget_mb as u64) * 1024 * 1024;
        Self {
            entries: Mutex::new(HashMap::new()),
            lru_order: Mutex::new(Vec::new()),
            memory_budget_bytes: budget_bytes,
            current_memory: AtomicU64::new(0),
            access_counter: AtomicU64::new(0),
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
            evictions: AtomicU64::new(0),
        }
    }

    /// Get a tile from the cache if present.
    pub fn get(&self, key: &CogTileCacheKey) -> Option<Vec<f32>> {
        let mut entries = self.entries.lock().ok()?;

        if let Some(entry) = entries.get_mut(key) {
            entry.last_access = self.access_counter.fetch_add(1, Ordering::Relaxed);
            self.hits.fetch_add(1, Ordering::Relaxed);

            if let Ok(mut lru) = self.lru_order.lock() {
                if let Some(pos) = lru.iter().position(|k| k == key) {
                    lru.remove(pos);
                }
                lru.push(*key);
            }

            Some(entry.data.clone())
        } else {
            self.misses.fetch_add(1, Ordering::Relaxed);
            None
        }
    }

    /// Insert a tile into the cache.
    pub fn insert(&self, key: CogTileCacheKey, data: Vec<f32>, memory_bytes: usize) {
        self.evict_to_budget(memory_bytes);

        let access = self.access_counter.fetch_add(1, Ordering::Relaxed);

        if let Ok(mut entries) = self.entries.lock() {
            if entries.contains_key(&key) {
                return;
            }

            entries.insert(
                key,
                CacheEntry {
                    data,
                    memory_bytes,
                    last_access: access,
                },
            );

            self.current_memory
                .fetch_add(memory_bytes as u64, Ordering::Relaxed);

            if let Ok(mut lru) = self.lru_order.lock() {
                lru.push(key);
            }
        }
    }

    /// Evict tiles until there's room for new_bytes.
    fn evict_to_budget(&self, new_bytes: usize) {
        let target = self.memory_budget_bytes.saturating_sub(new_bytes as u64);

        while self.current_memory.load(Ordering::Relaxed) > target {
            let key_to_evict = {
                let lru = self.lru_order.lock().ok();
                lru.and_then(|l| l.first().copied())
            };

            if let Some(key) = key_to_evict {
                self.evict(&key);
            } else {
                break;
            }
        }
    }

    /// Evict a specific tile.
    fn evict(&self, key: &CogTileCacheKey) {
        if let Ok(mut entries) = self.entries.lock() {
            if let Some(entry) = entries.remove(key) {
                self.current_memory
                    .fetch_sub(entry.memory_bytes as u64, Ordering::Relaxed);
                self.evictions.fetch_add(1, Ordering::Relaxed);

                if let Ok(mut lru) = self.lru_order.lock() {
                    if let Some(pos) = lru.iter().position(|k| k == key) {
                        lru.remove(pos);
                    }
                }
            }
        }
    }

    /// Get cache statistics.
    pub fn stats(&self) -> CogCacheStats {
        CogCacheStats {
            hits: self.hits.load(Ordering::Relaxed),
            misses: self.misses.load(Ordering::Relaxed),
            evictions: self.evictions.load(Ordering::Relaxed),
            memory_used_bytes: self.current_memory.load(Ordering::Relaxed),
            memory_budget_bytes: self.memory_budget_bytes,
        }
    }

    /// Clear all entries from the cache.
    pub fn clear(&self) {
        if let Ok(mut entries) = self.entries.lock() {
            entries.clear();
        }
        if let Ok(mut lru) = self.lru_order.lock() {
            lru.clear();
        }
        self.current_memory.store(0, Ordering::Relaxed);
    }

    /// Get current memory usage in bytes.
    pub fn memory_used(&self) -> u64 {
        self.current_memory.load(Ordering::Relaxed)
    }

    /// Get memory budget in bytes.
    pub fn memory_budget(&self) -> u64 {
        self.memory_budget_bytes
    }

    /// Get number of cached tiles.
    pub fn tile_count(&self) -> usize {
        self.entries.lock().map(|e| e.len()).unwrap_or(0)
    }
}
