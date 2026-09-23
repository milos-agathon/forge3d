//! P3.3: TIFF physical-tile cache with atomic LRU/accounting state.

use std::collections::{HashMap, VecDeque};
use std::sync::Mutex;

/// Stable physical identity: (IFD/overview index, physical tile x, tile y).
pub type CogTileCacheKey = (u32, u32, u32);

#[derive(Debug, Clone, Default)]
pub struct CogCacheStats {
    pub hits: u64,
    pub misses: u64,
    pub evictions: u64,
    pub memory_used_bytes: u64,
    pub memory_budget_bytes: u64,
    pub byte_cache_used_bytes: u64,
    pub byte_cache_budget_bytes: u64,
    pub disk_cache_used_bytes: u64,
    pub disk_cache_budget_bytes: u64,
}

struct CacheEntry {
    data: Vec<f32>,
    memory_bytes: usize,
}

#[derive(Default)]
struct CacheState {
    entries: HashMap<CogTileCacheKey, CacheEntry>,
    /// Least recent at the front. Every key occurs exactly once.
    lru: VecDeque<CogTileCacheKey>,
    used_bytes: u64,
    high_water_bytes: u64,
    hits: u64,
    misses: u64,
    evictions: u64,
}

pub struct CogTileCache {
    state: Mutex<CacheState>,
    memory_budget_bytes: u64,
}

impl CogTileCache {
    pub fn new(budget_mb: u32) -> Self {
        Self {
            state: Mutex::new(CacheState::default()),
            memory_budget_bytes: u64::from(budget_mb) * 1024 * 1024,
        }
    }

    pub fn get(&self, key: &CogTileCacheKey) -> Option<Vec<f32>> {
        let mut state = self.state.lock().unwrap_or_else(|p| p.into_inner());
        let data = match state.entries.get(key) {
            Some(entry) => entry.data.clone(),
            None => {
                state.misses += 1;
                return None;
            }
        };
        state.hits += 1;
        state.lru.retain(|candidate| candidate != key);
        state.lru.push_back(*key);
        Some(data)
    }

    pub fn insert(&self, key: CogTileCacheKey, data: Vec<f32>, memory_bytes: usize) {
        let incoming = memory_bytes as u64;
        let mut state = self.state.lock().unwrap_or_else(|p| p.into_inner());
        if incoming > self.memory_budget_bytes {
            return;
        }
        if state.entries.contains_key(&key) {
            state.lru.retain(|candidate| candidate != &key);
            state.lru.push_back(key);
            return;
        }
        while state.used_bytes.saturating_add(incoming) > self.memory_budget_bytes {
            let Some(victim) = state.lru.pop_front() else {
                return;
            };
            if let Some(evicted) = state.entries.remove(&victim) {
                state.used_bytes = state.used_bytes.saturating_sub(evicted.memory_bytes as u64);
                state.evictions += 1;
            }
        }
        state.entries.insert(key, CacheEntry { data, memory_bytes });
        state.lru.push_back(key);
        state.used_bytes += incoming;
        state.high_water_bytes = state.high_water_bytes.max(state.used_bytes);
        debug_assert_eq!(
            state.used_bytes,
            state
                .entries
                .values()
                .map(|entry| entry.memory_bytes as u64)
                .sum::<u64>()
        );
    }

    pub fn stats(&self) -> CogCacheStats {
        let state = self.state.lock().unwrap_or_else(|p| p.into_inner());
        CogCacheStats {
            hits: state.hits,
            misses: state.misses,
            evictions: state.evictions,
            memory_used_bytes: state.used_bytes,
            memory_budget_bytes: self.memory_budget_bytes,
            byte_cache_used_bytes: 0,
            byte_cache_budget_bytes: 0,
            disk_cache_used_bytes: 0,
            disk_cache_budget_bytes: 0,
        }
    }

    pub fn high_water_bytes(&self) -> u64 {
        self.state
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .high_water_bytes
    }

    pub fn clear(&self) {
        *self.state.lock().unwrap_or_else(|p| p.into_inner()) = CacheState::default();
    }

    pub fn memory_used(&self) -> u64 {
        self.state
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .used_bytes
    }

    pub fn memory_budget(&self) -> u64 {
        self.memory_budget_bytes
    }

    pub fn tile_count(&self) -> usize {
        self.state
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .entries
            .len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Barrier};

    #[test]
    fn duplicate_and_oversize_inserts_do_not_evict_resident_tiles() {
        let cache = CogTileCache::new(1);
        let a = (0, 0, 0);
        let b = (0, 1, 0);
        cache.insert(a, vec![1.0], 700 * 1024);
        cache.insert(b, vec![2.0], 300 * 1024);
        cache.insert(a, vec![9.0], 700 * 1024);
        cache.insert((0, 2, 0), vec![3.0], 2 * 1024 * 1024);
        assert_eq!(cache.get(&a), Some(vec![1.0]));
        assert_eq!(cache.get(&b), Some(vec![2.0]));
        assert_eq!(cache.memory_used(), 1000 * 1024);
    }

    #[test]
    fn runtime_get_touch_drives_true_lru_eviction() {
        let cache = CogTileCache::new(1);
        let a = (0, 0, 0);
        let b = (0, 1, 0);
        let c = (1, 0, 0);
        cache.insert(a, vec![1.0], 400 * 1024);
        cache.insert(b, vec![2.0], 400 * 1024);
        assert!(cache.get(&a).is_some());
        cache.insert(c, vec![3.0], 400 * 1024);
        assert!(cache.get(&a).is_some());
        assert!(cache.get(&b).is_none());
        assert!(cache.get(&c).is_some());
    }

    #[test]
    fn concurrent_insert_accounting_is_atomic_and_bounded() {
        let cache = Arc::new(CogTileCache::new(1));
        let barrier = Arc::new(Barrier::new(9));
        let mut threads = Vec::new();
        for x in 0..8 {
            let cache = cache.clone();
            let barrier = barrier.clone();
            threads.push(std::thread::spawn(move || {
                barrier.wait();
                cache.insert((0, x, 0), vec![x as f32], 256 * 1024);
            }));
        }
        barrier.wait();
        for thread in threads {
            thread.join().unwrap();
        }
        assert!(cache.memory_used() <= cache.memory_budget());
        assert!(cache.high_water_bytes() <= cache.memory_budget());
        assert_eq!(cache.memory_used(), cache.tile_count() as u64 * 256 * 1024);
    }
}
