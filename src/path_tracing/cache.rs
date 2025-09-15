//! Scene cache for high-quality path tracing
//!
//! Implements caching of BVH structures, material data, and texture bindings
//! to accelerate repeated renders with identical scene configurations.
//! Target: ≥30% faster re-renders with pixel-perfect identical output.

use std::collections::HashMap;
use wgpu::*;

/// Cache entry for reusable GPU resources
#[derive(Debug)]
pub struct CacheEntry {
    /// BVH buffer handle
    pub bvh_buffer: Option<Buffer>,
    /// Material data buffer
    pub material_buffer: Option<Buffer>,
    /// Texture bind group
    pub texture_bind_group: Option<BindGroup>,
    /// Cached hash for validation
    pub content_hash: u64,
    /// Creation timestamp for LRU eviction
    pub timestamp: std::time::Instant,
}

impl CacheEntry {
    pub fn new(content_hash: u64) -> Self {
        Self {
            bvh_buffer: None,
            material_buffer: None,
            texture_bind_group: None,
            content_hash,
            timestamp: std::time::Instant::now(),
        }
    }
}

/// Scene cache for GPU resources
pub struct SceneCache {
    /// Cache entries by scene content hash
    entries: HashMap<u64, CacheEntry>,
    /// Maximum cache entries
    max_entries: usize,
    /// GPU device reference
    device: std::sync::Arc<Device>,
}

impl SceneCache {
    /// Create new scene cache
    pub fn new(device: std::sync::Arc<Device>) -> Self {
        Self {
            entries: HashMap::new(),
            max_entries: 16, // Reasonable default for GPU memory
            device,
        }
    }

    /// Set maximum cache entries
    pub fn set_max_entries(&mut self, max_entries: usize) {
        self.max_entries = max_entries;
        self.evict_lru();
    }

    /// Compute content hash for scene
    pub fn compute_scene_hash(&self,
        bvh_data: &[u8],
        material_data: &[u8],
        texture_ids: &[u32]) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        bvh_data.hash(&mut hasher);
        material_data.hash(&mut hasher);
        texture_ids.hash(&mut hasher);
        hasher.finish()
    }

    /// Get cached entry if available and valid
    pub fn get_entry(&self, content_hash: u64) -> Option<&CacheEntry> {
        self.entries.get(&content_hash)
    }

    /// Cache BVH buffer
    pub fn cache_bvh_buffer(&mut self, content_hash: u64, buffer: Buffer) {
        let entry = self.entries.entry(content_hash)
            .or_insert_with(|| CacheEntry::new(content_hash));
        entry.bvh_buffer = Some(buffer);
        entry.timestamp = std::time::Instant::now();
        self.evict_lru();
    }

    /// Cache material buffer
    pub fn cache_material_buffer(&mut self, content_hash: u64, buffer: Buffer) {
        let entry = self.entries.entry(content_hash)
            .or_insert_with(|| CacheEntry::new(content_hash));
        entry.material_buffer = Some(buffer);
        entry.timestamp = std::time::Instant::now();
        self.evict_lru();
    }

    /// Cache texture bind group
    pub fn cache_texture_bind_group(&mut self, content_hash: u64, bind_group: BindGroup) {
        let entry = self.entries.entry(content_hash)
            .or_insert_with(|| CacheEntry::new(content_hash));
        entry.texture_bind_group = Some(bind_group);
        entry.timestamp = std::time::Instant::now();
        self.evict_lru();
    }

    /// Check if scene is cached and complete
    pub fn is_scene_cached(&self, content_hash: u64) -> bool {
        if let Some(entry) = self.entries.get(&content_hash) {
            entry.bvh_buffer.is_some() &&
            entry.material_buffer.is_some() &&
            entry.texture_bind_group.is_some()
        } else {
            false
        }
    }

    /// Reset cache (clear all entries)
    pub fn reset(&mut self) {
        self.entries.clear();
    }

    /// Get cache statistics
    pub fn get_stats(&self) -> CacheStats {
        let total_entries = self.entries.len();
        let complete_entries = self.entries.values()
            .filter(|e| e.bvh_buffer.is_some() && e.material_buffer.is_some() && e.texture_bind_group.is_some())
            .count();

        CacheStats {
            total_entries,
            complete_entries,
            max_entries: self.max_entries,
            memory_usage_estimate: total_entries * 1024 * 1024, // Rough estimate
        }
    }

    /// Evict least recently used entries
    fn evict_lru(&mut self) {
        while self.entries.len() > self.max_entries {
            // Find oldest entry
            let oldest_key = self.entries.iter()
                .min_by_key(|(_, entry)| entry.timestamp)
                .map(|(key, _)| *key);

            if let Some(key) = oldest_key {
                self.entries.remove(&key);
            } else {
                break;
            }
        }
    }
}

/// Cache statistics
#[derive(Debug, Clone)]
pub struct CacheStats {
    pub total_entries: usize,
    pub complete_entries: usize,
    pub max_entries: usize,
    pub memory_usage_estimate: usize,
}

/// Scene cache builder for easy configuration
pub struct SceneCacheBuilder {
    max_entries: usize,
}

impl Default for SceneCacheBuilder {
    fn default() -> Self {
        Self {
            max_entries: 16,
        }
    }
}

impl SceneCacheBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn max_entries(mut self, max_entries: usize) -> Self {
        self.max_entries = max_entries;
        self
    }

    pub fn build(self, device: std::sync::Arc<Device>) -> SceneCache {
        let mut cache = SceneCache::new(device);
        cache.set_max_entries(self.max_entries);
        cache
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cache_creation() {
        // Mock device creation would go here
        // let device = ...;
        // let cache = SceneCache::new(device);
        // assert_eq!(cache.entries.len(), 0);
    }

    #[test]
    fn test_hash_computation() {
        // Mock device for testing
        let device = std::sync::Arc::new(unsafe { std::mem::zeroed() }); // Placeholder
        let cache = SceneCache::new(device);

        let bvh_data = vec![1, 2, 3, 4];
        let material_data = vec![5, 6, 7, 8];
        let texture_ids = vec![100, 200];

        let hash1 = cache.compute_scene_hash(&bvh_data, &material_data, &texture_ids);
        let hash2 = cache.compute_scene_hash(&bvh_data, &material_data, &texture_ids);

        assert_eq!(hash1, hash2); // Same input should produce same hash
    }

    #[test]
    fn test_cache_builder() {
        let builder = SceneCacheBuilder::new()
            .max_entries(32);
        assert_eq!(builder.max_entries, 32);
    }
}