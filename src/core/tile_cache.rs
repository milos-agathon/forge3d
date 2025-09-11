//! LRU tile cache for virtual texture streaming
//!
//! This module provides an LRU (Least Recently Used) cache for managing
//! resident tiles in the virtual texture system with efficient lookup and eviction.

use std::collections::{HashMap, VecDeque};
use wgpu::TextureFormat;

/// Unique identifier for a virtual texture tile
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TileId {
    /// Tile X coordinate in virtual space
    pub x: u32,
    /// Tile Y coordinate in virtual space
    pub y: u32,
    /// Mip level of the tile
    pub mip_level: u32,
}

impl TileId {
    /// Create new tile ID
    pub fn new(x: u32, y: u32, mip_level: u32) -> Self {
        Self { x, y, mip_level }
    }
    
    /// Get parent tile ID (one level up in mip hierarchy)
    pub fn parent(&self) -> Option<Self> {
        if self.mip_level > 0 {
            Some(Self {
                x: self.x / 2,
                y: self.y / 2,
                mip_level: self.mip_level - 1,
            })
        } else {
            None
        }
    }
    
    /// Get child tile IDs (one level down in mip hierarchy)
    pub fn children(&self) -> [Self; 4] {
        let child_x = self.x * 2;
        let child_y = self.y * 2;
        let child_mip = self.mip_level + 1;
        
        [
            Self { x: child_x,     y: child_y,     mip_level: child_mip },
            Self { x: child_x + 1, y: child_y,     mip_level: child_mip },
            Self { x: child_x,     y: child_y + 1, mip_level: child_mip },
            Self { x: child_x + 1, y: child_y + 1, mip_level: child_mip },
        ]
    }
}

/// Physical location of a tile in the atlas texture
#[derive(Debug, Clone, Copy)]
pub struct AtlasSlot {
    /// X coordinate in atlas texture (pixels)
    pub atlas_x: u32,
    /// Y coordinate in atlas texture (pixels)
    pub atlas_y: u32,
    /// Normalized U coordinate in atlas (0.0-1.0)
    pub atlas_u: f32,
    /// Normalized V coordinate in atlas (0.0-1.0)
    pub atlas_v: f32,
    /// Mip bias for this atlas slot
    pub mip_bias: f32,
}

/// Tile data for loading and caching
#[derive(Debug, Clone)]
pub struct TileData {
    /// Tile identifier
    pub id: TileId,
    /// Raw pixel data
    pub data: Vec<u8>,
    /// Width in pixels
    pub width: u32,
    /// Height in pixels
    pub height: u32,
    /// Pixel format
    pub format: TextureFormat,
}

/// Cache entry for a resident tile
#[derive(Debug, Clone)]
struct CacheEntry {
    /// Tile ID
    tile_id: TileId,
    /// Atlas slot where tile is stored
    atlas_slot: AtlasSlot,
    /// Last access timestamp
    last_access: u64,
    /// Reference count
    ref_count: u32,
}

/// LRU cache statistics
#[derive(Debug, Clone, Default)]
pub struct CacheStats {
    /// Total cache capacity
    pub capacity: usize,
    /// Number of currently resident tiles
    pub resident_count: usize,
    /// Number of cache hits since last reset
    pub hits: u64,
    /// Number of cache misses since last reset  
    pub misses: u64,
    /// Number of evictions since last reset
    pub evictions: u64,
    /// Average access time in nanoseconds
    pub avg_access_time_ns: f64,
}

/// LRU tile cache for virtual texture streaming
pub struct TileCache {
    /// Maximum number of resident tiles
    capacity: usize,
    /// Map of tile ID to cache entry
    resident_tiles: HashMap<TileId, CacheEntry>,
    /// LRU queue for eviction order (most recent first)
    lru_queue: VecDeque<TileId>,
    /// Atlas slot allocator
    atlas_allocator: AtlasAllocator,
    /// Global access counter for timestamps
    access_counter: u64,
    /// Cache statistics
    stats: CacheStats,
}

impl TileCache {
    /// Create new tile cache with given capacity
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity,
            resident_tiles: HashMap::new(),
            lru_queue: VecDeque::new(),
            atlas_allocator: AtlasAllocator::new(),
            access_counter: 0,
            stats: CacheStats {
                capacity,
                ..Default::default()
            },
        }
    }
    
    /// Configure atlas dimensions for slot allocation
    pub fn configure_atlas(&mut self, atlas_width: u32, atlas_height: u32, tile_size: u32) {
        self.atlas_allocator = AtlasAllocator::new_with_dimensions(atlas_width, atlas_height, tile_size);
    }
    
    /// Check if a tile is resident in cache
    pub fn is_resident(&self, tile_id: &TileId) -> bool {
        self.resident_tiles.contains_key(tile_id)
    }
    
    /// Access a tile (updating LRU order)
    pub fn access_tile(&mut self, tile_id: &TileId) -> Option<AtlasSlot> {
        if let Some(entry) = self.resident_tiles.get_mut(tile_id) {
            // Update access time
            self.access_counter += 1;
            entry.last_access = self.access_counter;
            
            // Move to front of LRU queue
            if let Some(pos) = self.lru_queue.iter().position(|&id| id == *tile_id) {
                self.lru_queue.remove(pos);
            }
            self.lru_queue.push_front(*tile_id);
            
            self.stats.hits += 1;
            Some(entry.atlas_slot)
        } else {
            self.stats.misses += 1;
            None
        }
    }
    
    /// Allocate a tile in the cache (may evict LRU tile)
    pub fn allocate_tile(&mut self, tile_id: TileId) -> Option<AtlasSlot> {
        // If already resident, just access it
        if self.is_resident(&tile_id) {
            return self.access_tile(&tile_id);
        }
        
        // Check if we need to evict
        while self.resident_tiles.len() >= self.capacity {
            if !self.evict_lru_tile() {
                // Failed to evict, cache might be full of referenced tiles
                return None;
            }
        }
        
        // Allocate atlas slot
        if let Some(atlas_slot) = self.atlas_allocator.allocate() {
            // Add to cache
            self.access_counter += 1;
            
            let entry = CacheEntry {
                tile_id,
                atlas_slot,
                last_access: self.access_counter,
                ref_count: 1,
            };
            
            self.resident_tiles.insert(tile_id, entry);
            self.lru_queue.push_front(tile_id);
            
            self.stats.resident_count = self.resident_tiles.len();
            
            Some(atlas_slot)
        } else {
            None
        }
    }
    
    /// Evict least recently used tile
    fn evict_lru_tile(&mut self) -> bool {
        // Find LRU tile that can be evicted (ref_count == 0)
        while let Some(&lru_tile_id) = self.lru_queue.back() {
            if let Some(entry) = self.resident_tiles.get(&lru_tile_id) {
                if entry.ref_count == 0 {
                    // Can evict this tile
                    let entry = self.resident_tiles.remove(&lru_tile_id).unwrap();
                    self.lru_queue.pop_back();
                    
                    // Return atlas slot to allocator
                    self.atlas_allocator.deallocate(entry.atlas_slot);
                    
                    self.stats.evictions += 1;
                    self.stats.resident_count = self.resident_tiles.len();
                    
                    return true;
                } else {
                    // Tile is referenced, move it to front and try next
                    self.lru_queue.pop_back();
                    self.lru_queue.push_front(lru_tile_id);
                }
            } else {
                // Stale entry in LRU queue, remove it
                self.lru_queue.pop_back();
            }
        }
        
        false // No tiles could be evicted
    }
    
    /// Get atlas slot for a resident tile
    pub fn get_atlas_slot(&self, tile_id: &TileId) -> Option<AtlasSlot> {
        self.resident_tiles.get(tile_id).map(|entry| entry.atlas_slot)
    }
    
    /// Increment reference count for a tile
    pub fn add_ref(&mut self, tile_id: &TileId) -> bool {
        if let Some(entry) = self.resident_tiles.get_mut(tile_id) {
            entry.ref_count += 1;
            true
        } else {
            false
        }
    }
    
    /// Decrement reference count for a tile
    pub fn release(&mut self, tile_id: &TileId) -> bool {
        if let Some(entry) = self.resident_tiles.get_mut(tile_id) {
            if entry.ref_count > 0 {
                entry.ref_count -= 1;
            }
            true
        } else {
            false
        }
    }
    
    /// Force eviction of a specific tile
    pub fn evict_tile(&mut self, tile_id: &TileId) -> bool {
        if let Some(entry) = self.resident_tiles.remove(tile_id) {
            // Remove from LRU queue
            if let Some(pos) = self.lru_queue.iter().position(|&id| id == *tile_id) {
                self.lru_queue.remove(pos);
            }
            
            // Return atlas slot
            self.atlas_allocator.deallocate(entry.atlas_slot);
            
            self.stats.evictions += 1;
            self.stats.resident_count = self.resident_tiles.len();
            
            true
        } else {
            false
        }
    }
    
    /// Clear all tiles from cache
    pub fn clear(&mut self) {
        self.resident_tiles.clear();
        self.lru_queue.clear();
        self.atlas_allocator.clear();
        self.stats.resident_count = 0;
    }
    
    /// Get current cache statistics
    pub fn stats(&self) -> &CacheStats {
        &self.stats
    }
    
    /// Reset statistics counters
    pub fn reset_stats(&mut self) {
        self.stats.hits = 0;
        self.stats.misses = 0;
        self.stats.evictions = 0;
    }
    
    /// Get number of resident tiles
    pub fn resident_count(&self) -> usize {
        self.resident_tiles.len()
    }
    
    /// Get cache capacity
    pub fn capacity(&self) -> usize {
        self.capacity
    }
    
    /// Get all resident tile IDs
    pub fn resident_tiles(&self) -> Vec<TileId> {
        self.resident_tiles.keys().cloned().collect()
    }
    
    /// Perform cache maintenance (cleanup stale entries)
    pub fn maintain(&mut self) {
        // Remove stale LRU queue entries
        self.lru_queue.retain(|tile_id| self.resident_tiles.contains_key(tile_id));
        
        // Update stats
        self.stats.resident_count = self.resident_tiles.len();
    }
}

/// Atlas slot allocator for managing physical texture space
struct AtlasAllocator {
    /// Atlas dimensions
    atlas_width: u32,
    atlas_height: u32,
    /// Tile size
    tile_size: u32,
    /// Number of tiles in X direction
    tiles_x: u32,
    /// Number of tiles in Y direction
    tiles_y: u32,
    /// Free slot list
    free_slots: Vec<AtlasSlot>,
    /// Used slot tracker
    used_slots: Vec<bool>,
}

impl AtlasAllocator {
    /// Create new atlas allocator with default dimensions
    fn new() -> Self {
        Self::new_with_dimensions(2048, 2048, 128)
    }
    
    /// Create atlas allocator with specific dimensions
    fn new_with_dimensions(atlas_width: u32, atlas_height: u32, tile_size: u32) -> Self {
        let tiles_x = atlas_width / tile_size;
        let tiles_y = atlas_height / tile_size;
        let total_tiles = tiles_x * tiles_y;
        
        let mut free_slots = Vec::new();
        let used_slots = vec![false; total_tiles as usize];
        
        // Initialize free slots
        for y in 0..tiles_y {
            for x in 0..tiles_x {
                let atlas_x = x * tile_size;
                let atlas_y = y * tile_size;
                let atlas_u = atlas_x as f32 / atlas_width as f32;
                let atlas_v = atlas_y as f32 / atlas_height as f32;
                
                free_slots.push(AtlasSlot {
                    atlas_x,
                    atlas_y,
                    atlas_u,
                    atlas_v,
                    mip_bias: 0.0,
                });
            }
        }
        
        Self {
            atlas_width,
            atlas_height,
            tile_size,
            tiles_x,
            tiles_y,
            free_slots,
            used_slots,
        }
    }
    
    /// Allocate a free atlas slot
    fn allocate(&mut self) -> Option<AtlasSlot> {
        self.free_slots.pop()
    }
    
    /// Deallocate an atlas slot
    fn deallocate(&mut self, slot: AtlasSlot) {
        self.free_slots.push(slot);
    }
    
    /// Clear all allocations
    fn clear(&mut self) {
        let total_tiles = self.tiles_x * self.tiles_y;
        self.free_slots.clear();
        self.used_slots = vec![false; total_tiles as usize];
        
        // Reinitialize free slots
        for y in 0..self.tiles_y {
            for x in 0..self.tiles_x {
                let atlas_x = x * self.tile_size;
                let atlas_y = y * self.tile_size;
                let atlas_u = atlas_x as f32 / self.atlas_width as f32;
                let atlas_v = atlas_y as f32 / self.atlas_height as f32;
                
                self.free_slots.push(AtlasSlot {
                    atlas_x,
                    atlas_y,
                    atlas_u,
                    atlas_v,
                    mip_bias: 0.0,
                });
            }
        }
    }
    
    /// Get number of free slots
    fn free_count(&self) -> usize {
        self.free_slots.len()
    }
    
    /// Get total number of slots
    fn total_count(&self) -> usize {
        (self.tiles_x * self.tiles_y) as usize
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_tile_id_creation() {
        let tile = TileId::new(10, 20, 2);
        assert_eq!(tile.x, 10);
        assert_eq!(tile.y, 20);
        assert_eq!(tile.mip_level, 2);
    }
    
    #[test]
    fn test_tile_id_parent() {
        let tile = TileId::new(10, 20, 2);
        let parent = tile.parent().unwrap();
        assert_eq!(parent.x, 5);
        assert_eq!(parent.y, 10);
        assert_eq!(parent.mip_level, 1);
        
        let root = TileId::new(0, 0, 0);
        assert!(root.parent().is_none());
    }
    
    #[test]
    fn test_tile_id_children() {
        let tile = TileId::new(5, 10, 1);
        let children = tile.children();
        
        assert_eq!(children[0], TileId::new(10, 20, 2));
        assert_eq!(children[1], TileId::new(11, 20, 2));
        assert_eq!(children[2], TileId::new(10, 21, 2));
        assert_eq!(children[3], TileId::new(11, 21, 2));
    }
    
    #[test]
    fn test_tile_cache_creation() {
        let cache = TileCache::new(100);
        assert_eq!(cache.capacity(), 100);
        assert_eq!(cache.resident_count(), 0);
        assert!(!cache.is_resident(&TileId::new(0, 0, 0)));
    }
    
    #[test]
    fn test_tile_cache_allocation() {
        let mut cache = TileCache::new(10);
        cache.configure_atlas(512, 512, 64);
        
        let tile1 = TileId::new(0, 0, 0);
        let tile2 = TileId::new(1, 0, 0);
        
        // Allocate first tile
        let slot1 = cache.allocate_tile(tile1);
        assert!(slot1.is_some());
        assert!(cache.is_resident(&tile1));
        assert_eq!(cache.resident_count(), 1);
        
        // Allocate second tile
        let slot2 = cache.allocate_tile(tile2);
        assert!(slot2.is_some());
        assert!(cache.is_resident(&tile2));
        assert_eq!(cache.resident_count(), 2);
        
        // Access first tile (should update LRU order)
        let accessed_slot = cache.access_tile(&tile1);
        assert!(accessed_slot.is_some());
    }
    
    #[test]
    fn test_tile_cache_eviction() {
        let mut cache = TileCache::new(2); // Small capacity
        cache.configure_atlas(256, 256, 64);
        
        let tile1 = TileId::new(0, 0, 0);
        let tile2 = TileId::new(1, 0, 0);
        let tile3 = TileId::new(2, 0, 0);
        
        // Fill cache to capacity
        cache.allocate_tile(tile1);
        cache.allocate_tile(tile2);
        assert_eq!(cache.resident_count(), 2);
        
        // Allocate third tile (should evict LRU)
        cache.allocate_tile(tile3);
        assert_eq!(cache.resident_count(), 2);
        assert!(cache.is_resident(&tile3));
        
        // tile1 should have been evicted (was LRU)
        assert!(!cache.is_resident(&tile1));
    }
    
    #[test]
    fn test_atlas_allocator() {
        let mut allocator = AtlasAllocator::new_with_dimensions(256, 256, 64);
        
        assert_eq!(allocator.total_count(), 16); // 4x4 grid of 64x64 tiles
        assert_eq!(allocator.free_count(), 16);
        
        // Allocate a slot
        let slot = allocator.allocate();
        assert!(slot.is_some());
        assert_eq!(allocator.free_count(), 15);
        
        // Deallocate the slot
        allocator.deallocate(slot.unwrap());
        assert_eq!(allocator.free_count(), 16);
    }
    
    #[test]
    fn test_cache_stats() {
        let mut cache = TileCache::new(5);
        cache.configure_atlas(256, 256, 64);
        
        let stats = cache.stats();
        assert_eq!(stats.capacity, 5);
        assert_eq!(stats.resident_count, 0);
        assert_eq!(stats.hits, 0);
        assert_eq!(stats.misses, 0);
        
        // Cause some hits and misses
        let tile = TileId::new(0, 0, 0);
        
        // First access should be a miss
        cache.access_tile(&tile);
        assert_eq!(cache.stats().misses, 1);
        
        // Allocate and then access should be a hit
        cache.allocate_tile(tile);
        cache.access_tile(&tile);
        assert_eq!(cache.stats().hits, 1);
    }
    
    #[test]
    fn test_reference_counting() {
        let mut cache = TileCache::new(5);
        let tile = TileId::new(0, 0, 0);
        
        // Allocate tile
        cache.allocate_tile(tile);
        
        // Add references
        assert!(cache.add_ref(&tile));
        assert!(cache.add_ref(&tile));
        
        // Release references
        assert!(cache.release(&tile));
        assert!(cache.release(&tile));
        assert!(cache.release(&tile)); // Should handle ref_count going to 0
    }
}