use super::{allocator::AtlasAllocator, types::CacheEntry, AtlasSlot, TileId};
use std::collections::{HashMap, VecDeque};

/// LRU cache statistics
#[derive(Debug, Clone, Default)]
pub struct CacheStats {
    pub capacity: usize,
    pub resident_count: usize,
    pub hits: u64,
    pub misses: u64,
    pub evictions: u64,
    pub avg_access_time_ns: f64,
}

/// LRU tile cache for virtual texture streaming
pub struct TileCache {
    capacity: usize,
    resident_tiles: HashMap<TileId, CacheEntry>,
    lru_queue: VecDeque<TileId>,
    atlas_allocator: AtlasAllocator,
    access_counter: u64,
    stats: CacheStats,
}

impl TileCache {
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

    /// Configure atlas dimensions and slot size (spec 8.4).
    ///
    /// # Arguments
    /// * `atlas_width` - Physical atlas texture width in pixels
    /// * `atlas_height` - Physical atlas texture height in pixels
    /// * `slot_size` - Slot size in pixels (tile_size + 2*tile_border)
    pub fn configure_atlas(&mut self, atlas_width: u32, atlas_height: u32, slot_size: u32) {
        self.atlas_allocator =
            AtlasAllocator::new_with_dimensions(atlas_width, atlas_height, slot_size);
    }

    pub fn is_resident(&self, tile_id: &TileId) -> bool {
        self.resident_tiles.contains_key(tile_id)
    }

    pub fn access_tile(&mut self, tile_id: &TileId) -> Option<AtlasSlot> {
        if let Some(entry) = self.resident_tiles.get_mut(tile_id) {
            self.access_counter += 1;
            entry.last_access = self.access_counter;

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

    pub fn allocate_tile(&mut self, tile_id: TileId) -> Option<AtlasSlot> {
        self.allocate_tile_with_evicted(tile_id)
            .map(|(atlas_slot, _)| atlas_slot)
    }

    pub fn allocate_tile_with_evicted(
        &mut self,
        tile_id: TileId,
    ) -> Option<(AtlasSlot, Vec<TileId>)> {
        if self.is_resident(&tile_id) {
            return self.access_tile(&tile_id).map(|slot| (slot, Vec::new()));
        }

        let mut evicted = Vec::new();
        while self.resident_tiles.len() >= self.capacity {
            let Some(evicted_tile) = self.evict_lru_tile() else {
                return None;
            };
            evicted.push(evicted_tile);
        }

        if let Some(atlas_slot) = self.atlas_allocator.allocate() {
            self.access_counter += 1;

            self.resident_tiles.insert(
                tile_id,
                CacheEntry {
                    tile_id,
                    atlas_slot,
                    last_access: self.access_counter,
                    ref_count: 0,
                },
            );
            self.lru_queue.push_front(tile_id);
            self.stats.resident_count = self.resident_tiles.len();

            Some((atlas_slot, evicted))
        } else {
            None
        }
    }

    fn evict_lru_tile(&mut self) -> Option<TileId> {
        let mut remaining = self.lru_queue.len();

        while remaining > 0 {
            remaining -= 1;

            let Some(lru_tile_id) = self.lru_queue.pop_back() else {
                break;
            };

            if let Some(entry) = self.resident_tiles.get(&lru_tile_id) {
                if entry.ref_count == 0 {
                    let entry = self.resident_tiles.remove(&lru_tile_id).unwrap();
                    self.atlas_allocator.deallocate(entry.atlas_slot);
                    self.stats.evictions += 1;
                    self.stats.resident_count = self.resident_tiles.len();
                    return Some(lru_tile_id);
                }

                self.lru_queue.push_front(lru_tile_id);
            }
        }

        None
    }

    pub fn get_atlas_slot(&self, tile_id: &TileId) -> Option<AtlasSlot> {
        self.resident_tiles
            .get(tile_id)
            .map(|entry| entry.atlas_slot)
    }

    pub fn add_ref(&mut self, tile_id: &TileId) -> bool {
        if let Some(entry) = self.resident_tiles.get_mut(tile_id) {
            entry.ref_count += 1;
            true
        } else {
            false
        }
    }

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

    pub fn evict_tile(&mut self, tile_id: &TileId) -> bool {
        if let Some(entry) = self.resident_tiles.remove(tile_id) {
            if let Some(pos) = self.lru_queue.iter().position(|&id| id == *tile_id) {
                self.lru_queue.remove(pos);
            }

            self.atlas_allocator.deallocate(entry.atlas_slot);
            self.stats.evictions += 1;
            self.stats.resident_count = self.resident_tiles.len();

            true
        } else {
            false
        }
    }

    pub fn clear(&mut self) {
        self.resident_tiles.clear();
        self.lru_queue.clear();
        self.atlas_allocator.clear();
        self.stats.resident_count = 0;
    }

    pub fn stats(&self) -> &CacheStats {
        &self.stats
    }

    pub fn reset_stats(&mut self) {
        self.stats.hits = 0;
        self.stats.misses = 0;
        self.stats.evictions = 0;
    }

    pub fn resident_count(&self) -> usize {
        self.resident_tiles.len()
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    pub fn resident_tiles(&self) -> Vec<TileId> {
        self.resident_tiles.keys().cloned().collect()
    }

    pub fn maintain(&mut self) {
        self.lru_queue
            .retain(|tile_id| self.resident_tiles.contains_key(tile_id));
        self.stats.resident_count = self.resident_tiles.len();
    }
}
