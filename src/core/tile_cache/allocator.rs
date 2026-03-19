use super::AtlasSlot;

/// Atlas slot allocator for managing physical texture space
pub(super) struct AtlasAllocator {
    atlas_width: u32,
    atlas_height: u32,
    tile_size: u32,
    tiles_x: u32,
    tiles_y: u32,
    free_slots: Vec<AtlasSlot>,
    used_slots: Vec<bool>,
}

impl AtlasAllocator {
    pub(super) fn new() -> Self {
        Self::new_with_dimensions(2048, 2048, 128)
    }

    pub(super) fn new_with_dimensions(atlas_width: u32, atlas_height: u32, tile_size: u32) -> Self {
        let tiles_x = atlas_width / tile_size;
        let tiles_y = atlas_height / tile_size;
        let total_tiles = tiles_x * tiles_y;

        let mut free_slots = Vec::new();
        let used_slots = vec![false; total_tiles as usize];

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

    pub(super) fn allocate(&mut self) -> Option<AtlasSlot> {
        self.free_slots.pop()
    }

    pub(super) fn deallocate(&mut self, slot: AtlasSlot) {
        self.free_slots.push(slot);
    }

    pub(super) fn clear(&mut self) {
        let total_tiles = self.tiles_x * self.tiles_y;
        self.free_slots.clear();
        self.used_slots = vec![false; total_tiles as usize];

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

    #[allow(dead_code)]
    pub(super) fn free_count(&self) -> usize {
        self.free_slots.len()
    }

    #[allow(dead_code)]
    pub(super) fn total_count(&self) -> usize {
        (self.tiles_x * self.tiles_y) as usize
    }
}
