use super::AtlasSlot;

/// Atlas slot allocator for managing physical texture space
pub(super) struct AtlasAllocator {
    atlas_width: u32,
    atlas_height: u32,
    slot_size: u32,
    slots_x: u32,
    slots_y: u32,
    free_slots: Vec<AtlasSlot>,
    used_slots: Vec<bool>,
}

impl AtlasAllocator {
    pub(super) fn new() -> Self {
        // Default: atlas 2048×2048, slot_size 128 (tile_size=120, border=4)
        Self::new_with_dimensions(2048, 2048, 128)
    }

    /// Create allocator with specific atlas and slot dimensions.
    ///
    /// # Arguments
    /// * `atlas_width` - Physical atlas texture width in pixels
    /// * `atlas_height` - Physical atlas texture height in pixels
    /// * `slot_size` - Slot size in pixels (tile_size + 2*tile_border, spec 8.4)
    pub(super) fn new_with_dimensions(atlas_width: u32, atlas_height: u32, slot_size: u32) -> Self {
        let slots_x = atlas_width / slot_size;
        let slots_y = atlas_height / slot_size;
        let total_slots = slots_x * slots_y;

        let mut free_slots = Vec::new();
        let used_slots = vec![false; total_slots as usize];

        for y in 0..slots_y {
            for x in 0..slots_x {
                let atlas_x = x * slot_size;
                let atlas_y = y * slot_size;
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
            slot_size,
            slots_x,
            slots_y,
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
        let total_slots = self.slots_x * self.slots_y;
        self.free_slots.clear();
        self.used_slots = vec![false; total_slots as usize];

        for y in 0..self.slots_y {
            for x in 0..self.slots_x {
                let atlas_x = x * self.slot_size;
                let atlas_y = y * self.slot_size;
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
        (self.slots_x * self.slots_y) as usize
    }
}
