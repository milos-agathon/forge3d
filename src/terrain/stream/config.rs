#[derive(Debug, Clone, Copy)]
pub struct MosaicConfig {
    pub tile_size_px: u32,
    pub tiles_x: u32,
    pub tiles_y: u32,
    /// Optional fixed LOD; when set, slot = (tile_id.x, tile_id.y) and no LRU indirection
    pub fixed_lod: Option<u32>,
}

impl MosaicConfig {
    pub fn texture_size(&self) -> (u32, u32) {
        (
            self.tile_size_px * self.tiles_x,
            self.tile_size_px * self.tiles_y,
        )
    }
}
