use wgpu::TextureFormat;

/// Unique identifier for a virtual texture tile
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TileId {
    pub x: u32,
    pub y: u32,
    pub mip_level: u32,
}

impl TileId {
    pub fn new(x: u32, y: u32, mip_level: u32) -> Self {
        Self { x, y, mip_level }
    }

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

    pub fn children(&self) -> [Self; 4] {
        let child_x = self.x * 2;
        let child_y = self.y * 2;
        let child_mip = self.mip_level + 1;

        [
            Self {
                x: child_x,
                y: child_y,
                mip_level: child_mip,
            },
            Self {
                x: child_x + 1,
                y: child_y,
                mip_level: child_mip,
            },
            Self {
                x: child_x,
                y: child_y + 1,
                mip_level: child_mip,
            },
            Self {
                x: child_x + 1,
                y: child_y + 1,
                mip_level: child_mip,
            },
        ]
    }
}

/// Physical location of a tile in the atlas texture
#[derive(Debug, Clone, Copy)]
pub struct AtlasSlot {
    pub atlas_x: u32,
    pub atlas_y: u32,
    pub atlas_u: f32,
    pub atlas_v: f32,
    pub mip_bias: f32,
}

/// Tile data for loading and caching
#[derive(Debug, Clone)]
pub struct TileData {
    pub id: TileId,
    pub data: Vec<u8>,
    pub width: u32,
    pub height: u32,
    pub format: TextureFormat,
}

#[derive(Debug, Clone)]
pub(super) struct CacheEntry {
    #[allow(dead_code)]
    pub(super) tile_id: TileId,
    pub(super) atlas_slot: AtlasSlot,
    pub(super) last_access: u64,
    pub(super) ref_count: u32,
}
