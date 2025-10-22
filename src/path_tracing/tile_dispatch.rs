// src/path_tracing/tile_dispatch.rs
// Tiled dispatch manager for large resolution ray tracing
// Breaks renders into tiles to avoid Metal storage texture write limits

/// Configuration for tiled rendering
#[derive(Debug, Clone, Copy)]
pub struct TileConfig {
    /// Tile width in pixels (must be multiple of 8 for workgroup alignment)
    pub tile_width: u32,
    /// Tile height in pixels (must be multiple of 8)
    pub tile_height: u32,
    /// Samples per pixel per batch (for progressive rendering)
    pub batch_spp: u32,
}

impl Default for TileConfig {
    fn default() -> Self {
        Self {
            tile_width: 256,
            tile_height: 256,
            batch_spp: 4,
        }
    }
}

impl TileConfig {
    pub fn new(tile_width: u32, tile_height: u32, batch_spp: u32) -> Self {
        // Ensure tiles are multiples of 8 (workgroup size)
        let tw = ((tile_width + 7) / 8) * 8;
        let th = ((tile_height + 7) / 8) * 8;
        Self {
            tile_width: tw,
            tile_height: th,
            batch_spp,
        }
    }
    
    /// Check if tiling is needed for given resolution
    pub fn needs_tiling(&self, width: u32, height: u32) -> bool {
        width > self.tile_width || height > self.tile_height
    }
}

/// Represents a single tile region
#[derive(Debug, Clone, Copy)]
pub struct Tile {
    /// Tile origin X in image coordinates
    pub x: u32,
    /// Tile origin Y in image coordinates
    pub y: u32,
    /// Tile width (may be smaller than tile_width at edges)
    pub width: u32,
    /// Tile height (may be smaller than tile_height at edges)
    pub height: u32,
}

/// Iterator over tiles covering an image
pub struct TileIterator {
    image_width: u32,
    image_height: u32,
    tile_width: u32,
    tile_height: u32,
    current_x: u32,
    current_y: u32,
}

impl TileIterator {
    pub fn new(image_width: u32, image_height: u32, tile_width: u32, tile_height: u32) -> Self {
        Self {
            image_width,
            image_height,
            tile_width,
            tile_height,
            current_x: 0,
            current_y: 0,
        }
    }
    
    pub fn total_tiles(&self) -> usize {
        let nx = (self.image_width + self.tile_width - 1) / self.tile_width;
        let ny = (self.image_height + self.tile_height - 1) / self.tile_height;
        (nx * ny) as usize
    }
}

impl Iterator for TileIterator {
    type Item = Tile;
    
    fn next(&mut self) -> Option<Self::Item> {
        if self.current_y >= self.image_height {
            return None;
        }
        
        let tile_x = self.current_x;
        let tile_y = self.current_y;
        
        // Calculate actual tile dimensions (may be clipped at edges)
        let tile_w = self.tile_width.min(self.image_width - tile_x);
        let tile_h = self.tile_height.min(self.image_height - tile_y);
        
        // Advance to next tile
        self.current_x += self.tile_width;
        if self.current_x >= self.image_width {
            self.current_x = 0;
            self.current_y += self.tile_height;
        }
        
        Some(Tile {
            x: tile_x,
            y: tile_y,
            width: tile_w,
            height: tile_h,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_tile_iterator_exact_fit() {
        let tiles: Vec<_> = TileIterator::new(512, 512, 256, 256).collect();
        assert_eq!(tiles.len(), 4);
        assert_eq!(tiles[0], Tile { x: 0, y: 0, width: 256, height: 256 });
        assert_eq!(tiles[3], Tile { x: 256, y: 256, width: 256, height: 256 });
    }
    
    #[test]
    fn test_tile_iterator_partial() {
        let tiles: Vec<_> = TileIterator::new(640, 480, 256, 256).collect();
        assert_eq!(tiles.len(), 6); // 3x2 grid
        // Check edge tile is clipped
        assert_eq!(tiles[2], Tile { x: 512, y: 0, width: 128, height: 256 });
        assert_eq!(tiles[5], Tile { x: 512, y: 256, width: 128, height: 224 });
    }
}
