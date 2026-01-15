//! P2.1/M5: Clipmap streaming integration with HeightMosaic and PageTable.

use super::level::ClipmapLevel;
use super::ClipmapConfig;
use crate::terrain::lod::LodConfig;
use crate::terrain::page_table::PageTable;
use crate::terrain::stream::HeightMosaic;
use crate::terrain::tiling::TileId;
use glam::{Mat4, Vec2, Vec3};
use wgpu::Queue;

/// Clipmap streamer connecting clipmap mesh to tile streaming infrastructure.
pub struct ClipmapStreamer {
    pub clipmap: ClipmapLevel,
    pending_tiles: Vec<TileId>,
    loaded_tiles: Vec<TileId>,
}

impl ClipmapStreamer {
    /// Create a new clipmap streamer.
    pub fn new(config: ClipmapConfig, center: Vec2, terrain_extent: f32) -> Self {
        Self {
            clipmap: ClipmapLevel::new(config, center, terrain_extent),
            pending_tiles: Vec::new(),
            loaded_tiles: Vec::new(),
        }
    }

    /// Update clipmap based on camera position and request needed tiles.
    pub fn update(
        &mut self,
        camera_pos: Vec3,
        _view_matrix: Mat4,
        _proj_matrix: Mat4,
        _lod_config: &LodConfig,
    ) -> Vec<TileId> {
        // Update clipmap center to camera XZ position
        let new_center = Vec2::new(camera_pos.x, camera_pos.z);
        let required_tiles = self.clipmap.update_center(new_center);

        // Filter out already loaded tiles
        let new_tiles: Vec<TileId> = required_tiles
            .into_iter()
            .filter(|t| !self.loaded_tiles.contains(t) && !self.pending_tiles.contains(t))
            .collect();

        self.pending_tiles.extend(new_tiles.iter().cloned());
        new_tiles
    }

    /// Mark tiles as loaded (call after successful upload to mosaic).
    pub fn mark_loaded(&mut self, tiles: &[TileId]) {
        for tile in tiles {
            self.pending_tiles.retain(|t| t != tile);
            if !self.loaded_tiles.contains(tile) {
                self.loaded_tiles.push(*tile);
            }
        }
    }

    /// Get the clipmap mesh, generating if needed.
    pub fn mesh(&mut self) -> &super::level::ClipmapMesh {
        self.clipmap.mesh()
    }

    /// Map ring index to tile LOD level.
    pub fn ring_to_tile_lod(&self, ring_index: u32) -> u32 {
        self.clipmap.ring_lod(ring_index)
    }

    /// Get current clipmap center.
    pub fn center(&self) -> Vec2 {
        self.clipmap.center
    }

    /// Get terrain extent.
    pub fn terrain_extent(&self) -> f32 {
        self.clipmap.terrain_extent
    }

    /// Get pending tile count.
    pub fn pending_count(&self) -> usize {
        self.pending_tiles.len()
    }

    /// Get loaded tile count.
    pub fn loaded_count(&self) -> usize {
        self.loaded_tiles.len()
    }
}

/// Integration helper for uploading clipmap tiles to HeightMosaic.
pub fn upload_clipmap_tiles(
    mosaic: &mut HeightMosaic,
    queue: &Queue,
    tiles: &[(TileId, Vec<f32>)],
) -> Vec<TileId> {
    let mut uploaded = Vec::new();
    for (tile_id, height_data) in tiles {
        match mosaic.upload_tile(queue, *tile_id, height_data) {
            Ok(_slot) => uploaded.push(*tile_id),
            Err(e) => {
                eprintln!("[clipmap] Failed to upload tile {:?}: {}", tile_id, e);
            }
        }
    }
    uploaded
}

/// Sync clipmap tiles to page table.
pub fn sync_clipmap_page_table(page_table: &mut PageTable, queue: &Queue, mosaic: &HeightMosaic) {
    page_table.sync_from_mosaic(queue, mosaic);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_streamer_creation() {
        let config = ClipmapConfig::new(4, 64);
        let streamer = ClipmapStreamer::new(config, Vec2::ZERO, 1000.0);
        assert_eq!(streamer.center(), Vec2::ZERO);
        assert_eq!(streamer.terrain_extent(), 1000.0);
    }

    #[test]
    fn test_streamer_update_requests_tiles() {
        let config = ClipmapConfig::new(4, 64);
        let mut streamer = ClipmapStreamer::new(config, Vec2::ZERO, 1000.0);

        let lod_config = LodConfig::new(2.0, 1024, 768, 45.0_f32.to_radians());
        let camera = Vec3::new(100.0, 50.0, 100.0);
        let view = Mat4::look_at_rh(camera, Vec3::ZERO, Vec3::Y);
        let proj = Mat4::perspective_rh(45.0_f32.to_radians(), 1.33, 1.0, 1000.0);

        let tiles = streamer.update(camera, view, proj, &lod_config);
        // First update should request tiles
        assert!(!tiles.is_empty() || streamer.pending_count() > 0);
    }

    #[test]
    fn test_mark_loaded_clears_pending() {
        let config = ClipmapConfig::new(4, 64);
        let mut streamer = ClipmapStreamer::new(config, Vec2::ZERO, 1000.0);

        let lod_config = LodConfig::new(2.0, 1024, 768, 45.0_f32.to_radians());
        let camera = Vec3::new(100.0, 50.0, 100.0);
        let view = Mat4::look_at_rh(camera, Vec3::ZERO, Vec3::Y);
        let proj = Mat4::perspective_rh(45.0_f32.to_radians(), 1.33, 1.0, 1000.0);

        let tiles = streamer.update(camera, view, proj, &lod_config);
        let pending_before = streamer.pending_count();

        if !tiles.is_empty() {
            streamer.mark_loaded(&tiles);
            assert!(streamer.pending_count() < pending_before || pending_before == 0);
            assert!(streamer.loaded_count() > 0);
        }
    }
}
