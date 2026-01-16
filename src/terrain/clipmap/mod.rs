//! P2.1/M5: Geometry clipmap terrain system for true scalability.
//!
//! This module implements nested-ring clipmap meshes with geo-morphing
//! for seamless LOD transitions. Connects to existing HeightMosaic/PageTable
//! infrastructure for tile streaming.
//!
//! # Architecture
//!
//! ```text
//! Ring 3 (LOD 3): 8x spacing  ████████████████
//! Ring 2 (LOD 2): 4x spacing    ████████████
//! Ring 1 (LOD 1): 2x spacing      ████████
//! Ring 0 (LOD 0): 1x spacing        ████
//! Center (LOD 0): 1x spacing          ██
//! ```

pub mod geomorph;
pub mod gpu_lod;
pub mod level;
#[cfg(feature = "extension-module")]
pub mod py_bindings;
pub mod ring;
pub mod streaming;
pub mod vertex;

pub use level::{ClipmapLevel, ClipmapMesh};
pub use ring::{make_center_block, make_ring, make_ring_skirts};
pub use streaming::ClipmapStreamer;
pub use vertex::ClipmapVertex;

use glam::Vec2;

/// Configuration for clipmap terrain generation.
#[derive(Debug, Clone)]
pub struct ClipmapConfig {
    /// Number of LOD rings around the center (typically 4-6).
    pub ring_count: u32,
    /// Grid resolution per ring side (e.g., 64 = 64x64 cells per ring strip).
    pub ring_resolution: u32,
    /// Center block resolution (typically same as ring_resolution).
    pub center_resolution: u32,
    /// Depth of skirt vertices to hide seams between LOD levels.
    pub skirt_depth: f32,
    /// Distance range for geo-morphing blend [0.0-1.0].
    /// 0.3 means morphing starts at 70% of ring boundary.
    pub morph_range: f32,
}

impl Default for ClipmapConfig {
    fn default() -> Self {
        Self {
            ring_count: 4,
            ring_resolution: 64,
            center_resolution: 64,
            skirt_depth: 10.0,
            morph_range: 0.3,
        }
    }
}

impl ClipmapConfig {
    /// Create a new clipmap configuration.
    pub fn new(ring_count: u32, ring_resolution: u32) -> Self {
        Self {
            ring_count,
            ring_resolution,
            center_resolution: ring_resolution,
            ..Default::default()
        }
    }

    /// Set skirt depth for hiding LOD seams.
    pub fn with_skirt_depth(mut self, depth: f32) -> Self {
        self.skirt_depth = depth;
        self
    }

    /// Set morph range for geo-morphing blend distance.
    pub fn with_morph_range(mut self, range: f32) -> Self {
        self.morph_range = range.clamp(0.0, 1.0);
        self
    }

    /// Calculate the world-space extent of a given ring.
    /// Ring 0 is innermost (finest LOD), ring N-1 is outermost (coarsest).
    pub fn ring_extent(&self, ring_index: u32, base_cell_size: f32) -> f32 {
        let lod_scale = 1 << ring_index;
        let cell_size = base_cell_size * lod_scale as f32;
        cell_size * self.ring_resolution as f32
    }

    /// Calculate inner and outer extents for a ring.
    pub fn ring_bounds(&self, ring_index: u32, base_cell_size: f32, _center: Vec2) -> (f32, f32) {
        if ring_index == 0 {
            let center_extent = base_cell_size * self.center_resolution as f32 * 0.5;
            let outer = center_extent + self.ring_extent(0, base_cell_size);
            (center_extent, outer)
        } else {
            let mut inner = base_cell_size * self.center_resolution as f32 * 0.5;
            for i in 0..ring_index {
                inner += self.ring_extent(i, base_cell_size);
            }
            let outer = inner + self.ring_extent(ring_index, base_cell_size);
            (inner, outer)
        }
    }

    /// Get the LOD level for a ring index (ring 0 = LOD 0, ring N = LOD N).
    pub fn ring_lod(&self, ring_index: u32) -> u32 {
        ring_index
    }

    /// Estimate total vertex count for the complete clipmap.
    pub fn estimate_vertex_count(&self) -> u32 {
        let center_verts = (self.center_resolution + 1) * (self.center_resolution + 1);
        let ring_verts_per_ring = self.ring_resolution * 4 * 2; // Approximate for hollow ring
        center_verts + ring_verts_per_ring * self.ring_count
    }

    /// Estimate total triangle count for the complete clipmap.
    pub fn estimate_triangle_count(&self) -> u32 {
        let center_tris = self.center_resolution * self.center_resolution * 2;
        let ring_tris_per_ring = self.ring_resolution * 4 * 2; // Approximate
        center_tris + ring_tris_per_ring * self.ring_count
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_defaults() {
        let config = ClipmapConfig::default();
        assert_eq!(config.ring_count, 4);
        assert_eq!(config.ring_resolution, 64);
        assert_eq!(config.center_resolution, 64);
    }

    #[test]
    fn test_ring_extent_scaling() {
        let config = ClipmapConfig::new(4, 64);
        let base_cell = 1.0;

        // Ring 0: 1x scale
        assert_eq!(config.ring_extent(0, base_cell), 64.0);
        // Ring 1: 2x scale
        assert_eq!(config.ring_extent(1, base_cell), 128.0);
        // Ring 2: 4x scale
        assert_eq!(config.ring_extent(2, base_cell), 256.0);
        // Ring 3: 8x scale
        assert_eq!(config.ring_extent(3, base_cell), 512.0);
    }

    #[test]
    fn test_morph_range_clamped() {
        let config = ClipmapConfig::default().with_morph_range(1.5);
        assert_eq!(config.morph_range, 1.0);

        let config = ClipmapConfig::default().with_morph_range(-0.5);
        assert_eq!(config.morph_range, 0.0);
    }
}
