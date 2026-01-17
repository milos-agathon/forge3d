//! Tileset traversal for LOD selection

use glam::{Mat4, Vec3};
use super::bounds::BoundingVolume;
use super::sse::{SseParams, compute_sse};
use super::tile::{Tile, TileRefine};
use super::tileset::Tileset;

/// Result of traversal for a single tile
#[derive(Debug, Clone)]
pub struct VisibleTile<'a> {
    /// Reference to the tile
    pub tile: &'a Tile,
    /// World transform for this tile
    pub world_transform: Mat4,
    /// World-space bounding volume
    pub world_bounds: BoundingVolume,
    /// Computed SSE for this tile
    pub sse: f32,
    /// Depth in the hierarchy (0 = root)
    pub depth: usize,
}

/// Tileset traverser for LOD selection
#[derive(Debug)]
pub struct TilesetTraverser {
    /// SSE threshold for refinement (in pixels)
    pub sse_threshold: f32,
    /// Maximum depth to traverse
    pub max_depth: usize,
    /// SSE computation parameters
    pub sse_params: SseParams,
    /// Whether to cull tiles outside frustum
    pub frustum_cull: bool,
}

impl Default for TilesetTraverser {
    fn default() -> Self {
        Self {
            sse_threshold: 16.0,
            max_depth: 32,
            sse_params: SseParams::default(),
            frustum_cull: true,
        }
    }
}

impl TilesetTraverser {
    /// Create a new traverser with given SSE threshold
    pub fn new(sse_threshold: f32) -> Self {
        Self {
            sse_threshold,
            ..Default::default()
        }
    }

    /// Set SSE parameters
    pub fn with_sse_params(mut self, params: SseParams) -> Self {
        self.sse_params = params;
        self
    }

    /// Set maximum traversal depth
    pub fn with_max_depth(mut self, max_depth: usize) -> Self {
        self.max_depth = max_depth;
        self
    }

    /// Enable/disable frustum culling
    pub fn with_frustum_cull(mut self, enabled: bool) -> Self {
        self.frustum_cull = enabled;
        self
    }

    /// Traverse the tileset and return visible tiles
    pub fn visible_tiles<'a>(
        &self,
        tileset: &'a Tileset,
        camera_position: Vec3,
        view_proj: Option<&Mat4>,
    ) -> Vec<VisibleTile<'a>> {
        let mut result = Vec::new();
        let default_refine = tileset.default_refine();
        
        self.traverse_tile(
            tileset.root(),
            Mat4::IDENTITY,
            default_refine,
            camera_position,
            view_proj,
            0,
            &mut result,
        );
        
        result
    }

    /// Traverse a single tile and its children
    fn traverse_tile<'a>(
        &self,
        tile: &'a Tile,
        parent_transform: Mat4,
        parent_refine: TileRefine,
        camera_position: Vec3,
        view_proj: Option<&Mat4>,
        depth: usize,
        result: &mut Vec<VisibleTile<'a>>,
    ) {
        if depth > self.max_depth {
            return;
        }

        let local_transform = tile.get_transform();
        let world_transform = parent_transform * local_transform;
        let world_bounds = tile.bounding_volume.transform(&world_transform);

        // Frustum culling
        if self.frustum_cull {
            if let Some(vp) = view_proj {
                if !world_bounds.intersects_frustum(vp) {
                    return;
                }
            }
        }

        let sse = compute_sse(
            tile.geometric_error,
            &world_bounds,
            camera_position,
            &self.sse_params,
        );

        let refine = tile.effective_refine(parent_refine);
        let should_refine = sse > self.sse_threshold && !tile.children.is_empty();

        if should_refine {
            // Need higher LOD - traverse children
            match refine {
                TileRefine::Replace => {
                    // Children replace parent, don't render this tile
                    for child in &tile.children {
                        self.traverse_tile(
                            child,
                            world_transform,
                            refine,
                            camera_position,
                            view_proj,
                            depth + 1,
                            result,
                        );
                    }
                }
                TileRefine::Add => {
                    // Children add to parent, render this tile AND children
                    if tile.has_content() {
                        result.push(VisibleTile {
                            tile,
                            world_transform,
                            world_bounds: world_bounds.clone(),
                            sse,
                            depth,
                        });
                    }
                    for child in &tile.children {
                        self.traverse_tile(
                            child,
                            world_transform,
                            refine,
                            camera_position,
                            view_proj,
                            depth + 1,
                            result,
                        );
                    }
                }
            }
        } else {
            // SSE is acceptable, render this tile (if it has content)
            if tile.has_content() {
                result.push(VisibleTile {
                    tile,
                    world_transform,
                    world_bounds,
                    sse,
                    depth,
                });
            } else if !tile.children.is_empty() {
                // Empty tile with children - must render children
                for child in &tile.children {
                    self.traverse_tile(
                        child,
                        world_transform,
                        refine,
                        camera_position,
                        view_proj,
                        depth + 1,
                        result,
                    );
                }
            }
        }
    }

    /// Count tiles that would be visible at given SSE threshold
    pub fn count_visible_tiles(
        &self,
        tileset: &Tileset,
        camera_position: Vec3,
    ) -> usize {
        self.visible_tiles(tileset, camera_position, None).len()
    }

    /// Get statistics about the traversal
    pub fn traversal_stats(
        &self,
        tileset: &Tileset,
        camera_position: Vec3,
    ) -> TraversalStats {
        let visible = self.visible_tiles(tileset, camera_position, None);
        
        let max_depth = visible.iter().map(|t| t.depth).max().unwrap_or(0);
        let min_sse = visible.iter().map(|t| t.sse).fold(f32::MAX, f32::min);
        let max_sse = visible.iter().map(|t| t.sse).fold(0.0f32, f32::max);
        let avg_sse = if visible.is_empty() {
            0.0
        } else {
            visible.iter().map(|t| t.sse).sum::<f32>() / visible.len() as f32
        };

        TraversalStats {
            visible_tile_count: visible.len(),
            max_depth,
            min_sse,
            max_sse,
            avg_sse,
        }
    }
}

/// Statistics from a traversal
#[derive(Debug, Clone)]
pub struct TraversalStats {
    /// Number of visible tiles
    pub visible_tile_count: usize,
    /// Maximum depth reached
    pub max_depth: usize,
    /// Minimum SSE among visible tiles
    pub min_sse: f32,
    /// Maximum SSE among visible tiles
    pub max_sse: f32,
    /// Average SSE among visible tiles
    pub avg_sse: f32,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tiles3d::tileset::Tileset;
    use std::path::PathBuf;

    #[test]
    fn test_traverser_default() {
        let traverser = TilesetTraverser::default();
        assert_eq!(traverser.sse_threshold, 16.0);
        assert_eq!(traverser.max_depth, 32);
    }

    #[test]
    fn test_visible_tiles_decreases_with_sse() {
        let json = r#"{
            "asset": { "version": "1.0" },
            "geometricError": 500.0,
            "root": {
                "boundingVolume": { "sphere": [0, 0, 0, 100] },
                "geometricError": 100.0,
                "content": { "uri": "root.b3dm" },
                "children": [
                    {
                        "boundingVolume": { "sphere": [-50, 0, 0, 50] },
                        "geometricError": 10.0,
                        "content": { "uri": "tile1.b3dm" }
                    },
                    {
                        "boundingVolume": { "sphere": [50, 0, 0, 50] },
                        "geometricError": 10.0,
                        "content": { "uri": "tile2.b3dm" }
                    }
                ]
            }
        }"#;
        
        let tileset = Tileset::from_json(json, PathBuf::from(".")).unwrap();
        let camera = Vec3::new(0.0, 0.0, 500.0);
        
        // Low SSE threshold -> more tiles (refined)
        let traverser_low = TilesetTraverser::new(1.0);
        let visible_low = traverser_low.visible_tiles(&tileset, camera, None);
        
        // High SSE threshold -> fewer tiles (coarse)
        let traverser_high = TilesetTraverser::new(1000.0);
        let visible_high = traverser_high.visible_tiles(&tileset, camera, None);
        
        assert!(visible_low.len() >= visible_high.len());
    }
}
