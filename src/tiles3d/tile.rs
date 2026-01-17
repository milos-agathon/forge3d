//! Tile structure for 3D Tiles

use glam::Mat4;
use serde::{Deserialize, Serialize};
use super::bounds::BoundingVolume;

/// Refinement strategy for child tiles
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "UPPERCASE")]
pub enum TileRefine {
    /// Replace parent tile with children
    Replace,
    /// Add children to parent tile
    Add,
}

impl Default for TileRefine {
    fn default() -> Self {
        Self::Replace
    }
}

/// Content description for a tile
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TileContent {
    /// URI to the tile content (b3dm, pnts, i3dm, cmpt, etc.)
    pub uri: String,
    /// Optional bounding volume for the content (tighter than tile bounds)
    #[serde(rename = "boundingVolume")]
    pub bounding_volume: Option<BoundingVolume>,
}

/// A single tile in the 3D Tiles hierarchy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tile {
    /// Bounding volume enclosing the tile
    #[serde(rename = "boundingVolume")]
    pub bounding_volume: BoundingVolume,
    /// Geometric error in meters (controls LOD selection)
    #[serde(rename = "geometricError")]
    pub geometric_error: f32,
    /// Optional refinement strategy (inherited from parent if not specified)
    #[serde(default)]
    pub refine: Option<TileRefine>,
    /// Optional content (tile may be empty, containing only children)
    pub content: Option<TileContent>,
    /// Child tiles
    #[serde(default)]
    pub children: Vec<Tile>,
    /// Optional 4x4 transform matrix (column-major)
    #[serde(default)]
    pub transform: Option<[f64; 16]>,
    /// Viewer request volume (optional)
    #[serde(rename = "viewerRequestVolume")]
    pub viewer_request_volume: Option<BoundingVolume>,
}

impl Tile {
    /// Get the transform matrix for this tile
    pub fn get_transform(&self) -> Mat4 {
        self.transform
            .map(|t| {
                Mat4::from_cols_array(&[
                    t[0] as f32, t[1] as f32, t[2] as f32, t[3] as f32,
                    t[4] as f32, t[5] as f32, t[6] as f32, t[7] as f32,
                    t[8] as f32, t[9] as f32, t[10] as f32, t[11] as f32,
                    t[12] as f32, t[13] as f32, t[14] as f32, t[15] as f32,
                ])
            })
            .unwrap_or(Mat4::IDENTITY)
    }

    /// Get the world-space bounding volume (with transform applied)
    pub fn world_bounding_volume(&self, parent_transform: &Mat4) -> BoundingVolume {
        let local_transform = self.get_transform();
        let world_transform = *parent_transform * local_transform;
        self.bounding_volume.transform(&world_transform)
    }

    /// Get effective refinement strategy (inherits from parent if not specified)
    pub fn effective_refine(&self, parent_refine: TileRefine) -> TileRefine {
        self.refine.unwrap_or(parent_refine)
    }

    /// Check if this tile has renderable content
    pub fn has_content(&self) -> bool {
        self.content.is_some()
    }

    /// Get the content URI if present
    pub fn content_uri(&self) -> Option<&str> {
        self.content.as_ref().map(|c| c.uri.as_str())
    }

    /// Count total tiles in this subtree
    pub fn count_tiles(&self) -> usize {
        1 + self.children.iter().map(|c| c.count_tiles()).sum::<usize>()
    }

    /// Get maximum depth of this subtree
    pub fn max_depth(&self) -> usize {
        if self.children.is_empty() {
            1
        } else {
            1 + self.children.iter().map(|c| c.max_depth()).max().unwrap_or(0)
        }
    }
}

impl Default for Tile {
    fn default() -> Self {
        Self {
            bounding_volume: BoundingVolume::default(),
            geometric_error: 0.0,
            refine: None,
            content: None,
            children: Vec::new(),
            transform: None,
            viewer_request_volume: None,
        }
    }
}
