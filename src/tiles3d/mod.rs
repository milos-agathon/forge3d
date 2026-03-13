//! P5: 3D Tiles support for forge3d
//!
//! This module provides parsing, traversal, and rendering of OGC 3D Tiles datasets.
//! Supports tileset.json, b3dm (batched 3D model), and pnts (point cloud) payloads.

mod b3dm;
mod bounds;
mod error;
mod pnts;
mod renderer;
mod sse;
mod tile;
mod tileset;
mod traversal;

pub use b3dm::{decode_b3dm, load_b3dm, B3dmHeader, B3dmPayload};
pub use bounds::{BoundingBox, BoundingRegion, BoundingSphere, BoundingVolume};
pub use error::{Tiles3dError, Tiles3dResult};
pub use pnts::{decode_pnts, load_pnts, PntsHeader, PntsPayload};
pub use renderer::Tiles3dRenderer;
pub use sse::{
    compute_sse, compute_sse_surface, compute_sse_with_matrix, distance_to_surface, should_refine,
};
pub use tile::{Tile, TileContent, TileRefine};
pub use tileset::Tileset;
pub use traversal::TilesetTraverser;
