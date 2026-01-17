//! P5: 3D Tiles support for forge3d
//!
//! This module provides parsing, traversal, and rendering of OGC 3D Tiles datasets.
//! Supports tileset.json, b3dm (batched 3D model), and pnts (point cloud) payloads.

mod bounds;
mod error;
mod tile;
mod tileset;
mod b3dm;
mod pnts;
mod sse;
mod traversal;
mod renderer;

pub use bounds::{BoundingVolume, BoundingBox, BoundingSphere, BoundingRegion};
pub use error::{Tiles3dError, Tiles3dResult};
pub use tile::{Tile, TileContent, TileRefine};
pub use tileset::Tileset;
pub use b3dm::{B3dmPayload, B3dmHeader, decode_b3dm, load_b3dm};
pub use pnts::{PntsPayload, PntsHeader, decode_pnts, load_pnts};
pub use sse::{compute_sse, compute_sse_with_matrix, compute_sse_surface, should_refine, distance_to_surface};
pub use traversal::TilesetTraverser;
pub use renderer::Tiles3dRenderer;
