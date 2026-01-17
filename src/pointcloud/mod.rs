//! P5: Point Cloud support for forge3d
//!
//! This module provides parsing, LOD traversal, and rendering of point cloud datasets.
//! Supports COPC (Cloud Optimized Point Cloud) and EPT (Entwine Point Tile) formats.

mod error;
mod octree;
mod copc;
mod ept;
mod traversal;
mod renderer;

pub use error::{PointCloudError, PointCloudResult};
pub use octree::{OctreeNode, OctreeKey, OctreeBounds};
pub use copc::{CopcDataset, CopcHeader, CopcInfo};
pub use ept::{EptDataset, EptSchema, EptInfo};
pub use traversal::{PointCloudTraverser, VisibleNode, TraversalParams};
pub use renderer::{PointCloudRenderer, PointBuffer, RenderStats};
