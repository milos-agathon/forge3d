//! P5: Point Cloud support for forge3d
//!
//! This module provides parsing, LOD traversal, and rendering of point cloud datasets.
//! Supports COPC (Cloud Optimized Point Cloud) and EPT (Entwine Point Tile) formats.

mod copc;
mod copc_decode;
mod ept;
mod error;
mod octree;
mod renderer;
mod traversal;

pub use copc::{CopcDataset, CopcHeader, CopcInfo};
pub use ept::{EptDataset, EptInfo, EptSchema};
pub use error::{PointCloudError, PointCloudResult};
pub use octree::{OctreeBounds, OctreeKey, OctreeNode};
pub use renderer::{MemoryReport, PointBuffer, PointCloudRenderer, RenderStats};
pub use traversal::{PointCloudTraverser, TraversalParams, VisibleNode};
