//! Point cloud renderer with caching

use glam::Vec3;
use std::collections::HashMap;

use super::copc::{CopcDataset, PointData as CopcPointData};
use super::ept::{EptDataset, PointData as EptPointData};
use super::error::PointCloudResult;
use super::traversal::{PointCloudTraverser, TraversalParams, VisibleNode};

/// Point buffer ready for GPU upload
#[derive(Debug)]
pub struct PointBuffer {
    pub positions: Vec<f32>,
    pub colors: Option<Vec<u8>>,
    pub point_count: usize,
}

/// Number of floats per vertex in the interleaved GPU buffer: [x, y, z, r, g, b]
const GPU_FLOATS_PER_VERTEX: usize = 6;
/// Matches the viewer's `PointInstance3D` layout (48 bytes / 12 floats).
const VIEWER_FLOATS_PER_VERTEX: usize = 12;

impl PointBuffer {
    pub fn new() -> Self {
        Self {
            positions: Vec::new(),
            colors: None,
            point_count: 0,
        }
    }

    pub fn byte_size(&self) -> usize {
        self.positions.len() * 4 + self.colors.as_ref().map_or(0, |c| c.len())
    }

    /// Create interleaved GPU vertex data: [x, y, z, r, g, b] per point.
    ///
    /// Positions are taken from `self.positions` (packed [x,y,z,x,y,z,...]).
    /// Colors are taken from `self.colors` (packed [r,g,b,r,g,b,...] as u8, normalized to 0..1).
    /// If no colors are present, defaults to white (1.0, 1.0, 1.0).
    pub fn create_gpu_buffer(&self) -> Vec<f32> {
        if self.point_count == 0 {
            return Vec::new();
        }
        let mut out = Vec::with_capacity(self.point_count * GPU_FLOATS_PER_VERTEX);
        let colors = self.colors.as_deref();

        for i in 0..self.point_count {
            let pi = i * 3;
            // Position: x, y, z
            let x = self.positions.get(pi).copied().unwrap_or(0.0);
            let y = self.positions.get(pi + 1).copied().unwrap_or(0.0);
            let z = self.positions.get(pi + 2).copied().unwrap_or(0.0);
            out.push(x);
            out.push(y);
            out.push(z);

            // Color: r, g, b (normalized from u8 or default white)
            if let Some(cols) = colors {
                let ci = i * 3;
                let r = cols.get(ci).copied().unwrap_or(255) as f32 / 255.0;
                let g = cols.get(ci + 1).copied().unwrap_or(255) as f32 / 255.0;
                let b = cols.get(ci + 2).copied().unwrap_or(255) as f32 / 255.0;
                out.push(r);
                out.push(g);
                out.push(b);
            } else {
                out.push(1.0);
                out.push(1.0);
                out.push(1.0);
            }
        }
        out
    }

    /// Byte size of the interleaved GPU buffer that `create_gpu_buffer()` would produce.
    pub fn gpu_byte_size(&self) -> usize {
        self.point_count * GPU_FLOATS_PER_VERTEX * std::mem::size_of::<f32>()
    }

    /// Create viewer-compatible GPU buffer matching `PointInstance3D` layout.
    ///
    /// Layout per point (12 floats = 48 bytes):
    /// `[x, y, z, elevation_norm, r, g, b, intensity, size, pad, pad, pad]`
    ///
    /// `bounds_min`/`bounds_max` normalise elevation (Y axis) to \[0, 1\].
    pub fn create_viewer_gpu_buffer(&self, bounds_min: [f32; 3], bounds_max: [f32; 3]) -> Vec<f32> {
        if self.point_count == 0 {
            return Vec::new();
        }
        let elev_range = (bounds_max[1] - bounds_min[1]).max(0.001);
        let mut out = Vec::with_capacity(self.point_count * VIEWER_FLOATS_PER_VERTEX);
        let colors = self.colors.as_deref();

        for i in 0..self.point_count {
            let pi = i * 3;
            let x = self.positions.get(pi).copied().unwrap_or(0.0);
            let y = self.positions.get(pi + 1).copied().unwrap_or(0.0);
            let z = self.positions.get(pi + 2).copied().unwrap_or(0.0);
            let elev_norm = ((y - bounds_min[1]) / elev_range).clamp(0.0, 1.0);

            let (r, g, b) = if let Some(cols) = colors {
                let ci = i * 3;
                (
                    cols.get(ci).copied().unwrap_or(255) as f32 / 255.0,
                    cols.get(ci + 1).copied().unwrap_or(255) as f32 / 255.0,
                    cols.get(ci + 2).copied().unwrap_or(255) as f32 / 255.0,
                )
            } else {
                (1.0, 1.0, 1.0)
            };

            out.extend_from_slice(&[
                x, y, z,         // position
                elev_norm, // elevation_norm
                r, g, b,   // rgb
                0.5, // intensity (default)
                1.0, // size (default)
                0.0, 0.0, 0.0, // padding
            ]);
        }
        out
    }
}

impl Default for PointBuffer {
    fn default() -> Self {
        Self::new()
    }
}

struct CacheEntry {
    buffer: PointBuffer,
    last_used: std::time::Instant,
}

/// Render statistics
#[derive(Debug, Clone, Default)]
pub struct RenderStats {
    pub nodes_rendered: usize,
    pub points_rendered: u64,
    pub cache_hits: usize,
    pub cache_misses: usize,
}

/// Memory usage report for the point cloud cache.
#[derive(Debug, Clone)]
pub struct MemoryReport {
    /// Bytes currently used by the LRU cache.
    pub cache_used: usize,
    /// Maximum cache budget in bytes.
    pub cache_budget: usize,
    /// Utilization as a fraction (0.0 .. 1.0).
    pub utilization: f64,
    /// Number of entries currently in the cache.
    pub entry_count: usize,
}

/// Point cloud renderer
pub struct PointCloudRenderer {
    cache: HashMap<String, CacheEntry>,
    cache_budget: usize,
    cache_used: usize,
    traverser: PointCloudTraverser,
    stats: RenderStats,
}

impl Default for PointCloudRenderer {
    fn default() -> Self {
        Self::new()
    }
}

impl PointCloudRenderer {
    pub fn new() -> Self {
        Self::with_cache_budget(256 * 1024 * 1024)
    }

    pub fn with_cache_budget(budget: usize) -> Self {
        Self {
            cache: HashMap::new(),
            cache_budget: budget,
            cache_used: 0,
            traverser: PointCloudTraverser::default(),
            stats: RenderStats::default(),
        }
    }

    pub fn set_point_budget(&mut self, budget: u64) {
        self.traverser.set_point_budget(budget);
    }

    pub fn set_traversal_params(&mut self, params: TraversalParams) {
        self.traverser = PointCloudTraverser::new(params);
    }

    /// Get visible nodes from COPC dataset
    pub fn get_visible_copc(&self, dataset: &CopcDataset, camera_pos: Vec3) -> Vec<VisibleNode> {
        let root = dataset.root_node();
        self.traverser
            .visible_nodes(&root, camera_pos, None, |key| dataset.children(key))
    }

    /// Get visible nodes from EPT dataset
    pub fn get_visible_ept(&self, dataset: &EptDataset, camera_pos: Vec3) -> Vec<VisibleNode> {
        let root = dataset.root_node();
        self.traverser
            .visible_nodes(&root, camera_pos, None, |key| dataset.children(key))
    }

    /// Load points for visible nodes from COPC.
    pub fn load_copc_points(
        &mut self,
        dataset: &CopcDataset,
        visible: &[VisibleNode],
    ) -> PointCloudResult<PointBuffer> {
        self.load_points("copc", visible, |key| {
            dataset.read_points(key).map(copc_to_buffer)
        })
    }

    /// Load points for visible nodes from EPT.
    pub fn load_ept_points(
        &mut self,
        dataset: &EptDataset,
        visible: &[VisibleNode],
    ) -> PointCloudResult<PointBuffer> {
        self.load_points("ept", visible, |key| {
            dataset.read_points(key).map(ept_to_buffer)
        })
    }

    fn load_points<F>(
        &mut self,
        prefix: &str,
        visible: &[VisibleNode],
        read_fn: F,
    ) -> PointCloudResult<PointBuffer>
    where
        F: Fn(&super::octree::OctreeKey) -> PointCloudResult<PointBuffer>,
    {
        self.stats = RenderStats::default();
        let mut combined = PointBuffer::new();
        let mut has_colors = false;

        for node in visible {
            let cache_key = format!("{}:{}", prefix, node.key.to_string());

            if let Some(entry) = self.cache.get_mut(&cache_key) {
                entry.last_used = std::time::Instant::now();
                self.stats.cache_hits += 1;

                combined.positions.extend(&entry.buffer.positions);
                if let Some(ref cols) = entry.buffer.colors {
                    has_colors = true;
                    combined.colors.get_or_insert_with(Vec::new).extend(cols);
                }
                combined.point_count += entry.buffer.point_count;
            } else {
                self.stats.cache_misses += 1;

                match read_fn(&node.key) {
                    Ok(buffer) => {
                        let byte_size = buffer.byte_size();
                        combined.positions.extend(&buffer.positions);
                        if let Some(ref cols) = buffer.colors {
                            has_colors = true;
                            combined.colors.get_or_insert_with(Vec::new).extend(cols);
                        }
                        combined.point_count += buffer.point_count;

                        self.ensure_cache_space(byte_size);
                        self.cache.insert(
                            cache_key,
                            CacheEntry {
                                buffer,
                                last_used: std::time::Instant::now(),
                            },
                        );
                        self.cache_used += byte_size;
                    }
                    Err(_) => continue,
                }
            }
        }

        if !has_colors {
            combined.colors = None;
        }
        self.stats.nodes_rendered = visible.len();
        self.stats.points_rendered = combined.point_count as u64;
        Ok(combined)
    }

    fn ensure_cache_space(&mut self, needed: usize) {
        while self.cache_used + needed > self.cache_budget && !self.cache.is_empty() {
            let oldest = self
                .cache
                .iter()
                .min_by_key(|(_, e)| e.last_used)
                .map(|(k, _)| k.clone());

            if let Some(key) = oldest {
                if let Some(entry) = self.cache.remove(&key) {
                    self.cache_used = self.cache_used.saturating_sub(entry.buffer.byte_size());
                }
            }
        }
    }

    pub fn stats(&self) -> &RenderStats {
        &self.stats
    }

    /// Cache budget in bytes.
    pub fn cache_budget(&self) -> usize {
        self.cache_budget
    }

    /// Current cache usage in bytes.
    pub fn cache_used(&self) -> usize {
        self.cache_used
    }

    /// Report on memory usage vs budget.
    pub fn memory_report(&self) -> MemoryReport {
        let utilization = if self.cache_budget > 0 {
            self.cache_used as f64 / self.cache_budget as f64
        } else {
            0.0
        };
        MemoryReport {
            cache_used: self.cache_used,
            cache_budget: self.cache_budget,
            utilization,
            entry_count: self.cache.len(),
        }
    }

    pub fn clear_cache(&mut self) {
        self.cache.clear();
        self.cache_used = 0;
    }
}

fn copc_to_buffer(data: CopcPointData) -> PointBuffer {
    let point_count = data.positions.len() / 3;
    PointBuffer {
        positions: data.positions,
        colors: data.colors,
        point_count,
    }
}

fn ept_to_buffer(data: EptPointData) -> PointBuffer {
    let point_count = data.positions.len() / 3;
    PointBuffer {
        positions: data.positions,
        colors: data.colors,
        point_count,
    }
}
