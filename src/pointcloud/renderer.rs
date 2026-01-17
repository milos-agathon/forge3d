//! Point cloud renderer with caching

use std::collections::HashMap;
use glam::Vec3;

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

impl PointBuffer {
    pub fn new() -> Self {
        Self { positions: Vec::new(), colors: None, point_count: 0 }
    }

    pub fn byte_size(&self) -> usize {
        self.positions.len() * 4 + self.colors.as_ref().map_or(0, |c| c.len())
    }
}

impl Default for PointBuffer {
    fn default() -> Self { Self::new() }
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

/// Point cloud renderer
pub struct PointCloudRenderer {
    cache: HashMap<String, CacheEntry>,
    cache_budget: usize,
    cache_used: usize,
    traverser: PointCloudTraverser,
    stats: RenderStats,
}

impl Default for PointCloudRenderer {
    fn default() -> Self { Self::new() }
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
    pub fn get_visible_copc(
        &self,
        dataset: &CopcDataset,
        camera_pos: Vec3,
    ) -> Vec<VisibleNode> {
        let root = dataset.root_node();
        self.traverser.visible_nodes(
            &root,
            camera_pos,
            None,
            |key| dataset.children(key),
        )
    }

    /// Get visible nodes from EPT dataset
    pub fn get_visible_ept(
        &self,
        dataset: &EptDataset,
        camera_pos: Vec3,
    ) -> Vec<VisibleNode> {
        let root = dataset.root_node();
        self.traverser.visible_nodes(
            &root,
            camera_pos,
            None,
            |key| dataset.children(key),
        )
    }

    /// Load points for visible nodes from COPC
    pub fn load_copc_points(
        &mut self,
        dataset: &CopcDataset,
        visible: &[VisibleNode],
    ) -> PointCloudResult<PointBuffer> {
        self.stats = RenderStats::default();
        let mut combined = PointBuffer::new();
        let mut has_colors = false;
        
        for node in visible {
            let cache_key = format!("copc:{}", node.key.to_string());
            
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
                
                match dataset.read_points(&node.key) {
                    Ok(data) => {
                        let buffer = copc_to_buffer(data);
                        let byte_size = buffer.byte_size();
                        
                        combined.positions.extend(&buffer.positions);
                        if let Some(ref cols) = buffer.colors {
                            has_colors = true;
                            combined.colors.get_or_insert_with(Vec::new).extend(cols);
                        }
                        combined.point_count += buffer.point_count;
                        
                        self.ensure_cache_space(byte_size);
                        self.cache.insert(cache_key, CacheEntry {
                            buffer,
                            last_used: std::time::Instant::now(),
                        });
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

    /// Load points for visible nodes from EPT
    pub fn load_ept_points(
        &mut self,
        dataset: &EptDataset,
        visible: &[VisibleNode],
    ) -> PointCloudResult<PointBuffer> {
        self.stats = RenderStats::default();
        let mut combined = PointBuffer::new();
        let mut has_colors = false;
        
        for node in visible {
            let cache_key = format!("ept:{}", node.key.to_string());
            
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
                
                match dataset.read_points(&node.key) {
                    Ok(data) => {
                        let buffer = ept_to_buffer(data);
                        let byte_size = buffer.byte_size();
                        
                        combined.positions.extend(&buffer.positions);
                        if let Some(ref cols) = buffer.colors {
                            has_colors = true;
                            combined.colors.get_or_insert_with(Vec::new).extend(cols);
                        }
                        combined.point_count += buffer.point_count;
                        
                        self.ensure_cache_space(byte_size);
                        self.cache.insert(cache_key, CacheEntry {
                            buffer,
                            last_used: std::time::Instant::now(),
                        });
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
            let oldest = self.cache.iter()
                .min_by_key(|(_, e)| e.last_used)
                .map(|(k, _)| k.clone());
            
            if let Some(key) = oldest {
                if let Some(entry) = self.cache.remove(&key) {
                    self.cache_used = self.cache_used.saturating_sub(entry.buffer.byte_size());
                }
            }
        }
    }

    pub fn stats(&self) -> &RenderStats { &self.stats }
    
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
