//! 3D Tiles renderer with caching
//! Extended with P4: 3D Buildings Pipeline support

use glam::{Mat4, Vec3};
use std::collections::HashMap;
use std::path::PathBuf;

use super::b3dm::decode_b3dm;
use super::error::{Tiles3dError, Tiles3dResult};
use super::pnts::decode_pnts;
use super::sse::SseParams;
use super::tileset::Tileset;
use super::traversal::{TilesetTraverser, VisibleTile};
use crate::import::building_materials::BuildingMaterial;
use crate::import::cityjson::BuildingGeom;

/// Cached tile content
#[derive(Debug)]
pub enum TileContent {
    Mesh(MeshData),
    Points(PointData),
}

/// Mesh data from b3dm
#[derive(Debug)]
pub struct MeshData {
    pub positions: Vec<f32>,
    pub normals: Option<Vec<f32>>,
    pub colors: Option<Vec<u8>>,
    pub indices: Vec<u32>,
}

/// Point data from pnts
#[derive(Debug)]
pub struct PointData {
    pub positions: Vec<f32>,
    pub colors: Option<Vec<u8>>,
    pub normals: Option<Vec<f32>>,
}

struct CacheEntry {
    content: TileContent,
    byte_size: usize,
    last_used: std::time::Instant,
}

/// 3D Tiles renderer with LRU caching
pub struct Tiles3dRenderer {
    cache: HashMap<String, CacheEntry>,
    cache_budget: usize,
    cache_used: usize,
    traverser: TilesetTraverser,
    cache_hits: usize,
    cache_misses: usize,
}

impl Default for Tiles3dRenderer {
    fn default() -> Self {
        Self::new()
    }
}

impl Tiles3dRenderer {
    pub fn new() -> Self {
        Self::with_cache_budget(256 * 1024 * 1024)
    }

    pub fn with_cache_budget(budget_bytes: usize) -> Self {
        Self {
            cache: HashMap::new(),
            cache_budget: budget_bytes,
            cache_used: 0,
            traverser: TilesetTraverser::default(),
            cache_hits: 0,
            cache_misses: 0,
        }
    }

    pub fn set_sse_threshold(&mut self, threshold: f32) {
        self.traverser.sse_threshold = threshold;
    }

    pub fn set_sse_params(&mut self, params: SseParams) {
        self.traverser.sse_params = params;
    }

    /// Get visible tiles for rendering
    pub fn get_visible_tiles<'a>(
        &self,
        tileset: &'a Tileset,
        camera_pos: Vec3,
        view_proj: Option<&Mat4>,
    ) -> Vec<VisibleTile<'a>> {
        self.traverser.visible_tiles(tileset, camera_pos, view_proj)
    }

    /// Load tile content with caching
    pub fn load_tile_content(
        &mut self,
        tileset: &Tileset,
        uri: &str,
    ) -> Tiles3dResult<&TileContent> {
        if self.cache.contains_key(uri) {
            self.cache_hits += 1;
            if let Some(entry) = self.cache.get_mut(uri) {
                entry.last_used = std::time::Instant::now();
            }
            return Ok(&self.cache.get(uri).unwrap().content);
        }

        self.cache_misses += 1;
        let path = tileset.resolve_uri(uri);
        let content = self.load_content_from_path(&path)?;
        let byte_size = estimate_content_size(&content);

        self.ensure_cache_space(byte_size);
        
        self.cache.insert(uri.to_string(), CacheEntry {
            content,
            byte_size,
            last_used: std::time::Instant::now(),
        });
        self.cache_used += byte_size;

        Ok(&self.cache.get(uri).unwrap().content)
    }

    fn load_content_from_path(&self, path: &PathBuf) -> Tiles3dResult<TileContent> {
        let ext = path.extension()
            .and_then(|e| e.to_str())
            .unwrap_or("")
            .to_lowercase();

        let data = std::fs::read(path)?;

        match ext.as_str() {
            "b3dm" => {
                let payload = decode_b3dm(&data)?;
                Ok(TileContent::Mesh(MeshData {
                    positions: payload.positions,
                    normals: payload.normals,
                    colors: payload.colors,
                    indices: payload.indices,
                }))
            }
            "pnts" => {
                let payload = decode_pnts(&data)?;
                let colors = payload.colors_rgba.or(payload.colors);
                Ok(TileContent::Points(PointData {
                    positions: payload.positions,
                    colors,
                    normals: payload.normals,
                }))
            }
            _ => Err(Tiles3dError::Unsupported(format!("Unknown format: {}", ext))),
        }
    }

    fn ensure_cache_space(&mut self, needed: usize) {
        while self.cache_used + needed > self.cache_budget && !self.cache.is_empty() {
            let oldest = self.cache
                .iter()
                .min_by_key(|(_, e)| e.last_used)
                .map(|(k, _)| k.clone());
            
            if let Some(key) = oldest {
                if let Some(entry) = self.cache.remove(&key) {
                    self.cache_used = self.cache_used.saturating_sub(entry.byte_size);
                }
            }
        }
    }

    pub fn cache_stats(&self) -> CacheStats {
        CacheStats {
            hits: self.cache_hits,
            misses: self.cache_misses,
            entries: self.cache.len(),
            used_bytes: self.cache_used,
            budget_bytes: self.cache_budget,
        }
    }

    pub fn clear_cache(&mut self) {
        self.cache.clear();
        self.cache_used = 0;
    }
}

fn estimate_content_size(content: &TileContent) -> usize {
    match content {
        TileContent::Mesh(m) => {
            m.positions.len() * 4
                + m.normals.as_ref().map_or(0, |n| n.len() * 4)
                + m.colors.as_ref().map_or(0, |c| c.len())
                + m.indices.len() * 4
        }
        TileContent::Points(p) => {
            p.positions.len() * 4
                + p.colors.as_ref().map_or(0, |c| c.len())
                + p.normals.as_ref().map_or(0, |n| n.len() * 4)
        }
    }
}

#[derive(Debug, Clone)]
pub struct CacheStats {
    pub hits: usize,
    pub misses: usize,
    pub entries: usize,
    pub used_bytes: usize,
    pub budget_bytes: usize,
}

impl CacheStats {
    pub fn hit_rate(&self) -> f32 {
        let total = self.hits + self.misses;
        if total == 0 { 0.0 } else { self.hits as f32 / total as f32 }
    }
}

// ============================================================================
// P4: 3D Buildings Pipeline - Building Render Data
// ============================================================================

/// Prepared building data for GPU rendering
#[derive(Debug, Clone)]
pub struct BuildingRenderData {
    /// Flat position array [x, y, z, x, y, z, ...]
    pub positions: Vec<f32>,
    /// Flat normal array
    pub normals: Vec<f32>,
    /// Triangle indices
    pub indices: Vec<u32>,
    /// Per-building instance data (for instanced rendering)
    pub instances: Vec<BuildingInstance>,
    /// Total vertex count
    pub vertex_count: usize,
    /// Total triangle count
    pub triangle_count: usize,
}

/// Per-building instance data
#[derive(Debug, Clone, Copy)]
pub struct BuildingInstance {
    /// Index into positions/normals buffer
    pub vertex_offset: u32,
    /// Index into indices buffer
    pub index_offset: u32,
    /// Number of indices for this building
    pub index_count: u32,
    /// Building material properties
    pub material: BuildingMaterial,
}

impl Default for BuildingRenderData {
    fn default() -> Self {
        Self::new()
    }
}

impl BuildingRenderData {
    pub fn new() -> Self {
        Self {
            positions: Vec::new(),
            normals: Vec::new(),
            indices: Vec::new(),
            instances: Vec::new(),
            vertex_count: 0,
            triangle_count: 0,
        }
    }

    /// Create from a slice of BuildingGeom
    pub fn from_buildings(buildings: &[BuildingGeom]) -> Self {
        let mut data = Self::new();
        data.add_buildings(buildings);
        data
    }

    /// Add buildings to the render data
    pub fn add_buildings(&mut self, buildings: &[BuildingGeom]) {
        for building in buildings {
            if building.is_empty() {
                continue;
            }

            let vertex_offset = self.positions.len() as u32 / 3;
            let index_offset = self.indices.len() as u32;

            // Add positions
            self.positions.extend_from_slice(&building.positions);

            // Add normals (generate if missing)
            if let Some(ref normals) = building.normals {
                self.normals.extend_from_slice(normals);
            } else {
                // Pad with up-facing normals
                let n_verts = building.vertex_count();
                for _ in 0..n_verts {
                    self.normals.extend_from_slice(&[0.0, 0.0, 1.0]);
                }
            }

            // Add indices (offset by vertex base)
            for &idx in &building.indices {
                self.indices.push(idx + vertex_offset);
            }

            // Add instance
            self.instances.push(BuildingInstance {
                vertex_offset,
                index_offset,
                index_count: building.indices.len() as u32,
                material: building.material,
            });
        }

        self.vertex_count = self.positions.len() / 3;
        self.triangle_count = self.indices.len() / 3;
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.positions.is_empty()
    }

    /// Get building count
    pub fn building_count(&self) -> usize {
        self.instances.len()
    }

    /// Get bounding box [min_x, min_y, min_z, max_x, max_y, max_z]
    pub fn bounds(&self) -> Option<[f32; 6]> {
        if self.positions.is_empty() {
            return None;
        }

        let mut min_x = f32::MAX;
        let mut min_y = f32::MAX;
        let mut min_z = f32::MAX;
        let mut max_x = f32::MIN;
        let mut max_y = f32::MIN;
        let mut max_z = f32::MIN;

        for chunk in self.positions.chunks(3) {
            if chunk.len() < 3 {
                continue;
            }
            min_x = min_x.min(chunk[0]);
            min_y = min_y.min(chunk[1]);
            min_z = min_z.min(chunk[2]);
            max_x = max_x.max(chunk[0]);
            max_y = max_y.max(chunk[1]);
            max_z = max_z.max(chunk[2]);
        }

        Some([min_x, min_y, min_z, max_x, max_y, max_z])
    }
}

impl Tiles3dRenderer {
    /// P4.4: Prepare building render data from BuildingGeom slice
    ///
    /// This batches multiple buildings into a single draw call-ready format.
    /// Use with `get_building_render_data` to access the prepared data.
    pub fn prepare_buildings(&self, buildings: &[BuildingGeom]) -> BuildingRenderData {
        BuildingRenderData::from_buildings(buildings)
    }

    /// P4.4: Get render data for buildings visible from camera position
    ///
    /// Applies simple distance-based culling to filter buildings.
    pub fn get_visible_buildings(
        &self,
        buildings: &[BuildingGeom],
        camera_pos: Vec3,
        max_distance: f32,
    ) -> BuildingRenderData {
        let max_dist_sq = max_distance * max_distance;

        let visible: Vec<&BuildingGeom> = buildings
            .iter()
            .filter(|b| {
                if b.positions.len() < 3 {
                    return false;
                }
                // Use first vertex as building position
                let bx = b.positions[0];
                let by = b.positions[1];
                let bz = b.positions[2];
                let dist_sq = (bx - camera_pos.x).powi(2)
                    + (by - camera_pos.y).powi(2)
                    + (bz - camera_pos.z).powi(2);
                dist_sq <= max_dist_sq
            })
            .collect();

        let owned: Vec<BuildingGeom> = visible.into_iter().cloned().collect();
        BuildingRenderData::from_buildings(&owned)
    }
}
