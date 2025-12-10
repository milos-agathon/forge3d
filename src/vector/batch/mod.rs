//! Batching and visibility culling for vector primitives.
//!
//! Provides AABB computation, frustum culling, and bucketed batching
//! with performance counters.

mod aabb;
mod frustum;
mod stats;

pub use aabb::AABB;
pub use frustum::Frustum;
pub use stats::BatchingStats;

use crate::core::error::RenderError;
use crate::vector::api::{PointDef, PolygonDef, PolylineDef, VectorId};
use crate::vector::layer::Layer;
use glam::{Mat4, Vec2};

/// Batched primitive data ready for rendering.
#[derive(Debug)]
pub struct Batch {
    pub layer: Layer,
    pub primitive_type: PrimitiveType,
    pub aabb: AABB,
    pub primitive_ids: Vec<VectorId>,
    pub vertex_count: u32,
    pub instance_count: u32,
}

/// Type of primitive in a batch.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u32)]
pub enum PrimitiveType {
    Polygon,
    Line,
    Point,
    GraphNode,
    GraphEdge,
    Triangle,
}

/// Batching system for vector primitives with frustum culling.
pub struct BatchManager {
    stats: BatchingStats,
    frustum: Option<Frustum>,
    polygon_data: Vec<(VectorId, PolygonDef, AABB)>,
    line_data: Vec<(VectorId, PolylineDef, AABB)>,
    point_data: Vec<(VectorId, PointDef, AABB)>,
    max_batch_size: usize,
    area_threshold: f32,
}

impl BatchManager {
    /// Create a new batch manager with default settings.
    pub fn new() -> Self {
        Self {
            stats: BatchingStats::default(),
            frustum: None,
            polygon_data: Vec::new(),
            line_data: Vec::new(),
            point_data: Vec::new(),
            max_batch_size: 1000,
            area_threshold: 100.0,
        }
    }

    /// Set the view frustum from a view-projection matrix.
    pub fn set_view_frustum(&mut self, vp_matrix: &Mat4) {
        self.frustum = Some(Frustum::from_view_proj_matrix(vp_matrix));
    }

    /// Clear all primitive data and reset statistics.
    pub fn clear(&mut self) {
        self.polygon_data.clear();
        self.line_data.clear();
        self.point_data.clear();
        self.stats = BatchingStats::default();
    }

    /// Add a polygon primitive for batching.
    pub fn add_polygon(&mut self, id: VectorId, polygon: PolygonDef) -> Result<(), RenderError> {
        let aabb = AABB::from_points(&polygon.exterior)
            .ok_or_else(|| RenderError::Upload("Empty polygon exterior".to_string()))?;
        self.polygon_data.push((id, polygon, aabb));
        Ok(())
    }

    /// Add a line primitive for batching.
    pub fn add_line(&mut self, id: VectorId, line: PolylineDef) -> Result<(), RenderError> {
        let aabb = AABB::from_points(&line.path)
            .ok_or_else(|| RenderError::Upload("Empty line path".to_string()))?;
        self.line_data.push((id, line, aabb));
        Ok(())
    }

    /// Add a point primitive for batching.
    pub fn add_point(&mut self, id: VectorId, point: PointDef) -> Result<(), RenderError> {
        let half_size = point.style.point_size * 0.5;
        let aabb = AABB {
            min: point.position - Vec2::splat(half_size),
            max: point.position + Vec2::splat(half_size),
        };
        self.point_data.push((id, point, aabb));
        Ok(())
    }

    /// Perform visibility culling and generate batches.
    pub fn generate_batches(&mut self) -> Result<Vec<Batch>, RenderError> {
        let start_time = std::time::Instant::now();

        let mut batches = Vec::new();
        self.stats.total_primitives =
            self.polygon_data.len() + self.line_data.len() + self.point_data.len();
        self.stats.draw_calls_before_batching = self.stats.total_primitives;

        // Cull and batch polygons
        let (visible_polygons, poly_stats) = cull_primitives(&self.polygon_data, &self.frustum);
        self.stats.visible_primitives += poly_stats.0;
        self.stats.culled_primitives += poly_stats.1;
        self.stats.culling_time_ms += poly_stats.2;
        batches.extend(self.create_batches(
            visible_polygons.into_iter().map(|(id, _, aabb)| (id, aabb)).collect(),
            PrimitiveType::Polygon,
            Layer::Background,
        ));

        // Cull and batch lines
        let (visible_lines, line_stats) = cull_primitives(&self.line_data, &self.frustum);
        self.stats.visible_primitives += line_stats.0;
        self.stats.culled_primitives += line_stats.1;
        self.stats.culling_time_ms += line_stats.2;
        batches.extend(self.create_batches(
            visible_lines.into_iter().map(|(id, _, aabb)| (id, aabb)).collect(),
            PrimitiveType::Line,
            Layer::Vector,
        ));

        // Cull and batch points
        let (visible_points, point_stats) = cull_primitives(&self.point_data, &self.frustum);
        self.stats.visible_primitives += point_stats.0;
        self.stats.culled_primitives += point_stats.1;
        self.stats.culling_time_ms += point_stats.2;
        batches.extend(self.create_batches(
            visible_points.into_iter().map(|(id, _, aabb)| (id, aabb)).collect(),
            PrimitiveType::Point,
            Layer::Points,
        ));

        self.stats.draw_calls_after_batching = batches.len();
        self.stats.batching_time_ms = start_time.elapsed().as_secs_f32() * 1000.0;

        Ok(batches)
    }

    fn create_batches(
        &self,
        primitives: Vec<(VectorId, AABB)>,
        primitive_type: PrimitiveType,
        layer: Layer,
    ) -> Vec<Batch> {
        if primitives.is_empty() {
            return Vec::new();
        }

        let mut batches = Vec::new();
        let mut current_batch_ids = Vec::new();
        let mut current_batch_aabb: Option<AABB> = None;

        for (id, aabb) in primitives {
            let should_start_new_batch = current_batch_ids.len() >= self.max_batch_size
                || (current_batch_aabb.is_some()
                    && (aabb.area() > current_batch_aabb.unwrap().area() * self.area_threshold
                        || current_batch_aabb.unwrap().area() > aabb.area() * self.area_threshold));

            if should_start_new_batch && !current_batch_ids.is_empty() {
                batches.push(Batch {
                    layer,
                    primitive_type,
                    aabb: current_batch_aabb.unwrap(),
                    primitive_ids: std::mem::take(&mut current_batch_ids),
                    vertex_count: 0,
                    instance_count: 0,
                });
                current_batch_aabb = None;
            }

            current_batch_ids.push(id);
            current_batch_aabb = Some(match current_batch_aabb {
                Some(existing) => existing.union(&aabb),
                None => aabb,
            });
        }

        if !current_batch_ids.is_empty() {
            batches.push(Batch {
                layer,
                primitive_type,
                aabb: current_batch_aabb.unwrap(),
                primitive_ids: current_batch_ids,
                vertex_count: 0,
                instance_count: 0,
            });
        }

        batches
    }

    /// Get current batching statistics.
    pub fn get_stats(&self) -> &BatchingStats {
        &self.stats
    }

    /// Set maximum primitives per batch.
    pub fn set_max_batch_size(&mut self, size: usize) {
        self.max_batch_size = size;
    }

    /// Set area difference threshold for batching.
    pub fn set_area_threshold(&mut self, threshold: f32) {
        self.area_threshold = threshold;
    }
}

/// Cull primitives against frustum and return visible items with stats.
///
/// Returns: (visible_items, (visible_count, culled_count, time_ms))
fn cull_primitives<T: Clone>(
    data: &[(VectorId, T, AABB)],
    frustum: &Option<Frustum>,
) -> (Vec<(VectorId, T, AABB)>, (usize, usize, f32)) {
    let cull_start = std::time::Instant::now();
    let mut visible = Vec::new();
    let mut visible_count = 0usize;
    let mut culled_count = 0usize;

    for (id, item, aabb) in data {
        if let Some(ref f) = frustum {
            if f.test_aabb(aabb) {
                visible.push((*id, item.clone(), *aabb));
                visible_count += 1;
            } else {
                culled_count += 1;
            }
        } else {
            visible.push((*id, item.clone(), *aabb));
            visible_count += 1;
        }
    }

    let time_ms = cull_start.elapsed().as_secs_f32() * 1000.0;
    (visible, (visible_count, culled_count, time_ms))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_batch_manager() {
        let mut manager = BatchManager::new();

        let polygon = crate::vector::api::PolygonDef {
            exterior: vec![
                Vec2::new(0.0, 0.0),
                Vec2::new(1.0, 0.0),
                Vec2::new(0.5, 1.0),
            ],
            holes: vec![],
            style: crate::vector::api::VectorStyle::default(),
        };

        manager.add_polygon(VectorId(1), polygon).unwrap();

        let batches = manager.generate_batches().unwrap();
        assert!(!batches.is_empty());

        let stats = manager.get_stats();
        assert_eq!(stats.total_primitives, 1);
    }
}
