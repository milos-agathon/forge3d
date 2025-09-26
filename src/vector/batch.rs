//! H4,H10: Batching & visibility culling for vector primitives
//! AABB computation, frustum culling, and bucketed batching with performance counters

use crate::error::RenderError;
use crate::vector::api::{PointDef, PolygonDef, PolylineDef, VectorId};
use crate::vector::layer::Layer;
use glam::{Mat4, Vec2, Vec3, Vec4};

/// Axis-aligned bounding box
#[derive(Debug, Clone, Copy)]
pub struct AABB {
    pub min: Vec2,
    pub max: Vec2,
}

impl AABB {
    pub fn new(min: Vec2, max: Vec2) -> Self {
        Self { min, max }
    }

    pub fn from_points(points: &[Vec2]) -> Option<Self> {
        if points.is_empty() {
            return None;
        }

        let mut min = points[0];
        let mut max = points[0];

        for &point in points.iter().skip(1) {
            min = min.min(point);
            max = max.max(point);
        }

        Some(Self { min, max })
    }

    pub fn center(&self) -> Vec2 {
        (self.min + self.max) * 0.5
    }

    pub fn size(&self) -> Vec2 {
        self.max - self.min
    }

    pub fn area(&self) -> f32 {
        let size = self.size();
        size.x * size.y
    }

    pub fn contains_point(&self, point: Vec2) -> bool {
        point.x >= self.min.x
            && point.x <= self.max.x
            && point.y >= self.min.y
            && point.y <= self.max.y
    }

    pub fn intersects(&self, other: &AABB) -> bool {
        self.min.x <= other.max.x
            && self.max.x >= other.min.x
            && self.min.y <= other.max.y
            && self.max.y >= other.min.y
    }

    pub fn union(&self, other: &AABB) -> AABB {
        AABB {
            min: self.min.min(other.min),
            max: self.max.max(other.max),
        }
    }
}

/// Frustum for visibility culling
#[derive(Debug, Clone)]
pub struct Frustum {
    pub planes: [Vec4; 4], // Left, right, bottom, top (2D frustum)
}

impl Frustum {
    /// Create frustum from view-projection matrix
    pub fn from_view_proj_matrix(vp_matrix: &Mat4) -> Self {
        // Extract frustum planes from view-projection matrix
        // For 2D, we only need left, right, bottom, top planes
        let m = vp_matrix.transpose(); // Row-major access

        let left = (m.w_axis + m.x_axis).normalize();
        let right = (m.w_axis - m.x_axis).normalize();
        let bottom = (m.w_axis + m.y_axis).normalize();
        let top = (m.w_axis - m.y_axis).normalize();

        Self {
            planes: [left, right, bottom, top],
        }
    }

    /// Test AABB against frustum
    pub fn test_aabb(&self, aabb: &AABB) -> bool {
        // Test all 4 corners of AABB against all planes
        let corners = [
            Vec3::new(aabb.min.x, aabb.min.y, 0.0),
            Vec3::new(aabb.max.x, aabb.min.y, 0.0),
            Vec3::new(aabb.min.x, aabb.max.y, 0.0),
            Vec3::new(aabb.max.x, aabb.max.y, 0.0),
        ];

        for plane in &self.planes {
            let mut inside = false;
            for corner in &corners {
                let distance =
                    plane.x * corner.x + plane.y * corner.y + plane.z * corner.z + plane.w;
                if distance >= 0.0 {
                    inside = true;
                    break;
                }
            }
            if !inside {
                return false; // AABB is completely outside this plane
            }
        }

        true // AABB is at least partially inside frustum
    }

    pub fn from_view_proj(vp_matrix: &Mat4) -> Self {
        Self::from_view_proj_matrix(vp_matrix)
    }
}

/// Performance counters for batching system
#[derive(Debug, Default, Clone)]
pub struct BatchingStats {
    pub total_primitives: usize,
    pub visible_primitives: usize,
    pub culled_primitives: usize,
    pub draw_calls_before_batching: usize,
    pub draw_calls_after_batching: usize,
    pub batching_time_ms: f32,
    pub culling_time_ms: f32,
}

impl BatchingStats {
    pub fn culling_ratio(&self) -> f32 {
        if self.total_primitives == 0 {
            0.0
        } else {
            self.culled_primitives as f32 / self.total_primitives as f32
        }
    }

    pub fn batching_efficiency(&self) -> f32 {
        if self.draw_calls_before_batching == 0 {
            0.0
        } else {
            1.0 - (self.draw_calls_after_batching as f32 / self.draw_calls_before_batching as f32)
        }
    }
}

/// Batched primitive data
#[derive(Debug)]
pub struct Batch {
    pub layer: Layer,
    pub primitive_type: PrimitiveType,
    pub aabb: AABB,
    pub primitive_ids: Vec<VectorId>,
    pub vertex_count: u32,
    pub instance_count: u32,
}

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

/// Batching system for vector primitives
pub struct BatchManager {
    stats: BatchingStats,
    frustum: Option<Frustum>,
    // Primitive data with AABBs
    polygon_data: Vec<(VectorId, PolygonDef, AABB)>,
    line_data: Vec<(VectorId, PolylineDef, AABB)>,
    point_data: Vec<(VectorId, PointDef, AABB)>,
    // Batching parameters
    max_batch_size: usize,
    area_threshold: f32,
}

impl BatchManager {
    pub fn new() -> Self {
        Self {
            stats: BatchingStats::default(),
            frustum: None,
            polygon_data: Vec::new(),
            line_data: Vec::new(),
            point_data: Vec::new(),
            max_batch_size: 1000,
            area_threshold: 100.0, // Maximum area difference for batching
        }
    }

    pub fn set_view_frustum(&mut self, vp_matrix: &Mat4) {
        self.frustum = Some(Frustum::from_view_proj_matrix(vp_matrix));
    }

    pub fn clear(&mut self) {
        self.polygon_data.clear();
        self.line_data.clear();
        self.point_data.clear();
        self.stats = BatchingStats::default();
    }

    pub fn add_polygon(&mut self, id: VectorId, polygon: PolygonDef) -> Result<(), RenderError> {
        let aabb = AABB::from_points(&polygon.exterior)
            .ok_or_else(|| RenderError::Upload("Empty polygon exterior".to_string()))?;

        self.polygon_data.push((id, polygon, aabb));
        Ok(())
    }

    pub fn add_line(&mut self, id: VectorId, line: PolylineDef) -> Result<(), RenderError> {
        let aabb = AABB::from_points(&line.path)
            .ok_or_else(|| RenderError::Upload("Empty line path".to_string()))?;

        self.line_data.push((id, line, aabb));
        Ok(())
    }

    pub fn add_point(&mut self, id: VectorId, point: PointDef) -> Result<(), RenderError> {
        // Create small AABB around point based on size
        let half_size = point.style.point_size * 0.5;
        let aabb = AABB {
            min: point.position - Vec2::splat(half_size),
            max: point.position + Vec2::splat(half_size),
        };

        self.point_data.push((id, point, aabb));
        Ok(())
    }

    /// Perform visibility culling and generate batches
    pub fn generate_batches(&mut self) -> Result<Vec<Batch>, RenderError> {
        let start_time = std::time::Instant::now();

        let mut batches = Vec::new();
        self.stats.total_primitives =
            self.polygon_data.len() + self.line_data.len() + self.point_data.len();
        self.stats.draw_calls_before_batching = self.stats.total_primitives;

        // Cull and batch polygons
        let visible_polygons = self.cull_polygons();
        let polygon_batches = self.batch_polygons(visible_polygons);
        batches.extend(polygon_batches);

        // Cull and batch lines
        let visible_lines = self.cull_lines();
        let line_batches = self.batch_lines(visible_lines);
        batches.extend(line_batches);

        // Cull and batch points
        let visible_points = self.cull_points();
        let point_batches = self.batch_points(visible_points);
        batches.extend(point_batches);

        self.stats.draw_calls_after_batching = batches.len();
        self.stats.batching_time_ms = start_time.elapsed().as_secs_f32() * 1000.0;

        Ok(batches)
    }

    fn cull_polygons(&mut self) -> Vec<(VectorId, PolygonDef, AABB)> {
        let cull_start = std::time::Instant::now();
        let mut visible = Vec::new();

        for (id, polygon, aabb) in &self.polygon_data {
            if let Some(ref frustum) = self.frustum {
                if frustum.test_aabb(aabb) {
                    visible.push((*id, polygon.clone(), *aabb));
                    self.stats.visible_primitives += 1;
                } else {
                    self.stats.culled_primitives += 1;
                }
            } else {
                // No frustum culling, all visible
                visible.push((*id, polygon.clone(), *aabb));
                self.stats.visible_primitives += 1;
            }
        }

        self.stats.culling_time_ms += cull_start.elapsed().as_secs_f32() * 1000.0;
        visible
    }

    fn cull_lines(&mut self) -> Vec<(VectorId, PolylineDef, AABB)> {
        let cull_start = std::time::Instant::now();
        let mut visible = Vec::new();

        for (id, line, aabb) in &self.line_data {
            if let Some(ref frustum) = self.frustum {
                if frustum.test_aabb(aabb) {
                    visible.push((*id, line.clone(), *aabb));
                    self.stats.visible_primitives += 1;
                } else {
                    self.stats.culled_primitives += 1;
                }
            } else {
                visible.push((*id, line.clone(), *aabb));
                self.stats.visible_primitives += 1;
            }
        }

        self.stats.culling_time_ms += cull_start.elapsed().as_secs_f32() * 1000.0;
        visible
    }

    fn cull_points(&mut self) -> Vec<(VectorId, PointDef, AABB)> {
        let cull_start = std::time::Instant::now();
        let mut visible = Vec::new();

        for (id, point, aabb) in &self.point_data {
            if let Some(ref frustum) = self.frustum {
                if frustum.test_aabb(aabb) {
                    visible.push((*id, point.clone(), *aabb));
                    self.stats.visible_primitives += 1;
                } else {
                    self.stats.culled_primitives += 1;
                }
            } else {
                visible.push((*id, point.clone(), *aabb));
                self.stats.visible_primitives += 1;
            }
        }

        self.stats.culling_time_ms += cull_start.elapsed().as_secs_f32() * 1000.0;
        visible
    }

    fn batch_polygons(&self, visible: Vec<(VectorId, PolygonDef, AABB)>) -> Vec<Batch> {
        self.create_batches(
            visible
                .into_iter()
                .map(|(id, _, aabb)| (id, aabb))
                .collect(),
            PrimitiveType::Polygon,
            Layer::Background,
        )
    }

    fn batch_lines(&self, visible: Vec<(VectorId, PolylineDef, AABB)>) -> Vec<Batch> {
        self.create_batches(
            visible
                .into_iter()
                .map(|(id, _, aabb)| (id, aabb))
                .collect(),
            PrimitiveType::Line,
            Layer::Vector,
        )
    }

    fn batch_points(&self, visible: Vec<(VectorId, PointDef, AABB)>) -> Vec<Batch> {
        self.create_batches(
            visible
                .into_iter()
                .map(|(id, _, aabb)| (id, aabb))
                .collect(),
            PrimitiveType::Point,
            Layer::Points,
        )
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
                // Finish current batch
                batches.push(Batch {
                    layer,
                    primitive_type,
                    aabb: current_batch_aabb.unwrap(),
                    primitive_ids: std::mem::take(&mut current_batch_ids),
                    vertex_count: 0,   // Will be filled by renderer
                    instance_count: 0, // Will be filled by renderer
                });
                current_batch_aabb = None;
            }

            // Add to current batch
            current_batch_ids.push(id);
            current_batch_aabb = Some(match current_batch_aabb {
                Some(existing) => existing.union(&aabb),
                None => aabb,
            });
        }

        // Finish final batch
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

    pub fn get_stats(&self) -> &BatchingStats {
        &self.stats
    }

    pub fn set_max_batch_size(&mut self, size: usize) {
        self.max_batch_size = size;
    }

    pub fn set_area_threshold(&mut self, threshold: f32) {
        self.area_threshold = threshold;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Vec2;

    #[test]
    fn test_aabb_from_points() {
        let points = vec![
            Vec2::new(1.0, 2.0),
            Vec2::new(-1.0, 3.0),
            Vec2::new(2.0, -1.0),
        ];

        let aabb = AABB::from_points(&points).unwrap();
        assert_eq!(aabb.min, Vec2::new(-1.0, -1.0));
        assert_eq!(aabb.max, Vec2::new(2.0, 3.0));
        assert_eq!(aabb.center(), Vec2::new(0.5, 1.0));
        assert_eq!(aabb.size(), Vec2::new(3.0, 4.0));
        assert_eq!(aabb.area(), 12.0);
    }

    #[test]
    fn test_aabb_intersection() {
        let aabb1 = AABB::new(Vec2::new(0.0, 0.0), Vec2::new(2.0, 2.0));
        let aabb2 = AABB::new(Vec2::new(1.0, 1.0), Vec2::new(3.0, 3.0));
        let aabb3 = AABB::new(Vec2::new(3.0, 3.0), Vec2::new(4.0, 4.0));

        assert!(aabb1.intersects(&aabb2));
        assert!(aabb2.intersects(&aabb1));
        assert!(!aabb1.intersects(&aabb3));
        assert!(!aabb3.intersects(&aabb1));
    }

    #[test]
    fn test_batch_manager() {
        let mut manager = BatchManager::new();

        // Add some test primitives
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

    #[test]
    fn test_frustum_culling() {
        // Create identity matrix (no culling)
        let identity = Mat4::IDENTITY;
        let frustum = Frustum::from_view_proj_matrix(&identity);

        let aabb = AABB::new(Vec2::new(-0.5, -0.5), Vec2::new(0.5, 0.5));
        assert!(frustum.test_aabb(&aabb));
    }
}
