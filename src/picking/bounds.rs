// src/picking/bounds.rs
// AABB bounds testing for layer-based picking
// Part of Plan 2: Standard - GPU Ray Picking + Hover Support

use super::ray::Ray;

/// Axis-Aligned Bounding Box
#[derive(Debug, Clone, Copy, Default)]
pub struct AABB {
    pub min: [f32; 3],
    pub max: [f32; 3],
}

impl AABB {
    /// Create a new AABB from min and max points
    pub fn new(min: [f32; 3], max: [f32; 3]) -> Self {
        Self { min, max }
    }

    /// Create an empty (invalid) AABB
    pub fn empty() -> Self {
        Self {
            min: [f32::MAX, f32::MAX, f32::MAX],
            max: [f32::MIN, f32::MIN, f32::MIN],
        }
    }

    /// Check if AABB is valid (non-empty)
    pub fn is_valid(&self) -> bool {
        self.min[0] <= self.max[0] && self.min[1] <= self.max[1] && self.min[2] <= self.max[2]
    }

    /// Expand AABB to include a point
    pub fn expand_point(&mut self, point: [f32; 3]) {
        for i in 0..3 {
            self.min[i] = self.min[i].min(point[i]);
            self.max[i] = self.max[i].max(point[i]);
        }
    }

    /// Expand AABB to include another AABB
    pub fn expand_aabb(&mut self, other: &AABB) {
        if !other.is_valid() {
            return;
        }
        for i in 0..3 {
            self.min[i] = self.min[i].min(other.min[i]);
            self.max[i] = self.max[i].max(other.max[i]);
        }
    }

    /// Get center of AABB
    pub fn center(&self) -> [f32; 3] {
        [
            (self.min[0] + self.max[0]) * 0.5,
            (self.min[1] + self.max[1]) * 0.5,
            (self.min[2] + self.max[2]) * 0.5,
        ]
    }

    /// Get extents (half-widths) of AABB
    pub fn extents(&self) -> [f32; 3] {
        [
            (self.max[0] - self.min[0]) * 0.5,
            (self.max[1] - self.min[1]) * 0.5,
            (self.max[2] - self.min[2]) * 0.5,
        ]
    }

    /// Test ray-AABB intersection using slab method
    /// Returns (hit, t_near, t_far) where t values are ray parameters
    pub fn ray_intersect(&self, ray: &Ray) -> (bool, f32, f32) {
        let mut t_near = f32::MIN;
        let mut t_far = f32::MAX;

        for i in 0..3 {
            if ray.direction[i].abs() < 1e-10 {
                // Ray parallel to slab
                if ray.origin[i] < self.min[i] || ray.origin[i] > self.max[i] {
                    return (false, 0.0, 0.0);
                }
            } else {
                let inv_d = 1.0 / ray.direction[i];
                let mut t1 = (self.min[i] - ray.origin[i]) * inv_d;
                let mut t2 = (self.max[i] - ray.origin[i]) * inv_d;

                if t1 > t2 {
                    std::mem::swap(&mut t1, &mut t2);
                }

                t_near = t_near.max(t1);
                t_far = t_far.min(t2);

                if t_near > t_far || t_far < 0.0 {
                    return (false, 0.0, 0.0);
                }
            }
        }

        (true, t_near.max(0.0), t_far)
    }

    /// Check if point is inside AABB
    pub fn contains_point(&self, point: [f32; 3]) -> bool {
        point[0] >= self.min[0] && point[0] <= self.max[0]
            && point[1] >= self.min[1] && point[1] <= self.max[1]
            && point[2] >= self.min[2] && point[2] <= self.max[2]
    }

    /// Check if two AABBs overlap
    pub fn intersects(&self, other: &AABB) -> bool {
        self.min[0] <= other.max[0] && self.max[0] >= other.min[0]
            && self.min[1] <= other.max[1] && self.max[1] >= other.min[1]
            && self.min[2] <= other.max[2] && self.max[2] >= other.min[2]
    }
}

/// Bounds information for a pickable layer
#[derive(Debug, Clone)]
pub struct LayerBounds {
    /// Layer identifier
    pub layer_id: u32,
    /// Layer name
    pub name: String,
    /// World-space bounding box
    pub aabb: AABB,
    /// Whether the layer is visible
    pub visible: bool,
    /// Whether the layer is pickable
    pub pickable: bool,
}

impl LayerBounds {
    /// Create new layer bounds
    pub fn new(layer_id: u32, name: String) -> Self {
        Self {
            layer_id,
            name,
            aabb: AABB::empty(),
            visible: true,
            pickable: true,
        }
    }

    /// Update bounds from vertex positions
    pub fn update_from_vertices(&mut self, positions: &[[f32; 3]]) {
        self.aabb = AABB::empty();
        for pos in positions {
            self.aabb.expand_point(*pos);
        }
    }
}

/// Manager for layer bounds
#[derive(Debug, Default)]
pub struct BoundsManager {
    layers: Vec<LayerBounds>,
}

impl BoundsManager {
    /// Create new bounds manager
    pub fn new() -> Self {
        Self { layers: Vec::new() }
    }

    /// Add or update layer bounds
    pub fn set_layer_bounds(&mut self, bounds: LayerBounds) {
        if let Some(existing) = self.layers.iter_mut().find(|l| l.layer_id == bounds.layer_id) {
            *existing = bounds;
        } else {
            self.layers.push(bounds);
        }
    }

    /// Remove layer bounds
    pub fn remove_layer(&mut self, layer_id: u32) {
        self.layers.retain(|l| l.layer_id != layer_id);
    }

    /// Get all visible and pickable layers that intersect with a ray
    /// Returns layers sorted by distance (nearest first)
    pub fn get_candidate_layers(&self, ray: &Ray) -> Vec<(u32, f32)> {
        let mut candidates: Vec<(u32, f32)> = self
            .layers
            .iter()
            .filter(|l| l.visible && l.pickable && l.aabb.is_valid())
            .filter_map(|l| {
                let (hit, t_near, _) = l.aabb.ray_intersect(ray);
                if hit {
                    Some((l.layer_id, t_near))
                } else {
                    None
                }
            })
            .collect();

        candidates.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        candidates
    }

    /// Get layer bounds by ID
    pub fn get_layer(&self, layer_id: u32) -> Option<&LayerBounds> {
        self.layers.iter().find(|l| l.layer_id == layer_id)
    }

    /// Get all layer bounds
    pub fn all_layers(&self) -> &[LayerBounds] {
        &self.layers
    }

    /// Clear all bounds
    pub fn clear(&mut self) {
        self.layers.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_aabb_ray_intersect() {
        let aabb = AABB::new([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
        
        // Ray hitting the box
        let ray = Ray::new([-1.0, 0.5, 0.5], [1.0, 0.0, 0.0]);
        let (hit, t_near, _) = aabb.ray_intersect(&ray);
        assert!(hit);
        assert!((t_near - 1.0).abs() < 1e-6);

        // Ray missing the box
        let ray = Ray::new([2.0, 2.0, 0.5], [1.0, 0.0, 0.0]);
        let (hit, _, _) = aabb.ray_intersect(&ray);
        assert!(!hit);
    }

    #[test]
    fn test_aabb_contains_point() {
        let aabb = AABB::new([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
        assert!(aabb.contains_point([0.5, 0.5, 0.5]));
        assert!(!aabb.contains_point([2.0, 0.5, 0.5]));
    }
}
