// src/sdf/mod.rs
// Signed Distance Function (SDF) module for procedural geometry and CSG operations
// This module provides analytic SDF primitives and constructive solid geometry operations

pub mod hybrid;
pub mod operations;
pub mod primitives;

#[cfg(feature = "extension-module")]
pub mod py;

// Re-export commonly used types
pub use primitives::{
    SdfBox, SdfCapsule, SdfCylinder, SdfPlane, SdfPrimitive, SdfPrimitiveType, SdfSphere, SdfTorus,
};

pub use operations::{CsgNode, CsgOperation, CsgResult, CsgTree};

pub use hybrid::{HybridHitResult, HybridMetrics, HybridScene, Ray as HybridRay};

/// SDF scene containing primitives and CSG tree
#[derive(Clone, Debug)]
pub struct SdfScene {
    /// CSG tree defining the scene hierarchy
    pub csg_tree: CsgTree,
    /// Bounding box for the entire scene (for optimization)
    pub bounds: Option<(glam::Vec3, glam::Vec3)>, // (min, max)
}

impl SdfScene {
    /// Create a new empty SDF scene
    pub fn new() -> Self {
        Self {
            csg_tree: CsgTree::new(),
            bounds: None,
        }
    }

    /// Add a single primitive as the root
    pub fn single_primitive(primitive: SdfPrimitive) -> Self {
        let mut scene = Self::new();
        let prim_idx = scene.csg_tree.add_primitive(primitive);
        scene.csg_tree.add_leaf(prim_idx, primitive.material_id);
        scene
    }

    /// Set scene bounds for optimization
    pub fn with_bounds(mut self, min: glam::Vec3, max: glam::Vec3) -> Self {
        self.bounds = Some((min, max));
        self
    }

    /// Evaluate the scene at a point
    pub fn evaluate(&self, point: glam::Vec3) -> CsgResult {
        if let Some(root) = self.csg_tree.root_node() {
            self.csg_tree.evaluate(point, root)
        } else {
            CsgResult {
                distance: f32::INFINITY,
                material_id: 0,
            }
        }
    }

    /// Check if a point is inside the scene bounds
    pub fn in_bounds(&self, point: glam::Vec3) -> bool {
        if let Some((min_bounds, max_bounds)) = &self.bounds {
            point.x >= min_bounds.x
                && point.x <= max_bounds.x
                && point.y >= min_bounds.y
                && point.y <= max_bounds.y
                && point.z >= min_bounds.z
                && point.z <= max_bounds.z
        } else {
            true // No bounds set, assume infinite
        }
    }

    /// Explicitly set scene bounds
    pub fn set_bounds(&mut self, min: glam::Vec3, max: glam::Vec3) {
        self.bounds = Some((min, max));
    }

    /// Get the number of primitives in the scene
    pub fn primitive_count(&self) -> usize {
        self.csg_tree.primitives.len()
    }

    /// Get the number of CSG nodes in the scene
    pub fn node_count(&self) -> usize {
        self.csg_tree.nodes.len()
    }
}

impl Default for SdfScene {
    fn default() -> Self {
        Self::new()
    }
}

/// Builder pattern for constructing complex SDF scenes
pub struct SdfSceneBuilder {
    scene: SdfScene,
}

impl SdfSceneBuilder {
    /// Create a new scene builder
    pub fn new() -> Self {
        Self {
            scene: SdfScene::new(),
        }
    }

    fn add_primitive_internal(&mut self, primitive: SdfPrimitive, material_id: u32) -> u32 {
        let prim_idx = self.scene.csg_tree.add_primitive(primitive);
        self.scene.csg_tree.add_leaf(prim_idx, material_id)
    }

    fn add_operation_internal(
        &mut self,
        operation: CsgOperation,
        left: u32,
        right: u32,
        smoothing: f32,
        material_id: u32,
    ) -> u32 {
        self.scene
            .csg_tree
            .add_operation(operation, left, right, smoothing, material_id)
    }

    /// In-place sphere primitive helper
    pub fn add_sphere_mut(&mut self, center: glam::Vec3, radius: f32, material_id: u32) -> u32 {
        let primitive = SdfPrimitive::sphere(center, radius, material_id);
        self.add_primitive_internal(primitive, material_id)
    }

    /// Add a primitive and return a handle to it
    pub fn add_sphere(mut self, center: glam::Vec3, radius: f32, material_id: u32) -> (Self, u32) {
        let node_idx = self.add_sphere_mut(center, radius, material_id);
        (self, node_idx)
    }

    /// Add a box primitive
    pub fn add_box(
        mut self,
        center: glam::Vec3,
        extents: glam::Vec3,
        material_id: u32,
    ) -> (Self, u32) {
        let node_idx = self.add_box_mut(center, extents, material_id);
        (self, node_idx)
    }

    /// In-place box primitive helper
    pub fn add_box_mut(
        &mut self,
        center: glam::Vec3,
        extents: glam::Vec3,
        material_id: u32,
    ) -> u32 {
        let primitive = SdfPrimitive::box_primitive(center, extents, material_id);
        self.add_primitive_internal(primitive, material_id)
    }

    /// Add a cylinder primitive
    pub fn add_cylinder(
        mut self,
        center: glam::Vec3,
        radius: f32,
        height: f32,
        material_id: u32,
    ) -> (Self, u32) {
        let node_idx = self.add_cylinder_mut(center, radius, height, material_id);
        (self, node_idx)
    }

    /// In-place cylinder primitive helper
    pub fn add_cylinder_mut(
        &mut self,
        center: glam::Vec3,
        radius: f32,
        height: f32,
        material_id: u32,
    ) -> u32 {
        let primitive = SdfPrimitive::cylinder(center, radius, height, material_id);
        self.add_primitive_internal(primitive, material_id)
    }

    /// Add a plane primitive
    pub fn add_plane(mut self, normal: glam::Vec3, distance: f32, material_id: u32) -> (Self, u32) {
        let node_idx = self.add_plane_mut(normal, distance, material_id);
        (self, node_idx)
    }

    /// In-place plane primitive helper
    pub fn add_plane_mut(&mut self, normal: glam::Vec3, distance: f32, material_id: u32) -> u32 {
        let primitive = SdfPrimitive::plane(normal, distance, material_id);
        self.add_primitive_internal(primitive, material_id)
    }

    /// Add a torus primitive
    pub fn add_torus(
        mut self,
        center: glam::Vec3,
        major_radius: f32,
        minor_radius: f32,
        material_id: u32,
    ) -> (Self, u32) {
        let node_idx = self.add_torus_mut(center, major_radius, minor_radius, material_id);
        (self, node_idx)
    }

    /// In-place torus primitive helper
    pub fn add_torus_mut(
        &mut self,
        center: glam::Vec3,
        major_radius: f32,
        minor_radius: f32,
        material_id: u32,
    ) -> u32 {
        let primitive = SdfPrimitive::torus(center, major_radius, minor_radius, material_id);
        self.add_primitive_internal(primitive, material_id)
    }

    /// Add a capsule primitive
    pub fn add_capsule(
        mut self,
        point_a: glam::Vec3,
        point_b: glam::Vec3,
        radius: f32,
        material_id: u32,
    ) -> (Self, u32) {
        let node_idx = self.add_capsule_mut(point_a, point_b, radius, material_id);
        (self, node_idx)
    }

    /// In-place capsule primitive helper
    pub fn add_capsule_mut(
        &mut self,
        point_a: glam::Vec3,
        point_b: glam::Vec3,
        radius: f32,
        material_id: u32,
    ) -> u32 {
        let primitive = SdfPrimitive::capsule(point_a, point_b, radius, material_id);
        self.add_primitive_internal(primitive, material_id)
    }

    /// Union two nodes
    pub fn union(mut self, left: u32, right: u32, material_id: u32) -> (Self, u32) {
        let node_idx = self.union_mut(left, right, material_id);
        (self, node_idx)
    }

    /// In-place union operation helper
    pub fn union_mut(&mut self, left: u32, right: u32, material_id: u32) -> u32 {
        self.add_operation_internal(CsgOperation::Union, left, right, 0.0, material_id)
    }

    /// Smooth union two nodes
    pub fn smooth_union(
        mut self,
        left: u32,
        right: u32,
        smoothing: f32,
        material_id: u32,
    ) -> (Self, u32) {
        let node_idx = self.smooth_union_mut(left, right, smoothing, material_id);
        (self, node_idx)
    }

    /// In-place smooth union helper
    pub fn smooth_union_mut(
        &mut self,
        left: u32,
        right: u32,
        smoothing: f32,
        material_id: u32,
    ) -> u32 {
        self.add_operation_internal(
            CsgOperation::SmoothUnion,
            left,
            right,
            smoothing,
            material_id,
        )
    }

    /// Subtract right node from left node
    pub fn subtract(mut self, left: u32, right: u32, material_id: u32) -> (Self, u32) {
        let node_idx = self.subtract_mut(left, right, material_id);
        (self, node_idx)
    }

    /// In-place subtraction helper
    pub fn subtract_mut(&mut self, left: u32, right: u32, material_id: u32) -> u32 {
        self.add_operation_internal(CsgOperation::Subtraction, left, right, 0.0, material_id)
    }

    /// Intersect two nodes
    pub fn intersect(mut self, left: u32, right: u32, material_id: u32) -> (Self, u32) {
        let node_idx = self.intersect_mut(left, right, material_id);
        (self, node_idx)
    }

    /// In-place intersection helper
    pub fn intersect_mut(&mut self, left: u32, right: u32, material_id: u32) -> u32 {
        self.add_operation_internal(CsgOperation::Intersection, left, right, 0.0, material_id)
    }

    /// Smooth intersection of two nodes
    pub fn smooth_intersection(
        mut self,
        left: u32,
        right: u32,
        smoothing: f32,
        material_id: u32,
    ) -> (Self, u32) {
        let node_idx = self.smooth_intersection_mut(left, right, smoothing, material_id);
        (self, node_idx)
    }

    /// In-place smooth intersection helper
    pub fn smooth_intersection_mut(
        &mut self,
        left: u32,
        right: u32,
        smoothing: f32,
        material_id: u32,
    ) -> u32 {
        self.add_operation_internal(
            CsgOperation::SmoothIntersection,
            left,
            right,
            smoothing,
            material_id,
        )
    }

    /// Smooth subtraction helper (left - right with smoothing)
    pub fn smooth_subtraction(
        mut self,
        left: u32,
        right: u32,
        smoothing: f32,
        material_id: u32,
    ) -> (Self, u32) {
        let node_idx = self.smooth_subtraction_mut(left, right, smoothing, material_id);
        (self, node_idx)
    }

    /// In-place smooth subtraction helper
    pub fn smooth_subtraction_mut(
        &mut self,
        left: u32,
        right: u32,
        smoothing: f32,
        material_id: u32,
    ) -> u32 {
        self.add_operation_internal(
            CsgOperation::SmoothSubtraction,
            left,
            right,
            smoothing,
            material_id,
        )
    }

    /// Set scene bounds
    pub fn with_bounds(mut self, min: glam::Vec3, max: glam::Vec3) -> Self {
        self.with_bounds_mut(min, max);
        self
    }

    /// In-place scene bounds setter
    pub fn with_bounds_mut(&mut self, min: glam::Vec3, max: glam::Vec3) {
        self.scene.set_bounds(min, max);
    }

    /// Reset builder to an empty scene
    pub fn reset(&mut self) {
        self.scene = SdfScene::new();
    }

    /// Build the final scene
    pub fn build(self) -> SdfScene {
        self.scene
    }
}

impl Default for SdfSceneBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Vec3;

    #[test]
    fn test_single_primitive_scene() {
        let sphere = SdfPrimitive::sphere(Vec3::ZERO, 1.0, 1);
        let scene = SdfScene::single_primitive(sphere);

        let result = scene.evaluate(Vec3::ZERO);
        assert!(result.distance < 0.0); // Inside sphere
        assert_eq!(result.material_id, 1);

        let result = scene.evaluate(Vec3::new(2.0, 0.0, 0.0));
        assert!(result.distance > 0.0); // Outside sphere
    }

    #[test]
    fn test_scene_builder() {
        let (builder, sphere1) =
            SdfSceneBuilder::new().add_sphere(Vec3::new(-1.0, 0.0, 0.0), 0.8, 1);

        let (builder, sphere2) = builder.add_sphere(Vec3::new(1.0, 0.0, 0.0), 0.8, 2);

        let (builder, _union_node) = builder.union(sphere1, sphere2, 0);

        let scene = builder.build();

        // Test that we have the expected number of primitives and nodes
        assert_eq!(scene.primitive_count(), 2);
        assert_eq!(scene.node_count(), 3); // 2 leaves + 1 union

        // Test evaluation
        let result = scene.evaluate(Vec3::ZERO);
        // Should be inside the union of two spheres
        assert!(result.distance < 0.0);
    }

    #[test]
    fn test_scene_bounds() {
        let sphere = SdfPrimitive::sphere(Vec3::ZERO, 1.0, 1);
        let scene = SdfScene::single_primitive(sphere)
            .with_bounds(Vec3::new(-2.0, -2.0, -2.0), Vec3::new(2.0, 2.0, 2.0));

        assert!(scene.in_bounds(Vec3::ZERO));
        assert!(scene.in_bounds(Vec3::new(1.5, 1.5, 1.5)));
        assert!(!scene.in_bounds(Vec3::new(3.0, 0.0, 0.0)));
    }

    #[test]
    fn test_complex_csg() {
        let (builder, box1) =
            SdfSceneBuilder::new().add_box(Vec3::ZERO, Vec3::new(1.0, 1.0, 1.0), 1);

        let (builder, sphere1) = builder.add_sphere(Vec3::ZERO, 1.2, 2);

        let (builder, _result) = builder.subtract(box1, sphere1, 0); // Box with sphere subtracted

        let scene = builder.build();

        // The result should be a hollow box
        // Point at origin should be outside (positive distance) since sphere is subtracted
        let result = scene.evaluate(Vec3::ZERO);
        assert!(result.distance > 0.0);
    }
}
