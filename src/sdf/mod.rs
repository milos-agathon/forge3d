// src/sdf/mod.rs
// Signed Distance Function (SDF) module for procedural geometry and CSG operations
// This module provides analytic SDF primitives and constructive solid geometry operations

pub mod hybrid;
pub mod operations;
pub mod primitives;

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

    /// Add a primitive and return a handle to it
    pub fn add_sphere(mut self, center: glam::Vec3, radius: f32, material_id: u32) -> (Self, u32) {
        let primitive = SdfPrimitive::sphere(center, radius, material_id);
        let prim_idx = self.scene.csg_tree.add_primitive(primitive);
        let node_idx = self.scene.csg_tree.add_leaf(prim_idx, material_id);
        (self, node_idx)
    }

    /// Add a box primitive
    pub fn add_box(
        mut self,
        center: glam::Vec3,
        extents: glam::Vec3,
        material_id: u32,
    ) -> (Self, u32) {
        let primitive = SdfPrimitive::box_primitive(center, extents, material_id);
        let prim_idx = self.scene.csg_tree.add_primitive(primitive);
        let node_idx = self.scene.csg_tree.add_leaf(prim_idx, material_id);
        (self, node_idx)
    }

    /// Add a cylinder primitive
    pub fn add_cylinder(
        mut self,
        center: glam::Vec3,
        radius: f32,
        height: f32,
        material_id: u32,
    ) -> (Self, u32) {
        let primitive = SdfPrimitive::cylinder(center, radius, height, material_id);
        let prim_idx = self.scene.csg_tree.add_primitive(primitive);
        let node_idx = self.scene.csg_tree.add_leaf(prim_idx, material_id);
        (self, node_idx)
    }

    /// Union two nodes
    pub fn union(mut self, left: u32, right: u32, material_id: u32) -> (Self, u32) {
        let node_idx =
            self.scene
                .csg_tree
                .add_operation(CsgOperation::Union, left, right, 0.0, material_id);
        (self, node_idx)
    }

    /// Smooth union two nodes
    pub fn smooth_union(
        mut self,
        left: u32,
        right: u32,
        smoothing: f32,
        material_id: u32,
    ) -> (Self, u32) {
        let node_idx = self.scene.csg_tree.add_operation(
            CsgOperation::SmoothUnion,
            left,
            right,
            smoothing,
            material_id,
        );
        (self, node_idx)
    }

    /// Subtract right node from left node
    pub fn subtract(mut self, left: u32, right: u32, material_id: u32) -> (Self, u32) {
        let node_idx = self.scene.csg_tree.add_operation(
            CsgOperation::Subtraction,
            left,
            right,
            0.0,
            material_id,
        );
        (self, node_idx)
    }

    /// Intersect two nodes
    pub fn intersect(mut self, left: u32, right: u32, material_id: u32) -> (Self, u32) {
        let node_idx = self.scene.csg_tree.add_operation(
            CsgOperation::Intersection,
            left,
            right,
            0.0,
            material_id,
        );
        (self, node_idx)
    }

    /// Set scene bounds
    pub fn with_bounds(mut self, min: glam::Vec3, max: glam::Vec3) -> Self {
        self.scene.bounds = Some((min, max));
        self
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
