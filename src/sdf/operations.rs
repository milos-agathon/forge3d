// src/sdf/operations.rs
// Constructive Solid Geometry (CSG) operations for SDF primitives
// Implements union, intersection, subtraction, and smooth variants

use bytemuck::{Pod, Zeroable};

/// CSG operation types
#[repr(u32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CsgOperation {
    Union = 0,
    Intersection = 1,
    Subtraction = 2,
    SmoothUnion = 3,
    SmoothIntersection = 4,
    SmoothSubtraction = 5,
}

/// Parameters for CSG operations
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct CsgNode {
    /// Operation type
    pub operation: u32,
    /// Left child index (or primitive index if leaf)
    pub left_child: u32,
    /// Right child index (unused for leaf nodes)
    pub right_child: u32,
    /// Smoothing parameter for smooth operations
    pub smoothing: f32,
    /// Material ID for this node
    pub material_id: u32,
    /// Whether this is a leaf node (contains primitive)
    pub is_leaf: u32,
    /// Padding for alignment
    pub _pad: [u32; 2],
}

impl CsgNode {
    /// Create a leaf node containing a primitive
    pub fn leaf(primitive_index: u32, material_id: u32) -> Self {
        Self {
            operation: 0, // Unused for leaf
            left_child: primitive_index,
            right_child: 0, // Unused for leaf
            smoothing: 0.0,
            material_id,
            is_leaf: 1,
            _pad: [0; 2],
        }
    }

    /// Create an operation node
    pub fn operation(
        operation: CsgOperation,
        left_child: u32,
        right_child: u32,
        smoothing: f32,
        material_id: u32,
    ) -> Self {
        Self {
            operation: operation as u32,
            left_child,
            right_child,
            smoothing,
            material_id,
            is_leaf: 0,
            _pad: [0; 2],
        }
    }
}

/// CSG evaluation result including distance and material
#[derive(Clone, Copy, Debug)]
pub struct CsgResult {
    /// Signed distance
    pub distance: f32,
    /// Material ID
    pub material_id: u32,
}

/// CPU-side CSG evaluation functions
pub mod cpu_eval {
    use super::*;

    /// Smooth minimum function for smooth CSG operations
    fn smooth_min(a: f32, b: f32, k: f32) -> f32 {
        if k <= 0.0 {
            return a.min(b);
        }
        let h = (0.5 + 0.5 * (b - a) / k).clamp(0.0, 1.0);
        // Linear interpolation without unstable f32::lerp
        (1.0 - h) * b + h * a - k * h * (1.0 - h)
    }

    /// Smooth maximum function for smooth CSG operations
    fn smooth_max(a: f32, b: f32, k: f32) -> f32 {
        -smooth_min(-a, -b, k)
    }

    /// Union of two SDFs (minimum distance)
    pub fn union(a: CsgResult, b: CsgResult) -> CsgResult {
        if a.distance <= b.distance {
            a
        } else {
            b
        }
    }

    /// Intersection of two SDFs (maximum distance)
    pub fn intersection(a: CsgResult, b: CsgResult) -> CsgResult {
        if a.distance >= b.distance {
            a
        } else {
            b
        }
    }

    /// Subtraction of SDF b from SDF a
    pub fn subtraction(a: CsgResult, b: CsgResult) -> CsgResult {
        let neg_b = CsgResult {
            distance: -b.distance,
            material_id: b.material_id,
        };
        intersection(a, neg_b)
    }

    /// Smooth union of two SDFs
    pub fn smooth_union(a: CsgResult, b: CsgResult, k: f32) -> CsgResult {
        let distance = smooth_min(a.distance, b.distance, k);
        // Blend materials based on contribution
        let t = if a.distance + b.distance == 0.0 {
            0.5
        } else {
            (b.distance) / (a.distance + b.distance)
        };

        let material_id = if t < 0.5 { a.material_id } else { b.material_id };

        CsgResult {
            distance,
            material_id,
        }
    }

    /// Smooth intersection of two SDFs
    pub fn smooth_intersection(a: CsgResult, b: CsgResult, k: f32) -> CsgResult {
        let distance = smooth_max(a.distance, b.distance, k);
        let material_id = if a.distance >= b.distance {
            a.material_id
        } else {
            b.material_id
        };

        CsgResult {
            distance,
            material_id,
        }
    }

    /// Smooth subtraction of SDF b from SDF a
    pub fn smooth_subtraction(a: CsgResult, b: CsgResult, k: f32) -> CsgResult {
        let neg_b = CsgResult {
            distance: -b.distance,
            material_id: b.material_id,
        };
        smooth_intersection(a, neg_b, k)
    }

    /// Apply CSG operation to two results
    pub fn apply_operation(
        operation: CsgOperation,
        a: CsgResult,
        b: CsgResult,
        smoothing: f32,
    ) -> CsgResult {
        match operation {
            CsgOperation::Union => union(a, b),
            CsgOperation::Intersection => intersection(a, b),
            CsgOperation::Subtraction => subtraction(a, b),
            CsgOperation::SmoothUnion => smooth_union(a, b, smoothing),
            CsgOperation::SmoothIntersection => smooth_intersection(a, b, smoothing),
            CsgOperation::SmoothSubtraction => smooth_subtraction(a, b, smoothing),
        }
    }
}

/// CSG tree for organizing multiple primitives and operations
#[derive(Clone, Debug)]
pub struct CsgTree {
    /// CSG nodes (operations and leaves)
    pub nodes: Vec<CsgNode>,
    /// Primitives referenced by leaf nodes
    pub primitives: Vec<crate::sdf::primitives::SdfPrimitive>,
}

impl CsgTree {
    /// Create a new empty CSG tree
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            primitives: Vec::new(),
        }
    }

    /// Add a primitive and return its index
    pub fn add_primitive(&mut self, primitive: crate::sdf::primitives::SdfPrimitive) -> u32 {
        let index = self.primitives.len() as u32;
        self.primitives.push(primitive);
        index
    }

    /// Add a leaf node referencing a primitive
    pub fn add_leaf(&mut self, primitive_index: u32, material_id: u32) -> u32 {
        let node_index = self.nodes.len() as u32;
        self.nodes.push(CsgNode::leaf(primitive_index, material_id));
        node_index
    }

    /// Add an operation node
    pub fn add_operation(
        &mut self,
        operation: CsgOperation,
        left_child: u32,
        right_child: u32,
        smoothing: f32,
        material_id: u32,
    ) -> u32 {
        let node_index = self.nodes.len() as u32;
        self.nodes.push(CsgNode::operation(
            operation,
            left_child,
            right_child,
            smoothing,
            material_id,
        ));
        node_index
    }

    /// Evaluate the CSG tree at a point (CPU implementation)
    pub fn evaluate(&self, point: glam::Vec3, root_node: u32) -> CsgResult {
        if root_node as usize >= self.nodes.len() {
            return CsgResult {
                distance: f32::INFINITY,
                material_id: 0,
            };
        }

        let node = &self.nodes[root_node as usize];

        if node.is_leaf != 0 {
            // Leaf node: evaluate primitive
            if node.left_child as usize >= self.primitives.len() {
                return CsgResult {
                    distance: f32::INFINITY,
                    material_id: node.material_id,
                };
            }

            let primitive = &self.primitives[node.left_child as usize];
            let distance = crate::sdf::primitives::cpu_eval::evaluate_primitive(point, primitive);

            CsgResult {
                distance,
                material_id: node.material_id,
            }
        } else {
            // Operation node: evaluate children and apply operation
            let left_result = self.evaluate(point, node.left_child);
            let right_result = self.evaluate(point, node.right_child);

            let operation = match node.operation {
                0 => CsgOperation::Union,
                1 => CsgOperation::Intersection,
                2 => CsgOperation::Subtraction,
                3 => CsgOperation::SmoothUnion,
                4 => CsgOperation::SmoothIntersection,
                5 => CsgOperation::SmoothSubtraction,
                _ => CsgOperation::Union, // Fallback
            };

            cpu_eval::apply_operation(operation, left_result, right_result, node.smoothing)
        }
    }

    /// Get the root node index (assumes last added node is root)
    pub fn root_node(&self) -> Option<u32> {
        if self.nodes.is_empty() {
            None
        } else {
            Some((self.nodes.len() - 1) as u32)
        }
    }
}

impl Default for CsgTree {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sdf::primitives::SdfPrimitive;
    use glam::Vec3;

    #[test]
    fn test_csg_union() {
        let a = CsgResult {
            distance: 0.5,
            material_id: 1,
        };
        let b = CsgResult {
            distance: 1.0,
            material_id: 2,
        };

        let result = cpu_eval::union(a, b);
        assert_eq!(result.distance, 0.5);
        assert_eq!(result.material_id, 1);
    }

    #[test]
    fn test_csg_intersection() {
        let a = CsgResult {
            distance: 0.5,
            material_id: 1,
        };
        let b = CsgResult {
            distance: 1.0,
            material_id: 2,
        };

        let result = cpu_eval::intersection(a, b);
        assert_eq!(result.distance, 1.0);
        assert_eq!(result.material_id, 2);
    }

    #[test]
    fn test_csg_subtraction() {
        let a = CsgResult {
            distance: 0.5,
            material_id: 1,
        };
        let b = CsgResult {
            distance: 1.0,
            material_id: 2,
        };

        let result = cpu_eval::subtraction(a, b);
        // Subtraction: max(a, -b) = max(0.5, -1.0) = 0.5
        assert_eq!(result.distance, 0.5);
        assert_eq!(result.material_id, 1);
    }

    #[test]
    fn test_smooth_union() {
        let a = CsgResult {
            distance: 0.5,
            material_id: 1,
        };
        let b = CsgResult {
            distance: 1.0,
            material_id: 2,
        };

        let result = cpu_eval::smooth_union(a, b, 0.1);
        // Should be between min and smooth blend
        assert!(result.distance <= 0.5);
        assert!(result.distance >= 0.4); // Approximate smooth result
    }

    #[test]
    fn test_csg_tree() {
        let mut tree = CsgTree::new();

        // Add two sphere primitives
        let sphere1 = SdfPrimitive::sphere(Vec3::new(-0.5, 0.0, 0.0), 1.0, 1);
        let sphere2 = SdfPrimitive::sphere(Vec3::new(0.5, 0.0, 0.0), 1.0, 2);

        let prim1_idx = tree.add_primitive(sphere1);
        let prim2_idx = tree.add_primitive(sphere2);

        // Add leaf nodes
        let leaf1_idx = tree.add_leaf(prim1_idx, 1);
        let leaf2_idx = tree.add_leaf(prim2_idx, 2);

        // Add union operation
        let union_idx = tree.add_operation(
            CsgOperation::Union,
            leaf1_idx,
            leaf2_idx,
            0.0,
            0,
        );

        // Evaluate at origin (should be inside both spheres, union should be negative)
        let result = tree.evaluate(Vec3::ZERO, union_idx);
        assert!(result.distance < 0.0);
    }

    #[test]
    fn test_csg_node_creation() {
        let leaf = CsgNode::leaf(5, 42);
        assert_eq!(leaf.left_child, 5);
        assert_eq!(leaf.material_id, 42);
        assert_eq!(leaf.is_leaf, 1);

        let op = CsgNode::operation(CsgOperation::Union, 1, 2, 0.1, 3);
        assert_eq!(op.operation, CsgOperation::Union as u32);
        assert_eq!(op.left_child, 1);
        assert_eq!(op.right_child, 2);
        assert_eq!(op.smoothing, 0.1);
        assert_eq!(op.material_id, 3);
        assert_eq!(op.is_leaf, 0);
    }
}
