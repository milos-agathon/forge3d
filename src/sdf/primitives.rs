// src/sdf/primitives.rs
// Analytic Signed Distance Function (SDF) primitives for procedural geometry
// Implementation supports common SDF shapes for CSG operations and raymarching

use bytemuck::{Pod, Zeroable};
use glam::{Vec2, Vec3};

/// Parameters for SDF sphere primitive
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct SdfSphere {
    /// Center position
    pub center: [f32; 3],
    /// Radius
    pub radius: f32,
}

/// Parameters for SDF box primitive
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct SdfBox {
    /// Center position
    pub center: [f32; 3],
    /// Padding for alignment
    pub _pad1: f32,
    /// Half-extents (size/2) in each dimension
    pub extents: [f32; 3],
    /// Padding for alignment
    pub _pad2: f32,
}

/// Parameters for SDF cylinder primitive
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct SdfCylinder {
    /// Center position
    pub center: [f32; 3],
    /// Radius
    pub radius: f32,
    /// Height (total height, not half-height)
    pub height: f32,
    /// Padding for alignment
    pub _pad: [f32; 3],
}

/// Parameters for SDF plane primitive
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct SdfPlane {
    /// Plane normal (should be normalized)
    pub normal: [f32; 3],
    /// Distance from origin along normal
    pub distance: f32,
}

/// Parameters for SDF torus primitive
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct SdfTorus {
    /// Center position
    pub center: [f32; 3],
    /// Major radius (distance from center to tube center)
    pub major_radius: f32,
    /// Minor radius (tube thickness)
    pub minor_radius: f32,
    /// Padding for alignment
    pub _pad: [f32; 3],
}

/// Parameters for SDF capsule primitive
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct SdfCapsule {
    /// Start point
    pub point_a: [f32; 3],
    /// Radius
    pub radius: f32,
    /// End point
    pub point_b: [f32; 3],
    /// Padding for alignment
    pub _pad: f32,
}

/// Enumeration of SDF primitive types
#[repr(u32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SdfPrimitiveType {
    Sphere = 0,
    Box = 1,
    Cylinder = 2,
    Plane = 3,
    Torus = 4,
    Capsule = 5,
}

/// Generic SDF primitive combining type and parameters
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct SdfPrimitive {
    /// Primitive type
    pub primitive_type: u32,
    /// Material ID for shading
    pub material_id: u32,
    /// Padding for alignment
    pub _pad: [u32; 2],
    /// Primitive parameters (interpretation depends on type)
    pub params: [f32; 16], // Enough space for any primitive
}

impl SdfPrimitive {
    /// Create a sphere primitive
    pub fn sphere(center: Vec3, radius: f32, material_id: u32) -> Self {
        let sphere = SdfSphere {
            center: center.into(),
            radius,
        };

        let mut params = [0.0f32; 16];
        params[0..4].copy_from_slice(bytemuck::cast_slice(&[sphere]));

        Self {
            primitive_type: SdfPrimitiveType::Sphere as u32,
            material_id,
            _pad: [0, 0],
            params,
        }
    }

    /// Create a box primitive
    pub fn box_primitive(center: Vec3, extents: Vec3, material_id: u32) -> Self {
        let sdf_box = SdfBox {
            center: center.into(),
            _pad1: 0.0,
            extents: extents.into(),
            _pad2: 0.0,
        };

        let mut params = [0.0f32; 16];
        params[0..8].copy_from_slice(bytemuck::cast_slice(&[sdf_box]));

        Self {
            primitive_type: SdfPrimitiveType::Box as u32,
            material_id,
            _pad: [0, 0],
            params,
        }
    }

    /// Create a cylinder primitive
    pub fn cylinder(center: Vec3, radius: f32, height: f32, material_id: u32) -> Self {
        let cylinder = SdfCylinder {
            center: center.into(),
            radius,
            height,
            _pad: [0.0; 3],
        };

        let mut params = [0.0f32; 16];
        params[0..8].copy_from_slice(bytemuck::cast_slice(&[cylinder]));

        Self {
            primitive_type: SdfPrimitiveType::Cylinder as u32,
            material_id,
            _pad: [0, 0],
            params,
        }
    }

    /// Create a plane primitive
    pub fn plane(normal: Vec3, distance: f32, material_id: u32) -> Self {
        let plane = SdfPlane {
            normal: normal.normalize().into(),
            distance,
        };

        let mut params = [0.0f32; 16];
        params[0..4].copy_from_slice(bytemuck::cast_slice(&[plane]));

        Self {
            primitive_type: SdfPrimitiveType::Plane as u32,
            material_id,
            _pad: [0, 0],
            params,
        }
    }

    /// Create a torus primitive
    pub fn torus(center: Vec3, major_radius: f32, minor_radius: f32, material_id: u32) -> Self {
        let torus = SdfTorus {
            center: center.into(),
            major_radius,
            minor_radius,
            _pad: [0.0; 3],
        };

        let mut params = [0.0f32; 16];
        params[0..8].copy_from_slice(bytemuck::cast_slice(&[torus]));

        Self {
            primitive_type: SdfPrimitiveType::Torus as u32,
            material_id,
            _pad: [0, 0],
            params,
        }
    }

    /// Create a capsule primitive
    pub fn capsule(point_a: Vec3, point_b: Vec3, radius: f32, material_id: u32) -> Self {
        let capsule = SdfCapsule {
            point_a: point_a.into(),
            radius,
            point_b: point_b.into(),
            _pad: 0.0,
        };

        let mut params = [0.0f32; 16];
        params[0..8].copy_from_slice(bytemuck::cast_slice(&[capsule]));

        Self {
            primitive_type: SdfPrimitiveType::Capsule as u32,
            material_id,
            _pad: [0, 0],
            params,
        }
    }
}

/// CPU-side SDF evaluation functions for testing and validation
pub mod cpu_eval {
    use super::*;

    /// Evaluate sphere SDF
    pub fn sphere_sdf(point: Vec3, sphere: &SdfSphere) -> f32 {
        let center = Vec3::from(sphere.center);
        (point - center).length() - sphere.radius
    }

    /// Evaluate box SDF
    pub fn box_sdf(point: Vec3, sdf_box: &SdfBox) -> f32 {
        let center = Vec3::from(sdf_box.center);
        let extents = Vec3::from(sdf_box.extents);
        let local_point = (point - center).abs();
        let q = local_point - extents;
        q.max(Vec3::ZERO).length() + q.max_element().min(0.0)
    }

    /// Evaluate cylinder SDF (oriented along Y-axis)
    pub fn cylinder_sdf(point: Vec3, cylinder: &SdfCylinder) -> f32 {
        let center = Vec3::from(cylinder.center);
        let local_point = point - center;
        let half_height = cylinder.height * 0.5;

        let xz_dist = Vec2::new(local_point.x, local_point.z).length();
        let radial_dist = xz_dist - cylinder.radius;
        let vertical_dist = local_point.y.abs() - half_height;

        radial_dist.max(vertical_dist).max(0.0)
            + Vec2::new(radial_dist.max(0.0), vertical_dist.max(0.0)).length()
    }

    /// Evaluate plane SDF
    pub fn plane_sdf(point: Vec3, plane: &SdfPlane) -> f32 {
        let normal = Vec3::from(plane.normal);
        point.dot(normal) + plane.distance
    }

    /// Evaluate torus SDF
    pub fn torus_sdf(point: Vec3, torus: &SdfTorus) -> f32 {
        let center = Vec3::from(torus.center);
        let local_point = point - center;
        let xz_dist = Vec2::new(local_point.x, local_point.z).length();
        let q = Vec2::new(xz_dist - torus.major_radius, local_point.y);
        q.length() - torus.minor_radius
    }

    /// Evaluate capsule SDF
    pub fn capsule_sdf(point: Vec3, capsule: &SdfCapsule) -> f32 {
        let point_a = Vec3::from(capsule.point_a);
        let point_b = Vec3::from(capsule.point_b);
        let segment = point_b - point_a;
        let pa = point - point_a;

        let h = (pa.dot(segment) / segment.dot(segment)).clamp(0.0, 1.0);
        let closest = point_a + segment * h;
        (point - closest).length() - capsule.radius
    }

    /// Evaluate arbitrary SDF primitive
    pub fn evaluate_primitive(point: Vec3, primitive: &SdfPrimitive) -> f32 {
        match primitive.primitive_type {
            0 => {
                let p = &primitive.params;
                let sphere = SdfSphere {
                    center: [p[0], p[1], p[2]],
                    radius: p[3],
                };
                sphere_sdf(point, &sphere)
            }
            1 => {
                let p = &primitive.params;
                let sdf_box = SdfBox {
                    center: [p[0], p[1], p[2]],
                    _pad1: 0.0,
                    extents: [p[4], p[5], p[6]],
                    _pad2: 0.0,
                };
                box_sdf(point, &sdf_box)
            }
            2 => {
                let p = &primitive.params;
                let cylinder = SdfCylinder {
                    center: [p[0], p[1], p[2]],
                    radius: p[3],
                    height: p[4],
                    _pad: [0.0; 3],
                };
                cylinder_sdf(point, &cylinder)
            }
            3 => {
                let p = &primitive.params;
                let plane = SdfPlane {
                    normal: [p[0], p[1], p[2]],
                    distance: p[3],
                };
                plane_sdf(point, &plane)
            }
            4 => {
                let p = &primitive.params;
                let torus = SdfTorus {
                    center: [p[0], p[1], p[2]],
                    major_radius: p[3],
                    minor_radius: p[4],
                    _pad: [0.0; 3],
                };
                torus_sdf(point, &torus)
            }
            5 => {
                let p = &primitive.params;
                let capsule = SdfCapsule {
                    point_a: [p[0], p[1], p[2]],
                    radius: p[3],
                    point_b: [p[4], p[5], p[6]],
                    _pad: 0.0,
                };
                capsule_sdf(point, &capsule)
            }
            _ => f32::INFINITY,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sphere_sdf() {
        let sphere = SdfSphere {
            center: [0.0, 0.0, 0.0],
            radius: 1.0,
        };

        // Point on surface
        assert!((cpu_eval::sphere_sdf(Vec3::new(1.0, 0.0, 0.0), &sphere).abs()) < 1e-6);

        // Point inside
        assert!(cpu_eval::sphere_sdf(Vec3::new(0.5, 0.0, 0.0), &sphere) < 0.0);

        // Point outside
        assert!(cpu_eval::sphere_sdf(Vec3::new(2.0, 0.0, 0.0), &sphere) > 0.0);
    }

    #[test]
    fn test_box_sdf() {
        let sdf_box = SdfBox {
            center: [0.0, 0.0, 0.0],
            _pad1: 0.0,
            extents: [1.0, 1.0, 1.0],
            _pad2: 0.0,
        };

        // Point on surface
        assert!((cpu_eval::box_sdf(Vec3::new(1.0, 0.0, 0.0), &sdf_box).abs()) < 1e-6);

        // Point inside
        assert!(cpu_eval::box_sdf(Vec3::new(0.5, 0.5, 0.5), &sdf_box) < 0.0);

        // Point outside
        assert!(cpu_eval::box_sdf(Vec3::new(2.0, 0.0, 0.0), &sdf_box) > 0.0);
    }

    #[test]
    fn test_primitive_creation() {
        let sphere = SdfPrimitive::sphere(Vec3::new(1.0, 2.0, 3.0), 0.5, 42);
        assert_eq!(sphere.primitive_type, SdfPrimitiveType::Sphere as u32);
        assert_eq!(sphere.material_id, 42);

        let sdf_box =
            SdfPrimitive::box_primitive(Vec3::new(0.0, 0.0, 0.0), Vec3::new(1.0, 2.0, 3.0), 1);
        assert_eq!(sdf_box.primitive_type, SdfPrimitiveType::Box as u32);
        assert_eq!(sdf_box.material_id, 1);
    }

    #[test]
    fn test_primitive_evaluation() {
        let sphere = SdfPrimitive::sphere(Vec3::ZERO, 1.0, 0);

        // Test evaluation through generic interface
        let distance = cpu_eval::evaluate_primitive(Vec3::new(2.0, 0.0, 0.0), &sphere);
        assert!((distance - 1.0).abs() < 1e-6);
    }
}
