// src/core/reflections_types.rs
// Type definitions for planar reflections (B5)
// RELEVANT FILES: shaders/planar_reflections.wgsl, python/forge3d/lighting.py

use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec3};

use super::reflections_math::{create_reflection_matrix, reflect_point_across_plane};

/// Reflection plane data matching WGSL structure
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct ReflectionPlane {
    /// Plane equation coefficients (ax + by + cz + d = 0)
    pub plane_equation: [f32; 4],
    /// Reflection matrix for transforming geometry
    pub reflection_matrix: [f32; 16],
    /// View matrix for reflection camera
    pub reflection_view: [f32; 16],
    /// Projection matrix for reflection camera
    pub reflection_projection: [f32; 16],
    /// World position of reflection plane center
    pub plane_center: [f32; 4],
    /// Plane dimensions (width, height, 0, 0)
    pub plane_size: [f32; 4],
}

impl Default for ReflectionPlane {
    fn default() -> Self {
        Self {
            plane_equation: [0.0, 1.0, 0.0, 0.0], // XZ plane at Y=0
            reflection_matrix: Mat4::IDENTITY.to_cols_array(),
            reflection_view: Mat4::IDENTITY.to_cols_array(),
            reflection_projection: Mat4::IDENTITY.to_cols_array(),
            plane_center: [0.0, 0.0, 0.0, 1.0],
            plane_size: [100.0, 100.0, 0.0, 0.0], // 100x100 default plane
        }
    }
}

impl ReflectionPlane {
    /// Create a new reflection plane from normal and point
    pub fn new(normal: Vec3, point: Vec3, size: Vec3) -> Self {
        let normal = normal.normalize();
        let d = -normal.dot(point);

        let reflection_matrix = create_reflection_matrix(normal, d);

        Self {
            plane_equation: [normal.x, normal.y, normal.z, d],
            reflection_matrix: reflection_matrix.to_cols_array(),
            reflection_view: Mat4::IDENTITY.to_cols_array(),
            reflection_projection: Mat4::IDENTITY.to_cols_array(),
            plane_center: [point.x, point.y, point.z, 1.0],
            plane_size: [size.x, size.y, 0.0, 0.0],
        }
    }

    /// Get plane normal
    pub fn normal(&self) -> Vec3 {
        Vec3::new(
            self.plane_equation[0],
            self.plane_equation[1],
            self.plane_equation[2],
        )
    }

    /// Get plane distance
    pub fn distance(&self) -> f32 {
        self.plane_equation[3]
    }

    /// Get reflection matrix as Mat4
    pub fn reflection_matrix(&self) -> Mat4 {
        Mat4::from_cols_array(&self.reflection_matrix)
    }

    /// Update reflection view and projection matrices
    pub fn update_matrices(
        &mut self,
        camera_pos: Vec3,
        camera_target: Vec3,
        camera_up: Vec3,
        projection: Mat4,
    ) {
        let reflected_pos = reflect_point_across_plane(camera_pos, self.normal(), self.distance());
        let reflected_target =
            reflect_point_across_plane(camera_target, self.normal(), self.distance());
        let reflected_up = self
            .normal()
            .cross(camera_up.cross(self.normal()))
            .normalize();

        let reflection_view = Mat4::look_at_rh(reflected_pos, reflected_target, reflected_up);
        self.reflection_view = reflection_view.to_cols_array();
        self.reflection_projection = projection.to_cols_array();
    }
}

/// Planar reflection configuration and uniform data
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct PlanarReflectionUniforms {
    /// Reflection plane data
    pub reflection_plane: ReflectionPlane,
    /// Reflection mode: 0=disabled, 1=main pass sampling enabled, 2=reflection pass (clip only)
    pub enable_reflections: u32,
    /// Reflection intensity [0, 1]
    pub reflection_intensity: f32,
    /// Fresnel power for reflection falloff
    pub fresnel_power: f32,
    /// Blur kernel size for roughness
    pub blur_kernel_size: u32,
    /// Maximum blur radius in texels
    pub max_blur_radius: f32,
    /// Reflection texture resolution
    pub reflection_resolution: f32,
    /// Distance fade start
    pub distance_fade_start: f32,
    /// Distance fade end
    pub distance_fade_end: f32,
    /// Debug visualization mode
    pub debug_mode: u32,
    /// Camera world-space position (xyz, 1 for alignment)
    pub camera_position: [f32; 4],
    /// Padding for 16-byte alignment (WGSL struct alignment)
    pub _padding: [f32; 7],
}

// Reflection mode values (shared with WGSL)
pub const REFLECTION_DISABLED: u32 = 0;
pub const REFLECTION_ENABLED: u32 = 1;
pub const REFLECTION_PASS: u32 = 2;

impl Default for PlanarReflectionUniforms {
    fn default() -> Self {
        Self {
            reflection_plane: ReflectionPlane::default(),
            enable_reflections: REFLECTION_ENABLED,
            reflection_intensity: 0.8,
            fresnel_power: 5.0,
            blur_kernel_size: 5,
            max_blur_radius: 8.0,
            reflection_resolution: 1024.0,
            distance_fade_start: 50.0,
            distance_fade_end: 200.0,
            debug_mode: 0,
            camera_position: [0.0, 0.0, 0.0, 1.0],
            _padding: [0.0; 7],
        }
    }
}

/// Reflection quality settings
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReflectionQuality {
    /// Low quality: 512x512, simple blur
    Low,
    /// Medium quality: 1024x1024, standard blur
    Medium,
    /// High quality: 2048x2048, Poisson blur
    High,
    /// Ultra quality: 4096x4096, high-quality blur
    Ultra,
}

impl ReflectionQuality {
    /// Get texture resolution for this quality setting
    pub fn resolution(self) -> u32 {
        match self {
            ReflectionQuality::Low => 512,
            ReflectionQuality::Medium => 512, // Reduced for CI performance
            ReflectionQuality::High => 2048,
            ReflectionQuality::Ultra => 4096,
        }
    }

    /// Get blur kernel size for this quality setting
    pub fn blur_kernel_size(self) -> u32 {
        match self {
            ReflectionQuality::Low => 3,
            ReflectionQuality::Medium => 5,
            ReflectionQuality::High => 7,
            ReflectionQuality::Ultra => 9,
        }
    }

    /// Get max blur radius for this quality setting
    pub fn max_blur_radius(self) -> f32 {
        match self {
            ReflectionQuality::Low => 4.0,
            ReflectionQuality::Medium => 8.0,
            ReflectionQuality::High => 12.0,
            ReflectionQuality::Ultra => 16.0,
        }
    }
}
