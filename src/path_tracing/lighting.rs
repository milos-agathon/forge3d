// src/path_tracing/lighting.rs
// Area light GPU types and helpers for wavefront path tracing (P3: A4/A20/A25 wiring)
// RELEVANT FILES: src/shaders/pt_shade.wgsl, src/path_tracing/wavefront/pipeline.rs

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

/// GPU layout matching WGSL `AreaLight` in pt_shade.wgsl
///
/// struct AreaLight {
///     position: vec3<f32>,
///     radius: f32,
///     normal: vec3<f32>,
///     intensity: f32,
///     color: vec3<f32>,
///     importance: f32,
/// }
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct GpuAreaLight {
    pub position: [f32; 3],
    pub radius: f32,
    pub normal: [f32; 3],
    pub intensity: f32,
    pub color: [f32; 3],
    pub importance: f32,
}

impl GpuAreaLight {
    /// Create a disc area light with position, normal, radius, intensity, and color.
    pub fn disc(
        position: [f32; 3],
        normal: [f32; 3],
        radius: f32,
        intensity: f32,
        color: [f32; 3],
        importance: f32,
    ) -> Self {
        Self {
            position,
            radius: radius.max(0.0),
            normal,
            intensity: intensity.max(0.0),
            color,
            importance: importance.max(0.0),
        }
    }
}

/// Create an area lights storage buffer for the wavefront scene bind group (Group 1, binding=4)
pub fn create_area_lights_buffer(device: &wgpu::Device, lights: &[GpuAreaLight]) -> wgpu::Buffer {
    let bytes = bytemuck::cast_slice(lights);
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("area-lights-buffer"),
        contents: bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    })
}

/// Create an empty area lights buffer (0 elements) that still satisfies binding requirements
pub fn empty_area_lights_buffer(device: &wgpu::Device) -> wgpu::Buffer {
    // Allocate a single zeroed element for portability; WGSL can treat length as 0 if not used
    let zero = [GpuAreaLight::disc(
        [0.0; 3],
        [0.0, 1.0, 0.0],
        0.0,
        0.0,
        [0.0; 3],
        0.0,
    )];
    create_area_lights_buffer(device, &zero[..0])
}

/// GPU layout matching WGSL `DirectionalLight` in pt_shade.wgsl
///
/// struct DirectionalLight {
///     direction: vec3<f32>,
///     intensity: f32,
///     color: vec3<f32>,
///     importance: f32,
/// }
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct GpuDirectionalLight {
    pub direction: [f32; 3],
    pub intensity: f32,
    pub color: [f32; 3],
    pub importance: f32,
}

impl GpuDirectionalLight {
    pub fn new(direction: [f32; 3], intensity: f32, color: [f32; 3], importance: f32) -> Self {
        // Ensure direction is normalized on CPU side for consistency
        let len = (direction[0] * direction[0]
            + direction[1] * direction[1]
            + direction[2] * direction[2])
            .sqrt();
        let dir = if len > 0.0 {
            [direction[0] / len, direction[1] / len, direction[2] / len]
        } else {
            [0.0, -1.0, 0.0]
        };
        Self {
            direction: dir,
            intensity: intensity.max(0.0),
            color,
            importance: importance.max(0.0),
        }
    }
}

pub fn create_directional_lights_buffer(
    device: &wgpu::Device,
    lights: &[GpuDirectionalLight],
) -> wgpu::Buffer {
    let bytes = bytemuck::cast_slice(lights);
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("directional-lights-buffer"),
        contents: bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    })
}

pub fn empty_directional_lights_buffer(device: &wgpu::Device) -> wgpu::Buffer {
    let zero = [GpuDirectionalLight::new(
        [0.0, -1.0, 0.0],
        0.0,
        [0.0; 3],
        0.0,
    )];
    create_directional_lights_buffer(device, &zero[..0])
}
