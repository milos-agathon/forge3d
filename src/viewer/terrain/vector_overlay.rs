// src/viewer/terrain/vector_overlay.rs
// Option B: Vector overlay geometry rendered as additional lit passes
// Renders vector overlays (points/lines/polygons) as GPU geometry in world space,
// optionally draped onto terrain heightfield, with proper lighting and shadowing.

use crate::core::resource_tracker::TrackedBuffer;
use glam::DVec3;
use std::sync::Arc;

pub(crate) fn build_layer_bvh(
    layer_id: u32,
    name: &str,
    vertices: &[VectorVertex],
    indices: &[u32],
    primitive: OverlayPrimitive,
) -> Option<crate::picking::LayerBvhData> {
    use crate::accel::cpu_bvh::{build_bvh_cpu, BuildOptions, MeshCPU};
    use crate::accel::types::{Aabb, BvhNode, Triangle};

    if primitive != OverlayPrimitive::Triangles || indices.is_empty() {
        return None;
    }
    let positions = vertices
        .iter()
        .map(|vertex| vertex.position)
        .collect::<Vec<_>>();
    let triangles = indices
        .chunks_exact(3)
        .map(|chunk| [chunk[0], chunk[1], chunk[2]])
        .collect::<Vec<_>>();
    let mesh = MeshCPU::new(positions.clone(), triangles.clone());
    let bvh = build_bvh_cpu(&mesh, &BuildOptions::default()).ok()?;
    let mut layer = crate::picking::LayerBvhData::new(layer_id, name.to_string());
    layer.cpu_nodes = bvh
        .nodes
        .iter()
        .map(|node| BvhNode {
            aabb: Aabb::new(node.aabb_min, node.aabb_max),
            kind: u32::from(node.is_leaf()),
            left_idx: node.left,
            right_idx: node.right,
            parent_idx: 0,
        })
        .collect();
    layer.cpu_triangles = bvh
        .tri_indices
        .iter()
        .map(|&tri_idx| {
            let tri = triangles.get(tri_idx as usize).copied().unwrap_or([0; 3]);
            Triangle::new(
                positions.get(tri[0] as usize).copied().unwrap_or([0.0; 3]),
                positions.get(tri[1] as usize).copied().unwrap_or([0.0; 3]),
                positions.get(tri[2] as usize).copied().unwrap_or([0.0; 3]),
            )
        })
        .collect();
    layer.cpu_feature_ids = bvh
        .tri_indices
        .iter()
        .map(|&tri_idx| {
            triangles
                .get(tri_idx as usize)
                .and_then(|tri| vertices.get(tri[0] as usize))
                .map(|vertex| vertex.feature_id)
                .unwrap_or(layer_id)
        })
        .collect();
    Some(layer)
}

/// Primitive type for vector overlay
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum OverlayPrimitive {
    Points,
    Lines,
    LineStrip,
    #[default]
    Triangles,
    TriangleStrip,
}

impl OverlayPrimitive {
    /// Create from string (for IPC parsing)
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "points" => Some(OverlayPrimitive::Points),
            "lines" => Some(OverlayPrimitive::Lines),
            "line_strip" | "linestrip" => Some(OverlayPrimitive::LineStrip),
            "triangles" => Some(OverlayPrimitive::Triangles),
            "triangle_strip" | "trianglestrip" => Some(OverlayPrimitive::TriangleStrip),
            _ => None,
        }
    }
}

/// Vertex format for vector overlay geometry
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct VectorVertex {
    pub position: [f32; 3], // World XYZ (Y may be offset for drape)
    pub color: [f32; 4],    // RGBA vertex color
    pub uv: [f32; 2],       // Texture coords (for textured overlays)
    pub normal: [f32; 3],   // For lit overlays (default: up vec)
    pub feature_id: u32,    // Feature ID for picking (0 = no feature)
}

/// Persistent absolute vector vertex. This is never uploaded directly.
#[derive(Copy, Clone, Debug)]
pub struct VectorSourceVertex {
    pub position: DVec3,
    pub color: [f32; 4],
    pub feature_id: u32,
}

impl From<crate::viewer::viewer_enums::ViewerVectorVertex> for VectorSourceVertex {
    fn from(value: crate::viewer::viewer_enums::ViewerVectorVertex) -> Self {
        Self {
            position: DVec3::from(value.position),
            color: value.color,
            feature_id: value.feature_id,
        }
    }
}

impl VectorVertex {
    /// Create a new vertex with position, color, and feature ID
    pub fn with_feature_id(
        x: f32,
        y: f32,
        z: f32,
        r: f32,
        g: f32,
        b: f32,
        a: f32,
        feature_id: u32,
    ) -> Self {
        Self {
            position: [x, y, z],
            color: [r, g, b, a],
            uv: [0.0, 0.0],
            normal: [0.0, 1.0, 0.0],
            feature_id,
        }
    }

    /// Vertex buffer layout descriptor
    pub fn desc() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<VectorVertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                // position: vec3<f32>
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x3,
                },
                // color: vec4<f32>
                wgpu::VertexAttribute {
                    offset: 12,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x4,
                },
                // uv: vec2<f32>
                wgpu::VertexAttribute {
                    offset: 28,
                    shader_location: 2,
                    format: wgpu::VertexFormat::Float32x2,
                },
                // normal: vec3<f32>
                wgpu::VertexAttribute {
                    offset: 36,
                    shader_location: 3,
                    format: wgpu::VertexFormat::Float32x3,
                },
                // feature_id: u32
                wgpu::VertexAttribute {
                    offset: 48,
                    shader_location: 4,
                    format: wgpu::VertexFormat::Uint32,
                },
            ],
        }
    }
}

/// Vector overlay layer configuration
#[derive(Clone, Debug)]
pub struct VectorOverlayLayer {
    pub name: String,
    pub source_vertices: Vec<VectorSourceVertex>,
    pub vertices: Vec<VectorVertex>,
    pub indices: Vec<u32>,
    pub primitive: OverlayPrimitive,
    pub drape: bool,       // If true, drape onto terrain
    pub drape_offset: f32, // Height above terrain when draped
    pub opacity: f32,      // 0.0 - 1.0
    pub depth_bias: f32,   // Z-fighting prevention (0.01 - 1.0)
    pub line_width: f32,   // For Lines/LineStrip (1.0 - 10.0)
    pub point_size: f32,   // For Points (1.0 - 20.0)
    pub visible: bool,
    pub z_order: i32,
}

impl Default for VectorOverlayLayer {
    fn default() -> Self {
        Self {
            name: String::new(),
            source_vertices: Vec::new(),
            vertices: Vec::new(),
            indices: Vec::new(),
            primitive: OverlayPrimitive::Triangles,
            drape: false,
            drape_offset: 0.5,
            opacity: 1.0,
            depth_bias: 0.1,
            line_width: 2.0,
            point_size: 5.0,
            visible: true,
            z_order: 0,
        }
    }
}

/// GPU resources for a vector overlay layer
pub struct VectorOverlayGpu {
    pub config: VectorOverlayLayer,
    pub vertex_buffer: TrackedBuffer,
    pub index_buffer: TrackedBuffer,
    pub vertex_count: u32,
    pub index_count: u32,
    pub id: u32,
}

/// Overlay uniforms for GPU
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct VectorOverlayUniforms {
    pub view_proj: [[f32; 4]; 4],     // 64 bytes
    pub sun_dir: [f32; 4],            // 16 bytes
    pub lighting: [f32; 4],           // sun_intensity, ambient, shadow_strength, terrain_width
    pub layer_params: [f32; 4],       // opacity, depth_bias, line_width, point_size
    pub highlight_color: [f32; 4],    // Highlight color for selected features (RGBA)
    pub selected_feature_id: u32,     // Currently selected feature ID (0 = none)
    pub _pad: [u32; 7],               // Preserve the existing 160-byte prefix.
    pub render_origin_span: [f32; 4], // Anchored origin x/z and physical span x/z.
}

pub struct VectorOverlayStack {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    layers: Vec<VectorOverlayGpu>,
    next_id: u32,
    enabled: bool,
    dirty: bool,

    // Global settings
    global_opacity: f32,

    // GPU resources for rendering
    pub pipeline_triangles: Option<wgpu::RenderPipeline>,
    pub pipeline_lines: Option<wgpu::RenderPipeline>,
    pub pipeline_points: Option<wgpu::RenderPipeline>,
    pub bind_group_layout: Option<wgpu::BindGroupLayout>,
    pub uniform_buffer: Option<TrackedBuffer>,
    pub bind_group: Option<wgpu::BindGroup>,
    pub sampler: Option<wgpu::Sampler>,

    // P0.1/M1: OIT pipelines with WBOIT blend states
    pub oit_pipeline_triangles: Option<wgpu::RenderPipeline>,
    pub oit_pipeline_lines: Option<wgpu::RenderPipeline>,
    pub oit_pipeline_points: Option<wgpu::RenderPipeline>,
}

mod core;
mod pipelines;

fn sample_heightmap_bilinear(heightmap: &[f32], dims: (u32, u32), u: f32, v: f32) -> f32 {
    let (w, h) = dims;
    if w == 0 || h == 0 || heightmap.is_empty() {
        return 0.0;
    }

    // Clamp to valid range
    let u = u.clamp(0.0, 1.0);
    let v = v.clamp(0.0, 1.0);

    // Convert to pixel coordinates
    let x = u * (w - 1) as f32;
    let y = v * (h - 1) as f32;

    let x0 = x.floor() as u32;
    let y0 = y.floor() as u32;
    let x1 = (x0 + 1).min(w - 1);
    let y1 = (y0 + 1).min(h - 1);

    let fx = x.fract();
    let fy = y.fract();

    // Sample four corners
    let idx = |ix: u32, iy: u32| -> usize { (iy * w + ix) as usize };
    let h00 = heightmap.get(idx(x0, y0)).copied().unwrap_or(0.0);
    let h10 = heightmap.get(idx(x1, y0)).copied().unwrap_or(0.0);
    let h01 = heightmap.get(idx(x0, y1)).copied().unwrap_or(0.0);
    let h11 = heightmap.get(idx(x1, y1)).copied().unwrap_or(0.0);

    // Bilinear interpolation
    let h0 = h00 * (1.0 - fx) + h10 * fx;
    let h1 = h01 * (1.0 - fx) + h11 * fx;
    h0 * (1.0 - fy) + h1 * fy
}

/// Compute terrain normal at a point using central differences
fn compute_terrain_normal(
    heightmap: &[f32],
    dims: (u32, u32),
    u: f32,
    v: f32,
    terrain_span: [f32; 2],
) -> [f32; 3] {
    let (w, h) = dims;
    if w < 2 || h < 2 || heightmap.is_empty() {
        return [0.0, 1.0, 0.0]; // Default up
    }

    let du = 1.0 / w as f32;
    let dv = 1.0 / h as f32;

    let h_left = sample_heightmap_bilinear(heightmap, dims, u - du, v);
    let h_right = sample_heightmap_bilinear(heightmap, dims, u + du, v);
    let h_down = sample_heightmap_bilinear(heightmap, dims, u, v - dv);
    let h_up = sample_heightmap_bilinear(heightmap, dims, u, v + dv);

    // Gradient in height per world unit
    let world_du = terrain_span[0] * 2.0 * du;
    let world_dv = terrain_span[1] * 2.0 * dv;

    let dh_dx = (h_right - h_left) / world_du;
    let dh_dz = (h_up - h_down) / world_dv;

    // Normal = normalize(-dh/dx, 1, -dh/dz)
    let len = (dh_dx * dh_dx + 1.0 + dh_dz * dh_dz).sqrt();
    [-dh_dx / len, 1.0 / len, -dh_dz / len]
}

/// Repack persistent f64 vector sources into the current render frame. Drape
/// UVs are computed in f64 before the one Anchor narrowing boundary.
pub fn repack_source_vertices(
    layer: &mut VectorOverlayLayer,
    terrain: Option<&crate::viewer::terrain::scene::ViewerTerrainData>,
    anchor: &crate::camera::Anchor,
) {
    layer.vertices.resize(
        layer.source_vertices.len(),
        VectorVertex::with_feature_id(0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0),
    );
    for (source, render) in layer.source_vertices.iter().zip(layer.vertices.iter_mut()) {
        let mut world = source.position;
        let mut normal = [0.0, 1.0, 0.0];
        if layer.drape {
            if let Some(terrain) = terrain {
                let u = (world.x - terrain.world_origin_xz.x) / terrain.world_span_xz.x;
                let v = (world.z - terrain.world_origin_xz.y) / terrain.world_span_xz.y;
                let uv = crate::camera::Anchor::new().to_render_direction(DVec3::new(u, v, 0.0));
                let height =
                    sample_heightmap_bilinear(&terrain.heightmap, terrain.dimensions, uv.x, uv.y);
                world.y +=
                    f64::from((height - terrain.domain.0) * terrain.z_scale + layer.drape_offset);
                let span = crate::camera::Anchor::new().to_render_direction(DVec3::new(
                    terrain.world_span_xz.x,
                    terrain.world_span_xz.y,
                    0.0,
                ));
                normal = compute_terrain_normal(
                    &terrain.heightmap,
                    terrain.dimensions,
                    uv.x,
                    uv.y,
                    [span.x, span.y],
                );
            }
        }
        render.position = anchor.to_render_vec3(world).to_array();
        render.color = source.color;
        render.normal = normal;
        render.feature_id = source.feature_id;
    }
}

/// Parameters for render_layer_with_highlight
pub struct RenderLayerParams {
    pub layer_index: usize,
    pub view_proj: [[f32; 4]; 4],
    pub sun_dir: [f32; 3],
    pub lighting: [f32; 4],
    pub render_origin_span: [f32; 4],
    pub selected_feature_id: u32,
    pub highlight_color: [f32; 4],
}

// ============================================================================
// Shader source (Milestone 3)
// ============================================================================

/// WGSL shader for vector overlay rendering with lighting and shadows
pub const VECTOR_OVERLAY_SHADER: &str = r#"
// Vector Overlay Shader with Lighting, Shadow Integration, and Picking Highlight

struct Uniforms {
    view_proj: mat4x4<f32>,
    sun_dir: vec4<f32>,
    lighting: vec4<f32>,       // sun_intensity, ambient, shadow_strength, terrain_width
    layer_params: vec4<f32>,   // opacity, depth_bias, line_width, point_size
    highlight_color: vec4<f32>, // Highlight color for selected features (RGBA)
    selected_feature_id: u32,  // Currently selected feature ID (0 = none)
    _pad: vec3<u32>,           // Padding for alignment
    render_origin_span: vec4<f32>, // anchored origin x/z, physical span x/z
};

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var sun_vis_tex: texture_2d<f32>;
@group(0) @binding(2) var sun_vis_sampler: sampler;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) color: vec4<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) normal: vec3<f32>,
    @location(4) feature_id: u32,
};

struct VertexOutput {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) world_pos: vec3<f32>,
    @location(2) normal: vec3<f32>,
    @location(3) @interpolate(flat) feature_id: u32,
};

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    
    var pos = in.position;
    let depth_bias = u.layer_params.y;
    pos.y += depth_bias;
    
    out.clip_pos = u.view_proj * vec4<f32>(pos, 1.0);
    out.color = in.color;
    out.world_pos = pos;
    out.normal = in.normal;
    out.feature_id = in.feature_id;
    
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let opacity = u.layer_params.x;
    let sun_intensity = u.lighting.x;
    let ambient = u.lighting.y;
    let shadow_strength = u.lighting.z;
    
    // Normalize normal
    let normal = normalize(in.normal);
    let sun_dir = normalize(u.sun_dir.xyz);
    
    // Directional lighting (simple Lambertian)
    let ndotl = max(dot(normal, sun_dir), 0.0);
    let diffuse = ndotl * sun_intensity;
    
    // Sample sun visibility for shadow (use world position to compute UV)
    // UV is based on world XZ normalized to [0,1] (terrain goes from 0 to terrain_width)
    // Clamp to [0,1] to ensure valid texture sampling even for vertices slightly outside bounds
    let uv = clamp(
        (in.world_pos.xz - u.render_origin_span.xy) / u.render_origin_span.zw,
        vec2<f32>(0.0),
        vec2<f32>(1.0),
    );
    let sun_vis = textureSampleLevel(sun_vis_tex, sun_vis_sampler, uv, 0.0).r;
    
    // Shadow factor: 1.0 = fully lit, 0.0 = fully shadowed
    let shadow = mix(1.0, sun_vis, shadow_strength);
    
    // Combine lighting
    let light = ambient + diffuse * shadow;
    
    // Apply lighting to color
    var lit_color = in.color.rgb * light;
    
    // Apply highlight if this feature is selected
    if (u.selected_feature_id != 0u && in.feature_id == u.selected_feature_id) {
        // Blend with highlight color
        lit_color = mix(lit_color, u.highlight_color.rgb, u.highlight_color.a);
    }
    
    return vec4<f32>(lit_color, in.color.a * opacity);
}

// P0.1/M1: OIT fragment shader for WBOIT accumulation
// Outputs to two render targets: color accumulation and reveal accumulation
struct OitOutput {
    @location(0) color_accum: vec4<f32>,
    @location(1) reveal_accum: f32,
}

@fragment
fn fs_main_oit(in: VertexOutput) -> OitOutput {
    let opacity = u.layer_params.x;
    let sun_intensity = u.lighting.x;
    let ambient = u.lighting.y;
    let shadow_strength = u.lighting.z;
    
    let normal = normalize(in.normal);
    let sun_dir = normalize(u.sun_dir.xyz);
    let ndotl = max(dot(normal, sun_dir), 0.0);
    let diffuse = ndotl * sun_intensity;
    
    let uv = clamp(
        (in.world_pos.xz - u.render_origin_span.xy) / u.render_origin_span.zw,
        vec2<f32>(0.0),
        vec2<f32>(1.0),
    );
    let sun_vis = textureSampleLevel(sun_vis_tex, sun_vis_sampler, uv, 0.0).r;
    let shadow = mix(1.0, sun_vis, shadow_strength);
    let light = ambient + diffuse * shadow;
    
    var lit_color = in.color.rgb * light;
    
    if (u.selected_feature_id != 0u && in.feature_id == u.selected_feature_id) {
        lit_color = mix(lit_color, u.highlight_color.rgb, u.highlight_color.a);
    }
    
    let alpha = in.color.a * opacity;
    
    // WBOIT weight calculation based on depth and alpha
    // Weight formula: alpha * max(min(1.0, pow(clip_z, 3)), 0.01)
    let clip_z = in.clip_pos.z / in.clip_pos.w;
    let depth_weight = clamp(pow(1.0 - clip_z, 3.0), 0.01, 1.0);
    let weight = alpha * depth_weight * 8.0 + 0.01;
    
    var out: OitOutput;
    // Color accumulation: premultiplied color * weight, weight sum in alpha
    out.color_accum = vec4<f32>(lit_color * alpha * weight, alpha * weight);
    // Reveal accumulation: product of (1 - alpha) via multiplicative blending
    out.reveal_accum = alpha;
    
    return out;
}
"#;

#[cfg(test)]
mod option2_precision_tests {
    use super::*;

    fn layer_at(origin: DVec3) -> VectorOverlayLayer {
        VectorOverlayLayer {
            source_vertices: vec![
                VectorSourceVertex {
                    position: origin,
                    color: [1.0, 0.0, 0.0, 1.0],
                    feature_id: 7,
                },
                VectorSourceVertex {
                    position: origin + DVec3::new(0.001, 0.0, 0.0),
                    color: [0.0, 0.0, 1.0, 1.0],
                    feature_id: 8,
                },
            ],
            primitive: OverlayPrimitive::Points,
            ..VectorOverlayLayer::default()
        }
    }

    #[test]
    fn earth_scale_millimetre_survives_anchor_packing_and_rebases() {
        let origin = DVec3::new(6_378_137.0, 42.0, -5_500_000.0);
        let mut layer = layer_at(origin);
        for offset in [-500.0, -125.0, 0.0, 125.0, 500.0] {
            let mut anchor = crate::camera::Anchor::new();
            anchor.rebase_if_needed(origin + DVec3::new(offset, 0.0, 0.0));
            repack_source_vertices(&mut layer, None, &anchor);
            let packed_delta =
                f64::from(layer.vertices[1].position[0] - layer.vertices[0].position[0]);
            assert!(
                (packed_delta - 0.001).abs() <= 0.00025,
                "packed delta {packed_delta} at rebase offset {offset}"
            );
            assert_eq!(layer.vertices[0].feature_id, 7);
            assert_eq!(layer.vertices[1].feature_id, 8);
        }
    }

    #[test]
    fn zero_millimetre_negative_control_collapses_exactly() {
        let origin = DVec3::new(6_378_137.0, 42.0, -5_500_000.0);
        let mut layer = layer_at(origin);
        layer.source_vertices[1].position = origin;
        let mut anchor = crate::camera::Anchor::new();
        anchor.rebase_if_needed(origin);
        repack_source_vertices(&mut layer, None, &anchor);
        assert_eq!(layer.vertices[0].position, layer.vertices[1].position);
    }
}

// ============================================================================
// Pipeline and Rendering Implementation (Milestone 3)
// ============================================================================
