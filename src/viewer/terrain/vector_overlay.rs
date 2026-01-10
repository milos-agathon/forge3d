// src/viewer/terrain/vector_overlay.rs
// Option B: Vector overlay geometry rendered as additional lit passes
// Renders vector overlays (points/lines/polygons) as GPU geometry in world space,
// optionally draped onto terrain heightfield, with proper lighting and shadowing.

use std::sync::Arc;
use wgpu::util::DeviceExt;

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
    /// Convert to wgpu primitive topology
    pub fn to_wgpu_topology(&self) -> wgpu::PrimitiveTopology {
        match self {
            OverlayPrimitive::Points => wgpu::PrimitiveTopology::PointList,
            OverlayPrimitive::Lines => wgpu::PrimitiveTopology::LineList,
            OverlayPrimitive::LineStrip => wgpu::PrimitiveTopology::LineStrip,
            OverlayPrimitive::Triangles => wgpu::PrimitiveTopology::TriangleList,
            OverlayPrimitive::TriangleStrip => wgpu::PrimitiveTopology::TriangleStrip,
        }
    }
    
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
    pub position: [f32; 3],  // World XYZ (Y may be offset for drape)
    pub color: [f32; 4],     // RGBA vertex color
    pub uv: [f32; 2],        // Texture coords (for textured overlays)
    pub normal: [f32; 3],    // For lit overlays (default: up vec)
    pub feature_id: u32,     // Feature ID for picking (0 = no feature)
}

impl VectorVertex {
    /// Create a new vertex with position and color
    pub fn new(x: f32, y: f32, z: f32, r: f32, g: f32, b: f32, a: f32) -> Self {
        Self {
            position: [x, y, z],
            color: [r, g, b, a],
            uv: [0.0, 0.0],
            normal: [0.0, 1.0, 0.0], // Default up normal
            feature_id: 0,
        }
    }
    
    /// Create a new vertex with position, color, and feature ID
    pub fn with_feature_id(x: f32, y: f32, z: f32, r: f32, g: f32, b: f32, a: f32, feature_id: u32) -> Self {
        Self {
            position: [x, y, z],
            color: [r, g, b, a],
            uv: [0.0, 0.0],
            normal: [0.0, 1.0, 0.0],
            feature_id,
        }
    }
    
    /// Create with full parameters
    pub fn with_all(
        position: [f32; 3],
        color: [f32; 4],
        uv: [f32; 2],
        normal: [f32; 3],
    ) -> Self {
        Self { position, color, uv, normal, feature_id: 0 }
    }
    
    /// Create with full parameters including feature ID
    pub fn with_all_and_id(
        position: [f32; 3],
        color: [f32; 4],
        uv: [f32; 2],
        normal: [f32; 3],
        feature_id: u32,
    ) -> Self {
        Self { position, color, uv, normal, feature_id }
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
    
    /// Set feature ID on this vertex
    pub fn set_feature_id(&mut self, id: u32) {
        self.feature_id = id;
    }
}

/// Vector overlay layer configuration
#[derive(Clone, Debug)]
pub struct VectorOverlayLayer {
    pub name: String,
    pub vertices: Vec<VectorVertex>,
    pub indices: Vec<u32>,
    pub primitive: OverlayPrimitive,
    pub drape: bool,              // If true, drape onto terrain
    pub drape_offset: f32,        // Height above terrain when draped
    pub opacity: f32,             // 0.0 - 1.0
    pub depth_bias: f32,          // Z-fighting prevention (0.01 - 1.0)
    pub line_width: f32,          // For Lines/LineStrip (1.0 - 10.0)
    pub point_size: f32,          // For Points (1.0 - 20.0)
    pub visible: bool,
    pub z_order: i32,
}

impl Default for VectorOverlayLayer {
    fn default() -> Self {
        Self {
            name: String::new(),
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
    pub vertex_buffer: wgpu::Buffer,
    pub index_buffer: wgpu::Buffer,
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
    pub _pad: [u32; 7],               // Padding: WGSL vec3 has 16-byte alignment → 160 bytes total
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
    pub uniform_buffer: Option<wgpu::Buffer>,
    pub bind_group: Option<wgpu::BindGroup>,
    pub sampler: Option<wgpu::Sampler>,
    
    // ID buffer pipelines for picking
    pub id_pipeline_triangles: Option<wgpu::RenderPipeline>,
    pub id_pipeline_lines: Option<wgpu::RenderPipeline>,
    pub id_pipeline_points: Option<wgpu::RenderPipeline>,
    pub id_bind_group_layout: Option<wgpu::BindGroupLayout>,
    pub id_uniform_buffer: Option<wgpu::Buffer>,
    pub id_bind_group: Option<wgpu::BindGroup>,
}

impl VectorOverlayStack {
    /// Create a new vector overlay stack
    pub fn new(device: Arc<wgpu::Device>, queue: Arc<wgpu::Queue>) -> Self {
        Self {
            device,
            queue,
            layers: Vec::new(),
            next_id: 0,
            enabled: true,
            dirty: false,
            global_opacity: 1.0,
            pipeline_triangles: None,
            pipeline_lines: None,
            pipeline_points: None,
            bind_group_layout: None,
            uniform_buffer: None,
            bind_group: None,
            sampler: None,
            id_pipeline_triangles: None,
            id_pipeline_lines: None,
            id_pipeline_points: None,
            id_bind_group_layout: None,
            id_uniform_buffer: None,
            id_bind_group: None,
        }
    }
    
    /// Add a vector overlay layer. Returns layer ID.
    pub fn add_layer(&mut self, layer: VectorOverlayLayer) -> u32 {
        let id = self.next_id;
        self.next_id += 1;
        
        // Create vertex buffer
        let vertex_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(&format!("vector_overlay_vertices_{}", id)),
            contents: bytemuck::cast_slice(&layer.vertices),
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        });
        
        // Create index buffer
        let index_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(&format!("vector_overlay_indices_{}", id)),
            contents: bytemuck::cast_slice(&layer.indices),
            usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
        });
        
        let gpu_layer = VectorOverlayGpu {
            vertex_count: layer.vertices.len() as u32,
            index_count: layer.indices.len() as u32,
            config: layer,
            vertex_buffer,
            index_buffer,
            id,
        };
        
        self.layers.push(gpu_layer);
        self.dirty = true;
        
        // Sort by z_order
        self.layers.sort_by_key(|l| l.config.z_order);
        
        id
    }
    
    /// Update vertices for an existing layer (for animation)
    pub fn update_vertices(&mut self, id: u32, vertices: Vec<VectorVertex>) {
        if let Some(layer) = self.layers.iter_mut().find(|l| l.id == id) {
            layer.config.vertices = vertices.clone();
            layer.vertex_count = vertices.len() as u32;
            
            // Recreate vertex buffer
            layer.vertex_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(&format!("vector_overlay_vertices_{}", id)),
                contents: bytemuck::cast_slice(&vertices),
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            });
            
            self.dirty = true;
        }
    }
    
    /// Remove a vector overlay by ID. Returns true if found and removed.
    pub fn remove(&mut self, id: u32) -> bool {
        if let Some(pos) = self.layers.iter().position(|l| l.id == id) {
            self.layers.remove(pos);
            self.dirty = true;
            true
        } else {
            false
        }
    }
    
    /// Set vector overlay visibility
    pub fn set_visible(&mut self, id: u32, visible: bool) {
        if let Some(layer) = self.layers.iter_mut().find(|l| l.id == id) {
            layer.config.visible = visible;
            self.dirty = true;
        }
    }
    
    /// Set vector overlay opacity
    pub fn set_opacity(&mut self, id: u32, opacity: f32) {
        if let Some(layer) = self.layers.iter_mut().find(|l| l.id == id) {
            layer.config.opacity = opacity.clamp(0.0, 1.0);
            self.dirty = true;
        }
    }
    
    /// List all vector overlay IDs in z-order
    pub fn list_ids(&self) -> Vec<u32> {
        self.layers.iter().map(|l| l.id).collect()
    }
    
    /// Get number of layers
    pub fn len(&self) -> usize {
        self.layers.len()
    }
    
    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.layers.is_empty()
    }
    
    /// Check if any visible layers exist
    pub fn has_visible_layers(&self) -> bool {
        self.layers.iter().any(|l| l.config.visible)
    }
    
    /// Check if enabled
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }
    
    /// Enable or disable the overlay system
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }
    
    /// Set global opacity
    pub fn set_global_opacity(&mut self, opacity: f32) {
        self.global_opacity = opacity.clamp(0.0, 1.0);
    }
    
    /// Get visible layers in z-order
    pub fn visible_layers(&self) -> impl Iterator<Item = &VectorOverlayGpu> {
        self.layers.iter().filter(|l| l.config.visible)
    }
    
    /// Get global opacity
    pub fn global_opacity(&self) -> f32 {
        self.global_opacity
    }
}

// ============================================================================
// Draping Implementation (Milestone 2)
// ============================================================================

/// Sample heightmap with bilinear interpolation
fn sample_heightmap_bilinear(
    heightmap: &[f32],
    dims: (u32, u32),
    u: f32,
    v: f32,
) -> f32 {
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
    terrain_width: f32,
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
    let world_du = terrain_width * 2.0 * du;
    let world_dv = terrain_width * 2.0 * dv;
    
    let dh_dx = (h_right - h_left) / world_du;
    let dh_dz = (h_up - h_down) / world_dv;
    
    // Normal = normalize(-dh/dx, 1, -dh/dz)
    let len = (dh_dx * dh_dx + 1.0 + dh_dz * dh_dz).sqrt();
    [-dh_dx / len, 1.0 / len, -dh_dz / len]
}

/// Parameters for drape_vertices
pub struct DrapeParams<'a> {
    pub vertices: &'a mut [VectorVertex],
    pub heightmap: &'a [f32],
    pub dims: (u32, u32),
    pub terrain_width: f32,
    pub terrain_origin: (f32, f32),
    pub height_offset: f32,
    pub height_min: f32,
    pub height_scale: f32,
}

/// Parameters for render_layer_with_highlight
pub struct RenderLayerParams {
    pub layer_index: usize,
    pub view_proj: [[f32; 4]; 4],
    pub sun_dir: [f32; 3],
    pub lighting: [f32; 4],
    pub selected_feature_id: u32,
    pub highlight_color: [f32; 4],
}

/// Drape vertices onto terrain surface
/// 
/// # Arguments
/// * `params` - Struct containing all draping parameters
pub fn drape_vertices(params: DrapeParams) {
    let DrapeParams {
        vertices,
        heightmap,
        dims,
        terrain_width,
        terrain_origin,
        height_offset,
        height_min,
        height_scale,
    } = params;

    if heightmap.is_empty() || dims.0 == 0 || dims.1 == 0 {
        // If no heightmap, just add offset to Y
        for v in vertices.iter_mut() {
            v.position[1] = height_offset;
            v.normal = [0.0, 1.0, 0.0];
        }
        return;
    }
    
    for v in vertices.iter_mut() {
        // Convert world XZ to terrain UV
        let x = v.position[0] - terrain_origin.0;
        let z = v.position[2] - terrain_origin.1;
        
        // Normalize to [0, 1] range (terrain goes from 0 to terrain_width)
        let u = (x / terrain_width).clamp(0.0, 1.0);
        let vv = (z / terrain_width).clamp(0.0, 1.0);
        
        // Sample heightmap and normalize like the terrain shader does:
        // world_y = (h - min_h) / h_range * terrain_width * z_scale * 0.001
        // Here height_scale = terrain_width * z_scale * 0.001 / h_range
        let h = sample_heightmap_bilinear(heightmap, dims, u, vv);
        let terrain_height = (h - height_min) * height_scale;
        
        // Set vertex Y to terrain height + offset
        v.position[1] = terrain_height + height_offset;
        
        // Compute normal from terrain gradient for proper lighting
        v.normal = compute_terrain_normal(heightmap, dims, u, vv, terrain_width);
    }
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
    let terrain_width = u.lighting.w;
    
    // Normalize normal
    let normal = normalize(in.normal);
    let sun_dir = normalize(u.sun_dir.xyz);
    
    // Directional lighting (simple Lambertian)
    let ndotl = max(dot(normal, sun_dir), 0.0);
    let diffuse = ndotl * sun_intensity;
    
    // Sample sun visibility for shadow (use world position to compute UV)
    // UV is based on world XZ normalized to [0,1] (terrain goes from 0 to terrain_width)
    // Clamp to [0,1] to ensure valid texture sampling even for vertices slightly outside bounds
    let uv = clamp(vec2<f32>(
        (in.world_pos.x / terrain_width),
        (in.world_pos.z / terrain_width)
    ), vec2<f32>(0.0, 0.0), vec2<f32>(1.0, 1.0));
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
"#;

// ============================================================================
// Pipeline and Rendering Implementation (Milestone 3)
// ============================================================================

impl VectorOverlayStack {
    /// Initialize the vector overlay render pipelines
    pub fn init_pipelines(&mut self, surface_format: wgpu::TextureFormat) {
        // Create bind group layout with texture/sampler for shadow integration
        let bind_group_layout = self.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("vector_overlay_bind_group_layout"),
            entries: &[
                // Uniforms
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Sun visibility texture (non-filterable for R32Float compatibility)
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // Sampler (non-filtering for R32Float compatibility)
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                    count: None,
                },
            ],
        });
        
        // Create uniform buffer
        let uniform_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("vector_overlay_uniforms"),
            size: std::mem::size_of::<VectorOverlayUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        
        // Compile shader
        let shader = self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("vector_overlay_shader"),
            source: wgpu::ShaderSource::Wgsl(VECTOR_OVERLAY_SHADER.into()),
        });
        
        // Create pipeline layout
        let pipeline_layout = self.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("vector_overlay_pipeline_layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        // Common depth stencil state (depth test, no write for overlays)
        // Use LessEqual to be more forgiving with depth precision, and aggressive bias
        // to ensure overlay is clearly in front of terrain
        let depth_stencil = Some(wgpu::DepthStencilState {
            format: wgpu::TextureFormat::Depth32Float,
            depth_write_enabled: false,  // Read only - don't write to depth
            depth_compare: wgpu::CompareFunction::LessEqual,
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState {
                constant: -100,  // Strong bias towards camera
                slope_scale: -10.0,
                clamp: 0.0,
            },
        });

        // Triangle pipeline
        self.pipeline_triangles = Some(self.device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("vector_overlay_triangles_pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                buffers: &[VectorVertex::desc()],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,  // Draw both sides
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: depth_stencil.clone(),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
        }));

        // Lines pipeline
        self.pipeline_lines = Some(self.device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("vector_overlay_lines_pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                buffers: &[VectorVertex::desc()],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::LineList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: depth_stencil.clone(),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
        }));

        // Points pipeline
        self.pipeline_points = Some(self.device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("vector_overlay_points_pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                buffers: &[VectorVertex::desc()],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::PointList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
        }));

        self.bind_group_layout = Some(bind_group_layout);
        self.uniform_buffer = Some(uniform_buffer);
    }
    
    /// Check if pipelines are initialized
    pub fn pipelines_ready(&self) -> bool {
        self.pipeline_triangles.is_some() && self.bind_group_layout.is_some()
    }
    
    /// Prepare bind group for rendering (call before render pass)
    /// Creates bind group with sun visibility texture for shadow integration
    pub fn prepare_bind_group(&mut self, sun_vis_view: &wgpu::TextureView) {
        let bind_group_layout = match &self.bind_group_layout {
            Some(layout) => layout,
            None => return,
        };
        
        let uniform_buffer = match &self.uniform_buffer {
            Some(buf) => buf,
            None => return,
        };
        
        // Create sampler if not already created (non-filtering for R32Float)
        if self.sampler.is_none() {
            self.sampler = Some(self.device.create_sampler(&wgpu::SamplerDescriptor {
                label: Some("vector_overlay_sampler"),
                address_mode_u: wgpu::AddressMode::ClampToEdge,
                address_mode_v: wgpu::AddressMode::ClampToEdge,
                address_mode_w: wgpu::AddressMode::ClampToEdge,
                mag_filter: wgpu::FilterMode::Nearest,
                min_filter: wgpu::FilterMode::Nearest,
                mipmap_filter: wgpu::FilterMode::Nearest,
                ..Default::default()
            }));
        }
        
        let sampler = self.sampler.as_ref().unwrap();
        
        // Create bind group with uniforms, sun visibility texture, and sampler
        self.bind_group = Some(self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("vector_overlay_bind_group"),
            layout: bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(sun_vis_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(sampler),
                },
            ],
        }));
    }
    
    /// Render a single layer (call during render pass)
    /// Returns true if something was drawn
    pub fn render_layer<'a>(
        &'a self,
        pass: &mut wgpu::RenderPass<'a>,
        layer_index: usize,
        view_proj: [[f32; 4]; 4],
        sun_dir: [f32; 3],
        lighting: [f32; 4],
    ) -> bool {
        if !self.enabled {
            return false;
        }
        
        let bind_group = match &self.bind_group {
            Some(bg) => bg,
            None => return false,
        };
        
        let uniform_buffer = match &self.uniform_buffer {
            Some(buf) => buf,
            None => return false,
        };
        
        // Get visible layers
        let visible_layers: Vec<_> = self.layers.iter().filter(|l| l.config.visible).collect();
        
        if layer_index >= visible_layers.len() {
            return false;
        }
        
        let layer = visible_layers[layer_index];
        
        // Update uniforms for this layer (with default highlight - no selection)
        let uniforms = VectorOverlayUniforms {
            view_proj,
            sun_dir: [sun_dir[0], sun_dir[1], sun_dir[2], 0.0],
            lighting,
            layer_params: [
                layer.config.opacity * self.global_opacity,
                layer.config.depth_bias,
                layer.config.line_width,
                layer.config.point_size,
            ],
            highlight_color: [1.0, 0.8, 0.0, 0.5], // Default yellow highlight
            selected_feature_id: 0, // No selection by default
            _pad: [0; 7],
        };
        
        self.queue.write_buffer(uniform_buffer, 0, bytemuck::cast_slice(&[uniforms]));
        
        // Select pipeline based on primitive type
        let pipeline = match layer.config.primitive {
            OverlayPrimitive::Triangles | OverlayPrimitive::TriangleStrip => {
                self.pipeline_triangles.as_ref()
            }
            OverlayPrimitive::Lines | OverlayPrimitive::LineStrip => {
                self.pipeline_lines.as_ref()
            }
            OverlayPrimitive::Points => {
                self.pipeline_points.as_ref()
            }
        };
        
        if let Some(pipeline) = pipeline {
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, bind_group, &[]);
            pass.set_vertex_buffer(0, layer.vertex_buffer.slice(..));
            
            if layer.index_count > 0 {
                pass.set_index_buffer(layer.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                pass.draw_indexed(0..layer.index_count, 0, 0..1);
            } else {
                pass.draw(0..layer.vertex_count, 0..1);
            }
            true
        } else {
            false
        }
    }
    
    /// Get count of visible layers
    pub fn visible_layer_count(&self) -> usize {
        self.layers.iter().filter(|l| l.config.visible).count()
    }
    
    /// Get all layers (for ID buffer rendering)
    pub fn layers(&self) -> &[VectorOverlayGpu] {
        &self.layers
    }

    /// Render a single layer with highlight support for picking
    pub fn render_layer_with_highlight<'a>(
        &'a self,
        pass: &mut wgpu::RenderPass<'a>,
        params: RenderLayerParams,
    ) -> bool {
        let RenderLayerParams {
            layer_index,
            view_proj,
            sun_dir,
            lighting,
            selected_feature_id,
            highlight_color,
        } = params;

        if !self.enabled {
            return false;
        }
        
        let bind_group = match &self.bind_group {
            Some(bg) => bg,
            None => return false,
        };
        
        let uniform_buffer = match &self.uniform_buffer {
            Some(buf) => buf,
            None => return false,
        };
        
        let visible_layers: Vec<_> = self.layers.iter().filter(|l| l.config.visible).collect();
        
        if layer_index >= visible_layers.len() {
            return false;
        }
        
        let layer = visible_layers[layer_index];
        
        let uniforms = VectorOverlayUniforms {
            view_proj,
            sun_dir: [sun_dir[0], sun_dir[1], sun_dir[2], 0.0],
            lighting,
            layer_params: [
                layer.config.opacity * self.global_opacity,
                layer.config.depth_bias,
                layer.config.line_width,
                layer.config.point_size,
            ],
            highlight_color,
            selected_feature_id,
            _pad: [0; 7],
        };
        
        self.queue.write_buffer(uniform_buffer, 0, bytemuck::cast_slice(&[uniforms]));
        
        let pipeline = match layer.config.primitive {
            OverlayPrimitive::Triangles | OverlayPrimitive::TriangleStrip => {
                self.pipeline_triangles.as_ref()
            }
            OverlayPrimitive::Lines | OverlayPrimitive::LineStrip => {
                self.pipeline_lines.as_ref()
            }
            OverlayPrimitive::Points => {
                self.pipeline_points.as_ref()
            }
        };
        
        if let Some(pipeline) = pipeline {
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, bind_group, &[]);
            pass.set_vertex_buffer(0, layer.vertex_buffer.slice(..));
            
            if layer.index_count > 0 {
                pass.set_index_buffer(layer.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                pass.draw_indexed(0..layer.index_count, 0, 0..1);
            } else {
                pass.draw(0..layer.vertex_count, 0..1);
            }
            true
        } else {
            false
        }
    }
}

// ============================================================================
// ID Buffer Rendering for Picking (Feature B)
// ============================================================================

/// Uniforms for ID buffer rendering
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct IdBufferUniforms {
    pub view_proj: [[f32; 4]; 4],
    pub depth_bias: f32,
    pub _pad: [f32; 3],
}

/// WGSL shader for ID buffer rendering
pub const ID_BUFFER_SHADER: &str = r#"
struct Uniforms {
    view_proj: mat4x4<f32>,
    depth_bias: f32,
    _pad: vec3<f32>,
};

@group(0) @binding(0) var<uniform> u: Uniforms;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) color: vec4<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) normal: vec3<f32>,
    @location(4) feature_id: u32,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) @interpolate(flat) feature_id: u32,
};

@vertex
fn vs_id(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    var pos = in.position;
    pos.y += u.depth_bias;
    out.clip_position = u.view_proj * vec4<f32>(pos, 1.0);
    out.feature_id = in.feature_id;
    return out;
}

@fragment
fn fs_id(in: VertexOutput) -> @location(0) u32 {
    return in.feature_id;
}
"#;

impl VectorOverlayStack {
    /// Initialize ID buffer pipelines for picking
    pub fn init_id_pipelines(&mut self) {
        let bind_group_layout = self.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("vector_overlay_id_bind_group_layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        let uniform_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("vector_overlay_id_uniforms"),
            size: std::mem::size_of::<IdBufferUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("vector_overlay_id_bind_group"),
            layout: &bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            }],
        });

        let shader = self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("vector_overlay_id_shader"),
            source: wgpu::ShaderSource::Wgsl(ID_BUFFER_SHADER.into()),
        });

        let pipeline_layout = self.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("vector_overlay_id_pipeline_layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let depth_stencil = Some(wgpu::DepthStencilState {
            format: wgpu::TextureFormat::Depth32Float,
            depth_write_enabled: true,
            depth_compare: wgpu::CompareFunction::Less,
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState::default(),
        });

        // Triangle pipeline for ID buffer
        self.id_pipeline_triangles = Some(self.device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("vector_overlay_id_triangles_pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_id",
                buffers: &[VectorVertex::desc()],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_id",
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::R32Uint,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: depth_stencil.clone(),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
        }));

        // Lines pipeline for ID buffer
        self.id_pipeline_lines = Some(self.device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("vector_overlay_id_lines_pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_id",
                buffers: &[VectorVertex::desc()],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_id",
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::R32Uint,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::LineList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: depth_stencil.clone(),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
        }));

        // Points pipeline for ID buffer
        self.id_pipeline_points = Some(self.device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("vector_overlay_id_points_pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_id",
                buffers: &[VectorVertex::desc()],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_id",
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::R32Uint,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::PointList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
        }));

        self.id_bind_group_layout = Some(bind_group_layout);
        self.id_uniform_buffer = Some(uniform_buffer);
        self.id_bind_group = Some(bind_group);
    }

    /// Check if ID pipelines are initialized
    pub fn id_pipelines_ready(&self) -> bool {
        self.id_pipeline_triangles.is_some() && self.id_bind_group.is_some()
    }

    /// Render all visible layers to ID buffer
    pub fn render_to_id_buffer<'a>(
        &'a self,
        pass: &mut wgpu::RenderPass<'a>,
        view_proj: [[f32; 4]; 4],
        depth_bias: f32,
    ) {
        if !self.enabled || !self.id_pipelines_ready() {
            return;
        }

        let bind_group = match &self.id_bind_group {
            Some(bg) => bg,
            None => return,
        };

        let uniform_buffer = match &self.id_uniform_buffer {
            Some(buf) => buf,
            None => return,
        };

        // Update uniforms
        let uniforms = IdBufferUniforms {
            view_proj,
            depth_bias,
            _pad: [0.0; 3],
        };
        self.queue.write_buffer(uniform_buffer, 0, bytemuck::cast_slice(&[uniforms]));

        // Render each visible layer
        for layer in self.layers.iter().filter(|l| l.config.visible) {
            let pipeline = match layer.config.primitive {
                OverlayPrimitive::Triangles | OverlayPrimitive::TriangleStrip => {
                    self.id_pipeline_triangles.as_ref()
                }
                OverlayPrimitive::Lines | OverlayPrimitive::LineStrip => {
                    self.id_pipeline_lines.as_ref()
                }
                OverlayPrimitive::Points => {
                    self.id_pipeline_points.as_ref()
                }
            };

            if let Some(pipeline) = pipeline {
                pass.set_pipeline(pipeline);
                pass.set_bind_group(0, bind_group, &[]);
                pass.set_vertex_buffer(0, layer.vertex_buffer.slice(..));

                if layer.index_count > 0 {
                    pass.set_index_buffer(layer.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                    pass.draw_indexed(0..layer.index_count, 0, 0..1);
                } else {
                    pass.draw(0..layer.vertex_count, 0..1);
                }
            }
        }
    }
}
