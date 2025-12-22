// src/viewer/terrain/dof.rs
// Depth of Field post-process pass with separable blur

use std::sync::Arc;
use wgpu::util::DeviceExt;

/// DoF uniforms for the blur shader
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct DofUniforms {
    /// Screen dimensions: width, height, 1/width, 1/height
    pub screen_dims: [f32; 4],
    /// DoF params: focus_distance, f_stop, focal_length, max_blur_radius
    pub dof_params: [f32; 4],
    /// DoF params 2: near_plane, far_plane, blur_direction (0=horiz, 1=vert), quality
    pub dof_params2: [f32; 4],
    /// Camera params: sensor_height, blur_strength, _, _
    pub camera_params: [f32; 4],
}

/// Depth of Field configuration
#[derive(Debug, Clone)]
pub struct DofConfig {
    pub enabled: bool,
    pub focus_distance: f32,
    pub f_stop: f32,
    pub focal_length: f32,
    pub quality: u32,  // 4, 8, or 16 samples
    pub max_blur_radius: f32,
    pub blur_strength: f32,  // Artistic multiplier (1.0 = physical, higher = more blur)
}

impl Default for DofConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            focus_distance: 500.0,
            f_stop: 5.6,
            focal_length: 50.0,
            quality: 8,
            max_blur_radius: 16.0,
            blur_strength: 1000.0,  // High default for visible effect at landscape distances
        }
    }
}

/// Depth of Field pass manager with two-pass separable blur
pub struct DofPass {
    device: Arc<wgpu::Device>,
    pipeline: wgpu::RenderPipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
    uniform_buffer: wgpu::Buffer,
    // Input texture (scene renders here)
    input_texture: Option<wgpu::Texture>,
    pub input_view: Option<wgpu::TextureView>,
    // Intermediate texture for horizontal pass output
    intermediate_texture: Option<wgpu::Texture>,
    pub intermediate_view: Option<wgpu::TextureView>,
    current_size: (u32, u32),
}

impl DofPass {
    pub fn new(device: Arc<wgpu::Device>, surface_format: wgpu::TextureFormat) -> Self {
        // Create bind group layout
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("dof.bind_group_layout"),
            entries: &[
                // Color texture
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // Depth texture
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Depth,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // Color sampler (filtering)
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                // Uniforms
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        // Create pipeline layout
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("dof.pipeline_layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        // Create shader module
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("dof.shader"),
            source: wgpu::ShaderSource::Wgsl(DOF_SHADER.into()),
        });

        // Create render pipeline
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("dof.pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                unclipped_depth: false,
                polygon_mode: wgpu::PolygonMode::Fill,
                conservative: false,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
        });

        // Create color sampler (filtering)
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("dof.color_sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        // Create uniform buffer
        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("dof.uniforms"),
            contents: bytemuck::cast_slice(&[DofUniforms {
                screen_dims: [1.0, 1.0, 1.0, 1.0],
                dof_params: [500.0, 5.6, 50.0, 16.0],
                dof_params2: [1.0, 10000.0, 0.0, 8.0],
                camera_params: [24.0, 0.0, 0.0, 0.0],
            }]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        Self {
            device,
            pipeline,
            bind_group_layout,
            sampler,
            uniform_buffer,
            input_texture: None,
            input_view: None,
            intermediate_texture: None,
            intermediate_view: None,
            current_size: (0, 0),
        }
    }

    /// Ensure textures are allocated
    fn ensure_textures(&mut self, width: u32, height: u32, format: wgpu::TextureFormat) {
        if self.current_size != (width, height) || self.input_texture.is_none() {
            // Input texture (scene renders here)
            let input_tex = self.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("dof.input"),
                size: wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            });
            self.input_view = Some(input_tex.create_view(&wgpu::TextureViewDescriptor::default()));
            self.input_texture = Some(input_tex);
            
            // Intermediate texture for horizontal pass output
            let tex = self.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("dof.intermediate"),
                size: wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            });
            self.intermediate_view = Some(tex.create_view(&wgpu::TextureViewDescriptor::default()));
            self.intermediate_texture = Some(tex);
            self.current_size = (width, height);
        }
    }

    /// Get the input texture view (where scene should render to)
    pub fn get_input_view(&mut self, width: u32, height: u32, format: wgpu::TextureFormat) -> &wgpu::TextureView {
        self.ensure_textures(width, height, format);
        self.input_view.as_ref().unwrap()
    }

    /// Apply DoF effect using internal input texture (two-pass separable blur)
    pub fn apply_from_input(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        queue: &wgpu::Queue,
        depth_view: &wgpu::TextureView,
        output_view: &wgpu::TextureView,
        width: u32,
        height: u32,
        format: wgpu::TextureFormat,
        config: &DofConfig,
        near_plane: f32,
        far_plane: f32,
    ) {
        self.ensure_textures(width, height, format);
        
        let input_view = self.input_view.as_ref().unwrap();
        let intermediate_view = self.intermediate_view.as_ref().unwrap();

        // Pass 1: Horizontal blur from input -> intermediate
        self.render_pass(
            encoder,
            queue,
            input_view,
            depth_view,
            intermediate_view,
            width,
            height,
            config,
            near_plane,
            far_plane,
            0.0, // horizontal
        );

        // Pass 2: Vertical blur from intermediate -> output
        self.render_pass(
            encoder,
            queue,
            intermediate_view,
            depth_view,
            output_view,
            width,
            height,
            config,
            near_plane,
            far_plane,
            1.0, // vertical
        );
    }

    /// Apply DoF effect (two-pass separable blur) with external input
    #[allow(dead_code)]
    pub fn apply(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        queue: &wgpu::Queue,
        color_view: &wgpu::TextureView,
        depth_view: &wgpu::TextureView,
        output_view: &wgpu::TextureView,
        width: u32,
        height: u32,
        format: wgpu::TextureFormat,
        config: &DofConfig,
        near_plane: f32,
        far_plane: f32,
    ) {
        self.ensure_textures(width, height, format);

        // Pass 1: Horizontal blur from color_view -> intermediate
        self.render_pass(
            encoder,
            queue,
            color_view,
            depth_view,
            self.intermediate_view.as_ref().unwrap(),
            width,
            height,
            config,
            near_plane,
            far_plane,
            0.0, // horizontal
        );

        // Pass 2: Vertical blur from intermediate -> output
        self.render_pass(
            encoder,
            queue,
            self.intermediate_view.as_ref().unwrap(),
            depth_view,
            output_view,
            width,
            height,
            config,
            near_plane,
            far_plane,
            1.0, // vertical
        );
    }

    fn render_pass(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        queue: &wgpu::Queue,
        input_view: &wgpu::TextureView,
        depth_view: &wgpu::TextureView,
        output_view: &wgpu::TextureView,
        width: u32,
        height: u32,
        config: &DofConfig,
        near_plane: f32,
        far_plane: f32,
        blur_direction: f32,
    ) {
        // Update uniforms
        let uniforms = DofUniforms {
            screen_dims: [width as f32, height as f32, 1.0 / width as f32, 1.0 / height as f32],
            dof_params: [config.focus_distance, config.f_stop, config.focal_length, config.max_blur_radius],
            dof_params2: [near_plane, far_plane, blur_direction, config.quality as f32],
            camera_params: [24.0, config.blur_strength, 0.0, 0.0], // 24mm sensor height, blur_strength
        };
        queue.write_buffer(&self.uniform_buffer, 0, bytemuck::cast_slice(&[uniforms]));

        // Create bind group
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("dof.bind_group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(input_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(depth_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.uniform_buffer.as_entire_binding(),
                },
            ],
        });

        // Render
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("dof.render_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: output_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.draw(0..3, 0..1);
        }
    }
}

/// DoF shader with CoC calculation and separable Gaussian blur
const DOF_SHADER: &str = r#"
// Depth of Field shader with separable blur

struct Uniforms {
    screen_dims: vec4<f32>,    // width, height, 1/width, 1/height
    dof_params: vec4<f32>,     // focus_distance, f_stop, focal_length, max_blur_radius
    dof_params2: vec4<f32>,    // near_plane, far_plane, blur_direction (0=h, 1=v), quality
    camera_params: vec4<f32>,  // sensor_height, blur_strength, _, _
};

@group(0) @binding(0) var color_tex: texture_2d<f32>;
@group(0) @binding(1) var depth_tex: texture_depth_2d;
@group(0) @binding(2) var samp: sampler;
@group(0) @binding(3) var<uniform> u: Uniforms;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

// Full-screen triangle vertex shader
@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    let x = f32(i32(vertex_index & 1u) * 4 - 1);
    let y = f32(i32(vertex_index >> 1u) * 4 - 1);
    
    var out: VertexOutput;
    out.position = vec4<f32>(x, y, 0.0, 1.0);
    out.uv = vec2<f32>((x + 1.0) * 0.5, (1.0 - y) * 0.5);
    return out;
}

// Linearize depth from depth buffer (glam::perspective_rh uses standard Z: depth=0 at near, depth=1 at far)
fn linearize_depth(depth: f32, near: f32, far: f32) -> f32 {
    // Standard Z: depth=0 at near plane, depth=1 at far plane
    // z = near * far / (far - depth * (far - near))
    // Verify: depth=0 → z = near*far/far = near ✓
    // Verify: depth=1 → z = near*far/(far-far+near) = near*far/near = far ✓
    return near * far / (far - depth * (far - near));
}

// Calculate Circle of Confusion (CoC) diameter in pixels
// All inputs in consistent units (meters for distances, mm for optics)
fn calculate_coc(linear_depth: f32, focus_dist: f32, focal_length_mm: f32, f_stop: f32, sensor_height_mm: f32) -> f32 {
    // Convert focal length and sensor to meters for consistent units with depth
    let focal_length = focal_length_mm / 1000.0;
    let sensor_height = sensor_height_mm / 1000.0;
    
    // Aperture diameter in meters
    let aperture = focal_length / f_stop;
    
    // Hyperfocal distance (where everything from H/2 to infinity is acceptably sharp)
    // For thin lens: H = f^2 / (N * c) where c is circle of confusion limit
    // We'll use a simplified approach: magnification at focus distance
    
    // Subject magnification at focus distance
    let magnification = focal_length / max(focus_dist - focal_length, 0.001);
    
    // CoC on sensor (in meters)
    // CoC = |A * m * (d - s) / d| where A=aperture, m=magnification, d=subject dist, s=focus dist
    let depth_diff = abs(linear_depth - focus_dist);
    let coc_sensor = aperture * magnification * depth_diff / max(linear_depth, 0.001);
    
    // Convert to screen pixels
    let screen_height = u.screen_dims.y;
    let coc_pixels = coc_sensor * screen_height / sensor_height;
    
    return coc_pixels;
}

// Gaussian weight
fn gaussian_weight(offset: f32, sigma: f32) -> f32 {
    return exp(-0.5 * offset * offset / (sigma * sigma));
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let focus_dist = u.dof_params.x;
    let f_stop = u.dof_params.y;
    let focal_length = u.dof_params.z;
    let max_blur = u.dof_params.w;
    let near = u.dof_params2.x;
    let far = u.dof_params2.y;
    let direction = u.dof_params2.z;  // 0 = horizontal, 1 = vertical
    let quality = u.dof_params2.w;
    let sensor_height = u.camera_params.x;
    let blur_strength = u.camera_params.y;  // Artistic multiplier for landscape DoF
    
    // Sample depth at center (depth textures are read with textureLoad)
    let w = max(i32(u.screen_dims.x) - 1, 0);
    let h = max(i32(u.screen_dims.y) - 1, 0);
    let xy = vec2<i32>(
        clamp(i32(in.uv.x * u.screen_dims.x), 0, w),
        clamp(i32(in.uv.y * u.screen_dims.y), 0, h),
    );
    let depth_sample = textureLoad(depth_tex, xy, 0);
    let linear_depth = linearize_depth(depth_sample, near, far);
    
    // Simple depth-based blur: blur amount based on distance from focus plane
    // Normalized depth difference from focus distance, scaled by blur_strength and f_stop
    let depth_diff = abs(linear_depth - focus_dist);
    let norm_blur = depth_diff / max(far - near, 1.0);  // 0-1 range
    let aperture_factor = 16.0 / max(f_stop, 1.4);  // f/2.8 = 5.7x, f/16 = 1x
    let coc_raw = norm_blur * blur_strength * aperture_factor * 0.1;
    
    // Ensure minimum blur of 1.0 pixel for testing, then clamp to max
    let coc = clamp(coc_raw, 1.0, max_blur);
    
    // Blur direction vector
    var blur_dir: vec2<f32>;
    if direction < 0.5 {
        blur_dir = vec2<f32>(u.screen_dims.z, 0.0);  // horizontal
    } else {
        blur_dir = vec2<f32>(0.0, u.screen_dims.w);  // vertical
    }
    
    // Number of samples based on quality
    let num_samples = i32(quality);
    let sigma = coc / 3.0;  // Gaussian sigma from blur radius
    
    var color_sum = vec3<f32>(0.0);
    var weight_sum = 0.0;
    
    // Sample along blur direction
    for (var i = -num_samples; i <= num_samples; i++) {
        let offset = f32(i);
        let sample_uv = in.uv + blur_dir * offset * (coc / f32(num_samples));
        
        // Clamp to valid UV range
        let clamped_uv = clamp(sample_uv, vec2<f32>(0.001), vec2<f32>(0.999));
        
        // Sample color
        let sample_color = textureSample(color_tex, samp, clamped_uv).rgb;
        
        // Gaussian weight
        let weight = gaussian_weight(offset, sigma);
        
        color_sum += sample_color * weight;
        weight_sum += weight;
    }
    
    // Normalize
    let final_color = color_sum / max(weight_sum, 0.001);
    
    return vec4<f32>(final_color, 1.0);
}
"#;
