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
    /// Camera params: sensor_height, blur_strength, tilt_pitch, tilt_yaw
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
    pub tilt_pitch: f32,     // Tilt-shift pitch in radians (Scheimpflug effect)
    pub tilt_yaw: f32,       // Tilt-shift yaw in radians
}

impl Default for DofConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            focus_distance: 500.0,
            f_stop: 5.6,
            focal_length: 50.0,
            quality: 8,
            max_blur_radius: 32.0,
            blur_strength: 500.0,  // Landscape scale multiplier (physical CoC is tiny at 100s of meters)
            tilt_pitch: 0.0,
            tilt_yaw: 0.0,
        }
    }
}

/// Depth of Field pass manager with two-pass separable blur
pub struct DofPass {
    device: Arc<wgpu::Device>,
    pipeline: wgpu::RenderPipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
    uniform_buffer_h: wgpu::Buffer,  // Horizontal pass uniforms
    uniform_buffer_v: wgpu::Buffer,  // Vertical pass uniforms
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

        // Create uniform buffers (separate for each pass to avoid race conditions)
        let uniform_buffer_h = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("dof.uniforms_h"),
            contents: bytemuck::cast_slice(&[DofUniforms {
                screen_dims: [1.0, 1.0, 1.0, 1.0],
                dof_params: [500.0, 5.6, 50.0, 16.0],
                dof_params2: [1.0, 10000.0, 0.0, 8.0],  // direction=0 for horizontal
                camera_params: [24.0, 500.0, 0.0, 0.0],
            }]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        
        let uniform_buffer_v = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("dof.uniforms_v"),
            contents: bytemuck::cast_slice(&[DofUniforms {
                screen_dims: [1.0, 1.0, 1.0, 1.0],
                dof_params: [500.0, 5.6, 50.0, 16.0],
                dof_params2: [1.0, 10000.0, 1.0, 8.0],  // direction=1 for vertical
                camera_params: [24.0, 500.0, 0.0, 0.0],
            }]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        Self {
            device,
            pipeline,
            bind_group_layout,
            sampler,
            uniform_buffer_h,
            uniform_buffer_v,
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

        // Update uniforms for horizontal pass
        let uniforms_h = DofUniforms {
            screen_dims: [width as f32, height as f32, 1.0 / width as f32, 1.0 / height as f32],
            dof_params: [config.focus_distance, config.f_stop, config.focal_length, config.max_blur_radius],
            dof_params2: [near_plane, far_plane, 0.0, config.quality as f32],  // direction=0 horizontal
            camera_params: [24.0, config.blur_strength, config.tilt_pitch, config.tilt_yaw],
        };
        queue.write_buffer(&self.uniform_buffer_h, 0, bytemuck::cast_slice(&[uniforms_h]));

        // Update uniforms for vertical pass
        let uniforms_v = DofUniforms {
            screen_dims: [width as f32, height as f32, 1.0 / width as f32, 1.0 / height as f32],
            dof_params: [config.focus_distance, config.f_stop, config.focal_length, config.max_blur_radius],
            dof_params2: [near_plane, far_plane, 1.0, config.quality as f32],  // direction=1 vertical
            camera_params: [24.0, config.blur_strength, config.tilt_pitch, config.tilt_yaw],
        };
        queue.write_buffer(&self.uniform_buffer_v, 0, bytemuck::cast_slice(&[uniforms_v]));

        // Pass 1: Horizontal blur from input -> intermediate
        self.render_pass_with_buffer(
            encoder,
            input_view,
            depth_view,
            intermediate_view,
            &self.uniform_buffer_h,
        );

        // Pass 2: Vertical blur from intermediate -> output
        self.render_pass_with_buffer(
            encoder,
            intermediate_view,
            depth_view,
            output_view,
            &self.uniform_buffer_v,
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

        // Update uniforms for horizontal pass
        let uniforms_h = DofUniforms {
            screen_dims: [width as f32, height as f32, 1.0 / width as f32, 1.0 / height as f32],
            dof_params: [config.focus_distance, config.f_stop, config.focal_length, config.max_blur_radius],
            dof_params2: [near_plane, far_plane, 0.0, config.quality as f32],
            camera_params: [24.0, config.blur_strength, config.tilt_pitch, config.tilt_yaw],
        };
        queue.write_buffer(&self.uniform_buffer_h, 0, bytemuck::cast_slice(&[uniforms_h]));

        // Update uniforms for vertical pass
        let uniforms_v = DofUniforms {
            screen_dims: [width as f32, height as f32, 1.0 / width as f32, 1.0 / height as f32],
            dof_params: [config.focus_distance, config.f_stop, config.focal_length, config.max_blur_radius],
            dof_params2: [near_plane, far_plane, 1.0, config.quality as f32],
            camera_params: [24.0, config.blur_strength, config.tilt_pitch, config.tilt_yaw],
        };
        queue.write_buffer(&self.uniform_buffer_v, 0, bytemuck::cast_slice(&[uniforms_v]));

        // Pass 1: Horizontal blur from color_view -> intermediate
        self.render_pass_with_buffer(
            encoder,
            color_view,
            depth_view,
            self.intermediate_view.as_ref().unwrap(),
            &self.uniform_buffer_h,
        );

        // Pass 2: Vertical blur from intermediate -> output
        self.render_pass_with_buffer(
            encoder,
            self.intermediate_view.as_ref().unwrap(),
            depth_view,
            output_view,
            &self.uniform_buffer_v,
        );
    }

    fn render_pass_with_buffer(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        input_view: &wgpu::TextureView,
        depth_view: &wgpu::TextureView,
        output_view: &wgpu::TextureView,
        uniform_buffer: &wgpu::Buffer,
    ) {
        // Create bind group with the specified uniform buffer
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
                    resource: uniform_buffer.as_entire_binding(),
                },
            ],
        });

        // Render - use Clear for output to avoid undefined content
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("dof.render_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: output_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
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
// Depth of Field shader with separable blur and tilt-shift support

struct Uniforms {
    screen_dims: vec4<f32>,    // width, height, 1/width, 1/height
    dof_params: vec4<f32>,     // focus_distance, f_stop, focal_length, max_blur_radius
    dof_params2: vec4<f32>,    // near_plane, far_plane, blur_direction (0=h, 1=v), quality
    camera_params: vec4<f32>,  // sensor_height, blur_strength, tilt_pitch, tilt_yaw
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

// Linearize depth from depth buffer
fn linearize_depth(depth: f32, near: f32, far: f32) -> f32 {
    // Standard-Z (glam::perspective_rh): depth=0 at near, depth=1 at far
    return near * far / (far - depth * (far - near));
}

// Calculate screen-space blur for tilt-shift miniature effect
// Returns a blur multiplier based on distance from the tilted focus band
fn calculate_tilt_blur_factor(uv: vec2<f32>, tilt_pitch: f32, tilt_yaw: f32) -> f32 {
    // Convert UV to centered coordinates [-1, 1]
    let centered = (uv - 0.5) * 2.0;
    
    // Calculate signed distance from the tilted focus plane in screen space
    // tilt_pitch tilts the focus band (positive = top sharp, bottom blurry)
    // tilt_yaw rotates the band (positive = left sharp, right blurry)
    // For classic miniature: pitch tilts vertically, creating horizontal sharp band
    let pitch_contrib = centered.y * sin(tilt_pitch);
    let yaw_contrib = centered.x * sin(tilt_yaw);
    
    // Distance from the focus band (0 = on band, 1 = far from band)
    let distance_from_band = abs(pitch_contrib + yaw_contrib);
    
    // Smooth falloff from focus band
    return smoothstep(0.0, 1.0, distance_from_band);
}

// Calculate effective focus distance with tilt-shift (Scheimpflug principle)
fn calculate_tilted_focus_distance(uv: vec2<f32>, base_focus: f32, tilt_pitch: f32, tilt_yaw: f32) -> f32 {
    // Convert UV to centered coordinates [-1, 1]
    let centered = (uv - 0.5) * 2.0;
    
    // Calculate tilt offset: pitch affects Y, yaw affects X
    let tilt_offset = centered.y * tan(tilt_pitch) + centered.x * tan(tilt_yaw);
    
    // Scale by focus distance for realistic plane tilt
    let focus_variation = base_focus * tilt_offset * 0.5;
    
    return max(base_focus + focus_variation, 1.0);
}

// Calculate Circle of Confusion (CoC) in pixels using thin lens model
fn calculate_coc(linear_depth: f32, focus_dist: f32, focal_length_mm: f32, f_stop: f32, sensor_height_mm: f32) -> f32 {
    // Convert to meters
    let focal_length = focal_length_mm / 1000.0;
    let sensor_height = sensor_height_mm / 1000.0;
    
    // Aperture diameter
    let aperture = focal_length / max(f_stop, 1.4);
    
    // Signed distance from focus plane (positive = behind focus, negative = in front)
    let signed_depth_diff = linear_depth - focus_dist;
    
    // Thin lens CoC formula:
    // CoC = |aperture * focal_length * (depth - focus) / (depth * (focus - focal_length))|
    let denominator = linear_depth * max(focus_dist - focal_length, 0.001);
    let coc_sensor = abs(aperture * focal_length * signed_depth_diff / denominator);
    
    // Convert sensor CoC to screen pixels
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
    let base_focus_dist = u.dof_params.x;
    let f_stop = u.dof_params.y;
    let focal_length = u.dof_params.z;
    let max_blur = u.dof_params.w;
    let near = u.dof_params2.x;
    let far = u.dof_params2.y;
    let direction = u.dof_params2.z;
    let quality = u.dof_params2.w;
    let sensor_height = u.camera_params.x;
    let blur_strength = u.camera_params.y;
    let tilt_pitch = u.camera_params.z;
    let tilt_yaw = u.camera_params.w;
    
    // Sample depth at current pixel
    let w = max(i32(u.screen_dims.x) - 1, 0);
    let h = max(i32(u.screen_dims.y) - 1, 0);
    let xy = vec2<i32>(
        clamp(i32(in.uv.x * u.screen_dims.x), 0, w),
        clamp(i32(in.uv.y * u.screen_dims.y), 0, h),
    );
    let depth_sample = textureLoad(depth_tex, xy, 0);
    let linear_depth = linearize_depth(depth_sample, near, far);
    
    // Calculate effective focus distance (with tilt-shift if enabled)
    var focus_dist = base_focus_dist;
    let has_tilt = abs(tilt_pitch) > 0.001 || abs(tilt_yaw) > 0.001;
    if has_tilt {
        focus_dist = calculate_tilted_focus_distance(in.uv, base_focus_dist, tilt_pitch, tilt_yaw);
    }
    
    // Calculate CoC using thin lens model
    var coc = calculate_coc(linear_depth, focus_dist, focal_length, f_stop, sensor_height);
    
    // Apply blur strength multiplier for landscape scale
    coc = coc * blur_strength;
    
    // For tilt-shift: use screen-space blur for miniature effect
    // This creates the classic "fake miniature" look with a sharp horizontal band
    if has_tilt {
        let tilt_blur = calculate_tilt_blur_factor(in.uv, tilt_pitch, tilt_yaw);
        // Tilt blur OVERRIDES physical CoC to create proper miniature effect
        // tilt_blur = 0 at focus band (sharp), 1 at edges (full blur)
        // Scale to max_blur for full effect
        coc = tilt_blur * max_blur;
    }
    
    // Clamp to max blur radius
    coc = clamp(coc, 0.0, max_blur);
    
    // Get original color
    let original_color = textureSample(color_tex, samp, in.uv).rgb;
    
    // If CoC is very small, return original (no blur needed)
    if coc < 0.5 {
        return vec4<f32>(original_color, 1.0);
    }
    
    // Blur direction vector
    var blur_dir: vec2<f32>;
    if direction < 0.5 {
        blur_dir = vec2<f32>(u.screen_dims.z, 0.0);  // horizontal (1/width, 0)
    } else {
        blur_dir = vec2<f32>(0.0, u.screen_dims.w);  // vertical (0, 1/height)
    }
    
    // Number of samples based on quality
    let num_samples = i32(quality);
    let sigma = max(coc / 2.5, 1.0);
    
    var color_sum = vec3<f32>(0.0);
    var weight_sum = 0.0;
    
    // Gaussian blur along direction
    for (var i = -num_samples; i <= num_samples; i++) {
        let offset = f32(i);
        let sample_uv = in.uv + blur_dir * offset * (coc / f32(num_samples));
        
        // Clamp to valid UV range
        let clamped_uv = clamp(sample_uv, vec2<f32>(0.0), vec2<f32>(1.0));
        
        // Sample color
        let sample_color = textureSample(color_tex, samp, clamped_uv).rgb;
        
        // Gaussian weight only (no bilateral - it causes issues with separable blur)
        let weight = gaussian_weight(offset, sigma);
        
        color_sum += sample_color * weight;
        weight_sum += weight;
    }
    
    // Normalize
    let blurred_color = color_sum / max(weight_sum, 0.001);
    
    // Blend based on CoC strength
    let blend = smoothstep(0.5, 3.0, coc);
    let final_color = mix(original_color, blurred_color, blend);
    
    return vec4<f32>(final_color, 1.0);
}
"#;
