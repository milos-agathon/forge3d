//! P5: Screen-space effects system (SSAO/GTAO, SSGI, SSR)
//!
//! Provides GPU-accelerated screen-space techniques for ambient occlusion,
//! global illumination, and reflections.

use crate::core::gbuffer::{GBuffer, GBufferConfig};
use crate::error::RenderResult;
use wgpu::*;
use wgpu::util::DeviceExt;

/// Screen-space effect type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScreenSpaceEffect {
    /// Screen-Space Ambient Occlusion / Ground-Truth Ambient Occlusion
    SSAO,
    /// Screen-Space Global Illumination
    SSGI,
    /// Screen-Space Reflections
    SSR,
}

/// SSAO/GTAO settings
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SsaoSettings {
    pub radius: f32,
    pub intensity: f32,
    pub bias: f32,
    pub num_samples: u32,
    pub technique: u32, // 0=SSAO, 1=GTAO
    pub inv_resolution: [f32; 2],
    pub _pad0: [f32; 2],
}

impl Default for SsaoSettings {
    fn default() -> Self {
        Self {
            radius: 0.5,
            intensity: 1.0,
            bias: 0.025,
            num_samples: 16,
            technique: 0,
            inv_resolution: [1.0 / 1920.0, 1.0 / 1080.0],
            _pad0: [0.0; 2],
        }
    }
}

/// SSGI settings
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SsgiSettings {
    pub radius: f32,
    pub intensity: f32,
    pub num_steps: u32,
    pub step_size: f32,
    pub inv_resolution: [f32; 2],
    pub temporal_alpha: f32,
    pub use_half_res: u32,
    pub upsample_depth_sigma: f32,
    pub upsample_normal_exp: f32,
    pub use_edge_aware: u32,
    pub _pad1: u32,
}

impl Default for SsgiSettings {
    fn default() -> Self {
        Self {
            radius: 1.0,
            intensity: 0.5,
            num_steps: 16,
            step_size: 0.1,
            inv_resolution: [1.0 / 1920.0, 1.0 / 1080.0],
            temporal_alpha: 0.1,
            use_half_res: 0,
            // Depth sigma ~ 0.02 approximates the previous exp(-dz*50.0)
            upsample_depth_sigma: 0.02,
            // Normal exponent steers how strongly we preserve edges
            upsample_normal_exp: 8.0,
            // Enable edge-aware upsample by default
            use_edge_aware: 1,
            _pad1: 0,
        }
    }
}

/// SSR settings
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SsrSettings {
    pub max_steps: u32,
    pub thickness: f32,
    pub max_distance: f32,
    pub intensity: f32,
    pub inv_resolution: [f32; 2],
    pub _pad0: [f32; 2],
}

impl Default for SsrSettings {
    fn default() -> Self {
        Self {
            max_steps: 32,
            thickness: 0.1,
            max_distance: 10.0,
            intensity: 1.0,
            inv_resolution: [1.0 / 1920.0, 1.0 / 1080.0],
            _pad0: [0.0; 2],
        }
    }
}

/// Camera parameters for screen-space effects
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CameraParams {
    pub view_matrix: [[f32; 4]; 4],
    pub inv_view_matrix: [[f32; 4]; 4],
    pub proj_matrix: [[f32; 4]; 4],
    pub inv_proj_matrix: [[f32; 4]; 4],
    pub camera_pos: [f32; 3],
    pub _pad: f32,
}

impl Default for CameraParams {
    fn default() -> Self {
        let identity = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];
        Self {
            view_matrix: identity,
            inv_view_matrix: identity,
            proj_matrix: identity,
            inv_proj_matrix: identity,
            camera_pos: [0.0, 0.0, 0.0],
            _pad: 0.0,
        }
    }
}

/// SSAO renderer
pub struct SsaoRenderer {
    settings: SsaoSettings,
    settings_buffer: Buffer,
    camera_buffer: Buffer,
    
    // SSAO compute pipeline
    ssao_pipeline: ComputePipeline,
    ssao_bind_group_layout: BindGroupLayout,
    
    // Blur pipeline
    blur_pipeline: ComputePipeline,
    blur_bind_group_layout: BindGroupLayout,
    
    // Composite pipeline
    composite_pipeline: ComputePipeline,
    composite_bind_group_layout: BindGroupLayout,
    comp_uniform: Buffer,
    
    // Output textures
    ssao_texture: Texture,
    ssao_view: TextureView,
    ssao_blurred: Texture,
    ssao_blurred_view: TextureView,
    // Color composited with AO for display
    ssao_composited: Texture,
    ssao_composited_view: TextureView,
    
    width: u32,
    height: u32,
}

impl SsaoRenderer {
    pub fn new(device: &Device, width: u32, height: u32) -> RenderResult<Self> {
        let mut settings = SsaoSettings::default();
        settings.inv_resolution = [1.0 / width as f32, 1.0 / height as f32];
        
        // Create uniform buffers
        let settings_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("ssao_settings"),
            size: std::mem::size_of::<SsaoSettings>() as u64,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        
        let camera_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("ssao_camera"),
            size: std::mem::size_of::<CameraParams>() as u64,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        
        // Load shaders
        let shader = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("ssao_shader"),
            source: ShaderSource::Wgsl(include_str!("../../shaders/ssao.wgsl").into()),
        });
        
        // Create bind group layouts
        let ssao_bind_group_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("ssao_bind_group_layout"),
            entries: &[
                // Depth texture
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Texture {
                        sample_type: TextureSampleType::Float { filterable: false },
                        view_dimension: TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // Normal texture
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Texture {
                        sample_type: TextureSampleType::Float { filterable: false },
                        view_dimension: TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // Output
                BindGroupLayoutEntry {
                    binding: 2,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::StorageTexture {
                        access: StorageTextureAccess::WriteOnly,
                        format: TextureFormat::R16Float,
                        view_dimension: TextureViewDimension::D2,
                    },
                    count: None,
                },
                // Settings
                BindGroupLayoutEntry {
                    binding: 3,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Camera
                BindGroupLayoutEntry {
                    binding: 4,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        // Note: SSGI upsample bind group layout is defined in SsgiRenderer::new
        
        let blur_bind_group_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("ssao_blur_bind_group_layout"),
            entries: &[
                // SSAO input
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Texture {
                        sample_type: TextureSampleType::Float { filterable: false },
                        view_dimension: TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // Depth for bilateral
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Texture {
                        sample_type: TextureSampleType::Float { filterable: false },
                        view_dimension: TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // Output
                BindGroupLayoutEntry {
                    binding: 2,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::StorageTexture {
                        access: StorageTextureAccess::WriteOnly,
                        format: TextureFormat::R16Float,
                        view_dimension: TextureViewDimension::D2,
                    },
                    count: None,
                },
                // Settings
                BindGroupLayoutEntry {
                    binding: 3,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        
        let composite_bind_group_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("ssao_composite_bind_group_layout"),
            entries: &[
                // Color input
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Texture {
                        sample_type: TextureSampleType::Float { filterable: false },
                        view_dimension: TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // Output
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::StorageTexture {
                        access: StorageTextureAccess::WriteOnly,
                        format: TextureFormat::Rgba8Unorm,
                        view_dimension: TextureViewDimension::D2,
                    },
                    count: None,
                },
                // SSAO texture
                BindGroupLayoutEntry {
                    binding: 2,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Texture {
                        sample_type: TextureSampleType::Float { filterable: false },
                        view_dimension: TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // Composite params (multiplier)
                BindGroupLayoutEntry {
                    binding: 3,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        
        // Create pipelines
        let ssao_pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("ssao_pipeline"),
            layout: Some(&device.create_pipeline_layout(&PipelineLayoutDescriptor {
                label: Some("ssao_pipeline_layout"),
                bind_group_layouts: &[&ssao_bind_group_layout],
                push_constant_ranges: &[],
            })),
            module: &shader,
            entry_point: "cs_ssao",
        });
        
        let blur_pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("ssao_blur_pipeline"),
            layout: Some(&device.create_pipeline_layout(&PipelineLayoutDescriptor {
                label: Some("ssao_blur_pipeline_layout"),
                bind_group_layouts: &[&blur_bind_group_layout],
                push_constant_ranges: &[],
            })),
            module: &shader,
            entry_point: "cs_ssao_blur",
        });
        
        let composite_pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("ssao_composite_pipeline"),
            layout: Some(&device.create_pipeline_layout(&PipelineLayoutDescriptor {
                label: Some("ssao_composite_pipeline_layout"),
                bind_group_layouts: &[&composite_bind_group_layout],
                push_constant_ranges: &[],
            })),
            module: &shader,
            entry_point: "cs_ssao_composite",
        });
        
        // Create output textures
        let ssao_texture = device.create_texture(&TextureDescriptor {
            label: Some("ssao_texture"),
            size: Extent3d { width, height, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::R16Float,
            usage: TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let ssao_view = ssao_texture.create_view(&TextureViewDescriptor::default());
        
        let ssao_blurred = device.create_texture(&TextureDescriptor {
            label: Some("ssao_blurred"),
            size: Extent3d { width, height, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::R16Float,
            usage: TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let ssao_blurred_view = ssao_blurred.create_view(&TextureViewDescriptor::default());

        // Composited color (material * AO)
        let ssao_composited = device.create_texture(&TextureDescriptor {
            label: Some("ssao_composited"),
            size: Extent3d { width, height, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Rgba8Unorm,
            usage: TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let ssao_composited_view = ssao_composited.create_view(&TextureViewDescriptor::default());
        
        // Composite uniform (x=multiplier, yzw reserved)
        let comp_params: [f32; 4] = [1.0, 0.0, 0.0, 0.0];
        let comp_uniform = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("ssao.comp.uniform"),
            contents: bytemuck::cast_slice(&comp_params),
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        });

        Ok(Self {
            settings,
            settings_buffer,
            camera_buffer,
            ssao_pipeline,
            ssao_bind_group_layout,
            blur_pipeline,
            blur_bind_group_layout,
            composite_pipeline,
            composite_bind_group_layout,
            comp_uniform,
            ssao_texture,
            ssao_view,
            ssao_blurred,
            ssao_blurred_view,
            ssao_composited,
            ssao_composited_view,
            width,
            height,
        })
    }
    
    /// Update settings
    pub fn update_settings(&mut self, queue: &Queue, settings: SsaoSettings) {
        self.settings = settings;
        queue.write_buffer(&self.settings_buffer, 0, bytemuck::bytes_of(&settings));
    }

    pub fn get_settings(&self) -> SsaoSettings { self.settings }
    
    /// Update camera parameters
    pub fn update_camera(&mut self, queue: &Queue, camera: &CameraParams) {
        queue.write_buffer(&self.camera_buffer, 0, bytemuck::bytes_of(camera));
    }
    
    /// Execute SSAO pass
    pub fn execute(
        &self,
        device: &Device,
        encoder: &mut CommandEncoder,
        gbuffer: &GBuffer,
    ) -> RenderResult<()> {
        // Create bind group
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("ssao_bind_group"),
            layout: &self.ssao_bind_group_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::TextureView(&gbuffer.depth_view),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::TextureView(&gbuffer.normal_view),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: BindingResource::TextureView(&self.ssao_view),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: self.settings_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 4,
                    resource: self.camera_buffer.as_entire_binding(),
                },
            ],
        });
        
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("ssao_pass"),
            timestamp_writes: None,
        });
        
        pass.set_pipeline(&self.ssao_pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        
        let workgroup_x = (self.width + 7) / 8;
        let workgroup_y = (self.height + 7) / 8;
        pass.dispatch_workgroups(workgroup_x, workgroup_y, 1);
        
        drop(pass);
        
        // Blur pass
        let blur_bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("ssao_blur_bind_group"),
            layout: &self.blur_bind_group_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::TextureView(&self.ssao_view),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::TextureView(&gbuffer.depth_view),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: BindingResource::TextureView(&self.ssao_blurred_view),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: self.settings_buffer.as_entire_binding(),
                },
            ],
        });
        
        let mut blur_pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("ssao_blur_pass"),
            timestamp_writes: None,
        });
        
        blur_pass.set_pipeline(&self.blur_pipeline);
        blur_pass.set_bind_group(0, &blur_bind_group, &[]);
        blur_pass.dispatch_workgroups(workgroup_x, workgroup_y, 1);
        drop(blur_pass);

        // Composite pass: material * AO -> composited RGBA8
        let comp_bg = device.create_bind_group(&BindGroupDescriptor {
            label: Some("ssao_composite_bind_group"),
            layout: &self.composite_bind_group_layout,
            entries: &[
                // material input
                BindGroupEntry { binding: 0, resource: BindingResource::TextureView(&gbuffer.material_view) },
                // output
                BindGroupEntry { binding: 1, resource: BindingResource::TextureView(&self.ssao_composited_view) },
                // blurred AO
                BindGroupEntry { binding: 2, resource: BindingResource::TextureView(&self.ssao_blurred_view) },
                // composite params
                BindGroupEntry { binding: 3, resource: self.comp_uniform.as_entire_binding() },
            ],
        });

        let mut comp_pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("ssao_composite_pass"),
            timestamp_writes: None,
        });
        comp_pass.set_pipeline(&self.composite_pipeline);
        comp_pass.set_bind_group(0, &comp_bg, &[]);
        comp_pass.dispatch_workgroups(workgroup_x, workgroup_y, 1);

        Ok(())
    }
    
    /// Get blurred SSAO texture view
    pub fn get_output(&self) -> &TextureView {
        &self.ssao_blurred_view
    }

    /// Get composited color (material * AO)
    pub fn get_composited(&self) -> &TextureView {
        &self.ssao_composited_view
    }

    pub fn set_composite_multiplier(&mut self, queue: &Queue, mul: f32) {
        let m = mul.clamp(0.0, 1.0);
        let params: [f32; 4] = [m, 0.0, 0.0, 0.0];
        queue.write_buffer(&self.comp_uniform, 0, bytemuck::cast_slice(&params));
    }
}

/// Screen-space effects manager
pub struct ScreenSpaceEffectsManager {
    gbuffer: GBuffer,
    ssao_renderer: Option<SsaoRenderer>,
    ssgi_renderer: Option<SsgiRenderer>,
    ssr_renderer: Option<SsrRenderer>,
    enabled_effects: Vec<ScreenSpaceEffect>,
}

impl ScreenSpaceEffectsManager {
    pub fn new(device: &Device, width: u32, height: u32) -> RenderResult<Self> {
        let gbuffer_config = GBufferConfig {
            width,
            height,
            ..Default::default()
        };
        let gbuffer = GBuffer::new(device, gbuffer_config)?;
        
        Ok(Self {
            gbuffer,
            ssao_renderer: None,
            ssgi_renderer: None,
            ssr_renderer: None,
            enabled_effects: Vec::new(),
        })
    }
    
    /// Enable an effect
    pub fn enable_effect(&mut self, device: &Device, effect: ScreenSpaceEffect) -> RenderResult<()> {
        if !self.enabled_effects.contains(&effect) {
            self.enabled_effects.push(effect);
        }
        
        match effect {
            ScreenSpaceEffect::SSAO => {
                if self.ssao_renderer.is_none() {
                    let (width, height) = self.gbuffer.dimensions();
                    self.ssao_renderer = Some(SsaoRenderer::new(device, width, height)?);
                }
            }
            ScreenSpaceEffect::SSGI => {
                if self.ssgi_renderer.is_none() {
                    let (width, height) = self.gbuffer.dimensions();
                    self.ssgi_renderer = Some(SsgiRenderer::new(device, width, height)?);
                }
            }
            ScreenSpaceEffect::SSR => {
                if self.ssr_renderer.is_none() {
                    let (width, height) = self.gbuffer.dimensions();
                    self.ssr_renderer = Some(SsrRenderer::new(device, width, height)?);
                }
            }
        }
        
        Ok(())
    }
    
    /// Disable an effect
    pub fn disable_effect(&mut self, effect: ScreenSpaceEffect) {
        self.enabled_effects.retain(|&e| e != effect);
    }
    
    /// Get GBuffer
    pub fn gbuffer(&self) -> &GBuffer {
        &self.gbuffer
    }
    
    /// Get mutable GBuffer
    pub fn gbuffer_mut(&mut self) -> &mut GBuffer {
        &mut self.gbuffer
    }
    
    /// Execute all enabled effects
    pub fn execute(
        &self,
        device: &Device,
        encoder: &mut CommandEncoder,
    ) -> RenderResult<()> {
        for effect in &self.enabled_effects {
            match effect {
                ScreenSpaceEffect::SSAO => {
                    if let Some(ref ssao) = self.ssao_renderer {
                        ssao.execute(device, encoder, &self.gbuffer)?;
                    }
                }
                ScreenSpaceEffect::SSGI => {
                    if let Some(ref ssgi) = self.ssgi_renderer {
                        ssgi.execute(device, encoder, &self.gbuffer)?;
                    }
                }
                ScreenSpaceEffect::SSR => {
                    if let Some(ref ssr) = self.ssr_renderer {
                        ssr.execute(device, encoder, &self.gbuffer)?;
                    }
                }
            }
        }
        Ok(())
    }

    /// Update camera parameters for all active effects
    pub fn update_camera(&mut self, queue: &Queue, camera: &CameraParams) {
        if let Some(ref mut ssao) = self.ssao_renderer {
            ssao.update_camera(queue, camera);
        }
        if let Some(ref mut ssgi) = self.ssgi_renderer {
            ssgi.update_camera(queue, camera);
        }
        if let Some(ref mut ssr) = self.ssr_renderer {
            ssr.update_camera(queue, camera);
        }
    }

    /// Return a GI debug texture view if available (SSR preferred, else SSGI)
    pub fn gi_debug_view(&self) -> Option<&TextureView> {
        if let Some(ref ssr) = self.ssr_renderer {
            // SSR filtered view
            #[allow(dead_code)]
            {
                return Some(ssr.get_output());
            }
        }
        if let Some(ref ssgi) = self.ssgi_renderer {
            #[allow(dead_code)]
            {
                return Some(ssgi.get_output_for_display());
            }
        }
        None
    }

    /// Return the material composited with SSAO if available
    pub fn material_with_ao_view(&self) -> Option<&TextureView> {
        if let Some(ref ssao) = self.ssao_renderer {
            return Some(ssao.get_composited());
        }
        None
    }

    pub fn set_ssao_composite_multiplier(&mut self, queue: &Queue, mul: f32) {
        if let Some(ref mut ssao) = self.ssao_renderer {
            ssao.set_composite_multiplier(queue, mul);
        }
    }

    pub fn update_ssao_settings(&mut self, queue: &Queue, mut f: impl FnMut(&mut SsaoSettings)) {
        if let Some(ref mut r) = self.ssao_renderer {
            let mut s = r.get_settings();
            f(&mut s);
            r.update_settings(queue, s);
        }
    }

    pub fn update_ssgi_settings(&mut self, queue: &Queue, mut f: impl FnMut(&mut SsgiSettings)) {
        if let Some(ref mut r) = self.ssgi_renderer {
            let mut s = r.get_settings();
            f(&mut s);
            r.update_settings(queue, s);
        }
    }

    pub fn update_ssr_settings(&mut self, queue: &Queue, mut f: impl FnMut(&mut SsrSettings)) {
        if let Some(ref mut r) = self.ssr_renderer {
            let mut s = r.get_settings();
            f(&mut s);
            r.update_settings(queue, s);
        }
    }

    pub fn set_env_for_all(&mut self, device: &Device, env_texture: &Texture) {
        if let Some(ref mut ssgi) = self.ssgi_renderer {
            ssgi.set_environment_texture(device, env_texture);
        }
        if let Some(ref mut ssr) = self.ssr_renderer {
            ssr.set_environment_texture(device, env_texture);
        }
    }

    pub fn set_ssgi_env(&mut self, device: &Device, env_texture: &Texture) {
        if let Some(ref mut r) = self.ssgi_renderer {
            r.set_environment_texture(device, env_texture);
        }
    }

    pub fn set_ssr_env(&mut self, device: &Device, env_texture: &Texture) {
        if let Some(ref mut r) = self.ssr_renderer {
            r.set_environment_texture(device, env_texture);
        }
    }

    /// Toggle SSGI half-resolution mode and reallocate its internal targets
    pub fn set_ssgi_half_res(&mut self, device: &Device, on: bool) {
        if let Some(ref mut _ssgi) = self.ssgi_renderer {
            // SSGI set_half_res requires a queue to rewrite inv_resolution
            // We don't store a queue here, so this function should be called from the viewer with available queue.
            // Provide a no-op queue reference via device.poll then accept an external queue in an alternate method.
            // For simplicity, this helper only toggles an internal flag if required; actual reallocation is done via set_ssgi_half_res_with_queue.
            let _ = (device, on);
        }
    }

    /// Toggle SSGI half-resolution with a provided queue
    pub fn set_ssgi_half_res_with_queue(&mut self, device: &Device, queue: &Queue, on: bool) {
        if let Some(ref mut ssgi) = self.ssgi_renderer {
            let _ = ssgi.set_half_res(device, queue, on);
        }
    }
}

/// SSGI renderer
pub struct SsgiRenderer {
    settings: SsgiSettings,
    settings_buffer: Buffer,
    camera_buffer: Buffer,

    // Pipelines
    ssgi_pipeline: ComputePipeline,
    ssgi_bind_group_layout: BindGroupLayout,
    temporal_pipeline: ComputePipeline,
    temporal_bind_group_layout: BindGroupLayout,
    upsample_pipeline: ComputePipeline,
    upsample_bind_group_layout: BindGroupLayout,

    // Output and temporal textures
    ssgi_texture: Texture,
    ssgi_view: TextureView,
    ssgi_history: Texture,
    ssgi_history_view: TextureView,
    ssgi_filtered: Texture,
    ssgi_filtered_view: TextureView,
    // Full-resolution upscaled output for half-res mode
    ssgi_upscaled: Texture,
    ssgi_upscaled_view: TextureView,
    upsample_sampler: Sampler,

    // Env
    env_texture: Texture,
    env_view: TextureView,
    env_sampler: Sampler,

    width: u32,
    height: u32,
    half_res: bool,
}

impl SsgiRenderer {
    pub fn new(device: &Device, width: u32, height: u32) -> RenderResult<Self> {
        let mut settings = SsgiSettings::default();
        settings.inv_resolution = [1.0 / width as f32, 1.0 / height as f32];

        // Uniform buffers
        let settings_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("ssgi_settings"),
            size: std::mem::size_of::<SsgiSettings>() as u64,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let camera_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("ssgi_camera"),
            size: std::mem::size_of::<CameraParams>() as u64,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Shader
        let shader = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("ssgi_shader"),
            source: ShaderSource::Wgsl(include_str!("../../shaders/ssgi.wgsl").into()),
        });

        // Bind group layouts
        // group(0): depth, normal, color, output, settings, camera, ibl cube, sampler
        let ssgi_bind_group_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("ssgi_bind_group_layout"),
            entries: &[
                // depth
                BindGroupLayoutEntry { binding: 0, visibility: ShaderStages::COMPUTE, ty: BindingType::Texture { sample_type: TextureSampleType::Float { filterable: false }, view_dimension: TextureViewDimension::D2, multisampled: false }, count: None },
                // normal
                BindGroupLayoutEntry { binding: 1, visibility: ShaderStages::COMPUTE, ty: BindingType::Texture { sample_type: TextureSampleType::Float { filterable: false }, view_dimension: TextureViewDimension::D2, multisampled: false }, count: None },
                // color
                BindGroupLayoutEntry { binding: 2, visibility: ShaderStages::COMPUTE, ty: BindingType::Texture { sample_type: TextureSampleType::Float { filterable: false }, view_dimension: TextureViewDimension::D2, multisampled: false }, count: None },
                // output
                BindGroupLayoutEntry { binding: 3, visibility: ShaderStages::COMPUTE, ty: BindingType::StorageTexture { access: StorageTextureAccess::WriteOnly, format: TextureFormat::Rgba16Float, view_dimension: TextureViewDimension::D2 }, count: None },
                // settings
                BindGroupLayoutEntry { binding: 4, visibility: ShaderStages::COMPUTE, ty: BindingType::Buffer { ty: BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None },
                // camera
                BindGroupLayoutEntry { binding: 5, visibility: ShaderStages::COMPUTE, ty: BindingType::Buffer { ty: BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None },
                // ibl cube
                BindGroupLayoutEntry { binding: 6, visibility: ShaderStages::COMPUTE, ty: BindingType::Texture { sample_type: TextureSampleType::Float { filterable: true }, view_dimension: TextureViewDimension::Cube, multisampled: false }, count: None },
                // ibl sampler
                BindGroupLayoutEntry { binding: 7, visibility: ShaderStages::COMPUTE, ty: BindingType::Sampler(SamplerBindingType::Filtering), count: None },
            ],
        });

        // group(1): temporal current, history, filtered, settings (reuse settings layout for simplicity)
        let temporal_bind_group_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("ssgi_temporal_bind_group_layout"),
            entries: &[
                // current
                BindGroupLayoutEntry { binding: 0, visibility: ShaderStages::COMPUTE, ty: BindingType::Texture { sample_type: TextureSampleType::Float { filterable: false }, view_dimension: TextureViewDimension::D2, multisampled: false }, count: None },
                // history
                BindGroupLayoutEntry { binding: 1, visibility: ShaderStages::COMPUTE, ty: BindingType::Texture { sample_type: TextureSampleType::Float { filterable: false }, view_dimension: TextureViewDimension::D2, multisampled: false }, count: None },
                // filtered
                BindGroupLayoutEntry { binding: 2, visibility: ShaderStages::COMPUTE, ty: BindingType::StorageTexture { access: StorageTextureAccess::WriteOnly, format: TextureFormat::Rgba16Float, view_dimension: TextureViewDimension::D2 }, count: None },
                // settings
                BindGroupLayoutEntry { binding: 3, visibility: ShaderStages::COMPUTE, ty: BindingType::Buffer { ty: BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None },
            ],
        });

        // Pipelines
        let ssgi_pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("ssgi_pipeline"),
            layout: Some(&device.create_pipeline_layout(&PipelineLayoutDescriptor {
                label: Some("ssgi_pipeline_layout"),
                bind_group_layouts: &[&ssgi_bind_group_layout],
                push_constant_ranges: &[],
            })),
            module: &shader,
            entry_point: "cs_ssgi",
        });

        let temporal_pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("ssgi_temporal_pipeline"),
            layout: Some(&device.create_pipeline_layout(&PipelineLayoutDescriptor {
                label: Some("ssgi_temporal_pipeline_layout"),
                // The temporal shader uses @group(1); include the main group at index 0
                bind_group_layouts: &[&ssgi_bind_group_layout, &temporal_bind_group_layout],
                push_constant_ranges: &[],
            })),
            module: &shader,
            entry_point: "cs_ssgi_temporal",
        });

        // Define upsample bind group layout here (local variable)
        let upsample_bind_group_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("ssgi_upsample_bind_group_layout"),
            entries: &[
                BindGroupLayoutEntry { binding: 0, visibility: ShaderStages::COMPUTE, ty: BindingType::Texture { sample_type: TextureSampleType::Float { filterable: true }, view_dimension: TextureViewDimension::D2, multisampled: false }, count: None },
                BindGroupLayoutEntry { binding: 1, visibility: ShaderStages::COMPUTE, ty: BindingType::StorageTexture { access: StorageTextureAccess::WriteOnly, format: TextureFormat::Rgba16Float, view_dimension: TextureViewDimension::D2 }, count: None },
                BindGroupLayoutEntry { binding: 2, visibility: ShaderStages::COMPUTE, ty: BindingType::Sampler(SamplerBindingType::Filtering), count: None },
                BindGroupLayoutEntry { binding: 3, visibility: ShaderStages::COMPUTE, ty: BindingType::Texture { sample_type: TextureSampleType::Float { filterable: false }, view_dimension: TextureViewDimension::D2, multisampled: false }, count: None },
                // full-res normal for edge-aware weights
                BindGroupLayoutEntry { binding: 4, visibility: ShaderStages::COMPUTE, ty: BindingType::Texture { sample_type: TextureSampleType::Float { filterable: false }, view_dimension: TextureViewDimension::D2, multisampled: false }, count: None },
                // settings (for upsample weights)
                BindGroupLayoutEntry { binding: 5, visibility: ShaderStages::COMPUTE, ty: BindingType::Buffer { ty: BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None },
            ],
        });

        let upsample_pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("ssgi_upsample_pipeline"),
            layout: Some(&device.create_pipeline_layout(&PipelineLayoutDescriptor {
                label: Some("ssgi_upsample_pipeline_layout"),
                // Upsample shader uses @group(2); include main and temporal groups first
                bind_group_layouts: &[&ssgi_bind_group_layout, &temporal_bind_group_layout, &upsample_bind_group_layout],
                push_constant_ranges: &[],
            })),
            module: &shader,
            entry_point: "cs_ssgi_upsample",
        });

        // Output and temporal textures
        let ssgi_texture = device.create_texture(&TextureDescriptor {
            label: Some("ssgi_texture"),
            size: Extent3d { width, height, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Rgba16Float,
            usage: TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let ssgi_view = ssgi_texture.create_view(&TextureViewDescriptor::default());

        let ssgi_history = device.create_texture(&TextureDescriptor {
            label: Some("ssgi_history"),
            size: Extent3d { width, height, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Rgba16Float,
            usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let ssgi_history_view = ssgi_history.create_view(&TextureViewDescriptor::default());

        let ssgi_filtered = device.create_texture(&TextureDescriptor {
            label: Some("ssgi_filtered"),
            size: Extent3d { width, height, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Rgba16Float,
            usage: TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let ssgi_filtered_view = ssgi_filtered.create_view(&TextureViewDescriptor::default());

        // Full-resolution upscaled target
        let ssgi_upscaled = device.create_texture(&TextureDescriptor {
            label: Some("ssgi_upscaled"),
            size: Extent3d { width, height, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Rgba16Float,
            usage: TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let ssgi_upscaled_view = ssgi_upscaled.create_view(&TextureViewDescriptor::default());

        // Placeholder env cube texture (1x1x6 RGBA8)
        let env_texture = device.create_texture(&TextureDescriptor {
            label: Some("ssgi_env_cube"),
            size: Extent3d { width: 1, height: 1, depth_or_array_layers: 6 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Rgba8Unorm,
            usage: TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let env_view = env_texture.create_view(&TextureViewDescriptor {
            label: Some("ssgi_env_cube_view"),
            format: Some(TextureFormat::Rgba8Unorm),
            dimension: Some(TextureViewDimension::Cube),
            aspect: TextureAspect::All,
            base_mip_level: 0,
            mip_level_count: None,
            base_array_layer: 0,
            array_layer_count: None,
        });
        let env_sampler = device.create_sampler(&SamplerDescriptor::default());
        let upsample_sampler = device.create_sampler(&SamplerDescriptor {
            label: Some("ssgi.upsample.sampler"),
            mag_filter: FilterMode::Linear,
            min_filter: FilterMode::Linear,
            mipmap_filter: FilterMode::Linear,
            address_mode_u: AddressMode::ClampToEdge,
            address_mode_v: AddressMode::ClampToEdge,
            address_mode_w: AddressMode::ClampToEdge,
            ..Default::default()
        });

        Ok(Self {
            settings,
            settings_buffer,
            camera_buffer,
            ssgi_pipeline,
            ssgi_bind_group_layout,
            temporal_pipeline,
            temporal_bind_group_layout,
            upsample_pipeline,
            upsample_bind_group_layout,
            ssgi_texture,
            ssgi_view,
            ssgi_history,
            ssgi_history_view,
            ssgi_filtered,
            ssgi_filtered_view,
            ssgi_upscaled,
            ssgi_upscaled_view,
            upsample_sampler,
            env_texture,
            env_view,
            env_sampler,
            width,
            height,
            half_res: false,
        })
    }

    pub fn update_settings(&mut self, queue: &Queue, settings: SsgiSettings) {
        self.settings = settings;
        queue.write_buffer(&self.settings_buffer, 0, bytemuck::bytes_of(&self.settings));
    }

    pub fn update_camera(&mut self, queue: &Queue, camera: &CameraParams) {
        queue.write_buffer(&self.camera_buffer, 0, bytemuck::bytes_of(camera));
    }

    pub fn get_settings(&self) -> SsgiSettings { self.settings }

    pub fn set_environment(&mut self, _env_view: &TextureView, _env_sampler: &Sampler) {
        // Deprecated in favor of set_environment_texture; kept for API-compat.
    }

    pub fn set_environment_texture(&mut self, device: &Device, env_texture: &Texture) {
        // Create a cube view from the provided texture
        let view = env_texture.create_view(&TextureViewDescriptor {
            label: Some("gi.env.cube.view"),
            format: None,
            dimension: Some(TextureViewDimension::Cube),
            aspect: TextureAspect::All,
            base_mip_level: 0,
            mip_level_count: None,
            base_array_layer: 0,
            array_layer_count: None,
        });
        self.env_view = view;
        // Create a linear sampler suitable for sampling the environment
        self.env_sampler = device.create_sampler(&SamplerDescriptor {
            label: Some("gi.env.cube.sampler"),
            address_mode_u: AddressMode::ClampToEdge,
            address_mode_v: AddressMode::ClampToEdge,
            address_mode_w: AddressMode::ClampToEdge,
            mag_filter: FilterMode::Linear,
            min_filter: FilterMode::Linear,
            mipmap_filter: FilterMode::Linear,
            ..Default::default()
        });
    }

    pub fn set_half_res(&mut self, device: &Device, queue: &Queue, on: bool) -> RenderResult<()> {
        self.half_res = on;
        let (w, h) = if on { (self.width.max(2) / 2, self.height.max(2) / 2) } else { (self.width, self.height) };

        // Recreate output and temporal textures at new resolution
        self.ssgi_texture = device.create_texture(&TextureDescriptor {
            label: Some("ssgi_texture"),
            size: Extent3d { width: w, height: h, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Rgba16Float,
            usage: TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        self.ssgi_view = self.ssgi_texture.create_view(&TextureViewDescriptor::default());

        self.ssgi_history = device.create_texture(&TextureDescriptor {
            label: Some("ssgi_history"),
            size: Extent3d { width: w, height: h, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Rgba16Float,
            usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
            view_formats: &[],
        });
        self.ssgi_history_view = self.ssgi_history.create_view(&TextureViewDescriptor::default());

        self.ssgi_filtered = device.create_texture(&TextureDescriptor {
            label: Some("ssgi_filtered"),
            size: Extent3d { width: w, height: h, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Rgba16Float,
            usage: TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        self.ssgi_filtered_view = self.ssgi_filtered.create_view(&TextureViewDescriptor::default());

        // Update inv_resolution in settings
        self.settings.inv_resolution = [1.0 / w as f32, 1.0 / h as f32];
        self.settings.use_half_res = if on { 1 } else { 0 };
        queue.write_buffer(&self.settings_buffer, 0, bytemuck::bytes_of(&self.settings));
        Ok(())
    }

    pub fn execute(&self, device: &Device, encoder: &mut CommandEncoder, gbuffer: &GBuffer) -> RenderResult<()> {
        // Bind group for main pass
        let bg = device.create_bind_group(&BindGroupDescriptor {
            label: Some("ssgi_bind_group"),
            layout: &self.ssgi_bind_group_layout,
            entries: &[
                BindGroupEntry { binding: 0, resource: BindingResource::TextureView(&gbuffer.depth_view) },
                BindGroupEntry { binding: 1, resource: BindingResource::TextureView(&gbuffer.normal_view) },
                BindGroupEntry { binding: 2, resource: BindingResource::TextureView(&gbuffer.material_view) },
                BindGroupEntry { binding: 3, resource: BindingResource::TextureView(&self.ssgi_view) },
                BindGroupEntry { binding: 4, resource: self.settings_buffer.as_entire_binding() },
                BindGroupEntry { binding: 5, resource: self.camera_buffer.as_entire_binding() },
                BindGroupEntry { binding: 6, resource: BindingResource::TextureView(&self.env_view) },
                BindGroupEntry { binding: 7, resource: BindingResource::Sampler(&self.env_sampler) },
            ],
        });

        // Dispatch using the current SSGI output resolution. In half-res mode the
        // output textures are reallocated at half the base resolution via
        // `set_half_res()`, so we must also halve the dispatch grid here.
        let (w_out, h_out) = if self.half_res {
            (self.width.max(2) / 2, self.height.max(2) / 2)
        } else {
            (self.width, self.height)
        };
        let gx = (w_out + 7) / 8;
        let gy = (h_out + 7) / 8;

        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor { label: Some("ssgi_pass"), timestamp_writes: None });
            pass.set_pipeline(&self.ssgi_pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(gx, gy, 1);
        }

        // Temporal pass (optional)
        let temporal_bg = device.create_bind_group(&BindGroupDescriptor {
            label: Some("ssgi_temporal_bg"),
            layout: &self.temporal_bind_group_layout,
            entries: &[
                BindGroupEntry { binding: 0, resource: BindingResource::TextureView(&self.ssgi_view) },
                BindGroupEntry { binding: 1, resource: BindingResource::TextureView(&self.ssgi_history_view) },
                BindGroupEntry { binding: 2, resource: BindingResource::TextureView(&self.ssgi_filtered_view) },
                BindGroupEntry { binding: 3, resource: self.settings_buffer.as_entire_binding() },
            ],
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor { label: Some("ssgi_temporal"), timestamp_writes: None });
            pass.set_pipeline(&self.temporal_pipeline);
            // Bind main group(0) and temporal resources at group(1) per pipeline layout
            pass.set_bind_group(0, &bg, &[]);
            pass.set_bind_group(1, &temporal_bg, &[]);
            pass.dispatch_workgroups(gx, gy, 1);
        }

        // Copy temporal filtered output to history for next frame
        encoder.copy_texture_to_texture(
            ImageCopyTexture {
                texture: &self.ssgi_filtered,
                mip_level: 0,
                origin: Origin3d::ZERO,
                aspect: TextureAspect::All,
            },
            ImageCopyTexture {
                texture: &self.ssgi_history,
                mip_level: 0,
                origin: Origin3d::ZERO,
                aspect: TextureAspect::All,
            },
            Extent3d { width: w_out, height: h_out, depth_or_array_layers: 1 },
        );

        // If half-res, upsample filtered SSGI to full resolution for display
        if self.half_res {
            let up_bg = device.create_bind_group(&BindGroupDescriptor {
                label: Some("ssgi_upsample_bg"),
                layout: &self.upsample_bind_group_layout,
                entries: &[
                    BindGroupEntry { binding: 0, resource: BindingResource::TextureView(&self.ssgi_filtered_view) },
                    BindGroupEntry { binding: 1, resource: BindingResource::TextureView(&self.ssgi_upscaled_view) },
                    BindGroupEntry { binding: 2, resource: BindingResource::Sampler(&self.upsample_sampler) },
                    BindGroupEntry { binding: 3, resource: BindingResource::TextureView(&gbuffer.depth_view) },
                    BindGroupEntry { binding: 4, resource: BindingResource::TextureView(&gbuffer.normal_view) },
                    BindGroupEntry { binding: 5, resource: self.settings_buffer.as_entire_binding() },
                ],
            });
            let gx_full = (self.width + 7) / 8;
            let gy_full = (self.height + 7) / 8;
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor { label: Some("ssgi_upsample"), timestamp_writes: None });
            pass.set_pipeline(&self.upsample_pipeline);
            // Bind main group(0), temporal group(1) and upsample group(2) per pipeline layout
            pass.set_bind_group(0, &bg, &[]);
            pass.set_bind_group(1, &temporal_bg, &[]);
            pass.set_bind_group(2, &up_bg, &[]);
            pass.dispatch_workgroups(gx_full, gy_full, 1);
        }

        Ok(())
    }

    #[allow(dead_code)]
    pub fn get_output(&self) -> &TextureView { &self.ssgi_filtered_view }

    /// Get display-ready output (upscaled if running half-res)
    pub fn get_output_for_display(&self) -> &TextureView {
        if self.half_res { &self.ssgi_upscaled_view } else { &self.ssgi_filtered_view }
    }
}

/// SSR renderer
pub struct SsrRenderer {
    settings: SsrSettings,
    settings_buffer: Buffer,
    camera_buffer: Buffer,

    ssr_pipeline: ComputePipeline,
    ssr_bind_group_layout: BindGroupLayout,
    temporal_pipeline: ComputePipeline,
    temporal_bind_group_layout: BindGroupLayout,

    ssr_texture: Texture,
    ssr_view: TextureView,
    ssr_history: Texture,
    ssr_history_view: TextureView,
    ssr_filtered: Texture,
    ssr_filtered_view: TextureView,

    env_texture: Texture,
    env_view: TextureView,
    env_sampler: Sampler,

    width: u32,
    height: u32,
}

impl SsrRenderer {
    pub fn new(device: &Device, width: u32, height: u32) -> RenderResult<Self> {
        let mut settings = SsrSettings::default();
        settings.inv_resolution = [1.0 / width as f32, 1.0 / height as f32];

        let settings_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("ssr_settings"),
            size: std::mem::size_of::<SsrSettings>() as u64,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let camera_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("ssr_camera"),
            size: std::mem::size_of::<CameraParams>() as u64,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let shader = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("ssr_shader"),
            source: ShaderSource::Wgsl(include_str!("../../shaders/ssr.wgsl").into()),
        });

        let ssr_bind_group_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("ssr_bind_group_layout"),
            entries: &[
                // depth
                BindGroupLayoutEntry { binding: 0, visibility: ShaderStages::COMPUTE, ty: BindingType::Texture { sample_type: TextureSampleType::Float { filterable: false }, view_dimension: TextureViewDimension::D2, multisampled: false }, count: None },
                // normal
                BindGroupLayoutEntry { binding: 1, visibility: ShaderStages::COMPUTE, ty: BindingType::Texture { sample_type: TextureSampleType::Float { filterable: false }, view_dimension: TextureViewDimension::D2, multisampled: false }, count: None },
                // color
                BindGroupLayoutEntry { binding: 2, visibility: ShaderStages::COMPUTE, ty: BindingType::Texture { sample_type: TextureSampleType::Float { filterable: false }, view_dimension: TextureViewDimension::D2, multisampled: false }, count: None },
                // output
                BindGroupLayoutEntry { binding: 3, visibility: ShaderStages::COMPUTE, ty: BindingType::StorageTexture { access: StorageTextureAccess::WriteOnly, format: TextureFormat::Rgba16Float, view_dimension: TextureViewDimension::D2 }, count: None },
                // settings
                BindGroupLayoutEntry { binding: 4, visibility: ShaderStages::COMPUTE, ty: BindingType::Buffer { ty: BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None },
                // camera
                BindGroupLayoutEntry { binding: 5, visibility: ShaderStages::COMPUTE, ty: BindingType::Buffer { ty: BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None },
                // env cube
                BindGroupLayoutEntry { binding: 6, visibility: ShaderStages::COMPUTE, ty: BindingType::Texture { sample_type: TextureSampleType::Float { filterable: true }, view_dimension: TextureViewDimension::Cube, multisampled: false }, count: None },
                // env sampler
                BindGroupLayoutEntry { binding: 7, visibility: ShaderStages::COMPUTE, ty: BindingType::Sampler(SamplerBindingType::Filtering), count: None },
            ],
        });

        let temporal_bind_group_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("ssr_temporal_bind_group_layout"),
            entries: &[
                // current
                BindGroupLayoutEntry { binding: 0, visibility: ShaderStages::COMPUTE, ty: BindingType::Texture { sample_type: TextureSampleType::Float { filterable: false }, view_dimension: TextureViewDimension::D2, multisampled: false }, count: None },
                // history
                BindGroupLayoutEntry { binding: 1, visibility: ShaderStages::COMPUTE, ty: BindingType::Texture { sample_type: TextureSampleType::Float { filterable: false }, view_dimension: TextureViewDimension::D2, multisampled: false }, count: None },
                // filtered
                BindGroupLayoutEntry { binding: 2, visibility: ShaderStages::COMPUTE, ty: BindingType::StorageTexture { access: StorageTextureAccess::WriteOnly, format: TextureFormat::Rgba16Float, view_dimension: TextureViewDimension::D2 }, count: None },
            ],
        });

        let ssr_pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("ssr_pipeline"),
            layout: Some(&device.create_pipeline_layout(&PipelineLayoutDescriptor { label: Some("ssr_pipeline_layout"), bind_group_layouts: &[&ssr_bind_group_layout], push_constant_ranges: &[] })),
            module: &shader,
            entry_point: "cs_ssr",
        });
        let temporal_pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("ssr_temporal_pipeline"),
            layout: Some(&device.create_pipeline_layout(&PipelineLayoutDescriptor {
                label: Some("ssr_temporal_pipeline_layout"),
                // The temporal shader uses @group(1); include the main group at index 0
                bind_group_layouts: &[&ssr_bind_group_layout, &temporal_bind_group_layout],
                push_constant_ranges: &[],
            })),
            module: &shader,
            entry_point: "cs_ssr_temporal",
        });

        // Textures
        let ssr_texture = device.create_texture(&TextureDescriptor {
            label: Some("ssr_texture"),
            size: Extent3d { width, height, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Rgba16Float,
            usage: TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let ssr_view = ssr_texture.create_view(&TextureViewDescriptor::default());

        let ssr_history = device.create_texture(&TextureDescriptor {
            label: Some("ssr_history"),
            size: Extent3d { width, height, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Rgba16Float,
            usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let ssr_history_view = ssr_history.create_view(&TextureViewDescriptor::default());

        let ssr_filtered = device.create_texture(&TextureDescriptor {
            label: Some("ssr_filtered"),
            size: Extent3d { width, height, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Rgba16Float,
            usage: TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let ssr_filtered_view = ssr_filtered.create_view(&TextureViewDescriptor::default());

        // Env cube
        let env_texture = device.create_texture(&TextureDescriptor {
            label: Some("ssr_env_cube"),
            size: Extent3d { width: 1, height: 1, depth_or_array_layers: 6 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Rgba8Unorm,
            usage: TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let env_view = env_texture.create_view(&TextureViewDescriptor {
            label: Some("ssr_env_cube_view"),
            format: Some(TextureFormat::Rgba8Unorm),
            dimension: Some(TextureViewDimension::Cube),
            aspect: TextureAspect::All,
            base_mip_level: 0,
            mip_level_count: None,
            base_array_layer: 0,
            array_layer_count: None,
        });
        let env_sampler = device.create_sampler(&SamplerDescriptor::default());

        Ok(Self {
            settings,
            settings_buffer,
            camera_buffer,
            ssr_pipeline,
            ssr_bind_group_layout,
            temporal_pipeline,
            temporal_bind_group_layout,
            ssr_texture,
            ssr_view,
            ssr_history,
            ssr_history_view,
            ssr_filtered,
            ssr_filtered_view,
            env_texture,
            env_view,
            env_sampler,
            width,
            height,
        })
    }

    pub fn update_settings(&mut self, queue: &Queue, settings: SsrSettings) {
        self.settings = settings;
        queue.write_buffer(&self.settings_buffer, 0, bytemuck::bytes_of(&self.settings));
    }

    pub fn update_camera(&mut self, queue: &Queue, camera: &CameraParams) {
        queue.write_buffer(&self.camera_buffer, 0, bytemuck::bytes_of(camera));
    }

    pub fn execute(&self, device: &Device, encoder: &mut CommandEncoder, gbuffer: &GBuffer) -> RenderResult<()> {
        let bg = device.create_bind_group(&BindGroupDescriptor {
            label: Some("ssr_bind_group"),
            layout: &self.ssr_bind_group_layout,
            entries: &[
                BindGroupEntry { binding: 0, resource: BindingResource::TextureView(&gbuffer.depth_view) },
                BindGroupEntry { binding: 1, resource: BindingResource::TextureView(&gbuffer.normal_view) },
                BindGroupEntry { binding: 2, resource: BindingResource::TextureView(&gbuffer.material_view) },
                BindGroupEntry { binding: 3, resource: BindingResource::TextureView(&self.ssr_view) },
                BindGroupEntry { binding: 4, resource: self.settings_buffer.as_entire_binding() },
                BindGroupEntry { binding: 5, resource: self.camera_buffer.as_entire_binding() },
                BindGroupEntry { binding: 6, resource: BindingResource::TextureView(&self.env_view) },
                BindGroupEntry { binding: 7, resource: BindingResource::Sampler(&self.env_sampler) },
            ],
        });

        let (w, h) = (self.width, self.height);
        let gx = (w + 7) / 8;
        let gy = (h + 7) / 8;

        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor { label: Some("ssr_pass"), timestamp_writes: None });
            pass.set_pipeline(&self.ssr_pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(gx, gy, 1);
        }

        let temporal_bg = device.create_bind_group(&BindGroupDescriptor {
            label: Some("ssr_temporal_bg"),
            layout: &self.temporal_bind_group_layout,
            entries: &[
                BindGroupEntry { binding: 0, resource: BindingResource::TextureView(&self.ssr_view) },
                BindGroupEntry { binding: 1, resource: BindingResource::TextureView(&self.ssr_history_view) },
                BindGroupEntry { binding: 2, resource: BindingResource::TextureView(&self.ssr_filtered_view) },
            ],
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor { label: Some("ssr_temporal"), timestamp_writes: None });
            pass.set_pipeline(&self.temporal_pipeline);
            // Bind temporal resources at group(1) per pipeline layout
            pass.set_bind_group(1, &temporal_bg, &[]);
            pass.dispatch_workgroups(gx, gy, 1);
        }

        // Copy temporal filtered output to history for next frame
        encoder.copy_texture_to_texture(
            ImageCopyTexture {
                texture: &self.ssr_filtered,
                mip_level: 0,
                origin: Origin3d::ZERO,
                aspect: TextureAspect::All,
            },
            ImageCopyTexture {
                texture: &self.ssr_history,
                mip_level: 0,
                origin: Origin3d::ZERO,
                aspect: TextureAspect::All,
            },
            Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        );

        Ok(())
    }

    #[allow(dead_code)]
    pub fn get_output(&self) -> &TextureView { &self.ssr_filtered_view }

    pub fn get_settings(&self) -> SsrSettings { self.settings }

    pub fn set_environment(&mut self, _env_view: &TextureView, _env_sampler: &Sampler) {
        // Deprecated in favor of set_environment_texture; kept for API-compat.
    }

    pub fn set_environment_texture(&mut self, device: &Device, env_texture: &Texture) {
        let view = env_texture.create_view(&TextureViewDescriptor {
            label: Some("gi.env.cube.view"),
            format: None,
            dimension: Some(TextureViewDimension::Cube),
            aspect: TextureAspect::All,
            base_mip_level: 0,
            mip_level_count: None,
            base_array_layer: 0,
            array_layer_count: None,
        });
        self.env_view = view;
        self.env_sampler = device.create_sampler(&SamplerDescriptor {
            label: Some("gi.env.cube.sampler"),
            address_mode_u: AddressMode::ClampToEdge,
            address_mode_v: AddressMode::ClampToEdge,
            address_mode_w: AddressMode::ClampToEdge,
            mag_filter: FilterMode::Linear,
            min_filter: FilterMode::Linear,
            mipmap_filter: FilterMode::Linear,
            ..Default::default()
        });
    }
}
