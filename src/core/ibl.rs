// B15: Image-Based Lighting (IBL) Polish - Core Rust implementation
// Provides irradiance/specular prefiltering and BRDF LUT generation for physically-based IBL

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

/// IBL quality levels
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IBLQuality {
    Low = 0,    // 64x64 irradiance, 128x128 specular, 256x256 BRDF
    Medium = 1, // 128x128 irradiance, 256x256 specular, 512x512 BRDF
    High = 2,   // 256x256 irradiance, 512x512 specular, 512x512 BRDF
    Ultra = 3,  // 512x512 irradiance, 1024x1024 specular, 512x512 BRDF
}

impl Default for IBLQuality {
    fn default() -> Self {
        Self::Medium
    }
}

impl IBLQuality {
    pub fn irradiance_size(self) -> u32 {
        match self {
            Self::Low => 64,
            Self::Medium => 128,
            Self::High => 256,
            Self::Ultra => 512,
        }
    }

    pub fn specular_size(self) -> u32 {
        match self {
            Self::Low => 128,
            Self::Medium => 256,
            Self::High => 512,
            Self::Ultra => 1024,
        }
    }

    pub fn brdf_size(self) -> u32 {
        match self {
            Self::Low | Self::Medium | Self::High | Self::Ultra => 512,
        }
    }

    pub fn specular_mip_levels(self) -> u32 {
        match self {
            Self::Low => 5,    // 128 -> 8
            Self::Medium => 6, // 256 -> 8
            Self::High => 7,   // 512 -> 8
            Self::Ultra => 8,  // 1024 -> 8
        }
    }
}

/// Environment map types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EnvironmentMapType {
    Equirectangular,
    Cubemap,
    SphericalHDR,
}

/// IBL uniform data structure (matches WGSL layout exactly)
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct IBLUniforms {
    // Environment map properties (16 bytes)
    pub env_map_size: u32,
    pub target_face: u32,    // Cubemap face index (0-5)
    pub mip_level: u32,      // Target mip level for specular prefiltering
    pub max_mip_levels: u32, // Total mip levels in specular map

    // Filtering parameters (16 bytes)
    pub roughness: f32,    // Roughness level for specular prefiltering
    pub sample_count: u32, // Number of samples for Monte Carlo integration
    pub _pad0: f32,
    pub _pad1: f32,

    // BRDF LUT parameters (16 bytes)
    pub brdf_size: u32, // Size of BRDF LUT texture
    pub _pad2: f32,
    pub _pad3: f32,
    pub _pad4: f32,

    // Padding to 64 bytes
    pub _pad5: [f32; 4],
}

impl Default for IBLUniforms {
    fn default() -> Self {
        Self {
            env_map_size: 512,
            target_face: 0,
            mip_level: 0,
            max_mip_levels: 5,
            roughness: 0.0,
            sample_count: 1024,
            _pad0: 0.0,
            _pad1: 0.0,
            brdf_size: 512,
            _pad2: 0.0,
            _pad3: 0.0,
            _pad4: 0.0,
            _pad5: [0.0; 4],
        }
    }
}

/// IBL material parameters
#[derive(Debug, Clone, Copy)]
pub struct IBLMaterial {
    pub metallic: f32,
    pub roughness: f32,
    pub albedo: [f32; 3],
    pub f0: [f32; 3],
    pub ambient_occlusion: f32,
}

impl Default for IBLMaterial {
    fn default() -> Self {
        Self {
            metallic: 0.0,
            roughness: 0.5,
            albedo: [1.0, 1.0, 1.0],
            f0: [0.04, 0.04, 0.04],
            ambient_occlusion: 1.0,
        }
    }
}

/// Core IBL renderer for environment-based lighting
pub struct IBLRenderer {
    // Rendering pipelines
    irradiance_pipeline: wgpu::RenderPipeline,
    specular_pipeline: wgpu::RenderPipeline,
    brdf_pipeline: wgpu::RenderPipeline,
    _equirect_to_cube_pipeline: wgpu::RenderPipeline,

    // Bind group layouts
    ibl_bind_group_layout: wgpu::BindGroupLayout,
    pbr_bind_group_layout: wgpu::BindGroupLayout,

    // Buffers
    uniforms_buffer: wgpu::Buffer,

    // IBL textures
    environment_map: Option<wgpu::Texture>,
    environment_view: Option<wgpu::TextureView>,
    irradiance_map: Option<wgpu::Texture>,
    irradiance_view: Option<wgpu::TextureView>,
    specular_map: Option<wgpu::Texture>,
    specular_view: Option<wgpu::TextureView>,
    brdf_lut: Option<wgpu::Texture>,
    brdf_lut_view: Option<wgpu::TextureView>,

    // Samplers
    environment_sampler: wgpu::Sampler,
    irradiance_sampler: wgpu::Sampler,
    specular_sampler: wgpu::Sampler,
    brdf_sampler: wgpu::Sampler,

    // Bind groups
    ibl_bind_group: Option<wgpu::BindGroup>,
    pbr_bind_group: Option<wgpu::BindGroup>,

    // Configuration
    uniforms: IBLUniforms,
    quality: IBLQuality,
    is_initialized: bool,
}

impl IBLRenderer {
    pub fn new(device: &wgpu::Device, quality: IBLQuality) -> Self {
        // Load shader
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("ibl_shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/ibl.wgsl").into()),
        });

        // Create bind group layouts
        let ibl_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("ibl_bind_group_layout"),
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
                    // Environment map
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::Cube,
                            multisampled: false,
                        },
                        count: None,
                    },
                    // Environment sampler
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
            });

        let pbr_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("ibl_pbr_bind_group_layout"),
                entries: &[
                    // Irradiance map
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::Cube,
                            multisampled: false,
                        },
                        count: None,
                    },
                    // Irradiance sampler
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                    // Specular map
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::Cube,
                            multisampled: false,
                        },
                        count: None,
                    },
                    // Specular sampler
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                    // BRDF LUT
                    wgpu::BindGroupLayoutEntry {
                        binding: 4,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    // BRDF sampler
                    wgpu::BindGroupLayoutEntry {
                        binding: 5,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
            });

        // Create pipeline layouts
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("ibl_pipeline_layout"),
            bind_group_layouts: &[&ibl_bind_group_layout],
            push_constant_ranges: &[],
        });

        // Create irradiance convolution pipeline
        let irradiance_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("ibl_irradiance_pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_irradiance",
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_irradiance_convolution",
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Rgba16Float,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
        });

        // Create specular prefiltering pipeline
        let specular_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("ibl_specular_pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_irradiance",
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_specular_prefilter",
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Rgba16Float,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
        });

        // Create BRDF integration pipeline
        let brdf_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("ibl_brdf_pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_irradiance",
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_brdf_integration",
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Rg16Float,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
        });

        // Create equirectangular to cubemap pipeline
        let equirect_to_cube_pipeline =
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("ibl_equirect_to_cube_pipeline"),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: "vs_irradiance",
                    buffers: &[],
                    },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: "fs_equirect_to_cube",
                    targets: &[Some(wgpu::ColorTargetState {
                        format: wgpu::TextureFormat::Rgba16Float,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    }),
                primitive: wgpu::PrimitiveState::default(),
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                multiview: None,
                });

        // Create samplers
        let environment_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("environment_sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        let irradiance_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("irradiance_sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        let specular_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("specular_sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        let brdf_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("brdf_sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        // Create uniforms buffer
        let mut uniforms = IBLUniforms::default();
        uniforms.brdf_size = quality.brdf_size();
        uniforms.max_mip_levels = quality.specular_mip_levels();

        let uniforms_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("ibl_uniforms"),
            contents: bytemuck::cast_slice(&[uniforms]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        Self {
            irradiance_pipeline,
            specular_pipeline,
            brdf_pipeline,
            _equirect_to_cube_pipeline: equirect_to_cube_pipeline,
            ibl_bind_group_layout,
            pbr_bind_group_layout,
            uniforms_buffer,
            environment_map: None,
            environment_view: None,
            irradiance_map: None,
            irradiance_view: None,
            specular_map: None,
            specular_view: None,
            brdf_lut: None,
            brdf_lut_view: None,
            environment_sampler,
            irradiance_sampler,
            specular_sampler,
            brdf_sampler,
            ibl_bind_group: None,
            pbr_bind_group: None,
            uniforms,
            quality,
            is_initialized: false,
        }
    }

    /// Load environment map from HDR file
    pub fn load_environment_map(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        _hdr_data: &[f32],
        _width: u32,
        _height: u32,
    ) -> Result<(), String> {
        // Create cubemap texture from HDR data
        let size = 512; // Fixed cubemap size for now

        let environment_map = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("environment_cubemap"),
            size: wgpu::Extent3d {
                width: size,
                height: size,
                depth_or_array_layers: 6,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });

        let environment_view = environment_map.create_view(&wgpu::TextureViewDescriptor {
            label: Some("environment_cubemap_view"),
            format: Some(wgpu::TextureFormat::Rgba16Float),
            dimension: Some(wgpu::TextureViewDimension::Cube),
            aspect: wgpu::TextureAspect::All,
            base_mip_level: 0,
            mip_level_count: Some(1),
            base_array_layer: 0,
            array_layer_count: Some(6),
        });

        // TODO: Convert equirectangular HDR to cubemap
        // For now, create a simple gradient environment
        self.create_default_environment(device, queue, size)?;

        self.environment_map = Some(environment_map);
        self.environment_view = Some(environment_view);

        Ok(())
    }

    /// Create default environment for testing
    fn create_default_environment(
        &mut self,
        _device: &wgpu::Device,
        queue: &wgpu::Queue,
        size: u32,
    ) -> Result<(), String> {
        // Create a simple gradient environment for testing
        let mut env_data = vec![0.0f32; (size * size * 6 * 4) as usize];

        for face in 0..6 {
            for y in 0..size {
                for x in 0..size {
                    let idx = ((face * size * size + y * size + x) * 4) as usize;

                    // Create different colors for each face
                    let (r, g, b) = match face {
                        0 => (1.0, 0.5, 0.5), // +X - Red
                        1 => (0.5, 1.0, 0.5), // -X - Green
                        2 => (0.5, 0.5, 1.0), // +Y - Blue
                        3 => (1.0, 1.0, 0.5), // -Y - Yellow
                        4 => (1.0, 0.5, 1.0), // +Z - Magenta
                        5 => (0.5, 1.0, 1.0), // -Z - Cyan
                        _ => (0.5, 0.5, 0.5),
                    };

                    // Add some variation based on position
                    let fx = x as f32 / size as f32;
                    let fy = y as f32 / size as f32;
                    let intensity = 0.5 + 0.5 * (fx + fy) * 0.5;

                    env_data[idx] = r * intensity;
                    env_data[idx + 1] = g * intensity;
                    env_data[idx + 2] = b * intensity;
                    env_data[idx + 3] = 1.0;
                }
            }
        }

        // Convert to half precision for upload
        let env_data_bytes: Vec<u8> = env_data
            .chunks(4)
            .flat_map(|chunk| {
                let r = half::f16::from_f32(chunk[0]);
                let g = half::f16::from_f32(chunk[1]);
                let b = half::f16::from_f32(chunk[2]);
                let a = half::f16::from_f32(chunk[3]);
                [
                    r.to_le_bytes(),
                    g.to_le_bytes(),
                    b.to_le_bytes(),
                    a.to_le_bytes(),
                ]
                .concat()
            })
            .collect();

        // Upload to each face of the cubemap
        if let Some(ref env_map) = self.environment_map {
            for face in 0..6 {
                let face_offset = (face * size * size * 8) as usize; // 8 bytes per pixel (4 x f16)
                let face_data =
                    &env_data_bytes[face_offset..face_offset + (size * size * 8) as usize];

                queue.write_texture(
                    wgpu::ImageCopyTexture {
                        texture: env_map,
                        mip_level: 0,
                        origin: wgpu::Origin3d {
                            x: 0,
                            y: 0,
                            z: face,
                        },
                        aspect: wgpu::TextureAspect::All,
                    },
                    face_data,
                    wgpu::ImageDataLayout {
                        offset: 0,
                        bytes_per_row: Some(size * 8), // 8 bytes per pixel
                        rows_per_image: Some(size),
                    },
                    wgpu::Extent3d {
                        width: size,
                        height: size,
                        depth_or_array_layers: 1,
                    },
                );
            }
        }

        Ok(())
    }

    /// Generate irradiance map from environment map
    pub fn generate_irradiance_map(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> Result<(), String> {
        if self.environment_map.is_none() {
            return Err("Environment map not loaded".to_string());
        }

        let size = self.quality.irradiance_size();

        // Create irradiance cubemap
        let irradiance_map = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("irradiance_cubemap"),
            size: wgpu::Extent3d {
                width: size,
                height: size,
                depth_or_array_layers: 6,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });

        let irradiance_view = irradiance_map.create_view(&wgpu::TextureViewDescriptor {
            label: Some("irradiance_cubemap_view"),
            format: Some(wgpu::TextureFormat::Rgba16Float),
            dimension: Some(wgpu::TextureViewDimension::Cube),
            aspect: wgpu::TextureAspect::All,
            base_mip_level: 0,
            mip_level_count: Some(1),
            base_array_layer: 0,
            array_layer_count: Some(6),
        });

        // Generate irradiance for each face
        self.uniforms.env_map_size = size;
        self.uniforms.sample_count = 2048; // High quality for irradiance

        self.update_uniforms(queue);
        self.create_ibl_bind_group(device);

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("irradiance_generation"),
        });

        for face in 0..6 {
            self.uniforms.target_face = face;
            self.update_uniforms(queue);

            let face_view = irradiance_map.create_view(&wgpu::TextureViewDescriptor {
                label: Some(&format!("irradiance_face_{}", face)),
                format: Some(wgpu::TextureFormat::Rgba16Float),
                dimension: Some(wgpu::TextureViewDimension::D2),
                aspect: wgpu::TextureAspect::All,
                base_mip_level: 0,
                mip_level_count: Some(1),
                base_array_layer: face,
                array_layer_count: Some(1),
            });

            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some(&format!("irradiance_pass_face_{}", face)),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &face_view,
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

            if let Some(ref bind_group) = self.ibl_bind_group {
                render_pass.set_pipeline(&self.irradiance_pipeline);
                render_pass.set_bind_group(0, bind_group, &[]);
                render_pass.draw(0..3, 0..1);
            }
        }

        queue.submit(Some(encoder.finish()));

        self.irradiance_map = Some(irradiance_map);
        self.irradiance_view = Some(irradiance_view);

        Ok(())
    }

    /// Generate specular prefiltered map
    pub fn generate_specular_map(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> Result<(), String> {
        if self.environment_map.is_none() {
            return Err("Environment map not loaded".to_string());
        }

        let size = self.quality.specular_size();
        let mip_levels = self.quality.specular_mip_levels();

        // Create specular cubemap with mipmaps
        let specular_map = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("specular_cubemap"),
            size: wgpu::Extent3d {
                width: size,
                height: size,
                depth_or_array_layers: 6,
            },
            mip_level_count: mip_levels,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });

        let specular_view = specular_map.create_view(&wgpu::TextureViewDescriptor {
            label: Some("specular_cubemap_view"),
            format: Some(wgpu::TextureFormat::Rgba16Float),
            dimension: Some(wgpu::TextureViewDimension::Cube),
            aspect: wgpu::TextureAspect::All,
            base_mip_level: 0,
            mip_level_count: Some(mip_levels),
            base_array_layer: 0,
            array_layer_count: Some(6),
        });

        // Generate specular prefiltering for each mip level and face
        self.uniforms.max_mip_levels = mip_levels;
        self.uniforms.sample_count = 1024; // Good quality for specular

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("specular_generation"),
        });

        for mip in 0..mip_levels {
            let mip_size = size >> mip;
            if mip_size == 0 {
                break;
            }

            let roughness = mip as f32 / (mip_levels - 1) as f32;
            self.uniforms.mip_level = mip;
            self.uniforms.roughness = roughness;
            self.uniforms.env_map_size = mip_size;

            self.update_uniforms(queue);

            for face in 0..6 {
                self.uniforms.target_face = face;
                self.update_uniforms(queue);
                self.create_ibl_bind_group(device);

                let face_view = specular_map.create_view(&wgpu::TextureViewDescriptor {
                    label: Some(&format!("specular_face_{}_{}", face, mip)),
                    format: Some(wgpu::TextureFormat::Rgba16Float),
                    dimension: Some(wgpu::TextureViewDimension::D2),
                    aspect: wgpu::TextureAspect::All,
                    base_mip_level: mip,
                    mip_level_count: Some(1),
                    base_array_layer: face,
                    array_layer_count: Some(1),
                });

                let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some(&format!("specular_pass_face_{}_{}", face, mip)),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &face_view,
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

                if let Some(ref bind_group) = self.ibl_bind_group {
                    render_pass.set_pipeline(&self.specular_pipeline);
                    render_pass.set_bind_group(0, bind_group, &[]);
                    render_pass.draw(0..3, 0..1);
                }
            }
        }

        queue.submit(Some(encoder.finish()));

        self.specular_map = Some(specular_map);
        self.specular_view = Some(specular_view);

        Ok(())
    }

    /// Generate BRDF integration lookup table
    pub fn generate_brdf_lut(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> Result<(), String> {
        let size = self.quality.brdf_size();

        // Create BRDF LUT texture
        let brdf_lut = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("brdf_lut"),
            size: wgpu::Extent3d {
                width: size,
                height: size,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rg16Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });

        let brdf_lut_view = brdf_lut.create_view(&wgpu::TextureViewDescriptor::default());

        // Generate BRDF LUT
        self.uniforms.brdf_size = size;
        self.update_uniforms(queue);
        self.create_ibl_bind_group(device);

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("brdf_lut_generation"),
        });

        let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("brdf_lut_pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &brdf_lut_view,
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

        if let Some(ref bind_group) = self.ibl_bind_group {
            render_pass.set_pipeline(&self.brdf_pipeline);
            render_pass.set_bind_group(0, bind_group, &[]);
            render_pass.draw(0..3, 0..1);
        }

        drop(render_pass);
        queue.submit(Some(encoder.finish()));

        self.brdf_lut = Some(brdf_lut);
        self.brdf_lut_view = Some(brdf_lut_view);

        Ok(())
    }

    /// Initialize complete IBL pipeline
    pub fn initialize(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) -> Result<(), String> {
        // Create default environment if none loaded
        if self.environment_map.is_none() {
            self.create_default_environment(device, queue, 512)?;
        }

        // Generate all IBL textures
        self.generate_irradiance_map(device, queue)?;
        self.generate_specular_map(device, queue)?;
        self.generate_brdf_lut(device, queue)?;

        // Create PBR bind group
        self.create_pbr_bind_group(device);

        self.is_initialized = true;
        Ok(())
    }

    /// Update uniforms buffer
    fn update_uniforms(&self, queue: &wgpu::Queue) {
        queue.write_buffer(
            &self.uniforms_buffer,
            0,
            bytemuck::cast_slice(&[self.uniforms]),
        );
    }

    /// Create IBL bind group for prefiltering
    fn create_ibl_bind_group(&mut self, device: &wgpu::Device) {
        if let Some(ref env_view) = self.environment_view {
            self.ibl_bind_group = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("ibl_bind_group"),
                layout: &self.ibl_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: self.uniforms_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(env_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::Sampler(&self.environment_sampler),
                    },
                ],
            }));
        }
    }

    /// Create PBR bind group for runtime IBL
    fn create_pbr_bind_group(&mut self, device: &wgpu::Device) {
        if let (Some(ref irr_view), Some(ref spec_view), Some(ref brdf_view)) = (
            &self.irradiance_view,
            &self.specular_view,
            &self.brdf_lut_view,
        ) {
            self.pbr_bind_group = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("ibl_pbr_bind_group"),
                layout: &self.pbr_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(irr_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&self.irradiance_sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::TextureView(spec_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::Sampler(&self.specular_sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: wgpu::BindingResource::TextureView(brdf_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: wgpu::BindingResource::Sampler(&self.brdf_sampler),
                    },
                ],
            }));
        }
    }

    /// Set IBL quality
    pub fn set_quality(&mut self, quality: IBLQuality) {
        self.quality = quality;
        self.uniforms.brdf_size = quality.brdf_size();
        self.uniforms.max_mip_levels = quality.specular_mip_levels();
        self.is_initialized = false; // Need to regenerate
    }

    /// Get IBL bind group for PBR rendering
    pub fn pbr_bind_group(&self) -> Option<&wgpu::BindGroup> {
        self.pbr_bind_group.as_ref()
    }

    /// Check if IBL is initialized
    pub fn is_initialized(&self) -> bool {
        self.is_initialized
    }

    /// Get current quality setting
    pub fn quality(&self) -> IBLQuality {
        self.quality
    }

    /// Get IBL textures for inspection
    pub fn textures(
        &self,
    ) -> (
        Option<&wgpu::Texture>,
        Option<&wgpu::Texture>,
        Option<&wgpu::Texture>,
    ) {
        (
            self.irradiance_map.as_ref(),
            self.specular_map.as_ref(),
            self.brdf_lut.as_ref(),
        )
    }
}
