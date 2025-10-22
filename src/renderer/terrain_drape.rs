//! Terrain draping renderer: DEM displacement + categorical land-cover texture
//!
//! This module provides a GPU rasterization pipeline that:
//! 1. Displaces vertices using a DEM heightmap (R32F/R16F)
//! 2. Samples categorical land-cover colors (RGBA8, nearest filtering)
//! 3. Applies simple lighting for 3D relief visualization

use crate::core::uv_transform::UVTransform;
use crate::io::tex_upload::{upload_r32f_texture, upload_rgba8_texture, create_texture_2d};
use wgpu::*;
use wgpu::util::DeviceExt;

/// Camera and rendering globals (matches WGSL Globals struct)
/// Size: 336 bytes (expanded for shadow mapping support)
#[repr(C, align(16))]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct Globals {
    view: [[f32; 4]; 4],        // 64 bytes (offset 0)
    proj: [[f32; 4]; 4],        // 64 bytes (offset 64)
    camera_pos: [f32; 4],       // 16 bytes (offset 128) - xyz = position, w = unused
    
    // Packed parameters - vec4 alignment
    params0: [f32; 4],          // 16 bytes (offset 144) - [z_dir, zscale, light_type_f32, light_elevation]
    params1: [f32; 4],          // 16 bytes (offset 160) - [light_azimuth, light_intensity, ambient, shadow_intensity]
    params2: [f32; 4],          // 16 bytes (offset 176) - [gamma, fov, shadow_bias, enable_shadows]
    params3: [f32; 4],          // 16 bytes (offset 192) - [lighting_model, shininess, specular_strength, shadow_softness]
    params4: [f32; 4],          // 16 bytes (offset 208) - [background_r, background_g, background_b, background_a]
    params5: [f32; 4],          // 16 bytes (offset 224) - [shadow_map_res, tonemap_mode, gamma_correction, hdri_intensity]
    params6: [f32; 4],          // 16 bytes (offset 240) - [hdri_rotation_rad, enable_hdri, _unused, _unused]
    
    // Shadow mapping
    light_view_proj: [[f32; 4]; 4],  // 64 bytes (offset 256) - light space transform
    shadow_params: [f32; 4],         // 16 bytes (offset 320) - Reserved for additional shadow parameters
}

impl Default for Globals {
    fn default() -> Self {
        Self {
            view: glam::Mat4::IDENTITY.to_cols_array_2d(),
            proj: glam::Mat4::IDENTITY.to_cols_array_2d(),
            camera_pos: [0.0, 0.0, 0.0, 0.0],
            params0: [1.0, 1.0, 1.0, 45.0],  // [z_dir, zscale, light_type_f32, light_elevation]
            params1: [315.0, 1.0, 0.25, 0.6],  // [light_azimuth, light_intensity, ambient, shadow_intensity]
            params2: [0.0, 35.0, 0.0015, 1.0],  // [gamma, fov, shadow_bias, enable_shadows]
            params3: [2.0, 32.0, 0.3, 2.0],  // [lighting_model(blinn_phong), shininess, specular_strength, shadow_softness]
            params4: [1.0, 1.0, 1.0, 1.0],  // [background RGBA - white]
            params5: [2048.0, 1.0, 2.2, 1.0],  // [shadow_map_res, tonemap_mode(aces), gamma_correction, hdri_intensity]
            params6: [0.0, 0.0, 0.0, 0.0],  // [hdri_rotation_rad, enable_hdri, _unused, _unused]
            light_view_proj: glam::Mat4::IDENTITY.to_cols_array_2d(),
            shadow_params: [0.0, 0.0, 0.0, 0.0],
        }
    }
}

/// Terrain draping renderer configuration
pub struct TerrainDrapeConfig {
    pub width: u32,
    pub height: u32,
    pub sample_count: u32,
    
    // Displacement parameters
    pub z_dir: f32,
    pub zscale: f32,
    
    // Lighting parameters
    pub light_type: u32,        // 0=none, 1=directional, 2=hemisphere
    pub light_elevation: f32,   // degrees above horizon
    pub light_azimuth: f32,     // degrees from +X clockwise
    pub light_intensity: f32,
    pub ambient: f32,
    pub shadow_intensity: f32,
    
    // Advanced lighting
    pub lighting_model: u32,    // 0=lambert, 1=phong, 2=blinn_phong
    pub shininess: f32,         // Phong/Blinn-Phong exponent (1-256)
    pub specular_strength: f32, // Specular reflection strength (0-1)
    pub shadow_softness: f32,   // PCF kernel radius for soft shadows (1-5)
    
    // Camera parameters
    pub gamma: f32,             // camera roll in degrees
    pub fov: f32,               // field of view in degrees
    
    // Background color (RGBA, 0-1 range)
    pub background_color: [f32; 4],
    
    // Post-processing
    pub tonemap_mode: u32,      // 0=none, 1=aces, 2=reinhard
    pub gamma_correction: f32,  // Gamma value for output (typically 2.2)
    pub hdri_intensity: f32,    // HDRI environment intensity (0-2+)
    
    // Shadow mapping
    pub shadow_map_res: u32,    // Shadow map resolution (512-4096)
    pub shadow_bias: f32,       // Depth bias to prevent shadow acne (0.0001-0.01)
    pub enable_shadows: bool,   // Enable/disable shadow rendering
    
    // HDRI environment lighting
    pub hdri_path: Option<String>,  // Path to HDRI file (.hdr, .exr) or None
    pub hdri_rotation_deg: f32,     // Y-axis rotation in degrees (0-360)
    
    // Post-processing: denoising
    pub denoiser: String,           // "none", "oidn", or "bilateral"
    pub denoise_strength: f32,      // 0.0 - 1.0, denoising intensity
}

impl Default for TerrainDrapeConfig {
    fn default() -> Self {
        Self {
            width: 1280,
            height: 720,
            sample_count: 1,
            z_dir: 1.0,
            zscale: 1.0,
            light_type: 1,  // directional
            light_elevation: 45.0,
            light_azimuth: 315.0,
            light_intensity: 1.0,
            ambient: 0.25,
            shadow_intensity: 0.5,
            lighting_model: 2,  // blinn_phong
            shininess: 32.0,
            specular_strength: 0.3,
            shadow_softness: 2.0,
            gamma: 0.0,
            fov: 35.0,
            background_color: [1.0, 1.0, 1.0, 1.0],  // white
            tonemap_mode: 1,  // aces
            gamma_correction: 2.2,
            hdri_intensity: 1.0,
            shadow_map_res: 2048,
            shadow_bias: 0.0015,
            enable_shadows: true,
            hdri_path: None,
            hdri_rotation_deg: 0.0,
            denoiser: "oidn".to_string(),
            denoise_strength: 0.8,
        }
    }
}

/// Terrain draping renderer
pub struct TerrainDrapeRenderer {
    device: Device,
    queue: Queue,
    pipeline: RenderPipeline,
    shadow_pipeline: RenderPipeline,
    
    // Bind group layouts
    bgl_globals: BindGroupLayout,
    bgl_height: BindGroupLayout,
    bgl_landcover: BindGroupLayout,  // Includes HDRI (bindings 3-4)
    bgl_shadow: BindGroupLayout,
    bgl_shadow_pass: BindGroupLayout,  // For shadow pass globals
    
    // Render targets
    color_texture: Texture,
    color_view: TextureView,
    depth_texture: Texture,
    depth_view: TextureView,
    
    // Shadow mapping
    shadow_map: Texture,
    shadow_map_view: TextureView,
    shadow_sampler: Sampler,
    
    // Configuration
    config: TerrainDrapeConfig,
}

impl TerrainDrapeRenderer {
    /// Create a new terrain draping renderer
    pub async fn new(config: TerrainDrapeConfig) -> Result<Self, String> {
        // Create device and queue
        let instance = Instance::new(InstanceDescriptor {
            backends: Backends::all(),
            ..Default::default()
        });
        
        let adapter = instance
            .request_adapter(&RequestAdapterOptions {
                power_preference: PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .ok_or("Failed to find adapter")?;
        
        let (device, queue) = adapter
            .request_device(
                &DeviceDescriptor {
                    label: Some("terrain_drape_device"),
                    required_features: Features::empty(),
                    required_limits: Limits::default(),
                },
                None,
            )
            .await
            .map_err(|e| format!("Failed to create device: {}", e))?;
        
        // Create bind group layouts
        let bgl_globals = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("terrain_drape_bgl_globals"),
            entries: &[BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::VERTEX_FRAGMENT,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        
        // Check if device supports filterable R32Float
        let features = device.features();
        let height_filterable = features.contains(Features::FLOAT32_FILTERABLE);
        
        let bgl_height = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("terrain_drape_bgl_height"),
            entries: &[
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::VERTEX_FRAGMENT,
                    ty: BindingType::Texture {
                        sample_type: TextureSampleType::Float { filterable: height_filterable },
                        view_dimension: TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::VERTEX_FRAGMENT,
                    ty: BindingType::Sampler(if height_filterable {
                        SamplerBindingType::Filtering
                    } else {
                        SamplerBindingType::NonFiltering
                    }),
                    count: None,
                },
            ],
        });
        
        let bgl_landcover = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("terrain_drape_bgl_landcover"),
            entries: &[
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::FRAGMENT,
                    ty: BindingType::Texture {
                        sample_type: TextureSampleType::Float { filterable: true },
                        view_dimension: TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::FRAGMENT,
                    ty: BindingType::Sampler(SamplerBindingType::NonFiltering),
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 2,
                    visibility: ShaderStages::VERTEX_FRAGMENT,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // HDRI environment map (bindings 3-4)
                BindGroupLayoutEntry {
                    binding: 3,
                    visibility: ShaderStages::FRAGMENT,
                    ty: BindingType::Texture {
                        sample_type: TextureSampleType::Float { filterable: true },
                        view_dimension: TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 4,
                    visibility: ShaderStages::FRAGMENT,
                    ty: BindingType::Sampler(SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        
        // Shadow mapping bind group layout
        let bgl_shadow = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("terrain_drape_bgl_shadow"),
            entries: &[
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::FRAGMENT,
                    ty: BindingType::Texture {
                        sample_type: TextureSampleType::Depth,
                        view_dimension: TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::FRAGMENT,
                    ty: BindingType::Sampler(SamplerBindingType::Comparison),
                    count: None,
                },
            ],
        });
        
        // Shadow pass globals bind group layout (simpler than main globals)
        let bgl_shadow_pass = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("terrain_drape_bgl_shadow_pass"),
            entries: &[BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::VERTEX,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        
        // Create main pipeline layout (HDRI now in Group 2 landcover)
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("terrain_drape_pipeline_layout"),
            bind_group_layouts: &[&bgl_globals, &bgl_height, &bgl_landcover, &bgl_shadow],
            push_constant_ranges: &[],
        });
        
        // Create shadow pass pipeline layout (only needs globals and height texture)
        let shadow_pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("terrain_drape_shadow_pipeline_layout"),
            bind_group_layouts: &[&bgl_shadow_pass, &bgl_height],
            push_constant_ranges: &[],
        });
        
        // Load main shader with WGSL preprocessing for includes
        let shader_source = include_str!("../../shaders/terrain_drape.wgsl");
        let pcf_source = include_str!("../../shaders/common/shadows_pcf.wgsl");
        let hdri_source = include_str!("../../shaders/common/hdri.wgsl");
        // Simple preprocessor: replace #include with actual file content
        let processed_shader = shader_source
            .replace(
                "#include \"common/shadows_pcf.wgsl\"",
                pcf_source
            )
            .replace(
                "#include \"common/hdri.wgsl\"",
                hdri_source
            );
        let shader_module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("terrain_drape_shader"),
            source: ShaderSource::Wgsl(processed_shader.into()),
        });
        
        // Load shadow pass shader
        let shadow_shader_source = include_str!("../../shaders/terrain_drape_shadow.wgsl");
        let shadow_shader_module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("terrain_drape_shadow_shader"),
            source: ShaderSource::Wgsl(shadow_shader_source.into()),
        });
        
        // Create render pipeline
        let pipeline = device.create_render_pipeline(&RenderPipelineDescriptor {
            label: Some("terrain_drape_pipeline"),
            layout: Some(&pipeline_layout),
            vertex: VertexState {
                module: &shader_module,
                entry_point: "vs_main",
                buffers: &[
                    // Vertex position (XZ plane)
                    VertexBufferLayout {
                        array_stride: 8, // 2 * f32
                        step_mode: VertexStepMode::Vertex,
                        attributes: &vertex_attr_array![0 => Float32x2],
                    },
                    // UV coordinates
                    VertexBufferLayout {
                        array_stride: 8, // 2 * f32
                        step_mode: VertexStepMode::Vertex,
                        attributes: &vertex_attr_array![1 => Float32x2],
                    },
                ],
            },
            primitive: PrimitiveState {
                topology: PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: FrontFace::Ccw,
                cull_mode: Some(Face::Back),
                unclipped_depth: false,
                polygon_mode: PolygonMode::Fill,
                conservative: false,
            },
            depth_stencil: Some(DepthStencilState {
                format: TextureFormat::Depth32Float,
                depth_write_enabled: true,
                depth_compare: CompareFunction::Less,
                stencil: StencilState::default(),
                bias: DepthBiasState::default(),
            }),
            multisample: MultisampleState {
                count: config.sample_count,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            fragment: Some(FragmentState {
                module: &shader_module,
                entry_point: "fs_main",
                targets: &[Some(ColorTargetState {
                    format: TextureFormat::Rgba8UnormSrgb,
                    blend: Some(BlendState::ALPHA_BLENDING),
                    write_mask: ColorWrites::ALL,
                })],
            }),
            multiview: None,
        });
        
        // Create shadow pass pipeline (depth-only rendering)
        let shadow_pipeline = device.create_render_pipeline(&RenderPipelineDescriptor {
            label: Some("terrain_drape_shadow_pipeline"),
            layout: Some(&shadow_pipeline_layout),
            vertex: VertexState {
                module: &shadow_shader_module,
                entry_point: "vs_main",
                buffers: &[
                    // Same vertex layout as main pass
                    VertexBufferLayout {
                        array_stride: 8,
                        step_mode: VertexStepMode::Vertex,
                        attributes: &vertex_attr_array![0 => Float32x2],
                    },
                    VertexBufferLayout {
                        array_stride: 8,
                        step_mode: VertexStepMode::Vertex,
                        attributes: &vertex_attr_array![1 => Float32x2],
                    },
                ],
            },
            primitive: PrimitiveState {
                topology: PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: FrontFace::Ccw,
                cull_mode: Some(Face::Back),
                unclipped_depth: false,
                polygon_mode: PolygonMode::Fill,
                conservative: false,
            },
            depth_stencil: Some(DepthStencilState {
                format: TextureFormat::Depth32Float,
                depth_write_enabled: true,
                depth_compare: CompareFunction::Less,
                stencil: StencilState::default(),
                bias: DepthBiasState::default(),
            }),
            multisample: MultisampleState::default(),
            fragment: None,  // Depth-only pass, no fragment shader needed
            multiview: None,
        });
        
        // Create render targets
        let color_texture = create_texture_2d(
            &device,
            "terrain_drape_color",
            config.width,
            config.height,
            TextureFormat::Rgba8UnormSrgb,
            TextureUsages::RENDER_ATTACHMENT | TextureUsages::COPY_SRC,
        );
        let color_view = color_texture.create_view(&TextureViewDescriptor::default());
        
        let depth_texture = create_texture_2d(
            &device,
            "terrain_drape_depth",
            config.width,
            config.height,
            TextureFormat::Depth32Float,
            TextureUsages::RENDER_ATTACHMENT,
        );
        let depth_view = depth_texture.create_view(&TextureViewDescriptor::default());
        
        // Create shadow map texture
        let shadow_map = create_texture_2d(
            &device,
            "terrain_drape_shadow_map",
            config.shadow_map_res,
            config.shadow_map_res,
            TextureFormat::Depth32Float,
            TextureUsages::RENDER_ATTACHMENT | TextureUsages::TEXTURE_BINDING,
        );
        let shadow_map_view = shadow_map.create_view(&TextureViewDescriptor::default());
        
        // Create comparison sampler for shadow mapping
        let shadow_sampler = device.create_sampler(&SamplerDescriptor {
            label: Some("shadow_comparison_sampler"),
            address_mode_u: AddressMode::ClampToEdge,
            address_mode_v: AddressMode::ClampToEdge,
            address_mode_w: AddressMode::ClampToEdge,
            mag_filter: FilterMode::Linear,
            min_filter: FilterMode::Linear,
            mipmap_filter: FilterMode::Nearest,
            compare: Some(CompareFunction::LessEqual),  // Hardware PCF comparison
            ..Default::default()
        });
        
        Ok(Self {
            device,
            queue,
            pipeline,
            shadow_pipeline,
            bgl_globals,
            bgl_height,
            bgl_landcover,
            bgl_shadow,
            bgl_shadow_pass,
            color_texture,
            color_view,
            depth_texture,
            depth_view,
            shadow_map,
            shadow_map_view,
            shadow_sampler,
            config,
        })
    }
    
    /// Render terrain with DEM and land-cover textures
    pub fn render(
        &self,
        heightmap: &[f32],
        heightmap_width: u32,
        heightmap_height: u32,
        landcover: &[u8],
        landcover_width: u32,
        landcover_height: u32,
        uv_transform: &UVTransform,
        camera_view: &glam::Mat4,
        camera_proj: &glam::Mat4,
        camera_pos: &glam::Vec3,
    ) -> Result<Vec<u8>, String> {
        // Create and upload height texture
        let height_texture = create_texture_2d(
            &self.device,
            "dem_height",
            heightmap_width,
            heightmap_height,
            TextureFormat::R32Float,
            TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
        );
        upload_r32f_texture(&self.queue, &height_texture, heightmap, heightmap_width, heightmap_height)?;
        let height_view = height_texture.create_view(&TextureViewDescriptor::default());
        
        // Create and upload land-cover texture
        let landcover_texture = create_texture_2d(
            &self.device,
            "landcover",
            landcover_width,
            landcover_height,
            TextureFormat::Rgba8UnormSrgb,
            TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
        );
        upload_rgba8_texture(&self.queue, &landcover_texture, landcover, landcover_width, landcover_height)?;
        let landcover_view = landcover_texture.create_view(&TextureViewDescriptor::default());
        
        // Create samplers
        // Use linear filtering if supported, otherwise nearest
        let features = self.device.features();
        let can_filter = features.contains(Features::FLOAT32_FILTERABLE);
        let filter_mode = if can_filter { FilterMode::Linear } else { FilterMode::Nearest };
        
        let height_sampler = self.device.create_sampler(&SamplerDescriptor {
            label: Some("height_sampler"),
            address_mode_u: AddressMode::ClampToEdge,
            address_mode_v: AddressMode::ClampToEdge,
            address_mode_w: AddressMode::ClampToEdge,
            mag_filter: filter_mode,
            min_filter: filter_mode,
            mipmap_filter: filter_mode,
            ..Default::default()
        });
        
        let landcover_sampler = self.device.create_sampler(&SamplerDescriptor {
            label: Some("landcover_sampler"),
            address_mode_u: AddressMode::ClampToEdge,
            address_mode_v: AddressMode::ClampToEdge,
            address_mode_w: AddressMode::ClampToEdge,
            mag_filter: FilterMode::Nearest,
            min_filter: FilterMode::Nearest,
            mipmap_filter: FilterMode::Nearest,
            ..Default::default()
        });
        
        // Load or create HDRI environment map
        let (_hdri_texture, hdri_view, enable_hdri) = if let Some(hdri_path) = &self.config.hdri_path {
            // Load HDRI file
            use crate::formats::hdr::load_hdr;
            match load_hdr(hdri_path) {
                Ok(hdr_image) => {
                    let hdri_texture = create_texture_2d(
                        &self.device,
                        "hdri_environment",
                        hdr_image.width,
                        hdr_image.height,
                        TextureFormat::Rgba16Float,  // Rgba16Float is filterable on most GPUs
                        TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
                    );
                    
                    // Convert RGB f32 to RGBA f16 (half precision) for upload
                    let rgba_data = hdr_image.to_rgba();
                    let rgba16_data: Vec<half::f16> = rgba_data.iter()
                        .map(|&f| half::f16::from_f32(f))
                        .collect();
                    
                    // Convert f16 to bytes manually (f16 doesn't implement Pod)
                    let rgba16_bytes: Vec<u8> = rgba16_data.iter()
                        .flat_map(|f| f.to_le_bytes())
                        .collect();
                    
                    self.queue.write_texture(
                        ImageCopyTexture {
                            texture: &hdri_texture,
                            mip_level: 0,
                            origin: Origin3d::ZERO,
                            aspect: TextureAspect::All,
                        },
                        &rgba16_bytes,
                        ImageDataLayout {
                            offset: 0,
                            bytes_per_row: Some(hdr_image.width * 8), // 4 channels * 2 bytes (f16)
                            rows_per_image: Some(hdr_image.height),
                        },
                        Extent3d {
                            width: hdr_image.width,
                            height: hdr_image.height,
                            depth_or_array_layers: 1,
                        },
                    );
                    
                    let hdri_view = hdri_texture.create_view(&TextureViewDescriptor::default());
                    (hdri_texture, hdri_view, true)
                },
                Err(e) => {
                    eprintln!("Warning: Failed to load HDRI '{}': {}. Using fallback.", hdri_path, e);
                    // Create fallback 1x1 neutral gray texture
                    let fallback_texture = create_texture_2d(
                        &self.device,
                        "hdri_fallback",
                        1,
                        1,
                        TextureFormat::Rgba16Float,
                        TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
                    );
                    let gray_pixel: [half::f16; 4] = [
                        half::f16::from_f32(0.5),
                        half::f16::from_f32(0.5),
                        half::f16::from_f32(0.5),
                        half::f16::from_f32(1.0),
                    ];
                    let gray_bytes: Vec<u8> = gray_pixel.iter()
                        .flat_map(|f| f.to_le_bytes())
                        .collect();
                    self.queue.write_texture(
                        ImageCopyTexture {
                            texture: &fallback_texture,
                            mip_level: 0,
                            origin: Origin3d::ZERO,
                            aspect: TextureAspect::All,
                        },
                        &gray_bytes,
                        ImageDataLayout {
                            offset: 0,
                            bytes_per_row: Some(8),  // 4 channels * 2 bytes
                            rows_per_image: Some(1),
                        },
                        Extent3d {
                            width: 1,
                            height: 1,
                            depth_or_array_layers: 1,
                        },
                    );
                    let fallback_view = fallback_texture.create_view(&TextureViewDescriptor::default());
                    (fallback_texture, fallback_view, false)
                }
            }
        } else {
            // No HDRI specified, create 1x1 neutral texture
            let fallback_texture = create_texture_2d(
                &self.device,
                "hdri_none",
                1,
                1,
                TextureFormat::Rgba16Float,
                TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
            );
            let gray_pixel: [half::f16; 4] = [
                half::f16::from_f32(0.5),
                half::f16::from_f32(0.5),
                half::f16::from_f32(0.5),
                half::f16::from_f32(1.0),
            ];
            let gray_bytes: Vec<u8> = gray_pixel.iter()
                .flat_map(|f| f.to_le_bytes())
                .collect();
            self.queue.write_texture(
                ImageCopyTexture {
                    texture: &fallback_texture,
                    mip_level: 0,
                    origin: Origin3d::ZERO,
                    aspect: TextureAspect::All,
                },
                &gray_bytes,
                ImageDataLayout {
                    offset: 0,
                    bytes_per_row: Some(8),  // 4 channels * 2 bytes
                    rows_per_image: Some(1),
                },
                Extent3d {
                    width: 1,
                    height: 1,
                    depth_or_array_layers: 1,
                },
            );
            let fallback_view = fallback_texture.create_view(&TextureViewDescriptor::default());
            (fallback_texture, fallback_view, false)
        };
        
        // Create HDRI sampler (linear filtering for environment lookups)
        let hdri_sampler = self.device.create_sampler(&SamplerDescriptor {
            label: Some("hdri_sampler"),
            address_mode_u: AddressMode::Repeat,  // Wrap horizontally (longitude)
            address_mode_v: AddressMode::ClampToEdge,  // Clamp vertically (latitude)
            address_mode_w: AddressMode::ClampToEdge,
            mag_filter: FilterMode::Linear,
            min_filter: FilterMode::Linear,
            mipmap_filter: FilterMode::Linear,
            ..Default::default()
        });
        
        // Calculate light view-projection matrix for shadow mapping
        let light_view_proj = if self.config.enable_shadows {
            // Compute light direction from elevation and azimuth
            let light_elev_rad = self.config.light_elevation.to_radians();
            let light_azim_rad = self.config.light_azimuth.to_radians();
            let light_dir = glam::Vec3::new(
                light_elev_rad.cos() * light_azim_rad.cos(),
                light_elev_rad.sin(),
                -light_elev_rad.cos() * light_azim_rad.sin(),
            ).normalize();
            
            // Compute terrain bounds for light frustum
            let terrain_width = heightmap_width as f32;
            let terrain_height = heightmap_height as f32;
            let terrain_extent = terrain_width.max(terrain_height);
            
            // Position light to cover entire terrain
            let light_distance = terrain_extent * 2.0;
            let light_pos = -light_dir * light_distance;
            
            // Create orthographic projection for directional light
            let half_extent = terrain_extent * 0.75;
            let light_view = glam::Mat4::look_at_rh(
                light_pos,
                glam::Vec3::ZERO,
                glam::Vec3::Y,
            );
            let light_proj = glam::Mat4::orthographic_rh(
                -half_extent, half_extent,
                -half_extent, half_extent,
                0.1, light_distance * 2.5,
            );
            
            light_proj * light_view
        } else {
            glam::Mat4::IDENTITY
        };
        
        // Create globals uniform (packed into vec4s for std140 alignment)
        let globals = Globals {
            view: camera_view.to_cols_array_2d(),
            proj: camera_proj.to_cols_array_2d(),
            camera_pos: [camera_pos.x, camera_pos.y, camera_pos.z, 0.0],
            params0: [
                self.config.z_dir,
                self.config.zscale,
                self.config.light_type as f32,
                self.config.light_elevation,
            ],
            params1: [
                self.config.light_azimuth,
                self.config.light_intensity,
                self.config.ambient,
                self.config.shadow_intensity,
            ],
            params2: [
                self.config.gamma,
                self.config.fov,
                self.config.shadow_bias,
                if self.config.enable_shadows { 1.0 } else { 0.0 },
            ],
            params3: [
                self.config.lighting_model as f32,
                self.config.shininess,
                self.config.specular_strength,
                self.config.shadow_softness,
            ],
            params4: self.config.background_color,
            params5: [
                self.config.shadow_map_res as f32,
                self.config.tonemap_mode as f32,
                self.config.gamma_correction,
                self.config.hdri_intensity,
            ],
            params6: [
                self.config.hdri_rotation_deg.to_radians(),
                if enable_hdri { 1.0 } else { 0.0 },
                0.0,
                0.0,
            ],
            light_view_proj: light_view_proj.to_cols_array_2d(),
            shadow_params: [0.0, 0.0, 0.0, 0.0],
        };
        let globals_buffer = self.device.create_buffer_init(&util::BufferInitDescriptor {
            label: Some("globals_uniform"),
            contents: bytemuck::bytes_of(&globals),
            usage: BufferUsages::UNIFORM,
        });
        
        // Create UV transform uniform
        let uv_buffer = uv_transform.create_buffer(&self.device);
        
        // Create bind groups
        let bg_globals = self.device.create_bind_group(&BindGroupDescriptor {
            label: Some("bg_globals"),
            layout: &self.bgl_globals,
            entries: &[BindGroupEntry {
                binding: 0,
                resource: globals_buffer.as_entire_binding(),
            }],
        });
        
        let bg_height = self.device.create_bind_group(&BindGroupDescriptor {
            label: Some("bg_height"),
            layout: &self.bgl_height,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::TextureView(&height_view),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::Sampler(&height_sampler),
                },
            ],
        });
        
        let bg_landcover = self.device.create_bind_group(&BindGroupDescriptor {
            label: Some("bg_landcover"),
            layout: &self.bgl_landcover,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::TextureView(&landcover_view),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::Sampler(&landcover_sampler),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: uv_buffer.as_entire_binding(),
                },
                // HDRI environment map (bindings 3-4)
                BindGroupEntry {
                    binding: 3,
                    resource: BindingResource::TextureView(&hdri_view),
                },
                BindGroupEntry {
                    binding: 4,
                    resource: BindingResource::Sampler(&hdri_sampler),
                },
            ],
        });
        
        // Shadow bind group
        let bg_shadow = self.device.create_bind_group(&BindGroupDescriptor {
            label: Some("bg_shadow"),
            layout: &self.bgl_shadow,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::TextureView(&self.shadow_map_view),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::Sampler(&self.shadow_sampler),
                },
            ],
        });
        
        // Generate terrain mesh (simple grid)
        // Use terrain dimensions based on actual data size
        let terrain_width = heightmap_width as f32;
        let terrain_height = heightmap_height as f32;
        let (vertices, uvs, indices) = generate_terrain_grid(128, 128, terrain_width, terrain_height);
        
        let vertex_buffer = self.device.create_buffer_init(&util::BufferInitDescriptor {
            label: Some("vertex_buffer"),
            contents: bytemuck::cast_slice(&vertices),
            usage: BufferUsages::VERTEX,
        });
        
        let uv_buffer_vtx = self.device.create_buffer_init(&util::BufferInitDescriptor {
            label: Some("uv_buffer"),
            contents: bytemuck::cast_slice(&uvs),
            usage: BufferUsages::VERTEX,
        });
        
        let index_buffer = self.device.create_buffer_init(&util::BufferInitDescriptor {
            label: Some("index_buffer"),
            contents: bytemuck::cast_slice(&indices),
            usage: BufferUsages::INDEX,
        });
        
        // Create command encoder
        let mut encoder = self.device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("terrain_drape_encoder"),
        });
        
        // Shadow pass (if enabled)
        if self.config.enable_shadows {
            // Create shadow pass globals uniform (includes UV transform)
            #[repr(C, align(16))]
            #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
            struct ShadowGlobals {
                light_view_proj: [[f32; 4]; 4],  // 64 bytes
                z_dir: f32,                       // 4 bytes
                zscale: f32,                      // 4 bytes
                uv_scale_x: f32,                  // 4 bytes
                uv_scale_y: f32,                  // 4 bytes
                uv_offset_x: f32,                 // 4 bytes
                uv_offset_y: f32,                 // 4 bytes
                y_flip: u32,                      // 4 bytes
                _pad: u32,                        // 4 bytes
            }
            
            let shadow_globals = ShadowGlobals {
                light_view_proj: light_view_proj.to_cols_array_2d(),
                z_dir: self.config.z_dir,
                zscale: self.config.zscale,
                uv_scale_x: uv_transform.scale[0],
                uv_scale_y: uv_transform.scale[1],
                uv_offset_x: uv_transform.offset[0],
                uv_offset_y: uv_transform.offset[1],
                y_flip: uv_transform.y_flip as u32,
                _pad: 0,
            };
            
            let shadow_globals_buffer = self.device.create_buffer_init(&util::BufferInitDescriptor {
                label: Some("shadow_globals_uniform"),
                contents: bytemuck::bytes_of(&shadow_globals),
                usage: BufferUsages::UNIFORM,
            });
            
            let bg_shadow_pass = self.device.create_bind_group(&BindGroupDescriptor {
                label: Some("bg_shadow_pass"),
                layout: &self.bgl_shadow_pass,
                entries: &[BindGroupEntry {
                    binding: 0,
                    resource: shadow_globals_buffer.as_entire_binding(),
                }],
            });
            
            // Shadow render pass (depth-only)
            {
                let mut shadow_pass = encoder.begin_render_pass(&RenderPassDescriptor {
                    label: Some("terrain_drape_shadow_pass"),
                    color_attachments: &[],
                    depth_stencil_attachment: Some(RenderPassDepthStencilAttachment {
                        view: &self.shadow_map_view,
                        depth_ops: Some(Operations {
                            load: LoadOp::Clear(1.0),
                            store: StoreOp::Store,
                        }),
                        stencil_ops: None,
                    }),
                    ..Default::default()
                });
                
                shadow_pass.set_pipeline(&self.shadow_pipeline);
                shadow_pass.set_bind_group(0, &bg_shadow_pass, &[]);
                shadow_pass.set_bind_group(1, &bg_height, &[]);
                shadow_pass.set_vertex_buffer(0, vertex_buffer.slice(..));
                shadow_pass.set_vertex_buffer(1, uv_buffer_vtx.slice(..));
                shadow_pass.set_index_buffer(index_buffer.slice(..), IndexFormat::Uint32);
                shadow_pass.draw_indexed(0..indices.len() as u32, 0, 0..1);
            }
        }
        
        // Main render pass
        
        {
            let mut rpass = encoder.begin_render_pass(&RenderPassDescriptor {
                label: Some("terrain_drape_pass"),
                color_attachments: &[Some(RenderPassColorAttachment {
                    view: &self.color_view,
                    resolve_target: None,
                    ops: Operations {
                        load: LoadOp::Clear(Color {
                            r: self.config.background_color[0] as f64,
                            g: self.config.background_color[1] as f64,
                            b: self.config.background_color[2] as f64,
                            a: self.config.background_color[3] as f64,
                        }),
                        store: StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(RenderPassDepthStencilAttachment {
                    view: &self.depth_view,
                    depth_ops: Some(Operations {
                        load: LoadOp::Clear(1.0),
                        store: StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                ..Default::default()
            });
            
            rpass.set_pipeline(&self.pipeline);
            rpass.set_bind_group(0, &bg_globals, &[]);
            rpass.set_bind_group(1, &bg_height, &[]);
            rpass.set_bind_group(2, &bg_landcover, &[]);  // Includes HDRI at bindings 3-4
            rpass.set_bind_group(3, &bg_shadow, &[]);
            rpass.set_vertex_buffer(0, vertex_buffer.slice(..));
            rpass.set_vertex_buffer(1, uv_buffer_vtx.slice(..));
            rpass.set_index_buffer(index_buffer.slice(..), IndexFormat::Uint32);
            rpass.draw_indexed(0..indices.len() as u32, 0, 0..1);
        }
        
        self.queue.submit(Some(encoder.finish()));
        
        // Readback (simplified - would need proper async handling in production)
        let bytes_per_row = self.config.width * 4;
        let padded_bytes_per_row = crate::io::tex_upload::padded_bytes_per_row(bytes_per_row);
        let buffer_size = (padded_bytes_per_row * self.config.height) as u64;
        
        let readback_buffer = self.device.create_buffer(&BufferDescriptor {
            label: Some("readback_buffer"),
            size: buffer_size,
            usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        
        let mut encoder = self.device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("readback_encoder"),
        });
        
        encoder.copy_texture_to_buffer(
            ImageCopyTexture {
                texture: &self.color_texture,
                mip_level: 0,
                origin: Origin3d::ZERO,
                aspect: TextureAspect::All,
            },
            ImageCopyBuffer {
                buffer: &readback_buffer,
                layout: ImageDataLayout {
                    offset: 0,
                    bytes_per_row: Some(padded_bytes_per_row),
                    rows_per_image: Some(self.config.height),
                },
            },
            Extent3d {
                width: self.config.width,
                height: self.config.height,
                depth_or_array_layers: 1,
            },
        );
        
        self.queue.submit(Some(encoder.finish()));
        
        // Map and read
        let buffer_slice = readback_buffer.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        buffer_slice.map_async(MapMode::Read, move |result| {
            tx.send(result).ok();
        });
        self.device.poll(Maintain::Wait);
        rx.recv().unwrap().map_err(|e| format!("Buffer map failed: {:?}", e))?;
        
        let data = buffer_slice.get_mapped_range();
        let mut result = vec![0u8; (self.config.width * self.config.height * 4) as usize];
        
        for y in 0..self.config.height {
            let src_offset = (y * padded_bytes_per_row) as usize;
            let dst_offset = (y * bytes_per_row) as usize;
            result[dst_offset..dst_offset + bytes_per_row as usize]
                .copy_from_slice(&data[src_offset..src_offset + bytes_per_row as usize]);
        }
        
        drop(data);
        readback_buffer.unmap();
        
        // Apply denoising if requested
        let denoiser_type = crate::post::DenoiserType::from_str(&self.config.denoiser)
            .unwrap_or(crate::post::DenoiserType::None);
        
        let denoise_config = crate::post::DenoiseConfig {
            denoiser: denoiser_type,
            strength: self.config.denoise_strength,
        };
        
        let final_result = if denoiser_type != crate::post::DenoiserType::None {
            crate::post::denoise_rgba(&result, self.config.width, self.config.height, &denoise_config)
                .map_err(|e| format!("Denoising failed: {}", e))?
        } else {
            result
        };
        
        Ok(final_result)
    }
}

/// Generate a simple terrain grid mesh
fn generate_terrain_grid(
    grid_w: u32,
    grid_h: u32,
    size_x: f32,
    size_z: f32,
) -> (Vec<[f32; 2]>, Vec<[f32; 2]>, Vec<u32>) {
    let mut vertices = Vec::new();
    let mut uvs = Vec::new();
    let mut indices = Vec::new();
    
    // Generate vertices and UVs
    for y in 0..=grid_h {
        for x in 0..=grid_w {
            let u = x as f32 / grid_w as f32;
            let v = y as f32 / grid_h as f32;
            
            let pos_x = (u - 0.5) * size_x;
            let pos_z = (v - 0.5) * size_z;
            
            vertices.push([pos_x, pos_z]);
            uvs.push([u, v]);
        }
    }
    
    // Generate indices (two triangles per quad)
    for y in 0..grid_h {
        for x in 0..grid_w {
            let i0 = y * (grid_w + 1) + x;
            let i1 = i0 + 1;
            let i2 = i0 + (grid_w + 1);
            let i3 = i2 + 1;
            
            // Triangle 1
            indices.push(i0);
            indices.push(i2);
            indices.push(i1);
            
            // Triangle 2
            indices.push(i1);
            indices.push(i2);
            indices.push(i3);
        }
    }
    
    (vertices, uvs, indices)
}
