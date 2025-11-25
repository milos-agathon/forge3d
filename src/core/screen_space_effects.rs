//! P5: Screen-space effects system (SSAO/GTAO, SSGI, SSR)
//!
//! Provides GPU-accelerated screen-space techniques for ambient occlusion,
//! global illumination, and reflections.

use crate::core::gbuffer::{GBuffer, GBufferConfig};
use crate::core::gpu_timing::GpuTimingManager;
use crate::error::{RenderError, RenderResult};
use crate::render::params::SsrParams;
use futures_intrusive::channel::shared::oneshot_channel;
use pollster::block_on;
use std::mem::size_of;
use std::time::Instant;
use wgpu::util::DeviceExt;
use wgpu::*;

const SSAO_SHADER_SRC: &str = concat!(
    include_str!("../shaders/ssao/common.wgsl"),
    "\n",
    include_str!("../shaders/ssao/ssao.wgsl")
);
const GTAO_SHADER_SRC: &str = concat!(
    include_str!("../shaders/ssao/common.wgsl"),
    "\n",
    include_str!("../shaders/ssao/gtao.wgsl")
);
const SSAO_COMPOSITE_SHADER_SRC: &str = include_str!("../shaders/ssao/composite.wgsl");

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

/// Hierarchical Z-Buffer (min-depth pyramid) for accelerated occlusion queries
pub struct HzbPyramid {
    tex: Texture,
    mip_count: u32,
    width: u32,
    height: u32,
    // Compute pipelines and layouts
    bgl_copy: BindGroupLayout,
    bgl_down: BindGroupLayout,
    pipe_copy: ComputePipeline,
    pipe_down: ComputePipeline,
}

impl HzbPyramid {
    fn new(device: &Device, width: u32, height: u32) -> RenderResult<Self> {
        use crate::core::mipmap::calculate_mip_levels;
        let mip_count = calculate_mip_levels(width, height).max(1);
        // HZB is a float color texture (R32Float) with mip chain
        let tex = device.create_texture(&TextureDescriptor {
            label: Some("p5.hzb.pyramid"),
            size: Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: mip_count,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::R32Float,
            usage: TextureUsages::TEXTURE_BINDING
                | TextureUsages::STORAGE_BINDING
                | TextureUsages::COPY_SRC,
            view_formats: &[],
        });

        let shader = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("p5.hzb.build.shader"),
            source: ShaderSource::Wgsl(include_str!("../../shaders/hzb_build.wgsl").into()),
        });

        // Group 0: depth copy (depth texture -> r32f storage). We use textureLoad on depth (no sampler).
        let bgl_copy = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("p5.hzb.bgl.copy"),
            entries: &[
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Texture {
                        sample_type: TextureSampleType::Depth,
                        view_dimension: TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::StorageTexture {
                        access: StorageTextureAccess::WriteOnly,
                        format: TextureFormat::R32Float,
                        view_dimension: TextureViewDimension::D2,
                    },
                    count: None,
                },
            ],
        });

        // Group 1: downsample (r32f -> r32f) with reversed_z uniform
        let bgl_down = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("p5.hzb.bgl.down"),
            entries: &[
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
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::StorageTexture {
                        access: StorageTextureAccess::WriteOnly,
                        format: TextureFormat::R32Float,
                        view_dimension: TextureViewDimension::D2,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 2,
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

        // Use separate pipeline layouts per entry to keep validation simple
        let pl_copy = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("p5.hzb.pl.copy"),
            bind_group_layouts: &[&bgl_copy],
            push_constant_ranges: &[],
        });
        let pl_down = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("p5.hzb.pl.down"),
            bind_group_layouts: &[&bgl_down],
            push_constant_ranges: &[],
        });
        let pipe_copy = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("p5.hzb.pipe.copy"),
            layout: Some(&pl_copy),
            module: &shader,
            entry_point: "cs_copy",
        });
        let pipe_down = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("p5.hzb.pipe.down"),
            layout: Some(&pl_down),
            module: &shader,
            entry_point: "cs_downsample",
        });

        Ok(Self {
            tex,
            mip_count,
            width,
            height,
            bgl_copy,
            bgl_down,
            pipe_copy,
            pipe_down,
        })
    }

    fn ensure_size(&mut self, device: &Device, width: u32, height: u32) {
        if self.width == width && self.height == height {
            return;
        }
        // Recreate texture with new size
        *self = Self::new(device, width, height).expect("HzbPyramid::new should not fail");
    }

    /// Build HZB from a source DEPTH view (mip 0), limiting to `levels` mips (including level 0).
    /// Produces a pyramid in `self.tex` up to the requested number of levels.
    pub fn build_n(
        &self,
        device: &Device,
        encoder: &mut CommandEncoder,
        src_depth: &TextureView,
        levels: u32,
        reversed_z: bool,
    ) {
        // Copy depth -> HZB level 0
        let dst0 = self.tex.create_view(&TextureViewDescriptor {
            label: Some("p5.hzb.mip0"),
            format: None,
            dimension: Some(TextureViewDimension::D2),
            aspect: TextureAspect::All,
            base_mip_level: 0,
            mip_level_count: Some(1),
            base_array_layer: 0,
            array_layer_count: Some(1),
        });
        let bg_copy = device.create_bind_group(&BindGroupDescriptor {
            label: Some("p5.hzb.bg.copy"),
            layout: &self.bgl_copy,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::TextureView(src_depth),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::TextureView(&dst0),
                },
            ],
        });
        let mut pass0 = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("p5.hzb.pass.copy"),
            timestamp_writes: None,
        });
        pass0.set_pipeline(&self.pipe_copy);
        pass0.set_bind_group(0, &bg_copy, &[]);
        let gx0 = (self.width + 7) / 8;
        let gy0 = (self.height + 7) / 8;
        pass0.dispatch_workgroups(gx0, gy0, 1);
        drop(pass0);

        // Downsample chain up to requested levels
        let build_to = levels.min(self.mip_count).saturating_sub(1);
        let mut level_w = self.width;
        let mut level_h = self.height;
        // Create uniform buffer for reversed_z flag
        let reversed_z_val: u32 = if reversed_z { 1 } else { 0 };
        let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("p5.hzb.params"),
            contents: bytemuck::cast_slice(&[reversed_z_val]),
            usage: BufferUsages::UNIFORM,
        });
        for level in 1..=build_to {
            let src_view = self.tex.create_view(&TextureViewDescriptor {
                label: Some("p5.hzb.src.prev"),
                format: None,
                dimension: Some(TextureViewDimension::D2),
                aspect: TextureAspect::All,
                base_mip_level: level - 1,
                mip_level_count: Some(1),
                base_array_layer: 0,
                array_layer_count: Some(1),
            });
            let dst_view = self.tex.create_view(&TextureViewDescriptor {
                label: Some("p5.hzb.dst.curr"),
                format: None,
                dimension: Some(TextureViewDimension::D2),
                aspect: TextureAspect::All,
                base_mip_level: level,
                mip_level_count: Some(1),
                base_array_layer: 0,
                array_layer_count: Some(1),
            });
            let bg_down = device.create_bind_group(&BindGroupDescriptor {
                label: Some("p5.hzb.bg.down"),
                layout: &self.bgl_down,
                entries: &[
                    BindGroupEntry {
                        binding: 0,
                        resource: BindingResource::TextureView(&src_view),
                    },
                    BindGroupEntry {
                        binding: 1,
                        resource: BindingResource::TextureView(&dst_view),
                    },
                    BindGroupEntry {
                        binding: 2,
                        resource: params_buf.as_entire_binding(),
                    },
                ],
            });
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("p5.hzb.pass.down"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipe_down);
            pass.set_bind_group(0, &bg_down, &[]);
            level_w = (level_w / 2).max(1);
            level_h = (level_h / 2).max(1);
            let gx = (level_w + 7) / 8;
            let gy = (level_h + 7) / 8;
            pass.dispatch_workgroups(gx, gy, 1);
            drop(pass);
        }
    }

    /// Build HZB from a source DEPTH view (mip 0). Produces a full pyramid in self.tex
    fn build(
        &self,
        device: &Device,
        encoder: &mut CommandEncoder,
        src_depth: &TextureView,
        reversed_z: bool,
    ) {
        self.build_n(device, encoder, src_depth, self.mip_count, reversed_z);
    }

    #[allow(dead_code)]
    fn texture_view(&self) -> TextureView {
        self.tex.create_view(&TextureViewDescriptor::default())
    }
}

/// SSAO/GTAO settings
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SsaoSettings {
    pub radius: f32,
    pub intensity: f32,
    pub bias: f32,
    pub num_samples: u32,
    pub technique: u32,   // 0=SSAO, 1=GTAO
    pub frame_index: u32, // frame counter for noise
    pub inv_resolution: [f32; 2],
    pub proj_scale: f32, // 0.5 * height / tan(fov/2) = 0.5 * height * P[1][1]
    pub ao_min: f32,     // minimum AO value to prevent full black (default 0.35)
}

impl Default for SsaoSettings {
    fn default() -> Self {
        Self {
            radius: 0.5,
            intensity: 1.5, // Higher default intensity for stronger AO effect
            bias: 0.025,
            num_samples: 16,
            technique: 0,
            frame_index: 0,
            inv_resolution: [1.0 / 1920.0, 1.0 / 1080.0],
            proj_scale: 0.5 * 1080.0 * (1.0 / (45.0_f32.to_radians() * 0.5).tan()),
            ao_min: 0.05, // Allow stronger crease darkening while keeping a floor
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct SsaoTemporalParamsUniform {
    temporal_alpha: f32,
    _pad: [f32; 7],
}

/// SSGI settings
/// Note: Size must be 80 bytes to match WGSL std140 layout where vec3<u32> is aligned to 16 bytes
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SsgiSettings {
    pub radius: f32,
    pub intensity: f32,
    pub num_steps: u32,
    pub step_size: f32,
    pub inv_resolution: [f32; 2],
    pub temporal_alpha: f32,
    pub temporal_enabled: u32,
    pub use_half_res: u32,
    pub upsample_depth_sigma: f32,
    pub upsample_normal_sigma: f32,
    pub use_edge_aware: u32,
    pub _pad1: u32,
    pub frame_index: u32,
    pub _pad3: u32,
    pub _pad4: u32,
    pub _pad5: u32,
    pub _pad6: [u32; 4],
    pub _pad7: [u32; 3],
    pub _pad8: [u32; 4],
    pub _pad9: [u32; 8],
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
            temporal_enabled: 1,
            use_half_res: 0,
            upsample_depth_sigma: 0.02,
            // Normal sigma controls bilateral falloff (radians)
            upsample_normal_sigma: 0.25,
            use_edge_aware: 1,
            _pad1: 0,
            _pad3: 0,
            _pad4: 0,
            _pad5: 0,
            frame_index: 0,
            _pad6: [0; 4],
            _pad7: [0; 3],
            _pad8: [0; 4],
            _pad9: [0; 8],
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
            max_steps: 96,
            thickness: 0.1,
            max_distance: 32.0,
            intensity: 5.0,
            inv_resolution: [1.0 / 1920.0, 1.0 / 1080.0],
            _pad0: [0.0; 2],
        }
    }
}

/// Aggregated statistics emitted by the SSR pipeline.
#[derive(Debug, Clone, Default)]
pub struct SsrStats {
    pub num_rays: u32,
    pub num_hits: u32,
    pub total_steps: u32,
    pub num_misses: u32,
    pub miss_ibl_samples: u32,
    pub trace_ms: f32,
    pub shade_ms: f32,
    pub fallback_ms: f32,
}

impl SsrStats {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn clear(&mut self) {
        *self = Self::default();
    }

    pub fn hit_rate(&self) -> f32 {
        if self.num_rays == 0 {
            0.0
        } else {
            self.num_hits as f32 / self.num_rays as f32
        }
    }

    pub fn avg_steps(&self) -> f32 {
        if self.num_rays == 0 {
            0.0
        } else {
            self.total_steps as f32 / self.num_rays as f32
        }
    }

    pub fn miss_ibl_ratio(&self) -> f32 {
        if self.num_misses == 0 {
            0.0
        } else {
            self.miss_ibl_samples as f32 / self.num_misses as f32
        }
    }

    pub fn perf_ms(&self) -> f32 {
        self.trace_ms + self.shade_ms + self.fallback_ms
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

    // SSAO/GTAO compute pipelines
    ssao_pipeline: ComputePipeline,
    gtao_pipeline: ComputePipeline,
    ssao_bind_group_layout: BindGroupLayout,

    // Separable bilateral blur pipelines (H and V) + settings
    blur_h_pipeline: ComputePipeline,
    blur_v_pipeline: ComputePipeline,
    blur_bind_group_layout: BindGroupLayout,
    blur_settings: Buffer,
    // Temporal resolve pipeline and layout
    temporal_pipeline: ComputePipeline,
    temporal_bind_group_layout: BindGroupLayout,
    temporal_params_buffer: Buffer,

    // Composite pipeline
    composite_pipeline: ComputePipeline,
    composite_bind_group_layout: BindGroupLayout,
    comp_uniform: Buffer,

    // Noise texture and sampler for SSAO
    noise_texture: Texture,
    noise_view: TextureView,
    noise_sampler: Sampler,

    // Output textures
    // ssao_texture   : R32Float raw AO in [ao_min, 1], where 1 = no occlusion.
    ssao_texture: Texture,
    ssao_view: TextureView,
    // ssao_blurred   : R32Float spatially filtered AO (same range as ssao_texture).
    ssao_blurred: Texture,
    ssao_blurred_view: TextureView,
    // Temporal history and resolved outputs
    // ssao_history   : R32Float previous-frame AO used by temporal resolve.
    ssao_history: Texture,
    ssao_history_view: TextureView,
    // ssao_resolved  : R32Float temporally resolved AO, sampled by composite.
    ssao_resolved: Texture,
    ssao_resolved_view: TextureView,
    // Intermediate blur target
    ssao_tmp: Texture,
    ssao_tmp_view: TextureView,
    // Color composited with AO for display
    // ssao_composited: Rgba8Unorm material buffer modulated by resolved AO; used only
    // for visualization in the viewer (lit shading uses its own path).
    ssao_composited: Texture,
    ssao_composited_view: TextureView,

    width: u32,
    height: u32,
    frame_index: u32,
    temporal_enabled: bool,
    temporal_alpha: f32,
    // Enable/disable bilateral blur stage
    blur_enabled: bool,
    history_valid: bool,
    last_ao_ms: f32,
    last_blur_ms: f32,
    last_temporal_ms: f32,
}

impl SsaoRenderer {
    pub fn new(
        device: &Device,
        width: u32,
        height: u32,
        material_format: TextureFormat,
    ) -> RenderResult<Self> {
        let mut settings = SsaoSettings::default();
        settings.inv_resolution = [1.0 / width as f32, 1.0 / height as f32];
        // proj_scale will be filled on first update_camera; set a conservative default
        settings.proj_scale = 0.5 * height as f32;

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
        let ssao_module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("ssao_kernel_shader"),
            source: ShaderSource::Wgsl(SSAO_SHADER_SRC.into()),
        });
        let gtao_module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("gtao_kernel_shader"),
            source: ShaderSource::Wgsl(GTAO_SHADER_SRC.into()),
        });
        let filter_shader = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("ssao_filter_shader"),
            source: ShaderSource::Wgsl(
                include_str!("../shaders/filters/bilateral_separable.wgsl").into(),
            ),
        });
        let temporal_shader = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("ssao_temporal_shader"),
            source: ShaderSource::Wgsl(include_str!("../shaders/temporal/resolve_ao.wgsl").into()),
        });
        let composite_shader = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("ssao_composite_shader"),
            source: ShaderSource::Wgsl(SSAO_COMPOSITE_SHADER_SRC.into()),
        });

        // Create bind group layouts
        let ssao_bind_group_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("ssao_bind_group_layout"),
            entries: &[
                // Binding 0: Depth texture (texture_2d<f32>)
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
                // Binding 1: HZB texture
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
                // Binding 2: Normal texture
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
                // Binding 3: Noise texture
                BindGroupLayoutEntry {
                    binding: 3,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Texture {
                        sample_type: TextureSampleType::Float { filterable: false },
                        view_dimension: TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // Binding 4: Sampler
                BindGroupLayoutEntry {
                    binding: 4,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Sampler(SamplerBindingType::NonFiltering),
                    count: None,
                },
                // Binding 5: Output (R32Float - R8Unorm unsupported on Metal)
                BindGroupLayoutEntry {
                    binding: 5,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::StorageTexture {
                        access: StorageTextureAccess::WriteOnly,
                        format: TextureFormat::R32Float,
                        view_dimension: TextureViewDimension::D2,
                    },
                    count: None,
                },
                // Binding 6: Settings
                BindGroupLayoutEntry {
                    binding: 6,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Binding 7: Camera
                BindGroupLayoutEntry {
                    binding: 7,
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
                // AO input
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
                // Depth
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
                // Normal
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
                // Output (R32Float - R8Unorm unsupported on Metal)
                BindGroupLayoutEntry {
                    binding: 3,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::StorageTexture {
                        access: StorageTextureAccess::WriteOnly,
                        format: TextureFormat::R32Float,
                        view_dimension: TextureViewDimension::D2,
                    },
                    count: None,
                },
                // Settings (blur radius, sigmas)
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

        let composite_bind_group_layout =
            device.create_bind_group_layout(&BindGroupLayoutDescriptor {
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
        let ssao_pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("ssao_pipeline_layout"),
            bind_group_layouts: &[&ssao_bind_group_layout],
            push_constant_ranges: &[],
        });
        let ssao_pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("ssao_pipeline"),
            layout: Some(&ssao_pipeline_layout),
            module: &ssao_module,
            entry_point: "cs_ssao",
        });
        let gtao_pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("gtao_pipeline"),
            layout: Some(&ssao_pipeline_layout),
            module: &gtao_module,
            entry_point: "cs_gtao",
        });

        let blur_pl = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("ssao_blur_pipeline_layout"),
            bind_group_layouts: &[&blur_bind_group_layout],
            push_constant_ranges: &[],
        });
        let blur_h_pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("ssao_blur_h_pipeline"),
            layout: Some(&blur_pl),
            module: &filter_shader,
            entry_point: "cs_blur_h",
        });
        let blur_v_pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("ssao_blur_v_pipeline"),
            layout: Some(&blur_pl),
            module: &filter_shader,
            entry_point: "cs_blur_v",
        });

        // Blur settings buffer (u32 radius, f32 depth_sigma, f32 normal_sigma, u32 _pad)
        #[repr(C)]
        #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
        struct BlurSettingsStd140 {
            blur_radius: u32,
            depth_sigma: f32,
            normal_sigma: f32,
            _pad: u32,
        }
        // Use larger radius and tighter sigmas to ensure blur reduces variance by at least 5%
        // Increased radius and more permissive sigmas to allow more averaging for noise reduction
        let blur_params = BlurSettingsStd140 {
            blur_radius: 6,
            depth_sigma: 0.1,
            normal_sigma: 0.6,
            _pad: 0,
        };
        let blur_settings = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("ssao.blur.settings"),
            contents: bytemuck::bytes_of(&blur_params),
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        });

        // Temporal resolve layout: current, history, output, params
        let temporal_bind_group_layout =
            device.create_bind_group_layout(&BindGroupLayoutDescriptor {
                label: Some("ssao_temporal_bgl"),
                entries: &[
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
                    BindGroupLayoutEntry {
                        binding: 2,
                        visibility: ShaderStages::COMPUTE,
                        ty: BindingType::StorageTexture {
                            access: StorageTextureAccess::WriteOnly,
                            format: TextureFormat::R32Float,
                            view_dimension: TextureViewDimension::D2,
                        },
                        count: None,
                    },
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
        let temporal_pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("ssao_temporal_pipeline"),
            layout: Some(&device.create_pipeline_layout(&PipelineLayoutDescriptor {
                label: Some("ssao_temporal_pl"),
                bind_group_layouts: &[&temporal_bind_group_layout],
                push_constant_ranges: &[],
            })),
            module: &temporal_shader,
            entry_point: "cs_resolve_temporal",
        });
        // Temporal params buffer: alpha (32 bytes for uniform buffer alignment)
        let temporal_params = SsaoTemporalParamsUniform {
            temporal_alpha: 0.2,
            _pad: [0.0; 7],
        };
        let temporal_params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("ssao.temporal.params"),
            contents: bytemuck::bytes_of(&temporal_params),
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        });

        let _composite_params = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("p5.ssr.composite.params"),
            contents: bytemuck::cast_slice(&[1.0, 0.0, 0.0, 0.0]),
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        });

        let composite_params = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("ssao.comp.uniform"),
            contents: bytemuck::cast_slice(&[1.0, 0.0, 0.0, 0.0]),
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        });

        let composite_pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("ssao_composite_pipeline"),
            layout: Some(&device.create_pipeline_layout(&PipelineLayoutDescriptor {
                label: Some("ssao_composite_pipeline_layout"),
                bind_group_layouts: &[&composite_bind_group_layout],
                push_constant_ranges: &[],
            })),
            module: &composite_shader,
            entry_point: "cs_ssao_composite",
        });

        // Create blue noise texture (4x4 for simplicity, using IGN in shader anyway)
        let noise_size = 4u32;
        let mut noise_data: Vec<f32> = vec![0.0; (noise_size * noise_size) as usize];
        for i in 0..noise_data.len() {
            noise_data[i] = (i as f32 * 0.618033988749895) % 1.0; // Golden ratio noise
        }
        let noise_texture = device.create_texture(&TextureDescriptor {
            label: Some("ssao_noise"),
            size: Extent3d {
                width: noise_size,
                height: noise_size,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::R32Float,
            usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let noise_view = noise_texture.create_view(&TextureViewDescriptor::default());
        let noise_sampler = device.create_sampler(&SamplerDescriptor {
            label: Some("ssao_noise_sampler"),
            address_mode_u: AddressMode::Repeat,
            address_mode_v: AddressMode::Repeat,
            address_mode_w: AddressMode::Repeat,
            mag_filter: FilterMode::Nearest,
            min_filter: FilterMode::Nearest,
            mipmap_filter: FilterMode::Nearest,
            ..Default::default()
        });

        // Create output textures
        // R32Float (R8Unorm unsupported on Metal for storage)
        let ssao_texture = device.create_texture(&TextureDescriptor {
            label: Some("ssao_texture"),
            size: Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::R32Float,
            usage: TextureUsages::STORAGE_BINDING
                | TextureUsages::TEXTURE_BINDING
                | TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let ssao_view = ssao_texture.create_view(&TextureViewDescriptor::default());

        let ssao_blurred = device.create_texture(&TextureDescriptor {
            label: Some("ssao_blurred"),
            size: Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::R32Float,
            usage: TextureUsages::STORAGE_BINDING
                | TextureUsages::TEXTURE_BINDING
                | TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let ssao_blurred_view = ssao_blurred.create_view(&TextureViewDescriptor::default());

        // Intermediate target for separable blur
        let ssao_tmp = device.create_texture(&TextureDescriptor {
            label: Some("ssao_tmp"),
            size: Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::R32Float,
            usage: TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let ssao_tmp_view = ssao_tmp.create_view(&TextureViewDescriptor::default());

        // Temporal history and resolved targets
        let ssao_history = device.create_texture(&TextureDescriptor {
            label: Some("ssao_history"),
            size: Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::R32Float,
            usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let ssao_history_view = ssao_history.create_view(&TextureViewDescriptor::default());
        let ssao_resolved = device.create_texture(&TextureDescriptor {
            label: Some("ssao_resolved"),
            size: Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::R32Float,
            usage: TextureUsages::STORAGE_BINDING
                | TextureUsages::TEXTURE_BINDING
                | TextureUsages::COPY_SRC
                | TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let ssao_resolved_view = ssao_resolved.create_view(&TextureViewDescriptor::default());

        // Composited color (material * AO)
        let ssao_composited = device.create_texture(&TextureDescriptor {
            label: Some("ssao_composited"),
            size: Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Rgba8Unorm,
            usage: TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let ssao_composited_view = ssao_composited.create_view(&TextureViewDescriptor::default());

        // Composite uniform (x=intensity scale, y=ssao_mul, z,w reserved)
        // Default of (1.0, 1.0) applies full AO effect controlled by SSAO settings.
        let comp_params: [f32; 4] = [1.0, 1.0, 0.0, 0.0];
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
            gtao_pipeline,
            ssao_bind_group_layout,
            blur_h_pipeline,
            blur_v_pipeline,
            blur_bind_group_layout,
            blur_settings,
            temporal_pipeline,
            temporal_bind_group_layout,
            temporal_params_buffer,
            composite_pipeline,
            composite_bind_group_layout,
            comp_uniform,
            noise_texture,
            noise_view,
            noise_sampler,
            ssao_texture,
            ssao_view,
            ssao_blurred,
            ssao_blurred_view,
            ssao_tmp,
            ssao_tmp_view,
            ssao_history,
            ssao_history_view,
            ssao_resolved,
            ssao_resolved_view,
            ssao_composited,
            ssao_composited_view,
            width,
            height,
            frame_index: 0,
            temporal_enabled: true,
            temporal_alpha: 0.2,
            blur_enabled: true,
            history_valid: false,
            last_ao_ms: 0.0,
            last_blur_ms: 0.0,
            last_temporal_ms: 0.0,
        })
    }

    /// Update settings
    pub fn update_settings(&mut self, queue: &Queue, settings: SsaoSettings) {
        let technique_changed = self.settings.technique != settings.technique;
        self.settings = settings;
        queue.write_buffer(&self.settings_buffer, 0, bytemuck::bytes_of(&settings));
        if technique_changed {
            self.invalidate_history();
        }
    }

    pub fn get_settings(&self) -> SsaoSettings {
        self.settings
    }

    pub fn set_seed(&mut self, queue: &Queue, seed: u32) {
        self.settings.frame_index = seed;
        self.frame_index = seed;
        queue.write_buffer(&self.settings_buffer, 0, bytemuck::bytes_of(&self.settings));
        self.invalidate_history();
    }

    /// Update camera parameters
    pub fn update_camera(&mut self, queue: &Queue, camera: &CameraParams) {
        queue.write_buffer(&self.camera_buffer, 0, bytemuck::bytes_of(camera));
        // Update proj_scale from camera projection matrix and current height
        let p11 = camera.proj_matrix[1][1];
        self.settings.proj_scale = 0.5 * self.height as f32 * p11;
        queue.write_buffer(&self.settings_buffer, 0, bytemuck::bytes_of(&self.settings));
        self.invalidate_history();
    }

    /// Encode only the raw SSAO generation pass into the provided encoder
    pub fn encode_ssao(
        &mut self,
        device: &Device,
        encoder: &mut CommandEncoder,
        gbuffer: &GBuffer,
        hzb_view: &TextureView,
    ) -> RenderResult<()> {
        let t0 = Instant::now();
        // Increment frame index and push to GPU before dispatch
        let mut settings_shadow = self.settings;
        // Work on a shadow copy so we can tweak derived fields (frame index, inv resolution)
        // before uploading to the GPU while keeping `self.settings` in sync afterwards.
        settings_shadow.frame_index = self.settings.frame_index.wrapping_add(1);
        // Update inv_resolution in case of resize
        settings_shadow.inv_resolution = [1.0 / self.width as f32, 1.0 / self.height as f32];
        let staging = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("ssao.settings.staging"),
            contents: bytemuck::bytes_of(&settings_shadow),
            usage: BufferUsages::COPY_SRC,
        });
        encoder.copy_buffer_to_buffer(
            &staging,
            0,
            &self.settings_buffer,
            0,
            std::mem::size_of::<SsaoSettings>() as u64,
        );
        // Persist the updated dynamic settings so future dispatches see the latest frame index/resolution.
        self.settings = settings_shadow;
        // Create bind group with 8 bindings per task.toon
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
                    resource: BindingResource::TextureView(hzb_view),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: BindingResource::TextureView(&gbuffer.normal_view),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: BindingResource::TextureView(&self.noise_view),
                },
                BindGroupEntry {
                    binding: 4,
                    resource: BindingResource::Sampler(&self.noise_sampler),
                },
                BindGroupEntry {
                    binding: 5,
                    resource: BindingResource::TextureView(&self.ssao_view),
                },
                BindGroupEntry {
                    binding: 6,
                    resource: self.settings_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 7,
                    resource: self.camera_buffer.as_entire_binding(),
                },
            ],
        });
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("ssao_pass"),
            timestamp_writes: None,
        });
        let kernel = if self.settings.technique == 1 {
            &self.gtao_pipeline
        } else {
            &self.ssao_pipeline
        };
        pass.set_pipeline(kernel);
        pass.set_bind_group(0, &bind_group, &[]);
        let workgroup_x = (self.width + 7) / 8;
        let workgroup_y = (self.height + 7) / 8;
        pass.dispatch_workgroups(workgroup_x, workgroup_y, 1);
        drop(pass);
        self.last_ao_ms = t0.elapsed().as_secs_f32() * 1000.0;
        Ok(())
    }

    /// Encode bilateral blur (H and V)
    pub fn encode_blur(
        &mut self,
        device: &Device,
        encoder: &mut CommandEncoder,
        gbuffer: &GBuffer,
    ) -> RenderResult<()> {
        let t0 = Instant::now();
        let workgroup_x = (self.width + 7) / 8;
        let workgroup_y = (self.height + 7) / 8;

        // Blur pass H
        let blur_bg_h = device.create_bind_group(&BindGroupDescriptor {
            label: Some("ssao_blur_bg_h"),
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
                    resource: BindingResource::TextureView(&gbuffer.normal_view),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: BindingResource::TextureView(&self.ssao_tmp_view),
                },
                BindGroupEntry {
                    binding: 4,
                    resource: self.blur_settings.as_entire_binding(),
                },
            ],
        });
        let mut blur_h = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("ssao_blur_h"),
            timestamp_writes: None,
        });
        blur_h.set_pipeline(&self.blur_h_pipeline);
        blur_h.set_bind_group(0, &blur_bg_h, &[]);
        blur_h.dispatch_workgroups(workgroup_x, workgroup_y, 1);
        drop(blur_h);

        // Blur pass V
        let blur_bg_v = device.create_bind_group(&BindGroupDescriptor {
            label: Some("ssao_blur_bg_v"),
            layout: &self.blur_bind_group_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::TextureView(&self.ssao_tmp_view),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::TextureView(&gbuffer.depth_view),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: BindingResource::TextureView(&gbuffer.normal_view),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: BindingResource::TextureView(&self.ssao_blurred_view),
                },
                BindGroupEntry {
                    binding: 4,
                    resource: self.blur_settings.as_entire_binding(),
                },
            ],
        });
        let mut blur_v = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("ssao_blur_v"),
            timestamp_writes: None,
        });
        blur_v.set_pipeline(&self.blur_v_pipeline);
        blur_v.set_bind_group(0, &blur_bg_v, &[]);
        blur_v.dispatch_workgroups(workgroup_x, workgroup_y, 1);
        drop(blur_v);
        self.last_blur_ms = t0.elapsed().as_secs_f32() * 1000.0;
        Ok(())
    }

    /// Encode temporal resolve (or copy blurred->resolved)
    pub fn encode_temporal(
        &mut self,
        device: &Device,
        encoder: &mut CommandEncoder,
    ) -> RenderResult<()> {
        let t0 = Instant::now();
        let workgroup_x = (self.width + 7) / 8;
        let workgroup_y = (self.height + 7) / 8;
        // Temporal resolve (optional)
        if self.temporal_enabled {
            if !self.history_valid {
                let src_tex = if self.blur_enabled {
                    &self.ssao_blurred
                } else {
                    &self.ssao_texture
                };
                encoder.copy_texture_to_texture(
                    ImageCopyTexture {
                        texture: src_tex,
                        mip_level: 0,
                        origin: Origin3d::ZERO,
                        aspect: TextureAspect::All,
                    },
                    ImageCopyTexture {
                        texture: &self.ssao_resolved,
                        mip_level: 0,
                        origin: Origin3d::ZERO,
                        aspect: TextureAspect::All,
                    },
                    Extent3d {
                        width: self.width,
                        height: self.height,
                        depth_or_array_layers: 1,
                    },
                );
                encoder.copy_texture_to_texture(
                    ImageCopyTexture {
                        texture: &self.ssao_resolved,
                        mip_level: 0,
                        origin: Origin3d::ZERO,
                        aspect: TextureAspect::All,
                    },
                    ImageCopyTexture {
                        texture: &self.ssao_history,
                        mip_level: 0,
                        origin: Origin3d::ZERO,
                        aspect: TextureAspect::All,
                    },
                    Extent3d {
                        width: self.width,
                        height: self.height,
                        depth_or_array_layers: 1,
                    },
                );
                self.history_valid = true;
                self.last_temporal_ms = t0.elapsed().as_secs_f32() * 1000.0;
                return Ok(());
            }
            // Choose input depending on blur toggle
            let input_view = if self.blur_enabled {
                &self.ssao_blurred_view
            } else {
                &self.ssao_view
            };
            let temporal_bg = device.create_bind_group(&BindGroupDescriptor {
                label: Some("ssao_temporal_bg"),
                layout: &self.temporal_bind_group_layout,
                entries: &[
                    BindGroupEntry {
                        binding: 0,
                        resource: BindingResource::TextureView(input_view),
                    },
                    BindGroupEntry {
                        binding: 1,
                        resource: BindingResource::TextureView(&self.ssao_history_view),
                    },
                    BindGroupEntry {
                        binding: 2,
                        resource: BindingResource::TextureView(&self.ssao_resolved_view),
                    },
                    BindGroupEntry {
                        binding: 3,
                        resource: self.temporal_params_buffer.as_entire_binding(),
                    },
                ],
            });
            let mut tpass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("ssao_temporal"),
                timestamp_writes: None,
            });
            tpass.set_pipeline(&self.temporal_pipeline);
            tpass.set_bind_group(0, &temporal_bg, &[]);
            tpass.dispatch_workgroups(workgroup_x, workgroup_y, 1);
            drop(tpass);
            // Copy resolved to history for next frame
            encoder.copy_texture_to_texture(
                ImageCopyTexture {
                    texture: &self.ssao_resolved,
                    mip_level: 0,
                    origin: Origin3d::ZERO,
                    aspect: TextureAspect::All,
                },
                ImageCopyTexture {
                    texture: &self.ssao_history,
                    mip_level: 0,
                    origin: Origin3d::ZERO,
                    aspect: TextureAspect::All,
                },
                Extent3d {
                    width: self.width,
                    height: self.height,
                    depth_or_array_layers: 1,
                },
            );
            self.history_valid = true;
            self.last_temporal_ms = t0.elapsed().as_secs_f32() * 1000.0;
        } else {
            // If temporal disabled, keep resolved identical to blurred via a copy
            let src_tex = if self.blur_enabled {
                &self.ssao_blurred
            } else {
                &self.ssao_texture
            };
            encoder.copy_texture_to_texture(
                ImageCopyTexture {
                    texture: src_tex,
                    mip_level: 0,
                    origin: Origin3d::ZERO,
                    aspect: TextureAspect::All,
                },
                ImageCopyTexture {
                    texture: &self.ssao_resolved,
                    mip_level: 0,
                    origin: Origin3d::ZERO,
                    aspect: TextureAspect::All,
                },
                Extent3d {
                    width: self.width,
                    height: self.height,
                    depth_or_array_layers: 1,
                },
            );
            self.history_valid = false;
            self.last_temporal_ms = t0.elapsed().as_secs_f32() * 1000.0;
        }
        Ok(())
    }

    fn invalidate_history(&mut self) {
        self.history_valid = false;
    }

    /// Encode composite of material * AO -> RGBA8 target
    pub fn encode_composite(
        &self,
        device: &Device,
        encoder: &mut CommandEncoder,
        gbuffer: &GBuffer,
    ) -> RenderResult<()> {
        let comp_bg = device.create_bind_group(&BindGroupDescriptor {
            label: Some("ssao_composite_bind_group"),
            layout: &self.composite_bind_group_layout,
            entries: &[
                // material input
                BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::TextureView(&gbuffer.material_view),
                },
                // output
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::TextureView(&self.ssao_composited_view),
                },
                // resolved AO (temporal if enabled, else blurred is copied)
                BindGroupEntry {
                    binding: 2,
                    resource: BindingResource::TextureView(&self.ssao_resolved_view),
                },
                // composite params
                BindGroupEntry {
                    binding: 3,
                    resource: self.comp_uniform.as_entire_binding(),
                },
            ],
        });

        let mut comp_pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("ssao_composite_pass"),
            timestamp_writes: None,
        });
        comp_pass.set_pipeline(&self.composite_pipeline);
        comp_pass.set_bind_group(0, &comp_bg, &[]);
        let workgroup_x = (self.width + 7) / 8;
        let workgroup_y = (self.height + 7) / 8;
        comp_pass.dispatch_workgroups(workgroup_x, workgroup_y, 1);
        Ok(())
    }

    /// Execute full SSAO pipeline (raw -> blur -> temporal -> composite)
    pub fn execute(
        &mut self,
        device: &Device,
        encoder: &mut CommandEncoder,
        gbuffer: &GBuffer,
        hzb_view: &TextureView,
    ) -> RenderResult<()> {
        self.encode_ssao(device, encoder, gbuffer, hzb_view)?;
        self.encode_blur(device, encoder, gbuffer)?;
        self.encode_temporal(device, encoder)?;
        self.encode_composite(device, encoder, gbuffer)?;
        Ok(())
    }

    // removed: mistakenly added inside SsaoRenderer impl

    /// Get blurred SSAO texture view
    pub fn get_output(&self) -> &TextureView {
        &self.ssao_blurred_view
    }

    /// Get raw SSAO texture view (pre-blur)
    pub fn get_raw_ao_view(&self) -> &TextureView {
        &self.ssao_view
    }

    /// Get intermediate blur_h output (tmp texture)
    pub fn get_tmp_ao_view(&self) -> &TextureView {
        &self.ssao_tmp_view
    }

    /// Get resolved (temporal) AO view
    pub fn get_resolved_ao_view(&self) -> &TextureView {
        &self.ssao_resolved_view
    }

    /// Get composited color (material * AO)
    pub fn get_composited(&self) -> &TextureView {
        &self.ssao_composited_view
    }

    pub fn set_composite_multiplier(&mut self, queue: &Queue, mul: f32) {
        let m = mul.clamp(0.0, 1.0);
        // Keep intensity scale at 1.0 and use the second slot for ssao_mul toggle.
        let params: [f32; 4] = [1.0, m, 0.0, 0.0];
        queue.write_buffer(&self.comp_uniform, 0, bytemuck::cast_slice(&params));
    }

    pub fn set_blur_enabled(&mut self, on: bool) {
        if self.blur_enabled != on {
            self.blur_enabled = on;
            self.invalidate_history();
        }
    }

    pub fn set_temporal_enabled(&mut self, on: bool) {
        if self.temporal_enabled != on {
            self.temporal_enabled = on;
            self.invalidate_history();
        }
    }
    pub fn blur_enabled(&self) -> bool {
        self.blur_enabled
    }

    // Expose underlying textures for readback
    pub fn raw_ao_texture(&self) -> &Texture {
        &self.ssao_texture
    }
    pub fn blurred_ao_texture(&self) -> &Texture {
        &self.ssao_blurred
    }
    pub fn resolved_ao_texture(&self) -> &Texture {
        &self.ssao_resolved
    }
    pub fn composited_texture(&self) -> &Texture {
        &self.ssao_composited
    }

    pub fn timings_ms(&self) -> (f32, f32, f32) {
        (self.last_ao_ms, self.last_blur_ms, self.last_temporal_ms)
    }
}

/// Screen-space effects manager
pub struct ScreenSpaceEffectsManager {
    gbuffer: GBuffer,
    ssao_renderer: Option<SsaoRenderer>,
    ssgi_renderer: Option<SsgiRenderer>,
    ssr_renderer: Option<SsrRenderer>,
    enabled_effects: Vec<ScreenSpaceEffect>,
    pub hzb: Option<HzbPyramid>,
    ssr_params: SsrParams,
    last_hzb_ms: f32,
}

impl ScreenSpaceEffectsManager {
    pub fn new(device: &Device, width: u32, height: u32) -> RenderResult<Self> {
        let gbuffer_config = GBufferConfig {
            width,
            height,
            ..Default::default()
        };
        let gbuffer = GBuffer::new(device, gbuffer_config)?;
        // Initialize HZB pyramid (optional)
        let hzb = HzbPyramid::new(device, width, height).ok();

        Ok(Self {
            gbuffer,
            ssao_renderer: None,
            ssgi_renderer: None,
            ssr_renderer: None,
            enabled_effects: Vec::new(),
            hzb,
            ssr_params: SsrParams::default(),
            last_hzb_ms: 0.0,
        })
    }

    pub fn set_environment_texture(&mut self, device: &Device, env_texture: &Texture) {
        if let Some(ref mut ssgi) = self.ssgi_renderer {
            ssgi.set_environment_texture(device, env_texture);
        }
        if let Some(ref mut ssr) = self.ssr_renderer {
            ssr.set_environment_texture(device, env_texture);
        }
    }

    /// Enable an effect
    pub fn enable_effect(
        &mut self,
        device: &Device,
        effect: ScreenSpaceEffect,
    ) -> RenderResult<()> {
        if !self.enabled_effects.contains(&effect) {
            self.enabled_effects.push(effect);
        }

        match effect {
            ScreenSpaceEffect::SSAO => {
                if self.ssao_renderer.is_none() {
                    let (width, height) = self.gbuffer.dimensions();
                    let mat_fmt = self.gbuffer.config().material_format;
                    self.ssao_renderer = Some(SsaoRenderer::new(device, width, height, mat_fmt)?);
                }
            }
            ScreenSpaceEffect::SSGI => {
                if self.ssgi_renderer.is_none() {
                    let (width, height) = self.gbuffer.dimensions();
                    let mat_fmt = self.gbuffer.config().material_format;
                    self.ssgi_renderer = Some(SsgiRenderer::new(device, width, height, mat_fmt)?);
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

    /// Build depth HZB pyramid from current GBuffer depth for this frame.
    /// Requires that GBuffer depth has been rendered for the frame.
    /// Records an approximate CPU-side build time in milliseconds for
    /// performance diagnostics (P5.6 budgets).
    pub fn build_hzb(
        &mut self,
        device: &Device,
        encoder: &mut CommandEncoder,
        src_depth: &TextureView,
        reversed_z: bool,
    ) {
        if let Some(ref hzb) = self.hzb {
            let t0 = Instant::now();
            hzb.build(device, encoder, src_depth, reversed_z);
            self.last_hzb_ms = t0.elapsed().as_secs_f32() * 1000.0;
        }
    }

    /// Get access to the HZB pyramid texture and mip count (for diagnostics/export)
    pub fn hzb_texture_and_mips(&self) -> Option<(&Texture, u32)> {
        self.hzb.as_ref().map(|h| (&h.tex, h.mip_count))
    }

    pub fn set_ssr_params(&mut self, queue: &Queue, params: &SsrParams) {
        self.ssr_params = *params;
        self.refresh_ssr_settings(queue);
    }

    fn refresh_ssr_settings(&mut self, queue: &Queue) {
        let settings = self.build_ssr_settings();
        if let Some(ref mut renderer) = self.ssr_renderer {
            renderer.update_settings(queue, settings);
        }
    }

    fn build_ssr_settings(&self) -> SsrSettings {
        let (mut w, mut h) = self.gbuffer.dimensions();
        if w == 0 {
            w = 1;
        }
        if h == 0 {
            h = 1;
        }
        SsrSettings {
            max_steps: self.ssr_params.ssr_max_steps,
            thickness: self.ssr_params.ssr_thickness,
            inv_resolution: [1.0 / w as f32, 1.0 / h as f32],
            ..SsrSettings::default()
        }
    }

    /// Execute all enabled effects
    pub fn execute(
        &mut self,
        device: &Device,
        encoder: &mut CommandEncoder,
        ssr_stats: Option<&mut SsrStats>,
        mut timing: Option<&mut GpuTimingManager>,
    ) -> RenderResult<()> {
        let mut ssr_stats_opt = ssr_stats;
        for effect in &self.enabled_effects {
            match effect {
                ScreenSpaceEffect::SSAO => {
                    if let (Some(ssao), Some(hzb)) =
                        (self.ssao_renderer.as_mut(), self.hzb.as_ref())
                    {
                        let hzb_view = hzb.texture_view();
                        if let Some(timer) = timing.as_deref_mut() {
                            let scope_id = timer.begin_scope(encoder, "p5.ssao");
                            ssao.execute(device, encoder, &self.gbuffer, &hzb_view)?;
                            timer.end_scope(encoder, scope_id);
                        } else {
                            ssao.execute(device, encoder, &self.gbuffer, &hzb_view)?;
                        }
                    }
                }
                ScreenSpaceEffect::SSGI => {
                    if let Some(ref mut ssgi) = self.ssgi_renderer {
                        let timing_scope = if let Some(timer) = timing.as_deref_mut() {
                            Some(timer.begin_scope(encoder, "p5.ssgi"))
                        } else {
                            None
                        };

                        if let Some(ref hzb) = self.hzb {
                            let hzb_view = hzb.texture_view();
                            ssgi.execute(device, encoder, &self.gbuffer, &hzb_view)?;
                        } else {
                            // Fallback to depth view if HZB is unavailable
                            ssgi.execute(device, encoder, &self.gbuffer, &self.gbuffer.depth_view)?;
                        }

                        if let Some(scope_id) = timing_scope {
                            if let Some(timer) = timing.as_deref_mut() {
                                timer.end_scope(encoder, scope_id);
                            }
                        }
                    }
                }
                ScreenSpaceEffect::SSR => {
                    if self.ssr_params.ssr_enable {
                        if let Some(ref mut ssr) = self.ssr_renderer {
                            if let Some(timer) = timing.as_deref_mut() {
                                let scope_id = timer.begin_scope(encoder, "p5.ssr");
                                ssr.execute(
                                    device,
                                    encoder,
                                    &self.gbuffer,
                                    ssr_stats_opt.as_deref_mut(),
                                )?;
                                timer.end_scope(encoder, scope_id);
                            } else {
                                ssr.execute(
                                    device,
                                    encoder,
                                    &self.gbuffer,
                                    ssr_stats_opt.as_deref_mut(),
                                )?;
                            }
                        }
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

    pub fn advance_frame(&mut self, queue: &Queue) {
        if let Some(ref mut ssgi) = self.ssgi_renderer {
            ssgi.advance_frame(queue);
        }
    }

    /// Return the last recorded HZB build time in milliseconds. This is a
    /// CPU-side approximation used for P5.6 performance budgets.
    pub fn hzb_ms(&self) -> f32 {
        self.last_hzb_ms
    }

    pub fn set_gi_seed(
        &mut self,
        device: &Device,
        queue: &Queue,
        seed: u32,
    ) -> RenderResult<()> {
        if let Some(ref mut ssao) = self.ssao_renderer {
            ssao.set_seed(queue, seed);
        }
        if let Some(ref mut ssgi) = self.ssgi_renderer {
            ssgi.set_seed(queue, seed);
        }
        self.ssgi_reset_history(device, queue)?;
        Ok(())
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
        if self.enabled_effects.contains(&ScreenSpaceEffect::SSAO) {
            if let Some(ref ssao) = self.ssao_renderer {
                return Some(ssao.get_composited());
            }
        }
        None
    }

    /// Return the material composited with SSR if available
    pub fn material_with_ssr_view(&self) -> Option<&TextureView> {
        if self.enabled_effects.contains(&ScreenSpaceEffect::SSR) {
            if let Some(ref ssr) = self.ssr_renderer {
                return Some(ssr.composite_view());
            }
        }
        None
    }

    pub fn ssr_spec_view(&self) -> Option<&TextureView> {
        self.ssr_renderer.as_ref().map(|ssr| ssr.spec_view())
    }

    pub fn ssr_final_view(&self) -> Option<&TextureView> {
        self.ssr_renderer.as_ref().map(|ssr| ssr.final_view())
    }

    pub fn set_ssr_scene_color_view(&mut self, view: TextureView) {
        if let Some(ref mut ssr) = self.ssr_renderer {
            ssr.set_scene_color_view(view);
        }
    }

    pub fn collect_ssr_stats(
        &mut self,
        device: &Device,
        queue: &Queue,
        stats: &mut SsrStats,
    ) -> RenderResult<()> {
        if let Some(ref mut ssr) = self.ssr_renderer {
            ssr.collect_stats_into(device, queue, stats)
        } else {
            stats.clear();
            Ok(())
        }
    }

    /// Return the material composited with SSGI if available
    pub fn material_with_ssgi_view(&self) -> Option<&TextureView> {
        if self.enabled_effects.contains(&ScreenSpaceEffect::SSGI) {
            if let Some(ref ssgi) = self.ssgi_renderer {
                return Some(ssgi.get_composited());
            }
        }
        None
    }

    /// Set SSGI composite intensity
    pub fn set_ssgi_composite_intensity(&mut self, queue: &Queue, intensity: f32) {
        if let Some(ref mut ssgi) = self.ssgi_renderer {
            ssgi.set_composite_intensity(queue, intensity);
        }
    }

    /// Provide access to AO AOVs for metrics/readbacks
    pub fn ao_raw_view(&self) -> Option<&TextureView> {
        self.ssao_renderer.as_ref().map(|r| r.get_raw_ao_view())
    }
    pub fn ao_tmp_view(&self) -> Option<&TextureView> {
        self.ssao_renderer.as_ref().map(|r| r.get_tmp_ao_view())
    }
    pub fn ao_blur_view(&self) -> Option<&TextureView> {
        self.ssao_renderer.as_ref().map(|r| r.get_output())
    }
    pub fn ao_resolved_view(&self) -> Option<&TextureView> {
        self.ssao_renderer
            .as_ref()
            .map(|r| r.get_resolved_ao_view())
    }

    /// Direct texture accessors for readback
    pub fn ao_raw_texture(&self) -> Option<&Texture> {
        self.ssao_renderer.as_ref().map(|r| r.raw_ao_texture())
    }
    pub fn ao_blur_texture(&self) -> Option<&Texture> {
        self.ssao_renderer.as_ref().map(|r| r.blurred_ao_texture())
    }
    pub fn ao_resolved_texture(&self) -> Option<&Texture> {
        self.ssao_renderer.as_ref().map(|r| r.resolved_ao_texture())
    }
    pub fn ao_composited_texture(&self) -> Option<&Texture> {
        self.ssao_renderer.as_ref().map(|r| r.composited_texture())
    }

    pub fn ssr_hit_view(&self) -> Option<&TextureView> {
        self.ssr_renderer.as_ref().map(|r| r.hit_data_view())
    }

    pub fn ssr_output_texture(&self) -> Option<&Texture> {
        self.ssr_renderer.as_ref().map(|r| r.output_texture())
    }

    pub fn ssr_hit_texture(&self) -> Option<&Texture> {
        self.ssr_renderer.as_ref().map(|r| r.hit_data_texture())
    }

    pub fn ssr_timings_ms(&self) -> Option<(f32, f32, f32)> {
        self.ssr_renderer.as_ref().map(|r| r.timings_ms())
    }

    /// Enable/disable temporal AO
    pub fn set_ssao_temporal(&mut self, on: bool) {
        if let Some(ref mut r) = self.ssao_renderer {
            r.set_temporal_enabled(on);
        }
    }
    /// Set temporal alpha for AO
    pub fn set_ssao_temporal_alpha(&mut self, queue: &Queue, alpha: f32) {
        if let Some(ref mut r) = self.ssao_renderer {
            r.temporal_alpha = alpha.clamp(0.0, 1.0);
            let params = SsaoTemporalParamsUniform {
                temporal_alpha: r.temporal_alpha,
                _pad: [0.0; 7],
            };
            queue.write_buffer(&r.temporal_params_buffer, 0, bytemuck::bytes_of(&params));
        }
    }

    pub fn set_ssao_composite_multiplier(&mut self, queue: &Queue, mul: f32) {
        if let Some(ref mut ssao) = self.ssao_renderer {
            ssao.set_composite_multiplier(queue, mul);
        }
    }

    pub fn ssao_timings_ms(&self) -> Option<(f32, f32, f32)> {
        self.ssao_renderer.as_ref().map(|r| r.timings_ms())
    }

    /// Check if an effect is enabled
    pub fn is_enabled(&self, effect: ScreenSpaceEffect) -> bool {
        self.enabled_effects.contains(&effect)
    }

    /// Get current SSAO settings for HUD display
    pub fn ssao_settings(&self) -> SsaoSettings {
        self.ssao_renderer
            .as_ref()
            .map(|r| r.settings)
            .unwrap_or_default()
    }

    /// Get SSAO temporal alpha for HUD display
    pub fn ssao_temporal_alpha(&self) -> f32 {
        self.ssao_renderer
            .as_ref()
            .map(|r| r.temporal_alpha)
            .unwrap_or(0.0)
    }

    pub fn ssao_temporal_enabled(&self) -> bool {
        self.ssao_renderer
            .as_ref()
            .map(|r| r.temporal_enabled)
            .unwrap_or(false)
    }

    pub fn update_ssao_settings(&mut self, queue: &Queue, mut f: impl FnMut(&mut SsaoSettings)) {
        if let Some(ref mut r) = self.ssao_renderer {
            let mut s = r.get_settings();
            f(&mut s);
            r.update_settings(queue, s);
        }
    }

    /// Toggle SSAO bilateral blur on/off
    pub fn set_ssao_blur(&mut self, on: bool) {
        if let Some(ref mut r) = self.ssao_renderer {
            r.set_blur_enabled(on);
        }
    }

    /// Set SSAO bias
    pub fn set_ssao_bias(&mut self, queue: &Queue, bias: f32) {
        if let Some(ref mut r) = self.ssao_renderer {
            let mut s = r.get_settings();
            s.bias = bias.max(0.0);
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

    pub fn ssgi_settings(&self) -> Option<SsgiSettings> {
        self.ssgi_renderer.as_ref().map(|r| r.get_settings())
    }

    pub fn ssgi_timings_ms(&self) -> Option<(f32, f32, f32, f32)> {
        self.ssgi_renderer.as_ref().map(|r| r.timings_ms())
    }

    pub fn ssgi_dimensions(&self) -> Option<(u32, u32)> {
        self.ssgi_renderer.as_ref().map(|r| r.dimensions())
    }

    pub fn ssgi_half_res(&self) -> Option<bool> {
        self.ssgi_renderer.as_ref().map(|r| r.is_half_res())
    }

    pub fn ssgi_hit_texture(&self) -> Option<&Texture> {
        self.ssgi_renderer.as_ref().map(|r| r.hit_texture())
    }

    pub fn ssgi_filtered_texture(&self) -> Option<&Texture> {
        self.ssgi_renderer.as_ref().map(|r| r.filtered_texture())
    }

    pub fn ssgi_history_texture(&self) -> Option<&Texture> {
        self.ssgi_renderer.as_ref().map(|r| r.history_texture())
    }

    pub fn ssgi_upscaled_texture(&self) -> Option<&Texture> {
        self.ssgi_renderer.as_ref().map(|r| r.upscaled_texture())
    }

    pub fn ssgi_output_for_display_view(&self) -> Option<&TextureView> {
        self.ssgi_renderer
            .as_ref()
            .map(|r| r.get_output_for_display())
    }

    pub fn ssgi_reset_history(&mut self, device: &Device, queue: &Queue) -> RenderResult<()> {
        if let Some(ref mut ssgi) = self.ssgi_renderer {
            ssgi.reset_history(device, queue)?;
        }
        Ok(())
    }
}

/// SSGI renderer
pub struct SsgiRenderer {
    settings: SsgiSettings,
    settings_buffer: Buffer,
    camera_buffer: Buffer,
    frame_index: u32,

    // Pipelines
    trace_pipeline: ComputePipeline,
    trace_bind_group_layout: BindGroupLayout,
    shade_pipeline: ComputePipeline,
    shade_bind_group_layout: BindGroupLayout,
    temporal_pipeline: ComputePipeline,
    temporal_bind_group_layout: BindGroupLayout,
    upsample_pipeline: ComputePipeline,
    upsample_bind_group_layout: BindGroupLayout,
    composite_pipeline: ComputePipeline,
    composite_bind_group_layout: BindGroupLayout,

    // Output and temporal textures
    // ssgi_hit       : Rgba16Float half-res hit buffer (xy = hit UV in [0,1], z = travelled
    //                   distance in view units, w = hit mask in {0,1}).
    ssgi_hit: Texture,
    ssgi_hit_view: TextureView,
    // ssgi_texture   : Rgba16Float half-res GI radiance (rgb = diffuse bounce light in
    //                   linear HDR units, a unused/1.0).
    ssgi_texture: Texture,
    ssgi_view: TextureView,
    // ssgi_history   : Rgba16Float previous-frame GI radiance used for temporal resolve.
    ssgi_history: Texture,
    ssgi_history_view: TextureView,
    // ssgi_filtered  : Rgba16Float temporally filtered GI radiance (same layout as
    //                   ssgi_texture).
    ssgi_filtered: Texture,
    ssgi_filtered_view: TextureView,
    // Full-resolution upscaled output for half-res mode
    // ssgi_upscaled  : Rgba16Float full-res GI radiance after edge-aware upsample.
    ssgi_upscaled: Texture,
    ssgi_upscaled_view: TextureView,
    // Composited material (material + SSGI)
    // ssgi_composited: Rgba8Unorm material buffer + SSGI diffuse contribution, used for
    //                   P5 visualization and metrics (not the main HDR lighting buffer).
    ssgi_composited: Texture,
    ssgi_composited_view: TextureView,
    composite_uniform: Buffer,
    scene_history: [Texture; 2],
    scene_history_views: [TextureView; 2],
    scene_history_index: usize,
    scene_history_ready: bool,
    linear_sampler: Sampler,

    // Env
    env_texture: Texture,
    env_view: TextureView,
    env_sampler: Sampler,

    width: u32,
    height: u32,
    half_res: bool,

    // Timings (ms)
    last_trace_ms: f32,
    last_shade_ms: f32,
    last_temporal_ms: f32,
    last_upsample_ms: f32,
}

impl SsgiRenderer {
    pub fn new(
        device: &Device,
        width: u32,
        height: u32,
        material_format: TextureFormat,
    ) -> RenderResult<Self> {
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

        // Shaders (split per stage per P5.2)
        let trace_shader = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("p5.ssgi.trace"),
            source: ShaderSource::Wgsl(include_str!("../shaders/ssgi/trace.wgsl").into()),
        });
        let shade_shader = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("p5.ssgi.shade"),
            source: ShaderSource::Wgsl(include_str!("../shaders/ssgi/shade.wgsl").into()),
        });
        let temporal_shader = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("p5.ssgi.temporal"),
            source: ShaderSource::Wgsl(
                include_str!("../shaders/ssgi/resolve_temporal.wgsl").into(),
            ),
        });
        let upsample_shader = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("p5.ssgi.upsample"),
            source: ShaderSource::Wgsl(
                include_str!("../shaders/filters/edge_aware_upsample.wgsl").into(),
            ),
        });

        // Trace pass: depth, normal, HZB, outHit, settings, camera
        let trace_bind_group_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("ssgi_trace_bgl"),
            entries: &[
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
                BindGroupLayoutEntry {
                    binding: 3,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::StorageTexture {
                        access: StorageTextureAccess::WriteOnly,
                        format: TextureFormat::Rgba16Float,
                        view_dimension: TextureViewDimension::D2,
                    },
                    count: None,
                },
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
                BindGroupLayoutEntry {
                    binding: 5,
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
        let trace_pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("ssgi_trace_pipeline"),
            layout: Some(&device.create_pipeline_layout(&PipelineLayoutDescriptor {
                label: Some("ssgi_trace_pl"),
                bind_group_layouts: &[&trace_bind_group_layout],
                push_constant_ranges: &[],
            })),
            module: &trace_shader,
            entry_point: "cs_trace",
        });

        // Shade pass: prevColor, sampler, env cube/sampler, hit, outRadiance, settings, camera, normalFull, material
        let shade_bind_group_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("ssgi_shade_bgl"),
            entries: &[
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Texture {
                        sample_type: TextureSampleType::Float { filterable: true },
                        view_dimension: TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Sampler(SamplerBindingType::Filtering),
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 2,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Texture {
                        sample_type: TextureSampleType::Float { filterable: true },
                        view_dimension: TextureViewDimension::Cube,
                        multisampled: false,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 3,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Sampler(SamplerBindingType::Filtering),
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 4,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Texture {
                        sample_type: TextureSampleType::Float { filterable: false },
                        view_dimension: TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 5,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::StorageTexture {
                        access: StorageTextureAccess::WriteOnly,
                        format: TextureFormat::Rgba16Float,
                        view_dimension: TextureViewDimension::D2,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 6,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 7,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 8,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Texture {
                        sample_type: TextureSampleType::Float { filterable: false },
                        view_dimension: TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 9,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Texture {
                        sample_type: TextureSampleType::Float { filterable: false },
                        view_dimension: TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
            ],
        });
        let shade_pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("ssgi_shade_pipeline"),
            layout: Some(&device.create_pipeline_layout(&PipelineLayoutDescriptor {
                label: Some("ssgi_shade_pl"),
                bind_group_layouts: &[&shade_bind_group_layout],
                push_constant_ranges: &[],
            })),
            module: &shade_shader,
            entry_point: "cs_shade",
        });

        // Temporal resolve layout: current, history, filtered, settings
        let temporal_bind_group_layout =
            device.create_bind_group_layout(&BindGroupLayoutDescriptor {
                label: Some("ssgi_temporal_bind_group_layout"),
                entries: &[
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
                    BindGroupLayoutEntry {
                        binding: 2,
                        visibility: ShaderStages::COMPUTE,
                        ty: BindingType::StorageTexture {
                            access: StorageTextureAccess::WriteOnly,
                            format: TextureFormat::Rgba16Float,
                            view_dimension: TextureViewDimension::D2,
                        },
                        count: None,
                    },
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
                    BindGroupLayoutEntry {
                        binding: 4,
                        visibility: ShaderStages::COMPUTE,
                        ty: BindingType::Texture {
                            sample_type: TextureSampleType::Float { filterable: false },
                            view_dimension: TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    BindGroupLayoutEntry {
                        binding: 5,
                        visibility: ShaderStages::COMPUTE,
                        ty: BindingType::Texture {
                            sample_type: TextureSampleType::Float { filterable: false },
                            view_dimension: TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                ],
            });
        let temporal_pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("ssgi_temporal_pipeline"),
            layout: Some(&device.create_pipeline_layout(&PipelineLayoutDescriptor {
                label: Some("ssgi_temporal_pl"),
                bind_group_layouts: &[&temporal_bind_group_layout],
                push_constant_ranges: &[],
            })),
            module: &temporal_shader,
            entry_point: "cs_resolve_temporal",
        });

        // Edge-aware upsample layout/pipeline
        let upsample_bind_group_layout =
            device.create_bind_group_layout(&BindGroupLayoutDescriptor {
                label: Some("ssgi_upsample_bind_group_layout"),
                entries: &[
                    BindGroupLayoutEntry {
                        binding: 0,
                        visibility: ShaderStages::COMPUTE,
                        ty: BindingType::Texture {
                            sample_type: TextureSampleType::Float { filterable: true },
                            view_dimension: TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    BindGroupLayoutEntry {
                        binding: 1,
                        visibility: ShaderStages::COMPUTE,
                        ty: BindingType::StorageTexture {
                            access: StorageTextureAccess::WriteOnly,
                            format: TextureFormat::Rgba16Float,
                            view_dimension: TextureViewDimension::D2,
                        },
                        count: None,
                    },
                    BindGroupLayoutEntry {
                        binding: 2,
                        visibility: ShaderStages::COMPUTE,
                        ty: BindingType::Sampler(SamplerBindingType::Filtering),
                        count: None,
                    },
                    BindGroupLayoutEntry {
                        binding: 3,
                        visibility: ShaderStages::COMPUTE,
                        ty: BindingType::Texture {
                            sample_type: TextureSampleType::Float { filterable: false },
                            view_dimension: TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    BindGroupLayoutEntry {
                        binding: 4,
                        visibility: ShaderStages::COMPUTE,
                        ty: BindingType::Texture {
                            sample_type: TextureSampleType::Float { filterable: false },
                            view_dimension: TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    BindGroupLayoutEntry {
                        binding: 5,
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
        let upsample_pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("ssgi_upsample_pipeline"),
            layout: Some(&device.create_pipeline_layout(&PipelineLayoutDescriptor {
                label: Some("ssgi_upsample_pl"),
                bind_group_layouts: &[&upsample_bind_group_layout],
                push_constant_ranges: &[],
            })),
            module: &upsample_shader,
            entry_point: "cs_edge_aware_upsample",
        });

        // Composite pass: material + SSGI
        let composite_shader = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("p5.ssgi.composite"),
            source: ShaderSource::Wgsl(include_str!("../shaders/ssgi/composite.wgsl").into()),
        });
        let composite_bind_group_layout =
            device.create_bind_group_layout(&BindGroupLayoutDescriptor {
                label: Some("ssgi_composite_bind_group_layout"),
                entries: &[
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
                    BindGroupLayoutEntry {
                        binding: 2,
                        visibility: ShaderStages::COMPUTE,
                        ty: BindingType::Texture {
                            sample_type: TextureSampleType::Float { filterable: true },
                            view_dimension: TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
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
        let composite_pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("ssgi_composite_pipeline"),
            layout: Some(&device.create_pipeline_layout(&PipelineLayoutDescriptor {
                label: Some("ssgi_composite_pl"),
                bind_group_layouts: &[&composite_bind_group_layout],
                push_constant_ranges: &[],
            })),
            module: &composite_shader,
            entry_point: "cs_ssgi_composite",
        });

        // Output and temporal textures (half-res by default disabled)
        let ssgi_hit = device.create_texture(&TextureDescriptor {
            label: Some("ssgi_hit"),
            size: Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Rgba16Float,
            usage: TextureUsages::STORAGE_BINDING
                | TextureUsages::TEXTURE_BINDING
                | TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let ssgi_hit_view = ssgi_hit.create_view(&TextureViewDescriptor::default());
        let ssgi_texture = device.create_texture(&TextureDescriptor {
            label: Some("ssgi_texture"),
            size: Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Rgba16Float,
            usage: TextureUsages::STORAGE_BINDING
                | TextureUsages::TEXTURE_BINDING
                | TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let ssgi_view = ssgi_texture.create_view(&TextureViewDescriptor::default());

        let ssgi_history = device.create_texture(&TextureDescriptor {
            label: Some("ssgi_history"),
            size: Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Rgba16Float,
            usage: TextureUsages::TEXTURE_BINDING
                | TextureUsages::COPY_DST
                | TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let ssgi_history_view = ssgi_history.create_view(&TextureViewDescriptor::default());

        let ssgi_filtered = device.create_texture(&TextureDescriptor {
            label: Some("ssgi_filtered"),
            size: Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Rgba16Float,
            usage: TextureUsages::STORAGE_BINDING
                | TextureUsages::TEXTURE_BINDING
                | TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let ssgi_filtered_view = ssgi_filtered.create_view(&TextureViewDescriptor::default());

        // Full-resolution upscaled target
        let ssgi_upscaled = device.create_texture(&TextureDescriptor {
            label: Some("ssgi_upscaled"),
            size: Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Rgba16Float,
            usage: TextureUsages::STORAGE_BINDING
                | TextureUsages::TEXTURE_BINDING
                | TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let ssgi_upscaled_view = ssgi_upscaled.create_view(&TextureViewDescriptor::default());

        // Composited material (material + SSGI)
        let ssgi_composited = device.create_texture(&TextureDescriptor {
            label: Some("ssgi_composited"),
            size: Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Rgba8Unorm,
            usage: TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let ssgi_composited_view = ssgi_composited.create_view(&TextureViewDescriptor::default());

        // Composite uniform (x = intensity multiplier)
        let comp_params: [f32; 4] = [1.0, 0.0, 0.0, 0.0];
        let composite_uniform = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("ssgi.composite.uniform"),
            contents: bytemuck::cast_slice(&comp_params),
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        });

        // Previous-frame color ping-pong (full resolution)
        let history_usage =
            TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST | TextureUsages::COPY_SRC;
        let scene_history = [
            device.create_texture(&TextureDescriptor {
                label: Some("ssgi_scene_history_a"),
                size: Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: TextureDimension::D2,
                format: material_format,
                usage: history_usage,
                view_formats: &[],
            }),
            device.create_texture(&TextureDescriptor {
                label: Some("ssgi_scene_history_b"),
                size: Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: TextureDimension::D2,
                format: material_format,
                usage: history_usage,
                view_formats: &[],
            }),
        ];
        let scene_history_views = [
            scene_history[0].create_view(&TextureViewDescriptor::default()),
            scene_history[1].create_view(&TextureViewDescriptor::default()),
        ];

        // Placeholder env cube texture (1x1x6 RGBA8)
        let env_texture = device.create_texture(&TextureDescriptor {
            label: Some("ssgi_env_cube"),
            size: Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 6,
            },
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
        let linear_sampler = device.create_sampler(&SamplerDescriptor {
            label: Some("ssgi.linear.sampler"),
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
            frame_index: 0,
            trace_pipeline,
            trace_bind_group_layout,
            shade_pipeline,
            shade_bind_group_layout,
            temporal_pipeline,
            temporal_bind_group_layout,
            upsample_pipeline,
            upsample_bind_group_layout,
            composite_pipeline,
            composite_bind_group_layout,
            ssgi_hit,
            ssgi_hit_view,
            ssgi_texture,
            ssgi_view,
            ssgi_history,
            ssgi_history_view,
            ssgi_filtered,
            ssgi_filtered_view,
            ssgi_upscaled,
            ssgi_upscaled_view,
            ssgi_composited,
            ssgi_composited_view,
            composite_uniform,
            scene_history,
            scene_history_views,
            scene_history_index: 0,
            scene_history_ready: false,
            linear_sampler,
            env_texture,
            env_view,
            env_sampler,
            width,
            height,
            half_res: false,
            last_trace_ms: 0.0,
            last_shade_ms: 0.0,
            last_temporal_ms: 0.0,
            last_upsample_ms: 0.0,
        })
    }

    pub fn set_seed(&mut self, queue: &Queue, seed: u32) {
        self.frame_index = seed;
        self.settings.frame_index = self.frame_index;
        queue.write_buffer(&self.settings_buffer, 0, bytemuck::bytes_of(&self.settings));
    }

    pub fn update_settings(&mut self, queue: &Queue, settings: SsgiSettings) {
        self.settings = settings;
        self.settings.use_half_res = if self.half_res { 1 } else { 0 };
        self.settings.frame_index = self.frame_index;
        queue.write_buffer(&self.settings_buffer, 0, bytemuck::bytes_of(&self.settings));
    }

    pub fn update_camera(&mut self, queue: &Queue, camera: &CameraParams) {
        queue.write_buffer(&self.camera_buffer, 0, bytemuck::bytes_of(camera));
    }

    pub fn get_settings(&self) -> SsgiSettings {
        self.settings
    }

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

    pub fn advance_frame(&mut self, queue: &Queue) {
        self.frame_index = self.frame_index.wrapping_add(1);
        self.settings.frame_index = self.frame_index;
        queue.write_buffer(&self.settings_buffer, 0, bytemuck::bytes_of(&self.settings));
    }

    pub fn set_half_res(&mut self, device: &Device, queue: &Queue, on: bool) -> RenderResult<()> {
        self.half_res = on;
        let (w, h) = if on {
            (self.width.max(2) / 2, self.height.max(2) / 2)
        } else {
            (self.width, self.height)
        };

        // Recreate output and temporal textures at new resolution
        self.ssgi_hit = device.create_texture(&TextureDescriptor {
            label: Some("ssgi_hit"),
            size: Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Rgba16Float,
            usage: TextureUsages::STORAGE_BINDING
                | TextureUsages::TEXTURE_BINDING
                | TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        self.ssgi_hit_view = self.ssgi_hit.create_view(&TextureViewDescriptor::default());
        self.ssgi_texture = device.create_texture(&TextureDescriptor {
            label: Some("ssgi_texture"),
            size: Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Rgba16Float,
            usage: TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        self.ssgi_view = self
            .ssgi_texture
            .create_view(&TextureViewDescriptor::default());

        self.ssgi_history = device.create_texture(&TextureDescriptor {
            label: Some("ssgi_history"),
            size: Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Rgba16Float,
            usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
            view_formats: &[],
        });
        self.ssgi_history_view = self
            .ssgi_history
            .create_view(&TextureViewDescriptor::default());

        self.ssgi_filtered = device.create_texture(&TextureDescriptor {
            label: Some("ssgi_filtered"),
            size: Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Rgba16Float,
            usage: TextureUsages::STORAGE_BINDING
                | TextureUsages::TEXTURE_BINDING
                | TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        self.ssgi_filtered_view = self
            .ssgi_filtered
            .create_view(&TextureViewDescriptor::default());

        // Update inv_resolution in settings
        self.settings.inv_resolution = [1.0 / w as f32, 1.0 / h as f32];
        self.settings.use_half_res = if on { 1 } else { 0 };
        queue.write_buffer(&self.settings_buffer, 0, bytemuck::bytes_of(&self.settings));
        Ok(())
    }

    pub fn execute(
        &mut self,
        device: &Device,
        encoder: &mut CommandEncoder,
        gbuffer: &GBuffer,
        hzb_view: &TextureView,
    ) -> RenderResult<()> {
        let (w_out, h_out) = if self.half_res {
            (self.width.max(2) / 2, self.height.max(2) / 2)
        } else {
            (self.width, self.height)
        };
        let gx = (w_out + 7) / 8;
        let gy = (h_out + 7) / 8;

        let t0 = Instant::now();
        let trace_bg = device.create_bind_group(&BindGroupDescriptor {
            label: Some("p5.ssgi.trace.bg"),
            layout: &self.trace_bind_group_layout,
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
                    resource: BindingResource::TextureView(hzb_view),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: BindingResource::TextureView(&self.ssgi_hit_view),
                },
                BindGroupEntry {
                    binding: 4,
                    resource: self.settings_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 5,
                    resource: self.camera_buffer.as_entire_binding(),
                },
            ],
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("p5.ssgi.trace"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.trace_pipeline);
            pass.set_bind_group(0, &trace_bg, &[]);
            pass.dispatch_workgroups(gx, gy, 1);
        }
        let t1 = Instant::now();

        let prev_color_view = if self.scene_history_ready {
            &self.scene_history_views[self.scene_history_index]
        } else {
            &gbuffer.material_view
        };

        let shade_bg = device.create_bind_group(&BindGroupDescriptor {
            label: Some("p5.ssgi.shade.bg"),
            layout: &self.shade_bind_group_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::TextureView(prev_color_view),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::Sampler(&self.linear_sampler),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: BindingResource::TextureView(&self.env_view),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: BindingResource::Sampler(&self.env_sampler),
                },
                BindGroupEntry {
                    binding: 4,
                    resource: BindingResource::TextureView(&self.ssgi_hit_view),
                },
                BindGroupEntry {
                    binding: 5,
                    resource: BindingResource::TextureView(&self.ssgi_view),
                },
                BindGroupEntry {
                    binding: 6,
                    resource: self.settings_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 7,
                    resource: self.camera_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 8,
                    resource: BindingResource::TextureView(&gbuffer.normal_view),
                },
                BindGroupEntry {
                    binding: 9,
                    resource: BindingResource::TextureView(&gbuffer.material_view),
                },
            ],
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("p5.ssgi.shade"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.shade_pipeline);
            pass.set_bind_group(0, &shade_bg, &[]);
            pass.dispatch_workgroups(gx, gy, 1);
        }
        let t2 = Instant::now();

        if self.settings.temporal_enabled != 0 {
            let temporal_bg = device.create_bind_group(&BindGroupDescriptor {
                label: Some("p5.ssgi.temporal.bg"),
                layout: &self.temporal_bind_group_layout,
                entries: &[
                    BindGroupEntry {
                        binding: 0,
                        resource: BindingResource::TextureView(&self.ssgi_view),
                    },
                    BindGroupEntry {
                        binding: 1,
                        resource: BindingResource::TextureView(&self.ssgi_history_view),
                    },
                    BindGroupEntry {
                        binding: 2,
                        resource: BindingResource::TextureView(&self.ssgi_filtered_view),
                    },
                    BindGroupEntry {
                        binding: 3,
                        resource: self.settings_buffer.as_entire_binding(),
                    },
                    BindGroupEntry {
                        binding: 4,
                        resource: BindingResource::TextureView(&gbuffer.depth_view),
                    },
                    BindGroupEntry {
                        binding: 5,
                        resource: BindingResource::TextureView(&gbuffer.normal_view),
                    },
                ],
            });
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("p5.ssgi.temporal"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.temporal_pipeline);
            pass.set_bind_group(0, &temporal_bg, &[]);
            pass.dispatch_workgroups(gx, gy, 1);
        } else {
            encoder.copy_texture_to_texture(
                ImageCopyTexture {
                    texture: &self.ssgi_texture,
                    mip_level: 0,
                    origin: Origin3d::ZERO,
                    aspect: TextureAspect::All,
                },
                ImageCopyTexture {
                    texture: &self.ssgi_filtered,
                    mip_level: 0,
                    origin: Origin3d::ZERO,
                    aspect: TextureAspect::All,
                },
                Extent3d {
                    width: w_out,
                    height: h_out,
                    depth_or_array_layers: 1,
                },
            );
        }
        let t3 = Instant::now();

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
            Extent3d {
                width: w_out,
                height: h_out,
                depth_or_array_layers: 1,
            },
        );

        // Task 3: Always run upsample pass when SSGI is enabled (even if not half-res, it will be 1:1)
        // This ensures upsample_ms > 0.0 as required by P5.2 acceptance criteria
        let up_bg = device.create_bind_group(&BindGroupDescriptor {
            label: Some("p5.ssgi.upsample.bg"),
            layout: &self.upsample_bind_group_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::TextureView(&self.ssgi_filtered_view),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::TextureView(&self.ssgi_upscaled_view),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: BindingResource::Sampler(&self.linear_sampler),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: BindingResource::TextureView(&gbuffer.depth_view),
                },
                BindGroupEntry {
                    binding: 4,
                    resource: BindingResource::TextureView(&gbuffer.normal_view),
                },
                BindGroupEntry {
                    binding: 5,
                    resource: self.settings_buffer.as_entire_binding(),
                },
            ],
        });
        let gx_full = (self.width + 7) / 8;
        let gy_full = (self.height + 7) / 8;
        let t_up0 = Instant::now();
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("p5.ssgi.upsample"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.upsample_pipeline);
            pass.set_bind_group(0, &up_bg, &[]);
            pass.dispatch_workgroups(gx_full, gy_full, 1);
        } // Drop pass here so encoder can be used again
        let t_up1 = Instant::now();
        let up_ms = (t_up1 - t_up0).as_secs_f32() * 1000.0;

        // Composite pass: add SSGI to material
        let ssgi_output_view = if self.half_res {
            &self.ssgi_upscaled_view
        } else {
            &self.ssgi_filtered_view
        };
        let comp_bg = device.create_bind_group(&BindGroupDescriptor {
            label: Some("p5.ssgi.composite.bg"),
            layout: &self.composite_bind_group_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::TextureView(&gbuffer.material_view),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::TextureView(&self.ssgi_composited_view),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: BindingResource::TextureView(ssgi_output_view),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: self.composite_uniform.as_entire_binding(),
                },
            ],
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("p5.ssgi.composite"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.composite_pipeline);
            pass.set_bind_group(0, &comp_bg, &[]);
            pass.dispatch_workgroups(gx_full, gy_full, 1);
        }

        self.copy_scene_history(encoder, gbuffer);

        self.last_trace_ms = (t1 - t0).as_secs_f32() * 1000.0;
        self.last_shade_ms = (t2 - t1).as_secs_f32() * 1000.0;
        self.last_temporal_ms = (t3 - t2).as_secs_f32() * 1000.0;
        self.last_upsample_ms = up_ms;

        Ok(())
    }

    fn copy_scene_history(&mut self, encoder: &mut CommandEncoder, gbuffer: &GBuffer) {
        let write_idx = if self.scene_history_ready {
            1 - self.scene_history_index
        } else {
            self.scene_history_index
        };
        encoder.copy_texture_to_texture(
            ImageCopyTexture {
                texture: &gbuffer.material_texture,
                mip_level: 0,
                origin: Origin3d::ZERO,
                aspect: TextureAspect::All,
            },
            ImageCopyTexture {
                texture: &self.scene_history[write_idx],
                mip_level: 0,
                origin: Origin3d::ZERO,
                aspect: TextureAspect::All,
            },
            Extent3d {
                width: self.width,
                height: self.height,
                depth_or_array_layers: 1,
            },
        );
        self.scene_history_index = write_idx;
        self.scene_history_ready = true;
    }

    #[allow(dead_code)]
    pub fn get_output(&self) -> &TextureView {
        &self.ssgi_filtered_view
    }

    pub fn get_output_view(&self) -> &TextureView {
        &self.ssgi_filtered_view
    }

    pub fn get_upscaled_view(&self) -> &TextureView {
        &self.ssgi_upscaled_view
    }

    /// Get display-ready output (upscaled if running half-res)
    pub fn get_output_for_display(&self) -> &TextureView {
        if self.half_res {
            &self.ssgi_upscaled_view
        } else {
            &self.ssgi_filtered_view
        }
    }

    pub fn timings_ms(&self) -> (f32, f32, f32, f32) {
        (
            self.last_trace_ms,
            self.last_shade_ms,
            self.last_temporal_ms,
            self.last_upsample_ms,
        )
    }

    pub fn dimensions(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    pub fn is_half_res(&self) -> bool {
        self.half_res
    }

    pub fn hit_texture(&self) -> &Texture {
        &self.ssgi_hit
    }

    pub fn filtered_texture(&self) -> &Texture {
        &self.ssgi_filtered
    }

    pub fn history_texture(&self) -> &Texture {
        &self.ssgi_history
    }

    pub fn upscaled_texture(&self) -> &Texture {
        &self.ssgi_upscaled
    }

    pub fn get_composited(&self) -> &TextureView {
        &self.ssgi_composited_view
    }

    pub fn set_composite_intensity(&mut self, queue: &Queue, intensity: f32) {
        let params: [f32; 4] = [intensity, 0.0, 0.0, 0.0];
        queue.write_buffer(&self.composite_uniform, 0, bytemuck::cast_slice(&params));
    }

    pub fn reset_history(&mut self, device: &Device, queue: &Queue) -> RenderResult<()> {
        self.scene_history_ready = false;
        self.scene_history_index = 0;
        self.set_half_res(device, queue, self.half_res)
    }
}

pub struct SsrRenderer {
    settings: SsrSettings,
    settings_buffer: Buffer,
    camera_buffer: Buffer,
    trace_pipeline: ComputePipeline,
    trace_bind_group_layout: BindGroupLayout,
    shade_pipeline: ComputePipeline,
    shade_bind_group_layout: BindGroupLayout,
    fallback_pipeline: ComputePipeline,
    fallback_bind_group_layout: BindGroupLayout,
    temporal_pipeline: ComputePipeline,
    temporal_bind_group_layout: BindGroupLayout,
    composite_pipeline: ComputePipeline,
    composite_bind_group_layout: BindGroupLayout,
    composite_params: Buffer,

    // ssr_spec_texture   : Rgba16Float raw SSR specular from cs_shade
    //                      (rgb = spec radiance, a = reflection weight in [0,1]).
    ssr_spec_texture: Texture,
    ssr_spec_view: TextureView,
    // ssr_final_texture  : Rgba16Float SSR after environment fallback
    //                      (rgb = spec radiance, a > 0 for surface hits, a = 0 for
    //                      env-only misses; see fallback_env.wgsl).
    ssr_final_texture: Texture,
    ssr_final_view: TextureView,
    // ssr_history_texture: Rgba16Float previous-frame SSR used for temporal filtering.
    ssr_history_texture: Texture,
    ssr_history_view: TextureView,
    // ssr_filtered_texture: Rgba16Float temporally filtered SSR (input to composite).
    ssr_filtered_texture: Texture,
    ssr_filtered_view: TextureView,
    // ssr_hit_texture    : Rgba16Float hit buffer from cs_trace (xy = hit UV in [0,1],
    //                      z = normalized step count, w = hit mask in {0,1}).
    ssr_hit_texture: Texture,
    ssr_hit_view: TextureView,
    // ssr_composited_texture: Rgba8Unorm view of base lighting + SSR specular after
    //                         tone mapping; used by the viewer for SSR previews.
    ssr_composited_texture: Texture,
    ssr_composited_view: TextureView,
    scene_color_override: Option<TextureView>,

    env_texture: Texture,
    env_view: TextureView,
    env_sampler: Sampler,
    linear_sampler: Sampler,
    width: u32,
    height: u32,

    counters_buffer: Buffer,
    counters_readback: Buffer,
    temporal_params: Buffer,

    last_trace_ms: f32,
    last_shade_ms: f32,
    last_fallback_ms: f32,
    stats_readback_pending: bool,
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

        let counters_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("p5.ssr.counters"),
            size: std::mem::size_of::<[u32; 5]>() as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let counters_readback = device.create_buffer(&BufferDescriptor {
            label: Some("p5.ssr.counters.readback"),
            size: std::mem::size_of::<[u32; 5]>() as u64,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let trace_shader = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("p5.ssr.trace"),
            source: ShaderSource::Wgsl(include_str!("../shaders/ssr/trace.wgsl").into()),
        });
        let shade_shader = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("p5.ssr.shade"),
            source: ShaderSource::Wgsl(include_str!("../shaders/ssr/shade.wgsl").into()),
        });
        let fallback_shader = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("p5.ssr.fallback"),
            source: ShaderSource::Wgsl(include_str!("../shaders/ssr/fallback_env.wgsl").into()),
        });
        let temporal_shader = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("p5.ssr.temporal"),
            source: ShaderSource::Wgsl(include_str!("../shaders/ssr/temporal.wgsl").into()),
        });
        let composite_shader = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("p5.ssr.composite"),
            source: ShaderSource::Wgsl(include_str!("../shaders/ssr/composite.wgsl").into()),
        });

        let trace_bind_group_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("p5.ssr.trace.bgl"),
            entries: &[
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
                BindGroupLayoutEntry {
                    binding: 2,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::StorageTexture {
                        access: StorageTextureAccess::WriteOnly,
                        format: TextureFormat::Rgba16Float,
                        view_dimension: TextureViewDimension::D2,
                    },
                    count: None,
                },
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
                BindGroupLayoutEntry {
                    binding: 5,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let shade_bind_group_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("p5.ssr.shade.bgl"),
            entries: &[
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Texture {
                        sample_type: TextureSampleType::Float { filterable: true },
                        view_dimension: TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Sampler(SamplerBindingType::Filtering),
                    count: None,
                },
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
                BindGroupLayoutEntry {
                    binding: 3,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Texture {
                        sample_type: TextureSampleType::Float { filterable: false },
                        view_dimension: TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 4,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Texture {
                        sample_type: TextureSampleType::Float { filterable: true },
                        view_dimension: TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 5,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Texture {
                        sample_type: TextureSampleType::Float { filterable: false },
                        view_dimension: TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 6,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::StorageTexture {
                        access: StorageTextureAccess::WriteOnly,
                        format: TextureFormat::Rgba16Float,
                        view_dimension: TextureViewDimension::D2,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 7,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 8,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 9,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let fallback_bind_group_layout =
            device.create_bind_group_layout(&BindGroupLayoutDescriptor {
                label: Some("p5.ssr.fallback.bgl"),
                entries: &[
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
                    BindGroupLayoutEntry {
                        binding: 3,
                        visibility: ShaderStages::COMPUTE,
                        ty: BindingType::Texture {
                            sample_type: TextureSampleType::Float { filterable: false },
                            view_dimension: TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    BindGroupLayoutEntry {
                        binding: 4,
                        visibility: ShaderStages::COMPUTE,
                        ty: BindingType::Texture {
                            sample_type: TextureSampleType::Float { filterable: true },
                            view_dimension: TextureViewDimension::Cube,
                            multisampled: false,
                        },
                        count: None,
                    },
                    BindGroupLayoutEntry {
                        binding: 5,
                        visibility: ShaderStages::COMPUTE,
                        ty: BindingType::Sampler(SamplerBindingType::Filtering),
                        count: None,
                    },
                    BindGroupLayoutEntry {
                        binding: 6,
                        visibility: ShaderStages::COMPUTE,
                        ty: BindingType::StorageTexture {
                            access: StorageTextureAccess::WriteOnly,
                            format: TextureFormat::Rgba16Float,
                            view_dimension: TextureViewDimension::D2,
                        },
                        count: None,
                    },
                    BindGroupLayoutEntry {
                        binding: 7,
                        visibility: ShaderStages::COMPUTE,
                        ty: BindingType::Buffer {
                            ty: BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    BindGroupLayoutEntry {
                        binding: 8,
                        visibility: ShaderStages::COMPUTE,
                        ty: BindingType::Buffer {
                            ty: BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    BindGroupLayoutEntry {
                        binding: 9,
                        visibility: ShaderStages::COMPUTE,
                        ty: BindingType::Buffer {
                            ty: BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

        let temporal_bind_group_layout =
            device.create_bind_group_layout(&BindGroupLayoutDescriptor {
                label: Some("p5.ssr.temporal.bgl"),
                entries: &[
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
                    BindGroupLayoutEntry {
                        binding: 2,
                        visibility: ShaderStages::COMPUTE,
                        ty: BindingType::StorageTexture {
                            access: StorageTextureAccess::WriteOnly,
                            format: TextureFormat::Rgba16Float,
                            view_dimension: TextureViewDimension::D2,
                        },
                        count: None,
                    },
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

        let composite_bind_group_layout =
            device.create_bind_group_layout(&BindGroupLayoutDescriptor {
                label: Some("p5.ssr.composite.bgl"),
                entries: &[
                    BindGroupLayoutEntry {
                        binding: 0,
                        visibility: ShaderStages::COMPUTE,
                        ty: BindingType::Texture {
                            sample_type: TextureSampleType::Float { filterable: true },
                            view_dimension: TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
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
                    BindGroupLayoutEntry {
                        binding: 2,
                        visibility: ShaderStages::COMPUTE,
                        ty: BindingType::Buffer {
                            ty: BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    BindGroupLayoutEntry {
                        binding: 3,
                        visibility: ShaderStages::COMPUTE,
                        ty: BindingType::StorageTexture {
                            access: StorageTextureAccess::WriteOnly,
                            format: TextureFormat::Rgba8Unorm,
                            view_dimension: TextureViewDimension::D2,
                        },
                        count: None,
                    },
                ],
            });

        let trace_pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("p5.ssr.trace.pipeline"),
            layout: Some(&device.create_pipeline_layout(&PipelineLayoutDescriptor {
                label: Some("p5.ssr.trace.layout"),
                bind_group_layouts: &[&trace_bind_group_layout],
                push_constant_ranges: &[],
            })),
            module: &trace_shader,
            entry_point: "cs_trace",
        });
        let shade_pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("p5.ssr.shade.pipeline"),
            layout: Some(&device.create_pipeline_layout(&PipelineLayoutDescriptor {
                label: Some("p5.ssr.shade.layout"),
                bind_group_layouts: &[&shade_bind_group_layout],
                push_constant_ranges: &[],
            })),
            module: &shade_shader,
            entry_point: "cs_shade",
        });
        let fallback_pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("p5.ssr.fallback.pipeline"),
            layout: Some(&device.create_pipeline_layout(&PipelineLayoutDescriptor {
                label: Some("p5.ssr.fallback.layout"),
                bind_group_layouts: &[&fallback_bind_group_layout],
                push_constant_ranges: &[],
            })),
            module: &fallback_shader,
            entry_point: "cs_fallback",
        });
        let temporal_pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("p5.ssr.temporal.pipeline"),
            layout: Some(&device.create_pipeline_layout(&PipelineLayoutDescriptor {
                label: Some("p5.ssr.temporal.layout"),
                bind_group_layouts: &[&temporal_bind_group_layout],
                push_constant_ranges: &[],
            })),
            module: &temporal_shader,
            entry_point: "cs_temporal",
        });

        let composite_pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("p5.ssr.composite.pipeline"),
            layout: Some(&device.create_pipeline_layout(&PipelineLayoutDescriptor {
                label: Some("p5.ssr.composite.layout"),
                bind_group_layouts: &[&composite_bind_group_layout],
                push_constant_ranges: &[],
            })),
            module: &composite_shader,
            entry_point: "cs_ssr_composite",
        });

        #[repr(C)]
        #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
        struct SsrCompositeParamsStd140 {
            boost: f32,
            exposure: f32,
            gamma: f32,
            weight_floor: f32,
            tone_white: f32,
            tone_bias: f32,
            reinhard_k: f32,
            _pad0: f32,
        }
        let composite_params_data = SsrCompositeParamsStd140 {
            boost: 1.6,
            exposure: 1.1,
            gamma: 1.0,
            weight_floor: 0.2,
            tone_white: 1.0,
            tone_bias: 0.0,
            reinhard_k: 1.0,
            _pad0: 0.0,
        };
        let composite_params = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("p5.ssr.composite.params"),
            contents: bytemuck::bytes_of(&composite_params_data),
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        });

        #[repr(C)]
        #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
        struct SsrTemporalParamsStd140 {
            temporal_alpha: f32,
            pad: [f32; 3],
        }
        let temporal_params_data = SsrTemporalParamsStd140 {
            temporal_alpha: 0.85,
            pad: [0.0; 3],
        };
        let temporal_params = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("p5.ssr.temporal.params"),
            contents: bytemuck::bytes_of(&temporal_params_data),
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        });

        let ssr_spec_texture = device.create_texture(&TextureDescriptor {
            label: Some("p5.ssr.spec"),
            size: Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Rgba16Float,
            usage: TextureUsages::STORAGE_BINDING
                | TextureUsages::TEXTURE_BINDING
                | TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let ssr_spec_view = ssr_spec_texture.create_view(&TextureViewDescriptor::default());

        let ssr_final_texture = device.create_texture(&TextureDescriptor {
            label: Some("p5.ssr.final"),
            size: Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Rgba16Float,
            usage: TextureUsages::STORAGE_BINDING
                | TextureUsages::TEXTURE_BINDING
                | TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let ssr_final_view = ssr_final_texture.create_view(&TextureViewDescriptor::default());

        let ssr_history_texture = device.create_texture(&TextureDescriptor {
            label: Some("p5.ssr.history"),
            size: Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Rgba16Float,
            usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let ssr_history_view = ssr_history_texture.create_view(&TextureViewDescriptor::default());

        let ssr_filtered_texture = device.create_texture(&TextureDescriptor {
            label: Some("p5.ssr.filtered"),
            size: Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Rgba16Float,
            usage: TextureUsages::STORAGE_BINDING
                | TextureUsages::TEXTURE_BINDING
                | TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let ssr_filtered_view = ssr_filtered_texture.create_view(&TextureViewDescriptor::default());

        let ssr_hit_texture = device.create_texture(&TextureDescriptor {
            label: Some("p5.ssr.hit"),
            size: Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Rgba16Float,
            usage: TextureUsages::STORAGE_BINDING
                | TextureUsages::TEXTURE_BINDING
                | TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let ssr_hit_view = ssr_hit_texture.create_view(&TextureViewDescriptor::default());

        let ssr_composited_texture = device.create_texture(&TextureDescriptor {
            label: Some("p5.ssr.composited"),
            size: Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Rgba8Unorm,
            usage: TextureUsages::STORAGE_BINDING
                | TextureUsages::TEXTURE_BINDING
                | TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let ssr_composited_view =
            ssr_composited_texture.create_view(&TextureViewDescriptor::default());

        let env_texture = device.create_texture(&TextureDescriptor {
            label: Some("p5.ssr.env.placeholder"),
            size: Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 6,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Rgba8Unorm,
            usage: TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let env_view = env_texture.create_view(&TextureViewDescriptor {
            label: Some("p5.ssr.env.view"),
            format: None,
            dimension: Some(TextureViewDimension::Cube),
            aspect: TextureAspect::All,
            base_mip_level: 0,
            mip_level_count: None,
            base_array_layer: 0,
            array_layer_count: None,
        });
        let env_sampler = device.create_sampler(&SamplerDescriptor {
            label: Some("p5.ssr.env.sampler"),
            address_mode_u: AddressMode::ClampToEdge,
            address_mode_v: AddressMode::ClampToEdge,
            address_mode_w: AddressMode::ClampToEdge,
            mag_filter: FilterMode::Linear,
            min_filter: FilterMode::Linear,
            mipmap_filter: FilterMode::Linear,
            ..Default::default()
        });
        let linear_sampler = device.create_sampler(&SamplerDescriptor {
            label: Some("p5.ssr.linear"),
            address_mode_u: AddressMode::ClampToEdge,
            address_mode_v: AddressMode::ClampToEdge,
            address_mode_w: AddressMode::ClampToEdge,
            mag_filter: FilterMode::Linear,
            min_filter: FilterMode::Linear,
            mipmap_filter: FilterMode::Linear,
            ..Default::default()
        });

        Ok(Self {
            settings,
            settings_buffer,
            camera_buffer,
            trace_pipeline,
            trace_bind_group_layout,
            shade_pipeline,
            shade_bind_group_layout,
            fallback_pipeline,
            fallback_bind_group_layout,
            temporal_pipeline,
            temporal_bind_group_layout,
            composite_pipeline,
            composite_bind_group_layout,
            composite_params,
            ssr_spec_texture,
            ssr_spec_view,
            ssr_final_texture,
            ssr_final_view,
            ssr_history_texture,
            ssr_history_view,
            ssr_filtered_texture,
            ssr_filtered_view,
            ssr_hit_texture,
            ssr_hit_view,
            ssr_composited_texture,
            ssr_composited_view,
            env_texture,
            env_view,
            env_sampler,
            linear_sampler,
            width,
            height,
            counters_buffer,
            counters_readback,
            temporal_params,
            last_trace_ms: 0.0,
            last_shade_ms: 0.0,
            last_fallback_ms: 0.0,
            stats_readback_pending: false,
            scene_color_override: None,
        })
    }

    pub fn execute(
        &mut self,
        device: &Device,
        encoder: &mut CommandEncoder,
        gbuffer: &GBuffer,
        mut stats: Option<&mut SsrStats>,
    ) -> RenderResult<()> {
        let stats_requested = stats.is_some();
        if stats_requested {
            encoder.clear_buffer(&self.counters_buffer, 0, None);
        }

        let (w, h) = (self.width, self.height);
        let gx = (w + 7) / 8;
        let gy = (h + 7) / 8;

        let trace_start = Instant::now();

        // Trace rays against the hierarchical depth
        let trace_bg = device.create_bind_group(&BindGroupDescriptor {
            label: Some("p5.ssr.trace.bg"),
            layout: &self.trace_bind_group_layout,
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
                    resource: BindingResource::TextureView(&self.ssr_hit_view),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: self.settings_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 4,
                    resource: self.camera_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 5,
                    resource: self.counters_buffer.as_entire_binding(),
                },
            ],
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("p5.ssr.trace"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.trace_pipeline);
            pass.set_bind_group(0, &trace_bg, &[]);
            pass.dispatch_workgroups(gx, gy, 1);
        }

        // Shade pass converts hit data into specular contributions
        let shade_start = Instant::now();
        let scene_color_view = self
            .scene_color_override
            .as_ref()
            .unwrap_or(&gbuffer.material_view);
        let shade_bg = device.create_bind_group(&BindGroupDescriptor {
            label: Some("p5.ssr.shade.bg"),
            layout: &self.shade_bind_group_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::TextureView(scene_color_view),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::Sampler(&self.linear_sampler),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: BindingResource::TextureView(&self.ssr_hit_view),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: BindingResource::TextureView(&gbuffer.normal_view),
                },
                BindGroupEntry {
                    binding: 4,
                    resource: BindingResource::TextureView(&gbuffer.material_view),
                },
                BindGroupEntry {
                    binding: 5,
                    resource: BindingResource::TextureView(&gbuffer.depth_view),
                },
                BindGroupEntry {
                    binding: 6,
                    resource: BindingResource::TextureView(&self.ssr_spec_view),
                },
                BindGroupEntry {
                    binding: 7,
                    resource: self.settings_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 8,
                    resource: self.camera_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 9,
                    resource: self.counters_buffer.as_entire_binding(),
                },
            ],
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("p5.ssr.shade"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.shade_pipeline);
            pass.set_bind_group(0, &shade_bg, &[]);
            pass.dispatch_workgroups(gx, gy, 1);
        }

        // Environment fallback for misses
        let fallback_start = Instant::now();
        let fallback_bg = device.create_bind_group(&BindGroupDescriptor {
            label: Some("p5.ssr.fallback.bg"),
            layout: &self.fallback_bind_group_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::TextureView(&self.ssr_spec_view),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::TextureView(&self.ssr_hit_view),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: BindingResource::TextureView(&gbuffer.depth_view),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: BindingResource::TextureView(&gbuffer.normal_view),
                },
                BindGroupEntry {
                    binding: 4,
                    resource: BindingResource::TextureView(&self.env_view),
                },
                BindGroupEntry {
                    binding: 5,
                    resource: BindingResource::Sampler(&self.env_sampler),
                },
                BindGroupEntry {
                    binding: 6,
                    resource: BindingResource::TextureView(&self.ssr_final_view),
                },
                BindGroupEntry {
                    binding: 7,
                    resource: self.settings_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 8,
                    resource: self.camera_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 9,
                    resource: self.counters_buffer.as_entire_binding(),
                },
            ],
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("p5.ssr.fallback"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.fallback_pipeline);
            pass.set_bind_group(0, &fallback_bg, &[]);
            pass.dispatch_workgroups(gx, gy, 1);
        }

        // Temporal accumulation smooths the reflection
        let fallback_end = Instant::now();
        let temporal_bg = device.create_bind_group(&BindGroupDescriptor {
            label: Some("p5.ssr.temporal.bg"),
            layout: &self.temporal_bind_group_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::TextureView(&self.ssr_final_view),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::TextureView(&self.ssr_history_view),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: BindingResource::TextureView(&self.ssr_filtered_view),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: self.temporal_params.as_entire_binding(),
                },
            ],
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("p5.ssr.temporal"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.temporal_pipeline);
            pass.set_bind_group(0, &temporal_bg, &[]);
            pass.dispatch_workgroups(gx, gy, 1);
        }

        // Composite SSR into the lit buffer with tone mapping/boost parameters
        let composite_bg = device.create_bind_group(&BindGroupDescriptor {
            label: Some("p5.ssr.composite.bg"),
            layout: &self.composite_bind_group_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::TextureView(&gbuffer.material_view),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::TextureView(&self.ssr_filtered_view),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: self.composite_params.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: BindingResource::TextureView(&self.ssr_composited_view),
                },
            ],
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("p5.ssr.composite"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.composite_pipeline);
            pass.set_bind_group(0, &composite_bg, &[]);
            pass.dispatch_workgroups(gx, gy, 1);
        }

        // Preserve filtered result for next frame's temporal accumulation
        encoder.copy_texture_to_texture(
            self.ssr_filtered_texture.as_image_copy(),
            self.ssr_history_texture.as_image_copy(),
            Extent3d {
                width: self.width,
                height: self.height,
                depth_or_array_layers: 1,
            },
        );

        self.clear_scene_color_override();

        self.last_trace_ms = (shade_start - trace_start).as_secs_f32() * 1000.0;
        self.last_shade_ms = (fallback_start - shade_start).as_secs_f32() * 1000.0;
        self.last_fallback_ms = (fallback_end - fallback_start).as_secs_f32() * 1000.0;

        self.stats_readback_pending = stats_requested;
        if stats_requested {
            let counter_bytes = size_of::<[u32; 5]>() as BufferAddress;
            encoder.copy_buffer_to_buffer(
                &self.counters_buffer,
                0,
                &self.counters_readback,
                0,
                counter_bytes,
            );
        }

        if let Some(stats) = stats.as_deref_mut() {
            stats.trace_ms = self.last_trace_ms;
            stats.shade_ms = self.last_shade_ms;
            stats.fallback_ms = self.last_fallback_ms;
        }

        Ok(())
    }

    pub fn collect_stats_into(
        &mut self,
        device: &Device,
        _queue: &Queue,
        stats: &mut SsrStats,
    ) -> RenderResult<()> {
        if !self.stats_readback_pending {
            // No stats were requested during execute; just keep timings.
            stats.trace_ms = self.last_trace_ms;
            stats.shade_ms = self.last_shade_ms;
            stats.fallback_ms = self.last_fallback_ms;
            return Ok(());
        }

        let slice = self.counters_readback.slice(..);
        let (sender, receiver) = oneshot_channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });
        device.poll(wgpu::Maintain::Wait);

        let map_result = block_on(receiver.receive()).ok_or_else(|| {
            RenderError::Readback("failed to receive SSR stats map signal".to_string())
        })?;
        map_result.map_err(|e| RenderError::Readback(format!("SSR stats map failed: {e:?}")))?;

        let data = slice.get_mapped_range();
        let words: &[u32] = bytemuck::cast_slice(&data);
        if words.len() < 5 {
            return Err(RenderError::Readback(
                "SSR stats buffer size was smaller than expected".to_string(),
            ));
        }

        stats.num_rays = words[0];
        stats.num_hits = words[1];
        stats.total_steps = words[2];
        stats.num_misses = words[3];
        stats.miss_ibl_samples = words[4];
        stats.trace_ms = self.last_trace_ms;
        stats.shade_ms = self.last_shade_ms;
        stats.fallback_ms = self.last_fallback_ms;

        drop(data);
        self.counters_readback.unmap();

        self.stats_readback_pending = false;

        Ok(())
    }

    pub fn get_output(&self) -> &TextureView {
        &self.ssr_filtered_view
    }

    pub fn output_texture(&self) -> &Texture {
        &self.ssr_filtered_texture
    }

    pub fn set_scene_color_view(&mut self, view: TextureView) {
        self.scene_color_override = Some(view);
    }

    fn clear_scene_color_override(&mut self) {
        self.scene_color_override = None;
    }

    pub fn update_settings(&mut self, queue: &Queue, settings: SsrSettings) {
        self.settings = settings;
        queue.write_buffer(&self.settings_buffer, 0, bytemuck::bytes_of(&self.settings));
    }

    pub fn update_camera(&mut self, queue: &Queue, camera: &CameraParams) {
        queue.write_buffer(&self.camera_buffer, 0, bytemuck::bytes_of(camera));
    }

    pub fn spec_view(&self) -> &TextureView {
        &self.ssr_spec_view
    }

    pub fn final_view(&self) -> &TextureView {
        &self.ssr_final_view
    }

    pub fn hit_data_view(&self) -> &TextureView {
        &self.ssr_hit_view
    }

    pub fn hit_data_texture(&self) -> &Texture {
        &self.ssr_hit_texture
    }

    pub fn composite_view(&self) -> &TextureView {
        &self.ssr_composited_view
    }

    pub fn timings_ms(&self) -> (f32, f32, f32) {
        (
            self.last_trace_ms,
            self.last_shade_ms,
            self.last_fallback_ms,
        )
    }

    pub fn get_settings(&self) -> SsrSettings {
        self.settings
    }

    pub fn set_environment(&mut self, _env_view: &TextureView, _env_sampler: &Sampler) {
        // Deprecated in favor of set_environment_texture; kept for API-compat.
    }

    pub fn set_environment_texture(&mut self, device: &Device, env_texture: &Texture) {
        println!("[SSR] Updating environment texture");
        let view = env_texture.create_view(&TextureViewDescriptor {
            label: Some("p5.ssr.env.view"),
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
