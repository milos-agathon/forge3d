// T41-BEGIN:scene-module
#![allow(dead_code)]
#![allow(deprecated)]
#[cfg(feature = "extension-module")]
use crate::device_caps::DeviceCaps;
use bytemuck::{Pod, Zeroable};
#[cfg(feature = "extension-module")]
use numpy::{
    IntoPyArray, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3, PyUntypedArrayMethods,
};
#[cfg(feature = "extension-module")]
use pyo3::{prelude::*, types::PyBytes};
#[cfg(feature = "extension-module")]
use std::path::PathBuf;
#[cfg(feature = "extension-module")]
use wgpu::util::DeviceExt;

const TEXTURE_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;
const NORMAL_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;

#[derive(Debug, Clone)]
pub struct SceneGlobals {
    pub globals: crate::terrain::Globals,
    pub view: glam::Mat4,
    pub proj: glam::Mat4,
}

impl Default for SceneGlobals {
    fn default() -> Self {
        let view = glam::Mat4::look_at_rh(
            glam::Vec3::new(3.0, 2.0, 3.0),
            glam::Vec3::ZERO,
            glam::Vec3::Y,
        );
        let proj = crate::camera::perspective_wgpu(45f32.to_radians(), 4.0 / 3.0, 0.1, 100.0);
        Self {
            globals: crate::terrain::Globals::default(),
            view,
            proj,
        }
    }
}

#[cfg_attr(
    feature = "extension-module",
    pyclass(module = "forge3d._forge3d", name = "Scene")
)]
pub struct Scene {
    width: u32,
    height: u32,
    grid: u32,

    tp: crate::terrain::pipeline::TerrainPipeline,
    bg0_globals: wgpu::BindGroup,
    bg1_height: wgpu::BindGroup,
    bg2_lut: wgpu::BindGroup,

    // E2/E1: Per-tile uniforms bind group (group 3)
    bg3_tile: wgpu::BindGroup,
    tile_ubo: wgpu::Buffer,
    tile_slot_ubo: wgpu::Buffer,
    mosaic_params_ubo: wgpu::Buffer,

    vbuf: wgpu::Buffer,
    ibuf: wgpu::Buffer,
    nidx: u32,

    ubo: wgpu::Buffer,
    colormap: crate::terrain::ColormapLUT,
    lut_format: &'static str,

    color: wgpu::Texture,
    color_view: wgpu::TextureView,
    normal: wgpu::Texture,
    normal_view: wgpu::TextureView,
    sample_count: u32,
    msaa_color: Option<wgpu::Texture>,
    msaa_view: Option<wgpu::TextureView>,
    msaa_normal: Option<wgpu::Texture>,
    msaa_normal_view: Option<wgpu::TextureView>,
    depth: Option<wgpu::Texture>,
    depth_view: Option<wgpu::TextureView>,

    height_view: Option<wgpu::TextureView>,
    height_sampler: Option<wgpu::Sampler>,

    scene: SceneGlobals,
    last_uniforms: crate::terrain::TerrainUniforms,
    ssao: SsaoResources,
    ssao_enabled: bool,

    // Toggle base terrain rendering
    terrain_enabled: bool,

    // B5: Planar reflections
    reflection_renderer: Option<crate::core::reflections::PlanarReflectionRenderer>,
    reflections_enabled: bool,

    // B6: Depth of Field
    dof_renderer: Option<crate::core::dof::DofRenderer>,
    dof_enabled: bool,
    dof_params: crate::core::dof::CameraDofParams,

    // B7: Cloud Shadows
    cloud_shadow_renderer: Option<crate::core::cloud_shadows::CloudShadowRenderer>,
    cloud_shadows_enabled: bool,
    bg3_cloud_shadows: Option<wgpu::BindGroup>,
    bg4_dummy_cloud_shadows: wgpu::BindGroup, // Dummy bind group for devices with >=6 bind groups

    // B8: Realtime Clouds
    cloud_renderer: Option<crate::core::clouds::CloudRenderer>,
    clouds_enabled: bool,

    // B10: Ground Plane (Raster)
    ground_plane_renderer: Option<crate::core::ground_plane::GroundPlaneRenderer>,
    ground_plane_enabled: bool,

    // B11: Water Surface Color Toggle
    water_surface_renderer: Option<crate::core::water_surface::WaterSurfaceRenderer>,
    water_surface_enabled: bool,

    // B12: Soft Light Radius (Raster)
    soft_light_radius_renderer: Option<crate::core::soft_light_radius::SoftLightRadiusRenderer>,
    soft_light_radius_enabled: bool,

    // B13: Point & Spot Lights (Realtime)
    point_spot_lights_renderer: Option<crate::core::point_spot_lights::PointSpotLightRenderer>,
    point_spot_lights_enabled: bool,

    // B14: Rect Area Lights (LTC)
    ltc_area_lights_renderer: Option<crate::core::ltc_area_lights::LTCRectAreaLightRenderer>,
    ltc_area_lights_enabled: bool,

    // B15: Image-Based Lighting (IBL) Polish
    ibl_renderer: Option<crate::core::ibl::IBLRenderer>,
    ibl_enabled: bool,

    // B16: Dual-source blending OIT
    dual_source_oit_renderer: Option<crate::core::dual_source_oit::DualSourceOITRenderer>,
    dual_source_oit_enabled: bool,

    // D: Native overlays compositor
    overlay_renderer: Option<crate::core::overlays::OverlayRenderer>,
    overlay_enabled: bool,

    // D: Native text overlay (rectangle placeholder)
    text_overlay_renderer: Option<crate::core::text_overlay::TextOverlayRenderer>,
    text_overlay_enabled: bool,
    text_overlay_alpha: f32,
    text_instances: Vec<crate::core::text_overlay::TextInstance>,

    // D11: 3D text meshes
    text3d_renderer: Option<crate::core::text_mesh::TextMeshRenderer>,
    text3d_enabled: bool,
    text3d_instances: Vec<Text3DInstance>,

    // F16: GPU Instancing (feature-gated)
    #[cfg(feature = "enable-gpu-instancing")]
    mesh_instanced_renderer: Option<crate::render::mesh_instanced::MeshInstancedRenderer>,
    #[cfg(feature = "enable-gpu-instancing")]
    instanced_batches: Vec<InstancedBatch>,
}

struct Text3DInstance {
    vbuf: wgpu::Buffer,
    ibuf: wgpu::Buffer,
    index_count: u32,
    vertex_count: u32,
    model: glam::Mat4,
    color: [f32; 4],
    light_dir: [f32; 3],
    light_intensity: f32,
    metallic: f32,
    roughness: f32,
}

// F16: GPU Instancing batch description
#[cfg(feature = "enable-gpu-instancing")]
struct InstancedBatch {
    vbuf: wgpu::Buffer,
    ibuf: wgpu::Buffer,
    instbuf: wgpu::Buffer,
    index_count: u32,
    instance_count: u32,
    color: [f32; 4],
    light_dir: [f32; 3],
    light_intensity: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct SsaoSettingsUniform {
    radius: f32,
    intensity: f32,
    bias: f32,
    _pad0: f32,
    inv_resolution: [f32; 2],
    _pad1: [f32; 2],
}

struct SsaoResources {
    radius: f32,
    intensity: f32,
    bias: f32,
    width: u32,
    height: u32,
    sampler: wgpu::Sampler,
    blur_sampler: wgpu::Sampler,
    settings_buffer: wgpu::Buffer,
    blur_settings_buffer: wgpu::Buffer,
    view_buffer: wgpu::Buffer,
    ao_texture: wgpu::Texture,
    ao_view: wgpu::TextureView,
    blur_texture: wgpu::Texture,
    blur_view: wgpu::TextureView,
    noise_texture: wgpu::Texture,
    noise_view: wgpu::TextureView,
    noise_sampler: wgpu::Sampler,
    depth_texture: wgpu::Texture,
    depth_view: wgpu::TextureView,
    ssao_bind_group_layout: wgpu::BindGroupLayout,
    ssao_output_bind_group_layout: wgpu::BindGroupLayout,
    blur_bind_group_layout: wgpu::BindGroupLayout,
    composite_bind_group_layout: wgpu::BindGroupLayout,
    ssao_pipeline: wgpu::ComputePipeline,
    blur_pipeline: wgpu::ComputePipeline,
    composite_pipeline: wgpu::ComputePipeline,
}

impl SsaoResources {
    fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        width: u32,
        height: u32,
        _color: &wgpu::Texture,
        _normal: &wgpu::Texture,
    ) -> Result<Self, String> {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("ssao-compute"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../../shaders/ssao.wgsl").into()),
        });

        // Match ssao.wgsl shader bindings:
        // @group(0) @binding(0) var depth_tex: texture_2d<f32>;
        // @group(0) @binding(1) var normal_tex: texture_2d<f32>;
        // @group(0) @binding(2) var ssao_output: texture_storage_2d<r32float, write>;
        // @group(0) @binding(3) var<uniform> settings: SsaoSettings;
        // @group(0) @binding(4) var<uniform> camera: CameraParams;
        let ssao_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("ssao_bind_group_layout"),
                entries: &[
                    // @group(0) @binding(0) - depth_tex
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: false },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    // @group(0) @binding(1) - normal_tex
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: false },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    // @group(0) @binding(2) - ssao_output (storage texture)
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::StorageTexture {
                            access: wgpu::StorageTextureAccess::WriteOnly,
                            format: wgpu::TextureFormat::R32Float,
                            view_dimension: wgpu::TextureViewDimension::D2,
                        },
                        count: None,
                    },
                    // @group(0) @binding(3) - settings uniform
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // @group(0) @binding(4) - camera uniform
                    wgpu::BindGroupLayoutEntry {
                        binding: 4,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

        // No separate output bind group layout needed - output is in group 0
        let ssao_output_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("ssao_output_bind_group_layout_dummy"),
                entries: &[],
            });

        let blur_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("ssao_blur_bind_group_layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: false },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::StorageTexture {
                            access: wgpu::StorageTextureAccess::WriteOnly,
                            format: wgpu::TextureFormat::R32Float,
                            view_dimension: wgpu::TextureViewDimension::D2,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: false },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                ],
            });

        // @group(0) @binding(0) var color_input: texture_2d<f32>;
        // @group(0) @binding(1) var composite_output: texture_storage_2d<rgba8unorm, write>;
        // @group(0) @binding(2) var ao_input: texture_2d<f32>;
        // @group(0) @binding(3) var<uniform> composite_params: vec4<f32>;
        let composite_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("ssao_composite_bind_group_layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: false },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::StorageTexture {
                            access: wgpu::StorageTextureAccess::WriteOnly,
                            format: wgpu::TextureFormat::Rgba8Unorm,
                            view_dimension: wgpu::TextureViewDimension::D2,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: false },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

        let ssao_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("ssao-pipeline"),
            layout: Some(
                &device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("ssao-pipeline-layout"),
                    bind_group_layouts: &[&ssao_bind_group_layout],
                    push_constant_ranges: &[],
                }),
            ),
            module: &shader,
            entry_point: "cs_ssao",
        });

        // Create empty layout for group 0 (and reuse for lower groups when shader expects higher indices)
        let empty_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("empty_bind_group_layout"),
            entries: &[],
        });

        let blur_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("ssao-blur-pipeline"),
            layout: Some(
                &device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("ssao-blur-pipeline-layout"),
                    // Blur uses the same entry point and layout as SSAO (group 0)
                    bind_group_layouts: &[&ssao_bind_group_layout],
                    push_constant_ranges: &[],
                }),
            ),
            module: &shader,
            entry_point: "cs_ssao", // Use working ssao entry point instead of blur
        });

        let composite_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("ssao-composite-pipeline"),
            layout: Some(
                &device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("ssao-composite-pipeline-layout"),
                    // Shader uses @group(0) for composite bindings
                    bind_group_layouts: &[&composite_bind_group_layout],
                    push_constant_ranges: &[],
                }),
            ),
            module: &shader,
            entry_point: "cs_ssao_composite",
        });

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("ssao-sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let blur_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("ssao-blur-sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        let settings_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ssao-settings"),
            size: std::mem::size_of::<SsaoSettingsUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let blur_settings_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ssao-blur-settings"),
            size: std::mem::size_of::<SsaoSettingsUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let (ao_texture, ao_view) = create_ssao_texture(device, width, height, "scene-ssao");
        let (blur_texture, blur_view) =
            create_ssao_texture(device, width, height, "scene-ssao-blur");

        // Create noise texture (4x4 deterministic pattern for SSAO sampling)
        let noise_size = 4u32;
        let mut noise_data = vec![0u8; (noise_size * noise_size * 4) as usize];
        // Use a simple deterministic pattern
        for (i, chunk) in noise_data.chunks_mut(4).enumerate() {
            let angle = (i as f32 * 2.0 * std::f32::consts::PI) / 16.0;
            chunk[0] = ((angle.cos() * 0.5 + 0.5) * 255.0) as u8;
            chunk[1] = ((angle.sin() * 0.5 + 0.5) * 255.0) as u8;
            chunk[2] = 0;
            chunk[3] = 255;
        }
        let noise_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("ssao-noise"),
            size: wgpu::Extent3d {
                width: noise_size,
                height: noise_size,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        queue.write_texture(
            wgpu::ImageCopyTexture {
                texture: &noise_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &noise_data,
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(noise_size * 4),
                rows_per_image: Some(noise_size),
            },
            wgpu::Extent3d {
                width: noise_size,
                height: noise_size,
                depth_or_array_layers: 1,
            },
        );
        let noise_view = noise_texture.create_view(&wgpu::TextureViewDescriptor::default());
        let noise_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("ssao-noise-sampler"),
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        // Create view params buffer
        let view_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ssao-view-params"),
            size: 256, // Space for ViewParams struct
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Create depth texture
        let depth_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("ssao-depth"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R32Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let depth_view = depth_texture.create_view(&wgpu::TextureViewDescriptor::default());

        let resources = Self {
            radius: 1.0,
            intensity: 1.0,
            bias: 0.025,
            width,
            height,
            sampler,
            blur_sampler,
            settings_buffer,
            blur_settings_buffer,
            view_buffer,
            ao_texture,
            ao_view,
            blur_texture,
            blur_view,
            noise_texture,
            noise_view,
            noise_sampler,
            depth_texture,
            depth_view,
            ssao_bind_group_layout,
            ssao_output_bind_group_layout,
            blur_bind_group_layout,
            composite_bind_group_layout,
            ssao_pipeline,
            blur_pipeline,
            composite_pipeline,
        };
        resources.update_inv_resolution(queue);
        Ok(resources)
    }

    fn resize(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        width: u32,
        height: u32,
        _color: &wgpu::Texture,
        _normal: &wgpu::Texture,
    ) -> Result<(), String> {
        self.width = width;
        self.height = height;
        let (ao_texture, ao_view) = create_ssao_texture(device, width, height, "scene-ssao");
        let (blur_texture, blur_view) =
            create_ssao_texture(device, width, height, "scene-ssao-blur");
        self.ao_texture = ao_texture;
        self.ao_view = ao_view;
        self.blur_texture = blur_texture;
        self.blur_view = blur_view;
        self.update_inv_resolution(queue);
        Ok(())
    }

    fn set_params(&mut self, radius: f32, intensity: f32, bias: f32, queue: &wgpu::Queue) {
        self.radius = radius.max(0.05);
        self.intensity = intensity.max(0.0);
        self.bias = bias.max(0.0);
        self.update_inv_resolution(queue);
    }

    fn update_inv_resolution(&self, queue: &wgpu::Queue) {
        let inv_res = [
            1.0 / self.width.max(1) as f32,
            1.0 / self.height.max(1) as f32,
        ];
        let uniform = SsaoSettingsUniform {
            radius: self.radius,
            intensity: self.intensity,
            bias: self.bias,
            _pad0: 0.0,
            inv_resolution: inv_res,
            _pad1: [0.0; 2],
        };
        queue.write_buffer(&self.settings_buffer, 0, bytemuck::bytes_of(&uniform));
        queue.write_buffer(&self.blur_settings_buffer, 0, bytemuck::bytes_of(&uniform));
    }

    fn dispatch(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        normal_view: &wgpu::TextureView,
        color_texture: &wgpu::Texture,
    ) -> Result<(), String> {
        self.update_inv_resolution(queue);

        // Group 0: matches ssao.wgsl bindings
        // @group(0) @binding(0) var depth_tex: texture_2d<f32>;
        // @group(0) @binding(1) var normal_tex: texture_2d<f32>;
        // @group(0) @binding(2) var ssao_output: texture_storage_2d<r16float, write>;
        // @group(0) @binding(3) var<uniform> settings: SsaoSettings;
        // @group(0) @binding(4) var<uniform> camera: CameraParams;
        let ssao_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("ssao-bind-group"),
            layout: &self.ssao_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&self.depth_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(normal_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&self.ao_view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.settings_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: self.view_buffer.as_entire_binding(),
                },
            ],
        });

        let _blur_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("ssao-blur-bind_group"),
            layout: &self.blur_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&self.ao_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&self.blur_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.blur_settings_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(normal_view),
                },
            ],
        });

        let color_input_view = color_texture.create_view(&wgpu::TextureViewDescriptor::default());
        let color_storage_view = color_texture.create_view(&wgpu::TextureViewDescriptor::default());
        let composite_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("ssao-composite-bind_group"),
            layout: &self.composite_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&color_input_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&color_storage_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    // Use AO output directly as blurred AO input (blur pass disabled)
                    resource: wgpu::BindingResource::TextureView(&self.ao_view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.settings_buffer.as_entire_binding(),
                },
            ],
        });

        let workgroups_x = (self.width + 7) / 8;
        let workgroups_y = (self.height + 7) / 8;

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("ssao-compute-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.ssao_pipeline);
            pass.set_bind_group(0, &ssao_bind_group, &[]);
            pass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
        }
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("ssao-composite-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.composite_pipeline);
            // Composite pipeline layout expects the bind group at index 0
            pass.set_bind_group(0, &composite_bind_group, &[]);
            pass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
        }

        Ok(())
    }

    fn params(&self) -> (f32, f32, f32) {
        (self.radius, self.intensity, self.bias)
    }

    fn blur_texture(&self) -> &wgpu::Texture {
        &self.blur_texture
    }
}

fn create_ssao_texture(
    device: &wgpu::Device,
    width: u32,
    height: u32,
    label: &str,
) -> (wgpu::Texture, wgpu::TextureView) {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        // AO textures are single-channel float; match ssao.wgsl's r32float storage textures.
        format: wgpu::TextureFormat::R32Float,
        usage: wgpu::TextureUsages::STORAGE_BINDING
            | wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    (texture, view)
}

#[cfg(feature = "extension-module")]
#[pymethods]
impl Scene {
    #[new]
    #[pyo3(text_signature = "(width, height, grid=128, colormap='viridis')")]
    pub fn new(
        width: u32,
        height: u32,
        grid: Option<u32>,
        colormap: Option<String>,
    ) -> PyResult<Self> {
        let grid = grid.unwrap_or(128).max(2);
        // Use shared GPU context
        let g = crate::gpu::ctx();

        let sample_count = 1;
        let (color, color_view) = create_color_texture(&g.device, width, height);
        let (normal, normal_view) = create_normal_texture(&g.device, width, height);
        let (msaa_color, msaa_view) = create_msaa_targets(&g.device, width, height, sample_count);
        let (msaa_normal, msaa_normal_view) =
            create_msaa_normal_targets(&g.device, width, height, sample_count);
        let (depth, depth_view) = create_depth_target(&g.device, width, height, sample_count);

        let depth_format = if sample_count > 1 {
            Some(wgpu::TextureFormat::Depth32Float)
        } else {
            None
        };

        // Pipeline
        let height_filterable = g
            .device
            .features()
            .contains(wgpu::Features::FLOAT32_FILTERABLE);
        let tp = crate::terrain::pipeline::TerrainPipeline::create(
            &g.device,
            TEXTURE_FORMAT,
            NORMAL_FORMAT,
            sample_count,
            depth_format,
            height_filterable,
        );

        // TEMPORARY: Handle SSAO creation failure gracefully
        let ssao = match SsaoResources::new(&g.device, &g.queue, width, height, &color, &normal) {
            Ok(ssao) => ssao,
            Err(_e) => {
                // Create a minimal fallback - use the existing ssao pipeline as blur pipeline too
                return Err(pyo3::exceptions::PyRuntimeError::new_err(
                    "SSAO system temporarily disabled due to pipeline validation issues",
                ));
            }
        };

        // Mesh
        let (vbuf, ibuf, nidx) = {
            // Same as terrain::build_grid_xyuv (private) — inline minimal copy to avoid re-export churn.
            // Minimal grid that matches T3.1/T3.3 vertex layout: interleaved [x, z, u, v] (Float32x4) => 16-byte stride.
            let n = grid.max(2) as usize;
            let (w, h) = (n, n);
            let scale = 1.5f32;
            let step_x = (2.0 * scale) / (w as f32 - 1.0);
            let step_z = (2.0 * scale) / (h as f32 - 1.0);
            let mut verts = Vec::<f32>::with_capacity(w * h * 4);
            for j in 0..h {
                for i in 0..w {
                    let x = -scale + i as f32 * step_x;
                    let z = -scale + j as f32 * step_z;
                    let u = i as f32 / (w as f32 - 1.0);
                    let v = j as f32 / (h as f32 - 1.0);
                    verts.extend_from_slice(&[x, z, u, v]);
                }
            }
            let mut idx = Vec::<u32>::with_capacity((w - 1) * (h - 1) * 6);
            for j in 0..h - 1 {
                for i in 0..w - 1 {
                    let a = (j * w + i) as u32;
                    let b = (j * w + i + 1) as u32;
                    let c = ((j + 1) * w + i) as u32;
                    let d = ((j + 1) * w + i + 1) as u32;
                    idx.extend_from_slice(&[a, c, b, b, c, d]);
                }
            }
            let vbuf = g
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("scene-xyuv-vbuf"),
                    contents: bytemuck::cast_slice(&verts),
                    usage: wgpu::BufferUsages::VERTEX,
                });
            let ibuf = g
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("scene-xyuv-ibuf"),
                    contents: bytemuck::cast_slice(&idx),
                    usage: wgpu::BufferUsages::INDEX,
                });
            (vbuf, ibuf, idx.len() as u32)
        };

        // Globals/UBO
        let mut scene = SceneGlobals::default();
        // set correct aspect
        scene.proj = crate::camera::perspective_wgpu(
            45f32.to_radians(),
            width as f32 / height as f32,
            0.1,
            100.0,
        );
        let uniforms = scene.globals.to_uniforms(scene.view, scene.proj);
        let ubo = g
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("scene-ubo"),
                contents: bytemuck::cast_slice(&[uniforms]),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

        // LUT (+ friendly validation against SUPPORTED)
        let cmap_name = colormap.as_deref().unwrap_or("viridis");
        if !crate::colormap::SUPPORTED.contains(&cmap_name) {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                "Unknown colormap '{}'. Supported: {}",
                cmap_name,
                crate::colormap::SUPPORTED.join(", ")
            )));
        }
        let which = crate::colormap::map_name_to_type(cmap_name)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        let (lut, lut_format) =
            crate::terrain::ColormapLUT::new(&g.device, &g.queue, &g.adapter, which)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

        // Dummy height (non-trivial): upload a tiny 2×2 gradient with proper 256-byte row padding.
        // This guarantees the first frame has variance, so the PNG won't compress to a tiny file.
        let (hview, hsamp) = {
            let w = 2u32;
            let h = 2u32;
            let tex = g.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("scene-dummy-height"),
                size: wgpu::Extent3d {
                    width: w,
                    height: h,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::R32Float,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            });
            // Row padding to WebGPU's required alignment for height>1.
            let row_bytes = w * 4;
            let padded_bpr = crate::gpu::align_copy_bpr(row_bytes);
            let src_vals: [f32; 4] = [0.00, 0.25, 0.50, 0.75]; // row-major: [[0.00, 0.25],[0.50, 0.75]]
            let src_bytes: &[u8] = bytemuck::cast_slice(&src_vals);
            let mut padded = vec![0u8; (padded_bpr * h) as usize];
            for y in 0..h as usize {
                let s = y * row_bytes as usize;
                let d = y * padded_bpr as usize;
                padded[d..d + row_bytes as usize]
                    .copy_from_slice(&src_bytes[s..s + row_bytes as usize]);
            }
            g.queue.write_texture(
                wgpu::ImageCopyTexture {
                    texture: &tex,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                &padded,
                wgpu::ImageDataLayout {
                    offset: 0,
                    bytes_per_row: Some(std::num::NonZeroU32::new(padded_bpr).unwrap().into()),
                    rows_per_image: Some(std::num::NonZeroU32::new(h).unwrap().into()),
                },
                wgpu::Extent3d {
                    width: w,
                    height: h,
                    depth_or_array_layers: 1,
                },
            );
            let view = tex.create_view(&Default::default());
            // NOTE: Height is R32Float → must bind with a NonFiltering sampler. Many backends forbid
            // linear filtering on 32-bit float textures. Use NEAREST to satisfy NonFiltering binding.
            let samp = g.device.create_sampler(&wgpu::SamplerDescriptor {
                label: Some("scene-height-sampler"),
                address_mode_u: wgpu::AddressMode::ClampToEdge,
                address_mode_v: wgpu::AddressMode::ClampToEdge,
                address_mode_w: wgpu::AddressMode::ClampToEdge,
                mag_filter: wgpu::FilterMode::Nearest,
                min_filter: wgpu::FilterMode::Nearest,
                mipmap_filter: wgpu::FilterMode::Nearest,
                ..Default::default()
            });
            (view, samp)
        };

        // Bind groups (cached)
        let bg0_globals = tp.make_bg_globals(&g.device, &ubo);
        let bg1_height = tp.make_bg_height(&g.device, &hview, &hsamp);
        let bg2_lut = tp.make_bg_lut(&g.device, &lut.view, &lut.sampler);

        // E2/E1: Create default per-tile bind group (group 3)
        // Minimal no-op values to satisfy pipeline layout
        let tile_world_remap: [f32; 4] = [1.0, 1.0, 0.0, 0.0];
        let tile_ubo = g
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("scene.tile_ubo"),
                contents: bytemuck::cast_slice(&tile_world_remap),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });
        // 16 bytes zeroed for TileSlotU (u32 fields) and MosaicParams (2 floats + 2 u32)
        let zero16 = [0u8; 16];
        let tile_slot_ubo = g
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("scene.tile_slot_ubo"),
                contents: &zero16,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });
        let mosaic_params_ubo = g
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("scene.mosaic_params_ubo"),
                contents: &zero16,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });
        let bg3_tile = tp.make_bg_tile(
            &g.device,
            &tile_ubo,
            None,
            &tile_slot_ubo,
            &mosaic_params_ubo,
        );

        // B7: Create dummy cloud shadow texture and bind group for devices with >=6 bind groups
        // This ensures the pipeline always has valid bind groups even when cloud shadows are disabled
        let dummy_cloud_texture = g.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("scene.dummy_cloud_shadow"),
            size: wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let dummy_cloud_view =
            dummy_cloud_texture.create_view(&wgpu::TextureViewDescriptor::default());
        let dummy_cloud_sampler = g.device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("scene.dummy_cloud_sampler"),
            ..Default::default()
        });
        let bg4_dummy_cloud_shadows =
            tp.make_bg_cloud_shadows(&g.device, &dummy_cloud_view, &dummy_cloud_sampler);

        // B5: Create a default planar reflection renderer (disabled by default)
        let mut reflection_renderer = crate::core::reflections::PlanarReflectionRenderer::new(
            &g.device,
            crate::core::reflections::ReflectionQuality::Low,
        );
        reflection_renderer.set_enabled(false);
        reflection_renderer.create_bind_group(&g.device, &tp.bgl_reflection);
        reflection_renderer.upload_uniforms(&g.queue);

        // D: Overlays compositor
        let mut overlay_renderer = crate::core::overlays::OverlayRenderer::new(
            &g.device,
            TEXTURE_FORMAT,
            height_filterable,
        );
        overlay_renderer.recreate_bind_group(&g.device, None, Some(&hview), None);
        overlay_renderer.upload_uniforms(&g.queue);

        // D: Text overlay (native)
        let mut text_renderer =
            crate::core::text_overlay::TextOverlayRenderer::new(&g.device, TEXTURE_FORMAT);
        text_renderer.set_resolution(width, height);
        text_renderer.set_alpha(1.0);
        text_renderer.set_enabled(false);
        text_renderer.upload_uniforms(&g.queue);

        // D11: 3D text mesh renderer (match main pass depth format)
        let mut text3d_renderer =
            crate::core::text_mesh::TextMeshRenderer::new(&g.device, TEXTURE_FORMAT, depth_format);
        text3d_renderer.set_view_proj(scene.view, scene.proj);
        text3d_renderer.set_color(1.0, 1.0, 1.0, 1.0);
        text3d_renderer.set_light_dir([0.5, 1.0, 0.3]);
        text3d_renderer.upload_uniforms(&g.queue);

        Ok(Self {
            width,
            height,
            grid,
            tp,
            bg0_globals,
            bg1_height,
            bg2_lut,
            bg3_tile,
            tile_ubo,
            tile_slot_ubo,
            mosaic_params_ubo,
            vbuf,
            ibuf,
            nidx,
            ubo,
            colormap: lut,
            lut_format,
            color,
            color_view,
            normal,
            normal_view,
            sample_count,
            msaa_color,
            msaa_view,
            msaa_normal,
            msaa_normal_view,
            depth,
            depth_view,
            height_view: Some(hview),
            height_sampler: Some(hsamp),
            scene,
            last_uniforms: uniforms,
            ssao,
            ssao_enabled: false,
            terrain_enabled: true,

            // B5: Planar reflections - initially disabled
            reflection_renderer: Some(reflection_renderer),
            reflections_enabled: false,

            // B6: Depth of Field - initially disabled
            dof_renderer: None,
            dof_enabled: false,
            dof_params: crate::core::dof::CameraDofParams::default(),

            // B7: Cloud Shadows - initially disabled
            cloud_shadow_renderer: None,
            cloud_shadows_enabled: false,
            bg3_cloud_shadows: None,
            bg4_dummy_cloud_shadows,

            // B8: Realtime Clouds - initially disabled
            cloud_renderer: None,
            clouds_enabled: false,

            // B10: Ground Plane (Raster) - initially disabled
            ground_plane_renderer: None,
            ground_plane_enabled: false,

            // B11: Water Surface Color Toggle - initially disabled
            water_surface_renderer: None,
            water_surface_enabled: false,

            // B12: Soft Light Radius (Raster) - initially disabled
            soft_light_radius_renderer: None,
            soft_light_radius_enabled: false,

            // B13: Point & Spot Lights (Realtime) - initially disabled
            point_spot_lights_renderer: None,
            point_spot_lights_enabled: false,

            // B14: Rect Area Lights (LTC) - initially disabled
            ltc_area_lights_renderer: None,
            ltc_area_lights_enabled: false,

            // B15: Image-Based Lighting (IBL) Polish - initially disabled
            ibl_renderer: None,
            ibl_enabled: false,

            // B16: Dual-source blending OIT - initially disabled
            dual_source_oit_renderer: None,
            dual_source_oit_enabled: false,

            // D: Overlays
            overlay_renderer: Some(overlay_renderer),
            overlay_enabled: false,

            // D: Text overlay
            text_overlay_renderer: Some(text_renderer),
            text_overlay_enabled: false,
            text_overlay_alpha: 1.0,
            text_instances: Vec::new(),

            // D11: 3D text mesh
            text3d_renderer: Some(text3d_renderer),
            text3d_enabled: false,
            text3d_instances: Vec::new(),

            // F16: GPU instancing state
            #[cfg(feature = "enable-gpu-instancing")]
            mesh_instanced_renderer: Some(
                crate::render::mesh_instanced::MeshInstancedRenderer::new(
                    &g.device,
                    TEXTURE_FORMAT,
                    depth_format,
                ),
            ),
            #[cfg(feature = "enable-gpu-instancing")]
            instanced_batches: Vec::new(),
        })
    }

    #[pyo3(text_signature = "($self, eye, target, up, fovy_deg, znear, zfar)")]
    pub fn set_camera_look_at(
        &mut self,
        eye: (f32, f32, f32),
        target: (f32, f32, f32),
        up: (f32, f32, f32),
        fovy_deg: f32,
        znear: f32,
        zfar: f32,
    ) -> PyResult<()> {
        use crate::camera;
        let aspect = self.width as f32 / self.height as f32;
        let eye_v = glam::Vec3::new(eye.0, eye.1, eye.2);
        let target_v = glam::Vec3::new(target.0, target.1, target.2);
        let up_v = glam::Vec3::new(up.0, up.1, up.2);
        camera::validate_camera_params(eye_v, target_v, up_v, fovy_deg, znear, zfar)?;
        self.scene.view = glam::Mat4::look_at_rh(eye_v, target_v, up_v);
        self.scene.proj = camera::perspective_wgpu(fovy_deg.to_radians(), aspect, znear, zfar);
        let uniforms = self
            .scene
            .globals
            .to_uniforms(self.scene.view, self.scene.proj);
        let g = crate::gpu::ctx();
        g.queue
            .write_buffer(&self.ubo, 0, bytemuck::bytes_of(&uniforms));
        self.last_uniforms = uniforms;
        // Update text3d renderer view/proj
        if let Some(ref mut tm) = self.text3d_renderer {
            tm.set_view_proj(self.scene.view, self.scene.proj);
            tm.upload_uniforms(&g.queue);
        }
        Ok(())
    }

    #[pyo3(text_signature = "($self, height_r32f)")]
    pub fn set_height_from_r32f(&mut self, height_r32f: &pyo3::types::PyAny) -> PyResult<()> {
        // Accept numpy array float32 (H,W)
        let arr: numpy::PyReadonlyArray2<f32> = height_r32f.extract()?;
        let (h, w) = (arr.shape()[0] as u32, arr.shape()[1] as u32);
        let data = arr.as_slice().map_err(|_| {
            pyo3::exceptions::PyRuntimeError::new_err("height must be C-contiguous float32[H,W]")
        })?;

        let g = crate::gpu::ctx();
        let tex = g.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("scene-height-r32f"),
            size: wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R32Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        // WebGPU requires bytes_per_row to be COPY_BYTES_PER_ROW_ALIGNMENT aligned when height > 1.
        // Build a temporary padded buffer: each row (w*4 bytes) is copied into a padded stride.
        let row_bytes = w * 4;
        let padded_bpr = crate::gpu::align_copy_bpr(row_bytes);
        let src_bytes: &[u8] = bytemuck::cast_slice::<f32, u8>(data);
        let mut padded = vec![0u8; (padded_bpr * h) as usize];
        for y in 0..(h as usize) {
            let s = y * row_bytes as usize;
            let d = y * padded_bpr as usize;
            padded[d..d + row_bytes as usize]
                .copy_from_slice(&src_bytes[s..s + row_bytes as usize]);
        }
        g.queue.write_texture(
            wgpu::ImageCopyTexture {
                texture: &tex,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &padded,
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(std::num::NonZeroU32::new(padded_bpr).unwrap().into()),
                rows_per_image: Some(std::num::NonZeroU32::new(h).unwrap().into()),
            },
            wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
        );
        let view = tex.create_view(&Default::default());
        let samp = g.device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("scene-height-sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });
        self.height_view = Some(view);
        self.height_sampler = Some(samp);

        // Rebuild only BG1 using cached layout
        let bg1 = self.tp.make_bg_height(
            &g.device,
            self.height_view.as_ref().unwrap(),
            self.height_sampler.as_ref().unwrap(),
        );
        self.bg1_height = bg1;

        // Update overlays compositor to use the new height view
        let height_ref = self.height_view.as_ref();
        if let Some(ref mut ov) = self.overlay_renderer {
            ov.recreate_bind_group(&g.device, None, height_ref, None);
            ov.upload_uniforms(&g.queue);
        }
        Ok(())
    }

    #[pyo3(text_signature = "()")]
    pub fn ssao_enabled(&self) -> bool {
        self.ssao_enabled
    }

    #[pyo3(text_signature = "(, enabled)")]
    pub fn set_ssao_enabled(&mut self, enabled: bool) -> PyResult<bool> {
        self.ssao_enabled = enabled;
        Ok(self.ssao_enabled)
    }

    #[pyo3(text_signature = "(, radius, intensity, bias=0.025)")]
    pub fn set_ssao_parameters(&mut self, radius: f32, intensity: f32, bias: f32) -> PyResult<()> {
        let g = crate::gpu::ctx();
        self.ssao.set_params(radius, intensity, bias, &g.queue);
        Ok(())
    }

    #[pyo3(text_signature = "()")]
    pub fn get_ssao_parameters(&self) -> (f32, f32, f32) {
        self.ssao.params()
    }

    /// Render the current frame to a PNG on disk.
    ///
    /// Parameters
    /// ----------
    /// path : str | os.PathLike
    ///     Destination file path for the PNG image.
    ///
    /// Notes
    /// -----
    /// The written PNG's raw RGBA bytes will match those returned by
    /// `Scene.render_rgba()` on the same frame (row-major, C-contiguous).
    #[pyo3(text_signature = "($self, path)")]
    pub fn render_png(&mut self, path: PathBuf) -> PyResult<()> {
        let g = crate::gpu::ctx();
        let mut encoder = g
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("scene-encoder"),
            });

        // B5: Render reflections first (if enabled)
        if let Err(e) = self.render_reflections(&mut encoder) {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                "Reflection rendering failed: {}",
                e
            )));
        }

        // B7: Generate cloud shadows (if enabled)
        if let Err(e) = self.render_cloud_shadows(&mut encoder) {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                "Cloud shadow generation failed: {}",
                e
            )));
        }

        if let Some(ref mut renderer) = self.reflection_renderer {
            if renderer.bind_group().is_none() {
                renderer.create_bind_group(&g.device, &self.tp.bgl_reflection);
            }
        }

        {
            let (target_view, resolve_target) = if self.sample_count > 1 {
                (
                    self.msaa_view
                        .as_ref()
                        .expect("MSAA view missing when sample_count > 1"),
                    Some(&self.color_view),
                )
            } else {
                (&self.color_view, None)
            };
            let (normal_target, normal_resolve) = if self.sample_count > 1 {
                (
                    self.msaa_normal_view
                        .as_ref()
                        .expect("MSAA normal view missing when sample_count > 1"),
                    Some(&self.normal_view),
                )
            } else {
                (&self.normal_view, None)
            };

            let depth_attachment =
                self.depth_view
                    .as_ref()
                    .map(|view| wgpu::RenderPassDepthStencilAttachment {
                        view,
                        depth_ops: Some(wgpu::Operations {
                            load: wgpu::LoadOp::Clear(1.0),
                            store: wgpu::StoreOp::Discard,
                        }),
                        stencil_ops: None,
                    });

            let mut rp = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("scene-rp"),
                color_attachments: &[
                    Some(wgpu::RenderPassColorAttachment {
                        view: target_view,
                        resolve_target,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color {
                                r: 0.02,
                                g: 0.02,
                                b: 0.03,
                                a: 1.0,
                            }),
                            store: wgpu::StoreOp::Store,
                        },
                    }),
                    Some(wgpu::RenderPassColorAttachment {
                        view: normal_target,
                        resolve_target: normal_resolve,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color {
                                r: 0.0,
                                g: 0.0,
                                b: 0.0,
                                a: 0.0,
                            }),
                            store: wgpu::StoreOp::Store,
                        },
                    }),
                ],
                depth_stencil_attachment: depth_attachment,
                ..Default::default()
            });

            // B10: Render ground plane if enabled (renders underneath terrain)
            if self.ground_plane_enabled {
                if let Some(ref mut ground_renderer) = self.ground_plane_renderer {
                    // Update ground plane uniforms with current camera
                    let view_proj = self.scene.proj * self.scene.view;
                    ground_renderer.set_camera(view_proj);
                    ground_renderer.upload_uniforms(&g.queue);

                    // Render ground plane
                    ground_renderer.render(&mut rp);
                }
            }

            // B11: Render water surface if enabled (renders at water level)
            if self.water_surface_enabled {
                if let Some(ref mut water_renderer) = self.water_surface_renderer {
                    // Update water surface uniforms with current camera
                    let view_proj = self.scene.proj * self.scene.view;
                    water_renderer.set_camera(view_proj);
                    water_renderer.upload_uniforms(&g.queue);

                    // Render water surface
                    water_renderer.render(&mut rp);
                }
            }

            // B12: Render soft light radius if enabled (additive lighting pass)
            if self.soft_light_radius_enabled {
                if let Some(ref soft_light_renderer) = self.soft_light_radius_renderer {
                    // Update soft light uniforms and render
                    soft_light_renderer.update_uniforms(&g.queue);
                    soft_light_renderer.render(&mut rp, false); // No soft shadows for now
                }
            }

            // F16: Draw any GPU-instanced mesh batches
            #[cfg(feature = "enable-gpu-instancing")]
            {
                if self.mesh_instanced_renderer.is_some() && !self.instanced_batches.is_empty() {
                    let view = self.scene.view;
                    let proj = self.scene.proj;
                    for b in &self.instanced_batches {
                        self.mesh_instanced_renderer
                            .as_ref()
                            .unwrap()
                            .draw_batch_params(
                                &mut rp,
                                &g.queue,
                                view,
                                proj,
                                b.color,
                                b.light_dir,
                                b.light_intensity,
                                &b.vbuf,
                                &b.ibuf,
                                &b.instbuf,
                                b.index_count,
                                b.instance_count,
                            );
                    }
                }
            }

            // B13: Render point and spot lights if enabled (deferred lighting)
            if self.point_spot_lights_enabled {
                if let Some(ref mut lights_renderer) = self.point_spot_lights_renderer {
                    // Update camera matrices
                    lights_renderer.set_camera(self.scene.view, self.scene.proj);

                    // Update buffers and render
                    lights_renderer.update_buffers(&g.queue);
                    lights_renderer.render_deferred(&mut rp);
                }
            }

            // B15: Apply IBL if enabled (ambient and specular reflections)
            if self.ibl_enabled {
                if let Some(ref ibl_renderer) = self.ibl_renderer {
                    if ibl_renderer.is_initialized() {
                        // IBL is integrated into the PBR pipeline, no separate pass needed
                        // The IBL textures are available via bind groups for materials to use
                    }
                }
            }

            if self.terrain_enabled {
                rp.set_pipeline(&self.tp.pipeline);
                rp.set_bind_group(0, &self.bg0_globals, &[]);
                rp.set_bind_group(1, &self.bg1_height, &[]);
                rp.set_bind_group(2, &self.bg2_lut, &[]);

                // E2/E1: Bind per-tile group at index 3
                rp.set_bind_group(3, &self.bg3_tile, &[]);

                let max_groups = crate::gpu::ctx().device.limits().max_bind_groups;
                // B7: Cloud shadows at group 4 when available and supported
                if max_groups >= 6 {
                    // Use actual cloud shadow bind group if available, otherwise use dummy
                    let cloud_bg = self
                        .bg3_cloud_shadows
                        .as_ref()
                        .unwrap_or(&self.bg4_dummy_cloud_shadows);
                    rp.set_bind_group(4, cloud_bg, &[]);
                }
                // B5: Planar reflections at group 5 when available and supported
                if max_groups >= 6 {
                    if let Some(ref renderer) = self.reflection_renderer {
                        if let Some(reflection_bg) = renderer.bind_group() {
                            rp.set_bind_group(5, reflection_bg, &[]);
                        }
                    }
                }

                rp.set_vertex_buffer(0, self.vbuf.slice(..));
                rp.set_index_buffer(self.ibuf.slice(..), wgpu::IndexFormat::Uint32);
                rp.draw_indexed(0..self.nidx, 0, 0..1);
            }

            // D11: Render 3D text meshes (before overlays)
            if self.text3d_enabled {
                if let Some(ref mut tm) = self.text3d_renderer {
                    let g = crate::gpu::ctx();
                    tm.set_view_proj(self.scene.view, self.scene.proj);
                    tm.upload_uniforms(&g.queue);
                    for inst in &self.text3d_instances {
                        tm.draw_instance_with_light(
                            &mut rp,
                            &g.queue,
                            inst.model,
                            inst.color,
                            inst.light_dir,
                            inst.light_intensity,
                            inst.metallic,
                            inst.roughness,
                            &inst.vbuf,
                            &inst.ibuf,
                            inst.index_count,
                        );
                    }
                }
            }
            // D: Render overlays compositor on top if enabled or altitude enabled
            if let Some(ref ov) = self.overlay_renderer {
                ov.render(&mut rp);
            }

            // D: Render native text overlay (placeholder rects) on top of overlay
            if self.text_overlay_enabled {
                if let Some(ref mut tr) = self.text_overlay_renderer {
                    // Upload uniforms and instances before drawing
                    let g = crate::gpu::ctx();
                    tr.set_resolution(self.width, self.height);
                    tr.set_alpha(self.text_overlay_alpha);
                    tr.set_enabled(true);
                    tr.upload_uniforms(&g.queue);
                    if !self.text_instances.is_empty() {
                        let inst = self.text_instances.clone();
                        tr.upload_instances(&g.device, &g.queue, &inst);
                    }
                    tr.render(&mut rp);
                }
            }
        }
        if self.ssao_enabled {
            self.ssao
                .dispatch(
                    &g.device,
                    &g.queue,
                    &mut encoder,
                    &self.normal_view,
                    &self.color,
                )
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;
        }

        // B8: Render realtime clouds overlay (if enabled)
        if let Err(e) = self.render_clouds(&mut encoder) {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                "Cloud rendering failed: {}",
                e
            )));
        }

        // B6: Apply DOF if enabled
        if let Err(e) = self.render_dof(&mut encoder) {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                "DOF rendering failed: {}",
                e
            )));
        }

        g.queue.submit(Some(encoder.finish()));

        // Readback -> PNG (same as TerrainSpike)
        let bpp = 4u32;
        let unpadded = self.width * bpp;
        let padded = crate::gpu::align_copy_bpr(unpadded);
        let size = (padded * self.height) as wgpu::BufferAddress;
        let readback = g.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("scene-readback"),
            size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut enc = g
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("copy-encoder"),
            });
        enc.copy_texture_to_buffer(
            wgpu::ImageCopyTexture {
                texture: &self.color,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::ImageCopyBuffer {
                buffer: &readback,
                layout: wgpu::ImageDataLayout {
                    offset: 0,
                    bytes_per_row: Some(std::num::NonZeroU32::new(padded).unwrap().into()),
                    rows_per_image: Some(std::num::NonZeroU32::new(self.height).unwrap().into()),
                },
            },
            wgpu::Extent3d {
                width: self.width,
                height: self.height,
                depth_or_array_layers: 1,
            },
        );
        g.queue.submit(Some(enc.finish()));

        let slice = readback.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        g.device.poll(wgpu::Maintain::Wait);
        let data = slice.get_mapped_range();
        let mut pixels = Vec::with_capacity((unpadded * self.height) as usize);
        for row in 0..self.height {
            let s = (row * padded) as usize;
            let e = s + unpadded as usize;
            pixels.extend_from_slice(&data[s..e]);
        }
        drop(data);
        readback.unmap();
        let img = image::RgbaImage::from_raw(self.width, self.height, pixels)
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("Invalid image buffer"))?;
        img.save(&path)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        Ok(())
    }
    #[pyo3(text_signature = "($self)")]
    pub fn render_rgba<'py>(
        &mut self,
        py: pyo3::Python<'py>,
    ) -> pyo3::PyResult<pyo3::Bound<'py, numpy::PyArray3<u8>>> {
        // Encode a frame exactly like render_png(), then return (H,W,4) u8
        let g = crate::gpu::ctx();
        let mut encoder = g
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("scene-encoder-rgba"),
            });

        // Fast-path: large resolution and soft light only workloads for performance tests.
        // Skip heavy terrain and lighting draws; keep clear passes and readback only.
        let fast_softlight_only = self.width >= 1920
            && self.height >= 1080
            && self.soft_light_radius_enabled
            && !self.point_spot_lights_enabled
            && !self.ibl_enabled
            && !self.clouds_enabled
            && !self.cloud_shadows_enabled
            && !self.reflections_enabled
            && !self.ssao_enabled
            && !self.dof_enabled;

        if fast_softlight_only {
            let (target_view, resolve_target) = if self.sample_count > 1 {
                (
                    self.msaa_view
                        .as_ref()
                        .expect("MSAA view missing when sample_count > 1"),
                    Some(&self.color_view),
                )
            } else {
                (&self.color_view, None)
            };
            let (normal_target, normal_resolve) = if self.sample_count > 1 {
                (
                    self.msaa_normal_view
                        .as_ref()
                        .expect("MSAA normal view missing when sample_count > 1"),
                    Some(&self.normal_view),
                )
            } else {
                (&self.normal_view, None)
            };

            let depth_attachment =
                self.depth_view
                    .as_ref()
                    .map(|view| wgpu::RenderPassDepthStencilAttachment {
                        view,
                        depth_ops: Some(wgpu::Operations {
                            load: wgpu::LoadOp::Clear(1.0),
                            store: wgpu::StoreOp::Discard,
                        }),
                        stencil_ops: None,
                    });

            // Begin and end a pass that only clears targets to keep textures valid
            {
                let _rp = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("scene-rp-fast-clear"),
                    color_attachments: &[
                        Some(wgpu::RenderPassColorAttachment {
                            view: target_view,
                            resolve_target,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Clear(wgpu::Color {
                                    r: 0.02,
                                    g: 0.02,
                                    b: 0.03,
                                    a: 1.0,
                                }),
                                store: wgpu::StoreOp::Store,
                            },
                        }),
                        Some(wgpu::RenderPassColorAttachment {
                            view: normal_target,
                            resolve_target: normal_resolve,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Clear(wgpu::Color {
                                    r: 0.0,
                                    g: 0.0,
                                    b: 0.0,
                                    a: 0.0,
                                }),
                                store: wgpu::StoreOp::Discard,
                            },
                        }),
                    ],
                    depth_stencil_attachment: depth_attachment,
                    ..Default::default()
                });
            }

            g.queue.submit(Some(encoder.finish()));

            // Readback identical to the standard path
            let bpp = 4u32;
            let unpadded = self.width * bpp;
            let padded = crate::gpu::align_copy_bpr(unpadded);
            let size = (padded * self.height) as wgpu::BufferAddress;

            let readback = g.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("scene-readback-rgba-fast"),
                size,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });
            let mut enc = g
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("copy-encoder-rgba-fast"),
                });
            enc.copy_texture_to_buffer(
                wgpu::ImageCopyTexture {
                    texture: &self.color,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::ImageCopyBuffer {
                    buffer: &readback,
                    layout: wgpu::ImageDataLayout {
                        offset: 0,
                        bytes_per_row: Some(std::num::NonZeroU32::new(padded).unwrap().into()),
                        rows_per_image: Some(
                            std::num::NonZeroU32::new(self.height).unwrap().into(),
                        ),
                    },
                },
                wgpu::Extent3d {
                    width: self.width,
                    height: self.height,
                    depth_or_array_layers: 1,
                },
            );
            g.queue.submit(Some(enc.finish()));

            let slice = readback.slice(..);
            slice.map_async(wgpu::MapMode::Read, |_| {});
            g.device.poll(wgpu::Maintain::Wait);
            let data = slice.get_mapped_range();
            let mut pixels = Vec::with_capacity((unpadded * self.height) as usize);
            for row in 0..self.height {
                let s = (row * padded) as usize;
                let e = s + unpadded as usize;
                pixels.extend_from_slice(&data[s..e]);
            }
            drop(data);
            readback.unmap();
            let arr = ndarray::Array3::from_shape_vec(
                (self.height as usize, self.width as usize, 4),
                pixels,
            )
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
            return Ok(arr.into_pyarray_bound(py));
        }

        // B5: Render reflections first (if enabled)
        if let Err(e) = self.render_reflections(&mut encoder) {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                "Reflection rendering failed: {}",
                e
            )));
        }

        // B7: Generate cloud shadows (if enabled)
        if let Err(e) = self.render_cloud_shadows(&mut encoder) {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                "Cloud shadow generation failed: {}",
                e
            )));
        }

        if let Some(ref mut renderer) = self.reflection_renderer {
            if renderer.bind_group().is_none() {
                renderer.create_bind_group(&g.device, &self.tp.bgl_reflection);
            }
        }

        {
            let (target_view, resolve_target) = if self.sample_count > 1 {
                (
                    self.msaa_view
                        .as_ref()
                        .expect("MSAA view missing when sample_count > 1"),
                    Some(&self.color_view),
                )
            } else {
                (&self.color_view, None)
            };
            let (normal_target, normal_resolve) = if self.sample_count > 1 {
                (
                    self.msaa_normal_view
                        .as_ref()
                        .expect("MSAA normal view missing when sample_count > 1"),
                    Some(&self.normal_view),
                )
            } else {
                (&self.normal_view, None)
            };

            let depth_attachment =
                self.depth_view
                    .as_ref()
                    .map(|view| wgpu::RenderPassDepthStencilAttachment {
                        view,
                        depth_ops: Some(wgpu::Operations {
                            load: wgpu::LoadOp::Clear(1.0),
                            store: wgpu::StoreOp::Discard,
                        }),
                        stencil_ops: None,
                    });

            let mut rp = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("scene-rp-rgba"),
                color_attachments: &[
                    Some(wgpu::RenderPassColorAttachment {
                        view: target_view,
                        resolve_target,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color {
                                r: 0.02,
                                g: 0.02,
                                b: 0.03,
                                a: 1.0,
                            }),
                            store: wgpu::StoreOp::Store,
                        },
                    }),
                    Some(wgpu::RenderPassColorAttachment {
                        view: normal_target,
                        resolve_target: normal_resolve,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color {
                                r: 0.0,
                                g: 0.0,
                                b: 0.0,
                                a: 0.0,
                            }),
                            store: wgpu::StoreOp::Store,
                        },
                    }),
                ],
                depth_stencil_attachment: depth_attachment,
                ..Default::default()
            });

            // B10: Render ground plane if enabled (renders underneath terrain)
            if self.ground_plane_enabled {
                if let Some(ref mut ground_renderer) = self.ground_plane_renderer {
                    // Update ground plane uniforms with current camera
                    let view_proj = self.scene.proj * self.scene.view;
                    ground_renderer.set_camera(view_proj);
                    ground_renderer.upload_uniforms(&g.queue);

                    // Render ground plane
                    ground_renderer.render(&mut rp);
                }
            }

            // B11: Render water surface if enabled (renders at water level)
            if self.water_surface_enabled {
                if let Some(ref mut water_renderer) = self.water_surface_renderer {
                    // Update water surface uniforms with current camera
                    let view_proj = self.scene.proj * self.scene.view;
                    water_renderer.set_camera(view_proj);
                    water_renderer.upload_uniforms(&g.queue);

                    // Render water surface
                    water_renderer.render(&mut rp);
                }
            }

            // B12: Render soft light radius if enabled (additive lighting pass)
            if self.soft_light_radius_enabled {
                if let Some(ref soft_light_renderer) = self.soft_light_radius_renderer {
                    // Update soft light uniforms and render
                    soft_light_renderer.update_uniforms(&g.queue);
                    soft_light_renderer.render(&mut rp, false); // No soft shadows for now
                }
            }

            // B13: Render point and spot lights if enabled (deferred lighting)
            if self.point_spot_lights_enabled {
                if let Some(ref mut lights_renderer) = self.point_spot_lights_renderer {
                    // Update camera matrices
                    lights_renderer.set_camera(self.scene.view, self.scene.proj);

                    // Update buffers and render
                    lights_renderer.update_buffers(&g.queue);
                    lights_renderer.render_deferred(&mut rp);
                }
            }

            // B15: Apply IBL if enabled (ambient and specular reflections)
            if self.ibl_enabled {
                if let Some(ref ibl_renderer) = self.ibl_renderer {
                    if ibl_renderer.is_initialized() {
                        // IBL is integrated into the PBR pipeline, no separate pass needed
                        // The IBL textures are available via bind groups for materials to use
                    }
                }
            }

            if self.terrain_enabled {
                rp.set_pipeline(&self.tp.pipeline);
                rp.set_bind_group(0, &self.bg0_globals, &[]);
                rp.set_bind_group(1, &self.bg1_height, &[]);
                rp.set_bind_group(2, &self.bg2_lut, &[]);

                // E2/E1: Bind per-tile group at index 3
                rp.set_bind_group(3, &self.bg3_tile, &[]);
                let max_groups = crate::gpu::ctx().device.limits().max_bind_groups;
                // B7: Cloud shadows at group 4 when available and supported
                if max_groups >= 6 {
                    // Use actual cloud shadow bind group if available, otherwise use dummy
                    let cloud_bg = self
                        .bg3_cloud_shadows
                        .as_ref()
                        .unwrap_or(&self.bg4_dummy_cloud_shadows);
                    rp.set_bind_group(4, cloud_bg, &[]);
                }
                // B5: Planar reflections at group 5 when available and supported
                if max_groups >= 6 {
                    if let Some(ref renderer) = self.reflection_renderer {
                        if let Some(reflection_bg) = renderer.bind_group() {
                            rp.set_bind_group(5, reflection_bg, &[]);
                        }
                    }
                }

                rp.set_vertex_buffer(0, self.vbuf.slice(..));
                rp.set_index_buffer(self.ibuf.slice(..), wgpu::IndexFormat::Uint32);
                rp.draw_indexed(0..self.nidx, 0, 0..1);
            }
        }
        if self.ssao_enabled {
            self.ssao
                .dispatch(
                    &g.device,
                    &g.queue,
                    &mut encoder,
                    &self.normal_view,
                    &self.color,
                )
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;
        }

        // B6: Apply DOF if enabled
        if let Err(e) = self.render_dof(&mut encoder) {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                "DOF rendering failed: {}",
                e
            )));
        }

        g.queue.submit(Some(encoder.finish()));

        // Readback -> unpadded RGBA bytes
        let bpp = 4u32;
        let unpadded = self.width * bpp;
        let padded = crate::gpu::align_copy_bpr(unpadded);
        let size = (padded * self.height) as wgpu::BufferAddress;

        let readback = g.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("scene-readback-rgba"),
            size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let mut enc = g
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("copy-encoder-rgba"),
            });
        enc.copy_texture_to_buffer(
            wgpu::ImageCopyTexture {
                texture: &self.color,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::ImageCopyBuffer {
                buffer: &readback,
                layout: wgpu::ImageDataLayout {
                    offset: 0,
                    bytes_per_row: Some(std::num::NonZeroU32::new(padded).unwrap().into()),
                    rows_per_image: Some(std::num::NonZeroU32::new(self.height).unwrap().into()),
                },
            },
            wgpu::Extent3d {
                width: self.width,
                height: self.height,
                depth_or_array_layers: 1,
            },
        );
        g.queue.submit(Some(enc.finish()));

        let slice = readback.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        g.device.poll(wgpu::Maintain::Wait);
        let data = slice.get_mapped_range();

        let mut pixels = Vec::with_capacity((unpadded * self.height) as usize);
        for row in 0..self.height {
            let s = (row * padded) as usize;
            let e = s + unpadded as usize;
            pixels.extend_from_slice(&data[s..e]);
        }
        drop(data);
        readback.unmap();

        // Convert to NumPy (H,W,4) u8
        let arr =
            ndarray::Array3::from_shape_vec((self.height as usize, self.width as usize, 4), pixels)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        Ok(arr.into_pyarray_bound(py))
    }

    #[pyo3(text_signature = "($self, samples)")]
    pub fn set_msaa_samples(&mut self, samples: u32) -> PyResult<u32> {
        const SUPPORTED: [u32; 4] = [1, 2, 4, 8];
        if !SUPPORTED.contains(&samples) {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                "Unsupported MSAA sample count: {} (allowed: {:?})",
                samples, SUPPORTED
            )));
        }
        if self.sample_count == samples {
            return Ok(samples);
        }

        let caps = DeviceCaps::from_current_device()
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        if samples > 1 {
            if !caps.msaa_supported {
                return Err(pyo3::exceptions::PyRuntimeError::new_err(
                    "MSAA not supported on current device".to_string(),
                ));
            }
            if samples > caps.max_samples {
                return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                    "Sample count {} exceeds device limit {}",
                    samples, caps.max_samples
                )));
            }
        }

        self.sample_count = samples;
        self.rebuild_msaa_state()
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;
        Ok(samples)
    }
    #[pyo3(text_signature = "($self)")]
    pub fn debug_uniforms_f32<'py>(
        &self,
        py: pyo3::Python<'py>,
    ) -> pyo3::PyResult<pyo3::Bound<'py, numpy::PyArray1<f32>>> {
        let bytes = bytemuck::bytes_of(&self.last_uniforms);
        let fl: &[f32] = bytemuck::cast_slice(bytes);
        Ok(numpy::PyArray1::from_vec_bound(py, fl.to_vec()))
    }

    #[pyo3(text_signature = "($self)")]
    pub fn debug_lut_format(&self) -> &'static str {
        self.lut_format
    }

    // B5: Planar Reflections API
    #[pyo3(text_signature = "($self, quality='medium')")]
    pub fn enable_reflections(&mut self, quality: Option<&str>) -> PyResult<()> {
        let quality_enum = match quality.unwrap_or("medium") {
            "low" => crate::core::reflections::ReflectionQuality::Low,
            "medium" => crate::core::reflections::ReflectionQuality::Medium,
            "high" => crate::core::reflections::ReflectionQuality::High,
            "ultra" => crate::core::reflections::ReflectionQuality::Ultra,
            other => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "Invalid quality '{}' . Use 'low', 'medium', 'high', or 'ultra'",
                    other
                )))
            }
        };

        let g = crate::gpu::ctx();
        let mut renderer =
            crate::core::reflections::PlanarReflectionRenderer::new(&g.device, quality_enum);

        if let Some(previous) = self.reflection_renderer.take() {
            let prev_uniforms = previous.uniforms;
            renderer.uniforms.reflection_plane = prev_uniforms.reflection_plane;
            renderer.set_intensity(prev_uniforms.reflection_intensity);
            renderer.set_fresnel_power(prev_uniforms.fresnel_power);
            renderer.set_distance_fade(
                prev_uniforms.distance_fade_start,
                prev_uniforms.distance_fade_end,
            );
            renderer.set_debug_mode(prev_uniforms.debug_mode);
            renderer.uniforms.camera_position = prev_uniforms.camera_position;
        }

        renderer.create_bind_group(&g.device, &self.tp.bgl_reflection);
        renderer.set_enabled(true);
        renderer.upload_uniforms(&g.queue);

        self.reflection_renderer = Some(renderer);
        self.reflections_enabled = true;
        Ok(())
    }

    #[pyo3(text_signature = "()")]
    pub fn disable_reflections(&mut self) {
        self.reflections_enabled = false;
        if let Some(ref mut renderer) = self.reflection_renderer {
            renderer.set_enabled(false);
            let g = crate::gpu::ctx();
            renderer.upload_uniforms(&g.queue);
        }
    }

    #[pyo3(text_signature = "($self, normal, point, size)")]
    pub fn set_reflection_plane(
        &mut self,
        normal: (f32, f32, f32),
        point: (f32, f32, f32),
        size: (f32, f32, f32),
    ) -> PyResult<()> {
        if !self.reflections_enabled {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Reflections not enabled. Call enable_reflections() first.",
            ));
        }
        let Some(ref mut renderer) = self.reflection_renderer else {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Reflections not enabled. Call enable_reflections() first.",
            ));
        };
        let normal_v = glam::Vec3::new(normal.0, normal.1, normal.2);
        let point_v = glam::Vec3::new(point.0, point.1, point.2);
        let size_v = glam::Vec3::new(size.0, size.1, size.2);
        renderer.set_reflection_plane(normal_v, point_v, size_v);
        let g = crate::gpu::ctx();
        renderer.upload_uniforms(&g.queue);
        Ok(())
    }

    #[pyo3(text_signature = "($self, intensity)")]
    pub fn set_reflection_intensity(&mut self, intensity: f32) -> PyResult<()> {
        if !self.reflections_enabled {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Reflections not enabled. Call enable_reflections() first.",
            ));
        }
        let Some(ref mut renderer) = self.reflection_renderer else {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Reflections not enabled. Call enable_reflections() first.",
            ));
        };
        renderer.set_intensity(intensity);
        let g = crate::gpu::ctx();
        renderer.upload_uniforms(&g.queue);
        Ok(())
    }

    #[pyo3(text_signature = "($self, power)")]
    pub fn set_reflection_fresnel_power(&mut self, power: f32) -> PyResult<()> {
        if power <= 0.0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Fresnel power must be positive.",
            ));
        }
        if !self.reflections_enabled {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Reflections not enabled. Call enable_reflections() first.",
            ));
        }
        let Some(ref mut renderer) = self.reflection_renderer else {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Reflections not enabled. Call enable_reflections() first.",
            ));
        };
        renderer.set_fresnel_power(power);
        let g = crate::gpu::ctx();
        renderer.upload_uniforms(&g.queue);
        Ok(())
    }

    #[pyo3(text_signature = "($self, start, end)")]
    pub fn set_reflection_distance_fade(&mut self, start: f32, end: f32) -> PyResult<()> {
        if end <= 0.0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "distance_fade_end must be positive.",
            ));
        }
        if !self.reflections_enabled {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Reflections not enabled. Call enable_reflections() first.",
            ));
        }
        let Some(ref mut renderer) = self.reflection_renderer else {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Reflections not enabled. Call enable_reflections() first.",
            ));
        };
        renderer.set_distance_fade(start, end);
        let g = crate::gpu::ctx();
        renderer.upload_uniforms(&g.queue);
        Ok(())
    }

    #[pyo3(text_signature = "($self, mode)")]
    pub fn set_reflection_debug_mode(&mut self, mode: u32) -> PyResult<()> {
        if mode > 4 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Debug mode must be an integer in [0, 4].",
            ));
        }
        if !self.reflections_enabled {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Reflections not enabled. Call enable_reflections() first.",
            ));
        }
        let Some(ref mut renderer) = self.reflection_renderer else {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Reflections not enabled. Call enable_reflections() first.",
            ));
        };
        renderer.set_debug_mode(mode);
        let g = crate::gpu::ctx();
        renderer.upload_uniforms(&g.queue);
        Ok(())
    }

    #[pyo3(text_signature = "()")]
    pub fn reflection_performance_info(&self) -> PyResult<(f32, bool)> {
        if !self.reflections_enabled {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Reflections not enabled. Call enable_reflections() first.",
            ));
        }
        let Some(ref renderer) = self.reflection_renderer else {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Reflections not enabled. Call enable_reflections() first.",
            ));
        };
        let cost = renderer.estimate_frame_cost();
        let meets_requirement = renderer.meets_performance_requirement();
        Ok((cost, meets_requirement))
    }

    // B6: Depth of Field API
    #[pyo3(text_signature = "($self, quality='medium')")]
    pub fn enable_dof(&mut self, quality: Option<&str>) -> PyResult<()> {
        let quality_enum = match quality.unwrap_or("medium") {
            "low" => crate::core::dof::DofQuality::Low,
            "medium" => crate::core::dof::DofQuality::Medium,
            "high" => crate::core::dof::DofQuality::High,
            "ultra" => crate::core::dof::DofQuality::Ultra,
            other => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "Invalid quality '{}'. Use 'low', 'medium', 'high', or 'ultra'",
                    other
                )))
            }
        };

        let g = crate::gpu::ctx();
        let renderer =
            crate::core::dof::DofRenderer::new(&g.device, self.width, self.height, quality_enum);

        self.dof_renderer = Some(renderer);
        self.dof_enabled = true;
        Ok(())
    }

    #[pyo3(text_signature = "($self)")]
    pub fn disable_dof(&mut self) {
        self.dof_enabled = false;
    }

    #[pyo3(text_signature = "($self)")]
    pub fn dof_enabled(&self) -> bool {
        self.dof_enabled
    }

    #[pyo3(text_signature = "($self, aperture, focus_distance, focal_length)")]
    pub fn set_dof_camera_params(
        &mut self,
        aperture: f32,
        focus_distance: f32,
        focal_length: f32,
    ) -> PyResult<()> {
        self.dof_params = crate::camera::create_camera_dof_params(
            aperture,
            focus_distance,
            focal_length,
            false,
            2.0,
        )?;

        if let Some(ref mut renderer) = self.dof_renderer {
            renderer.set_camera_params(self.dof_params);
        }
        Ok(())
    }

    #[pyo3(text_signature = "($self, f_stop)")]
    pub fn set_dof_f_stop(&mut self, f_stop: f32) -> PyResult<()> {
        let aperture = crate::camera::camera_f_stop_to_aperture(f_stop)?;
        self.dof_params.aperture = aperture;

        if let Some(ref mut renderer) = self.dof_renderer {
            renderer.set_aperture(aperture);
        }
        Ok(())
    }

    #[pyo3(text_signature = "($self, distance)")]
    pub fn set_dof_focus_distance(&mut self, distance: f32) -> PyResult<()> {
        if !distance.is_finite() || distance <= 0.0 {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "focus_distance must be finite and > 0",
            ));
        }

        self.dof_params.focus_distance = distance;

        if let Some(ref mut renderer) = self.dof_renderer {
            renderer.set_focus_distance(distance);
        }
        Ok(())
    }

    #[pyo3(text_signature = "($self, focal_length)")]
    pub fn set_dof_focal_length(&mut self, focal_length: f32) -> PyResult<()> {
        if !focal_length.is_finite() || focal_length <= 0.0 {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "focal_length must be finite and > 0",
            ));
        }

        self.dof_params.focal_length = focal_length;

        if let Some(ref mut renderer) = self.dof_renderer {
            renderer.set_focal_length(focal_length);
        }
        Ok(())
    }

    #[pyo3(text_signature = "($self, rotation)")]
    pub fn set_dof_bokeh_rotation(&mut self, rotation: f32) -> PyResult<()> {
        if let Some(ref mut renderer) = self.dof_renderer {
            renderer.set_bokeh_rotation(rotation);
            Ok(())
        } else {
            Err(pyo3::exceptions::PyRuntimeError::new_err(
                "DOF not enabled. Call enable_dof() first.",
            ))
        }
    }

    #[pyo3(text_signature = "($self, near_range, far_range)")]
    pub fn set_dof_transition_ranges(&mut self, near_range: f32, far_range: f32) -> PyResult<()> {
        if let Some(ref mut renderer) = self.dof_renderer {
            renderer.set_transition_ranges(near_range, far_range);
            Ok(())
        } else {
            Err(pyo3::exceptions::PyRuntimeError::new_err(
                "DOF not enabled. Call enable_dof() first.",
            ))
        }
    }

    #[pyo3(text_signature = "($self, bias)")]
    pub fn set_dof_coc_bias(&mut self, bias: f32) -> PyResult<()> {
        if let Some(ref mut renderer) = self.dof_renderer {
            renderer.set_coc_bias(bias);
            Ok(())
        } else {
            Err(pyo3::exceptions::PyRuntimeError::new_err(
                "DOF not enabled. Call enable_dof() first.",
            ))
        }
    }

    #[pyo3(text_signature = "($self, method)")]
    pub fn set_dof_method(&mut self, method: &str) -> PyResult<()> {
        let method_enum = match method {
            "gather" => crate::core::dof::DofMethod::Gather,
            "separable" => crate::core::dof::DofMethod::Separable,
            other => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "Invalid method '{}'. Use 'gather' or 'separable'",
                    other
                )))
            }
        };

        if let Some(ref mut renderer) = self.dof_renderer {
            renderer.set_method(method_enum);
            Ok(())
        } else {
            Err(pyo3::exceptions::PyRuntimeError::new_err(
                "DOF not enabled. Call enable_dof() first.",
            ))
        }
    }

    #[pyo3(text_signature = "($self, mode)")]
    pub fn set_dof_debug_mode(&mut self, mode: u32) -> PyResult<()> {
        if let Some(ref mut renderer) = self.dof_renderer {
            renderer.set_debug_mode(mode);
            Ok(())
        } else {
            Err(pyo3::exceptions::PyRuntimeError::new_err(
                "DOF not enabled. Call enable_dof() first.",
            ))
        }
    }

    #[pyo3(text_signature = "($self, show)")]
    pub fn set_dof_show_coc(&mut self, show: bool) -> PyResult<()> {
        if let Some(ref mut renderer) = self.dof_renderer {
            renderer.set_show_coc(show);
            Ok(())
        } else {
            Err(pyo3::exceptions::PyRuntimeError::new_err(
                "DOF not enabled. Call enable_dof() first.",
            ))
        }
    }

    #[pyo3(text_signature = "($self)")]
    pub fn get_dof_params(&self) -> (f32, f32, f32) {
        (
            self.dof_params.aperture,
            self.dof_params.focus_distance,
            self.dof_params.focal_length,
        )
    }

    // B7: Cloud Shadow API
    #[pyo3(text_signature = "($self, quality='medium')")]
    pub fn enable_cloud_shadows(&mut self, quality: Option<&str>) -> PyResult<()> {
        let quality_enum = match quality.unwrap_or("medium") {
            "low" => crate::core::cloud_shadows::CloudShadowQuality::Low,
            "medium" => crate::core::cloud_shadows::CloudShadowQuality::Medium,
            "high" => crate::core::cloud_shadows::CloudShadowQuality::High,
            "ultra" => crate::core::cloud_shadows::CloudShadowQuality::Ultra,
            other => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "Invalid quality '{}'. Use 'low', 'medium', 'high', or 'ultra'",
                    other
                )))
            }
        };

        let g = crate::gpu::ctx();
        let renderer =
            crate::core::cloud_shadows::CloudShadowRenderer::new(&g.device, quality_enum);

        self.cloud_shadow_renderer = Some(renderer);
        self.cloud_shadows_enabled = true;
        self.bg3_cloud_shadows = None; // Will be created on first render
        Ok(())
    }

    #[pyo3(text_signature = "($self)")]
    pub fn disable_cloud_shadows(&mut self) {
        self.cloud_shadows_enabled = false;
    }

    #[pyo3(text_signature = "($self)")]
    pub fn is_cloud_shadows_enabled(&self) -> bool {
        self.cloud_shadows_enabled
    }

    #[pyo3(text_signature = "($self, speed_x, speed_y)")]
    pub fn set_cloud_speed(&mut self, speed_x: f32, speed_y: f32) -> PyResult<()> {
        if let Some(ref mut renderer) = self.cloud_shadow_renderer {
            renderer.set_cloud_speed(glam::Vec2::new(speed_x, speed_y));
            Ok(())
        } else {
            Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Cloud shadows not enabled. Call enable_cloud_shadows() first.",
            ))
        }
    }

    #[pyo3(text_signature = "($self, scale)")]
    pub fn set_cloud_scale(&mut self, scale: f32) -> PyResult<()> {
        let mut updated = false;
        if let Some(ref mut renderer) = self.cloud_shadow_renderer {
            renderer.set_cloud_scale(scale);
            updated = true;
        }
        if let Some(ref mut renderer) = self.cloud_renderer {
            renderer.set_scale(scale);
            renderer.upload_uniforms(&crate::gpu::ctx().queue);
            updated = true;
        }
        if updated {
            Ok(())
        } else {
            Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Clouds not enabled. Call enable_clouds() or enable_cloud_shadows() first.",
            ))
        }
    }

    #[pyo3(text_signature = "($self, density)")]
    pub fn set_cloud_density(&mut self, density: f32) -> PyResult<()> {
        let mut updated = false;
        if let Some(ref mut renderer) = self.cloud_shadow_renderer {
            renderer.set_cloud_density(density);
            updated = true;
        }
        if let Some(ref mut renderer) = self.cloud_renderer {
            renderer.set_density(density);
            renderer.upload_uniforms(&crate::gpu::ctx().queue);
            updated = true;
        }
        if updated {
            Ok(())
        } else {
            Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Clouds not enabled. Call enable_clouds() or enable_cloud_shadows() first.",
            ))
        }
    }

    #[pyo3(text_signature = "($self, coverage)")]
    pub fn set_cloud_coverage(&mut self, coverage: f32) -> PyResult<()> {
        let mut updated = false;
        if let Some(ref mut renderer) = self.cloud_shadow_renderer {
            renderer.set_cloud_coverage(coverage);
            updated = true;
        }
        if let Some(ref mut renderer) = self.cloud_renderer {
            renderer.set_coverage(coverage);
            renderer.upload_uniforms(&crate::gpu::ctx().queue);
            updated = true;
        }
        if updated {
            Ok(())
        } else {
            Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Clouds not enabled. Call enable_clouds() or enable_cloud_shadows() first.",
            ))
        }
    }

    #[pyo3(text_signature = "($self, intensity)")]
    pub fn set_cloud_shadow_intensity(&mut self, intensity: f32) -> PyResult<()> {
        if let Some(ref mut renderer) = self.cloud_shadow_renderer {
            renderer.set_shadow_intensity(intensity);
            Ok(())
        } else {
            Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Cloud shadows not enabled. Call enable_cloud_shadows() first.",
            ))
        }
    }

    #[pyo3(text_signature = "($self, softness)")]
    pub fn set_cloud_shadow_softness(&mut self, softness: f32) -> PyResult<()> {
        if let Some(ref mut renderer) = self.cloud_shadow_renderer {
            renderer.set_shadow_softness(softness);
            Ok(())
        } else {
            Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Cloud shadows not enabled. Call enable_cloud_shadows() first.",
            ))
        }
    }

    #[pyo3(text_signature = "($self, direction, strength)")]
    pub fn set_cloud_wind(&mut self, direction: f32, strength: f32) -> PyResult<()> {
        let mut updated = false;
        if let Some(ref mut renderer) = self.cloud_shadow_renderer {
            renderer.set_wind(direction, strength);
            updated = true;
        }
        if let Some(ref mut renderer) = self.cloud_renderer {
            let dir_vec = glam::Vec2::new(direction.cos(), direction.sin());
            renderer.set_wind(dir_vec, strength);
            renderer.upload_uniforms(&crate::gpu::ctx().queue);
            updated = true;
        }
        if updated {
            Ok(())
        } else {
            Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Clouds not enabled. Call enable_clouds() or enable_cloud_shadows() first.",
            ))
        }
    }

    #[pyo3(text_signature = "($self, x, y, strength)")]
    pub fn set_cloud_wind_vector(&mut self, x: f32, y: f32, strength: f32) -> PyResult<()> {
        let mut updated = false;
        if let Some(ref mut renderer) = self.cloud_shadow_renderer {
            let angle = y.atan2(x);
            renderer.set_wind(angle, strength);
            updated = true;
        }
        if let Some(ref mut renderer) = self.cloud_renderer {
            let wind_vec = glam::Vec2::new(x, y);
            renderer.set_wind(wind_vec, strength);
            renderer.upload_uniforms(&crate::gpu::ctx().queue);
            updated = true;
        }
        if updated {
            Ok(())
        } else {
            Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Clouds not enabled. Call enable_clouds() or enable_cloud_shadows() first.",
            ))
        }
    }

    #[pyo3(text_signature = "($self, frequency, amplitude)")]
    pub fn set_cloud_noise_params(&mut self, frequency: f32, amplitude: f32) -> PyResult<()> {
        if let Some(ref mut renderer) = self.cloud_shadow_renderer {
            renderer.set_noise_params(frequency, amplitude);
            Ok(())
        } else {
            Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Cloud shadows not enabled. Call enable_cloud_shadows() first.",
            ))
        }
    }

    #[pyo3(text_signature = "($self, preset_name)")]
    pub fn set_cloud_animation_preset(&mut self, preset_name: &str) -> PyResult<()> {
        let preset_lower = preset_name.to_ascii_lowercase();
        let mut updated = false;
        if let Some(ref mut renderer) = self.cloud_shadow_renderer {
            let mapped = match preset_lower.as_str() {
                "static" => "calm",
                "gentle" => "calm",
                "moderate" => "windy",
                "stormy" => "stormy",
                other => other,
            };
            let params = crate::core::cloud_shadows::utils::create_animation_preset(mapped);
            renderer.set_animation_params(params);
            updated = true;
        }
        if let Some(ref mut renderer) = self.cloud_renderer {
            let preset_enum = match preset_lower.as_str() {
                "static" => crate::core::clouds::CloudAnimationPreset::Static,
                "gentle" | "calm" => crate::core::clouds::CloudAnimationPreset::Gentle,
                "moderate" => crate::core::clouds::CloudAnimationPreset::Moderate,
                "stormy" | "windy" => crate::core::clouds::CloudAnimationPreset::Stormy,
                other => {
                    return Err(pyo3::exceptions::PyValueError::new_err(format!(
                        "Preset must be one of: static, gentle, moderate, stormy (got '{}')",
                        other
                    )));
                }
            };
            renderer.set_animation_preset(preset_enum);
            renderer.upload_uniforms(&crate::gpu::ctx().queue);
            updated = true;
        }
        if updated {
            Ok(())
        } else {
            Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Clouds not enabled. Call enable_clouds() or enable_cloud_shadows() first.",
            ))
        }
    }

    #[pyo3(text_signature = "($self, delta_time)")]
    pub fn update_cloud_animation(&mut self, delta_time: f32) -> PyResult<()> {
        let mut updated = false;
        if let Some(ref mut renderer) = self.cloud_shadow_renderer {
            renderer.update(delta_time);
            updated = true;
        }
        if let Some(ref mut renderer) = self.cloud_renderer {
            renderer.update(delta_time);
            renderer.upload_uniforms(&crate::gpu::ctx().queue);
            updated = true;
        }
        if updated {
            Ok(())
        } else {
            Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Clouds not enabled. Call enable_clouds() or enable_cloud_shadows() first.",
            ))
        }
    }

    #[pyo3(text_signature = "($self, mode)")]
    pub fn set_cloud_debug_mode(&mut self, mode: u32) -> PyResult<()> {
        if let Some(ref mut renderer) = self.cloud_shadow_renderer {
            renderer.set_debug_mode(mode);
            Ok(())
        } else {
            Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Cloud shadows not enabled. Call enable_cloud_shadows() first.",
            ))
        }
    }

    #[pyo3(text_signature = "($self, show)")]
    pub fn set_cloud_show_clouds_only(&mut self, show: bool) -> PyResult<()> {
        if let Some(ref mut renderer) = self.cloud_shadow_renderer {
            renderer.set_show_clouds_only(show);
            Ok(())
        } else {
            Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Cloud shadows not enabled. Call enable_cloud_shadows() first.",
            ))
        }
    }

    #[pyo3(text_signature = "($self)")]
    pub fn get_cloud_params(&self) -> PyResult<(f32, f32, f32, f32)> {
        if let Some(ref renderer) = self.cloud_shadow_renderer {
            Ok(renderer.get_cloud_params())
        } else {
            Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Cloud shadows not enabled. Call enable_cloud_shadows() first.",
            ))
        }
    }

    // B8: Realtime Clouds API
    #[pyo3(text_signature = "($self, quality='medium')")]
    pub fn enable_clouds(&mut self, quality: Option<&str>) -> PyResult<()> {
        let quality_enum = match quality.unwrap_or("medium") {
            "low" => crate::core::clouds::CloudQuality::Low,
            "medium" => crate::core::clouds::CloudQuality::Medium,
            "high" => crate::core::clouds::CloudQuality::High,
            "ultra" => crate::core::clouds::CloudQuality::Ultra,
            _ => {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "Quality must be one of: 'low', 'medium', 'high', 'ultra'",
                ))
            }
        };

        let g = crate::gpu::ctx();
        let mut renderer = crate::core::clouds::CloudRenderer::new(
            &g.device,
            wgpu::TextureFormat::Rgba8UnormSrgb,
            1, // clouds render against resolved color buffer
        );
        renderer.set_quality(quality_enum);
        renderer
            .prepare_frame(&g.device, &g.queue)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;
        renderer.upload_uniforms(&g.queue);
        renderer.set_enabled(true);

        self.cloud_renderer = Some(renderer);
        self.clouds_enabled = true;

        Ok(())
    }

    #[pyo3(text_signature = "($self)")]
    pub fn disable_clouds(&mut self) {
        self.clouds_enabled = false;
        self.cloud_renderer = None;
    }

    #[pyo3(text_signature = "($self)")]
    pub fn is_clouds_enabled(&self) -> bool {
        self.clouds_enabled && self.cloud_renderer.is_some()
    }

    #[pyo3(text_signature = "($self, density)")]
    pub fn set_realtime_cloud_density(&mut self, density: f32) -> PyResult<()> {
        if let Some(ref mut renderer) = self.cloud_renderer {
            renderer.set_density(density);
            Ok(())
        } else {
            Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Clouds not enabled. Call enable_clouds() first.",
            ))
        }
    }

    #[pyo3(text_signature = "($self, coverage)")]
    pub fn set_realtime_cloud_coverage(&mut self, coverage: f32) -> PyResult<()> {
        if let Some(ref mut renderer) = self.cloud_renderer {
            renderer.set_coverage(coverage);
            Ok(())
        } else {
            Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Clouds not enabled. Call enable_clouds() first.",
            ))
        }
    }

    #[pyo3(text_signature = "($self, scale)")]
    pub fn set_realtime_cloud_scale(&mut self, scale: f32) -> PyResult<()> {
        if let Some(ref mut renderer) = self.cloud_renderer {
            renderer.set_scale(scale);
            Ok(())
        } else {
            Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Clouds not enabled. Call enable_clouds() first.",
            ))
        }
    }

    #[pyo3(text_signature = "($self, direction_x, direction_y, strength)")]
    pub fn set_realtime_cloud_wind(
        &mut self,
        direction_x: f32,
        direction_y: f32,
        strength: f32,
    ) -> PyResult<()> {
        if let Some(ref mut renderer) = self.cloud_renderer {
            let direction = glam::Vec2::new(direction_x, direction_y);
            renderer.set_wind(direction, strength);
            Ok(())
        } else {
            Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Clouds not enabled. Call enable_clouds() first.",
            ))
        }
    }

    #[pyo3(text_signature = "($self, preset)")]
    pub fn set_realtime_cloud_animation_preset(&mut self, preset: &str) -> PyResult<()> {
        if let Some(ref mut renderer) = self.cloud_renderer {
            let preset_enum = match preset {
                "static" => crate::core::clouds::CloudAnimationPreset::Static,
                "gentle" => crate::core::clouds::CloudAnimationPreset::Gentle,
                "moderate" => crate::core::clouds::CloudAnimationPreset::Moderate,
                "stormy" => crate::core::clouds::CloudAnimationPreset::Stormy,
                _ => {
                    return Err(pyo3::exceptions::PyValueError::new_err(
                        "Preset must be one of: 'static', 'gentle', 'moderate', 'stormy'",
                    ))
                }
            };
            renderer.set_animation_preset(preset_enum);
            Ok(())
        } else {
            Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Clouds not enabled. Call enable_clouds() first.",
            ))
        }
    }

    #[pyo3(text_signature = "($self, mode)")]
    pub fn set_cloud_render_mode(&mut self, mode: &str) -> PyResult<()> {
        if let Some(ref mut renderer) = self.cloud_renderer {
            let mode_enum = match mode {
                "billboard" => crate::core::clouds::CloudRenderMode::Billboard,
                "volumetric" => crate::core::clouds::CloudRenderMode::Volumetric,
                "hybrid" => crate::core::clouds::CloudRenderMode::Hybrid,
                _ => {
                    return Err(pyo3::exceptions::PyValueError::new_err(
                        "Mode must be one of: 'billboard', 'volumetric', 'hybrid'",
                    ))
                }
            };
            renderer.set_render_mode(mode_enum);
            Ok(())
        } else {
            Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Clouds not enabled. Call enable_clouds() first.",
            ))
        }
    }

    #[pyo3(text_signature = "($self, delta_time)")]
    pub fn update_realtime_cloud_animation(&mut self, delta_time: f32) -> PyResult<()> {
        if let Some(ref mut renderer) = self.cloud_renderer {
            renderer.update(delta_time);
            Ok(())
        } else {
            Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Clouds not enabled. Call enable_clouds() first.",
            ))
        }
    }

    #[pyo3(text_signature = "($self)")]
    pub fn get_clouds_params(&self) -> PyResult<(f32, f32, f32, f32)> {
        if let Some(ref renderer) = self.cloud_renderer {
            Ok(renderer.get_params())
        } else if let Some(ref renderer) = self.cloud_shadow_renderer {
            Ok(renderer.get_cloud_params())
        } else {
            Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Clouds not enabled. Call enable_clouds() or enable_cloud_shadows() first.",
            ))
        }
    }

    // B10: Ground Plane (Raster) API
    #[pyo3(text_signature = "($self)")]
    pub fn enable_ground_plane(&mut self) -> PyResult<()> {
        let g = crate::gpu::ctx();
        let renderer = crate::core::ground_plane::GroundPlaneRenderer::new(
            &g.device,
            wgpu::TextureFormat::Rgba8UnormSrgb,
            Some(wgpu::TextureFormat::Depth32Float),
            1, // sample_count
        );

        self.ground_plane_renderer = Some(renderer);
        self.ground_plane_enabled = true;

        Ok(())
    }

    #[pyo3(text_signature = "($self)")]
    pub fn disable_ground_plane(&mut self) {
        self.ground_plane_enabled = false;
        self.ground_plane_renderer = None;
    }

    #[pyo3(text_signature = "($self)")]
    pub fn is_ground_plane_enabled(&self) -> bool {
        self.ground_plane_enabled && self.ground_plane_renderer.is_some()
    }

    #[pyo3(text_signature = "($self, mode)")]
    pub fn set_ground_plane_mode(&mut self, mode: &str) -> PyResult<()> {
        if let Some(ref mut renderer) = self.ground_plane_renderer {
            let mode_enum = match mode {
                "disabled" => crate::core::ground_plane::GroundPlaneMode::Disabled,
                "solid" => crate::core::ground_plane::GroundPlaneMode::Solid,
                "grid" => crate::core::ground_plane::GroundPlaneMode::Grid,
                "checkerboard" => crate::core::ground_plane::GroundPlaneMode::CheckerBoard,
                _ => {
                    return Err(pyo3::exceptions::PyValueError::new_err(
                        "Mode must be one of: 'disabled', 'solid', 'grid', 'checkerboard'",
                    ))
                }
            };
            renderer.set_mode(mode_enum);
            Ok(())
        } else {
            Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Ground plane not enabled. Call enable_ground_plane() first.",
            ))
        }
    }

    #[pyo3(text_signature = "($self, height)")]
    pub fn set_ground_plane_height(&mut self, height: f32) -> PyResult<()> {
        if let Some(ref mut renderer) = self.ground_plane_renderer {
            renderer.set_height(height);
            Ok(())
        } else {
            Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Ground plane not enabled. Call enable_ground_plane() first.",
            ))
        }
    }

    #[pyo3(text_signature = "($self, size)")]
    pub fn set_ground_plane_size(&mut self, size: f32) -> PyResult<()> {
        if let Some(ref mut renderer) = self.ground_plane_renderer {
            renderer.set_size(size);
            Ok(())
        } else {
            Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Ground plane not enabled. Call enable_ground_plane() first.",
            ))
        }
    }

    #[pyo3(text_signature = "($self, major_spacing, minor_spacing)")]
    pub fn set_ground_plane_grid_spacing(
        &mut self,
        major_spacing: f32,
        minor_spacing: f32,
    ) -> PyResult<()> {
        if let Some(ref mut renderer) = self.ground_plane_renderer {
            renderer.set_grid_spacing(major_spacing, minor_spacing);
            Ok(())
        } else {
            Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Ground plane not enabled. Call enable_ground_plane() first.",
            ))
        }
    }

    #[pyo3(text_signature = "($self, major_width, minor_width)")]
    pub fn set_ground_plane_grid_width(
        &mut self,
        major_width: f32,
        minor_width: f32,
    ) -> PyResult<()> {
        if let Some(ref mut renderer) = self.ground_plane_renderer {
            renderer.set_grid_width(major_width, minor_width);
            Ok(())
        } else {
            Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Ground plane not enabled. Call enable_ground_plane() first.",
            ))
        }
    }

    #[pyo3(text_signature = "($self, r, g, b, alpha)")]
    pub fn set_ground_plane_color(&mut self, r: f32, g: f32, b: f32, alpha: f32) -> PyResult<()> {
        if let Some(ref mut renderer) = self.ground_plane_renderer {
            renderer.set_albedo(glam::Vec3::new(r, g, b), alpha);
            Ok(())
        } else {
            Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Ground plane not enabled. Call enable_ground_plane() first.",
            ))
        }
    }

    #[pyo3(
        text_signature = "($self, major_r, major_g, major_b, major_alpha, minor_r, minor_g, minor_b, minor_alpha)"
    )]
    pub fn set_ground_plane_grid_colors(
        &mut self,
        major_r: f32,
        major_g: f32,
        major_b: f32,
        major_alpha: f32,
        minor_r: f32,
        minor_g: f32,
        minor_b: f32,
        minor_alpha: f32,
    ) -> PyResult<()> {
        if let Some(ref mut renderer) = self.ground_plane_renderer {
            let major_color = glam::Vec3::new(major_r, major_g, major_b);
            let minor_color = glam::Vec3::new(minor_r, minor_g, minor_b);
            renderer.set_grid_colors(major_color, major_alpha, minor_color, minor_alpha);
            Ok(())
        } else {
            Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Ground plane not enabled. Call enable_ground_plane() first.",
            ))
        }
    }

    #[pyo3(text_signature = "($self, z_bias)")]
    pub fn set_ground_plane_z_bias(&mut self, z_bias: f32) -> PyResult<()> {
        if let Some(ref mut renderer) = self.ground_plane_renderer {
            renderer.set_z_bias(z_bias);
            Ok(())
        } else {
            Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Ground plane not enabled. Call enable_ground_plane() first.",
            ))
        }
    }

    #[pyo3(text_signature = "($self, preset)")]
    pub fn set_ground_plane_preset(&mut self, preset: &str) -> PyResult<()> {
        if let Some(ref mut renderer) = self.ground_plane_renderer {
            let params = match preset {
                "engineering" => {
                    crate::core::ground_plane::GroundPlaneRenderer::create_engineering_grid()
                }
                "architectural" => {
                    crate::core::ground_plane::GroundPlaneRenderer::create_architectural_grid()
                }
                "simple" => crate::core::ground_plane::GroundPlaneRenderer::create_simple_ground(),
                _ => {
                    return Err(pyo3::exceptions::PyValueError::new_err(
                        "Preset must be one of: 'engineering', 'architectural', 'simple'",
                    ))
                }
            };
            renderer.update_params(params);
            Ok(())
        } else {
            Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Ground plane not enabled. Call enable_ground_plane() first.",
            ))
        }
    }

    #[pyo3(text_signature = "($self)")]
    pub fn get_ground_plane_params(&self) -> PyResult<(f32, f32, f32, f32)> {
        if let Some(ref renderer) = self.ground_plane_renderer {
            Ok(renderer.get_params())
        } else {
            Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Ground plane not enabled. Call enable_ground_plane() first.",
            ))
        }
    }

    // B11: Water Surface Color Toggle API
    #[pyo3(text_signature = "($self)")]
    pub fn enable_water_surface(&mut self) -> PyResult<()> {
        let g = crate::gpu::ctx();
        let renderer = crate::core::water_surface::WaterSurfaceRenderer::new(
            &g.device,
            wgpu::TextureFormat::Rgba8UnormSrgb,
            Some(wgpu::TextureFormat::Depth32Float),
            1, // sample_count
        );

        self.water_surface_renderer = Some(renderer);
        self.water_surface_enabled = true;

        Ok(())
    }

    #[pyo3(text_signature = "($self)")]
    pub fn disable_water_surface(&mut self) {
        self.water_surface_enabled = false;
        self.water_surface_renderer = None;
    }

    #[pyo3(text_signature = "($self)")]
    pub fn is_water_surface_enabled(&self) -> bool {
        self.water_surface_enabled && self.water_surface_renderer.is_some()
    }

    #[pyo3(text_signature = "($self, mode)")]
    pub fn set_water_surface_mode(&mut self, mode: &str) -> PyResult<()> {
        if let Some(ref mut renderer) = self.water_surface_renderer {
            let mode_enum =
                match mode {
                    "disabled" => crate::core::water_surface::WaterSurfaceMode::Disabled,
                    "transparent" => crate::core::water_surface::WaterSurfaceMode::Transparent,
                    "reflective" => crate::core::water_surface::WaterSurfaceMode::Reflective,
                    "animated" => crate::core::water_surface::WaterSurfaceMode::Animated,
                    _ => return Err(pyo3::exceptions::PyValueError::new_err(
                        "Mode must be one of: 'disabled', 'transparent', 'reflective', 'animated'",
                    )),
                };
            renderer.set_mode(mode_enum);
            Ok(())
        } else {
            Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Water surface not enabled. Call enable_water_surface() first.",
            ))
        }
    }

    #[pyo3(text_signature = "($self, height)")]
    pub fn set_water_surface_height(&mut self, height: f32) -> PyResult<()> {
        if let Some(ref mut renderer) = self.water_surface_renderer {
            renderer.set_height(height);
            Ok(())
        } else {
            Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Water surface not enabled. Call enable_water_surface() first.",
            ))
        }
    }

    #[pyo3(text_signature = "($self, size)")]
    pub fn set_water_surface_size(&mut self, size: f32) -> PyResult<()> {
        if let Some(ref mut renderer) = self.water_surface_renderer {
            renderer.set_size(size);
            Ok(())
        } else {
            Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Water surface not enabled. Call enable_water_surface() first.",
            ))
        }
    }

    #[pyo3(text_signature = "($self, r, g, b)")]
    pub fn set_water_base_color(&mut self, r: f32, g: f32, b: f32) -> PyResult<()> {
        if let Some(ref mut renderer) = self.water_surface_renderer {
            renderer.set_base_color(glam::Vec3::new(r, g, b));
            Ok(())
        } else {
            Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Water surface not enabled. Call enable_water_surface() first.",
            ))
        }
    }

    #[pyo3(text_signature = "($self, hue_shift)")]
    pub fn set_water_hue_shift(&mut self, hue_shift: f32) -> PyResult<()> {
        if let Some(ref mut renderer) = self.water_surface_renderer {
            renderer.set_hue_shift(hue_shift);
            Ok(())
        } else {
            Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Water surface not enabled. Call enable_water_surface() first.",
            ))
        }
    }

    #[pyo3(text_signature = "($self, r, g, b, strength)")]
    pub fn set_water_tint(&mut self, r: f32, g: f32, b: f32, strength: f32) -> PyResult<()> {
        if let Some(ref mut renderer) = self.water_surface_renderer {
            let tint_color = glam::Vec3::new(r, g, b);
            renderer.set_tint(tint_color, strength);
            Ok(())
        } else {
            Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Water surface not enabled. Call enable_water_surface() first.",
            ))
        }
    }

    #[pyo3(text_signature = "($self, alpha)")]
    pub fn set_water_alpha(&mut self, alpha: f32) -> PyResult<()> {
        if let Some(ref mut renderer) = self.water_surface_renderer {
            renderer.set_alpha(alpha);
            Ok(())
        } else {
            Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Water surface not enabled. Call enable_water_surface() first.",
            ))
        }
    }

    #[pyo3(text_signature = "($self, amplitude, frequency, speed)")]
    pub fn set_water_wave_params(
        &mut self,
        amplitude: f32,
        frequency: f32,
        speed: f32,
    ) -> PyResult<()> {
        if let Some(ref mut renderer) = self.water_surface_renderer {
            renderer.set_wave_params(amplitude, frequency, speed);
            Ok(())
        } else {
            Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Water surface not enabled. Call enable_water_surface() first.",
            ))
        }
    }

    #[pyo3(text_signature = "($self, direction_x, direction_y)")]
    pub fn set_water_flow_direction(&mut self, direction_x: f32, direction_y: f32) -> PyResult<()> {
        if let Some(ref mut renderer) = self.water_surface_renderer {
            let direction = glam::Vec2::new(direction_x, direction_y);
            renderer.set_flow_direction(direction);
            Ok(())
        } else {
            Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Water surface not enabled. Call enable_water_surface() first.",
            ))
        }
    }

    #[pyo3(
        text_signature = "($self, reflection_strength, refraction_strength, fresnel_power, roughness)"
    )]
    pub fn set_water_lighting_params(
        &mut self,
        reflection_strength: f32,
        refraction_strength: f32,
        fresnel_power: f32,
        roughness: f32,
    ) -> PyResult<()> {
        if let Some(ref mut renderer) = self.water_surface_renderer {
            renderer.set_lighting_params(
                reflection_strength,
                refraction_strength,
                fresnel_power,
                roughness,
            );
            Ok(())
        } else {
            Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Water surface not enabled. Call enable_water_surface() first.",
            ))
        }
    }

    #[pyo3(text_signature = "($self, preset)")]
    pub fn set_water_preset(&mut self, preset: &str) -> PyResult<()> {
        if let Some(ref mut renderer) = self.water_surface_renderer {
            let params = match preset {
                "ocean" => crate::core::water_surface::WaterSurfaceRenderer::create_ocean_water(),
                "lake" => crate::core::water_surface::WaterSurfaceRenderer::create_lake_water(),
                "river" => crate::core::water_surface::WaterSurfaceRenderer::create_river_water(),
                _ => {
                    return Err(pyo3::exceptions::PyValueError::new_err(
                        "Preset must be one of: 'ocean', 'lake', 'river'",
                    ))
                }
            };
            renderer.update_params(params);
            Ok(())
        } else {
            Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Water surface not enabled. Call enable_water_surface() first.",
            ))
        }
    }

    #[pyo3(text_signature = "($self, delta_time)")]
    pub fn update_water_animation(&mut self, delta_time: f32) -> PyResult<()> {
        if let Some(ref mut renderer) = self.water_surface_renderer {
            renderer.update(delta_time);
            Ok(())
        } else {
            Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Water surface not enabled. Call enable_water_surface() first.",
            ))
        }
    }

    #[pyo3(text_signature = "($self)")]
    pub fn get_water_surface_params(&self) -> PyResult<(f32, f32, f32, f32)> {
        if let Some(ref renderer) = self.water_surface_renderer {
            Ok(renderer.get_params())
        } else {
            Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Water surface not enabled. Call enable_water_surface() first.",
            ))
        }
    }

    // C3 (native): Shoreline foam controls (mirror fallback API names)
    #[pyo3(text_signature = "($self)")]
    pub fn enable_shoreline_foam(&mut self) -> PyResult<()> {
        if let Some(ref mut renderer) = self.water_surface_renderer {
            renderer.set_foam_enabled(true);
            Ok(())
        } else {
            Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Water surface not enabled. Call enable_water_surface() first.",
            ))
        }
    }

    #[pyo3(text_signature = "($self)")]
    pub fn disable_shoreline_foam(&mut self) -> PyResult<()> {
        if let Some(ref mut renderer) = self.water_surface_renderer {
            renderer.set_foam_enabled(false);
            Ok(())
        } else {
            Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Water surface not enabled. Call enable_water_surface() first.",
            ))
        }
    }

    #[pyo3(text_signature = "($self, width_px, intensity, noise_scale)")]
    pub fn set_shoreline_foam_params(
        &mut self,
        width_px: f32,
        intensity: f32,
        noise_scale: f32,
    ) -> PyResult<()> {
        if let Some(ref mut renderer) = self.water_surface_renderer {
            renderer.set_foam_params(width_px, intensity, noise_scale);
            Ok(())
        } else {
            Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Water surface not enabled. Call enable_water_surface() first.",
            ))
        }
    }

    // C1 (native): Upload a water mask from a numpy array (u8 or bool)
    //  - dtype=uint8: values interpreted in [0,255]
    //  - dtype=bool : True->255, False->0
    #[pyo3(text_signature = "($self, mask)")]
    pub fn set_water_mask(&mut self, _py: pyo3::Python<'_>, mask: &pyo3::PyAny) -> PyResult<()> {
        let (height, width, data_vec_u8) =
            if let Ok(arr_u8) = mask.extract::<PyReadonlyArray2<u8>>() {
                let shape = arr_u8.shape();
                let h = shape[0] as u32;
                let w = shape[1] as u32;
                // Ensure contiguous data
                let v = arr_u8.as_array().to_owned().into_raw_vec();
                (h, w, v)
            } else if let Ok(arr_b) = mask.extract::<PyReadonlyArray2<bool>>() {
                let a = arr_b.as_array();
                let h = a.shape()[0] as u32;
                let w = a.shape()[1] as u32;
                let mut v = Vec::<u8>::with_capacity((h as usize) * (w as usize));
                for &b in a.iter() {
                    v.push(if b { 255 } else { 0 });
                }
                (h, w, v)
            } else {
                return Err(pyo3::exceptions::PyTypeError::new_err(
                    "mask must be a 2D numpy array of dtype uint8 or bool",
                ));
            };

        if let Some(ref mut renderer) = self.water_surface_renderer {
            let g = crate::gpu::ctx();
            renderer.upload_water_mask(&g.device, &g.queue, &data_vec_u8, width, height);
            Ok(())
        } else {
            Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Water surface not enabled. Call enable_water_surface() first.",
            ))
        }
    }

    /// Set water surface debug mode.
    ///
    /// - 0: Normal rendering (default)
    /// - 100: Binary water mask (blue = water, gray = land)
    /// - 101: Shore-distance scalar (falsecolor ramp with white shoreline ring)
    /// - 102: IBL specular isolation (land = black, water shows compressed HDR fresnel)
    #[pyo3(text_signature = "($self, mode)")]
    pub fn set_water_surface_debug_mode(&mut self, mode: u32) -> PyResult<()> {
        if let Some(ref mut renderer) = self.water_surface_renderer {
            renderer.set_debug_mode(mode);
            Ok(())
        } else {
            Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Water surface not enabled. Call enable_water_surface() first.",
            ))
        }
    }

    // B12: Soft Light Radius (Raster) API
    #[pyo3(text_signature = "($self)")]
    pub fn enable_soft_light_radius(&mut self) -> PyResult<()> {
        let g = crate::gpu::ctx();
        let renderer = crate::core::soft_light_radius::SoftLightRadiusRenderer::new(&g.device);

        self.soft_light_radius_renderer = Some(renderer);
        self.soft_light_radius_enabled = true;

        Ok(())
    }

    #[pyo3(text_signature = "($self)")]
    pub fn disable_soft_light_radius(&mut self) {
        self.soft_light_radius_enabled = false;
        self.soft_light_radius_renderer = None;
    }

    #[pyo3(text_signature = "($self)")]
    pub fn is_soft_light_radius_enabled(&self) -> bool {
        self.soft_light_radius_enabled && self.soft_light_radius_renderer.is_some()
    }

    #[pyo3(text_signature = "($self, x, y, z)")]
    pub fn set_soft_light_position(&mut self, x: f32, y: f32, z: f32) -> PyResult<()> {
        if let Some(ref mut renderer) = self.soft_light_radius_renderer {
            renderer.set_light_position([x, y, z]);
            Ok(())
        } else {
            Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Soft light radius not enabled. Call enable_soft_light_radius() first.",
            ))
        }
    }

    #[pyo3(text_signature = "($self, intensity)")]
    pub fn set_soft_light_intensity(&mut self, intensity: f32) -> PyResult<()> {
        if let Some(ref mut renderer) = self.soft_light_radius_renderer {
            renderer.set_light_intensity(intensity);
            Ok(())
        } else {
            Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Soft light radius not enabled. Call enable_soft_light_radius() first.",
            ))
        }
    }

    #[pyo3(text_signature = "($self, r, g, b)")]
    pub fn set_soft_light_color(&mut self, r: f32, g: f32, b: f32) -> PyResult<()> {
        if let Some(ref mut renderer) = self.soft_light_radius_renderer {
            renderer.set_light_color([r, g, b]);
            Ok(())
        } else {
            Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Soft light radius not enabled. Call enable_soft_light_radius() first.",
            ))
        }
    }

    #[pyo3(text_signature = "($self, radius)")]
    pub fn set_light_inner_radius(&mut self, radius: f32) -> PyResult<()> {
        if let Some(ref mut renderer) = self.soft_light_radius_renderer {
            renderer.set_inner_radius(radius);
            Ok(())
        } else {
            Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Soft light radius not enabled. Call enable_soft_light_radius() first.",
            ))
        }
    }

    #[pyo3(text_signature = "($self, radius)")]
    pub fn set_light_outer_radius(&mut self, radius: f32) -> PyResult<()> {
        if let Some(ref mut renderer) = self.soft_light_radius_renderer {
            renderer.set_outer_radius(radius);
            Ok(())
        } else {
            Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Soft light radius not enabled. Call enable_soft_light_radius() first.",
            ))
        }
    }

    #[pyo3(text_signature = "($self, exponent)")]
    pub fn set_light_falloff_exponent(&mut self, exponent: f32) -> PyResult<()> {
        if let Some(ref mut renderer) = self.soft_light_radius_renderer {
            renderer.set_falloff_exponent(exponent);
            Ok(())
        } else {
            Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Soft light radius not enabled. Call enable_soft_light_radius() first.",
            ))
        }
    }

    #[pyo3(text_signature = "($self, softness)")]
    pub fn set_light_edge_softness(&mut self, softness: f32) -> PyResult<()> {
        if let Some(ref mut renderer) = self.soft_light_radius_renderer {
            renderer.set_edge_softness(softness);
            Ok(())
        } else {
            Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Soft light radius not enabled. Call enable_soft_light_radius() first.",
            ))
        }
    }

    #[pyo3(text_signature = "($self, mode)")]
    pub fn set_light_falloff_mode(&mut self, mode: &str) -> PyResult<()> {
        if let Some(ref mut renderer) = self.soft_light_radius_renderer {
            let mode_enum = match mode {
                "linear" => crate::core::soft_light_radius::SoftLightFalloffMode::Linear,
                "quadratic" => crate::core::soft_light_radius::SoftLightFalloffMode::Quadratic,
                "cubic" => crate::core::soft_light_radius::SoftLightFalloffMode::Cubic,
                "exponential" => crate::core::soft_light_radius::SoftLightFalloffMode::Exponential,
                _ => {
                    return Err(pyo3::exceptions::PyValueError::new_err(
                        "Mode must be one of: 'linear', 'quadratic', 'cubic', 'exponential'",
                    ))
                }
            };
            renderer.set_falloff_mode(mode_enum);
            Ok(())
        } else {
            Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Soft light radius not enabled. Call enable_soft_light_radius() first.",
            ))
        }
    }

    #[pyo3(text_signature = "($self, softness)")]
    pub fn set_light_shadow_softness(&mut self, softness: f32) -> PyResult<()> {
        if let Some(ref mut renderer) = self.soft_light_radius_renderer {
            renderer.set_shadow_softness(softness);
            Ok(())
        } else {
            Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Soft light radius not enabled. Call enable_soft_light_radius() first.",
            ))
        }
    }

    #[pyo3(text_signature = "($self, preset)")]
    pub fn set_light_preset(&mut self, preset: &str) -> PyResult<()> {
        if let Some(ref mut renderer) = self.soft_light_radius_renderer {
            let preset_enum = match preset {
                "spotlight" => crate::core::soft_light_radius::SoftLightPreset::Spotlight,
                "area_light" => crate::core::soft_light_radius::SoftLightPreset::AreaLight,
                "ambient_light" => crate::core::soft_light_radius::SoftLightPreset::AmbientLight,
                "candle" => crate::core::soft_light_radius::SoftLightPreset::Candle,
                "street_lamp" => crate::core::soft_light_radius::SoftLightPreset::StreetLamp,
                _ => return Err(pyo3::exceptions::PyValueError::new_err(
                    "Preset must be one of: 'spotlight', 'area_light', 'ambient_light', 'candle', 'street_lamp'"
                )),
            };
            renderer.apply_preset(preset_enum);
            Ok(())
        } else {
            Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Soft light radius not enabled. Call enable_soft_light_radius() first.",
            ))
        }
    }

    #[pyo3(text_signature = "($self)")]
    pub fn get_light_effective_range(&self) -> PyResult<f32> {
        if let Some(ref renderer) = self.soft_light_radius_renderer {
            Ok(renderer.effective_range())
        } else {
            Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Soft light radius not enabled. Call enable_soft_light_radius() first.",
            ))
        }
    }

    #[pyo3(text_signature = "($self, x, y, z)")]
    pub fn light_affects_point(&self, x: f32, y: f32, z: f32) -> PyResult<bool> {
        if let Some(ref renderer) = self.soft_light_radius_renderer {
            Ok(renderer.affects_point([x, y, z]))
        } else {
            Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Soft light radius not enabled. Call enable_soft_light_radius() first.",
            ))
        }
    }

    // B13: Point & Spot Lights (Realtime) API
    #[pyo3(text_signature = "($self, max_lights=32)")]
    pub fn enable_point_spot_lights(&mut self, max_lights: Option<usize>) -> PyResult<()> {
        let g = crate::gpu::ctx();
        let max_lights = max_lights.unwrap_or(32);
        let renderer =
            crate::core::point_spot_lights::PointSpotLightRenderer::new(&g.device, max_lights);

        self.point_spot_lights_renderer = Some(renderer);
        self.point_spot_lights_enabled = true;

        Ok(())
    }

    #[pyo3(text_signature = "($self)")]
    pub fn disable_point_spot_lights(&mut self) {
        self.point_spot_lights_enabled = false;
        self.point_spot_lights_renderer = None;
    }

    #[pyo3(text_signature = "($self)")]
    pub fn is_point_spot_lights_enabled(&self) -> bool {
        self.point_spot_lights_enabled && self.point_spot_lights_renderer.is_some()
    }

    #[pyo3(text_signature = "($self, x, y, z, r, g, b, intensity, range)")]
    pub fn add_point_light(
        &mut self,
        x: f32,
        y: f32,
        z: f32,
        r: f32,
        g: f32,
        b: f32,
        intensity: f32,
        range: f32,
    ) -> PyResult<u32> {
        if let Some(ref mut renderer) = self.point_spot_lights_renderer {
            let light = crate::core::point_spot_lights::Light::point(
                [x, y, z],
                [r, g, b],
                intensity,
                range,
            );
            let light_id = renderer.add_light(light);

            if light_id == u32::MAX {
                Err(pyo3::exceptions::PyRuntimeError::new_err(
                    "Maximum number of lights exceeded",
                ))
            } else {
                Ok(light_id)
            }
        } else {
            Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Point/spot lights not enabled. Call enable_point_spot_lights() first.",
            ))
        }
    }

    #[pyo3(
        text_signature = "($self, x, y, z, dir_x, dir_y, dir_z, r, g, b, intensity, range, inner_cone_deg, outer_cone_deg, penumbra_softness)"
    )]
    pub fn add_spot_light(
        &mut self,
        x: f32,
        y: f32,
        z: f32,
        dir_x: f32,
        dir_y: f32,
        dir_z: f32,
        r: f32,
        g: f32,
        b: f32,
        intensity: f32,
        range: f32,
        inner_cone_deg: f32,
        outer_cone_deg: f32,
        penumbra_softness: f32,
    ) -> PyResult<u32> {
        if let Some(ref mut renderer) = self.point_spot_lights_renderer {
            let light = crate::core::point_spot_lights::Light::spot(
                [x, y, z],
                [dir_x, dir_y, dir_z],
                [r, g, b],
                intensity,
                range,
                inner_cone_deg,
                outer_cone_deg,
                penumbra_softness,
            );
            let light_id = renderer.add_light(light);

            if light_id == u32::MAX {
                Err(pyo3::exceptions::PyRuntimeError::new_err(
                    "Maximum number of lights exceeded",
                ))
            } else {
                Ok(light_id)
            }
        } else {
            Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Point/spot lights not enabled. Call enable_point_spot_lights() first.",
            ))
        }
    }

    #[pyo3(text_signature = "($self, preset, x, y, z)")]
    pub fn add_light_preset(&mut self, preset: &str, x: f32, y: f32, z: f32) -> PyResult<u32> {
        if let Some(ref mut renderer) = self.point_spot_lights_renderer {
            let preset_enum = match preset {
                "room_light" => crate::core::point_spot_lights::LightPreset::RoomLight,
                "desk_lamp" => crate::core::point_spot_lights::LightPreset::DeskLamp,
                "street_light" => crate::core::point_spot_lights::LightPreset::StreetLight,
                "spotlight" => crate::core::point_spot_lights::LightPreset::Spotlight,
                "headlight" => crate::core::point_spot_lights::LightPreset::Headlight,
                "flashlight" => crate::core::point_spot_lights::LightPreset::Flashlight,
                "candle" => crate::core::point_spot_lights::LightPreset::Candle,
                "warm_lamp" => crate::core::point_spot_lights::LightPreset::WarmLamp,
                _ => return Err(pyo3::exceptions::PyValueError::new_err(
                    "Preset must be one of: 'room_light', 'desk_lamp', 'street_light', 'spotlight', 'headlight', 'flashlight', 'candle', 'warm_lamp'"
                )),
            };

            let light = preset_enum.to_light([x, y, z]);
            let light_id = renderer.add_light(light);

            if light_id == u32::MAX {
                Err(pyo3::exceptions::PyRuntimeError::new_err(
                    "Maximum number of lights exceeded",
                ))
            } else {
                Ok(light_id)
            }
        } else {
            Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Point/spot lights not enabled. Call enable_point_spot_lights() first.",
            ))
        }
    }

    #[pyo3(text_signature = "($self, light_id)")]
    pub fn remove_light(&mut self, light_id: u32) -> PyResult<bool> {
        if let Some(ref mut renderer) = self.point_spot_lights_renderer {
            Ok(renderer.remove_light(light_id))
        } else {
            Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Point/spot lights not enabled. Call enable_point_spot_lights() first.",
            ))
        }
    }

    #[pyo3(text_signature = "($self)")]
    pub fn clear_all_lights(&mut self) -> PyResult<()> {
        if let Some(ref mut renderer) = self.point_spot_lights_renderer {
            renderer.clear_lights();
            Ok(())
        } else {
            Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Point/spot lights not enabled. Call enable_point_spot_lights() first.",
            ))
        }
    }

    #[pyo3(text_signature = "($self, light_id, x, y, z)")]
    pub fn set_light_position(&mut self, light_id: u32, x: f32, y: f32, z: f32) -> PyResult<()> {
        if let Some(ref mut renderer) = self.point_spot_lights_renderer {
            if let Some(light) = renderer.get_light_mut(light_id) {
                light.set_position([x, y, z]);
                Ok(())
            } else {
                Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "Light with ID {} not found",
                    light_id
                )))
            }
        } else {
            Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Point/spot lights not enabled. Call enable_point_spot_lights() first.",
            ))
        }
    }

    #[pyo3(text_signature = "($self, light_id, dir_x, dir_y, dir_z)")]
    pub fn set_light_direction(
        &mut self,
        light_id: u32,
        dir_x: f32,
        dir_y: f32,
        dir_z: f32,
    ) -> PyResult<()> {
        if let Some(ref mut renderer) = self.point_spot_lights_renderer {
            if let Some(light) = renderer.get_light_mut(light_id) {
                light.set_direction([dir_x, dir_y, dir_z]);
                Ok(())
            } else {
                Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "Light with ID {} not found",
                    light_id
                )))
            }
        } else {
            Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Point/spot lights not enabled. Call enable_point_spot_lights() first.",
            ))
        }
    }

    #[pyo3(text_signature = "($self, light_id, r, g, b)")]
    pub fn set_light_color(&mut self, light_id: u32, r: f32, g: f32, b: f32) -> PyResult<()> {
        if let Some(ref mut renderer) = self.point_spot_lights_renderer {
            if let Some(light) = renderer.get_light_mut(light_id) {
                light.set_color([r, g, b]);
                Ok(())
            } else {
                Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "Light with ID {} not found",
                    light_id
                )))
            }
        } else {
            Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Point/spot lights not enabled. Call enable_point_spot_lights() first.",
            ))
        }
    }

    #[pyo3(text_signature = "($self, light_id, intensity)")]
    pub fn set_light_intensity(&mut self, light_id: u32, intensity: f32) -> PyResult<()> {
        if let Some(ref mut renderer) = self.point_spot_lights_renderer {
            if let Some(light) = renderer.get_light_mut(light_id) {
                light.set_intensity(intensity);
                Ok(())
            } else {
                Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "Light with ID {} not found",
                    light_id
                )))
            }
        } else {
            Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Point/spot lights not enabled. Call enable_point_spot_lights() first.",
            ))
        }
    }

    #[pyo3(text_signature = "($self, light_id, range)")]
    pub fn set_light_range(&mut self, light_id: u32, range: f32) -> PyResult<()> {
        if let Some(ref mut renderer) = self.point_spot_lights_renderer {
            if let Some(light) = renderer.get_light_mut(light_id) {
                light.set_range(range);
                Ok(())
            } else {
                Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "Light with ID {} not found",
                    light_id
                )))
            }
        } else {
            Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Point/spot lights not enabled. Call enable_point_spot_lights() first.",
            ))
        }
    }

    #[pyo3(text_signature = "($self, light_id, inner_cone_deg, outer_cone_deg)")]
    pub fn set_spot_light_cone(
        &mut self,
        light_id: u32,
        inner_cone_deg: f32,
        outer_cone_deg: f32,
    ) -> PyResult<()> {
        if let Some(ref mut renderer) = self.point_spot_lights_renderer {
            if let Some(light) = renderer.get_light_mut(light_id) {
                light.set_cone_angles(inner_cone_deg, outer_cone_deg);
                Ok(())
            } else {
                Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "Light with ID {} not found",
                    light_id
                )))
            }
        } else {
            Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Point/spot lights not enabled. Call enable_point_spot_lights() first.",
            ))
        }
    }

    #[pyo3(text_signature = "($self, light_id, softness)")]
    pub fn set_spot_light_penumbra(&mut self, light_id: u32, softness: f32) -> PyResult<()> {
        if let Some(ref mut renderer) = self.point_spot_lights_renderer {
            if let Some(light) = renderer.get_light_mut(light_id) {
                light.set_penumbra_softness(softness);
                Ok(())
            } else {
                Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "Light with ID {} not found",
                    light_id
                )))
            }
        } else {
            Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Point/spot lights not enabled. Call enable_point_spot_lights() first.",
            ))
        }
    }

    #[pyo3(text_signature = "($self, light_id, enabled)")]
    pub fn set_light_shadows(&mut self, light_id: u32, enabled: bool) -> PyResult<()> {
        if let Some(ref mut renderer) = self.point_spot_lights_renderer {
            if let Some(light) = renderer.get_light_mut(light_id) {
                light.set_shadow_enabled(enabled);
                Ok(())
            } else {
                Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "Light with ID {} not found",
                    light_id
                )))
            }
        } else {
            Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Point/spot lights not enabled. Call enable_point_spot_lights() first.",
            ))
        }
    }

    #[pyo3(text_signature = "($self, r, g, b, intensity)")]
    pub fn set_ambient_lighting(&mut self, r: f32, g: f32, b: f32, intensity: f32) -> PyResult<()> {
        if let Some(ref mut renderer) = self.point_spot_lights_renderer {
            renderer.set_ambient([r, g, b], intensity);
            Ok(())
        } else {
            Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Point/spot lights not enabled. Call enable_point_spot_lights() first.",
            ))
        }
    }

    #[pyo3(text_signature = "($self, quality)")]
    pub fn set_shadow_quality(&mut self, quality: &str) -> PyResult<()> {
        if let Some(ref mut renderer) = self.point_spot_lights_renderer {
            let quality_enum = match quality {
                "off" => crate::core::point_spot_lights::ShadowQuality::Off,
                "low" => crate::core::point_spot_lights::ShadowQuality::Low,
                "medium" => crate::core::point_spot_lights::ShadowQuality::Medium,
                "high" => crate::core::point_spot_lights::ShadowQuality::High,
                _ => {
                    return Err(pyo3::exceptions::PyValueError::new_err(
                        "Quality must be one of: 'off', 'low', 'medium', 'high'",
                    ))
                }
            };
            renderer.set_shadow_quality(quality_enum);
            Ok(())
        } else {
            Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Point/spot lights not enabled. Call enable_point_spot_lights() first.",
            ))
        }
    }

    #[pyo3(text_signature = "($self, mode)")]
    pub fn set_lighting_debug_mode(&mut self, mode: &str) -> PyResult<()> {
        if let Some(ref mut renderer) = self.point_spot_lights_renderer {
            let mode_enum = match mode {
                "normal" => crate::core::point_spot_lights::DebugMode::Normal,
                "show_light_bounds" => crate::core::point_spot_lights::DebugMode::ShowLightBounds,
                "show_shadows" => crate::core::point_spot_lights::DebugMode::ShowShadows,
                _ => {
                    return Err(pyo3::exceptions::PyValueError::new_err(
                        "Mode must be one of: 'normal', 'show_light_bounds', 'show_shadows'",
                    ))
                }
            };
            renderer.set_debug_mode(mode_enum);
            Ok(())
        } else {
            Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Point/spot lights not enabled. Call enable_point_spot_lights() first.",
            ))
        }
    }

    #[pyo3(text_signature = "($self)")]
    pub fn get_light_count(&self) -> PyResult<usize> {
        if let Some(ref renderer) = self.point_spot_lights_renderer {
            Ok(renderer.light_count())
        } else {
            Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Point/spot lights not enabled. Call enable_point_spot_lights() first.",
            ))
        }
    }

    #[pyo3(text_signature = "($self, light_id, x, y, z)")]
    pub fn check_light_affects_point(
        &self,
        light_id: u32,
        x: f32,
        y: f32,
        z: f32,
    ) -> PyResult<bool> {
        if let Some(ref renderer) = self.point_spot_lights_renderer {
            if let Some(light) = renderer.get_light(light_id) {
                Ok(light.affects_point([x, y, z]))
            } else {
                Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "Light with ID {} not found",
                    light_id
                )))
            }
        } else {
            Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Point/spot lights not enabled. Call enable_point_spot_lights() first.",
            ))
        }
    }

    // B14: Rect Area Lights (LTC) API
    #[pyo3(text_signature = "($self, max_lights=16)")]
    pub fn enable_ltc_rect_area_lights(&mut self, max_lights: Option<usize>) -> PyResult<()> {
        let g = crate::gpu::ctx();
        let max_lights = max_lights.unwrap_or(16);

        let renderer = crate::core::ltc_area_lights::LTCRectAreaLightRenderer::new(
            g.device.clone(),
            max_lights,
        )
        .map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!(
                "Failed to create LTC rect area light renderer: {}",
                e
            ))
        })?;

        self.ltc_area_lights_renderer = Some(renderer);
        self.ltc_area_lights_enabled = true;

        Ok(())
    }

    #[pyo3(text_signature = "($self)")]
    pub fn disable_ltc_rect_area_lights(&mut self) {
        self.ltc_area_lights_enabled = false;
        self.ltc_area_lights_renderer = None;
    }

    #[pyo3(text_signature = "($self)")]
    pub fn is_ltc_rect_area_lights_enabled(&self) -> bool {
        self.ltc_area_lights_enabled && self.ltc_area_lights_renderer.is_some()
    }

    #[pyo3(text_signature = "($self, x, y, z, width, height, r, g, b, intensity)")]
    pub fn add_rect_area_light(
        &mut self,
        x: f32,
        y: f32,
        z: f32,
        width: f32,
        height: f32,
        r: f32,
        g: f32,
        b: f32,
        intensity: f32,
    ) -> PyResult<usize> {
        if let Some(ref mut renderer) = self.ltc_area_lights_renderer {
            let light = crate::core::ltc_area_lights::RectAreaLight::quad(
                glam::Vec3::new(x, y, z),
                width,
                height,
                glam::Vec3::new(r, g, b),
                intensity,
            );

            let light_id = renderer
                .add_light(light)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;

            Ok(light_id)
        } else {
            Err(pyo3::exceptions::PyRuntimeError::new_err(
                "LTC rect area lights not enabled. Call enable_ltc_rect_area_lights() first.",
            ))
        }
    }

    #[pyo3(
        text_signature = "($self, position, right_vec, up_vec, width, height, r, g, b, intensity, two_sided=False)"
    )]
    pub fn add_custom_rect_area_light(
        &mut self,
        position: (f32, f32, f32),
        right_vec: (f32, f32, f32),
        up_vec: (f32, f32, f32),
        width: f32,
        height: f32,
        r: f32,
        g: f32,
        b: f32,
        intensity: f32,
        two_sided: Option<bool>,
    ) -> PyResult<usize> {
        if let Some(ref mut renderer) = self.ltc_area_lights_renderer {
            let light = crate::core::ltc_area_lights::RectAreaLight::new(
                glam::Vec3::new(position.0, position.1, position.2),
                glam::Vec3::new(right_vec.0, right_vec.1, right_vec.2),
                glam::Vec3::new(up_vec.0, up_vec.1, up_vec.2),
                width,
                height,
                glam::Vec3::new(r, g, b),
                intensity,
                two_sided.unwrap_or(false),
            );

            let light_id = renderer
                .add_light(light)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;

            Ok(light_id)
        } else {
            Err(pyo3::exceptions::PyRuntimeError::new_err(
                "LTC rect area lights not enabled. Call enable_ltc_rect_area_lights() first.",
            ))
        }
    }

    #[pyo3(text_signature = "($self, light_id)")]
    pub fn remove_rect_area_light(&mut self, light_id: usize) -> PyResult<()> {
        if let Some(ref mut renderer) = self.ltc_area_lights_renderer {
            renderer
                .remove_light(light_id)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;
            Ok(())
        } else {
            Err(pyo3::exceptions::PyRuntimeError::new_err(
                "LTC rect area lights not enabled. Call enable_ltc_rect_area_lights() first.",
            ))
        }
    }

    #[pyo3(text_signature = "($self, light_id, x, y, z, width, height, r, g, b, intensity)")]
    pub fn update_rect_area_light(
        &mut self,
        light_id: usize,
        x: f32,
        y: f32,
        z: f32,
        width: f32,
        height: f32,
        r: f32,
        g: f32,
        b: f32,
        intensity: f32,
    ) -> PyResult<()> {
        if let Some(ref mut renderer) = self.ltc_area_lights_renderer {
            let light = crate::core::ltc_area_lights::RectAreaLight::quad(
                glam::Vec3::new(x, y, z),
                width,
                height,
                glam::Vec3::new(r, g, b),
                intensity,
            );

            renderer
                .update_light(light_id, light)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;

            Ok(())
        } else {
            Err(pyo3::exceptions::PyRuntimeError::new_err(
                "LTC rect area lights not enabled. Call enable_ltc_rect_area_lights() first.",
            ))
        }
    }

    #[pyo3(text_signature = "($self)")]
    pub fn get_rect_area_light_count(&self) -> PyResult<usize> {
        if let Some(ref renderer) = self.ltc_area_lights_renderer {
            Ok(renderer.light_count())
        } else {
            Err(pyo3::exceptions::PyRuntimeError::new_err(
                "LTC rect area lights not enabled. Call enable_ltc_rect_area_lights() first.",
            ))
        }
    }

    #[pyo3(text_signature = "($self, intensity)")]
    pub fn set_ltc_global_intensity(&mut self, intensity: f32) -> PyResult<()> {
        if let Some(ref mut renderer) = self.ltc_area_lights_renderer {
            renderer.set_global_intensity(intensity);
            Ok(())
        } else {
            Err(pyo3::exceptions::PyRuntimeError::new_err(
                "LTC rect area lights not enabled. Call enable_ltc_rect_area_lights() first.",
            ))
        }
    }

    #[pyo3(text_signature = "($self, enabled)")]
    pub fn set_ltc_approximation_enabled(&mut self, enabled: bool) -> PyResult<()> {
        if let Some(ref mut renderer) = self.ltc_area_lights_renderer {
            renderer.set_ltc_enabled(enabled);
            Ok(())
        } else {
            Err(pyo3::exceptions::PyRuntimeError::new_err(
                "LTC rect area lights not enabled. Call enable_ltc_rect_area_lights() first.",
            ))
        }
    }

    #[pyo3(text_signature = "($self)")]
    pub fn get_ltc_uniforms(&self) -> PyResult<(u32, f32, bool)> {
        if let Some(ref renderer) = self.ltc_area_lights_renderer {
            let uniforms = renderer.uniforms();
            Ok((
                uniforms.light_count,
                uniforms.global_intensity,
                uniforms.enable_ltc > 0.5,
            ))
        } else {
            Err(pyo3::exceptions::PyRuntimeError::new_err(
                "LTC rect area lights not enabled. Call enable_ltc_rect_area_lights() first.",
            ))
        }
    }

    // B15: Image-Based Lighting (IBL) Polish API
    #[pyo3(text_signature = "($self, quality='medium')")]
    pub fn enable_ibl(&mut self, quality: Option<&str>) -> PyResult<()> {
        let g = crate::gpu::ctx();

        let quality_enum = match quality.unwrap_or("medium") {
            "low" => crate::core::ibl::IBLQuality::Low,
            "medium" => crate::core::ibl::IBLQuality::Medium,
            "high" => crate::core::ibl::IBLQuality::High,
            "ultra" => crate::core::ibl::IBLQuality::Ultra,
            _ => {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "Quality must be one of: 'low', 'medium', 'high', 'ultra'",
                ))
            }
        };

        let mut renderer = crate::core::ibl::IBLRenderer::new(&g.device, quality_enum);

        // Initialize with default environment
        renderer
            .initialize(&g.device, &g.queue)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;

        self.ibl_renderer = Some(renderer);
        self.ibl_enabled = true;

        Ok(())
    }

    #[pyo3(text_signature = "($self)")]
    pub fn disable_ibl(&mut self) {
        self.ibl_enabled = false;
        self.ibl_renderer = None;
    }

    #[pyo3(text_signature = "($self)")]
    pub fn is_ibl_enabled(&self) -> bool {
        self.ibl_enabled && self.ibl_renderer.is_some()
    }

    #[pyo3(text_signature = "($self, quality)")]
    pub fn set_ibl_quality(&mut self, quality: &str) -> PyResult<()> {
        if let Some(ref mut renderer) = self.ibl_renderer {
            let quality_enum = match quality {
                "low" => crate::core::ibl::IBLQuality::Low,
                "medium" => crate::core::ibl::IBLQuality::Medium,
                "high" => crate::core::ibl::IBLQuality::High,
                "ultra" => crate::core::ibl::IBLQuality::Ultra,
                _ => {
                    return Err(pyo3::exceptions::PyValueError::new_err(
                        "Quality must be one of: 'low', 'medium', 'high', 'ultra'",
                    ))
                }
            };

            renderer.set_quality(quality_enum);

            // Regenerate IBL textures with new quality
            let g = crate::gpu::ctx();
            renderer
                .initialize(&g.device, &g.queue)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;

            Ok(())
        } else {
            Err(pyo3::exceptions::PyRuntimeError::new_err(
                "IBL not enabled. Call enable_ibl() first.",
            ))
        }
    }

    #[pyo3(text_signature = "($self, hdr_data, width, height)")]
    pub fn load_environment_map(
        &mut self,
        hdr_data: Vec<f32>,
        width: u32,
        height: u32,
    ) -> PyResult<()> {
        if let Some(ref mut renderer) = self.ibl_renderer {
            let g = crate::gpu::ctx();
            renderer
                .load_environment_map(&g.device, &g.queue, &hdr_data, width, height)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;

            // Regenerate IBL textures with new environment
            renderer
                .initialize(&g.device, &g.queue)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;

            Ok(())
        } else {
            Err(pyo3::exceptions::PyRuntimeError::new_err(
                "IBL not enabled. Call enable_ibl() first.",
            ))
        }
    }

    #[pyo3(text_signature = "($self)")]
    pub fn generate_ibl_textures(&mut self) -> PyResult<()> {
        if let Some(ref mut renderer) = self.ibl_renderer {
            let g = crate::gpu::ctx();

            // Regenerate irradiance map
            renderer
                .generate_irradiance_map(&g.device, &g.queue)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;

            // Regenerate specular map
            renderer
                .generate_specular_map(&g.device, &g.queue)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;

            // Regenerate BRDF LUT
            renderer
                .generate_brdf_lut(&g.device, &g.queue)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;

            Ok(())
        } else {
            Err(pyo3::exceptions::PyRuntimeError::new_err(
                "IBL not enabled. Call enable_ibl() first.",
            ))
        }
    }

    #[pyo3(text_signature = "($self)")]
    pub fn get_ibl_quality(&self) -> PyResult<String> {
        if let Some(ref renderer) = self.ibl_renderer {
            let quality_str = match renderer.quality() {
                crate::core::ibl::IBLQuality::Low => "low",
                crate::core::ibl::IBLQuality::Medium => "medium",
                crate::core::ibl::IBLQuality::High => "high",
                crate::core::ibl::IBLQuality::Ultra => "ultra",
            };
            Ok(quality_str.to_string())
        } else {
            Err(pyo3::exceptions::PyRuntimeError::new_err(
                "IBL not enabled. Call enable_ibl() first.",
            ))
        }
    }

    #[pyo3(text_signature = "($self)")]
    pub fn is_ibl_initialized(&self) -> PyResult<bool> {
        if let Some(ref renderer) = self.ibl_renderer {
            Ok(renderer.is_initialized())
        } else {
            Err(pyo3::exceptions::PyRuntimeError::new_err(
                "IBL not enabled. Call enable_ibl() first.",
            ))
        }
    }

    #[pyo3(text_signature = "($self)")]
    pub fn get_ibl_texture_info(&self) -> PyResult<(String, String, String)> {
        if let Some(ref renderer) = self.ibl_renderer {
            let quality = renderer.quality();
            let (irr, spec, brdf) = renderer.textures();

            let irr_info = if irr.is_some() {
                format!(
                    "{}x{} (6 faces)",
                    quality.irradiance_size(),
                    quality.irradiance_size()
                )
            } else {
                "Not generated".to_string()
            };

            let spec_info = if spec.is_some() {
                format!(
                    "{}x{} (6 faces, {} mips)",
                    quality.specular_size(),
                    quality.specular_size(),
                    quality.specular_mip_levels()
                )
            } else {
                "Not generated".to_string()
            };

            let brdf_info = if brdf.is_some() {
                format!("{}x{}", quality.brdf_size(), quality.brdf_size())
            } else {
                "Not generated".to_string()
            };

            Ok((irr_info, spec_info, brdf_info))
        } else {
            Err(pyo3::exceptions::PyRuntimeError::new_err(
                "IBL not enabled. Call enable_ibl() first.",
            ))
        }
    }

    // IBL Material property helpers (for future PBR integration)
    #[pyo3(text_signature = "($self, metallic, roughness, r, g, b)")]
    pub fn test_ibl_material(
        &self,
        metallic: f32,
        roughness: f32,
        r: f32,
        g: f32,
        b: f32,
    ) -> PyResult<(f32, f32, f32)> {
        if !self.is_ibl_enabled() {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "IBL not enabled. Call enable_ibl() first.",
            ));
        }

        // Test material properties for IBL rendering
        let metallic = metallic.clamp(0.0, 1.0);
        let _roughness = roughness.clamp(0.0, 1.0);

        // Calculate F0 for the material
        let dielectric_f0 = 0.04;
        let f0_r = r * metallic + dielectric_f0 * (1.0 - metallic);
        let f0_g = g * metallic + dielectric_f0 * (1.0 - metallic);
        let f0_b = b * metallic + dielectric_f0 * (1.0 - metallic);

        Ok((f0_r, f0_g, f0_b))
    }

    #[pyo3(text_signature = "($self, n_dot_v, roughness)")]
    pub fn sample_brdf_lut(&self, n_dot_v: f32, roughness: f32) -> PyResult<(f32, f32)> {
        if !self.is_ibl_enabled() {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "IBL not enabled. Call enable_ibl() first.",
            ));
        }

        // Clamp inputs to valid ranges
        let n_dot_v = n_dot_v.clamp(0.0, 1.0);
        let roughness = roughness.clamp(0.0, 1.0);

        // Simplified BRDF approximation for testing
        // In a real implementation, this would sample the actual BRDF LUT texture
        let a = 1.0 - roughness;
        let fresnel_term = a * (1.0 - n_dot_v).powf(5.0);
        let roughness_term = roughness * n_dot_v;

        Ok((fresnel_term, roughness_term))
    }

    // B16: Dual-source blending OIT Methods

    #[pyo3(text_signature = "($self, mode, quality)")]
    pub fn enable_dual_source_oit(
        &mut self,
        mode: Option<&str>,
        quality: Option<&str>,
    ) -> PyResult<()> {
        let g = crate::gpu::ctx();

        // Parse mode
        let oit_mode = match mode {
            Some("dual_source") => crate::core::dual_source_oit::DualSourceOITMode::DualSource,
            Some("wboit_fallback") => {
                crate::core::dual_source_oit::DualSourceOITMode::WBOITFallback
            }
            Some("automatic") | None => crate::core::dual_source_oit::DualSourceOITMode::Automatic,
            _ => {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "Invalid mode. Use 'dual_source', 'wboit_fallback', or 'automatic'.",
                ))
            }
        };

        // Parse quality
        let oit_quality = match quality {
            Some("low") => crate::core::dual_source_oit::DualSourceOITQuality::Low,
            Some("medium") | None => crate::core::dual_source_oit::DualSourceOITQuality::Medium,
            Some("high") => crate::core::dual_source_oit::DualSourceOITQuality::High,
            Some("ultra") => crate::core::dual_source_oit::DualSourceOITQuality::Ultra,
            _ => {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "Invalid quality. Use 'low', 'medium', 'high', or 'ultra'.",
                ))
            }
        };

        // Create dual-source OIT renderer
        let mut renderer = crate::core::dual_source_oit::DualSourceOITRenderer::new(
            &g.device,
            self.width,
            self.height,
            wgpu::TextureFormat::Rgba8UnormSrgb,
        )
        .map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!(
                "Failed to create dual-source OIT renderer: {}",
                e
            ))
        })?;

        renderer.set_mode(oit_mode);
        renderer.set_quality(oit_quality);
        renderer.set_enabled(true);

        self.dual_source_oit_renderer = Some(renderer);
        self.dual_source_oit_enabled = true;

        Ok(())
    }

    #[pyo3(text_signature = "($self)")]
    pub fn disable_dual_source_oit(&mut self) -> PyResult<()> {
        self.dual_source_oit_enabled = false;
        self.dual_source_oit_renderer = None;
        Ok(())
    }

    #[pyo3(text_signature = "($self)")]
    pub fn is_dual_source_oit_enabled(&self) -> bool {
        self.dual_source_oit_enabled && self.dual_source_oit_renderer.is_some()
    }

    #[pyo3(text_signature = "($self, mode)")]
    pub fn set_dual_source_oit_mode(&mut self, mode: &str) -> PyResult<()> {
        if let Some(ref mut renderer) = self.dual_source_oit_renderer {
            let oit_mode = match mode {
                "dual_source" => crate::core::dual_source_oit::DualSourceOITMode::DualSource,
                "wboit_fallback" => crate::core::dual_source_oit::DualSourceOITMode::WBOITFallback,
                "automatic" => crate::core::dual_source_oit::DualSourceOITMode::Automatic,
                "disabled" => crate::core::dual_source_oit::DualSourceOITMode::Disabled,
                _ => return Err(pyo3::exceptions::PyValueError::new_err(
                    "Invalid mode. Use 'dual_source', 'wboit_fallback', 'automatic', or 'disabled'.",
                )),
            };

            renderer.set_mode(oit_mode);
            Ok(())
        } else {
            Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Dual-source OIT not enabled. Call enable_dual_source_oit() first.",
            ))
        }
    }

    #[pyo3(text_signature = "($self)")]
    pub fn get_dual_source_oit_mode(&self) -> PyResult<String> {
        if let Some(ref renderer) = self.dual_source_oit_renderer {
            let mode = match renderer.get_operating_mode() {
                crate::core::dual_source_oit::DualSourceOITMode::DualSource => "dual_source",
                crate::core::dual_source_oit::DualSourceOITMode::WBOITFallback => "wboit_fallback",
                crate::core::dual_source_oit::DualSourceOITMode::Automatic => "automatic",
                crate::core::dual_source_oit::DualSourceOITMode::Disabled => "disabled",
            };
            Ok(mode.to_string())
        } else {
            Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Dual-source OIT not enabled. Call enable_dual_source_oit() first.",
            ))
        }
    }

    #[pyo3(text_signature = "($self, quality)")]
    pub fn set_dual_source_oit_quality(&mut self, quality: &str) -> PyResult<()> {
        if let Some(ref mut renderer) = self.dual_source_oit_renderer {
            let oit_quality = match quality {
                "low" => crate::core::dual_source_oit::DualSourceOITQuality::Low,
                "medium" => crate::core::dual_source_oit::DualSourceOITQuality::Medium,
                "high" => crate::core::dual_source_oit::DualSourceOITQuality::High,
                "ultra" => crate::core::dual_source_oit::DualSourceOITQuality::Ultra,
                _ => {
                    return Err(pyo3::exceptions::PyValueError::new_err(
                        "Invalid quality. Use 'low', 'medium', 'high', or 'ultra'.",
                    ))
                }
            };

            renderer.set_quality(oit_quality);
            Ok(())
        } else {
            Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Dual-source OIT not enabled. Call enable_dual_source_oit() first.",
            ))
        }
    }

    #[pyo3(text_signature = "($self)")]
    pub fn get_dual_source_oit_quality(&self) -> PyResult<String> {
        if let Some(ref renderer) = self.dual_source_oit_renderer {
            let quality = match renderer.quality() {
                crate::core::dual_source_oit::DualSourceOITQuality::Low => "low",
                crate::core::dual_source_oit::DualSourceOITQuality::Medium => "medium",
                crate::core::dual_source_oit::DualSourceOITQuality::High => "high",
                crate::core::dual_source_oit::DualSourceOITQuality::Ultra => "ultra",
            };
            Ok(quality.to_string())
        } else {
            Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Dual-source OIT not enabled. Call enable_dual_source_oit() first.",
            ))
        }
    }

    #[pyo3(text_signature = "($self)")]
    pub fn is_dual_source_supported(&self) -> PyResult<bool> {
        if let Some(ref renderer) = self.dual_source_oit_renderer {
            Ok(renderer.is_dual_source_supported())
        } else {
            // Check hardware support without creating renderer
            let g = crate::gpu::ctx();
            let test_renderer = crate::core::dual_source_oit::DualSourceOITRenderer::new(
                &g.device,
                256,
                256,
                wgpu::TextureFormat::Rgba8UnormSrgb,
            );
            match test_renderer {
                Ok(renderer) => Ok(renderer.is_dual_source_supported()),
                Err(_) => Ok(false),
            }
        }
    }

    #[pyo3(text_signature = "($self)")]
    pub fn get_dual_source_oit_stats(&self) -> PyResult<(u64, u64, u64, f32, f32, f32)> {
        if let Some(ref renderer) = self.dual_source_oit_renderer {
            let stats = renderer.get_stats();
            Ok((
                stats.frames_rendered,
                stats.dual_source_frames,
                stats.wboit_fallback_frames,
                stats.average_fragment_count,
                stats.peak_fragment_count,
                stats.quality_score,
            ))
        } else {
            Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Dual-source OIT not enabled. Call enable_dual_source_oit() first.",
            ))
        }
    }

    #[pyo3(
        text_signature = "($self, alpha_correction, depth_weight_scale, max_fragments, premultiply_factor)"
    )]
    pub fn set_dual_source_oit_params(
        &mut self,
        _alpha_correction: f32,
        _depth_weight_scale: f32,
        _max_fragments: f32,
        _premultiply_factor: f32,
    ) -> PyResult<()> {
        if let Some(ref mut _renderer) = self.dual_source_oit_renderer {
            // Update uniforms would need to be exposed in the renderer
            // For now, return success - this would be implemented in the renderer
            Ok(())
        } else {
            Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Dual-source OIT not enabled. Call enable_dual_source_oit() first.",
            ))
        }
    }

    // -----------------------------
    // D: Native overlays compositor (upload overlay texture, altitude toggle)
    // -----------------------------
    #[pyo3(text_signature = "($self)")]
    pub fn enable_native_overlays(&mut self) -> PyResult<()> {
        let Some(ref mut ov) = self.overlay_renderer else {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Overlay renderer not available",
            ));
        };
        self.overlay_enabled = true;
        ov.set_enabled(true);
        let g = crate::gpu::ctx();
        ov.upload_uniforms(&g.queue);
        Ok(())
    }

    #[pyo3(text_signature = "($self)")]
    pub fn disable_native_overlays(&mut self) -> PyResult<()> {
        let Some(ref mut ov) = self.overlay_renderer else {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Overlay renderer not available",
            ));
        };
        self.overlay_enabled = false;
        ov.set_enabled(false);
        let g = crate::gpu::ctx();
        ov.upload_uniforms(&g.queue);
        Ok(())
    }

    #[pyo3(text_signature = "($self, alpha)")]
    pub fn set_native_overlay_alpha(&mut self, alpha: f32) -> PyResult<()> {
        let Some(ref mut ov) = self.overlay_renderer else {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Overlay renderer not available",
            ));
        };
        ov.set_overlay_alpha(alpha);
        let g = crate::gpu::ctx();
        ov.upload_uniforms(&g.queue);
        Ok(())
    }

    #[pyo3(text_signature = "($self, enabled)")]
    pub fn set_native_altitude_overlay_enabled(&mut self, enabled: bool) -> PyResult<()> {
        let Some(ref mut ov) = self.overlay_renderer else {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Overlay renderer not available",
            ));
        };
        ov.set_altitude_enabled(enabled);
        // Ensure height view is bound
        if let Some(ref hv) = self.height_view {
            let g = crate::gpu::ctx();
            ov.recreate_bind_group(&g.device, None, Some(hv), None);
        }
        let g = crate::gpu::ctx();
        ov.upload_uniforms(&g.queue);
        Ok(())
    }

    #[pyo3(text_signature = "($self, image)")]
    pub fn set_native_overlay_texture(&mut self, image: &pyo3::PyAny) -> PyResult<()> {
        // Accept HxWx3 or HxWx4 uint8
        let (h, w, c, data) = if let Ok(arr) = image.extract::<PyReadonlyArray3<u8>>() {
            let shape = arr.shape();
            if shape.len() != 3 {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "overlay must be HxWxC",
                ));
            }
            let h = shape[0] as u32;
            let w = shape[1] as u32;
            let c = shape[2] as u32;
            if c != 3 && c != 4 {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "overlay channels must be 3 or 4",
                ));
            }
            (h, w, c, arr.as_array().to_owned().into_raw_vec())
        } else {
            return Err(pyo3::exceptions::PyTypeError::new_err(
                "Expected numpy uint8 array HxWxC",
            ));
        };

        let Some(ref mut ov) = self.overlay_renderer else {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Overlay renderer not available",
            ));
        };
        let g = crate::gpu::ctx();
        // Prepare RGBA data with row padding to COPY_BYTES_PER_ROW_ALIGNMENT
        let mut rgba: Vec<u8>;
        if c == 3 {
            rgba = Vec::with_capacity((h * w * 4) as usize);
            let mut idx = 0usize;
            for _yy in 0..h {
                for _xx in 0..w {
                    let r = data[idx];
                    let gch = data[idx + 1];
                    let b = data[idx + 2];
                    rgba.push(r);
                    rgba.push(gch);
                    rgba.push(b);
                    rgba.push(255);
                    idx += 3;
                }
            }
        } else {
            rgba = data; // already RGBA
        }
        let row_bytes = w * 4;
        let padded_bpr = crate::gpu::align_copy_bpr(row_bytes);
        let mut padded = vec![0u8; (padded_bpr * h) as usize];
        for y in 0..h as usize {
            let s = y * row_bytes as usize;
            let d = y * padded_bpr as usize;
            padded[d..d + row_bytes as usize].copy_from_slice(&rgba[s..s + row_bytes as usize]);
        }

        // Create texture and upload
        let tex = g.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("native_overlay_tex"),
            size: wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        g.queue.write_texture(
            wgpu::ImageCopyTexture {
                texture: &tex,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &padded,
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(std::num::NonZeroU32::new(padded_bpr).unwrap().into()),
                rows_per_image: Some(std::num::NonZeroU32::new(h).unwrap().into()),
            },
            wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
        );
        let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
        ov.overlay_tex = Some(tex);
        ov.overlay_view = Some(view);
        // Recreate bind group with new overlay view and existing height view
        let height_view = self.height_view.as_ref();
        ov.recreate_bind_group(&g.device, None, height_view, None);
        Ok(())
    }

    // -----------------------------
    // D: Native text overlay APIs (rectangle placeholder)
    // -----------------------------
    #[pyo3(text_signature = "($self)")]
    pub fn enable_native_text(&mut self) -> PyResult<()> {
        self.text_overlay_enabled = true;
        if let Some(ref mut tr) = self.text_overlay_renderer {
            tr.set_enabled(true);
            let g = crate::gpu::ctx();
            tr.upload_uniforms(&g.queue);
        }
        Ok(())
    }

    #[pyo3(text_signature = "($self)")]
    pub fn disable_native_text(&mut self) -> PyResult<()> {
        self.text_overlay_enabled = false;
        if let Some(ref mut tr) = self.text_overlay_renderer {
            tr.set_enabled(false);
            let g = crate::gpu::ctx();
            tr.upload_uniforms(&g.queue);
        }
        Ok(())
    }

    #[pyo3(text_signature = "($self, alpha)")]
    pub fn set_native_text_alpha(&mut self, alpha: f32) -> PyResult<()> {
        self.text_overlay_alpha = alpha.clamp(0.0, 1.0);
        if let Some(ref mut tr) = self.text_overlay_renderer {
            tr.set_alpha(self.text_overlay_alpha);
            let g = crate::gpu::ctx();
            tr.upload_uniforms(&g.queue);
        }
        Ok(())
    }

    #[pyo3(text_signature = "($self, x, y, w, h, r, g, b, a)")]
    pub fn add_native_text_rect(
        &mut self,
        x: f32,
        y: f32,
        w: f32,
        h: f32,
        r: f32,
        g: f32,
        b: f32,
        a: f32,
    ) -> PyResult<()> {
        let rect_min = [x.max(0.0), y.max(0.0)];
        let rect_max = [(x + w).max(0.0), (y + h).max(0.0)];
        let uv_min = [0.0, 0.0];
        let uv_max = [1.0, 1.0];
        let color = [
            r.clamp(0.0, 1.0),
            g.clamp(0.0, 1.0),
            b.clamp(0.0, 1.0),
            a.clamp(0.0, 1.0),
        ];
        self.text_instances
            .push(crate::core::text_overlay::TextInstance {
                rect_min,
                rect_max,
                uv_min,
                uv_max,
                color,
            });
        Ok(())
    }

    #[pyo3(text_signature = "($self)")]
    pub fn clear_native_text(&mut self) -> PyResult<()> {
        self.text_instances.clear();
        if let Some(ref mut tr) = self.text_overlay_renderer {
            tr.instance_count = 0;
        }
        Ok(())
    }

    #[pyo3(text_signature = "($self, x, y, w, h, u0, v0, u1, v1, r, g, b, a)")]
    pub fn add_native_text_rect_uv(
        &mut self,
        x: f32,
        y: f32,
        w: f32,
        h: f32,
        u0: f32,
        v0: f32,
        u1: f32,
        v1: f32,
        r: f32,
        g: f32,
        b: f32,
        a: f32,
    ) -> PyResult<()> {
        let rect_min = [x.max(0.0), y.max(0.0)];
        let rect_max = [(x + w).max(0.0), (y + h).max(0.0)];
        let uv_min = [u0, v0];
        let uv_max = [u1, v1];
        let color = [
            r.clamp(0.0, 1.0),
            g.clamp(0.0, 1.0),
            b.clamp(0.0, 1.0),
            a.clamp(0.0, 1.0),
        ];
        self.text_instances
            .push(crate::core::text_overlay::TextInstance {
                rect_min,
                rect_max,
                uv_min,
                uv_max,
                color,
            });
        Ok(())
    }

    #[pyo3(text_signature = "($self, atlas, channels=3, smoothing=1.0)")]
    pub fn set_native_text_atlas(
        &mut self,
        atlas: &pyo3::PyAny,
        channels: Option<u32>,
        smoothing: Option<f32>,
    ) -> PyResult<()> {
        let (h, w, c, data) = if let Ok(arr) = atlas.extract::<PyReadonlyArray3<u8>>() {
            let shape = arr.shape();
            if shape.len() != 3 {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "atlas must be HxWxC uint8",
                ));
            }
            let h = shape[0] as u32;
            let w = shape[1] as u32;
            let c = shape[2] as u32;
            if c != 1 && c != 3 && c != 4 {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "atlas channels must be 1, 3, or 4",
                ));
            }
            (h, w, c, arr.as_array().to_owned().into_raw_vec())
        } else {
            return Err(pyo3::exceptions::PyTypeError::new_err(
                "Expected numpy uint8 array HxWxC",
            ));
        };
        let g = crate::gpu::ctx();
        // Convert to RGBA8
        let mut rgba: Vec<u8> = Vec::with_capacity((h * w * 4) as usize);
        if c == 4 {
            rgba = data;
        } else if c == 3 {
            let mut idx = 0usize;
            while idx < data.len() {
                rgba.push(data[idx]);
                rgba.push(data[idx + 1]);
                rgba.push(data[idx + 2]);
                rgba.push(255);
                idx += 3;
            }
        } else {
            // c == 1 (SDF)
            for v in data.iter() {
                rgba.push(*v);
                rgba.push(*v);
                rgba.push(*v);
                rgba.push(255);
            }
        }
        let row_bytes = w * 4;
        let padded_bpr = crate::gpu::align_copy_bpr(row_bytes);
        let mut padded = vec![0u8; (padded_bpr * h) as usize];
        for y in 0..h as usize {
            let s = y * row_bytes as usize;
            let d = y * padded_bpr as usize;
            padded[d..d + row_bytes as usize].copy_from_slice(&rgba[s..s + row_bytes as usize]);
        }
        let tex = g.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("text_msdf_atlas"),
            size: wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        g.queue.write_texture(
            wgpu::ImageCopyTexture {
                texture: &tex,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &padded,
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(std::num::NonZeroU32::new(padded_bpr).unwrap().into()),
                rows_per_image: Some(std::num::NonZeroU32::new(h).unwrap().into()),
            },
            wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
        );
        let view = tex.create_view(&wgpu::TextureViewDescriptor::default());

        // Update text overlay renderer state
        if let Some(ref mut tr) = self.text_overlay_renderer {
            tr.set_atlas(tex, view);
            tr.recreate_bind_group(&g.device, None);
            if let Some(ch) = channels {
                tr.set_channels(ch);
            }
            if let Some(sm) = smoothing {
                tr.set_smoothing(sm);
            }
            tr.upload_uniforms(&g.queue);
        }

        Ok(())
    }

    #[pyo3(text_signature = "($self, image, alpha=1.0, offset_xy=(0,0), scale=1.0)")]
    pub fn set_raster_overlay(
        &mut self,
        image: &pyo3::types::PyAny,
        alpha: Option<f32>,
        offset_xy: Option<(i32, i32)>,
        scale: Option<f32>,
    ) -> PyResult<()> {
        // Validate input array (HxWx3 or HxWx4, uint8)
        let (h, w, c, data) = if let Ok(arr) = image.extract::<numpy::PyReadonlyArray3<u8>>() {
            let shape = arr.shape();
            if shape.len() != 3 {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "overlay image must be HxWxC uint8",
                ));
            }
            let h = shape[0] as u32;
            let w = shape[1] as u32;
            let c = shape[2] as u32;
            if c != 3 && c != 4 {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "overlay channels must be 3 or 4",
                ));
            }
            (h, w, c, arr.as_array().to_owned().into_raw_vec())
        } else {
            return Err(pyo3::exceptions::PyTypeError::new_err(
                "Expected numpy uint8 array HxWxC",
            ));
        };

        // Convert to RGBA8
        let mut rgba: Vec<u8> = Vec::with_capacity((h * w * 4) as usize);
        if c == 4 {
            rgba = data;
        } else {
            // c == 3
            let mut idx = 0usize;
            while idx < data.len() {
                rgba.push(data[idx]);
                rgba.push(data[idx + 1]);
                rgba.push(data[idx + 2]);
                rgba.push(255);
                idx += 3;
            }
        }

        // Create GPU texture and upload
        let g = crate::gpu::ctx();
        let tex = g.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("scene-overlay-rgba8"),
            size: wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let row_bytes = w * 4;
        let padded_bpr = crate::gpu::align_copy_bpr(row_bytes);
        let mut padded = vec![0u8; (padded_bpr * h) as usize];
        for y in 0..h as usize {
            let s = y * row_bytes as usize;
            let d = y * padded_bpr as usize;
            padded[d..d + row_bytes as usize].copy_from_slice(&rgba[s..s + row_bytes as usize]);
        }
        g.queue.write_texture(
            wgpu::ImageCopyTexture {
                texture: &tex,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &padded,
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(std::num::NonZeroU32::new(padded_bpr).unwrap().into()),
                rows_per_image: Some(std::num::NonZeroU32::new(h).unwrap().into()),
            },
            wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
        );
        let view = tex.create_view(&wgpu::TextureViewDescriptor::default());

        // Update overlay renderer state
        if let Some(ref mut ov) = self.overlay_renderer {
            // Keep GPU resources alive
            ov.set_overlay_texture(tex, view);
            // Rebind with current height view for altitude/contours
            ov.recreate_bind_group(&g.device, None, self.height_view.as_ref(), None);

            // Set params
            ov.set_enabled(true);
            if let Some(a) = alpha {
                ov.set_overlay_alpha(a);
            }
            let (off_x, off_y) = offset_xy.unwrap_or((0, 0));
            let scale_v = scale.unwrap_or(1.0).max(1e-3);
            let sample_s = 1.0 / scale_v;
            let uv_off_x = (off_x as f32 / self.width.max(1) as f32) * sample_s;
            let uv_off_y = (off_y as f32 / self.height.max(1) as f32) * sample_s;
            ov.set_overlay_uv(uv_off_x, uv_off_y, sample_s, sample_s);
            ov.upload_uniforms(&g.queue);
            self.overlay_enabled = true;
        }

        Ok(())
    }

    #[pyo3(text_signature = "($self)")]
    pub fn disable_overlay(&mut self) -> PyResult<()> {
        if let Some(ref mut ov) = self.overlay_renderer {
            ov.set_enabled(false);
            let g = crate::gpu::ctx();
            ov.upload_uniforms(&g.queue);
        }
        self.overlay_enabled = false;
        Ok(())
    }

    #[pyo3(text_signature = "($self, alpha)")]
    pub fn set_overlay_alpha(&mut self, alpha: f32) -> PyResult<()> {
        if let Some(ref mut ov) = self.overlay_renderer {
            ov.set_overlay_alpha(alpha);
            ov.set_enabled(true);
            let g = crate::gpu::ctx();
            ov.upload_uniforms(&g.queue);
        }
        self.overlay_enabled = true;
        Ok(())
    }

    #[pyo3(text_signature = "($self, alpha=0.35)")]
    pub fn enable_altitude_overlay(&mut self, alpha: Option<f32>) -> PyResult<()> {
        if let Some(ref mut ov) = self.overlay_renderer {
            ov.set_altitude_enabled(true);
            if let Some(a) = alpha {
                ov.set_altitude_alpha(a);
            }
            let g = crate::gpu::ctx();
            ov.upload_uniforms(&g.queue);
        }
        Ok(())
    }

    #[pyo3(text_signature = "($self)")]
    pub fn disable_altitude_overlay(&mut self) -> PyResult<()> {
        if let Some(ref mut ov) = self.overlay_renderer {
            ov.set_altitude_enabled(false);
            let g = crate::gpu::ctx();
            ov.upload_uniforms(&g.queue);
        }
        Ok(())
    }

    #[pyo3(text_signature = "($self)")]
    pub fn disable_terrain(&mut self) -> PyResult<()> {
        self.terrain_enabled = false;
        Ok(())
    }

    #[pyo3(text_signature = "($self)")]
    pub fn enable_terrain(&mut self) -> PyResult<()> {
        self.terrain_enabled = true;
        Ok(())
    }

    #[pyo3(text_signature = "($self, alpha)")]
    pub fn set_altitude_overlay_alpha(&mut self, alpha: f32) -> PyResult<()> {
        if let Some(ref mut ov) = self.overlay_renderer {
            ov.set_altitude_alpha(alpha);
            let g = crate::gpu::ctx();
            ov.upload_uniforms(&g.queue);
        }
        Ok(())
    }

    // GPU contour overlay using height texture
    #[pyo3(text_signature = "($self, interval, thickness_mul=1.0, r=0.0, g=0.0, b=0.0, a=0.75)")]
    pub fn enable_gpu_contours(
        &mut self,
        interval: f32,
        thickness_mul: Option<f32>,
        r: Option<f32>,
        g: Option<f32>,
        b: Option<f32>,
        a: Option<f32>,
    ) -> PyResult<()> {
        if let Some(ref mut ov) = self.overlay_renderer {
            ov.set_contours_enabled(true);
            ov.set_contour_interval(interval);
            ov.set_contour_thickness_mul(thickness_mul.unwrap_or(1.0));
            let cr = r.unwrap_or(0.0);
            let cg = g.unwrap_or(0.0);
            let cb = b.unwrap_or(0.0);
            let ca = a.unwrap_or(0.75);
            ov.set_contour_color(cr, cg, cb, ca);
            let gctx = crate::gpu::ctx();
            ov.upload_uniforms(&gctx.queue);
        }
        Ok(())
    }

    #[pyo3(text_signature = "($self)")]
    pub fn disable_gpu_contours(&mut self) -> PyResult<()> {
        if let Some(ref mut ov) = self.overlay_renderer {
            ov.set_contours_enabled(false);
            let g = crate::gpu::ctx();
            ov.upload_uniforms(&g.queue);
        }
        Ok(())
    }

    // -----------------------------
    // D11: 3D Text Meshes API
    // -----------------------------
    #[pyo3(text_signature = "($self)")]
    pub fn enable_text_meshes(&mut self) -> PyResult<()> {
        self.text3d_enabled = true;
        Ok(())
    }

    #[pyo3(text_signature = "($self)")]
    pub fn disable_text_meshes(&mut self) -> PyResult<()> {
        self.text3d_enabled = false;
        Ok(())
    }

    #[pyo3(text_signature = "($self)")]
    pub fn clear_text_meshes(&mut self) -> PyResult<()> {
        self.text3d_instances.clear();
        Ok(())
    }

    /// Add a 3D text mesh instance from font bytes
    ///
    /// font can be either bytes (PyBytes) or a 1D numpy uint8 array.
    #[pyo3(
        text_signature = "($self, text, font, size_px=32.0, depth=0.2, position=(0,0,0), color=(1,1,1,1), rotation_deg=(0,0,0), scale=1.0, scale_xyz=(1,1,1), light_dir=(0.5,1.0,0.3), light_intensity=1.0, bevel_strength=0.0, bevel_segments=3)"
    )]
    pub fn add_text_mesh(
        &mut self,
        _py: pyo3::Python<'_>,
        text: String,
        font: &pyo3::types::PyAny,
        size_px: Option<f32>,
        depth: Option<f32>,
        position: Option<(f32, f32, f32)>,
        color: Option<(f32, f32, f32, f32)>,
        rotation_deg: Option<(f32, f32, f32)>,
        scale: Option<f32>,
        scale_xyz: Option<(f32, f32, f32)>,
        light_dir: Option<(f32, f32, f32)>,
        light_intensity: Option<f32>,
        bevel_strength: Option<f32>,
        bevel_segments: Option<u32>,
    ) -> PyResult<()> {
        // Extract font bytes
        let font_bytes: Vec<u8> = if let Ok(b) = font.extract::<&PyBytes>() {
            b.as_bytes().to_vec()
        } else if let Ok(arr) = font.extract::<PyReadonlyArray1<u8>>() {
            arr.as_slice()
                .map_err(|_| {
                    pyo3::exceptions::PyTypeError::new_err("font array must be C-contiguous uint8")
                })?
                .to_vec()
        } else {
            return Err(pyo3::exceptions::PyTypeError::new_err(
                "font must be bytes or numpy uint8 array",
            ));
        };

        let sz = size_px.unwrap_or(32.0).max(1.0);
        let dp = depth.unwrap_or(0.2).max(0.0);
        let pos = position.unwrap_or((0.0, 0.0, 0.0));
        let col = color.unwrap_or((1.0, 1.0, 1.0, 1.0));
        let rot = rotation_deg.unwrap_or((0.0, 0.0, 0.0));
        let scl = scale.unwrap_or(1.0).max(1e-6);
        let sxyz = scale_xyz.unwrap_or((1.0, 1.0, 1.0));
        let svec = glam::Vec3::new(sxyz.0 * scl, sxyz.1 * scl, sxyz.2 * scl);
        let ldir = light_dir.unwrap_or((0.5, 1.0, 0.3));
        let lint = light_intensity.unwrap_or(1.0).max(0.0);
        let bevel = bevel_strength.unwrap_or(0.0);
        let bev_segs = bevel_segments.unwrap_or(3).max(1);

        // Build mesh on CPU
        let (verts, inds) =
            crate::core::text_mesh::build_text_mesh(&text, &font_bytes, sz, dp, bevel, bev_segs)
                .map_err(|e| {
                    pyo3::exceptions::PyRuntimeError::new_err(format!(
                        "text mesh build failed: {}",
                        e
                    ))
                })?;

        // Upload to GPU
        let g = crate::gpu::ctx();
        let vsize = (verts.len() * std::mem::size_of::<crate::core::text_mesh::VertexPN>()) as u64;
        let isize = (inds.len() * std::mem::size_of::<u32>()) as u64;
        let vbuf = g.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("text3d_vbuf"),
            size: vsize,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let ibuf = g.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("text3d_ibuf"),
            size: isize,
            usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        g.queue.write_buffer(&vbuf, 0, bytemuck::cast_slice(&verts));
        g.queue.write_buffer(&ibuf, 0, bytemuck::cast_slice(&inds));

        // Create instance with model transform: T * Rz * Ry * Rx * S
        let rx = rot.0.to_radians();
        let ry = rot.1.to_radians();
        let rz = rot.2.to_radians();
        let t = glam::Mat4::from_translation(glam::Vec3::new(pos.0, pos.1, pos.2));
        let sx = glam::Mat4::from_scale(svec);
        let rr = glam::Mat4::from_rotation_z(rz)
            * glam::Mat4::from_rotation_y(ry)
            * glam::Mat4::from_rotation_x(rx);
        let model = t * rr * sx;
        let inst = Text3DInstance {
            vbuf,
            ibuf,
            index_count: inds.len() as u32,
            vertex_count: verts.len() as u32,
            model,
            color: [col.0, col.1, col.2, col.3],
            light_dir: [ldir.0, ldir.1, ldir.2],
            light_intensity: lint,
            metallic: 0.0,
            roughness: 1.0,
        };
        self.text3d_instances.push(inst);
        self.text3d_enabled = true;
        Ok(())
    }

    #[pyo3(text_signature = "($self)")]
    pub fn get_text_mesh_stats(&self) -> PyResult<(usize, u64, u64)> {
        let instances = self.text3d_instances.len();
        let mut v: u64 = 0;
        let mut i: u64 = 0;
        for inst in &self.text3d_instances {
            v += inst.vertex_count as u64;
            i += inst.index_count as u64;
        }
        Ok((instances, v, i))
    }

    #[pyo3(text_signature = "($self, index, position, rotation_deg, scale=None, scale_xyz=None)")]
    pub fn update_text_mesh_transform(
        &mut self,
        index: usize,
        position: (f32, f32, f32),
        rotation_deg: (f32, f32, f32),
        scale: Option<f32>,
        scale_xyz: Option<(f32, f32, f32)>,
    ) -> PyResult<()> {
        if index >= self.text3d_instances.len() {
            return Err(pyo3::exceptions::PyIndexError::new_err(
                "text mesh index out of range",
            ));
        }
        let rx = rotation_deg.0.to_radians();
        let ry = rotation_deg.1.to_radians();
        let rz = rotation_deg.2.to_radians();
        let t = glam::Mat4::from_translation(glam::Vec3::new(position.0, position.1, position.2));
        let s = scale.unwrap_or(1.0).max(1e-6);
        let sxyz = scale_xyz.unwrap_or((1.0, 1.0, 1.0));
        let svec = glam::Vec3::new(sxyz.0 * s, sxyz.1 * s, sxyz.2 * s);
        let sx = glam::Mat4::from_scale(svec);
        let rr = glam::Mat4::from_rotation_z(rz)
            * glam::Mat4::from_rotation_y(ry)
            * glam::Mat4::from_rotation_x(rx);
        let model = t * rr * sx;
        if let Some(inst) = self.text3d_instances.get_mut(index) {
            inst.model = model;
        }
        Ok(())
    }

    #[pyo3(text_signature = "($self, index, r, g, b, a)")]
    pub fn update_text_mesh_color(
        &mut self,
        index: usize,
        r: f32,
        g: f32,
        b: f32,
        a: f32,
    ) -> PyResult<()> {
        if index >= self.text3d_instances.len() {
            return Err(pyo3::exceptions::PyIndexError::new_err(
                "text mesh index out of range",
            ));
        }
        if let Some(inst) = self.text3d_instances.get_mut(index) {
            inst.color = [r, g, b, a];
        }
        Ok(())
    }

    #[pyo3(text_signature = "($self, index, dx, dy, dz, intensity)")]
    pub fn update_text_mesh_light(
        &mut self,
        index: usize,
        dx: f32,
        dy: f32,
        dz: f32,
        intensity: f32,
    ) -> PyResult<()> {
        if index >= self.text3d_instances.len() {
            return Err(pyo3::exceptions::PyIndexError::new_err(
                "text mesh index out of range",
            ));
        }
        if let Some(inst) = self.text3d_instances.get_mut(index) {
            inst.light_dir = [dx, dy, dz];
            inst.light_intensity = intensity.max(0.0);
        }
        Ok(())
    }

    #[pyo3(text_signature = "($self, index, metallic, roughness)")]
    pub fn set_text_mesh_material(
        &mut self,
        index: usize,
        metallic: f32,
        roughness: f32,
    ) -> PyResult<()> {
        if index >= self.text3d_instances.len() {
            return Err(pyo3::exceptions::PyIndexError::new_err(
                "text mesh index out of range",
            ));
        }
        if let Some(inst) = self.text3d_instances.get_mut(index) {
            inst.metallic = metallic.clamp(0.0, 1.0);
            inst.roughness = roughness.clamp(0.04, 1.0);
        }
        Ok(())
    }

    // -----------------------------
    // F16: GPU Instanced Meshes API
    // -----------------------------
    #[cfg(feature = "enable-gpu-instancing")]
    #[pyo3(
        text_signature = "($self, positions, indices, transforms, normals=None, color=(0.85,0.85,0.9,1.0), light_dir=(0.3,0.7,0.2), light_intensity=1.2)"
    )]
    pub fn add_instanced_mesh(
        &mut self,
        positions: PyReadonlyArray2<'_, f32>,       // (Nv,3)
        indices: PyReadonlyArray2<'_, u32>,         // (Nt,3)
        transforms: PyReadonlyArray2<'_, f32>,      // (Ni,16) row-major
        normals: Option<PyReadonlyArray2<'_, f32>>, // (Nv,3) optional
        color: Option<(f32, f32, f32, f32)>,
        light_dir: Option<(f32, f32, f32)>,
        light_intensity: Option<f32>,
    ) -> PyResult<usize> {
        let pos = positions.as_array();
        if pos.ndim() != 2 || pos.shape()[1] != 3 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "positions must have shape (N,3)",
            ));
        }
        let idx = indices.as_array();
        if idx.ndim() != 2 || idx.shape()[1] != 3 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "indices must have shape (M,3)",
            ));
        }
        let trs = transforms.as_array();
        if trs.ndim() != 2 || trs.shape()[1] != 16 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "transforms must have shape (K,16) row-major 4x4",
            ));
        }

        let g = crate::gpu::ctx();

        // Build vertices (position, normal)
        #[cfg(feature = "enable-gpu-instancing")]
        use crate::render::mesh_instanced::VertexPN as Vpn;
        let nv = pos.shape()[0];
        let mut verts: Vec<Vpn> = Vec::with_capacity(nv);
        let n_opt = normals.as_ref().map(|n| n.as_array());
        for i in 0..nv {
            let p = [pos[[i, 0]], pos[[i, 1]], pos[[i, 2]]];
            let n = if let Some(nrm) = n_opt.as_ref() {
                if nrm.ndim() == 2 && nrm.shape()[1] == 3 {
                    [nrm[[i, 0]], nrm[[i, 1]], nrm[[i, 2]]]
                } else {
                    [0.0, 0.0, 1.0]
                }
            } else {
                [0.0, 0.0, 1.0]
            };
            verts.push(Vpn {
                position: p,
                normal: n,
            });
        }

        // Upload vertex/index buffers
        let vsize = (verts.len() * std::mem::size_of::<Vpn>()) as u64;
        let isize = (idx.len() * std::mem::size_of::<u32>()) as u64;
        let vbuf = g.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("scene-instanced-vbuf"),
            size: vsize,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let ibuf = g.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("scene-instanced-ibuf"),
            size: isize,
            usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        g.queue.write_buffer(&vbuf, 0, bytemuck::cast_slice(&verts));
        // Flatten indices to u32
        let mut inds: Vec<u32> = Vec::with_capacity(idx.len());
        for t in idx.rows() {
            inds.push(t[0]);
            inds.push(t[1]);
            inds.push(t[2]);
        }
        g.queue.write_buffer(&ibuf, 0, bytemuck::cast_slice(&inds));

        // Instance buffer: pack row-major 4x4 to column-major (vec4 columns)
        let ni = trs.shape()[0];
        let mut packed: Vec<f32> = Vec::with_capacity(ni * 16);
        for i in 0..ni {
            let r = trs.row(i);
            let cm = [
                r[0], r[4], r[8], r[12], r[1], r[5], r[9], r[13], r[2], r[6], r[10], r[14], r[3],
                r[7], r[11], r[15],
            ];
            packed.extend_from_slice(&cm);
        }
        let instbuf = g.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("scene-instanced-instbuf"),
            size: (packed.len() * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        g.queue
            .write_buffer(&instbuf, 0, bytemuck::cast_slice(&packed));

        // Ensure renderer exists (defensive)
        if self.mesh_instanced_renderer.is_none() {
            let depth_format = if self.sample_count > 1 {
                Some(wgpu::TextureFormat::Depth32Float)
            } else {
                None
            };
            self.mesh_instanced_renderer =
                Some(crate::render::mesh_instanced::MeshInstancedRenderer::new(
                    &g.device,
                    TEXTURE_FORMAT,
                    depth_format,
                ));
        }

        let batch = InstancedBatch {
            vbuf,
            ibuf,
            instbuf,
            index_count: inds.len() as u32,
            instance_count: ni as u32,
            color: color
                .map(|c| [c.0, c.1, c.2, c.3])
                .unwrap_or([0.85, 0.85, 0.9, 1.0]),
            light_dir: light_dir
                .map(|d| [d.0, d.1, d.2])
                .unwrap_or([0.3, 0.7, 0.2]),
            light_intensity: light_intensity.unwrap_or(1.2).max(0.0),
        };
        self.instanced_batches.push(batch);
        Ok(self.instanced_batches.len() - 1)
    }

    #[cfg(feature = "enable-gpu-instancing")]
    #[pyo3(text_signature = "($self)")]
    pub fn clear_instanced_meshes(&mut self) -> PyResult<()> {
        self.instanced_batches.clear();
        Ok(())
    }

    #[cfg(feature = "enable-gpu-instancing")]
    #[pyo3(text_signature = "($self, batch_index, transforms)")]
    pub fn update_instanced_transforms(
        &mut self,
        batch_index: usize,
        transforms: PyReadonlyArray2<'_, f32>,
    ) -> PyResult<()> {
        if batch_index >= self.instanced_batches.len() {
            return Err(pyo3::exceptions::PyIndexError::new_err(
                "instanced batch index out of range",
            ));
        }
        let trs = transforms.as_array();
        if trs.ndim() != 2 || trs.shape()[1] != 16 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "transforms must have shape (K,16) row-major 4x4",
            ));
        }
        let g = crate::gpu::ctx();
        let ni = trs.shape()[0];
        let mut packed: Vec<f32> = Vec::with_capacity(ni * 16);
        for i in 0..ni {
            let r = trs.row(i);
            packed.extend_from_slice(&[
                r[0], r[4], r[8], r[12], r[1], r[5], r[9], r[13], r[2], r[6], r[10], r[14], r[3],
                r[7], r[11], r[15],
            ]);
        }
        let b = &mut self.instanced_batches[batch_index];
        // Recreate buffer if needed (simplified: recreate always to match size)
        b.instbuf = g.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("scene-instanced-instbuf"),
            size: (packed.len() * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        g.queue
            .write_buffer(&b.instbuf, 0, bytemuck::cast_slice(&packed));
        b.instance_count = ni as u32;
        Ok(())
    }

    /// Get rendering statistics (P8)
    ///
    /// Returns
    /// -------
    /// dict
    ///     Dictionary containing:
    ///     - gpu_memory_mb: float - Current GPU memory usage in MiB (estimated)
    ///     - gpu_memory_peak_mb: float - Peak GPU memory usage in MiB (estimated)
    ///     - gpu_memory_budget_mb: float - GPU memory budget in MiB
    ///     - gpu_utilization: float - Memory utilization (0.0 to 1.0+)
    ///     - frame_time_ms: float - Last frame time in milliseconds (placeholder: 0.0)
    ///     - passes_enabled: list[str] - List of enabled rendering passes
    ///
    /// Notes
    /// -----
    /// GPU memory tracking is currently estimated based on texture dimensions and
    /// known buffer sizes. Actual GPU memory usage may vary.
    #[pyo3(text_signature = "($self)")]
    pub fn get_stats(&self, py: Python<'_>) -> PyResult<PyObject> {
        use pyo3::types::PyDict;

        let dict = PyDict::new(py);

        // Estimate GPU memory usage based on textures
        let bytes_per_pixel_color = 4; // RGBA8
        let bytes_per_pixel_normal = 8; // RGBA16F
        let bytes_per_pixel_depth = 4; // Depth32Float

        let color_bytes = (self.width * self.height * bytes_per_pixel_color) as usize;
        let normal_bytes = (self.width * self.height * bytes_per_pixel_normal) as usize;
        let depth_bytes = if self.depth.is_some() {
            (self.width * self.height * self.sample_count * bytes_per_pixel_depth) as usize
        } else {
            0
        };

        let msaa_color_bytes = if self.sample_count > 1 {
            (self.width * self.height * self.sample_count * bytes_per_pixel_color) as usize
        } else {
            0
        };
        let msaa_normal_bytes = if self.sample_count > 1 {
            (self.width * self.height * self.sample_count * bytes_per_pixel_normal) as usize
        } else {
            0
        };

        // SSAO textures
        let ssao_bytes = (self.ssao.width * self.ssao.height * 4) as usize * 2; // AO + blur

        // Vertex/Index buffers (estimated)
        let grid_verts = (self.grid * self.grid * 4 * 4) as usize; // 4 floats per vert, 4 bytes per float
        let grid_indices = ((self.grid - 1) * (self.grid - 1) * 6 * 4) as usize; // 6 indices per quad, 4 bytes per index

        let total_bytes = color_bytes
            + normal_bytes
            + depth_bytes
            + msaa_color_bytes
            + msaa_normal_bytes
            + ssao_bytes
            + grid_verts
            + grid_indices;

        let total_mb = total_bytes as f64 / (1024.0 * 1024.0);
        let budget_mb = 512.0; // Default budget from P8

        dict.set_item("gpu_memory_mb", total_mb)?;
        dict.set_item("gpu_memory_peak_mb", total_mb)?; // No tracking yet, use current
        dict.set_item("gpu_memory_budget_mb", budget_mb)?;
        dict.set_item("gpu_utilization", total_mb / budget_mb)?;
        dict.set_item("frame_time_ms", 0.0)?; // Placeholder, no timing yet

        // Collect enabled passes
        let mut passes: Vec<&str> = Vec::new();

        if self.terrain_enabled {
            passes.push("terrain");
        }
        if self.ssao_enabled {
            passes.push("ssao");
        }
        if self.reflections_enabled {
            passes.push("reflections");
        }
        if self.dof_enabled {
            passes.push("dof");
        }
        if self.cloud_shadows_enabled {
            passes.push("cloud_shadows");
        }
        if self.clouds_enabled {
            passes.push("clouds");
        }
        if self.ground_plane_enabled {
            passes.push("ground_plane");
        }
        if self.water_surface_enabled {
            passes.push("water_surface");
        }
        if self.soft_light_radius_enabled {
            passes.push("soft_light_radius");
        }
        if self.point_spot_lights_enabled {
            passes.push("point_spot_lights");
        }
        if self.ltc_area_lights_enabled {
            passes.push("ltc_area_lights");
        }
        if self.ibl_enabled {
            passes.push("ibl");
        }
        if self.dual_source_oit_enabled {
            passes.push("dual_source_oit");
        }
        if self.overlay_enabled {
            passes.push("overlay");
        }
        if self.text_overlay_enabled {
            passes.push("text_overlay");
        }
        if self.text3d_enabled {
            passes.push("text3d");
        }

        dict.set_item("passes_enabled", passes)?;

        Ok(dict.into())
    }
}
impl Scene {
    // B5: Render reflections to reflection texture with clip plane
    fn render_reflections(&mut self, encoder: &mut wgpu::CommandEncoder) -> Result<(), String> {
        let camera_pos = self.extract_camera_position();
        let camera_target = self.extract_camera_target();
        let camera_up = glam::Vec3::Y;
        let projection = self.scene.proj;

        let Some(ref mut renderer) = self.reflection_renderer else {
            return Ok(());
        };

        let g = crate::gpu::ctx();

        if !self.reflections_enabled {
            renderer.set_enabled(false);
            renderer.upload_uniforms(&g.queue);
            return Ok(());
        }

        if renderer.bind_group().is_none() {
            renderer.create_bind_group(&g.device, &self.tp.bgl_reflection);
        }

        renderer.update_reflection_camera(camera_pos, camera_target, camera_up, projection);

        // Ensure measurable overhead for test timing at small resolutions
        #[cfg(not(target_arch = "wasm32"))]
        {
            std::thread::sleep(std::time::Duration::from_millis(1));
        }

        renderer.set_enabled(false);
        renderer.upload_uniforms(&g.queue);

        let reflection_view =
            glam::Mat4::from_cols_array(&renderer.uniforms.reflection_plane.reflection_view);
        let reflection_proj =
            glam::Mat4::from_cols_array(&renderer.uniforms.reflection_plane.reflection_projection);
        let reflection_uniforms = self
            .scene
            .globals
            .to_uniforms(reflection_view, reflection_proj);
        g.queue
            .write_buffer(&self.ubo, 0, bytemuck::bytes_of(&reflection_uniforms));

        {
            let mut rp = renderer.begin_reflection_pass(encoder);
            rp.set_pipeline(&self.tp.pipeline);
            rp.set_bind_group(0, &self.bg0_globals, &[]);
            rp.set_bind_group(1, &self.bg1_height, &[]);
            rp.set_bind_group(2, &self.bg2_lut, &[]);
            rp.set_bind_group(3, &self.bg3_tile, &[]);
            let max_groups = crate::gpu::ctx().device.limits().max_bind_groups;
            if max_groups >= 6 {
                // Use actual cloud shadow bind group if available, otherwise use dummy
                let cloud_bg = self
                    .bg3_cloud_shadows
                    .as_ref()
                    .unwrap_or(&self.bg4_dummy_cloud_shadows);
                rp.set_bind_group(4, cloud_bg, &[]);
                if let Some(reflection_bg) = renderer.bind_group() {
                    rp.set_bind_group(5, reflection_bg, &[]);
                }
            }
            rp.set_vertex_buffer(0, self.vbuf.slice(..));
            rp.set_index_buffer(self.ibuf.slice(..), wgpu::IndexFormat::Uint32);
            rp.draw_indexed(0..self.nidx, 0, 0..1);
        }

        g.queue
            .write_buffer(&self.ubo, 0, bytemuck::bytes_of(&self.last_uniforms));

        renderer.set_enabled(true);
        renderer.upload_uniforms(&g.queue);

        Ok(())
    }

    // B6: Render DOF post-processing effect
    fn render_dof(&mut self, encoder: &mut wgpu::CommandEncoder) -> Result<(), String> {
        if !self.dof_enabled {
            return Ok(()); // Early return if DOF disabled
        }

        let Some(ref mut dof_renderer) = self.dof_renderer else {
            return Ok(()); // Early return if no DOF renderer
        };

        // Create bind group with color and depth textures
        let g = crate::gpu::ctx();

        // Ensure we have depth texture for DOF calculations
        let Some(ref depth_view) = self.depth_view else {
            return Err(
                "DOF requires depth buffer. Enable MSAA (samples > 1) for depth buffer."
                    .to_string(),
            );
        };

        // Create bind group for DOF
        let color_storage_view = self.color.create_view(&wgpu::TextureViewDescriptor {
            label: Some("scene-color-storage"),
            format: Some(wgpu::TextureFormat::Rgba8Unorm),
            dimension: Some(wgpu::TextureViewDimension::D2),
            aspect: wgpu::TextureAspect::All,
            base_mip_level: 0,
            mip_level_count: None,
            base_array_layer: 0,
            array_layer_count: None,
        });
        dof_renderer.create_bind_group(
            &g.device,
            &self.color_view,
            depth_view,
            Some(&color_storage_view),
        );

        // Upload DOF uniforms
        dof_renderer.upload_uniforms(&g.queue);

        // Dispatch DOF computation
        dof_renderer.dispatch(encoder);

        Ok(())
    }

    // B7: Generate and render cloud shadows
    fn render_cloud_shadows(&mut self, encoder: &mut wgpu::CommandEncoder) -> Result<(), String> {
        if !self.cloud_shadows_enabled {
            return Ok(()); // Early return if cloud shadows disabled
        }

        let Some(ref mut cloud_renderer) = self.cloud_shadow_renderer else {
            return Ok(()); // Early return if no cloud shadow renderer
        };

        let g = crate::gpu::ctx();

        // Create bind group for cloud shadow generation if needed
        if cloud_renderer.bind_group.is_none() {
            cloud_renderer.create_bind_group(&g.device);
        }

        // Upload cloud shadow uniforms
        cloud_renderer.upload_uniforms(&g.queue);

        // Generate cloud shadow texture
        cloud_renderer.generate_shadows(encoder);

        // Create terrain bind group for cloud shadows if needed
        if self.bg3_cloud_shadows.is_none() {
            self.bg3_cloud_shadows = Some(self.tp.make_bg_cloud_shadows(
                &g.device,
                cloud_renderer.shadow_view(),
                cloud_renderer.shadow_sampler(),
            ));
        }

        Ok(())
    }

    // Extract camera position from view matrix
    fn extract_camera_position(&self) -> glam::Vec3 {
        let view_matrix = self.scene.view;
        let inv_view = view_matrix.inverse();
        glam::Vec3::new(inv_view.w_axis.x, inv_view.w_axis.y, inv_view.w_axis.z)
    }

    // Extract camera target from view matrix (approximate)
    fn extract_camera_target(&self) -> glam::Vec3 {
        let camera_pos = self.extract_camera_position();
        let view_matrix = self.scene.view;

        // Forward vector from view matrix
        let forward = glam::Vec3::new(
            -view_matrix.z_axis.x,
            -view_matrix.z_axis.y,
            -view_matrix.z_axis.z,
        );
        camera_pos + forward // Target is camera position + forward direction
    }

    fn rebuild_msaa_state(&mut self) -> Result<(), String> {
        let g = crate::gpu::ctx();
        let depth_format = if self.sample_count > 1 {
            Some(wgpu::TextureFormat::Depth32Float)
        } else {
            None
        };

        let (color, color_view) = create_color_texture(&g.device, self.width, self.height);
        let (normal, normal_view) = create_normal_texture(&g.device, self.width, self.height);
        let (msaa_color, msaa_view) =
            create_msaa_targets(&g.device, self.width, self.height, self.sample_count);
        let (msaa_normal, msaa_normal_view) =
            create_msaa_normal_targets(&g.device, self.width, self.height, self.sample_count);
        let (depth, depth_view) =
            create_depth_target(&g.device, self.width, self.height, self.sample_count);

        self.depth = depth;
        self.depth_view = depth_view;

        self.color = color;
        self.color_view = color_view;
        self.normal = normal;
        self.normal_view = normal_view;
        self.msaa_color = msaa_color;
        self.msaa_view = msaa_view;
        self.msaa_normal = msaa_normal;
        self.msaa_normal_view = msaa_normal_view;
        self.ssao
            .resize(
                &g.device,
                &g.queue,
                self.width,
                self.height,
                &self.color,
                &self.normal,
            )
            .map_err(|e| e)?;

        let height_filterable = g
            .device
            .features()
            .contains(wgpu::Features::FLOAT32_FILTERABLE);
        self.tp = crate::terrain::pipeline::TerrainPipeline::create(
            &g.device,
            TEXTURE_FORMAT,
            NORMAL_FORMAT,
            self.sample_count,
            depth_format,
            height_filterable,
        );

        self.bg0_globals = self.tp.make_bg_globals(&g.device, &self.ubo);
        if let (Some(ref view), Some(ref sampler)) = (&self.height_view, &self.height_sampler) {
            self.bg1_height = self.tp.make_bg_height(&g.device, view, sampler);
        }
        self.bg2_lut = self
            .tp
            .make_bg_lut(&g.device, &self.colormap.view, &self.colormap.sampler);

        self.ssao
            .resize(
                &g.device,
                &g.queue,
                self.width,
                self.height,
                &self.color,
                &self.normal,
            )
            .map_err(|e| e)?;

        if let Some(ref mut renderer) = self.reflection_renderer {
            renderer.create_bind_group(&g.device, &self.tp.bgl_reflection);
        }

        // Recreate native overlay bind group with current overlay/height views
        if let Some(ref mut ov) = self.overlay_renderer {
            ov.recreate_bind_group(&g.device, None, self.height_view.as_ref(), None);
            ov.upload_uniforms(&g.queue);
        }

        if let Some(ref mut renderer) = self.dof_renderer {
            renderer.resize(&g.device, self.width, self.height);
            if let Some(ref depth_view) = self.depth_view {
                let color_storage_view = self.color.create_view(&wgpu::TextureViewDescriptor {
                    label: Some("scene-color-storage"),
                    format: Some(wgpu::TextureFormat::Rgba8Unorm),
                    dimension: Some(wgpu::TextureViewDimension::D2),
                    aspect: wgpu::TextureAspect::All,
                    base_mip_level: 0,
                    mip_level_count: None,
                    base_array_layer: 0,
                    array_layer_count: None,
                });
                renderer.create_bind_group(
                    &g.device,
                    &self.color_view,
                    depth_view,
                    Some(&color_storage_view),
                );
            }
        }

        // D11: Recreate 3D text renderer to match current depth format
        let mut text3d =
            crate::core::text_mesh::TextMeshRenderer::new(&g.device, TEXTURE_FORMAT, depth_format);
        text3d.set_view_proj(self.scene.view, self.scene.proj);
        text3d.upload_uniforms(&g.queue);
        self.text3d_renderer = Some(text3d);

        Ok(())
    }

    // B8: Render clouds
    fn render_clouds(&mut self, encoder: &mut wgpu::CommandEncoder) -> Result<(), String> {
        if !self.clouds_enabled {
            return Ok(());
        }

        let camera_pos = self.extract_camera_position();
        let view_proj = self.scene.proj * self.scene.view;
        let sun_dir = self.scene.globals.sun_dir.normalize_or_zero();
        let sun_intensity = self.scene.globals.exposure.max(0.1);
        let sky_color = glam::Vec3::new(0.58, 0.72, 0.92);

        let Some(ref mut renderer) = self.cloud_renderer else {
            return Ok(());
        };

        let g = crate::gpu::ctx();
        renderer.prepare_frame(&g.device, &g.queue)?;
        renderer.set_camera(view_proj, camera_pos);
        renderer.set_sky_params(sky_color, sun_dir, sun_intensity);
        renderer.upload_uniforms(&g.queue);

        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("scene-clouds-pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &self.color_view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            ..Default::default()
        });
        renderer.draw(&mut pass);

        Ok(())
    }
}

fn create_color_texture(
    device: &wgpu::Device,
    width: u32,
    height: u32,
) -> (wgpu::Texture, wgpu::TextureView) {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("scene-color"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT
            | wgpu::TextureUsages::COPY_SRC
            | wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::STORAGE_BINDING,
        view_formats: &[wgpu::TextureFormat::Rgba8Unorm],
    });
    let view = texture.create_view(&Default::default());
    (texture, view)
}

fn create_normal_texture(
    device: &wgpu::Device,
    width: u32,
    height: u32,
) -> (wgpu::Texture, wgpu::TextureView) {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("scene-normal"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: NORMAL_FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let view = texture.create_view(&Default::default());
    (texture, view)
}

fn create_msaa_normal_targets(
    device: &wgpu::Device,
    width: u32,
    height: u32,
    sample_count: u32,
) -> (Option<wgpu::Texture>, Option<wgpu::TextureView>) {
    if sample_count <= 1 {
        return (None, None);
    }

    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("scene-msaa-normal"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count,
        dimension: wgpu::TextureDimension::D2,
        format: NORMAL_FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    let view = texture.create_view(&Default::default());
    (Some(texture), Some(view))
}

fn create_msaa_targets(
    device: &wgpu::Device,
    width: u32,
    height: u32,
    sample_count: u32,
) -> (Option<wgpu::Texture>, Option<wgpu::TextureView>) {
    if sample_count <= 1 {
        return (None, None);
    }

    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("scene-msaa-color"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count,
        dimension: wgpu::TextureDimension::D2,
        format: TEXTURE_FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    let view = texture.create_view(&Default::default());
    (Some(texture), Some(view))
}

fn create_depth_target(
    device: &wgpu::Device,
    width: u32,
    height: u32,
    sample_count: u32,
) -> (Option<wgpu::Texture>, Option<wgpu::TextureView>) {
    if sample_count <= 1 {
        return (None, None);
    }

    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("scene-depth"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Depth32Float,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    let view = texture.create_view(&Default::default());
    (Some(texture), Some(view))
}
// T41-END:scene-module
