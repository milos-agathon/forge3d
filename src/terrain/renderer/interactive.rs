use super::*;
use crate::core::gbuffer::GBuffer;
use crate::core::resource_tracker::{
    tracked_create_buffer_init, tracked_create_texture, TrackedBuffer, TrackedTexture,
};
use crate::core::screen_space_effects::{
    CameraParams, ScreenSpaceEffect, ScreenSpaceEffectsManager,
};

#[derive(Clone, Copy)]
pub(super) enum ComposePass {
    Ao,
    Gi,
    Reflections,
    Fog,
}

mod cartographic_depth;
mod execution;
mod fog;
mod labels;
mod overlays;
mod points;
mod settings;
pub(crate) mod shaders;
mod validation;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(super) struct FrameUniforms {
    pub(super) view: [[f32; 4]; 4],
    inv_view: [[f32; 4]; 4],
    pub(super) proj: [[f32; 4]; 4],
    inv_proj: [[f32; 4]; 4],
    previous_view_proj: [[f32; 4]; 4],
    cloud_scale_bias: [f32; 4],
    effects: [u32; 4],
    geometry: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ComposeUniforms {
    inv_view: [[f32; 4]; 4],
    inv_proj: [[f32; 4]; 4],
    terrain_scale_bias: [f32; 4],
    weights: [f32; 4],
}

/// One owner for geometry data and temporal history. Moved into RenderTargets
/// while encoding and returned to TerrainScene after the complete submission.
pub(super) struct ConcordiaFrame {
    settings_key: String,
    height_ao_active: bool,
    height_ao_pipeline: wgpu::ComputePipeline,
    terrain_signature: Option<u64>,
    radiance_signature: Option<[u8; 32]>,
    geometry_signature: Option<[u8; 32]>,
    chronos_identity: Option<(u64, u64)>,
    pub(super) current_uniforms: FrameUniforms,
    geometry_hdr: Option<Arc<TrackedTexture>>,
    point_layers: Vec<points::PointLayer>,
    label_batches: Vec<labels::LabelBatch>,
    vector_batches: Vec<crate::core::overlays::OverlayRenderer>,
    cartography_depth: Option<cartographic_depth::CartographyDepth>,
    _surface_kind: TrackedTexture,
    pub(super) surface_kind_view: wgpu::TextureView,
    pub(super) effects: ScreenSpaceEffectsManager,
    pub(super) geometry_pipeline: wgpu::RenderPipeline,
    pub(super) frame_layout: wgpu::BindGroupLayout,
    pub(super) frame_bind_group: wgpu::BindGroup,
    frame_uniform: TrackedBuffer,
    pub(super) cloud: crate::core::cloud_shadows::CloudShadowRenderer,
    taa: crate::core::taa::TaaRenderer,
    atmosphere_pipeline: wgpu::ComputePipeline,
    atmosphere_empty_group: wgpu::BindGroup,
    atmosphere_layout: wgpu::BindGroupLayout,
    pub(super) atmosphere_group: Mutex<Option<wgpu::BindGroup>>,
    previous_view_proj: glam::Mat4,
    previous_camera_mode: String,
    frame_index: u32,
    pub(super) background_pipeline: wgpu::RenderPipeline,
    pub(super) background_hdr_pipeline: wgpu::RenderPipeline,
    compose_layout: wgpu::BindGroupLayout,
    ao_pipeline: wgpu::ComputePipeline,
    gi_pipeline: wgpu::ComputePipeline,
    ssr_pipeline: wgpu::ComputePipeline,
    fog_pipeline: wgpu::ComputePipeline,
    froxels: Option<fog::FroxelState>,
    compose_uniform: TrackedBuffer,
    scratch: [TrackedTexture; 2],
    scratch_views: [wgpu::TextureView; 2],
    white: TrackedTexture,
    white_view: wgpu::TextureView,
    current_scratch: usize,
    composition_encoded: bool,
}

fn sampled(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable: false },
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: false,
        },
        count: None,
    }
}

fn texture_extent(texture: &wgpu::Texture) -> [u32; 3] {
    [
        texture.width(),
        texture.height(),
        texture.depth_or_array_layers(),
    ]
}

fn hdr_texture(
    device: &wgpu::Device,
    width: u32,
    height: u32,
    label: &str,
) -> Result<TrackedTexture> {
    Ok(tracked_create_texture(
        device,
        &wgpu::TextureDescriptor {
            label: Some(label),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::COPY_SRC
                | wgpu::TextureUsages::COPY_DST
                | wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        },
    )?)
}

impl ConcordiaFrame {
    pub(super) fn new(
        scene: &TerrainScene,
        params: &crate::terrain::render_params::TerrainRenderParams,
    ) -> Result<Self> {
        let (width, height) = params.size_px;
        let device = scene.device.as_ref();
        if device.limits().max_bind_groups < 8 {
            return Err(anyhow!("CONCORDIA terrain/atmosphere bindings require eight bind groups; the device granted {}",device.limits().max_bind_groups));
        }
        let queue = scene.queue.as_ref();
        let frame_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("concordia.frame.layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT | wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT | wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT | wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        let frame_uniform = tracked_create_buffer_init(
            device,
            &wgpu::util::BufferInitDescriptor {
                label: Some("concordia.frame.uniform"),
                contents: bytemuck::bytes_of(&FrameUniforms::zeroed()),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            },
        )?;
        let mut cloud = crate::core::cloud_shadows::CloudShadowRenderer::new(
            device,
            crate::core::cloud_shadows::CloudShadowQuality::Medium,
        )?;
        cloud.create_bind_group(device);
        let frame_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("concordia.frame.bind_group"),
            layout: &frame_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: frame_uniform.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(cloud.shadow_view()),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(cloud.shadow_sampler()),
                },
            ],
        });
        let geometry_pipeline = scene.create_concordia_pipeline(params, &frame_layout)?;
        let mut post_entries = vec![
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            },
            sampled(3),
            sampled(4),
        ];
        post_entries.push(wgpu::BindGroupLayoutEntry {
            binding: 5,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::StorageTexture {
                access: wgpu::StorageTextureAccess::WriteOnly,
                format: wgpu::TextureFormat::Rgba16Float,
                view_dimension: wgpu::TextureViewDimension::D2,
            },
            count: None,
        });
        let atmosphere_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("concordia.atmosphere.layout"),
            entries: &post_entries,
        });
        let empty = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("concordia.unused.layout"),
            entries: &[],
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("concordia.atmosphere.pipeline_layout"),
            bind_group_layouts: &[
                &empty,
                &empty,
                &empty,
                &empty,
                &scene.fog_bind_group_layout,
                &empty,
                &empty,
                &atmosphere_layout,
            ],
            push_constant_ranges: &[],
        });
        let shader = crate::core::shader_registry::create_labeled_shader_module(
            device,
            "concordia.atmosphere.shader",
            &shaders::atmosphere_source()?,
        );
        let atmosphere_empty_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("concordia.atmosphere.unused"),
            layout: &empty,
            entries: &[],
        });
        let atmosphere_pipeline = crate::core::shader_registry::create_compute_pipeline_scoped(
            device,
            &wgpu::ComputePipelineDescriptor {
                label: Some("concordia.atmosphere.pipeline"),
                layout: Some(&layout),
                module: &shader,
                entry_point: "cs_concordia_atmosphere",
            },
        );
        let mut effects = ScreenSpaceEffectsManager::new_linear_depth(device, width, height)?;
        if effects.hzb.is_none() {
            return Err(anyhow!("CONCORDIA requires the native depth pyramid"));
        }
        let settings = &params.decoded().screen_space;
        if settings.ssao_enabled {
            effects.enable_effect(device, ScreenSpaceEffect::SSAO)?;
            effects.update_ssao_settings(queue, |value| {
                value.technique = 1;
                value.radius = settings.ssao_radius;
                value.intensity = settings.ssao_intensity;
                value.inv_resolution = [1.0 / width as f32, 1.0 / height as f32];
                value.proj_scale =
                    0.5 * height as f32 / (params.fov_y_deg.to_radians() * 0.5).tan();
            });
        }
        if settings.ssgi_enabled {
            effects.enable_effect(device, ScreenSpaceEffect::SSGI)?;
        }
        if settings.ssr_enabled {
            effects.enable_effect(device, ScreenSpaceEffect::SSR)?;
            let ssr = crate::render::params::SsrParams {
                ssr_enable: true,
                ..Default::default()
            };
            effects.set_ssr_params(queue, &ssr);
        }
        let mut taa = crate::core::taa::TaaRenderer::new_shared_geometry(device, width, height)?;
        taa.set_enabled(settings.taa_enabled);
        taa.set_history_weight(1.0 - settings.temporal_alpha);
        let mut entries: Vec<_> = (0..6).map(sampled).collect();
        entries.push(wgpu::BindGroupLayoutEntry {
            binding: 8,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Uint,
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        });
        let surface_kind = tracked_create_texture(
            device,
            &wgpu::TextureDescriptor {
                label: Some("concordia.surface_kind"),
                size: wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::R8Uint,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::TEXTURE_BINDING
                    | wgpu::TextureUsages::COPY_SRC,
                view_formats: &[],
            },
        )?;
        let surface_kind_view = surface_kind.create_view(&Default::default());
        effects.configure_concordia_gtao(device)?;
        if settings.ssao_enabled && params.decoded().height_ao.enabled {
            effects.set_ssao_contact_kind(device, surface_kind.create_view(&Default::default()))?;
        }
        entries.push(wgpu::BindGroupLayoutEntry {
            binding: 6,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        });
        entries.push(wgpu::BindGroupLayoutEntry {
            binding: 7,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::StorageTexture {
                access: wgpu::StorageTextureAccess::WriteOnly,
                format: wgpu::TextureFormat::Rgba16Float,
                view_dimension: wgpu::TextureViewDimension::D2,
            },
            count: None,
        });
        let compose_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("concordia.compose.layout"),
            entries: &entries,
        });
        let shader = crate::core::shader_registry::create_labeled_shader_module(
            device,
            "concordia.compose.shader",
            include_str!("../../shaders/concordia_compose.wgsl"),
        );
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("concordia.compose.pipeline_layout"),
            bind_group_layouts: &[&compose_layout],
            push_constant_ranges: &[],
        });
        let pipeline = |entry: &str| {
            crate::core::shader_registry::create_compute_pipeline_scoped(
                device,
                &wgpu::ComputePipelineDescriptor {
                    label: Some(entry),
                    layout: Some(&layout),
                    module: &shader,
                    entry_point: entry,
                },
            )
        };
        let ao_pipeline = pipeline("combine_ao");
        let gi_pipeline = pipeline("add_gi");
        let ssr_pipeline = pipeline("add_reflections");
        let fog_pipeline = pipeline("composite_fog");
        let froxels = if params.decoded().volumetrics.enabled {
            Some(fog::FroxelState::new(scene, params, &frame_layout)?)
        } else {
            None
        };
        let compose_uniform = tracked_create_buffer_init(
            device,
            &wgpu::util::BufferInitDescriptor {
                label: Some("concordia.compose.uniform"),
                contents: bytemuck::bytes_of(&ComposeUniforms::zeroed()),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            },
        )?;
        let scratch = [
            hdr_texture(device, width, height, "concordia.hdr.0")?,
            hdr_texture(device, width, height, "concordia.hdr.1")?,
        ];
        let scratch_views = [
            scratch[0].create_view(&Default::default()),
            scratch[1].create_view(&Default::default()),
        ];
        let white = tracked_create_texture(
            device,
            &wgpu::TextureDescriptor {
                label: Some("concordia.visibility.identity"),
                size: wgpu::Extent3d {
                    width: 1,
                    height: 1,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8Unorm,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            },
        )?;
        queue.write_texture(
            wgpu::ImageCopyTexture {
                texture: &white,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &[255u8; 4],
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(4),
                rows_per_image: Some(1),
            },
            wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
        );
        let white_view = white.create_view(&Default::default());
        Ok(Self {
            settings_key: settings::key(params),
            height_ao_active: params.decoded().height_ao.enabled,
            height_ao_pipeline: {
                let source = shaders::height_ao_source()?;
                let module = crate::core::shader_registry::create_labeled_shader_module(
                    device,
                    "concordia.height_ao.shader",
                    &source,
                );
                let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("concordia.height_ao.layout"),
                    bind_group_layouts: &[&scene.height_ao_bind_group_layout],
                    push_constant_ranges: &[],
                });
                crate::core::shader_registry::try_create_compute_pipeline_scoped(
                    device,
                    &wgpu::ComputePipelineDescriptor {
                        label: Some("concordia.height_ao.pipeline"),
                        layout: Some(&layout),
                        module: &module,
                        entry_point: "main",
                    },
                )
                .map_err(anyhow::Error::msg)?
            },
            terrain_signature: None,
            radiance_signature: None,
            geometry_signature: None,
            chronos_identity: None,
            current_uniforms: FrameUniforms::zeroed(),
            geometry_hdr: None,
            point_layers: Vec::new(),
            label_batches: Vec::new(),
            vector_batches: Vec::new(),
            cartography_depth: None,
            _surface_kind: surface_kind,
            surface_kind_view,
            effects,
            geometry_pipeline,
            frame_layout,
            frame_bind_group,
            frame_uniform,
            cloud,
            taa,
            atmosphere_pipeline,
            atmosphere_empty_group,
            atmosphere_layout,
            atmosphere_group: Mutex::new(None),
            previous_view_proj: glam::Mat4::IDENTITY,
            previous_camera_mode: String::new(),
            frame_index: 0,
            background_pipeline: TerrainScene::create_depth_blit_pipeline(
                device,
                &scene.blit_bind_group_layout,
                wgpu::TextureFormat::Rgba16Float,
                1,
            ),
            background_hdr_pipeline: TerrainScene::create_aether_depth_blit_pipeline(
                device,
                &scene.blit_bind_group_layout,
                wgpu::TextureFormat::Rgba16Float,
                1,
            ),
            compose_layout,
            ao_pipeline,
            gi_pipeline,
            ssr_pipeline,
            fog_pipeline,
            froxels,
            compose_uniform,
            scratch,
            scratch_views,
            white,
            white_view,
            current_scratch: 0,
            composition_encoded: false,
        })
    }

    pub(super) fn prepare(
        &mut self,
        scene: &TerrainScene,
        params: &crate::terrain::render_params::TerrainRenderParams,
        clouds: bool,
        terrain_signature: u64,
        radiance_signature: [u8; 32],
    ) -> Result<bool> {
        self.composition_encoded = false;
        let (eye, view, mut proj) = camera_matrices(params);
        let current = proj * view;
        let chronos = params
            .chronos_frame
            .as_ref()
            .map(|frame| (frame.base_seed(), frame.frame_index()));
        let discontinuous = match (self.chronos_identity, chronos) {
            (Some((old_seed, old_index)), Some((seed, index))) => {
                old_seed != seed || old_index.checked_add(1) != Some(index)
            }
            (None, None) => false,
            _ => self.frame_index > 0,
        };
        let reset = self.frame_index == 0
            || self.previous_camera_mode != params.camera_mode
            || self.terrain_signature != Some(terrain_signature)
            || self.geometry_signature != scene.concordia_geometry_signature
            || self.radiance_signature != Some(radiance_signature)
            || discontinuous;
        if reset {
            self.frame_index = 0;
            let seed = params
                .chronos_frame
                .as_ref()
                .map(|frame| frame.base_seed())
                .or(params.aa_seed)
                .unwrap_or(0);
            self.effects
                .set_gi_seed(&scene.device, &scene.queue, (seed ^ (seed >> 32)) as u32)?;
            self.previous_view_proj = current;
            self.taa.reset_history();
            self.effects.ssgi_reset_history();
        }
        let jitter = if params.decoded().screen_space.taa_enabled && !reset {
            let (x, y) = crate::core::jitter::halton_2_3(
                self.frame_index,
                crate::core::jitter::DEFAULT_JITTER_SEQUENCE_LENGTH,
            );
            // Homogeneous clip translation: offset is multiplied by clip.w.
            let translation = glam::Mat4::from_translation(glam::Vec3::new(
                2.0 * x / params.size_px.0 as f32,
                -2.0 * y / params.size_px.1 as f32,
                0.0,
            ));
            proj = translation * proj;
            [x, y]
        } else {
            [0.0, 0.0]
        };
        let scale = 1.0 / params.terrain_span.max(f32::MIN_POSITIVE);
        let cloud_scale_bias = [scale, scale, 0.5, 0.5];
        let uniforms = FrameUniforms {
            view: view.to_cols_array_2d(),
            inv_view: view.inverse().to_cols_array_2d(),
            proj: proj.to_cols_array_2d(),
            inv_proj: proj.inverse().to_cols_array_2d(),
            previous_view_proj: self.previous_view_proj.to_cols_array_2d(),
            cloud_scale_bias,
            geometry: [
                (params.decoded().clamp.height_range.1 - params.decoded().clamp.height_range.0)
                    * params.z_scale
                    * 0.5,
                0.0, // Reserved; geometry.zw contain temporal jitter.
                jitter[0],
                jitter[1],
            ],
            effects: [
                u32::from(clouds),
                params.size_px.0,
                params.size_px.1,
                params.exposure.to_bits(),
            ],
        };
        self.current_uniforms = uniforms;
        scene
            .queue
            .write_buffer(&self.frame_uniform, 0, bytemuck::bytes_of(&uniforms));
        let settings = &params.decoded().screen_space;
        scene.queue.write_buffer(
            &self.compose_uniform,
            0,
            bytemuck::bytes_of(&ComposeUniforms {
                inv_view: view.inverse().to_cols_array_2d(),
                inv_proj: proj.inverse().to_cols_array_2d(),
                terrain_scale_bias: cloud_scale_bias,
                weights: [
                    settings.ssao_intensity,
                    settings.ssgi_intensity,
                    settings.ssr_intensity,
                    0.0,
                ],
            }),
        );
        self.effects.update_camera(
            &scene.queue,
            &CameraParams {
                view_matrix: view.to_cols_array_2d(),
                inv_view_matrix: view.inverse().to_cols_array_2d(),
                proj_matrix: proj.to_cols_array_2d(),
                inv_proj_matrix: proj.inverse().to_cols_array_2d(),
                prev_view_proj_matrix: self.previous_view_proj.to_cols_array_2d(),
                camera_pos: eye.to_array(),
                frame_index: self.frame_index,
                jitter_offset: jitter,
                ..Default::default()
            },
        );
        self.taa
            .update_settings(&scene.queue, [-jitter[0], -jitter[1]], self.frame_index);
        self.previous_view_proj = current;
        self.previous_camera_mode = params.camera_mode.clone();
        self.frame_index = self.frame_index.wrapping_add(1);
        self.current_scratch = 0;
        self.terrain_signature = Some(terrain_signature);
        self.radiance_signature = Some(radiance_signature);
        self.geometry_signature = scene.concordia_geometry_signature;
        self.chronos_identity = chronos;
        Ok(reset)
    }

    pub(super) fn cartographic_depth_view(&self) -> &wgpu::TextureView {
        self.cartography_depth
            .as_ref()
            .map(|depth| &depth.linear_view)
            .unwrap_or(&self.gbuffer().depth_view)
    }

    pub(super) fn gbuffer(&self) -> &GBuffer {
        self.effects.gbuffer()
    }

    pub(super) fn compose(
        &mut self,
        scene: &TerrainScene,
        encoder: &mut wgpu::CommandEncoder,
        input: &wgpu::TextureView,
        horizon: &wgpu::TextureView,
        effect: ComposePass,
    ) -> Result<()> {
        let output_index = 1 - self.current_scratch;
        let entries = [
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(input),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(
                    self.effects.ao_resolved_view().unwrap_or(&self.white_view),
                ),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::TextureView(horizon),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: wgpu::BindingResource::TextureView(
                    if matches!(effect, ComposePass::Fog) {
                        self.froxels.as_ref().expect("enabled froxels").view()
                    } else {
                        self.effects
                            .ssgi_output_for_display_view()
                            .unwrap_or(&self.white_view)
                    },
                ),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: wgpu::BindingResource::TextureView(
                    self.effects.ssr_final_view().unwrap_or(&self.white_view),
                ),
            },
            wgpu::BindGroupEntry {
                binding: 5,
                resource: wgpu::BindingResource::TextureView(&self.gbuffer().depth_view),
            },
            wgpu::BindGroupEntry {
                binding: 6,
                resource: self.compose_uniform.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 7,
                resource: wgpu::BindingResource::TextureView(&self.scratch_views[output_index]),
            },
            wgpu::BindGroupEntry {
                binding: 8,
                resource: wgpu::BindingResource::TextureView(&self.surface_kind_view),
            },
        ];
        let group = scene.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("concordia.compose.bind_group"),
            layout: &self.compose_layout,
            entries: &entries,
        });
        crate::core::shader_registry::record_shader_use("concordia.compose.shader");
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("concordia.compose"),
            timestamp_writes: None,
        });
        pass.set_pipeline(match effect {
            ComposePass::Ao => &self.ao_pipeline,
            ComposePass::Gi => &self.gi_pipeline,
            ComposePass::Reflections => &self.ssr_pipeline,
            ComposePass::Fog => &self.fog_pipeline,
        });
        pass.set_bind_group(0, &group, &[]);
        let (width, height) = self.gbuffer().dimensions();
        pass.dispatch_workgroups(width.div_ceil(8), height.div_ceil(8), 1);
        self.current_scratch = output_index;
        self.composition_encoded = true;
        Ok(())
    }
}

impl ConcordiaFrame {
    fn atmosphere(
        &mut self,
        scene: &TerrainScene,
        encoder: &mut wgpu::CommandEncoder,
        input: &wgpu::TextureView,
    ) -> Result<wgpu::TextureView> {
        let index = 1 - self.current_scratch;
        let group = scene.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("concordia.atmosphere.bind_group"),
            layout: &self.atmosphere_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.frame_uniform.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(self.cloud.shadow_view()),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(self.cloud.shadow_sampler()),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(input),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::TextureView(&self.gbuffer().depth_view),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: wgpu::BindingResource::TextureView(&self.scratch_views[index]),
                },
            ],
        });
        let atmosphere = self
            .atmosphere_group
            .lock()
            .map_err(|_| anyhow!("CONCORDIA atmosphere mutex poisoned"))?;
        let atmosphere = atmosphere
            .as_ref()
            .ok_or_else(|| anyhow!("geometry did not supply the atmosphere resources"))?;
        crate::core::shader_registry::record_shader_use("concordia.atmosphere.shader");
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("concordia.atmosphere"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.atmosphere_pipeline);
        for index in [0, 1, 2, 3, 5, 6] {
            pass.set_bind_group(index, &self.atmosphere_empty_group, &[]);
        }
        pass.set_bind_group(4, atmosphere, &[]);
        pass.set_bind_group(7, &group, &[]);
        let (width, height) = self.gbuffer().dimensions();
        pass.dispatch_workgroups(width.div_ceil(8), height.div_ceil(8), 1);
        self.current_scratch = index;
        self.composition_encoded = true;
        Ok(self.scratch[index].create_view(&Default::default()))
    }
}

/// Private geometry selector; public recipe camera keys are unchanged.
pub(crate) const SCREEN_GEOMETRY_MODE: &str = "mesh:zup:concordia-screen";

/// Screen mode uses full-tile orthographic XY framing and real shared depth.
/// The legacy fullscreen UV clamp is deliberately not reproduced here.
pub(crate) fn camera_matrices(
    params: &crate::terrain::render_params::TerrainRenderParams,
) -> (glam::Vec3, glam::Mat4, glam::Mat4) {
    if params.camera_mode != "screen" && params.camera_mode != SCREEN_GEOMETRY_MODE {
        return TerrainScene::build_camera_matrices(params);
    }
    let half = params.terrain_span * 0.5;
    let range = params.decoded().clamp.height_range;
    let eye = glam::Vec3::new(
        0.0,
        0.0,
        params.cam_radius + (range.1 - range.0).abs() * params.z_scale.abs(),
    );
    (
        eye,
        glam::Mat4::look_at_rh(eye, glam::Vec3::ZERO, glam::Vec3::Y),
        glam::Mat4::orthographic_rh(-half, half, -half, half, params.clip.0, params.clip.1),
    )
}
