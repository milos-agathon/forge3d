#[cfg(feature = "enable-gpu-instancing")]
use super::core::TERRAIN_DEPTH_FORMAT;
use super::*;

impl TerrainScene {
    /// Internal constructor used by Python and (later) the viewer.
    pub fn new(
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
        adapter: Arc<wgpu::Adapter>,
    ) -> Result<Self> {
        let base_layouts = create_base_bind_group_layouts(device.as_ref());
        let bind_group_layout = base_layouts.bind_group_layout;
        let ibl_bind_group_layout = base_layouts.ibl_bind_group_layout;
        let blit_bind_group_layout = base_layouts.blit_bind_group_layout;

        let base_resources = create_base_init_resources(device.as_ref(), queue.as_ref())?;
        let sampler_linear = base_resources.sampler_linear;
        let height_curve_lut_sampler = base_resources.height_curve_lut_sampler;
        let atmosphere_resources =
            create_atmosphere_init_resources(device.as_ref(), queue.as_ref());
        let sky_bind_group_layout0 = atmosphere_resources.sky_bind_group_layout0;
        let sky_bind_group_layout1 = atmosphere_resources.sky_bind_group_layout1;
        let sky_pipeline = atmosphere_resources.sky_pipeline;
        let sky_fallback_texture = atmosphere_resources.sky_fallback_texture;
        let sky_fallback_view = atmosphere_resources.sky_fallback_view;
        let height_curve_identity_texture = base_resources.height_curve_identity_texture;
        let height_curve_identity_view = base_resources.height_curve_identity_view;
        let water_mask_fallback_texture = base_resources.water_mask_fallback_texture;
        let water_mask_fallback_view = base_resources.water_mask_fallback_view;
        let detail_normal_fallback_view = base_resources.detail_normal_fallback_view;
        let detail_normal_sampler = base_resources.detail_normal_sampler;

        let heightfield_resources =
            create_heightfield_init_resources(device.as_ref(), queue.as_ref());
        let ao_debug_sampler = heightfield_resources.ao_debug_sampler;
        let ao_debug_fallback_texture = heightfield_resources.ao_debug_fallback_texture;
        let ao_debug_fallback_view = heightfield_resources.ao_debug_fallback_view;
        let height_ao_fallback_view = heightfield_resources.height_ao_fallback_view;
        let height_ao_sampler = heightfield_resources.height_ao_sampler;
        let height_ao_compute_pipeline = heightfield_resources.height_ao_compute_pipeline;
        let height_ao_bind_group_layout = heightfield_resources.height_ao_bind_group_layout;
        let height_ao_uniform_buffer = heightfield_resources.height_ao_uniform_buffer;
        let sun_vis_fallback_view = heightfield_resources.sun_vis_fallback_view;
        let sun_vis_sampler = heightfield_resources.sun_vis_sampler;
        let sun_vis_compute_pipeline = heightfield_resources.sun_vis_compute_pipeline;
        let sun_vis_bind_group_layout = heightfield_resources.sun_vis_bind_group_layout;
        let sun_vis_uniform_buffer = heightfield_resources.sun_vis_uniform_buffer;

        let light_buffer = LightBuffer::new(&device);
        let color_format = wgpu::TextureFormat::Rgba8Unorm;
        let light_buffer_layout = light_buffer.bind_group_layout();

        let shadow_bind_group_layout = Self::create_shadow_bind_group_layout(device.as_ref());

        let fog_bind_group_layout = Self::create_fog_bind_group_layout(device.as_ref());
        let fog_uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("terrain.fog.uniform_buffer"),
            contents: bytemuck::bytes_of(&FogUniforms::disabled()),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let water_reflection_bind_group_layout =
            Self::create_water_reflection_bind_group_layout(device.as_ref());
        let water_reflection_resources =
            create_water_reflection_init_resources(device.as_ref(), queue.as_ref(), color_format);
        let water_reflection_uniform_buffer =
            water_reflection_resources.water_reflection_uniform_buffer;
        let water_reflection_texture = water_reflection_resources.water_reflection_texture;
        let water_reflection_view = water_reflection_resources.water_reflection_view;
        let water_reflection_sampler = water_reflection_resources.water_reflection_sampler;
        let water_reflection_depth_texture =
            water_reflection_resources.water_reflection_depth_texture;
        let water_reflection_depth_view = water_reflection_resources.water_reflection_depth_view;
        let water_reflection_size = water_reflection_resources.water_reflection_size;
        let water_reflection_fallback_view =
            water_reflection_resources.water_reflection_fallback_view;

        let material_layer_bind_group_layout =
            Self::create_material_layer_bind_group_layout(device.as_ref());
        let material_layer_uniform_buffer =
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("terrain.material_layer.uniform_buffer"),
                contents: bytemuck::bytes_of(&MaterialLayerUniforms::disabled()),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

        // VT fallback resources
        let vt_atlas_fallback_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("vt_atlas_fallback"),
            size: wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });

        let vt_atlas_fallback_view =
            vt_atlas_fallback_texture.create_view(&wgpu::TextureViewDescriptor {
                label: Some("vt_atlas_fallback_view"),
                format: Some(wgpu::TextureFormat::Rgba8UnormSrgb),
                dimension: Some(wgpu::TextureViewDimension::D2),
                ..Default::default()
            });

        queue.write_texture(
            wgpu::ImageCopyTexture {
                texture: &vt_atlas_fallback_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &[255, 255, 255, 255],
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(4),
                rows_per_image: None,
            },
            wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
        );

        let vt_page_table_fallback_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("vt_page_table_fallback"),
            size: wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: super::core::MATERIAL_LAYER_CAPACITY as u32,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba32Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });

        let vt_page_table_fallback_view =
            vt_page_table_fallback_texture.create_view(&wgpu::TextureViewDescriptor {
                label: Some("vt_page_table_fallback_view"),
                format: Some(wgpu::TextureFormat::Rgba32Float),
                dimension: Some(wgpu::TextureViewDimension::D2Array),
                base_mip_level: 0,
                mip_level_count: Some(1),
                base_array_layer: 0,
                array_layer_count: Some(super::core::MATERIAL_LAYER_CAPACITY as u32),
                ..Default::default()
            });

        let vt_page_table_fallback_data = vec![0u8; 16 * super::core::MATERIAL_LAYER_CAPACITY];
        queue.write_texture(
            wgpu::ImageCopyTexture {
                texture: &vt_page_table_fallback_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &vt_page_table_fallback_data,
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(16),
                rows_per_image: Some(1),
            },
            wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: super::core::MATERIAL_LAYER_CAPACITY as u32,
            },
        );

        let vt_uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("vt_uniforms"),
            size: 64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let vt_fallback_uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("vt_fallback_colors"),
            size: 256,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let vt_feedback_fallback_buffer =
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("vt_feedback_fallback"),
                contents: bytemuck::cast_slice(&[0u32; 4]),
                usage: wgpu::BufferUsages::STORAGE,
            });

        let vt_atlas_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("vt_atlas_sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });
        let probe_grid_uniform_buffer =
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("terrain.probes.grid_uniform_buffer"),
                contents: bytemuck::bytes_of(
                    &crate::terrain::probes::ProbeGridUniformsGpu::disabled(),
                ),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });
        let probe_ssbo_init = [crate::terrain::probes::GpuProbeData::zeroed()];
        let probe_ssbo = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("terrain.probes.ssbo"),
            contents: bytemuck::cast_slice(&probe_ssbo_init),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });
        let probe_grid_uniform_alloc_bytes =
            std::mem::size_of::<crate::terrain::probes::ProbeGridUniformsGpu>() as u64;
        let probe_ssbo_alloc_bytes =
            std::mem::size_of::<crate::terrain::probes::GpuProbeData>() as u64;
        let reflection_probe_grid_uniform_buffer =
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("terrain.reflection_probes.grid_uniform_buffer"),
                contents: bytemuck::bytes_of(
                    &crate::terrain::probes::ReflectionProbeGridUniformsGpu::disabled(),
                ),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });
        let reflection_probe_grid_uniform_alloc_bytes =
            std::mem::size_of::<crate::terrain::probes::ReflectionProbeGridUniformsGpu>() as u64;
        let reflection_probe_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("terrain.reflection_probes.sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let reflection_probe_fallback_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("terrain.reflection_probes.fallback_texture"),
            size: wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 6,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let reflection_probe_fallback_zeroes = [0u16; 4 * 6];
        queue.write_texture(
            wgpu::ImageCopyTexture {
                texture: &reflection_probe_fallback_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            bytemuck::cast_slice(&reflection_probe_fallback_zeroes),
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(8),
                rows_per_image: Some(1),
            },
            wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 6,
            },
        );
        let reflection_probe_fallback_view =
            reflection_probe_fallback_texture.create_view(&wgpu::TextureViewDescriptor {
                label: Some("terrain.reflection_probes.fallback_view"),
                format: Some(wgpu::TextureFormat::Rgba16Float),
                dimension: Some(wgpu::TextureViewDimension::D2Array),
                aspect: wgpu::TextureAspect::All,
                base_mip_level: 0,
                mip_level_count: Some(1),
                base_array_layer: 0,
                array_layer_count: Some(6),
            });
        let reflection_probe_view =
            reflection_probe_fallback_texture.create_view(&wgpu::TextureViewDescriptor {
                label: Some("terrain.reflection_probes.active_view"),
                format: Some(wgpu::TextureFormat::Rgba16Float),
                dimension: Some(wgpu::TextureViewDimension::D2Array),
                aspect: wgpu::TextureAspect::All,
                base_mip_level: 0,
                mip_level_count: Some(1),
                base_array_layer: 0,
                array_layer_count: Some(6),
            });

        let pipeline = Self::create_render_pipeline(
            device.as_ref(),
            &bind_group_layout,
            light_buffer_layout,
            &ibl_bind_group_layout,
            &shadow_bind_group_layout,
            &fog_bind_group_layout,
            &water_reflection_bind_group_layout,
            &material_layer_bind_group_layout,
            color_format,
            1,
        );

        let water_reflection_pipeline = Self::create_render_pipeline(
            device.as_ref(),
            &bind_group_layout,
            light_buffer_layout,
            &ibl_bind_group_layout,
            &shadow_bind_group_layout,
            &fog_bind_group_layout,
            &water_reflection_bind_group_layout,
            &material_layer_bind_group_layout,
            color_format,
            1,
        );

        let blit_pipeline =
            Self::create_blit_pipeline(device.as_ref(), &blit_bind_group_layout, color_format, 1);
        let aov_blit_pipeline = Self::create_blit_pipeline(
            device.as_ref(),
            &blit_bind_group_layout,
            wgpu::TextureFormat::Rgba16Float,
            1,
        );
        let background_blit_pipeline = Self::create_depth_blit_pipeline(
            device.as_ref(),
            &blit_bind_group_layout,
            color_format,
            1,
        );
        let normal_blit_pipeline = Self::create_normal_blit_pipeline(
            device.as_ref(),
            &blit_bind_group_layout,
            wgpu::TextureFormat::Rgba16Float,
            1,
        );
        let offline_compute = Self::create_offline_compute_resources(device.as_ref());

        #[cfg(feature = "enable-gpu-instancing")]
        // Terrain scatter is composited after the terrain pass. Keep the shared instancing
        // renderer reusable, but avoid inheriting terrain-depth mismatches into this path.
        let scatter_renderer =
            crate::render::mesh_instanced::MeshInstancedRenderer::new_with_depth_state(
                device.as_ref(),
                color_format,
                Some(TERRAIN_DEPTH_FORMAT),
                1,
                wgpu::CompareFunction::Always,
                false,
            );

        let noop_shadow =
            Self::create_noop_shadow(device.as_ref(), queue.as_ref(), &shadow_bind_group_layout)?;

        let shadow_debug_mode = crate::core::shadows::parse_shadow_debug_env();
        let csm_config = crate::shadows::CsmConfig {
            cascade_count: 4,
            shadow_map_size: 2048,
            max_shadow_distance: 3000.0,
            pcf_kernel_size: 3,
            depth_bias: 0.0005,
            slope_bias: 0.001,
            peter_panning_offset: 0.0002,
            enable_evsm: true,
            stabilize_cascades: true,
            cascade_blend_range: 0.1,
            debug_mode: shadow_debug_mode,
            ..Default::default()
        };
        if shadow_debug_mode > 0 {
            log::info!(
                target: "terrain.shadow",
                "Shadow debug mode enabled: {} (FORGE3D_TERRAIN_SHADOW_DEBUG)",
                shadow_debug_mode
            );
        }
        let csm_renderer = crate::shadows::CsmRenderer::new(device.as_ref(), csm_config);

        let shadow_depth_bind_group_layout =
            Self::create_shadow_depth_bind_group_layout(device.as_ref());
        let shadow_depth_pipeline =
            Self::create_shadow_depth_pipeline(device.as_ref(), &shadow_depth_bind_group_layout);

        let tracker = crate::core::memory_tracker::global_tracker();
        tracker.track_buffer_allocation(probe_grid_uniform_alloc_bytes, false);
        tracker.track_buffer_allocation(probe_ssbo_alloc_bytes, false);
        tracker.track_buffer_allocation(reflection_probe_grid_uniform_alloc_bytes, false);

        let pipeline_cache = PipelineCache {
            sample_count: 1,
            pipeline,
        };

        Ok(Self {
            device,
            queue,
            adapter,
            pipeline: Mutex::new(pipeline_cache),
            bind_group_layout,
            ibl_bind_group_layout,
            blit_bind_group_layout,
            blit_pipeline,
            aov_blit_pipeline,
            background_blit_pipeline,
            normal_blit_pipeline,
            offline_compute,
            sampler_linear,
            sky_bind_group_layout0,
            sky_bind_group_layout1,
            sky_pipeline,
            sky_fallback_texture,
            sky_fallback_view,
            height_curve_identity_texture,
            height_curve_identity_view,
            water_mask_fallback_texture,
            water_mask_fallback_view,
            ao_debug_fallback_texture,
            ao_debug_fallback_view,
            ao_debug_sampler,
            ao_debug_view: None,
            coarse_ao_texture: None,
            coarse_ao_view: None,
            detail_normal_fallback_view,
            detail_normal_sampler,
            height_ao_fallback_view,
            height_ao_sampler,
            sun_vis_fallback_view,
            sun_vis_sampler,
            height_ao_compute_pipeline,
            height_ao_bind_group_layout,
            height_ao_uniform_buffer,
            height_ao_texture: Mutex::new(None),
            height_ao_storage_view: Mutex::new(None),
            height_ao_sample_view: Mutex::new(None),
            height_ao_size: Mutex::new((0, 0)),
            sun_vis_compute_pipeline,
            sun_vis_bind_group_layout,
            sun_vis_uniform_buffer,
            sun_vis_texture: Mutex::new(None),
            sun_vis_storage_view: Mutex::new(None),
            sun_vis_sample_view: Mutex::new(None),
            sun_vis_size: Mutex::new((0, 0)),
            height_curve_lut_sampler,
            color_format,
            light_buffer: Arc::new(Mutex::new(light_buffer)),
            noop_shadow,
            csm_renderer,
            shadow_depth_pipeline,
            shadow_depth_bind_group_layout,
            shadow_bind_group_layout,
            shadow_pcss_radius: 0.0,
            shadow_technique: 1,
            moment_pass: None,
            fog_bind_group_layout,
            fog_uniform_buffer,
            water_reflection_bind_group_layout,
            water_reflection_uniform_buffer,
            water_reflection_texture: Mutex::new(water_reflection_texture),
            water_reflection_view: Mutex::new(water_reflection_view),
            water_reflection_sampler,
            water_reflection_depth_texture: Mutex::new(water_reflection_depth_texture),
            water_reflection_depth_view: Mutex::new(water_reflection_depth_view),
            water_reflection_size: Mutex::new(water_reflection_size),
            water_reflection_fallback_view,
            water_reflection_pipeline,
            material_layer_bind_group_layout,
            material_layer_uniform_buffer,
            vt_uniform_buffer,
            vt_fallback_uniform_buffer,
            vt_atlas_fallback_texture,
            vt_atlas_fallback_view,
            vt_page_table_fallback_texture,
            vt_page_table_fallback_view,
            vt_feedback_fallback_buffer,
            vt_atlas_sampler,
            probe_grid_uniform_buffer,
            probe_ssbo,
            probe_grid_uniform_alloc_bytes,
            probe_ssbo_alloc_bytes,
            probe_grid_uniform_bytes: 0,
            probe_ssbo_bytes: 0,
            probe_cache_key: None,
            probe_cached_grid: None,
            probe_cached_data: Vec::new(),
            reflection_probe_grid_uniform_buffer,
            reflection_probe_sampler,
            reflection_probe_fallback_texture,
            reflection_probe_fallback_view,
            reflection_probe_texture: None,
            reflection_probe_view,
            reflection_probe_grid_uniform_alloc_bytes,
            reflection_probe_grid_uniform_bytes: 0,
            reflection_probe_texture_alloc_bytes: 0,
            reflection_probe_texture_bytes: 0,
            reflection_probe_cache_key: None,
            reflection_probe_cached_grid: None,
            reflection_probe_count: 0,
            reflection_probe_resolution: 0,
            reflection_probe_mip_levels: 0,
            #[cfg(feature = "enable-renderer-config")]
            config: Arc::new(Mutex::new(crate::render::params::RendererConfig::default())),
            aov_pipeline: Mutex::new(None),
            aov_pipeline_sample_count: Mutex::new(1),
            dof_renderer: Mutex::new(None),
            offline_state: Mutex::new(None),
            #[cfg(feature = "enable-gpu-instancing")]
            scatter_renderer,
            #[cfg(feature = "enable-gpu-instancing")]
            scatter_renderer_sample_count: 1,
            #[cfg(feature = "enable-gpu-instancing")]
            scatter_batches: Vec::new(),
            #[cfg(feature = "enable-gpu-instancing")]
            scatter_last_frame_stats: crate::terrain::scatter::TerrainScatterFrameStats::default(),
            material_vt: Mutex::new(super::virtual_texture::TerrainMaterialVT::new()),
            viewer_heightmap: None,
        })
    }

    pub(in crate::terrain::renderer) fn map_filter_mode(
        mode: FilterModeNative,
    ) -> wgpu::FilterMode {
        match mode {
            FilterModeNative::Linear => wgpu::FilterMode::Linear,
            FilterModeNative::Nearest => wgpu::FilterMode::Nearest,
        }
    }

    pub(in crate::terrain::renderer) fn map_address_mode(
        mode: AddressModeNative,
    ) -> wgpu::AddressMode {
        match mode {
            AddressModeNative::Repeat => wgpu::AddressMode::Repeat,
            AddressModeNative::ClampToEdge => wgpu::AddressMode::ClampToEdge,
            AddressModeNative::MirrorRepeat => wgpu::AddressMode::MirrorRepeat,
        }
    }
}
