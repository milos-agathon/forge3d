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
            Self::create_blit_pipeline(device.as_ref(), &blit_bind_group_layout, color_format);

        let accumulation_resources = create_accumulation_init_resources(device.as_ref());
        let accumulation_bind_group_layout = accumulation_resources.accumulation_bind_group_layout;
        let accumulation_pipeline = accumulation_resources.accumulation_pipeline;
        let accumulation_params_buffer = accumulation_resources.accumulation_params_buffer;

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
            sampler_linear,
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
            accumulation_bind_group_layout,
            accumulation_pipeline,
            accumulation_texture: Mutex::new(None),
            accumulation_view: Mutex::new(None),
            accumulation_size: Mutex::new((0, 0)),
            accumulation_params_buffer,
            material_layer_bind_group_layout,
            material_layer_uniform_buffer,
            #[cfg(feature = "enable-renderer-config")]
            config: Arc::new(Mutex::new(crate::render::params::RendererConfig::default())),
            aov_pipeline: Mutex::new(None),
            aov_pipeline_sample_count: Mutex::new(1),
            dof_renderer: Mutex::new(None),
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
