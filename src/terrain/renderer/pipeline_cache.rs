use super::*;
use crate::terrain::renderer::core::TERRAIN_DEPTH_FORMAT;

impl TerrainScene {
    fn create_fullscreen_blit_pipeline(
        device: &wgpu::Device,
        bind_group_layout: &wgpu::BindGroupLayout,
        color_format: wgpu::TextureFormat,
        sample_count: u32,
        pipeline_label: &'static str,
        shader_label: &'static str,
        shader_source: &'static str,
        depth_stencil: Option<wgpu::DepthStencilState>,
    ) -> wgpu::RenderPipeline {
        let shader = crate::core::shader_registry::create_labeled_shader_module(
            device,
            shader_label,
            shader_source,
        );

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("terrain.blit.pipeline_layout"),
            bind_group_layouts: &[bind_group_layout],
            push_constant_ranges: &[],
        });

        crate::core::shader_registry::with_error_scope(device, pipeline_label, || {
            crate::core::shader_registry::create_render_pipeline_scoped(
                device,
                &wgpu::RenderPipelineDescriptor {
                    label: Some(pipeline_label),
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
                            format: color_format,
                            blend: None,
                            write_mask: wgpu::ColorWrites::ALL,
                        })],
                    }),
                    primitive: wgpu::PrimitiveState::default(),
                    depth_stencil,
                    multisample: wgpu::MultisampleState {
                        count: sample_count,
                        ..Default::default()
                    },
                    multiview: None,
                },
            )
        })
    }

    /// Preprocess terrain shader by resolving #include directives
    /// WGSL doesn't have a preprocessor, so we manually expand includes
    pub(super) fn preprocess_terrain_shader() -> String {
        crate::shader_sources::terrain()
    }

    pub(super) fn create_render_pipeline(
        device: &wgpu::Device,
        bind_group_layout: &wgpu::BindGroupLayout,
        light_buffer_layout: &wgpu::BindGroupLayout,
        ibl_bind_group_layout: &wgpu::BindGroupLayout,
        shadow_bind_group_layout: &wgpu::BindGroupLayout,
        fog_bind_group_layout: &wgpu::BindGroupLayout,
        water_reflection_bind_group_layout: &wgpu::BindGroupLayout,
        material_layer_bind_group_layout: &wgpu::BindGroupLayout,
        color_format: wgpu::TextureFormat,
        sample_count: u32,
    ) -> wgpu::RenderPipeline {
        let shader_source = Self::preprocess_terrain_shader();
        let shader = crate::core::shader_registry::create_labeled_shader_module(
            device,
            "terrain_pbr_pom.shader",
            &shader_source,
        );

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("terrain_pbr_pom.pipeline_layout"),
            bind_group_layouts: &[
                bind_group_layout,         // @group(0): terrain uniforms/textures (bindings 0-11)
                light_buffer_layout,       // @group(1): lights (bindings 3-5)
                ibl_bind_group_layout,     // @group(2): IBL (bindings 0-4)
                &shadow_bind_group_layout, // @group(3): shadows (bindings 0-4)
                fog_bind_group_layout,     // @group(4): fog (binding 0)
                water_reflection_bind_group_layout, // @group(5): water reflections (bindings 0-2)
                material_layer_bind_group_layout, // @group(6): material layers + probes
            ],
            push_constant_ranges: &[],
        });

        crate::core::shader_registry::with_error_scope(device, "terrain_pbr_pom.pipeline", || {
            crate::core::shader_registry::create_render_pipeline_scoped(
                device,
                &wgpu::RenderPipelineDescriptor {
                    label: Some("terrain_pbr_pom.pipeline"),
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
                            format: color_format,
                            blend: None,
                            write_mask: wgpu::ColorWrites::ALL,
                        })],
                    }),
                    primitive: wgpu::PrimitiveState::default(),
                    depth_stencil: Some(wgpu::DepthStencilState {
                        format: TERRAIN_DEPTH_FORMAT,
                        depth_write_enabled: true,
                        depth_compare: wgpu::CompareFunction::Less,
                        stencil: wgpu::StencilState::default(),
                        bias: wgpu::DepthBiasState::default(),
                    }),
                    multisample: wgpu::MultisampleState {
                        count: sample_count,
                        ..Default::default()
                    },
                    multiview: None,
                },
            )
        })
    }

    pub(super) fn create_clipmap_render_pipeline(
        device: &wgpu::Device,
        bind_group_layout: &wgpu::BindGroupLayout,
        light_buffer_layout: &wgpu::BindGroupLayout,
        ibl_bind_group_layout: &wgpu::BindGroupLayout,
        shadow_bind_group_layout: &wgpu::BindGroupLayout,
        fog_bind_group_layout: &wgpu::BindGroupLayout,
        water_reflection_bind_group_layout: &wgpu::BindGroupLayout,
        material_layer_bind_group_layout: &wgpu::BindGroupLayout,
        color_format: wgpu::TextureFormat,
        sample_count: u32,
    ) -> wgpu::RenderPipeline {
        // `vs_clipmap_main` lives in the shared terrain_pbr_pom.wgsl module, so
        // the clipmap geometry path shades through the exact same PBR fragment
        // stage as the procedural-grid path.
        let shader_source = Self::preprocess_terrain_shader();
        let shader = crate::core::shader_registry::create_labeled_shader_module(
            device,
            "terrain_pbr_pom.clipmap.shader",
            &shader_source,
        );

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("terrain_pbr_pom.clipmap.pipeline_layout"),
            bind_group_layouts: &[
                bind_group_layout,
                light_buffer_layout,
                ibl_bind_group_layout,
                &shadow_bind_group_layout,
                fog_bind_group_layout,
                water_reflection_bind_group_layout,
                material_layer_bind_group_layout,
            ],
            push_constant_ranges: &[],
        });

        crate::core::shader_registry::with_error_scope(
            device,
            "terrain_pbr_pom.clipmap.pipeline",
            || {
                crate::core::shader_registry::create_render_pipeline_scoped(
                    device,
                    &wgpu::RenderPipelineDescriptor {
                        label: Some("terrain_pbr_pom.clipmap.pipeline"),
                        layout: Some(&pipeline_layout),
                        vertex: wgpu::VertexState {
                            module: &shader,
                            entry_point: "vs_clipmap_main",
                            buffers: &[crate::terrain::clipmap::ClipmapVertex::desc()],
                        },
                        fragment: Some(wgpu::FragmentState {
                            module: &shader,
                            entry_point: "fs_main",
                            targets: &[Some(wgpu::ColorTargetState {
                                format: color_format,
                                blend: None,
                                write_mask: wgpu::ColorWrites::ALL,
                            })],
                        }),
                        primitive: wgpu::PrimitiveState::default(),
                        depth_stencil: Some(wgpu::DepthStencilState {
                            format: TERRAIN_DEPTH_FORMAT,
                            depth_write_enabled: true,
                            depth_compare: wgpu::CompareFunction::Less,
                            stencil: wgpu::StencilState::default(),
                            bias: wgpu::DepthBiasState::default(),
                        }),
                        multisample: wgpu::MultisampleState {
                            count: sample_count,
                            ..Default::default()
                        },
                        multiview: None,
                    },
                )
            },
        )
    }

    pub(super) fn create_blit_pipeline(
        device: &wgpu::Device,
        bind_group_layout: &wgpu::BindGroupLayout,
        color_format: wgpu::TextureFormat,
        sample_count: u32,
    ) -> wgpu::RenderPipeline {
        Self::create_fullscreen_blit_pipeline(
            device,
            bind_group_layout,
            color_format,
            sample_count,
            "terrain.blit.pipeline",
            "terrain.blit.shader",
            include_str!("../../shaders/terrain_blit.wgsl"),
            None,
        )
    }

    pub(super) fn create_depth_blit_pipeline(
        device: &wgpu::Device,
        bind_group_layout: &wgpu::BindGroupLayout,
        color_format: wgpu::TextureFormat,
        sample_count: u32,
    ) -> wgpu::RenderPipeline {
        // This pass clears depth for the terrain scene and draws only color, so it needs a
        // depth-compatible pipeline that never overwrites the cleared depth buffer.
        Self::create_fullscreen_blit_pipeline(
            device,
            bind_group_layout,
            color_format,
            sample_count,
            "terrain.blit.depth.pipeline",
            "terrain.blit.depth.shader",
            include_str!("../../shaders/terrain_blit.wgsl"),
            Some(wgpu::DepthStencilState {
                format: TERRAIN_DEPTH_FORMAT,
                depth_write_enabled: false,
                depth_compare: wgpu::CompareFunction::Always,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
        )
    }

    pub(super) fn create_normal_blit_pipeline(
        device: &wgpu::Device,
        bind_group_layout: &wgpu::BindGroupLayout,
        color_format: wgpu::TextureFormat,
        sample_count: u32,
    ) -> wgpu::RenderPipeline {
        Self::create_fullscreen_blit_pipeline(
            device,
            bind_group_layout,
            color_format,
            sample_count,
            "terrain.blit.normal.pipeline",
            "terrain.blit.normal.shader",
            include_str!("../../shaders/terrain_normal_blit.wgsl"),
            None,
        )
    }

    /// M1: Create AOV-enabled render pipeline with multiple render targets
    /// This pipeline outputs to 4 color targets: beauty, albedo, normal, depth.
    /// VERITAS: with `include_source_id` a 5th `R32Uint` target carries the
    /// per-pixel VT source-id map (requires sample_count == 1 — R32Uint is
    /// not multisample-capable).
    #[allow(clippy::too_many_arguments)]
    pub(super) fn create_aov_render_pipeline(
        device: &wgpu::Device,
        bind_group_layout: &wgpu::BindGroupLayout,
        light_buffer_layout: &wgpu::BindGroupLayout,
        ibl_bind_group_layout: &wgpu::BindGroupLayout,
        shadow_bind_group_layout: &wgpu::BindGroupLayout,
        fog_bind_group_layout: &wgpu::BindGroupLayout,
        water_reflection_bind_group_layout: &wgpu::BindGroupLayout,
        material_layer_bind_group_layout: &wgpu::BindGroupLayout,
        color_format: wgpu::TextureFormat,
        sample_count: u32,
        include_source_id: bool,
        clipmap_geometry: bool,
    ) -> wgpu::RenderPipeline {
        let shader_source = Self::preprocess_terrain_shader();
        let shader = crate::core::shader_registry::create_labeled_shader_module(
            device,
            "terrain_pbr_pom.aov.shader",
            &shader_source,
        );

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("terrain_pbr_pom.aov.pipeline_layout"),
            bind_group_layouts: &[
                bind_group_layout,
                light_buffer_layout,
                ibl_bind_group_layout,
                shadow_bind_group_layout,
                fog_bind_group_layout,
                water_reflection_bind_group_layout,
                material_layer_bind_group_layout,
            ],
            push_constant_ranges: &[],
        });

        // M1: AOV pipeline with 4 color targets
        // Target 0: Beauty (tonemapped color)
        // Target 1: Albedo (base color before lighting)
        // Target 2: Normal (normalized world-space normal, signed float)
        // Target 3: Depth (linear depth normalized)
        // Target 4 (optional, VERITAS): per-pixel VT source id (R32Uint)
        let mut targets = vec![
            // Target 0: Beauty
            Some(wgpu::ColorTargetState {
                format: color_format,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            }),
            // Target 1: Albedo
            Some(wgpu::ColorTargetState {
                format: wgpu::TextureFormat::Rgba16Float,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            }),
            // Target 2: Normal
            Some(wgpu::ColorTargetState {
                format: wgpu::TextureFormat::Rgba16Float,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            }),
            // Target 3: Depth
            Some(wgpu::ColorTargetState {
                format: wgpu::TextureFormat::Rgba16Float,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            }),
        ];
        if include_source_id {
            targets.push(Some(wgpu::ColorTargetState {
                format: wgpu::TextureFormat::R32Uint,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            }));
        }
        let clipmap_vertex_buffers = [crate::terrain::clipmap::ClipmapVertex::desc()];
        let (vertex_entry, vertex_buffers): (&str, &[wgpu::VertexBufferLayout]) =
            if clipmap_geometry {
                ("vs_clipmap_main", &clipmap_vertex_buffers)
            } else {
                ("vs_main", &[])
            };
        let pipeline_label = if clipmap_geometry {
            "terrain_pbr_pom.aov.clipmap.pipeline"
        } else {
            "terrain_pbr_pom.aov.pipeline"
        };
        crate::core::shader_registry::with_error_scope(device, pipeline_label, || {
            crate::core::shader_registry::create_render_pipeline_scoped(
                device,
                &wgpu::RenderPipelineDescriptor {
                    label: Some(pipeline_label),
                    layout: Some(&pipeline_layout),
                    vertex: wgpu::VertexState {
                        module: &shader,
                        entry_point: vertex_entry,
                        buffers: vertex_buffers,
                    },
                    fragment: Some(wgpu::FragmentState {
                        module: &shader,
                        entry_point: "fs_main",
                        targets: &targets,
                    }),
                    primitive: wgpu::PrimitiveState::default(),
                    depth_stencil: Some(wgpu::DepthStencilState {
                        format: TERRAIN_DEPTH_FORMAT,
                        depth_write_enabled: true,
                        depth_compare: wgpu::CompareFunction::Less,
                        stencil: wgpu::StencilState::default(),
                        bias: wgpu::DepthBiasState::default(),
                    }),
                    multisample: wgpu::MultisampleState {
                        count: sample_count,
                        ..Default::default()
                    },
                    multiview: None,
                },
            )
        })
    }
}
