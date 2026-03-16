use super::*;

impl TerrainScene {
    /// Preprocess terrain shader by resolving #include directives
    /// WGSL doesn't have a preprocessor, so we manually expand includes
    pub(super) fn preprocess_terrain_shader() -> String {
        // Helper to strip #include lines from a shader source
        fn strip_includes(source: &str) -> String {
            source
                .lines()
                .filter(|l| !l.trim_start().starts_with("#include"))
                .collect::<Vec<_>>()
                .join("\n")
        }

        // Load nested includes for lighting.wgsl
        let lights = include_str!("../../shaders/lights.wgsl");

        // Load BRDF dispatch and its includes
        let brdf_common = include_str!("../../shaders/brdf/common.wgsl");
        let brdf_lambert = include_str!("../../shaders/brdf/lambert.wgsl");
        let brdf_phong = include_str!("../../shaders/brdf/phong.wgsl");
        let brdf_oren_nayar = include_str!("../../shaders/brdf/oren_nayar.wgsl");
        let brdf_cook_torrance = include_str!("../../shaders/brdf/cook_torrance.wgsl");
        let brdf_disney_principled = include_str!("../../shaders/brdf/disney_principled.wgsl");
        let brdf_ashikhmin_shirley = include_str!("../../shaders/brdf/ashikhmin_shirley.wgsl");
        let brdf_ward = include_str!("../../shaders/brdf/ward.wgsl");
        let brdf_toon = include_str!("../../shaders/brdf/toon.wgsl");
        let brdf_minnaert = include_str!("../../shaders/brdf/minnaert.wgsl");

        let brdf_dispatch_raw = include_str!("../../shaders/brdf/dispatch.wgsl");
        let brdf_dispatch = strip_includes(brdf_dispatch_raw);

        // Load lighting.wgsl and strip its includes
        let lighting_raw = include_str!("../../shaders/lighting.wgsl");
        let lighting = strip_includes(lighting_raw);

        // Load lighting_ibl.wgsl (no includes)
        let lighting_ibl = include_str!("../../shaders/lighting_ibl.wgsl");

        // Load main terrain shader and strip includes
        // Shader version: 2024-01-water-blue-fix
        let terrain_raw = include_str!("../../shaders/terrain_pbr_pom.wgsl");
        let terrain = strip_includes(terrain_raw);

        // Concatenate in dependency order
        format!(
            "{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}",
            lights,
            brdf_common,
            brdf_lambert,
            brdf_phong,
            brdf_oren_nayar,
            brdf_cook_torrance,
            brdf_disney_principled,
            brdf_ashikhmin_shirley,
            brdf_ward,
            brdf_toon,
            brdf_minnaert,
            brdf_dispatch,
            lighting,
            lighting_ibl,
            terrain
        )
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
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("terrain_pbr_pom.shader"),
            source: wgpu::ShaderSource::Wgsl(shader_source.into()),
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("terrain_pbr_pom.pipeline_layout"),
            bind_group_layouts: &[
                bind_group_layout,         // @group(0): terrain uniforms/textures (bindings 0-11)
                light_buffer_layout,       // @group(1): lights (bindings 3-5)
                ibl_bind_group_layout,     // @group(2): IBL (bindings 0-4)
                &shadow_bind_group_layout, // @group(3): shadows (bindings 0-4)
                fog_bind_group_layout,     // @group(4): fog (binding 0)
                water_reflection_bind_group_layout, // @group(5): water reflections (bindings 0-2)
                material_layer_bind_group_layout, // @group(6): material layers (binding 0)
            ],
            push_constant_ranges: &[],
        });

        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
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
            depth_stencil: None,
            multisample: wgpu::MultisampleState {
                count: sample_count,
                ..Default::default()
            },
            multiview: None,
        })
    }

    pub(super) fn create_blit_pipeline(
        device: &wgpu::Device,
        bind_group_layout: &wgpu::BindGroupLayout,
        color_format: wgpu::TextureFormat,
    ) -> wgpu::RenderPipeline {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("terrain.blit.shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../../shaders/terrain_blit.wgsl").into(),
            ),
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("terrain.blit.pipeline_layout"),
            bind_group_layouts: &[bind_group_layout],
            push_constant_ranges: &[],
        });

        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("terrain.blit.pipeline"),
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
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
        })
    }

    /// M1: Create AOV-enabled render pipeline with multiple render targets
    /// This pipeline outputs to 4 color targets: beauty, albedo, normal, depth
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
    ) -> wgpu::RenderPipeline {
        let shader_source = Self::preprocess_terrain_shader();
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("terrain_pbr_pom.aov.shader"),
            source: wgpu::ShaderSource::Wgsl(shader_source.into()),
        });

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
        // Target 2: Normal (world-space normals encoded to [0,1])
        // Target 3: Depth (linear depth normalized)
        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("terrain_pbr_pom.aov.pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_main",
                targets: &[
                    // Target 0: Beauty
                    Some(wgpu::ColorTargetState {
                        format: color_format,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    }),
                    // Target 1: Albedo (Rgba8Unorm for storage efficiency)
                    Some(wgpu::ColorTargetState {
                        format: wgpu::TextureFormat::Rgba8Unorm,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    }),
                    // Target 2: Normal (Rgba8Unorm, normals encoded to [0,1])
                    Some(wgpu::ColorTargetState {
                        format: wgpu::TextureFormat::Rgba8Unorm,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    }),
                    // Target 3: Depth (Rgba8Unorm for compatibility, could use R32Float)
                    Some(wgpu::ColorTargetState {
                        format: wgpu::TextureFormat::Rgba8Unorm,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    }),
                ],
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState {
                count: sample_count,
                ..Default::default()
            },
            multiview: None,
        })
    }
}
