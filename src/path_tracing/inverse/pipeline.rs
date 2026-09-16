// src/path_tracing/inverse/pipeline.rs
// DIFFERENTIA: bind-group layouts and compute pipelines for the inverse pass.
//
// The inverse re-dispatches the REAL forward entry points — `main_terrain`,
// `main_terrain_publish`, `main_terrain_gbuffer` and the standalone
// pt_restir_{temporal,spatial} passes — so the layouts must be supersets of
// what the forward kernels bind:
//
//   group 0: base Uniforms + LightingUniforms + InvParams (inverse-only)
//   group 1: scene storage (spheres dummy, hybrid uniforms, mesh dummies)
//   group 2: the forward accum/terrain/reservoir bindings (0..7, 10, 16)
//            plus the consolidated inverse buffers (11 = vec4 adjoint/weight
//            history, 12 = atomic<u32> scalars/loss/albedo-gradient)
//   group 3: the forward output + AOV storage textures (0..7) plus the
//            sampled observation target (8); the edge pass instead binds the
//            Normal/Depth AOVs as SAMPLED textures (9, 10) under its own
//            group-3 layout — a texture cannot be storage and sampled in the
//            same dispatch.
//
// main_terrain_gbuffer uses the forward's G-buffer group-2 variant
// (bindings 1,2,3 + 8,9,10) — the same arrangement as the forward driver.
// RELEVANT FILES: src/shaders/pt_inverse_*.wgsl, src/shader_sources.rs,
//                 src/path_tracing/hybrid_compute/layouts.rs

use crate::core::error::RenderError;

fn uniform_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn storage_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn sampled_texture_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
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

fn storage_texture_entry(binding: u32, format: wgpu::TextureFormat) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::StorageTexture {
            access: wgpu::StorageTextureAccess::WriteOnly,
            format,
            view_dimension: wgpu::TextureViewDimension::D2,
        },
        count: None,
    }
}

/// All pipelines + layouts for one inverse solve. The terrain/publish/gbuffer
/// and ReSTIR pipelines are the forward entry points re-dispatched under
/// inverse-superset layouts; the four inverse kernels are appended to the
/// same WGSL module so every forward declaration is shared verbatim.
pub struct InversePipelines {
    // ---- layouts ----
    pub inv_g0: wgpu::BindGroupLayout,
    pub inv_g1: wgpu::BindGroupLayout,
    pub inv_g2: wgpu::BindGroupLayout,
    pub inv_g3: wgpu::BindGroupLayout,
    /// Sampled-AOV group 3 for the edge pass (Normal + Depth as textures).
    pub inv_g3_edge: wgpu::BindGroupLayout,
    /// Forward G-buffer group-2 variant (terrain tex/uniforms + nr/pos).
    pub inv_g2_gbuffer: wgpu::BindGroupLayout,
    pub restir_g0: wgpu::BindGroupLayout,
    pub restir_empty: wgpu::BindGroupLayout,
    pub restir_temporal_g2: wgpu::BindGroupLayout,
    pub restir_spatial_g1: wgpu::BindGroupLayout,
    pub restir_spatial_g2: wgpu::BindGroupLayout,
    // ---- forward entries (the real primal path) ----
    pub terrain: wgpu::ComputePipeline,
    pub terrain_publish: wgpu::ComputePipeline,
    pub terrain_gbuffer: wgpu::ComputePipeline,
    pub restir_temporal: wgpu::ComputePipeline,
    pub restir_spatial: wgpu::ComputePipeline,
    // ---- inverse entries ----
    pub clear: wgpu::ComputePipeline,
    /// Gradient-region clear — once per multi-replicate eval.
    pub clear_grad: wgpu::ComputePipeline,
    /// Replicate-mean image accumulation (1/R-weighted linear radiance).
    pub accum_mean: wgpu::ComputePipeline,
    /// Reservoir-only clear for the checkpointed adjoint replay.
    pub replay_clear: wgpu::ComputePipeline,
    pub wsnap: wgpu::ComputePipeline,
    pub loss: wgpu::ComputePipeline,
    pub shade: wgpu::ComputePipeline,
    pub edge: wgpu::ComputePipeline,
}

impl InversePipelines {
    pub fn new(device: &wgpu::Device) -> Result<Self, RenderError> {
        let inv_g0 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("inv-bgl0"),
            entries: &[
                uniform_entry(0), // Uniforms (the forward's own struct)
                uniform_entry(1), // LightingUniforms
                uniform_entry(2), // InvParams
            ],
        });
        // Identical to the forward scene layout (spheres, hybrid uniforms,
        // mesh vertices/indices/bvh).
        let inv_g1 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("inv-bgl1-scene"),
            entries: &[
                storage_entry(0, true),
                uniform_entry(1),
                storage_entry(2, true),
                storage_entry(3, true),
                storage_entry(4, true),
            ],
        });
        // Forward accum layout (0..7, earth curvature at 10, albedo at 16)
        // plus the consolidated inverse buffers at 11 (vec4) and 12
        // (atomic<u32>).
        let inv_g2 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("inv-bgl2-state"),
            entries: &[
                storage_entry(0, false),   // accum_hdr
                sampled_texture_entry(1),  // terrain_height_tex
                sampled_texture_entry(2),  // terrain_minmax_tex
                uniform_entry(3),          // terrain uniforms
                storage_entry(4, false),   // terrain_welford
                storage_entry(5, false),   // terrain_reservoirs_curr
                sampled_texture_entry(6),  // terrain_env_tex
                storage_entry(7, false),   // terrain_reservoirs_prev
                uniform_entry(10),         // earth_curvature
                storage_entry(11, false),  // inv_v4 (adjoint + w history)
                storage_entry(12, false),  // inv_u32 (scalars/loss/d albedo)
                sampled_texture_entry(16), // terrain_albedo_tex
            ],
        });
        // Forward output layout (out_tex + 7 AOV storage textures) plus the
        // sampled observation target at 8.
        let inv_g3 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("inv-bgl3-out"),
            entries: &[
                storage_texture_entry(0, wgpu::TextureFormat::Rgba16Float), // out
                storage_texture_entry(1, wgpu::TextureFormat::Rgba16Float), // albedo
                storage_texture_entry(2, wgpu::TextureFormat::Rgba16Float), // normal
                storage_texture_entry(3, wgpu::TextureFormat::R32Float),    // depth
                storage_texture_entry(4, wgpu::TextureFormat::Rgba16Float), // direct
                storage_texture_entry(5, wgpu::TextureFormat::Rgba16Float), // indirect
                storage_texture_entry(6, wgpu::TextureFormat::Rgba16Float), // emission
                storage_texture_entry(7, wgpu::TextureFormat::Rgba8Unorm),  // visibility
                sampled_texture_entry(8),                                   // target
            ],
        });
        // Edge pass: the frame-0 Normal/Depth AOVs re-bound as SAMPLED
        // textures (the storage bindings stay exclusive to the primal).
        let inv_g3_edge = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("inv-bgl3-edge-aov"),
            entries: &[
                sampled_texture_entry(9),  // aov_normal (sampled)
                sampled_texture_entry(10), // aov_depth (sampled)
            ],
        });
        // Forward terrain-G-buffer group 2 (bindings 1,2,3 + 8,9,10) — the same
        // arrangement the forward driver binds for main_terrain_gbuffer.
        let inv_g2_gbuffer = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("inv-bgl2-gbuffer"),
            entries: &[
                sampled_texture_entry(1),
                sampled_texture_entry(2),
                uniform_entry(3),
                storage_entry(8, false),
                storage_entry(9, false),
                uniform_entry(10),
            ],
        });

        // One module carries the forward kernel + the inverse files (see
        // shader_sources::inverse_kernel); each entry point gets a pipeline.
        let shader = crate::core::shader_registry::create_labeled_shader_module(
            device,
            "inverse-pt",
            &crate::shader_sources::inverse_kernel(),
        );
        let main_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("inverse-pt-main-layout"),
            bind_group_layouts: &[&inv_g0, &inv_g1, &inv_g2, &inv_g3],
            push_constant_ranges: &[],
        });
        let gbuffer_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("inverse-pt-gbuffer-layout"),
            bind_group_layouts: &[&inv_g0, &inv_g1, &inv_g2_gbuffer],
            push_constant_ranges: &[],
        });
        let edge_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("inverse-pt-edge-layout"),
            bind_group_layouts: &[&inv_g0, &inv_g1, &inv_g2, &inv_g3_edge],
            push_constant_ranges: &[],
        });
        let mk = |label: &'static str, layout: &wgpu::PipelineLayout, entry: &'static str| {
            crate::core::shader_registry::try_create_compute_pipeline_scoped(
                device,
                &wgpu::ComputePipelineDescriptor {
                    label: Some(label),
                    layout: Some(layout),
                    module: &shader,
                    entry_point: entry,
                },
            )
            .map_err(|e| RenderError::Render(format!("{label}: {e}")))
        };
        let terrain = mk("inverse-pt-terrain", &main_layout, "main_terrain")?;
        let terrain_publish = mk(
            "inverse-pt-terrain-publish",
            &main_layout,
            "main_terrain_publish",
        )?;
        let terrain_gbuffer = mk(
            "inverse-pt-terrain-gbuffer",
            &gbuffer_layout,
            "main_terrain_gbuffer",
        )?;
        let clear = mk("inverse-pt-clear", &main_layout, "main_inverse_clear")?;
        let clear_grad = mk("inverse-pt-clear-grad", &main_layout, "main_inv_clear_grad")?;
        let accum_mean = mk("inverse-pt-accum-mean", &main_layout, "main_inv_accum_mean")?;
        let replay_clear = mk(
            "inverse-pt-replay-clear",
            &main_layout,
            "main_inv_replay_clear",
        )?;
        let wsnap = mk("inverse-pt-wsnap", &main_layout, "main_inv_wsnap")?;
        let loss = mk("inverse-pt-loss", &main_layout, "main_inverse_loss")?;
        let shade = mk("inverse-pt-shade", &main_layout, "main_inverse_shade")?;
        let edge = mk("inverse-pt-edge", &edge_layout, "main_inverse_edge")?;

        // Standalone ReSTIR passes — same modules and layout shapes the
        // forward driver uses (group 0 is the shared base-uniform buffer;
        // the standalone modules only read fields at identical offsets).
        let restir_g0 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("inv-restir-bgl0"),
            entries: &[uniform_entry(0)],
        });
        let restir_empty = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("inv-restir-bgl-empty"),
            entries: &[],
        });
        let restir_temporal_g2 =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("inv-restir-bgl2-temporal"),
                entries: &[
                    storage_entry(0, true),
                    storage_entry(1, true),
                    storage_entry(2, false),
                ],
            });
        let restir_spatial_g1 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("inv-restir-bgl1-spatial-scene"),
            entries: &[
                storage_entry(4, true),
                storage_entry(5, true),
                storage_entry(10, true),
                storage_entry(11, true),
            ],
        });
        let restir_spatial_g2 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("inv-restir-bgl2-spatial"),
            entries: &[storage_entry(0, true), storage_entry(1, false)],
        });

        let temporal_shader = crate::core::shader_registry::create_labeled_shader_module(
            device,
            "inv-restir-temporal",
            include_str!("../../shaders/pt_restir_temporal.wgsl"),
        );
        let temporal_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("inv-restir-temporal-layout"),
            bind_group_layouts: &[&restir_g0, &restir_empty, &restir_temporal_g2],
            push_constant_ranges: &[],
        });
        let restir_temporal = crate::core::shader_registry::try_create_compute_pipeline_scoped(
            device,
            &wgpu::ComputePipelineDescriptor {
                label: Some("inv-restir-temporal-compute"),
                layout: Some(&temporal_layout),
                module: &temporal_shader,
                entry_point: "main",
            },
        )
        .map_err(|e| RenderError::Render(format!("inv-restir-temporal: {e}")))?;

        let spatial_shader = crate::core::shader_registry::create_labeled_shader_module(
            device,
            "inv-restir-spatial",
            include_str!("../../shaders/pt_restir_spatial.wgsl"),
        );
        let spatial_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("inv-restir-spatial-layout"),
            bind_group_layouts: &[&restir_g0, &restir_spatial_g1, &restir_spatial_g2],
            push_constant_ranges: &[],
        });
        let restir_spatial = crate::core::shader_registry::try_create_compute_pipeline_scoped(
            device,
            &wgpu::ComputePipelineDescriptor {
                label: Some("inv-restir-spatial-compute"),
                layout: Some(&spatial_layout),
                module: &spatial_shader,
                entry_point: "main",
            },
        )
        .map_err(|e| RenderError::Render(format!("inv-restir-spatial: {e}")))?;

        Ok(Self {
            inv_g0,
            inv_g1,
            inv_g2,
            inv_g3,
            inv_g3_edge,
            inv_g2_gbuffer,
            restir_g0,
            restir_empty,
            restir_temporal_g2,
            restir_spatial_g1,
            restir_spatial_g2,
            terrain,
            terrain_publish,
            terrain_gbuffer,
            restir_temporal,
            restir_spatial,
            clear,
            clear_grad,
            accum_mean,
            replay_clear,
            wsnap,
            loss,
            shade,
            edge,
        })
    }
}
