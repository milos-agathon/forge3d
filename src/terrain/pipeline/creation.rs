use super::TerrainPipeline;
use std::borrow::Cow;
use wgpu::*;

/// Create the terrain pipeline. Does **not** record commands or create bind groups.
pub fn create_terrain_pipeline(
    device: &Device,
    color_format: TextureFormat,
    normal_format: TextureFormat,
    sample_count: u32,
    depth_format: Option<TextureFormat>,
    height_filterable: bool,
) -> TerrainPipeline {
    // Detect descriptor indexing capabilities from current device
    let features = device.features();
    let limits = device.limits();
    let descriptor_indexing = features.contains(Features::TEXTURE_BINDING_ARRAY)
        && features
            .contains(Features::SAMPLED_TEXTURE_AND_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING);
    let sample_count = sample_count.max(1);
    let max_palette_textures = if descriptor_indexing {
        limits.max_texture_array_layers.min(64)
    } else {
        1
    };
    // ---- Bind group layouts -------------------------------------------------
    // group(0) — Globals UBO (@group(0) @binding(0) var<uniform> globals : Globals)
    let bgl_globals = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
        label: Some("vf.Terrain.bgl.globals"),
        entries: &[BindGroupLayoutEntry {
            binding: 0,
            visibility: ShaderStages::VERTEX_FRAGMENT,
            ty: BindingType::Buffer {
                ty: BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None, // WGSL layout validated by naga
            },
            count: None,
        }],
    });

    // group(1) — height texture + sampler
    // Note: height_filterable is provided by caller — can be true either when
    // - device supports FLOAT32_FILTERABLE for R32F, or
    // - we're using a filterable fallback format (e.g., RG16Float)
    let bgl_height = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
        label: Some("vf.Terrain.bgl.height"),
        entries: &[
            BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::VERTEX_FRAGMENT,
                ty: BindingType::Texture {
                    sample_type: TextureSampleType::Float {
                        filterable: height_filterable,
                    },
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

    // E2/E1: group(3) — Per-tile uniforms (uv/world remap) + PageTable storage buffer
    let bgl_tile = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
        label: Some("vf.Terrain.bgl.tile"),
        entries: &[
            BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::VERTEX_FRAGMENT,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            BindGroupLayoutEntry {
                binding: 1,
                visibility: ShaderStages::VERTEX_FRAGMENT,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
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
            BindGroupLayoutEntry {
                binding: 3,
                visibility: ShaderStages::VERTEX_FRAGMENT,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    });

    // group(2) — LUT RGBA8UnormSrgb texture + sampler
    // Support texture arrays when descriptor indexing is available
    let bgl_lut = if descriptor_indexing {
        device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("vf.Terrain.bgl.lut.array"),
            entries: &[
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::FRAGMENT,
                    ty: BindingType::Texture {
                        sample_type: TextureSampleType::Float { filterable: true },
                        view_dimension: TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: Some(std::num::NonZeroU32::new(max_palette_textures).unwrap()),
                },
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::FRAGMENT,
                    ty: BindingType::Sampler(SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        })
    } else {
        device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("vf.Terrain.bgl.lut.single"),
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
                    ty: BindingType::Sampler(SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        })
    };

    // B7: group(3) — Cloud shadow texture + sampler
    let bgl_cloud_shadows = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
        label: Some("vf.Terrain.bgl.cloud_shadows"),
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
                ty: BindingType::Sampler(SamplerBindingType::Filtering),
                count: None,
            },
        ],
    });
    // B5: group(4) - Planar reflection uniforms + textures
    let bgl_reflection = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
        label: Some("vf.Terrain.bgl.reflection"),
        entries: &[
            BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::VERTEX_FRAGMENT,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            BindGroupLayoutEntry {
                binding: 1,
                visibility: ShaderStages::FRAGMENT,
                ty: BindingType::Texture {
                    sample_type: TextureSampleType::Float { filterable: true },
                    view_dimension: TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            BindGroupLayoutEntry {
                binding: 2,
                visibility: ShaderStages::FRAGMENT,
                ty: BindingType::Sampler(SamplerBindingType::Filtering),
                count: None,
            },
            BindGroupLayoutEntry {
                binding: 3,
                visibility: ShaderStages::FRAGMENT,
                ty: BindingType::Texture {
                    sample_type: TextureSampleType::Depth,
                    view_dimension: TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
        ],
    });

    // Respect device bind group limit (some devices only support 4)
    let max_groups = limits.max_bind_groups;
    let use_minimal_layout = max_groups < 6;

    let mut bgls: Vec<&BindGroupLayout> = vec![&bgl_globals, &bgl_height, &bgl_lut];
    // Always put tile at group(3)
    bgls.push(&bgl_tile);
    if !use_minimal_layout {
        bgls.push(&bgl_cloud_shadows); // group(4)
        bgls.push(&bgl_reflection); // group(5)
    }

    let layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
        label: Some("vf.Terrain.pipelineLayout"),
        bind_group_layouts: &bgls,
        push_constant_ranges: &[],
    });

    // ---- Shader module ------------------------------------------------------
    // Choose shader variant: minimal (<=4 bind groups) or full
    let shader = if use_minimal_layout {
        device.create_shader_module(ShaderModuleDescriptor {
            label: Some("vf.Terrain.shader.minimal"),
            source: ShaderSource::Wgsl(Cow::Borrowed(include_str!(
                "../../shaders/terrain_minimal.wgsl"
            ))),
        })
    } else {
        device.create_shader_module(ShaderModuleDescriptor {
            label: Some("vf.Terrain.shader.full"),
            source: ShaderSource::Wgsl(Cow::Borrowed(include_str!("../../shaders/terrain.wgsl"))),
        })
    };

    // ---- Vertex buffer layout ----------------------------------------------
    // Matches T3.1: location0 = position.xy (Float32x2), location1 = uv (Float32x2)
    const STRIDE: BufferAddress = 4 * 4; // 16 bytes
    let vertex_buffers = [VertexBufferLayout {
        array_stride: STRIDE,
        step_mode: VertexStepMode::Vertex,
        attributes: &vertex_attr_array![0 => Float32x2, 1 => Float32x2],
    }];

    // ---- Render pipeline ----------------------------------------------------
    let pipeline = device.create_render_pipeline(&RenderPipelineDescriptor {
        label: Some("vf.Terrain.pipeline"),
        layout: Some(&layout),
        vertex: VertexState {
            module: &shader,
            entry_point: "vs_main", // must match T3.1
            buffers: &vertex_buffers,
        },
        fragment: Some(FragmentState {
            module: &shader,
            entry_point: "fs_main", // must match T3.2
            targets: &[
                Some(ColorTargetState {
                    format: color_format, // Rgba8UnormSrgb recommended
                    blend: None, // straight alpha by default; no blending for opaque terrain
                    write_mask: ColorWrites::ALL,
                }),
                Some(ColorTargetState {
                    format: normal_format,
                    blend: None,
                    write_mask: ColorWrites::ALL,
                }),
            ],
        }),
        primitive: PrimitiveState {
            topology: PrimitiveTopology::TriangleList,
            strip_index_format: None,
            front_face: FrontFace::Ccw,
            cull_mode: Some(Face::Back),
            unclipped_depth: false,
            polygon_mode: PolygonMode::Fill,
            conservative: false,
        },
        depth_stencil: depth_format.map(|format| DepthStencilState {
            format,
            depth_write_enabled: true,
            depth_compare: CompareFunction::LessEqual,
            stencil: StencilState::default(),
            bias: DepthBiasState::default(),
        }),
        multisample: MultisampleState {
            count: sample_count,
            mask: !0,
            alpha_to_coverage_enabled: false,
        },
        multiview: None,
    });

    TerrainPipeline {
        layout,
        pipeline,
        bgl_globals,
        bgl_height,
        bgl_lut,
        bgl_cloud_shadows, // B7: Add cloud shadows bind group layout
        bgl_reflection,    // B5: Planar reflection bind group layout
        bgl_tile,
        descriptor_indexing,
        max_palette_textures,
        sample_count,
        depth_format,
        normal_format,
    }
}
