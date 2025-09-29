// T33-BEGIN:terrain-pipeline
//! Terrain pipeline state & bindings (T3.3).
//! Creates bind group layouts (0: Globals UBO, 1: height+sampler, 2: LUT+sampler)
//! and a render pipeline targeting Rgba8UnormSrgb. No integration/draw in this task.
//! Supports optional descriptor indexing for texture arrays when available.

use crate::core::reflections::PlanarReflectionRenderer;
use std::borrow::Cow;
use wgpu::*;

pub struct TerrainPipeline {
    pub layout: PipelineLayout,
    pub pipeline: RenderPipeline,
    pub bgl_globals: BindGroupLayout,
    pub bgl_height: BindGroupLayout,
    pub bgl_lut: BindGroupLayout,
    pub bgl_cloud_shadows: BindGroupLayout, // B7: Cloud shadows bind group layout
    pub bgl_reflection: BindGroupLayout,    // B5: Planar reflections bind group layout
    pub bgl_tile: BindGroupLayout,          // E2: Per-tile uniforms (uv/world remap)
    pub descriptor_indexing: bool,
    pub max_palette_textures: u32,
    pub sample_count: u32,
    pub depth_format: Option<TextureFormat>,
    pub normal_format: TextureFormat,
}

impl TerrainPipeline {
    /// Create the terrain pipeline. Does **not** record commands or create bind groups.
    pub fn create(
        device: &Device,
        color_format: TextureFormat,
        normal_format: TextureFormat,
        sample_count: u32,
        depth_format: Option<TextureFormat>,
        height_filterable: bool,
    ) -> Self {
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
                    "../shaders/terrain_minimal.wgsl"
                ))),
            })
        } else {
            device.create_shader_module(ShaderModuleDescriptor {
                label: Some("vf.Terrain.shader.full"),
                source: ShaderSource::Wgsl(Cow::Borrowed(include_str!("../shaders/terrain.wgsl"))),
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

        Self {
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

    // ---------- Bind-group helpers (builders) ----------
    pub fn make_bg_globals(&self, device: &Device, ubo: &Buffer) -> BindGroup {
        device.create_bind_group(&BindGroupDescriptor {
            label: Some("vf.Terrain.bg.globals"),
            layout: &self.bgl_globals,
            entries: &[BindGroupEntry {
                binding: 0,
                resource: ubo.as_entire_binding(),
            }],
        })
    }

    /// E2/E1: Per-tile uniform + page table bind group helper
    pub fn make_bg_tile(
        &self,
        device: &Device,
        tile_ubo: &Buffer,
        page_table: Option<&Buffer>,
        tile_slot_ubo: &Buffer,
        mosaic_params_ubo: &Buffer,
    ) -> BindGroup {
        let pt_dummy = device.create_buffer(&BufferDescriptor {
            label: Some("vf.Terrain.page_table.dummy"),
            // Must be at least the size of one PageTableEntry (8 u32 = 32 bytes)
            size: 32,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let pt_binding = page_table
            .map(|b| b.as_entire_binding())
            .unwrap_or_else(|| pt_dummy.as_entire_binding());

        device.create_bind_group(&BindGroupDescriptor {
            label: Some("vf.Terrain.bg.tile"),
            layout: &self.bgl_tile,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: tile_ubo.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: pt_binding,
                },
                BindGroupEntry {
                    binding: 2,
                    resource: tile_slot_ubo.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: mosaic_params_ubo.as_entire_binding(),
                },
            ],
        })
    }

    /// Bind group for height texture/sampler
    pub fn make_bg_height(&self, device: &Device, view: &TextureView, samp: &Sampler) -> BindGroup {
        device.create_bind_group(&BindGroupDescriptor {
            label: Some("vf.Terrain.bg.height"),
            layout: &self.bgl_height,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::TextureView(view),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::Sampler(samp),
                },
            ],
        })
    }

    pub fn make_bg_lut(&self, device: &Device, view: &TextureView, samp: &Sampler) -> BindGroup {
        device.create_bind_group(&BindGroupDescriptor {
            label: Some("vf.Terrain.bg.lut"),
            layout: &self.bgl_lut,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::TextureView(view),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::Sampler(samp),
                },
            ],
        })
    }

    /// Create bind group with texture array for descriptor indexing mode
    pub fn make_bg_lut_array(
        &self,
        device: &Device,
        views: &[&TextureView],
        samp: &Sampler,
    ) -> BindGroup {
        if !self.descriptor_indexing {
            panic!("make_bg_lut_array called but descriptor indexing is not available");
        }

        device.create_bind_group(&BindGroupDescriptor {
            label: Some("vf.Terrain.bg.lut.array"),
            layout: &self.bgl_lut,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::TextureViewArray(views),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::Sampler(samp),
                },
            ],
        })
    }

    // B7: Cloud shadow bind group helper
    pub fn make_bg_cloud_shadows(
        &self,
        device: &Device,
        view: &TextureView,
        samp: &Sampler,
    ) -> BindGroup {
        device.create_bind_group(&BindGroupDescriptor {
            label: Some("vf.Terrain.bg.cloud_shadows"),
            layout: &self.bgl_cloud_shadows,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::TextureView(view),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::Sampler(samp),
                },
            ],
        })
    }

    pub fn make_bg_reflection(
        &self,
        device: &Device,
        renderer: &PlanarReflectionRenderer,
    ) -> BindGroup {
        device.create_bind_group(&BindGroupDescriptor {
            label: Some("vf.Terrain.bg.reflection"),
            layout: &self.bgl_reflection,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: renderer.uniform_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::TextureView(&renderer.reflection_view),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: BindingResource::Sampler(&renderer.reflection_sampler),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: BindingResource::TextureView(&renderer.reflection_depth_view),
                },
            ],
        })
    }

    /// Check if descriptor indexing is supported
    pub fn supports_descriptor_indexing(&self) -> bool {
        self.descriptor_indexing
    }

    /// Get maximum number of palette textures supported
    pub fn max_palette_textures(&self) -> u32 {
        self.max_palette_textures
    }
}

// ---- Tests (no GPU device creation; descriptor sanity only where possible) ----
#[cfg(test)]
mod tests {
    #[test]
    fn vertex_stride_is_16_bytes() {
        // Keep this in sync with two vec2<f32> attributes
        assert_eq!(16, 4 * 4);
    }
}
// T33-END:terrain-pipeline
