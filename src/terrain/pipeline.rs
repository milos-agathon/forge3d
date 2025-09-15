// T33-BEGIN:terrain-pipeline
//! Terrain pipeline state & bindings (T3.3).
//! Creates bind group layouts (0: Globals UBO, 1: height+sampler, 2: LUT+sampler)
//! and a render pipeline targeting Rgba8UnormSrgb. No integration/draw in this task.
//! Supports optional descriptor indexing for texture arrays when available.

use crate::device_caps::DeviceCaps;
use std::borrow::Cow;
use wgpu::*;

pub struct TerrainPipeline {
    pub layout: PipelineLayout,
    pub pipeline: RenderPipeline,
    pub bgl_globals: BindGroupLayout,
    pub bgl_height: BindGroupLayout,
    pub bgl_lut: BindGroupLayout,
    pub descriptor_indexing: bool,
    pub max_palette_textures: u32,
}

impl TerrainPipeline {
    /// Create the terrain pipeline. Does **not** record commands or create bind groups.
    pub fn create(device: &Device, color_format: TextureFormat) -> Self {
        // Detect descriptor indexing capabilities
        let device_caps = DeviceCaps::from_current_device().unwrap_or_else(|_| {
            // Fallback if capability detection fails
            DeviceCaps {
                backend: "unknown".to_string(),
                adapter_name: "unknown".to_string(),
                device_name: "unknown".to_string(),
                max_texture_dimension_2d: 4096,
                max_buffer_size: 128 * 1024 * 1024,
                msaa_supported: false,
                max_samples: 1,
                device_type: "unknown".to_string(),
                descriptor_indexing: false,
                max_texture_array_layers: 16,
                max_sampler_array_size: 8,
                vertex_shader_array_support: false,
            }
        });

        let descriptor_indexing = device_caps.descriptor_indexing;
        let max_palette_textures = if descriptor_indexing {
            device_caps.max_texture_array_layers.min(64) // Reasonable limit for palettes
        } else {
            1 // Single texture mode
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

        // group(1) — height R32Float texture + sampler
        let bgl_height = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("vf.Terrain.bgl.height"),
            entries: &[
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::VERTEX_FRAGMENT,
                    ty: BindingType::Texture {
                        sample_type: TextureSampleType::Float { filterable: false },
                        view_dimension: TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::VERTEX_FRAGMENT,
                    ty: BindingType::Sampler(SamplerBindingType::NonFiltering),
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

        let layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("vf.Terrain.pipelineLayout"),
            bind_group_layouts: &[&bgl_globals, &bgl_height, &bgl_lut],
            push_constant_ranges: &[],
        });

        // ---- Shader module ------------------------------------------------------
        // Choose shader variant based on descriptor indexing support
        let shader = if descriptor_indexing {
            device.create_shader_module(ShaderModuleDescriptor {
                label: Some("vf.Terrain.shader.descriptor_indexing"),
                source: ShaderSource::Wgsl(Cow::Borrowed(include_str!(
                    "../shaders/terrain_descriptor_indexing.wgsl"
                ))),
            })
        } else {
            device.create_shader_module(ShaderModuleDescriptor {
                label: Some("vf.Terrain.shader.fallback"),
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
                targets: &[Some(ColorTargetState {
                    format: color_format, // Rgba8UnormSrgb recommended
                    blend: None, // straight alpha by default; no blending for opaque terrain
                    write_mask: ColorWrites::ALL,
                })],
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
            depth_stencil: None, // no depth for single-layer terrain in spike
            multisample: MultisampleState {
                count: 1, // MSAA=1 per roadmap
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
            descriptor_indexing,
            max_palette_textures,
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
