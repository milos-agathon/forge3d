// src/offscreen/pipeline.rs
// P7-03: PBR pipeline for offscreen BRDF tile rendering
// Simple pipeline with BRDF dispatch and NDF-only debug mode
// RELEVANT FILES: src/offscreen/brdf_tile.rs, src/shaders/brdf_tile.wgsl

use anyhow::Result;
use bytemuck::{Pod, Zeroable};

/// Uniforms for camera and transform matrices
#[repr(C)]
#[derive(Clone, Copy, Debug, Zeroable, Pod)]
pub struct Uniforms {
    pub model_matrix: [[f32; 4]; 4],
    pub view_matrix: [[f32; 4]; 4],
    pub projection_matrix: [[f32; 4]; 4],
}

/// Material and lighting parameters for BRDF tile rendering
#[repr(C)]
#[derive(Clone, Copy, Debug, Zeroable, Pod)]
pub struct BrdfTileParams {
    pub light_dir: [f32; 3],
    pub _pad0: f32,
    pub light_color: [f32; 3],
    pub light_intensity: f32,
    // M0: carry exposure via shading params (binding 2); kept here for future-proofing if needed
    pub camera_pos: [f32; 3],
    pub _pad1: f32,
    pub base_color: [f32; 3],
    pub metallic: f32,
    pub roughness: f32,
    pub ndf_only: u32,
    pub g_only: u32,    // Milestone 0: 1 = output Smith G as grayscale
    pub dfg_only: u32,  // Milestone 0: 1 = output D*F*G (pre-division)
    pub spec_only: u32, // Milestone 0: 1 = output specular-only (Cook–Torrance)
    pub roughness_visualize: u32,  // Milestone 0: 1 = output vec3(r) for uniform validation
    pub f0: [f32; 3],             // Milestone 0: explicit F0 vector
    pub _pad_f0: f32,
    // M4: Disney Principled BRDF extensions
    pub clearcoat: f32,
    pub clearcoat_roughness: f32,
    pub sheen: f32,
    pub sheen_tint: f32,
    pub specular_tint: f32,
    // M2: Debug toggles
    pub debug_lambert_only: u32,   // 1 = output diffuse lambert only (disable specular)
    pub debug_energy: u32,         // 1 = output (kS + kD)
    pub debug_d: u32,              // 1 = output D only (grayscale)
    pub debug_g_dbg: u32,          // 1 = output G only (grayscale, correlated)
    pub debug_spec_no_nl: u32,     // 1 = output spec only without NL/Li
    pub debug_angle_sweep: u32,    // 1 = override normal with sweep across uv.x and force V=L=+Z
    pub debug_angle_component: u32,// 0=spec,1=diffuse,2=combined
    pub debug_no_srgb: u32,        // 1 = bypass sRGB conversion at end
    // Preserve total size with padding (brdf_tile.rs overprovisions to 256 bytes)
    pub _pad2: u32,
    pub _pad3: u32,
    pub _pad4: u32,
    pub _pad5: u32,
    pub _pad6: u32,
    pub _pad7: u32,
}

/// Shading parameters matching ShadingParamsGPU in lighting.wgsl
#[repr(C)]
#[derive(Clone, Copy, Debug, Zeroable, Pod)]
pub struct ShadingParamsGPU {
    pub brdf: u32,
    pub metallic: f32,
    pub roughness: f32,
    pub sheen: f32,
    pub clearcoat: f32,
    pub subsurface: f32,
    pub anisotropy: f32,
    pub exposure: f32, // Milestone 0: carry exposure for deterministic output
    // M2: Output encoding selection (0=linear, 1=srgb)
    pub output_mode: u32,
    pub _pad_out0: u32,
    pub _pad_out1: u32,
    pub _pad_out2: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Zeroable, Pod)]
pub struct DebugPush {
    pub mode: u32,
    pub roughness: f32,
    pub _pad: [f32; 2],
}

/// BRDF tile rendering pipeline
pub struct BrdfTilePipeline {
    pipeline: wgpu::RenderPipeline,
    bind_group_layout: wgpu::BindGroupLayout,
}

impl BrdfTilePipeline {
    pub fn new(device: &wgpu::Device) -> Result<Self> {
        Self::new_with_format(device, wgpu::TextureFormat::Rgba8Unorm)
    }

    /// Create the pipeline with a specific color target format.
    /// Common formats:
    /// - Rgba8Unorm: linear 8-bit per channel (shader can apply sRGB)
    /// - Rgba16Float: linear 16-bit float per channel (HDR)
    pub fn new_with_format(device: &wgpu::Device, color_format: wgpu::TextureFormat) -> Result<Self> {
        // Load standalone shader (no includes needed - all BRDF code is inline)
        let shader_src = include_str!("../shaders/brdf_tile.wgsl");

        // Milestone 0: Log shader version stamp at pipeline creation for CI diff tracking
        // Parse BRDF_SHADER_VERSION constant from WGSL source to avoid drift
        let shader_version = shader_src
            .lines()
            .find(|l| l.contains("BRDF_SHADER_VERSION"))
            .and_then(|l| l.split('=').nth(1))
            .map(|rhs| rhs.chars().filter(|c| c.is_ascii_digit()).collect::<String>())
            .and_then(|digits| if digits.is_empty() { None } else { Some(digits) })
            .unwrap_or_else(|| "unknown".to_string());
        log::info!("BRDF_SHADER_VERSION = {}", shader_version);
        
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("brdf_tile.shader"),
            source: wgpu::ShaderSource::Wgsl(shader_src.into()),
        });
        
        // Create bind group layout
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("brdf_tile.bind_group_layout"),
            entries: &[
                // @binding(0): Uniforms
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // @binding(1): BrdfTileParams
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // @binding(2): ShadingParamsGPU
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // @binding(3): M1 Debug buffer for min/max N·L, N·V
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // @binding(7): WI-3 debug selector
                wgpu::BindGroupLayoutEntry {
                    binding: 7,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("brdf_tile.pipeline_layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });
        
        // Create render pipeline
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("brdf_tile.pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                buffers: &[
                    // TbnVertex layout from sphere.rs
                    wgpu::VertexBufferLayout {
                        array_stride: std::mem::size_of::<crate::offscreen::sphere::TbnVertex>() as wgpu::BufferAddress,
                        step_mode: wgpu::VertexStepMode::Vertex,
                        attributes: &[
                            // position
                            wgpu::VertexAttribute {
                                offset: 0,
                                shader_location: 0,
                                format: wgpu::VertexFormat::Float32x3,
                            },
                            // uv
                            wgpu::VertexAttribute {
                                offset: 12,
                                shader_location: 1,
                                format: wgpu::VertexFormat::Float32x2,
                            },
                            // normal
                            wgpu::VertexAttribute {
                                offset: 20,
                                shader_location: 2,
                                format: wgpu::VertexFormat::Float32x3,
                            },
                            // tangent
                            wgpu::VertexAttribute {
                                offset: 32,
                                shader_location: 3,
                                format: wgpu::VertexFormat::Float32x3,
                            },
                            // bitangent
                            wgpu::VertexAttribute {
                                offset: 44,
                                shader_location: 4,
                                format: wgpu::VertexFormat::Float32x3,
                            },
                        ],
                    },
                ],
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
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: Some(wgpu::Face::Back),
                unclipped_depth: false,
                polygon_mode: wgpu::PolygonMode::Fill,
                conservative: false,
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth24Plus,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
        });
        
        Ok(Self {
            pipeline,
            bind_group_layout,
        })
    }
    
    /// Create bind group for rendering
    pub fn create_bind_group(
        &self,
        device: &wgpu::Device,
        uniforms_buffer: &wgpu::Buffer,
        params_buffer: &wgpu::Buffer,
        shading_buffer: &wgpu::Buffer,
        debug_buffer: &wgpu::Buffer,
        debug_push_buffer: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("brdf_tile.bind_group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniforms_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: shading_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: debug_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: debug_push_buffer.as_entire_binding(),
                },
            ],
        })
    }
    
    pub fn pipeline(&self) -> &wgpu::RenderPipeline {
        &self.pipeline
    }
}
