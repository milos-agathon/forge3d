// src/path_tracing/hybrid_compute.rs
// Hybrid path tracer combining SDF raymarching with mesh BVH traversal
// Extends the existing path tracing compute pipeline with hybrid scene support

use std::num::NonZeroU32;

use bytemuck::{Pod, Zeroable};
use half::f16;
use wgpu::util::DeviceExt;

use crate::error::RenderError;
use crate::gpu::{align_copy_bpr, ctx};
use crate::path_tracing::aov::{AovFrames, AovKind};
use crate::path_tracing::compute::{Sphere, Uniforms};
use crate::sdf::HybridScene;
use std::borrow::Cow;

/// Load the hybrid kernel WGSL and expand simple `#include "..."` directives.
///
/// WGSL has no preprocessor, so we inline our shader snippets. This function
/// repeatedly expands known includes and removes duplicate include directives to
/// avoid parser errors and symbol redefinitions.
fn load_hybrid_kernel_src() -> String {
    // Deterministic assembly to avoid duplicate definitions
    // 1) Core SDF primitives (types and basic evaluators)
    let sdf_primitives = include_str!("../shaders/sdf_primitives.wgsl");

    // 2) CSG operations (strip its own includes)
    let sdf_operations_raw = include_str!("../shaders/sdf_operations.wgsl");
    let sdf_operations = sdf_operations_raw
        .lines()
        .filter(|l| !l.trim_start().starts_with("#include"))
        .collect::<Vec<_>>()
        .join("\n");

    // 3) Hybrid traversal utilities (strip includes)
    let hybrid_traversal_raw = include_str!("../shaders/hybrid_traversal.wgsl");
    let hybrid_traversal = hybrid_traversal_raw
        .lines()
        .filter(|l| !l.trim_start().starts_with("#include"))
        .collect::<Vec<_>>()
        .join("\n");

    // 4) Kernel entry (strip includes)
    let kernel_raw = include_str!("../shaders/hybrid_kernel.wgsl");
    let kernel = kernel_raw
        .lines()
        .filter(|l| !l.trim_start().starts_with("#include"))
        .collect::<Vec<_>>()
        .join("\n");

    // Concatenate in dependency order
    [sdf_primitives, &sdf_operations, &hybrid_traversal, &kernel].join("\n")
}

/// Additional uniforms for hybrid traversal
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug)]
pub struct HybridUniforms {
    pub sdf_primitive_count: u32,
    pub sdf_node_count: u32,
    pub mesh_vertex_count: u32,
    pub mesh_index_count: u32,
    pub mesh_bvh_node_count: u32,
    pub traversal_mode: u32, // 0 = hybrid, 1 = SDF only, 2 = mesh only
    pub _pad: [u32; 2],
}

/// Traversal mode for hybrid rendering
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum TraversalMode {
    Hybrid = 0,
    SdfOnly = 1,
    MeshOnly = 2,
}

impl Default for TraversalMode {
    fn default() -> Self {
        Self::Hybrid
    }
}

/// Hybrid path tracer parameters
#[derive(Clone, Debug)]
pub struct HybridTracerParams {
    /// Base path tracer uniforms
    pub base_uniforms: Uniforms,
    /// Traversal mode
    pub traversal_mode: TraversalMode,
    /// Early exit distance for optimization
    pub early_exit_distance: f32,
    /// Soft shadow softness factor
    pub shadow_softness: f32,
}

impl Default for HybridTracerParams {
    fn default() -> Self {
        Self {
            base_uniforms: Uniforms {
                width: 512,
                height: 512,
                frame_index: 0,
                aov_flags: 0,
                cam_origin: [0.0, 0.0, 0.0],
                cam_fov_y: std::f32::consts::PI / 4.0,
                cam_right: [1.0, 0.0, 0.0],
                cam_aspect: 1.0,
                cam_up: [0.0, 1.0, 0.0],
                cam_exposure: 1.0,
                cam_forward: [0.0, 0.0, -1.0],
                seed_hi: 12345,
                seed_lo: 67890,
                _pad_end: [0, 0, 0],
            },
            traversal_mode: TraversalMode::Hybrid,
            early_exit_distance: 0.01,
            shadow_softness: 4.0,
        }
    }
}

/// Hybrid path tracer implementation
pub struct HybridPathTracer {
    /// Shader module for hybrid traversal
    #[allow(dead_code)]
    shader: wgpu::ShaderModule,
    /// Bind group layouts
    layouts: HybridBindGroupLayouts,
    /// Pipeline layout
    #[allow(dead_code)]
    pipeline_layout: wgpu::PipelineLayout,
    /// Compute pipeline
    pipeline: wgpu::ComputePipeline,
}

/// Bind group layouts for hybrid path tracer
struct HybridBindGroupLayouts {
    uniforms: wgpu::BindGroupLayout, // Group 0: camera uniforms
    scene: wgpu::BindGroupLayout,    // Group 1: spheres + legacy mesh buffers
    accum: wgpu::BindGroupLayout,    // Group 2: accumulation buffer
    output: wgpu::BindGroupLayout,   // Group 3: primary output texture + AOVs
}

impl HybridPathTracer {
    /// Create a new hybrid path tracer
    pub fn new() -> Result<Self, RenderError> {
        let device = &ctx().device;

        // Load hybrid kernel shader (expand simple includes)
        let shader_src = load_hybrid_kernel_src();
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("hybrid-pt-kernel"),
            source: wgpu::ShaderSource::Wgsl(shader_src.into()),
        });

        // Create bind group layouts
        let layouts = HybridBindGroupLayouts {
            uniforms: Self::create_uniforms_layout(device),
            scene: Self::create_scene_layout(device),
            accum: Self::create_accum_layout(device),
            output: Self::create_output_layout(device),
        };

        // Create pipeline layout
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("hybrid-pt-pipeline-layout"),
            bind_group_layouts: &[
                &layouts.uniforms,
                &layouts.scene,
                &layouts.accum,
                &layouts.output,
            ],
            push_constant_ranges: &[],
        });

        // Create compute pipeline
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("hybrid-pt-compute"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: "main",
        });

        Ok(Self {
            shader,
            layouts,
            pipeline_layout,
            pipeline,
        })
    }

    /// Render using hybrid traversal
    pub fn render(
        &self,
        width: u32,
        height: u32,
        spheres: &[Sphere],
        hybrid_scene: &HybridScene,
        params: HybridTracerParams,
    ) -> Result<Vec<u8>, RenderError> {
        let device = &ctx().device;
        let queue = &ctx().queue;

        // Prepare hybrid uniforms
        let hybrid_uniforms = HybridUniforms {
            sdf_primitive_count: hybrid_scene.sdf_scene.primitive_count() as u32,
            sdf_node_count: hybrid_scene.sdf_scene.node_count() as u32,
            mesh_vertex_count: hybrid_scene.vertices.len() as u32,
            mesh_index_count: hybrid_scene.indices.len() as u32,
            mesh_bvh_node_count: match &hybrid_scene.bvh {
                Some(bvh) => match &bvh.backend {
                    crate::accel::BvhBackend::Cpu(cpu_data) => cpu_data.nodes.len() as u32,
                    crate::accel::BvhBackend::Gpu(gpu_data) => gpu_data.node_count,
                },
                None => 0,
            },
            traversal_mode: params.traversal_mode as u32,
            _pad: [0; 2],
        };

        // Create buffers for base uniforms
        let base_ubo = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("hybrid-pt-base-ubo"),
            contents: bytemuck::bytes_of(&params.base_uniforms),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let hybrid_ubo = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("hybrid-pt-hybrid-ubo"),
            contents: bytemuck::bytes_of(&hybrid_uniforms),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // Scene buffer (spheres)
        let scene_bytes: Cow<[u8]> = if spheres.is_empty() {
            Cow::Owned(vec![0u8; std::mem::size_of::<Sphere>()])
        } else {
            Cow::Borrowed(bytemuck::cast_slice(spheres))
        };
        let scene_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("hybrid-pt-scene"),
            contents: scene_bytes.as_ref(),
            usage: wgpu::BufferUsages::STORAGE,
        });

        // Accumulation buffer
        let accum_size = (width as u64) * (height as u64) * 16; // vec4<f32>
        let accum_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("hybrid-pt-accum"),
            size: accum_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Output texture
        let out_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("hybrid-pt-out-tex"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let out_view = out_tex.create_view(&wgpu::TextureViewDescriptor::default());

        // AOV textures
        let aovs_all = [
            AovKind::Albedo,
            AovKind::Normal,
            AovKind::Depth,
            AovKind::Direct,
            AovKind::Indirect,
            AovKind::Emission,
            AovKind::Visibility,
        ];
        let aov_frames = AovFrames::new(device, width, height, &aovs_all);
        let aov_views: Vec<wgpu::TextureView> = aovs_all
            .iter()
            .map(|k| {
                aov_frames
                    .get_texture(*k)
                    .unwrap()
                    .create_view(&wgpu::TextureViewDescriptor::default())
            })
            .collect();

        // Create bind groups
        let bg0 = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("hybrid-pt-bg0"),
            layout: &self.layouts.uniforms,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: base_ubo.as_entire_binding(),
            }],
        });

        // Scene + Hybrid bind group (Group 1)
        let mut bg1_entries = vec![
            wgpu::BindGroupEntry {
                binding: 0,
                resource: scene_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: hybrid_ubo.as_entire_binding(),
            },
        ];

        // Add mesh buffer entries at bindings 2..4
        bg1_entries.extend(
            hybrid_scene
                .get_mesh_bind_entries()
                .into_iter()
                .enumerate()
                .map(|(i, mut entry)| {
                    entry.binding = (i + 2) as u32;
                    entry
                }),
        );

        let bg1 = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("hybrid-pt-bg1"),
            layout: &self.layouts.scene,
            entries: &bg1_entries,
        });

        let bg2 = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("hybrid-pt-bg2"),
            layout: &self.layouts.accum,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: accum_buf.as_entire_binding(),
            }],
        });

        // Output + AOVs bind group (Group 3)
        let mut bg3_entries = vec![wgpu::BindGroupEntry {
            binding: 0,
            resource: wgpu::BindingResource::TextureView(&out_view),
        }];
        // Map AOVs to bindings 1..7 in the same group
        for (i, view) in aov_views.iter().enumerate() {
            bg3_entries.push(wgpu::BindGroupEntry {
                binding: (i as u32) + 1,
                resource: wgpu::BindingResource::TextureView(view),
            });
        }
        let bg3 = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("hybrid-pt-bg3"),
            layout: &self.layouts.output,
            entries: &bg3_entries,
        });

        // Removed separate AOV (bg4) and Hybrid (bg5) groups; merged into bg3 and bg1 respectively.

        // Dispatch compute shader
        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("hybrid-pt-encoder"),
        });

        {
            let mut cpass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("hybrid-pt-cpass"),
                ..Default::default()
            });
            cpass.set_pipeline(&self.pipeline);
            cpass.set_bind_group(0, &bg0, &[]);
            cpass.set_bind_group(1, &bg1, &[]);
            cpass.set_bind_group(2, &bg2, &[]);
            cpass.set_bind_group(3, &bg3, &[]);

            let gx = (width + 7) / 8;
            let gy = (height + 7) / 8;
            cpass.dispatch_workgroups(gx, gy, 1);
        }

        // Copy texture to buffer for readback
        let row_bytes = width * 8; // rgba16f = 8 bytes per pixel
        let padded_bpr = align_copy_bpr(row_bytes);
        let read_size = (padded_bpr as u64) * (height as u64);
        let read_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("hybrid-pt-read"),
            size: read_size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        enc.copy_texture_to_buffer(
            wgpu::ImageCopyTexture {
                texture: &out_tex,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::ImageCopyBuffer {
                buffer: &read_buf,
                layout: wgpu::ImageDataLayout {
                    offset: 0,
                    bytes_per_row: Some(NonZeroU32::new(padded_bpr).unwrap().into()),
                    rows_per_image: Some(NonZeroU32::new(height).unwrap().into()),
                },
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );

        queue.submit([enc.finish()]);
        device.poll(wgpu::Maintain::Wait);

        // Read back and convert to RGBA8
        let slice = read_buf.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |res| {
            let _ = tx.send(res);
        });
        device.poll(wgpu::Maintain::Wait);
        rx.recv()
            .map_err(|_| RenderError::Readback("map_async channel closed".into()))?
            .map_err(|e| RenderError::Readback(format!("MapAsync failed: {:?}", e)))?;

        let data = slice.get_mapped_range();
        let mut out = vec![0u8; (width as usize) * (height as usize) * 4];
        let src_stride = padded_bpr as usize;
        let dst_stride = (width as usize) * 4;

        for y in 0..(height as usize) {
            let row = &data[y * src_stride..y * src_stride + (width as usize) * 8];
            for x in 0..(width as usize) {
                let o = x * 8;
                let r = f16::from_bits(u16::from_le_bytes([row[o + 0], row[o + 1]])).to_f32();
                let g = f16::from_bits(u16::from_le_bytes([row[o + 2], row[o + 3]])).to_f32();
                let b = f16::from_bits(u16::from_le_bytes([row[o + 4], row[o + 5]])).to_f32();

                let ix = y * dst_stride + x * 4;
                out[ix + 0] = (r.clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
                out[ix + 1] = (g.clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
                out[ix + 2] = (b.clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
                out[ix + 3] = 255u8;
            }
        }

        drop(data);
        read_buf.unmap();
        Ok(out)
    }

    // Bind group layout creation methods
    fn create_uniforms_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("hybrid-pt-bgl0-uniforms"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        })
    }

    fn create_scene_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("hybrid-pt-bgl1-scene"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        })
    }

    fn create_accum_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("hybrid-pt-bgl2-accum"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        })
    }

    fn create_output_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("hybrid-pt-bgl3-out"),
            entries: &[
                // out_tex at binding 0
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba16Float,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
                // AOVs binding 1..7
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba16Float,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba16Float,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::R32Float,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba16Float,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba16Float,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba16Float,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 7,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba8Unorm,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
            ],
        })
    }

    fn create_aov_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("hybrid-pt-bgl4-aovs"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba16Float,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba16Float,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::R32Float,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba16Float,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba16Float,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba16Float,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba16Float,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 7,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba8Unorm,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
            ],
        })
    }

    fn create_hybrid_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("hybrid-pt-bgl5-hybrid"),
            entries: &[
                // Hybrid uniforms
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
                // SDF primitives
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // SDF nodes
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Mesh vertices
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Mesh indices
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // BVH nodes
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        })
    }
}
