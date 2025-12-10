// src/path_tracing/compute.rs
// Minimal GPU compute path tracer implementation (A1) using WGSL kernel and wgpu.
// This exists to create the compute pipeline, allocate buffers/textures, dispatch, and read back RGBA and AOVs.
// RELEVANT FILES:src/path_tracing/mod.rs,src/shaders/pt_kernel.wgsl,python/forge3d/path_tracing.py,src/lib.rs

use std::num::NonZeroU32;

use bytemuck::{Pod, Zeroable};
use half::f16;
use wgpu::util::DeviceExt;

use crate::core::error::RenderError;
use crate::core::gpu::{align_copy_bpr, ctx};
use crate::path_tracing::aov::{AovFrames, AovKind};
use crate::path_tracing::mesh::create_empty_mesh_buffers;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug)]
pub struct Uniforms {
    pub width: u32,
    pub height: u32,
    pub frame_index: u32,
    pub aov_flags: u32,
    pub cam_origin: [f32; 3],
    pub cam_fov_y: f32,
    pub cam_right: [f32; 3],
    pub cam_aspect: f32,
    pub cam_up: [f32; 3],
    pub cam_exposure: f32,
    pub cam_forward: [f32; 3],
    pub seed_hi: u32,
    pub seed_lo: u32,
    pub _pad_end: [u32; 3],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug)]
pub struct Sphere {
    // 16-byte group 0
    pub center: [f32; 3],
    pub radius: f32,
    // 16-byte group 1
    pub albedo: [f32; 3],
    pub metallic: f32,
    // 16-byte group 2
    pub emissive: [f32; 3],
    pub roughness: f32,
    // 16-byte group 3
    pub ior: f32,
    pub ax: f32,
    pub ay: f32,
    pub _pad1: f32,
}

pub struct PathTracerGPU;

impl PathTracerGPU {
    pub fn render(
        width: u32,
        height: u32,
        spheres: &[Sphere],
        uniforms: Uniforms,
    ) -> Result<Vec<u8>, RenderError> {
        let g = ctx();
        // Shader
        let shader = g.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("pt_kernel"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/pt_kernel.wgsl").into()),
        });

        // Bind group layouts
        let bgl0 = g
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("pt-bgl0-uniforms"),
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
            });
        let bgl1 = g
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("pt-bgl1-scene"),
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
                    // Optional mesh bindings (vertices, indices, bvh)
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
                ],
            });
        let bgl2 = g
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("pt-bgl2-accum"),
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
            });
        let bgl3 = g
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("pt-bgl3-out"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba16Float,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                }],
            });

        // Group 4: AOVs (bind 0..6)
        let bgl4 = g
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("pt-bgl4-aovs"),
                entries: &[
                    // 0: albedo
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
                    // 1: normal
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
                    // 2: depth
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
                    // 3: direct
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
                    // 4: indirect
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
                    // 5: emission
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
                    // 6: visibility
                    wgpu::BindGroupLayoutEntry {
                        binding: 6,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::StorageTexture {
                            access: wgpu::StorageTextureAccess::WriteOnly,
                            format: wgpu::TextureFormat::Rgba8Unorm,
                            view_dimension: wgpu::TextureViewDimension::D2,
                        },
                        count: None,
                    },
                ],
            });

        let pipeline_layout = g
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("pt-pipeline-layout"),
                bind_group_layouts: &[&bgl0, &bgl1, &bgl2, &bgl3, &bgl4],
                push_constant_ranges: &[],
            });

        let pipeline = g
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("pt-compute"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: "main",
            });

        // Buffers
        let ubo = g
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("pt-ubo"),
                contents: bytemuck::bytes_of(&uniforms),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

        let scene_bytes = bytemuck::cast_slice(spheres);
        let scene_buf = g
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("pt-scene"),
                contents: scene_bytes,
                usage: wgpu::BufferUsages::STORAGE,
            });
        // Empty mesh buffers to satisfy shader bindings (A3 prep)
        let (mesh_vertices, mesh_indices, mesh_bvh) = create_empty_mesh_buffers(&g.device);

        let accum_size = (width as u64) * (height as u64) * 16; // vec4<f32>
        let accum_buf = g.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("pt-accum"),
            size: accum_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Output texture (RGBA16F)
        let out_tex = g.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("pt-out-tex"),
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

        // Bind groups
        let bg0 = g.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("pt-bg0"),
            layout: &bgl0,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: ubo.as_entire_binding(),
            }],
        });
        let bg1 = g.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("pt-bg1"),
            layout: &bgl1,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: scene_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: mesh_vertices.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: mesh_indices.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: mesh_bvh.as_entire_binding(),
                },
            ],
        });
        let bg2 = g.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("pt-bg2"),
            layout: &bgl2,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: accum_buf.as_entire_binding(),
            }],
        });
        let bg3 = g.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("pt-bg3"),
            layout: &bgl3,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&out_view),
            }],
        });

        // Create AOV textures and bind group
        let aovs_all = [
            AovKind::Albedo,
            AovKind::Normal,
            AovKind::Depth,
            AovKind::Direct,
            AovKind::Indirect,
            AovKind::Emission,
            AovKind::Visibility,
        ];
        let aov_frames = AovFrames::new(&g.device, width, height, &aovs_all);
        let aov_views: Vec<wgpu::TextureView> = aovs_all
            .iter()
            .map(|k| {
                aov_frames
                    .get_texture(*k)
                    .unwrap()
                    .create_view(&wgpu::TextureViewDescriptor::default())
            })
            .collect();
        let bg4 = g.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("pt-bg4"),
            layout: &bgl4,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&aov_views[0]),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&aov_views[1]),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&aov_views[2]),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(&aov_views[3]),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::TextureView(&aov_views[4]),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: wgpu::BindingResource::TextureView(&aov_views[5]),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: wgpu::BindingResource::TextureView(&aov_views[6]),
                },
            ],
        });

        // Dispatch
        let mut enc = g
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("pt-encoder"),
            });
        {
            let mut cpass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("pt-cpass"),
                ..Default::default()
            });
            cpass.set_pipeline(&pipeline);
            cpass.set_bind_group(0, &bg0, &[]);
            cpass.set_bind_group(1, &bg1, &[]);
            cpass.set_bind_group(2, &bg2, &[]);
            cpass.set_bind_group(3, &bg3, &[]);
            cpass.set_bind_group(4, &bg4, &[]);
            let gx = (width + 7) / 8;
            let gy = (height + 7) / 8;
            cpass.dispatch_workgroups(gx, gy, 1);
        }

        // Copy texture to buffer
        let row_bytes = width * 8; // rgba16f = 8 bytes per pixel
        let padded_bpr = align_copy_bpr(row_bytes);
        let read_size = (padded_bpr as u64) * (height as u64);
        let read_buf = g.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("pt-read"),
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

        g.queue.submit([enc.finish()]);
        g.device.poll(wgpu::Maintain::Wait);

        // Map and convert to RGBA8
        let slice = read_buf.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |res| {
            let _ = tx.send(res);
        });
        g.device.poll(wgpu::Maintain::Wait);
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
                let gch = f16::from_bits(u16::from_le_bytes([row[o + 2], row[o + 3]])).to_f32();
                let b = f16::from_bits(u16::from_le_bytes([row[o + 4], row[o + 5]])).to_f32();
                // already tonemapped in shader; clamp and convert
                let ix = y * dst_stride + x * 4;
                out[ix + 0] = (r.clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
                out[ix + 1] = (gch.clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
                out[ix + 2] = (b.clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
                out[ix + 3] = 255u8;
            }
        }

        drop(data);
        read_buf.unmap();
        Ok(out)
    }

    /// Render AOVs and read them back to CPU buffers.
    pub fn render_aovs(
        width: u32,
        height: u32,
        spheres: &[Sphere],
        mut uniforms: Uniforms,
        aov_mask: u32,
    ) -> Result<std::collections::HashMap<AovKind, Vec<u8>>, RenderError> {
        // Reuse the RGBA path setup but capture AOV textures as well and sequentially copy out
        let g = ctx();
        // Force aov flags
        uniforms.aov_flags = aov_mask;

        // Build shader/pipeline and resources identically as in render()
        // For simplicity, call into render() to execute the dispatch once, then re-create the AOV resources here as they were used.
        // Duplicate minimal setup to get access to AOV textures for readback.

        // Shader
        let shader = g.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("pt_kernel"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/pt_kernel.wgsl").into()),
        });

        // BGLs 0..4 (same as in render)
        let bgl0 = g
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("pt-bgl0-uniforms"),
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
            });
        let bgl1 = g
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("pt-bgl1-scene"),
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
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
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
                ],
            });
        let bgl2 = g
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("pt-bgl2-accum"),
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
            });
        let bgl3 = g
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("pt-bgl3-out"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba16Float,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                }],
            });
        let bgl4 = g
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("pt-bgl4-aovs"),
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
                            format: wgpu::TextureFormat::R8Unorm,
                            view_dimension: wgpu::TextureViewDimension::D2,
                        },
                        count: None,
                    },
                ],
            });

        let pipeline_layout = g
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("pt-pipeline-layout"),
                bind_group_layouts: &[&bgl0, &bgl1, &bgl2, &bgl3, &bgl4],
                push_constant_ranges: &[],
            });
        let pipeline = g
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("pt-compute"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: "main",
            });

        // Buffers
        let ubo = g
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("pt-ubo"),
                contents: bytemuck::bytes_of(&uniforms),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });
        let scene_bytes = bytemuck::cast_slice(spheres);
        let scene_buf = g
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("pt-scene"),
                contents: scene_bytes,
                usage: wgpu::BufferUsages::STORAGE,
            });
        let accum_size = (width as u64) * (height as u64) * 16; // vec4<f32>
        let accum_buf = g.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("pt-accum"),
            size: accum_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let out_tex = g.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("pt-out-tex"),
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

        let bg0 = g.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("pt-bg0"),
            layout: &bgl0,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: ubo.as_entire_binding(),
            }],
        });
        let (mesh_vertices, mesh_indices, mesh_bvh) = create_empty_mesh_buffers(&g.device);
        let bg1 = g.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("pt-bg1"),
            layout: &bgl1,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: scene_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: mesh_vertices.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: mesh_indices.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: mesh_bvh.as_entire_binding(),
                },
            ],
        });
        let bg2 = g.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("pt-bg2"),
            layout: &bgl2,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: accum_buf.as_entire_binding(),
            }],
        });
        let bg3 = g.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("pt-bg3"),
            layout: &bgl3,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&out_view),
            }],
        });

        let aovs_all = [
            AovKind::Albedo,
            AovKind::Normal,
            AovKind::Depth,
            AovKind::Direct,
            AovKind::Indirect,
            AovKind::Emission,
            AovKind::Visibility,
        ];
        let aov_frames = AovFrames::new(&g.device, width, height, &aovs_all);
        let aov_views: Vec<wgpu::TextureView> = aovs_all
            .iter()
            .map(|k| {
                aov_frames
                    .get_texture(*k)
                    .unwrap()
                    .create_view(&wgpu::TextureViewDescriptor::default())
            })
            .collect();
        let bg4 = g.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("pt-bg4"),
            layout: &bgl4,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&aov_views[0]),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&aov_views[1]),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&aov_views[2]),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(&aov_views[3]),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::TextureView(&aov_views[4]),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: wgpu::BindingResource::TextureView(&aov_views[5]),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: wgpu::BindingResource::TextureView(&aov_views[6]),
                },
            ],
        });

        // Dispatch
        let mut enc = g
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("pt-encoder"),
            });
        {
            let mut cpass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("pt-cpass"),
                ..Default::default()
            });
            cpass.set_pipeline(&pipeline);
            cpass.set_bind_group(0, &bg0, &[]);
            cpass.set_bind_group(1, &bg1, &[]);
            cpass.set_bind_group(2, &bg2, &[]);
            cpass.set_bind_group(3, &bg3, &[]);
            cpass.set_bind_group(4, &bg4, &[]);
            let gx = (width + 7) / 8;
            let gy = (height + 7) / 8;
            cpass.dispatch_workgroups(gx, gy, 1);
        }
        g.queue.submit([enc.finish()]);
        g.device.poll(wgpu::Maintain::Wait);

        // Prepare a single staging buffer reused per AOV
        let mut out_map: std::collections::HashMap<AovKind, Vec<u8>> =
            std::collections::HashMap::new();

        // Helper closure to copy a texture to CPU vector
        let mut copy_tex = |kind: AovKind| -> Result<(), RenderError> {
            if (aov_mask & (1u32 << kind.flag_bit())) == 0 {
                return Ok(());
            }
            let format = kind.texture_format();
            let (bytes_per_pixel, convert_rgba16f_to_rgb_f32): (u32, bool) = match format {
                wgpu::TextureFormat::Rgba16Float => (8, true),
                wgpu::TextureFormat::R32Float => (4, false),
                wgpu::TextureFormat::R8Unorm => (1, false),
                _ => return Err(RenderError::Readback("Unsupported AOV format".into())),
            };

            let tex = aov_frames
                .get_texture(kind)
                .ok_or_else(|| RenderError::Readback("Missing AOV texture".into()))?;
            let row_bytes = width * bytes_per_pixel;
            let padded_bpr = align_copy_bpr(row_bytes);
            let read_size = (padded_bpr as u64) * (height as u64);
            let read_buf = g.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("pt-aov-read"),
                size: read_size,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });

            // Encode copy
            let mut enc = g
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("pt-aov-copy-enc"),
                });
            enc.copy_texture_to_buffer(
                wgpu::ImageCopyTexture {
                    texture: tex,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::ImageCopyBuffer {
                    buffer: &read_buf,
                    layout: wgpu::ImageDataLayout {
                        offset: 0,
                        bytes_per_row: Some(std::num::NonZeroU32::new(padded_bpr).unwrap().into()),
                        rows_per_image: Some(std::num::NonZeroU32::new(height).unwrap().into()),
                    },
                },
                wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
            );
            g.queue.submit([enc.finish()]);
            g.device.poll(wgpu::Maintain::Wait);

            let slice = read_buf.slice(..);
            let (tx, rx) = std::sync::mpsc::channel();
            slice.map_async(wgpu::MapMode::Read, move |res| {
                let _ = tx.send(res);
            });
            g.device.poll(wgpu::Maintain::Wait);
            rx.recv()
                .map_err(|_| RenderError::Readback("map_async channel closed".into()))?
                .map_err(|e| RenderError::Readback(format!("MapAsync failed: {:?}", e)))?;
            let data = slice.get_mapped_range();

            if convert_rgba16f_to_rgb_f32 {
                let mut out = vec![0u8; (width as usize) * (height as usize) * 3 * 4];
                let dst_stride = (width as usize) * 3 * 4;
                for y in 0..(height as usize) {
                    let row = &data[(y as usize) * (padded_bpr as usize)
                        ..(y as usize) * (padded_bpr as usize) + (width as usize) * 8];
                    for x in 0..(width as usize) {
                        let o = x * 8;
                        let r = half::f16::from_bits(u16::from_le_bytes([row[o + 0], row[o + 1]]))
                            .to_f32();
                        let gch =
                            half::f16::from_bits(u16::from_le_bytes([row[o + 2], row[o + 3]]))
                                .to_f32();
                        let b = half::f16::from_bits(u16::from_le_bytes([row[o + 4], row[o + 5]]))
                            .to_f32();
                        let ix = y * dst_stride + x * 12;
                        out[ix + 0..ix + 4].copy_from_slice(&r.to_le_bytes());
                        out[ix + 4..ix + 8].copy_from_slice(&gch.to_le_bytes());
                        out[ix + 8..ix + 12].copy_from_slice(&b.to_le_bytes());
                    }
                }
                out_map.insert(kind, out);
            } else if bytes_per_pixel == 4 {
                let mut out = vec![0u8; (width as usize) * (height as usize) * 4];
                let dst_stride = (width as usize) * 4;
                for y in 0..(height as usize) {
                    let src = &data[(y as usize) * (padded_bpr as usize)
                        ..(y as usize) * (padded_bpr as usize) + (width as usize) * 4];
                    let dst = &mut out[y * dst_stride..y * dst_stride + (width as usize) * 4];
                    dst.copy_from_slice(src);
                }
                out_map.insert(kind, out);
            } else {
                // r8
                let mut out = vec![0u8; (width as usize) * (height as usize)];
                let dst_stride = width as usize;
                for y in 0..(height as usize) {
                    let src = &data[(y as usize) * (padded_bpr as usize)
                        ..(y as usize) * (padded_bpr as usize) + (width as usize) * 1];
                    let dst = &mut out[y * dst_stride..y * dst_stride + (width as usize) * 1];
                    dst.copy_from_slice(src);
                }
                out_map.insert(kind, out);
            }

            drop(data);
            read_buf.unmap();
            Ok(())
        };

        for &k in &aovs_all {
            copy_tex(k)?;
        }

        Ok(out_map)
    }
}
