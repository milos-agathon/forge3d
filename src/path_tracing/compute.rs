// src/path_tracing/compute.rs
// Minimal GPU compute path tracer implementation (A1) using WGSL kernel and wgpu.
// This exists to create the compute pipeline, allocate buffers/textures, dispatch, and read back RGBA.
// RELEVANT FILES:src/path_tracing/mod.rs,src/shaders/pt_kernel.wgsl,python/forge3d/path_tracing.py,src/lib.rs

use std::num::NonZeroU32;
use std::sync::Arc;

use bytemuck::{Pod, Zeroable};
use half::f16;
use wgpu::util::DeviceExt;

use crate::error::RenderError;
use crate::gpu::{align_copy_bpr, ctx};

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug)]
pub struct Uniforms {
    pub width: u32,
    pub height: u32,
    pub frame_index: u32,
    pub _pad0: u32,
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
    pub center: [f32; 3],
    pub radius: f32,
    pub albedo: [f32; 3],
    pub _pad0: f32,
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
        let bgl0 = g.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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
        let bgl1 = g.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("pt-bgl1-scene"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        let bgl2 = g.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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
        let bgl3 = g.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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

        let pipeline_layout = g.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("pt-pipeline-layout"),
            bind_group_layouts: &[&bgl0, &bgl1, &bgl2, &bgl3],
            push_constant_ranges: &[],
        });

        let pipeline = g.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("pt-compute"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: "main",
        });

        // Buffers
        let ubo = g.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("pt-ubo"),
            contents: bytemuck::bytes_of(&uniforms),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let scene_bytes = bytemuck::cast_slice(spheres);
        let scene_buf = g.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
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

        // Output texture (RGBA16F)
        let out_tex = g.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("pt-out-tex"),
            size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
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
            entries: &[wgpu::BindGroupEntry { binding: 0, resource: ubo.as_entire_binding() }],
        });
        let bg1 = g.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("pt-bg1"),
            layout: &bgl1,
            entries: &[wgpu::BindGroupEntry { binding: 0, resource: scene_buf.as_entire_binding() }],
        });
        let bg2 = g.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("pt-bg2"),
            layout: &bgl2,
            entries: &[wgpu::BindGroupEntry { binding: 0, resource: accum_buf.as_entire_binding() }],
        });
        let bg3 = g.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("pt-bg3"),
            layout: &bgl3,
            entries: &[wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&out_view) }],
        });

        // Dispatch
        let mut enc = g.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("pt-encoder") });
        {
            let mut cpass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("pt-cpass"), ..Default::default() });
            cpass.set_pipeline(&pipeline);
            cpass.set_bind_group(0, &bg0, &[]);
            cpass.set_bind_group(1, &bg1, &[]);
            cpass.set_bind_group(2, &bg2, &[]);
            cpass.set_bind_group(3, &bg3, &[]);
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
            wgpu::ImageCopyTexture { texture: &out_tex, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
            wgpu::ImageCopyBuffer {
                buffer: &read_buf,
                layout: wgpu::ImageDataLayout {
                    offset: 0,
                    bytes_per_row: Some(NonZeroU32::new(padded_bpr).unwrap().into()),
                    rows_per_image: Some(NonZeroU32::new(height).unwrap().into()),
                },
            },
            wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
        );

        g.queue.submit([enc.finish()]);
        g.device.poll(wgpu::Maintain::Wait);

        // Map and convert to RGBA8
        let slice = read_buf.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |res| { let _ = tx.send(res); });
        g.device.poll(wgpu::Maintain::Wait);
        rx.recv().map_err(|_| RenderError::Readback("map_async channel closed".into()))?
            .map_err(|e| RenderError::Readback(format!("MapAsync failed: {:?}", e)))?;
        let data = slice.get_mapped_range();

        let mut out = vec![0u8; (width as usize) * (height as usize) * 4];
        let src_stride = padded_bpr as usize;
        let dst_stride = (width as usize) * 4;
        for y in 0..(height as usize) {
            let row = &data[y * src_stride .. y * src_stride + (width as usize) * 8];
            for x in 0..(width as usize) {
                let o = x * 8;
                let r = f16::from_bits(u16::from_le_bytes([row[o+0], row[o+1]])).to_f32();
                let gch = f16::from_bits(u16::from_le_bytes([row[o+2], row[o+3]])).to_f32();
                let b = f16::from_bits(u16::from_le_bytes([row[o+4], row[o+5]])).to_f32();
                // already tonemapped in shader; clamp and convert
                let ix = y * dst_stride + x * 4;
                out[ix+0] = (r.clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
                out[ix+1] = (gch.clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
                out[ix+2] = (b.clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
                out[ix+3] = 255u8;
            }
        }

        drop(data);
        read_buf.unmap();
        Ok(out)
    }
}
