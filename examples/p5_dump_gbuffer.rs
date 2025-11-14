// examples/p5_dump_gbuffer.rs
// P5.0: Standalone exporter that renders one frame (GI off), builds HZB from
// the real Depth32Float attachment, and writes required artifacts under reports/p5/.
// Artifacts:
//  - p5_gbuffer_normals.png: RGB in [0,1] of view-space normal encode
//  - p5_gbuffer_material.png: RGB from material/albedo target
//  - p5_gbuffer_depth_mips.png: horizontal strip of HZB mips 0..4 remapped to [0,1]
//  - p5_meta.json: metadata with formats, sizes, adapter name and WGSL hashes
//
// Usage:
//   cargo run --release --example p5_dump_gbuffer -- \
//     --size 1280 720 \
//     --out reports/p5
//
// Note: This example keeps behavior identical to the viewer's current pipeline.

use forge3d::core::screen_space_effects::ScreenSpaceEffectsManager;
use forge3d::util::image_write;
use glam::Mat4;
use sha2::{Digest, Sha256};
use std::fs;
use std::path::PathBuf;
use wgpu::util::DeviceExt;

fn parse_args() -> (u32, u32, PathBuf) {
    let mut w: u32 = 1280;
    let mut h: u32 = 720;
    let mut out = PathBuf::from("reports/p5");
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--size" => {
                if let (Some(ws), Some(hs)) = (it.next(), it.next()) {
                    if let (Ok(wp), Ok(hp)) = (ws.parse::<u32>(), hs.parse::<u32>()) {
                        w = wp;
                        h = hp;
                    }
                }
            }
            "--out" => {
                if let Some(p) = it.next() {
                    out = PathBuf::from(p);
                }
            }
            _ => {}
        }
    }
    (w.max(1), h.max(1), out)
}

fn to_arr4(m: Mat4) -> [[f32; 4]; 4] {
    let c = m.to_cols_array();
    [
        [c[0], c[1], c[2], c[3]],
        [c[4], c[5], c[6], c[7]],
        [c[8], c[9], c[10], c[11]],
        [c[12], c[13], c[14], c[15]],
    ]
}

fn main() -> anyhow::Result<()> {
    // Parse CLI
    let (width, height, out_dir) = parse_args();
    fs::create_dir_all(&out_dir)?;

    // Init WGPU (headless)
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::default());
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
    }))
    .ok_or_else(|| anyhow::anyhow!("No suitable GPU adapter found"))?;
    let (device, queue) = pollster::block_on(adapter.request_device(
        &wgpu::DeviceDescriptor {
            label: Some("P5 Exporter Device"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::default(),
        },
        None,
    ))?;

    // Build GI manager (gbuffer + optional HZB holder)
    println!("[P5] Creating GI manager {}x{}", width, height);
    let gi = ScreenSpaceEffectsManager::new(&device, width, height)
        .map_err(|e| anyhow::anyhow!("GI manager init failed: {}", e))?;

    // Depth32Float attachment for raster depth test
    let z_texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("p5.export.depth32"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Depth32Float,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let z_view = z_texture.create_view(&wgpu::TextureViewDescriptor::default());

    // Camera buffer
    let cam_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("p5.export.cam"),
        size: (std::mem::size_of::<[[f32; 4]; 4]>() * 2) as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    // Geometry BGL: camera + albedo + sampler
    let geom_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("p5.export.geom.bgl"),
        entries: &[
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
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            },
        ],
    });

    // Shaders: include gbuffer helpers, then pipeline shader
    let wgsl_common = include_str!("../shaders/gbuffer/common.wgsl");
    let wgsl_pack = include_str!("../shaders/gbuffer/pack.wgsl");
    let geom_src = format!(
        r#"
        {common}
        {pack}
        struct Camera {{ view: mat4x4<f32>, proj: mat4x4<f32> }};
        @group(0) @binding(0) var<uniform> uCam : Camera;
        @group(0) @binding(1) var tAlbedo : texture_2d<f32>;
        @group(0) @binding(2) var sAlbedo : sampler;
        struct VSIn {{ @location(0) pos: vec3<f32>, @location(1) nrm: vec3<f32>, @location(2) uv: vec2<f32> }};
        struct VSOut {{ @builtin(position) pos: vec4<f32>, @location(0) v_nrm_vs: vec3<f32>, @location(1) v_depth_vs: f32, @location(2) v_uv: vec2<f32> }};
        @vertex
        fn vs_main(inp: VSIn) -> VSOut {{
            var out: VSOut;
            let pos_ws = vec4<f32>(inp.pos, 1.0);
            let pos_vs = uCam.view * pos_ws;
            out.pos = uCam.proj * pos_vs;
            let nrm_vs = (uCam.view * vec4<f32>(inp.nrm, 0.0)).xyz;
            out.v_nrm_vs = normalize(nrm_vs);
            out.v_depth_vs = -pos_vs.z; // positive view-space depth
            out.v_uv = inp.uv;
            return out;
        }}
        struct FSOut {{ @location(0) normal_rgba: vec4<f32>, @location(1) albedo_rgba: vec4<f32>, @location(2) depth_r: f32 }};
        @fragment
        fn fs_main(inp: VSOut) -> FSOut {{
            var out: FSOut;
            // Pack normal to [0,1] but still store in RGBA16F target (no behavior change downstream)
            let enc = pack_normal(normalize(inp.v_nrm_vs));
            out.normal_rgba = vec4<f32>(enc, 1.0);
            let color = textureSample(tAlbedo, sAlbedo, inp.v_uv);
            out.albedo_rgba = vec4<f32>(color.rgb, 1.0);
            out.depth_r = inp.v_depth_vs;
            return out;
        }}
    "#,
        common = wgsl_common,
        pack = wgsl_pack
    );

    let geom_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("p5.export.geom.shader"),
        source: wgpu::ShaderSource::Wgsl(geom_src.into()),
    });

    // Geometry vertex buffer (cube)
    let verts: [f32; 36 * 8] = {
        // Embed a small cube inline to keep this example self-contained
        [
            // +Z face
            -1.0, -1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, -1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0,
            1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, -1.0, -1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0,
            1.0, 0.0, 0.0, 1.0, 1.0, 1.0, -1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0,
            // -Z face
            -1.0, -1.0, -1.0, 0.0, 0.0, -1.0, 1.0, 0.0, 1.0, 1.0, -1.0, 0.0, 0.0, -1.0, 0.0, 1.0,
            1.0, -1.0, -1.0, 0.0, 0.0, -1.0, 0.0, 0.0, -1.0, -1.0, -1.0, 0.0, 0.0, -1.0, 1.0, 0.0,
            -1.0, 1.0, -1.0, 0.0, 0.0, -1.0, 1.0, 1.0, 1.0, 1.0, -1.0, 0.0, 0.0, -1.0, 0.0, 1.0,
            // +X face
            1.0, -1.0, -1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0,
            -1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, -1.0, -1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0,
            -1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0,
            // -X face
            -1.0, -1.0, -1.0, -1.0, 0.0, 0.0, 1.0, 0.0, -1.0, -1.0, 1.0, -1.0, 0.0, 0.0, 0.0, 0.0,
            -1.0, 1.0, 1.0, -1.0, 0.0, 0.0, 0.0, 1.0, -1.0, -1.0, -1.0, -1.0, 0.0, 0.0, 1.0, 0.0,
            -1.0, 1.0, 1.0, -1.0, 0.0, 0.0, 0.0, 1.0, -1.0, 1.0, -1.0, -1.0, 0.0, 0.0, 1.0, 1.0,
            // +Y face
            -1.0, 1.0, -1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0,
            1.0, -1.0, 0.0, 1.0, 0.0, 1.0, 1.0, -1.0, 1.0, -1.0, 0.0, 1.0, 0.0, 0.0, 1.0, -1.0,
            1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0,
            // -Y face
            -1.0, -1.0, -1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0, -1.0, -1.0, 0.0, -1.0, 0.0, 1.0, 0.0,
            1.0, -1.0, 1.0, 0.0, -1.0, 0.0, 1.0, 1.0, -1.0, -1.0, -1.0, 0.0, -1.0, 0.0, 0.0, 0.0,
            1.0, -1.0, 1.0, 0.0, -1.0, 0.0, 1.0, 1.0, -1.0, -1.0, 1.0, 0.0, -1.0, 0.0, 0.0, 1.0,
        ]
    };
    let geom_vb = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("p5.export.geom.vb"),
        contents: bytemuck::cast_slice(&verts),
        usage: wgpu::BufferUsages::VERTEX,
    });

    // Procedural albedo texture
    let tex_size = 256u32;
    let mut pixels = vec![0u8; (tex_size * tex_size * 4) as usize];
    for y in 0..tex_size {
        for x in 0..tex_size {
            let idx = ((y * tex_size + x) * 4) as usize;
            let c = if ((x / 32) + (y / 32)) % 2 == 0 {
                230
            } else {
                50
            };
            pixels[idx] = c;
            pixels[idx + 1] = 180;
            pixels[idx + 2] = 80;
            pixels[idx + 3] = 255;
        }
    }
    let albedo_texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("p5.export.albedo"),
        size: wgpu::Extent3d {
            width: tex_size,
            height: tex_size,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8UnormSrgb,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    let albedo_view = albedo_texture.create_view(&wgpu::TextureViewDescriptor::default());
    queue.write_texture(
        wgpu::ImageCopyTexture {
            texture: &albedo_texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        &pixels,
        wgpu::ImageDataLayout {
            offset: 0,
            bytes_per_row: Some(tex_size * 4),
            rows_per_image: Some(tex_size),
        },
        wgpu::Extent3d {
            width: tex_size,
            height: tex_size,
            depth_or_array_layers: 1,
        },
    );
    let albedo_sampler = device.create_sampler(&wgpu::SamplerDescriptor::default());

    let geom_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("p5.export.geom.bg"),
        layout: &geom_bgl,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: cam_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(&albedo_view),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::Sampler(&albedo_sampler),
            },
        ],
    });

    // Pipeline
    let geom_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("p5.export.geom.pl"),
        bind_group_layouts: &[&geom_bgl],
        push_constant_ranges: &[],
    });
    let gb = gi.gbuffer();
    let gb_cfg = gb.config();
    let geom_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("p5.export.geom.pipeline"),
        layout: Some(&geom_pl),
        vertex: wgpu::VertexState {
            module: &geom_shader,
            entry_point: "vs_main",
            buffers: &[wgpu::VertexBufferLayout {
                array_stride: (8 * std::mem::size_of::<f32>()) as u64,
                step_mode: wgpu::VertexStepMode::Vertex,
                attributes: &[
                    wgpu::VertexAttribute {
                        shader_location: 0,
                        offset: 0,
                        format: wgpu::VertexFormat::Float32x3,
                    },
                    wgpu::VertexAttribute {
                        shader_location: 1,
                        offset: (3 * 4) as u64,
                        format: wgpu::VertexFormat::Float32x3,
                    },
                    wgpu::VertexAttribute {
                        shader_location: 2,
                        offset: (6 * 4) as u64,
                        format: wgpu::VertexFormat::Float32x2,
                    },
                ],
            }],
        },
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil: Some(wgpu::DepthStencilState {
            format: wgpu::TextureFormat::Depth32Float,
            depth_write_enabled: true,
            depth_compare: wgpu::CompareFunction::Less,
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState::default(),
        }),
        multisample: wgpu::MultisampleState::default(),
        fragment: Some(wgpu::FragmentState {
            module: &geom_shader,
            entry_point: "fs_main",
            targets: &[
                Some(wgpu::ColorTargetState {
                    format: gb_cfg.normal_format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                }),
                Some(wgpu::ColorTargetState {
                    format: gb_cfg.material_format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                }),
                Some(wgpu::ColorTargetState {
                    format: gb_cfg.depth_format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                }),
            ],
        }),
        multiview: None,
    });

    // Camera matrices
    let aspect = width as f32 / height as f32;
    let fov = 45.0_f32.to_radians();
    let proj = Mat4::perspective_rh(fov, aspect, 0.1, 1000.0);
    // Simple orbit camera looking at origin
    let eye = glam::Vec3::new(3.0, 2.0, 5.0);
    let target = glam::Vec3::ZERO;
    let up = glam::Vec3::Y;
    let view = Mat4::look_at_rh(eye, target, up);
    queue.write_buffer(
        &cam_buf,
        0,
        bytemuck::cast_slice(&[to_arr4(view), to_arr4(proj)]),
    );

    // Encode pass
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("p5.export.encoder"),
    });
    {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("p5.export.gbuf.pass"),
            color_attachments: &[
                Some(wgpu::RenderPassColorAttachment {
                    view: &gb.normal_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                }),
                Some(wgpu::RenderPassColorAttachment {
                    view: &gb.material_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                }),
                Some(wgpu::RenderPassColorAttachment {
                    view: &gb.depth_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                }),
            ],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &z_view,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(1.0),
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
        });
        pass.set_pipeline(&geom_pipeline);
        pass.set_bind_group(0, &geom_bg, &[]);
        pass.set_vertex_buffer(0, geom_vb.slice(..));
        pass.draw(0..36, 0..1);
    }

    // Build HZB from real depth attachment (5 levels for export: mips 0..4)
    // Reversed-Z detection will happen after render, so use regular-Z (false) for now
    println!("[P5] Building HZB pyramid (5 levels)");
    if let Some(ref hzb) = gi.hzb {
        hzb.build_n(&device, &mut encoder, &z_view, 5, false);
    }

    println!("[P5] Submitting render + HZB build");
    queue.submit(std::iter::once(encoder.finish()));
    device.poll(wgpu::Maintain::Wait);
    println!("[P5] Render complete, starting readbacks");

    // Detect reversed-Z by reading from HZB mip0 (which stores depth as R32F, readable)
    // Reversed-Z: clear=0, near geometry ≈0; Regular-Z: clear=1, near geometry <1
    let (hzb_tex, _mip_count) = gi
        .hzb_texture_and_mips()
        .ok_or_else(|| anyhow::anyhow!("HZB not initialized"))?;
    let bpp_depth = 4u32; // R32F
    let tight_bpr_depth = width * bpp_depth;
    let pad = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
    let padded_bpr_depth = ((tight_bpr_depth + pad - 1) / pad) * pad;
    let depth_staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("p5.depth_sample_staging"),
        size: (padded_bpr_depth * height) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut depth_sample_encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("p5.depth_sample"),
    });
    depth_sample_encoder.copy_texture_to_buffer(
        wgpu::ImageCopyTexture {
            texture: hzb_tex,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::ImageCopyBuffer {
            buffer: &depth_staging,
            layout: wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(padded_bpr_depth),
                rows_per_image: Some(height),
            },
        },
        wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
    );
    queue.submit(std::iter::once(depth_sample_encoder.finish()));
    device.poll(wgpu::Maintain::Wait);
    let sample_slice = depth_staging.slice(..);
    let (tx_s, rx_s) = futures_intrusive::channel::shared::oneshot_channel();
    sample_slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx_s.send(r);
    });
    device.poll(wgpu::Maintain::Wait);
    pollster::block_on(rx_s.receive())
        .ok_or_else(|| anyhow::anyhow!("depth sample map failed"))??;
    let sample_data = sample_slice.get_mapped_range();
    // Sample center pixel from HZB mip0
    let center_y = (height / 2) as usize;
    let center_x = (width / 2) as usize;
    let center_offset = center_y * (padded_bpr_depth as usize) + center_x * 4;
    let depth_center = f32::from_le_bytes([
        sample_data[center_offset],
        sample_data[center_offset + 1],
        sample_data[center_offset + 2],
        sample_data[center_offset + 3],
    ]);
    drop(sample_data);
    depth_staging.unmap();
    // Heuristic: if center depth < 0.5, likely reversed-Z (near=0); else regular-Z (near approaches 0 from 1)
    let reversed_z = depth_center < 0.5;
    println!(
        "[P5] Detected depth convention: {} (center depth = {:.6})",
        if reversed_z {
            "reversed-Z"
        } else {
            "regular-Z"
        },
        depth_center
    );

    // Batch readback: normals (RGBA16F) and material (RGBA8) in a single GPU submit + map
    let bytes_per_pixel_norm = 8u32; // RGBA16F
    let bytes_per_pixel_mat = 4u32; // RGBA8
    let tight_bpr_norm = width * bytes_per_pixel_norm;
    let tight_bpr_mat = width * bytes_per_pixel_mat;
    let pad = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
    let padded_bpr_norm = ((tight_bpr_norm + pad - 1) / pad) * pad;
    let padded_bpr_mat = ((tight_bpr_mat + pad - 1) / pad) * pad;
    // Offsets in a single staging buffer (align to 4 bytes for safety)
    let off_norm: u64 = 0;
    let off_mat: u64 = ((off_norm + (padded_bpr_norm as u64 * height as u64)) + 3) & !3u64;
    let staging_sz = off_mat + (padded_bpr_mat as u64 * height as u64);
    let staging_nm = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("p5.export.gbuffer.staging.nm"),
        size: staging_sz,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut enc_nm = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("p5.export.read.nm"),
    });
    // Copy normals (RGBA16F)
    enc_nm.copy_texture_to_buffer(
        wgpu::ImageCopyTexture {
            texture: &gb.normal_texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::ImageCopyBuffer {
            buffer: &staging_nm,
            layout: wgpu::ImageDataLayout {
                offset: off_norm,
                bytes_per_row: Some(padded_bpr_norm),
                rows_per_image: Some(height),
            },
        },
        wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
    );
    // Copy material (RGBA8)
    enc_nm.copy_texture_to_buffer(
        wgpu::ImageCopyTexture {
            texture: &gb.material_texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::ImageCopyBuffer {
            buffer: &staging_nm,
            layout: wgpu::ImageDataLayout {
                offset: off_mat,
                bytes_per_row: Some(padded_bpr_mat),
                rows_per_image: Some(height),
            },
        },
        wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
    );
    queue.submit(std::iter::once(enc_nm.finish()));
    device.poll(wgpu::Maintain::Wait);
    let slice_nm = staging_nm.slice(..);
    let (tx_nm, rx_nm) = futures_intrusive::channel::shared::oneshot_channel();
    slice_nm.map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx_nm.send(r);
    });
    device.poll(wgpu::Maintain::Wait);
    pollster::block_on(rx_nm.receive()).ok_or_else(|| anyhow::anyhow!("map failed"))??;
    let data_nm = slice_nm.get_mapped_range();
    // Build tight outputs
    let mut norm_rgba8 = vec![0u8; (width * height * 4) as usize];
    for y in 0..height as usize {
        let row_start = off_norm as usize + y * (padded_bpr_norm as usize);
        let row = &data_nm[row_start..row_start + (tight_bpr_norm as usize)];
        // Each pixel: 8 bytes (RGBA16F)
        for x in 0..width as usize {
            let off = x * 8;
            let rx = half::f16::from_le_bytes([row[off + 0], row[off + 1]]).to_f32();
            let ry = half::f16::from_le_bytes([row[off + 2], row[off + 3]]).to_f32();
            let rz = half::f16::from_le_bytes([row[off + 4], row[off + 5]]).to_f32();
            let r = (rx.clamp(0.0, 1.0) * 255.0) as u8;
            let g = (ry.clamp(0.0, 1.0) * 255.0) as u8;
            let b = (rz.clamp(0.0, 1.0) * 255.0) as u8;
            let o8 = (y * width as usize + x) * 4;
            norm_rgba8[o8] = r;
            norm_rgba8[o8 + 1] = g;
            norm_rgba8[o8 + 2] = b;
            norm_rgba8[o8 + 3] = 255;
        }
    }
    println!("[P5] Writing normals PNG");
    image_write::write_png_rgba8(
        &out_dir.join("p5_gbuffer_normals.png"),
        &norm_rgba8,
        width,
        height,
    )?;
    // Material rows: depad into tight RGBA8 buffer
    let mut mat_tight = vec![0u8; (width * height * 4) as usize];
    for y in 0..height as usize {
        let src_off = off_mat as usize + y * (padded_bpr_mat as usize);
        let dst_off = y * (tight_bpr_mat as usize);
        mat_tight[dst_off..dst_off + (tight_bpr_mat as usize)]
            .copy_from_slice(&data_nm[src_off..src_off + (tight_bpr_mat as usize)]);
    }
    println!("[P5] Writing material PNG");
    image_write::write_png_rgba8(
        &out_dir.join("p5_gbuffer_material.png"),
        &mat_tight,
        width,
        height,
    )?;
    drop(data_nm);
    staging_nm.unmap();

    // Depth HZB: read mips 0..4 and pack into a horizontal mosaic with 8px gutters
    println!("[P5] Reading HZB mips for mosaic");
    let (hzb_tex, mip_count) = gi
        .hzb_texture_and_mips()
        .ok_or_else(|| anyhow::anyhow!("HZB not initialized"))?;
    let show = mip_count.min(5);
    // Compute sizes for each mip and mosaic canvas (with 8px gutters between tiles)
    let gutter = 8u32;
    let mut dims: Vec<(u32, u32)> = Vec::new();
    let mut cw = width;
    let mut ch = height;
    let mut canvas_w = 0u32;
    let mut canvas_h = 0u32;
    for i in 0..show {
        dims.push((cw, ch));
        canvas_w += cw;
        if i < show - 1 {
            canvas_w += gutter;
        }
        canvas_h = canvas_h.max(ch);
        cw = (cw / 2).max(1);
        ch = (ch / 2).max(1);
    }

    // Batch all mip readbacks in one submission to minimize GPU-CPU sync costs
    let mut canvas = vec![0u8; (canvas_w * canvas_h * 4) as usize];
    // Precompute per-mip copy layout and offsets in a single staging buffer
    let bpp = 4u32; // R32F
    let mut copy_plans: Vec<(
        u64, /*offset*/
        u32, /*padded_bpr*/
        u32, /*mw*/
        u32, /*mh*/
    )> = Vec::with_capacity(dims.len());
    let mut total_bytes: u64 = 0;
    for (mw, mh) in dims.iter().copied() {
        let tight_bpr = mw * bpp;
        let pad = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        let padded_bpr = ((tight_bpr + pad - 1) / pad) * pad; // multiple of 256
                                                              // Ensure buffer offset alignment (wgpu::COPY_BUFFER_ALIGNMENT is 4 bytes)
        let offset = (total_bytes + 3) & !3u64; // align up to 4
        copy_plans.push((offset, padded_bpr, mw, mh));
        total_bytes = offset + (padded_bpr as u64) * (mh as u64);
    }

    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("p5.export.hzb.staging.batch"),
        size: total_bytes,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("p5.export.hzb.read.batch"),
    });
    for (level, (offset, padded_bpr, mw, mh)) in copy_plans.iter().copied().enumerate() {
        enc.copy_texture_to_buffer(
            wgpu::ImageCopyTexture {
                texture: hzb_tex,
                mip_level: level as u32,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::ImageCopyBuffer {
                buffer: &staging,
                layout: wgpu::ImageDataLayout {
                    offset,
                    bytes_per_row: Some(padded_bpr),
                    rows_per_image: Some(mh),
                },
            },
            wgpu::Extent3d {
                width: mw,
                height: mh,
                depth_or_array_layers: 1,
            },
        );
    }
    queue.submit(std::iter::once(enc.finish()));
    device.poll(wgpu::Maintain::Wait);

    let slice = staging.slice(..);
    let (tx, rx) = futures_intrusive::channel::shared::oneshot_channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    device.poll(wgpu::Maintain::Wait);
    pollster::block_on(rx.receive()).ok_or_else(|| anyhow::anyhow!("map failed"))??;
    let data = slice.get_mapped_range();

    // Per-mip normalization: compute min/max per mip and normalize independently
    let mut xoff = 0u32;
    for (i, (_offset, padded_bpr, mw, mh)) in copy_plans.iter().copied().enumerate() {
        let base = copy_plans[i].0 as usize;
        let tight_bpr = mw * bpp;
        // First pass: collect all depth values for this mip
        let mut depths = Vec::with_capacity((mw * mh) as usize);
        for y in 0..mh as usize {
            let row_start = base + y * (padded_bpr as usize);
            let row = &data[row_start..row_start + (tight_bpr as usize)];
            for x in 0..mw as usize {
                let off = x * 4;
                let val = f32::from_le_bytes([row[off], row[off + 1], row[off + 2], row[off + 3]]);
                depths.push(val);
            }
        }
        // Compute min/max for this mip
        let mut min_d = f32::INFINITY;
        let mut max_d = f32::NEG_INFINITY;
        for &d in &depths {
            if d.is_finite() {
                min_d = min_d.min(d);
                max_d = max_d.max(d);
            }
        }
        // Avoid division by zero
        let range = (max_d - min_d).max(1e-6);
        // Second pass: normalize and write to canvas
        let mut depth_idx = 0;
        for y in 0..mh as usize {
            for x in 0..mw as usize {
                let val = depths[depth_idx];
                depth_idx += 1;
                // Normalize to [0,1] using this mip's min/max
                let normalized = ((val - min_d) / range).clamp(0.0, 1.0);
                let g = (normalized * 255.0) as u8;
                let gx = (xoff + x as u32) as usize;
                let gy = y as usize;
                let goff = (gy * (canvas_w as usize) + gx) * 4;
                canvas[goff] = g;
                canvas[goff + 1] = g;
                canvas[goff + 2] = g;
                canvas[goff + 3] = 255;
            }
        }
        xoff += mw + gutter;
    }
    drop(data);
    staging.unmap();
    println!("[P5] Writing depth mosaic PNG");
    image_write::write_png_rgba8(
        &out_dir.join("p5_gbuffer_depth_mips.png"),
        &canvas,
        canvas_w,
        canvas_h,
    )?;

    // Meta JSON with WGSL hashes
    fn fmt_fmt(f: wgpu::TextureFormat) -> String {
        format!("{:?}", f)
    }
    fn sha256_hex(bytes: &[u8]) -> String {
        let mut hasher = Sha256::new();
        hasher.update(bytes);
        format!("{:x}", hasher.finalize())
    }

    // Hash WGSL sources as consumed by wgpu (exact include_str! contents)
    let mut wgsl_hashes = serde_json::Map::new();
    wgsl_hashes.insert(
        "hzb_build.wgsl".to_string(),
        serde_json::Value::String(sha256_hex(
            include_str!("../shaders/hzb_build.wgsl").as_bytes(),
        )),
    );
    wgsl_hashes.insert(
        "gbuffer/common.wgsl".to_string(),
        serde_json::Value::String(sha256_hex(
            include_str!("../shaders/gbuffer/common.wgsl").as_bytes(),
        )),
    );
    wgsl_hashes.insert(
        "gbuffer/pack.wgsl".to_string(),
        serde_json::Value::String(sha256_hex(
            include_str!("../shaders/gbuffer/pack.wgsl").as_bytes(),
        )),
    );

    let info = adapter.get_info();
    let gb_cfg = gi.gbuffer().config();
    let meta = serde_json::json!({
        "adapter": info.name,
        "device": "P5 Exporter Device",
        "rt_size": [width, height],
        "reversed_z": reversed_z,
        "depth_sample_center": depth_center,
        "gbuffer": {"depth_format": fmt_fmt(gb_cfg.depth_format), "normal_format": fmt_fmt(gb_cfg.normal_format), "material_format": fmt_fmt(gb_cfg.material_format)},
        "hzb": {"format": "R32Float", "mips": show, "base_size": [width, height], "total_pyramid_mips": mip_count},
        "depth_mips_exported": show,
        "depth_mosaic_path": "p5_gbuffer_depth_mips.png",
        "depth_mosaic_png": true,
        "wgsl_hashes": wgsl_hashes
    });
    println!("[P5] Writing metadata JSON");
    std::fs::write(
        out_dir.join("p5_meta.json"),
        serde_json::to_vec_pretty(&meta)?,
    )?;

    println!("[P5] Wrote reports/p5 artifacts to {}", out_dir.display());
    Ok(())
}
