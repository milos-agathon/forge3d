// src/viewer/viewer_render_helpers.rs
// GPU render helper functions for the interactive viewer
// RELEVANT FILES: src/viewer/mod.rs

use crate::renderer::readback::read_texture_tight;
use wgpu::util::DeviceExt;

/// Arguments for render_view_to_rgba8_ex
pub struct RenderViewArgs<'a> {
    pub device: &'a wgpu::Device,
    pub queue: &'a wgpu::Queue,
    pub comp_pl: &'a wgpu::RenderPipeline,
    pub comp_bgl: &'a wgpu::BindGroupLayout,
    pub sky_view: &'a wgpu::TextureView,
    pub depth_view: &'a wgpu::TextureView,
    pub fog_view: &'a wgpu::TextureView,
    pub surface_format: wgpu::TextureFormat,
    pub width: u32,
    pub height: u32,
    pub far: f32,
    pub src_view: &'a wgpu::TextureView,
    pub mode: u32,
}

/// Render a texture view through the compositor pipeline and return as RGBA8 bytes
pub fn render_view_to_rgba8_ex(args: RenderViewArgs) -> anyhow::Result<Vec<u8>> {
    use anyhow::Context;
    let RenderViewArgs {
        device,
        queue,
        comp_pl,
        comp_bgl,
        sky_view,
        depth_view,
        fog_view,
        surface_format,
        width,
        height,
        far,
        src_view,
        mode,
    } = args;

    // Uniform for mode and far
    let params: [f32; 4] = [mode as f32, far, 0.0, 0.0];
    let ub = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("p51.comp.params"),
        contents: bytemuck::cast_slice(&params),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });
    // Sampler
    let comp_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("p51.comp.sampler"),
        mag_filter: wgpu::FilterMode::Nearest,
        min_filter: wgpu::FilterMode::Nearest,
        mipmap_filter: wgpu::FilterMode::Nearest,
        ..Default::default()
    });
    // Bind group
    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("p51.comp.bg"),
        layout: comp_bgl,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(src_view),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Sampler(&comp_sampler),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: ub.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: wgpu::BindingResource::TextureView(sky_view),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: wgpu::BindingResource::TextureView(depth_view),
            },
            wgpu::BindGroupEntry {
                binding: 5,
                resource: wgpu::BindingResource::TextureView(fog_view),
            },
        ],
    });
    // Offscreen texture
    let tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("p51.offscreen"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: surface_format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("p51.comp.encoder"),
    });
    {
        let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("p51.comp.pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color {
                        r: 0.0,
                        g: 0.0,
                        b: 0.0,
                        a: 1.0,
                    }),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        });
        pass.set_pipeline(comp_pl);
        pass.set_bind_group(0, &bg, &[]);
        pass.draw(0..3, 0..1);
    }
    queue.submit(std::iter::once(enc.finish()));
    // Read back and format
    let mut data = read_texture_tight(device, queue, &tex, (width, height), surface_format)
        .context("read back offscreen")?;
    match surface_format {
        wgpu::TextureFormat::Bgra8Unorm | wgpu::TextureFormat::Bgra8UnormSrgb => {
            for px in data.chunks_exact_mut(4) {
                px.swap(0, 2);
            }
        }
        _ => {}
    }
    for px in data.chunks_exact_mut(4) {
        px[3] = 255;
    }
    Ok(data)
}
