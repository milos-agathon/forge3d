//! Headless renderer for a deterministic triangle with off-screen target + readback.
//! Rust: wgpu 0.19, PyO3 0.21 (abi3). Returns (H,W,4) u8 arrays via numpy.

use std::num::NonZeroU32;

use bytemuck::{Pod, Zeroable};
use image::ImageBuffer;
use once_cell::sync::OnceCell;
use pyo3::prelude::*;
use numpy::{PyArray3, IntoPyArray};
use ndarray::Array3;
use wgpu::util::DeviceExt;

const TEXTURE_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8UnormSrgb;
const CLEAR_COLOR: wgpu::Color = wgpu::Color { r: 1.0, g: 1.0, b: 1.0, a: 1.0 };

// ---------- WGPU device/queue singleton ----------

static WGPU_CTX: OnceCell<WgpuContext> = OnceCell::new();

struct WgpuContext {
    device: wgpu::Device,
    queue: wgpu::Queue,
}

impl WgpuContext {
    fn get() -> &'static Self {
        WGPU_CTX.get_or_init(|| {
            let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
                backends: wgpu::Backends::all(),
                ..Default::default()
            });

            let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            }))
            .expect("No suitable GPU adapter");

            let needed_features = wgpu::Features::empty();
            let limits = wgpu::Limits::downlevel_defaults();

            let (device, queue) = pollster::block_on(adapter.request_device(
                &wgpu::DeviceDescriptor {
                    required_features: needed_features,
                    required_limits: limits,
                    label: Some("vulkan-forge-device"),
                },
                None,
            ))
            .expect("request_device failed");

            Self { device, queue }
        })
    }
}

// ---------- Geometry & pipeline ----------

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Vertex {
    pos: [f32; 2],
    color: [f32; 3],
}

fn triangle_geometry(device: &wgpu::Device) -> (wgpu::Buffer, wgpu::Buffer, u32) {
    let verts: &[Vertex] = &[
        Vertex { pos: [-0.8, -0.8], color: [1.0, 0.2, 0.2] },
        Vertex { pos: [ 0.8, -0.8], color: [0.2, 1.0, 0.2] },
        Vertex { pos: [ 0.0,  0.8], color: [0.2, 0.2, 1.0] },
    ];
    let indices: &[u16] = &[0, 1, 2];

    let vbuf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("triangle-vertices"),
        contents: bytemuck::cast_slice(verts),
        usage: wgpu::BufferUsages::VERTEX,
    });
    let ibuf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("triangle-indices"),
        contents: bytemuck::cast_slice(indices),
        usage: wgpu::BufferUsages::INDEX,
    });
    (vbuf, ibuf, indices.len() as u32)
}

fn create_pipeline(device: &wgpu::Device, format: wgpu::TextureFormat) -> wgpu::RenderPipeline {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("triangle-shader"),
        source: wgpu::ShaderSource::Wgsl(include_str!("shaders/triangle.wgsl").into()),
    });

    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("triangle-pipeline-layout"),
        bind_group_layouts: &[],
        push_constant_ranges: &[],
    });

    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("triangle-pipeline"),
        layout: Some(&layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: "vs_main",
            buffers: &[wgpu::VertexBufferLayout {
                array_stride: std::mem::size_of::<Vertex>() as u64,
                step_mode: wgpu::VertexStepMode::Vertex,
                attributes: &wgpu::vertex_attr_array![0 => Float32x2, 1 => Float32x3],
            }],
        },
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            strip_index_format: None,
            front_face: wgpu::FrontFace::Ccw,
            cull_mode: Some(wgpu::Face::Back),
            ..Default::default()
        },
        depth_stencil: None,
        multisample: wgpu::MultisampleState {
            count: 1,
            mask: !0,
            alpha_to_coverage_enabled: false,
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: "fs_main",
            targets: &[Some(wgpu::ColorTargetState {
                format,
                blend: None, // A1.2 invariant
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        multiview: None,
    })
}

// ---------- Off-screen creation & readback ----------

fn create_offscreen(device: &wgpu::Device, width: u32, height: u32, format: wgpu::TextureFormat) -> (wgpu::Texture, wgpu::TextureView) {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("offscreen-color"),
        size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    (texture, view)
}

fn align256(n: u32) -> u32 { ((n + 255) / 256) * 256 }

fn copy_texture_to_rgba_unpadded(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    src_tex: &wgpu::Texture,
    readback_buf: &wgpu::Buffer,
    width: u32,
    height: u32,
) -> Vec<u8> {
    let row_bytes = width * 4;
    let padded_bpr = align256(row_bytes);

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("copy-encoder"),
    });

    let copy_src = wgpu::ImageCopyTexture {
        texture: src_tex,
        mip_level: 0,
        origin: wgpu::Origin3d::ZERO,
        aspect: wgpu::TextureAspect::All,
    };
    let copy_dst = wgpu::ImageCopyBuffer {
        buffer: readback_buf,
        layout: wgpu::ImageDataLayout {
            offset: 0,
            // Your wgpu build expects Option<u32> (not Option<NonZeroU32>): convert then wrap.
            bytes_per_row: Some(NonZeroU32::new(padded_bpr).unwrap().into()),
            rows_per_image: Some(NonZeroU32::new(height).unwrap().into()),
        },
    };
    let extent = wgpu::Extent3d { width, height, depth_or_array_layers: 1 };
    encoder.copy_texture_to_buffer(copy_src, copy_dst, extent);
    queue.submit([encoder.finish()]);
    device.poll(wgpu::Maintain::Wait);

    let slice = readback_buf.slice(..);

    // map_async with callback + poll (wgpu in your build)
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |res| { let _ = tx.send(res); });
    device.poll(wgpu::Maintain::Wait);
    rx.recv().expect("map_async channel closed").expect("MapAsync failed");

    let data = slice.get_mapped_range();

    // Unpad each row into a tightly-packed RGBA output
    let mut out = vec![0u8; (row_bytes * height) as usize];
    let src_stride = padded_bpr as usize;
    let dst_stride = row_bytes as usize;
    for y in 0..(height as usize) {
        let src_off = y * src_stride;
        let dst_off = y * dst_stride;
        out[dst_off..dst_off + dst_stride].copy_from_slice(&data[src_off..src_off + dst_stride]);
    }

    drop(data);
    readback_buf.unmap();
    out
}

// ---------- Python class ----------

#[pyclass]
pub struct Renderer {
    width: u32,
    height: u32,
    // persistent GPU resources
    color_tex: wgpu::Texture,
    color_view: wgpu::TextureView,
    readback_buf: wgpu::Buffer,
    readback_size: u64,
    pipeline: wgpu::RenderPipeline,
    vbuf: wgpu::Buffer,
    ibuf: wgpu::Buffer,
    icount: u32,
}

#[pymethods]
impl Renderer {
    #[new]
    pub fn new(width: u32, height: u32) -> Self {
        let ctx = WgpuContext::get();
        let pipeline = create_pipeline(&ctx.device, TEXTURE_FORMAT);
        let (vbuf, ibuf, icount) = triangle_geometry(&ctx.device);
        let (color_tex, color_view) = create_offscreen(&ctx.device, width, height, TEXTURE_FORMAT);
        // small buffer; will grow on demand
        let readback_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback-buffer"),
            size: 4,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        Self {
            width, height,
            color_tex, color_view,
            readback_buf, readback_size: 4,
            pipeline, vbuf, ibuf, icount,
        }
    }

    /// **Added**: matches python/examples/triangle.py
    pub fn info(&self) -> PyResult<String> {
        Ok(format!("Renderer {}x{}, format={:?}", self.width, self.height, TEXTURE_FORMAT))
    }

    /// Returns a (H,W,4) uint8 numpy array of the triangle.
    pub fn render_triangle_rgba<'py>(&mut self, py: Python<'py>) -> PyResult<&'py PyArray3<u8>> {
        let ctx = WgpuContext::get();
        self.render_into_offscreen(ctx)?;

        let need = (align256(self.width * 4) as u64) * (self.height as u64);
        if need > self.readback_size {
            self.readback_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("readback-buffer"),
                size: need,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });
            self.readback_size = need;
        }

        let pixels = copy_texture_to_rgba_unpadded(
            &ctx.device, &ctx.queue, &self.color_tex, &self.readback_buf, self.width, self.height);

        // ndarray -> numpy array (stable path across numpy versions)
        let arr3 = Array3::from_shape_vec(
            (self.height as usize, self.width as usize, 4), pixels
        ).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        Ok(arr3.into_pyarray(py))
    }

    /// Renders and saves a PNG to the given path.
    pub fn render_triangle_png(&mut self, path: String) -> PyResult<()> {
        let ctx = WgpuContext::get();
        self.render_into_offscreen(ctx)?;

        let need = (align256(self.width * 4) as u64) * (self.height as u64);
        if need > self.readback_size {
            self.readback_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("readback-buffer"),
                size: need,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });
            self.readback_size = need;
        }

        let pixels = copy_texture_to_rgba_unpadded(
            &ctx.device, &ctx.queue, &self.color_tex, &self.readback_buf, self.width, self.height);

        let img: ImageBuffer<image::Rgba<u8>, _> = ImageBuffer::from_raw(self.width, self.height, pixels)
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("ImageBuffer::from_raw failed"))?;
        img.save(path).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        Ok(())
    }
}

impl Renderer {
    fn render_into_offscreen(&mut self, ctx: &WgpuContext) -> PyResult<()> {
        // Recreate offscreen if size/format changed
        let size = self.color_tex.size();
        if size.width != self.width || size.height != self.height || self.color_tex.format() != TEXTURE_FORMAT {
            let (tex, view) = create_offscreen(&ctx.device, self.width, self.height, TEXTURE_FORMAT);
            self.color_tex = tex;
            self.color_view = view;
        }

        let mut encoder = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("render-encoder"),
        });
        {
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("triangle-pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.color_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(CLEAR_COLOR),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            rpass.set_viewport(0.0, 0.0, self.width as f32, self.height as f32, 0.0, 1.0);
            rpass.set_scissor_rect(0, 0, self.width, self.height);
            rpass.set_pipeline(&self.pipeline);
            rpass.set_vertex_buffer(0, self.vbuf.slice(..));
            rpass.set_index_buffer(self.ibuf.slice(..), wgpu::IndexFormat::Uint16);
            rpass.draw_indexed(0..self.icount, 0, 0..1);
        }
        ctx.queue.submit([encoder.finish()]);
        Ok(())
    }
}

// ---------- Python module ----------

// IMPORTANT: the module name must be _vulkan_forge to satisfy PyInit__vulkan_forge
#[pymodule]
fn _vulkan_forge(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Renderer>()?;
    Ok(())
}
