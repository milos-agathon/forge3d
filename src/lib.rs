use std::num::NonZeroU32;
use bytemuck::{Pod, Zeroable};
use image::{ImageBuffer, Rgba};
use once_cell::sync::OnceCell;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use numpy::{PyArray3, IntoPyArray};
use ndarray::Array3;
use thiserror::Error;
use wgpu::util::DeviceExt;

static WGPU_CTX: OnceCell<WgpuContext> = OnceCell::new();

#[derive(Error, Debug)]
enum VsError {
    #[error("wgpu error: {0}")]
    Wgpu(String),
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("image error: {0}")]
    Image(#[from] image::ImageError),
}

impl From<VsError> for PyErr {
    fn from(e: VsError) -> Self {
        PyRuntimeError::new_err(e.to_string())
    }
}

struct WgpuContext {
    instance: wgpu::Instance,
    adapter: wgpu::Adapter,
    device: wgpu::Device,
    queue: wgpu::Queue,
}

impl WgpuContext {
    fn new() -> Result<Self, VsError> {
        let backends = wgpu::Backends::all();
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends,
            ..Default::default()
        });
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        })).ok_or_else(|| VsError::Wgpu("No suitable GPU adapter found".into()))?;

        let needed_features = wgpu::Features::empty();
        let limits = wgpu::Limits::downlevel_defaults();

        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                required_features: needed_features,
                required_limits: limits,
                label: Some("vulkan-forge-device"),
            },
            None,
        )).map_err(|e| VsError::Wgpu(format!("request_device failed: {e}")))?;

        Ok(Self { instance, adapter, device, queue })
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Vertex {
    pos: [f32; 2],
    color: [f32; 3],
}

fn triangle_vertices() -> (wgpu::Buffer, wgpu::Buffer, u32) {
    let verts: &[Vertex] = &[
        Vertex { pos: [-0.8, -0.8], color: [1.0, 0.2, 0.2] },
        Vertex { pos: [ 0.8, -0.8], color: [0.2, 1.0, 0.2] },
        Vertex { pos: [ 0.0,  0.8], color: [0.2, 0.2, 1.0] },
    ];
    let indices: &[u16] = &[0, 1, 2];

    let ctx = WGPU_CTX.get().expect("WGPU context not initialized");
    let vbuf = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("triangle-vertices"),
        contents: bytemuck::cast_slice(verts),
        usage: wgpu::BufferUsages::VERTEX,
    });
    let ibuf = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("triangle-indices"),
        contents: bytemuck::cast_slice(indices),
        usage: wgpu::BufferUsages::INDEX,
    });
    (vbuf, ibuf, indices.len() as u32)
}

fn create_pipeline(format: wgpu::TextureFormat) -> wgpu::RenderPipeline {
    let ctx = WGPU_CTX.get().expect("WGPU context not initialized");
    let shader = ctx.device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("triangle-shader"),
        source: wgpu::ShaderSource::Wgsl(include_str!("shaders/triangle.wgsl").into()),
    });

    let layout = ctx.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("triangle-pipeline-layout"),
        bind_group_layouts: &[],
        push_constant_ranges: &[],
    });

    ctx.device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
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
                blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        multiview: None,
    })
}

fn create_target_texture(width: u32, height: u32, format: wgpu::TextureFormat) -> (wgpu::Texture, wgpu::TextureView) {
    let ctx = WGPU_CTX.get().expect("WGPU context not initialized");
    let texture = ctx.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("offscreen-target"),
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

fn read_texture_to_rgba(width: u32, height: u32, texture: &wgpu::Texture) -> Result<Vec<u8>, VsError> {
    let ctx = WGPU_CTX.get().expect("WGPU context not initialized");
    let bytes_per_pixel = 4u32;
    let unpadded_bytes_per_row = width * bytes_per_pixel;
    let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
    let padded_bytes_per_row = ((unpadded_bytes_per_row + align - 1) / align) * align;
    let output_buffer_size = (padded_bytes_per_row * height) as u64;

    let output = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("readback-buffer"),
        size: output_buffer_size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let mut encoder = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("copy-encoder"),
    });

    encoder.copy_texture_to_buffer(
        wgpu::ImageCopyTexture {
            texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::ImageCopyBuffer {
            buffer: &output,
            layout: wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(NonZeroU32::new(padded_bytes_per_row).unwrap().into()),
                rows_per_image: Some(NonZeroU32::new(height).unwrap().into()),
            },
        },
        wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
    );

    ctx.queue.submit(Some(encoder.finish()));

    // Map buffer and copy out, removing row padding
    let buffer_slice = output.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    buffer_slice.map_async(wgpu::MapMode::Read, move |res| { tx.send(res).ok(); });
    ctx.device.poll(wgpu::Maintain::Wait);
    rx.recv().unwrap().map_err(|e| VsError::Wgpu(format!("map_async failed: {e:?}")))?;

    let data = buffer_slice.get_mapped_range();
    let mut pixels = Vec::with_capacity((width * height * 4) as usize);
    for chunk in data.chunks(padded_bytes_per_row as usize) {
        // Rust uses `..` for slicing (not `:`)
        pixels.extend_from_slice(&chunk[..unpadded_bytes_per_row as usize]);
    }
    drop(data);
    output.unmap();
    Ok(pixels)
}

#[pyclass]
pub struct Renderer {
    width: u32,
    height: u32,
    clear: [f64; 4],
    format: wgpu::TextureFormat,
}

#[pymethods]
impl Renderer {
    #[new]
    pub fn new(width: u32, height: u32) -> PyResult<Self> {
        WGPU_CTX.get_or_try_init(|| WgpuContext::new())?;
        Ok(Self {
            width,
            height,
            clear: [1.0, 1.0, 1.0, 1.0],
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
        })
    }

    pub fn set_clear(&mut self, r: f64, g: f64, b: f64, a: f64) { self.clear = [r, g, b, a]; }

    pub fn info(&self) -> PyResult<String> {
        let ctx = WGPU_CTX.get().unwrap();
        let ad = ctx.adapter.get_info();
        Ok(format!("Adapter: {} ({:?}), backend: {:?}", ad.name, ad.device_type, ad.backend))
    }

    pub fn render_triangle_rgba<'py>(&self, py: Python<'py>) -> PyResult<&'py PyArray3<u8>> {
        let ctx = WGPU_CTX.get().unwrap();
        let (texture, view) = create_target_texture(self.width, self.height, self.format);
        let pipeline = create_pipeline(self.format);
        let (vbuf, ibuf, icount) = triangle_vertices();

        let mut encoder = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("render-encoder") });
        {
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("triangle-pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view, resolve_target: None,
                    ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color { r: self.clear[0], g: self.clear[1], b: self.clear[2], a: self.clear[3] }), store: wgpu::StoreOp::Store },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            rpass.set_pipeline(&pipeline);
            rpass.set_vertex_buffer(0, vbuf.slice(..));
            rpass.set_index_buffer(ibuf.slice(..), wgpu::IndexFormat::Uint16);
            rpass.draw_indexed(0..icount, 0, 0..1);
        }
        ctx.queue.submit(Some(encoder.finish()));

        let pixels = read_texture_to_rgba(self.width, self.height, &texture)?;
        // numpy 0.21: create via ndarray then convert
        let arr = Array3::from_shape_vec(
            (self.height as usize, self.width as usize, 4),
            pixels
        ).expect("failed to create ndarray")
         .into_pyarray(py);
        Ok(arr)
    }

    pub fn render_triangle_png(&self, path: &str) -> PyResult<()> {
        let ctx = WGPU_CTX.get().unwrap();
        let (texture, view) = create_target_texture(self.width, self.height, self.format);
        let pipeline = create_pipeline(self.format);
        let (vbuf, ibuf, icount) = triangle_vertices();

        let mut encoder = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("render-encoder") });
        {
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("triangle-pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view, resolve_target: None,
                    ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color { r: self.clear[0], g: self.clear[1], b: self.clear[2], a: self.clear[3] }), store: wgpu::StoreOp::Store },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            rpass.set_pipeline(&pipeline);
            rpass.set_vertex_buffer(0, vbuf.slice(..));
            rpass.set_index_buffer(ibuf.slice(..), wgpu::IndexFormat::Uint16);
            rpass.draw_indexed(0..icount, 0, 0..1);
        }
        ctx.queue.submit(Some(encoder.finish()));

        let pixels = read_texture_to_rgba(self.width, self.height, &texture)?;
        let img: ImageBuffer<Rgba<u8>, _> = ImageBuffer::from_raw(self.width, self.height, pixels)
            .ok_or_else(|| VsError::Wgpu("Failed to construct image buffer".into()))?;
        use pyo3::exceptions::PyIOError;
        img.save(path).map_err(|e| PyIOError::new_err(e.to_string()))?;
        Ok(())
    }
}

#[pymodule]
fn _vulkan_forge(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Renderer>()?;
    Ok(())
}
