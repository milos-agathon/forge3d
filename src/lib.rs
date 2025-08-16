//! Headless renderer for a deterministic triangle with off-screen target + readback.
//! Rust: wgpu 0.19, PyO3 0.21 (abi3). Returns (H,W,4) u8 arrays via numpy.

use std::num::NonZeroU32;

use bytemuck::{Pod, Zeroable};
use image::ImageBuffer;
use once_cell::sync::OnceCell;
use pyo3::prelude::*;
use pyo3::Bound;
use numpy::{PyArray3, IntoPyArray, PyArray2, PyReadonlyArray2, PyArray1};
use numpy::PyUntypedArrayMethods; // needed for contiguous checks
use ndarray::Array3;
use wgpu::util::DeviceExt;
use pyo3::types::{PyDict, PyList};
use pyo3::wrap_pyfunction;

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
                blend: None,
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
            bytes_per_row: Some(NonZeroU32::new(padded_bpr).unwrap().into()),
            rows_per_image: Some(NonZeroU32::new(height).unwrap().into()),
        },
    };
    let extent = wgpu::Extent3d { width, height, depth_or_array_layers: 1 };
    encoder.copy_texture_to_buffer(copy_src, copy_dst, extent);
    queue.submit([encoder.finish()]);
    device.poll(wgpu::Maintain::Wait);

    let slice = readback_buf.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |res| { let _ = tx.send(res); });
    device.poll(wgpu::Maintain::Wait);
    rx.recv().expect("map_async channel closed").expect("MapAsync failed");

    let data = slice.get_mapped_range();

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
    color_tex: wgpu::Texture,
    color_view: wgpu::TextureView,
    readback_buf: wgpu::Buffer,
    readback_size: u64,
    pipeline: wgpu::RenderPipeline,
    vbuf: wgpu::Buffer,
    ibuf: wgpu::Buffer,
    icount: u32,
    terrain: Option<TerrainData>,
    terrain_meta: renderer::TerrainMeta,
    height_tex: Option<wgpu::Texture>,
    height_view: Option<wgpu::TextureView>,
    height_sampler: Option<wgpu::Sampler>,
    // T22-BEGIN:sun-and-exposure
    #[cfg(feature = "terrain_spike")]
    globals: terrain::Globals,
    #[cfg(feature = "terrain_spike")]
    globals_dirty: bool,
    // T22-END:sun-and-exposure
}

#[pymethods]
impl Renderer {
    #[new]
    #[pyo3(text_signature = "(width, height)")]
    /// Create a headless renderer.
    pub fn new(width: u32, height: u32) -> Self {
        let ctx = WgpuContext::get();
        let pipeline = create_pipeline(&ctx.device, TEXTURE_FORMAT);
        let (vbuf, ibuf, icount) = triangle_geometry(&ctx.device);
        let (color_tex, color_view) = create_offscreen(&ctx.device, width, height, TEXTURE_FORMAT);
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
            terrain: None,
            terrain_meta: renderer::TerrainMeta::default(),
            height_tex: None,
            height_view: None,
            height_sampler: None,
            // T22-BEGIN:sun-and-exposure
            #[cfg(feature = "terrain_spike")]
            globals: terrain::Globals::default(),
            #[cfg(feature = "terrain_spike")]
            globals_dirty: true,
            // T22-END:sun-and-exposure
        }
    }

    #[pyo3(text_signature = "($self)")]
    pub fn info(&self) -> PyResult<String> {
        Ok(format!("Renderer {}x{}, format={:?}", self.width, self.height, TEXTURE_FORMAT))
    }

    #[pyo3(text_signature = "($self)")]
    pub fn render_triangle_rgba<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyArray3<u8>>> {
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

        let arr3 = Array3::from_shape_vec(
            (self.height as usize, self.width as usize, 4), pixels
        ).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        Ok(arr3.into_pyarray_bound(py))
    }

    #[pyo3(text_signature = "($self, path)")]
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

    #[pyo3(text_signature = "($self, heightmap, spacing, exaggeration=1.0, *, colormap='viridis')")]
    pub fn add_terrain(
        &mut self,
        heightmap: &PyAny,
        spacing: (f32, f32),
        exaggeration: f32,
        colormap: String,
    ) -> pyo3::PyResult<()> {
        if spacing.0 <= 0.0 || spacing.1 <= 0.0 {
            return Err(pyo3::exceptions::PyRuntimeError::new_err("spacing components must be > 0"));
        }
        if exaggeration <= 0.0 {
            return Err(pyo3::exceptions::PyRuntimeError::new_err("exaggeration must be > 0"));
        }

        let as_f32: Result<(Vec<f32>, usize, usize), pyo3::PyErr> = (|| {
            let arr32: &PyArray2<f32> = heightmap.downcast()?;
            let ro32: PyReadonlyArray2<f32> = arr32.readonly();

            if !ro32.is_c_contiguous() {
                return Err(pyo3::exceptions::PyRuntimeError::new_err("heightmap must be C-contiguous (row-major)"));
            }
            let view = ro32.as_array();
            let (h, w) = (view.shape()[0], view.shape()[1]);
            let mut v = Vec::with_capacity(h * w);
            for val in view.iter() {
                v.push(*val * exaggeration);
            }
            Ok((v, w, h))
        })();

        let (heights, width, height) = match as_f32 {
            Ok(ok) => ok,
            Err(_) => {
                let arr64: &PyArray2<f64> = heightmap
                    .downcast()
                    .map_err(|_| pyo3::exceptions::PyRuntimeError::new_err(
                        "heightmap must be a 2-D NumPy array of dtype float32 or float64"
                    ))?;
                let ro64: PyReadonlyArray2<f64> = arr64.readonly();

                if !ro64.is_c_contiguous() {
                    return Err(pyo3::exceptions::PyRuntimeError::new_err("heightmap must be C-contiguous (row-major)"));
                }
                let view = ro64.as_array();
                let (h, w) = (view.shape()[0], view.shape()[1]);
                let mut v = Vec::with_capacity(h * w);
                for val in view.iter() {
                    v.push((*val as f32) * exaggeration);
                }
                (v, w, h)
            }
        };

        if width == 0 || height == 0 {
            return Err(pyo3::exceptions::PyRuntimeError::new_err("heightmap cannot be empty"));
        }

        // Compute height range for colormap normalization using terrain_stats
        // T02-BEGIN:invoke-dem-range
        self.terrain_meta.compute_and_store_h_range(&heights);
        // T02-END:invoke-dem-range
        
        // Validate colormap parameter via central SUPPORTED list
        // T33-BEGIN:colormap-validation
        {
            use crate::colormap::SUPPORTED;
            if !SUPPORTED.contains(&colormap.as_str()) {
                return Err(pyo3::exceptions::PyRuntimeError::new_err(
                    format!("Unknown colormap '{}'. Supported: {}", colormap, SUPPORTED.join(", "))
                ));
            }
        }
        // T33-END:colormap-validation

        self.terrain = Some(TerrainData {
            width: width as u32,
            height: height as u32,
            spacing,
            exaggeration,
            colormap,
            heights,
        });

        Ok(())
    }

    #[pyo3(text_signature = "($self)")]
    pub fn terrain_stats(&self) -> pyo3::PyResult<(f32, f32, f32, f32)> {
        let terr = self.terrain.as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("no terrain uploaded; call add_terrain() first"))?;
        let stats = dem_stats_from_slice(&terr.heights);
        Ok((stats.min, stats.max, stats.mean, stats.std))
    }

    /// Override the height normalization range used for color & lighting.
    /// Raises `ValueError` if `min >= max`.
    // T02-BEGIN:set-height-range-python
    #[pyo3(text_signature = "($self, min, max)")]
    pub fn set_height_range(&mut self, min: f32, max: f32) -> pyo3::PyResult<()> {
        self.terrain_meta.set_height_range(min, max)
    }
    // T02-END:set-height-range-python

    // T22-BEGIN:sun-and-exposure
    #[cfg(feature = "terrain_spike")]
    /// Set sun by spherical angles (degrees).
    /// Basis: Y-up, right-handed; azimuth=0° along +X (CCW toward +Z), elevation=0° on horizon.
    fn set_sun_dir_spherical(&mut self, elevation_deg: f32, azimuth_deg: f32) {
        #[inline] fn deg(x: f32) -> f32 { x * std::f32::consts::PI / 180.0 }
        let el = deg(elevation_deg);
        let az = deg(azimuth_deg);
        let (se, ce) = (el.sin(), el.cos());
        let (sa, ca) = (az.sin(), az.cos());
        let dir = glam::Vec3::new(ce * ca, se, ce * sa).normalize_or_zero();
        self.globals.sun_dir = dir;
        self.globals_dirty = true;
    }

    #[cfg(feature = "terrain_spike")]
    #[pyo3(text_signature = "($self, elevation_deg, azimuth_deg)")]
    fn set_sun(&mut self, elevation_deg: f32, azimuth_deg: f32) -> PyResult<()> {
        if !elevation_deg.is_finite() || !azimuth_deg.is_finite() {
            return Err(pyo3::exceptions::PyValueError::new_err("angles must be finite"));
        }
        self.set_sun_dir_spherical(elevation_deg, azimuth_deg);
        Ok(())
    }

    #[cfg(feature = "terrain_spike")]
    #[pyo3(text_signature = "($self, exposure)")]
    fn set_exposure(&mut self, exposure: f32) -> PyResult<()> {
        if !exposure.is_finite() || exposure <= 0.0 {
            return Err(pyo3::exceptions::PyValueError::new_err("exposure must be > 0"));
        }
        self.globals.exposure = exposure;
        self.globals_dirty = true;
        Ok(())
    }
    // T22-END:sun-and-exposure

    #[pyo3(text_signature = "($self, mode, range=None, eps=1e-8)")]
    pub fn normalize_terrain(&mut self, mode: &str, range: Option<(f32, f32)>, eps: Option<f32>) -> pyo3::PyResult<()> {
        let terr = self.terrain.as_mut()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("no terrain uploaded; call add_terrain() first"))?;

        let mode = match mode.to_lowercase().as_str() {
            "minmax" => NormalizeMode::MinMax,
            "zscore" => NormalizeMode::ZScore,
            _ => return Err(pyo3::exceptions::PyRuntimeError::new_err("mode must be 'minmax' or 'zscore'")),
        };
        let eps = eps.unwrap_or(1e-8_f32);
        let range = range.unwrap_or((0.0, 1.0));

        let stats = dem_stats_from_slice(&terr.heights);
        normalize_in_place(&mut terr.heights, mode, eps, range, &stats);
        Ok(())
    }

    #[pyo3(text_signature = "($self)")]
    pub fn upload_height_r32f(&mut self) -> pyo3::PyResult<()> {
        let ctx = WgpuContext::get();

        let terr = self.terrain.as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("no terrain uploaded; call add_terrain() first"))?;

        let width = terr.width;
        let height = terr.height;
        if width == 0 || height == 0 {
            return Err(pyo3::exceptions::PyRuntimeError::new_err("terrain dimensions are zero"));
        }

        let tex = ctx.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("terrain-height-r32f"),
            size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R32Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
        let samp = ctx.device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("terrain-height-sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        // Build temporary padded upload buffer for 256-byte row alignment
        let row_bytes = width * 4;
        let padded_bpr = ((row_bytes + 255) / 256) * 256;
        
        // Create padded buffer
        let padded_data = {
            let mut data = vec![0u8; (padded_bpr * height) as usize];
            let input_data = bytemuck::cast_slice::<f32, u8>(&terr.heights);
            
            for y in 0..height {
                let src_offset = (y * row_bytes) as usize;
                let dst_offset = (y * padded_bpr) as usize;
                let src_end = src_offset + row_bytes as usize;
                let dst_end = dst_offset + row_bytes as usize;
                
                data[dst_offset..dst_end].copy_from_slice(&input_data[src_offset..src_end]);
            }
            data
        };

        ctx.queue.write_texture(
            wgpu::ImageCopyTexture {
                texture: &tex,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &padded_data,
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(NonZeroU32::new(padded_bpr).unwrap().into()),
                rows_per_image: Some(NonZeroU32::new(height).unwrap().into()),
            },
            wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
        );
        ctx.device.poll(wgpu::Maintain::Wait);

        self.height_tex = Some(tex);
        self.height_view = Some(view);
        self.height_sampler = Some(samp);
        Ok(())
    }

    #[pyo3(text_signature = "($self, x, y, w, h)")]
    pub fn debug_read_height_patch<'py>(&mut self, py: Python<'py>, x: u32, y: u32, w: u32, h: u32)
        -> pyo3::PyResult<Bound<'py, PyArray2<f32>>> {
        if w == 0 || h == 0 {
            return Err(pyo3::exceptions::PyRuntimeError::new_err("patch dimensions must be > 0"));
        }

        if self.height_tex.is_none() {
            let zeros = vec![0f32; (w * h) as usize];
            let rows: Vec<Vec<f32>> = zeros
                .chunks_exact(w as usize)
                .map(|row| row.to_vec())
                .collect();
            let out = numpy::PyArray2::<f32>::from_vec2_bound(py, &rows)?;
            return Ok(out);
        }

        let ctx = WgpuContext::get();
        let tex = self.height_tex.as_ref().unwrap();
        let tex_size = tex.size();
        if x + w > tex_size.width {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                "requested patch exceeds texture bounds in x: x+w ({}) > width ({})",
                x + w, tex_size.width
            )));
        }
        if y + h > tex_size.height {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                "requested patch exceeds texture bounds in y: y+h ({}) > height ({})",
                y + h, tex_size.height
            )));
        }

        let row_bytes = (w * 4) as u32;
        let padded_bpr = ((row_bytes + 255) / 256) * 256;
        let buf_size = padded_bpr as u64 * h as u64;
        let readback = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("height-patch-readback"),
            size: buf_size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let mut encoder = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("height-patch-encoder"),
        });

        encoder.copy_texture_to_buffer(
            wgpu::ImageCopyTexture {
                texture: tex,
                mip_level: 0,
                origin: wgpu::Origin3d { x, y, z: 0 },
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::ImageCopyBuffer {
                buffer: &readback,
                layout: wgpu::ImageDataLayout {
                    offset: 0,
                    bytes_per_row: Some(NonZeroU32::new(padded_bpr).unwrap().into()),
                    rows_per_image: Some(NonZeroU32::new(h).unwrap().into()),
                },
            },
            wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        );

        ctx.queue.submit([encoder.finish()]);
        ctx.device.poll(wgpu::Maintain::Wait);

        let slice = readback.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |res| { let _ = tx.send(res); });
        ctx.device.poll(wgpu::Maintain::Wait);
        rx.recv().expect("map_async channel closed").expect("MapAsync failed");
        let data = slice.get_mapped_range();

        let mut out = vec![0u8; (row_bytes * h) as usize];
        let src_stride = padded_bpr as usize;
        let dst_stride = row_bytes as usize;
        for j in 0..(h as usize) {
            let s = j * src_stride;
            let d = j * dst_stride;
            out[d..d + dst_stride].copy_from_slice(&data[s..s + dst_stride]);
        }
        drop(data);
        readback.unmap();

        let floats: &[f32] = bytemuck::cast_slice(&out);
        let rows: Vec<Vec<f32>> = floats
            .chunks_exact(w as usize)
            .map(|row| row.to_vec())
            .collect();
        let arr = numpy::PyArray2::<f32>::from_vec2_bound(py, &rows)?;
        Ok(arr)
    }

    #[pyo3(text_signature = "($self)")]
    pub fn read_full_height_texture<'py>(&mut self, py: Python<'py>)
        -> pyo3::PyResult<Bound<'py, PyArray2<f32>>> {
        let terr = self.terrain.as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("no terrain uploaded; call add_terrain() first"))?;

        if self.height_tex.is_none() {
            return Err(pyo3::exceptions::PyRuntimeError::new_err("no height texture uploaded; call upload_height_r32f() first"));
        }

        let width = terr.width;
        let height = terr.height;
        self.debug_read_height_patch(py, 0, 0, width, height)
    }
}

impl Renderer {
    fn render_into_offscreen(&mut self, ctx: &WgpuContext) -> PyResult<()> {
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

fn backend_str(b: wgpu::Backend) -> &'static str {
    match b {
        wgpu::Backend::Vulkan => "VULKAN",
        wgpu::Backend::Dx12 => "DX12",
        wgpu::Backend::Metal => "METAL",
        wgpu::Backend::Gl => "GL",
        wgpu::Backend::BrowserWebGpu => "WEBGPU",
        _ => "UNKNOWN",
    }
}
fn devtype_str(t: wgpu::DeviceType) -> &'static str {
    match t {
        wgpu::DeviceType::Other => "Other",
        wgpu::DeviceType::IntegratedGpu => "IntegratedGpu",
        wgpu::DeviceType::DiscreteGpu => "DiscreteGpu",
        wgpu::DeviceType::VirtualGpu => "VirtualGpu",
        wgpu::DeviceType::Cpu => "Cpu",
    }
}

#[pyfunction]
#[pyo3(text_signature = "()")]
fn enumerate_adapters(py: Python<'_>) -> PyResult<PyObject> {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::all(),
        ..Default::default()
    });

    let mut adapters: Vec<wgpu::Adapter> = instance.enumerate_adapters(wgpu::Backends::all());
    if adapters.is_empty() {
        if let Some(adapter) = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        })) {
            adapters.push(adapter);
        }
    }

    let out = PyList::empty_bound(py);
    for ad in adapters {
        let info = ad.get_info();
        let d = PyDict::new_bound(py);
        d.set_item("name", info.name).ok();
        d.set_item("backend", backend_str(info.backend)).ok();
        d.set_item("device_type", devtype_str(info.device_type)).ok();
        d.set_item("vendor_id", info.vendor as u32).ok();
        d.set_item("device_id", info.device as u32).ok();
        d.set_item("features", format!("{:?}", ad.features())).ok();
        d.set_item("limits", format!("{:?}", ad.limits())).ok();
        out.append(d).ok();
    }
    Ok(out.into_any().unbind())
}

#[pyfunction]
#[pyo3(text_signature = "(backend=None)")]
fn device_probe(py: Python<'_>, backend: Option<String>) -> PyResult<PyObject> {
    use std::time::Instant;
    let b = backend.unwrap_or_else(|| "AUTO".to_string()).to_uppercase();

    let backends = match b.as_str() {
        "VULKAN" => wgpu::Backends::VULKAN,
        "DX12"   => wgpu::Backends::DX12,
        "METAL"  => wgpu::Backends::METAL,
        "GL"     => wgpu::Backends::GL,
        "AUTO" | _ => wgpu::Backends::all(),
    };

    let inst = wgpu::Instance::new(wgpu::InstanceDescriptor { backends, ..Default::default() });

    let dict = PyDict::new_bound(py);
    dict.set_item("backend_request", b.clone()).ok();

    let t0 = Instant::now();
    let adapter = pollster::block_on(inst.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
    }));
    let adapter = match adapter {
        Some(a) => a,
        None => {
            dict.set_item("status", "unsupported").ok();
            dict.set_item("message", "No suitable GPU adapter").ok();
            dict.set_item("millis", t0.elapsed().as_secs_f64() * 1000.0).ok();
            return Ok(dict.into_any().unbind());
        }
    };

    let info = adapter.get_info();
    dict.set_item("adapter_name", info.name.clone()).ok();
    dict.set_item("backend", backend_str(info.backend)).ok();
    dict.set_item("device_type", devtype_str(info.device_type)).ok();
    dict.set_item("vendor_id", info.vendor as u32).ok();
    dict.set_item("device_id", info.device as u32).ok();
    dict.set_item("features", format!("{:?}", adapter.features())).ok();
    dict.set_item("limits", format!("{:?}", adapter.limits())).ok();

    let needed_features = wgpu::Features::empty();
    let limits = wgpu::Limits::downlevel_defaults();
    let (_device, _queue) = match pollster::block_on(adapter.request_device(
        &wgpu::DeviceDescriptor{
            required_features: needed_features,
            required_limits: limits,
            label: Some("diag-device"),
        },
        None,
    )) {
        Ok(pair) => pair,
        Err(e) => {
            dict.set_item("status", "error").ok();
            dict.set_item("message", format!("request_device failed: {}", e)).ok();
            dict.set_item("millis", t0.elapsed().as_secs_f64() * 1000.0).ok();
            return Ok(dict.into_any().unbind());
        }
    };

    dict.set_item("status", "ok").ok();
    dict.set_item("millis", t0.elapsed().as_secs_f64() * 1000.0).ok();
    Ok(dict.into_any().unbind())
}

// ---------- Python module ----------

// T33-BEGIN:lib-terrain-mod
pub mod terrain;
pub use terrain::pipeline::TerrainPipeline;
// T33-END:lib-terrain-mod
// T33-BEGIN:colormap-registry
pub mod colormap;
// T33-END:colormap-registry
pub mod camera;

// T2.1 Infrastructure re-exports for easy access
#[cfg(feature = "terrain_spike")]
pub use terrain::{TerrainUniforms, Globals};
// (re-exporting camera_utils/verify_t21_infrastructure is optional; omitted to avoid cfg/name drift)

// mod grid; // T11: disabled - grid_generate moved to terrain::mesh
mod terrain_stats;
mod renderer;

#[derive(Clone)]
struct TerrainData {
    width:  u32,
    height: u32,
    spacing: (f32, f32),
    exaggeration: f32,
    colormap: String,
    /// Row-major, length = width*height, units = heightmap * exaggeration
    heights: Vec<f32>,
}

#[derive(Debug, Clone)]
struct DemStats {
    min: f32,
    max: f32,
    mean: f32,
    std: f32,
}

#[derive(Debug, Clone)]
enum NormalizeMode {
    MinMax,
    ZScore,
}

impl NormalizeMode {
    fn from_str(s: &str) -> Result<Self, String> {
        match s.to_lowercase().as_str() {
            "minmax" => Ok(NormalizeMode::MinMax),
            "zscore" => Ok(NormalizeMode::ZScore),
            _ => Err(format!("Unknown normalization mode: {}", s)),
        }
    }
}

fn dem_stats_from_slice(heights: &[f32]) -> DemStats {
    if heights.is_empty() {
        return DemStats { min: 0.0, max: 0.0, mean: 0.0, std: 0.0 };
    }

    let mut min = heights[0];
    let mut max = heights[0];
    let mut sum = 0.0;

    for &h in heights {
        if h < min { min = h; }
        if h > max { max = h; }
        sum += h;
    }

    let mean = sum / heights.len() as f32;

    let mut variance_sum = 0.0;
    for &h in heights {
        let diff = h - mean;
        variance_sum += diff * diff;
    }

    let variance = variance_sum / heights.len() as f32;
    let std = variance.sqrt();

    DemStats { min, max, mean, std }
}

fn normalize_in_place(heights: &mut [f32], mode: NormalizeMode, eps: f32, range: (f32, f32), stats: &DemStats) {
    match mode {
        NormalizeMode::MinMax => {
            let (lo, hi) = range;
            let denom = (stats.max - stats.min).abs().max(eps);
            let scale = (hi - lo) / denom;
            for v in heights.iter_mut() {
                *v = (*v - stats.min) * scale + lo;
            }
        }
        NormalizeMode::ZScore => {
            let denom = stats.std.max(eps);
            for v in heights.iter_mut() {
                *v = (*v - stats.mean) / denom;
            }
        }
    }
}

#[pyfunction]
#[pyo3(text_signature = "(nx, nz, spacing=(1.0,1.0), origin='center')")]
fn grid_generate(py: Python<'_>, nx: u32, nz: u32, spacing: (f32, f32), origin: Option<String>)
    -> PyResult<(Bound<'_, PyArray2<f32>>, Bound<'_, PyArray2<f32>>, Bound<'_, PyArray1<u32>>)>
{
    terrain::mesh::grid_generate(py, nx, nz, spacing, origin)
}

#[allow(deprecated)]
#[pymodule]
fn _vulkan_forge(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_class::<Renderer>()?;
    #[cfg(feature = "terrain_spike")]
    { m.add_class::<terrain::TerrainSpike>()?; }
    m.add_function(wrap_pyfunction!(enumerate_adapters, m)?)?;
    m.add_function(wrap_pyfunction!(device_probe, m)?)?;
    m.add_function(wrap_pyfunction!(grid_generate, m)?)?;
    m.add_function(wrap_pyfunction!(colormap::colormap_supported, m)?)?;
    m.add_function(wrap_pyfunction!(camera::camera_look_at, m)?)?;
    m.add_function(wrap_pyfunction!(camera::camera_perspective, m)?)?;
    m.add_function(wrap_pyfunction!(camera::camera_view_proj, m)?)?;
    Ok(())
}
