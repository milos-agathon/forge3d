// A2-BEGIN:terrain-module
#![allow(dead_code)]

// T11-BEGIN:terrain-mesh-mod
pub mod mesh;
pub use mesh::{GridMesh, GridVertex, Indices, make_grid};
// T11-END:terrain-mesh-mod

use pyo3::prelude::*;
use std::num::NonZeroU32;
use wgpu::util::DeviceExt;

// T33-BEGIN:colormap-imports
use crate::colormap::{resolve_bytes, SUPPORTED, ColormapType, map_name_to_type};
// T33-END:colormap-imports

// ---------- Colormaps ----------

/// Helper function to get palette data from colormap type
fn get_palette_data_for_type(colormap_type: ColormapType) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    let name = match colormap_type {
        ColormapType::Viridis => "viridis",
        ColormapType::Magma => "magma",
        ColormapType::Terrain => "terrain",
    };
    let png = resolve_bytes(name).map_err(|_| "unknown colormap")?;
    let img = image::load_from_memory(png)?;
    let rgba = img.to_rgba8();
    Ok(rgba.as_raw().clone())
}

/// Convert sRGB bytes to linear for UNORM fallback
fn srgb_to_linear_byte(b: u8) -> u8 {
    let c = (b as f32) / 255.0;
    let l = if c <= 0.04045 { c / 12.92 } else { ((c + 0.055) / 1.055).powf(2.4) };
    (l.clamp(0.0, 1.0) * 255.0 + 0.5) as u8
}

/// Convert sRGB palette to linear (in-place) for UNORM fallback
fn convert_palette_srgb_to_linear(palette: &mut [u8]) {
    // Process RGBA data, convert RGB channels only (skip alpha)
    for chunk in palette.chunks_exact_mut(4) {
        chunk[0] = srgb_to_linear_byte(chunk[0]); // R
        chunk[1] = srgb_to_linear_byte(chunk[1]); // G 
        chunk[2] = srgb_to_linear_byte(chunk[2]); // B
        // Skip alpha channel[3]
    }
}

pub struct ColormapLUT {
    pub texture: wgpu::Texture,
    pub view: wgpu::TextureView,
    pub sampler: wgpu::Sampler,
    pub format: wgpu::TextureFormat,
}

impl ColormapLUT {
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        which: ColormapType,
        prefer_srgb: bool,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let mut palette = get_palette_data_for_type(which)?;
        
        let (format, needs_linearization) = if prefer_srgb {
            (wgpu::TextureFormat::Rgba8UnormSrgb, false)
        } else {
            (wgpu::TextureFormat::Rgba8Unorm, true)
        };
        
        // Convert palette from sRGB to linear if using UNORM fallback
        if needs_linearization {
            convert_palette_srgb_to_linear(&mut palette);
        }
        
        // 256×1 RGBA8
        let tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("colormap-lut"),
            size: wgpu::Extent3d {
                width: 256,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        queue.write_texture(
            wgpu::ImageCopyTexture {
                texture: &tex,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &palette,
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(NonZeroU32::new(256 * 4).unwrap().into()),
                rows_per_image: Some(NonZeroU32::new(1).unwrap().into()),
            },
            wgpu::Extent3d {
                width: 256,
                height: 1,
                depth_or_array_layers: 1,
            },
        );
        let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("colormap-lut-sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });
        Ok(Self { texture: tex, view, sampler, format })
    }
}

// ---------- Uniforms (std140-compatible, 176 bytes) ----------

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct TerrainUniforms {
    pub view: [[f32; 4]; 4],           // 64 B
    pub proj: [[f32; 4]; 4],           // 64 B
    pub sun_exposure: [f32; 4],        // (sun_dir.xyz, exposure) -> 16 B
    pub spacing_h_exag_pad: [f32; 4],  // (spacing, h_range, exaggeration, 0) -> 16 B
    pub _pad_tail: [f32; 4],           // final 16 B lane to satisfy 176 total
}

impl TerrainUniforms {
    /// Note: `h_range = max - min` (pass a single range, not min/max separately)
    pub fn new(
        view: glam::Mat4,
        proj: glam::Mat4,
        sun_dir: glam::Vec3,
        exposure: f32,
        spacing: f32,
        h_range: f32,
        exaggeration: f32,
    ) -> Self {
        Self {
            view: view.to_cols_array_2d(),
            proj: proj.to_cols_array_2d(),
            sun_exposure: [sun_dir.x, sun_dir.y, sun_dir.z, exposure],
            spacing_h_exag_pad: [spacing, h_range, exaggeration, 0.0],
            _pad_tail: [0.0; 4],
        }
    }

    pub fn from_mvp_legacy(
        mvp: glam::Mat4,
        light: glam::Vec3,
        h_min: f32,
        h_max: f32,
        exaggeration: f32,
    ) -> Self {
        let view = glam::Mat4::IDENTITY;
        let h_range = h_max - h_min;
        Self::new(view, mvp, light, 1.0, 1.0, h_range, exaggeration)
    }

    pub fn for_rendering(
        view_matrix: glam::Mat4,
        proj_matrix: glam::Mat4,
        sun_direction: glam::Vec3,
        exposure: f32,
        terrain_spacing: f32,
        height_range: f32,
        height_exaggeration: f32,
    ) -> Self {
        Self::new(
            view_matrix,
            proj_matrix,
            sun_direction,
            exposure,
            terrain_spacing,
            height_range,
            height_exaggeration,
        )
    }
}

// T2.1 Global state
#[derive(Debug, Clone)]
pub struct Globals {
    pub sun_dir: glam::Vec3,
    pub exposure: f32,
    pub spacing: f32,
    pub h_min: f32,
    pub h_max: f32,
    pub exaggeration: f32,
}

impl Default for Globals {
    fn default() -> Self {
        Self {
            sun_dir: glam::Vec3::new(0.5, 0.8, 0.6).normalize(),
            exposure: 1.0,
            spacing: 1.0,
            // choose a sane range matching our analytic spike heights (~±0.5)
            h_min: -0.5,
            h_max: 0.5,
            exaggeration: 1.0,
        }
    }
}

impl Globals {
    pub fn to_uniforms(&self, view: glam::Mat4, proj: glam::Mat4) -> TerrainUniforms {
        let h_range = self.h_max - self.h_min;
        TerrainUniforms::new(
            view,
            proj,
            self.sun_dir,
            self.exposure,
            self.spacing,
            h_range,
            self.exaggeration,
        )
    }
}

// ---------- Render spike object used by tests ----------

const TEXTURE_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8UnormSrgb;
const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

#[pyclass(module = "_vulkan_forge", name = "TerrainSpike")]
pub struct TerrainSpike {
    width: u32,
    height: u32,
    grid: u32,

    device: wgpu::Device,
    queue: wgpu::Queue,

    pipeline: wgpu::RenderPipeline,
    vbuf: wgpu::Buffer,
    ibuf: wgpu::Buffer,
    nidx: u32,

    ubo: wgpu::Buffer,
    ubo_bind_group: wgpu::BindGroup,
    colormap_lut: ColormapLUT,

    color: wgpu::Texture,
    color_view: wgpu::TextureView,
    depth: wgpu::Texture,
    depth_view: wgpu::TextureView,

    globals: Globals,
}

#[pymethods]
impl TerrainSpike {
    #[new]
    #[pyo3(text_signature = "(width, height, grid=128, colormap='viridis')")]
    pub fn new(width: u32, height: u32, grid: Option<u32>, colormap: Option<String>) -> PyResult<Self> {
        let grid = grid.unwrap_or(128).max(2);

        let colormap_name = colormap.as_deref().unwrap_or("viridis");
        
        // Validate colormap against central SUPPORTED list
        if !SUPPORTED.contains(&colormap_name) {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                format!("Unknown colormap '{}'. Supported: {}", colormap_name, SUPPORTED.join(", "))
            ));
        }
        
        let which = map_name_to_type(colormap_name)
            .map_err(|_| pyo3::exceptions::PyRuntimeError::new_err(
                format!("Unknown colormap '{}'. Supported: {}", colormap_name, SUPPORTED.join(", "))
            ))?;

        // Instance/adapter/device
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        })).ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("No suitable GPU adapter"))?;

        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor{
                label: Some("terrain-device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::downlevel_defaults(),
            },
            None,
        )).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

        // R2: Check device texture format support
        let mut prefer_srgb = adapter.get_texture_format_features(wgpu::TextureFormat::Rgba8UnormSrgb)
            .allowed_usages
            .contains(wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST);
        
        // Honor env toggle for CI
        if std::env::var("VF_FORCE_LUT_UNORM") == Ok("1".into()) {
            prefer_srgb = false;
        }

        // Offscreen color + depth
        let color = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("terrain-color"),
            size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: TEXTURE_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let color_view = color.create_view(&Default::default());

        let depth = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("terrain-depth"),
            size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: DEPTH_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let depth_view = depth.create_view(&Default::default());

        // Shader + pipeline
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("terrain.wgsl"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/terrain.wgsl").into()),
        });

        let ubo_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor{
            label: Some("ubo-layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry{
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer{
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None, // wgpu will validate vs shader (expects 176)
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry{
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture{
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry{
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor{
            label: Some("terrain-pipeline-layout"),
            bind_group_layouts: &[&ubo_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("terrain-pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: (4 * 3 + 4 * 3) as wgpu::BufferAddress, // pos(vec3) + normal(vec3)
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &[
                        wgpu::VertexAttribute { shader_location: 0, offset: 0,                     format: wgpu::VertexFormat::Float32x3 },
                        wgpu::VertexAttribute { shader_location: 1, offset: (4 * 3) as u64,        format: wgpu::VertexFormat::Float32x3 },
                    ],
                }],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format: TEXTURE_FORMAT,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                cull_mode: Some(wgpu::Face::Back),
                front_face: wgpu::FrontFace::Ccw,
                polygon_mode: wgpu::PolygonMode::Fill,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: DEPTH_FORMAT,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
        });

        // Mesh + uniforms
        let (vbuf, ibuf, nidx) = build_grid_mesh(&device, grid);
        let (view, proj, light) = build_view_matrices(width, height);

        let mut globals = Globals::default();
        // R4: Seed globals.sun_dir from computed light
        globals.sun_dir = light;
        // Use globals (with h_min/h_max) -> h_range is computed inside to_uniforms()
        let uniforms = globals.to_uniforms(view, proj);

        let ubo = device.create_buffer_init(&wgpu::util::BufferInitDescriptor{
            label: Some("terrain-ubo"),
            contents: bytemuck::cast_slice(&[uniforms]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let lut = ColormapLUT::new(&device, &queue, which, prefer_srgb)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

        let ubo_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor{
            label: Some("ubo-bdg"),
            layout: &ubo_layout,
            entries: &[
                wgpu::BindGroupEntry{ binding: 0, resource: ubo.as_entire_binding() },
                wgpu::BindGroupEntry{ binding: 1, resource: wgpu::BindingResource::TextureView(&lut.view) },
                wgpu::BindGroupEntry{ binding: 2, resource: wgpu::BindingResource::Sampler(&lut.sampler) },
            ],
        });

        Ok(Self{
            width, height, grid,
            device, queue,
            pipeline,
            vbuf, ibuf, nidx,
            ubo, ubo_bind_group,
            colormap_lut: lut,
            color, color_view,
            depth, depth_view,
            globals,
        })
    }

    #[pyo3(text_signature = "($self, path)")]
    pub fn render_png(&mut self, path: String) -> PyResult<()> {
        // Encode pass
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor{ label: Some("terrain-encoder") });
        {
            let mut rp = encoder.begin_render_pass(&wgpu::RenderPassDescriptor{
                label: Some("terrain-rp"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment{
                    view: &self.color_view,
                    resolve_target: None,
                    ops: wgpu::Operations{
                        load:  wgpu::LoadOp::Clear(wgpu::Color{ r: 0.02, g: 0.02, b: 0.03, a: 1.0 }),
                        store: wgpu::StoreOp::Store,
                    }
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment{
                    view: &self.depth_view,
                    depth_ops: Some(wgpu::Operations{ load: wgpu::LoadOp::Clear(1.0), store: wgpu::StoreOp::Store }),
                    stencil_ops: None,
                }),
                ..Default::default()
            });
            rp.set_pipeline(&self.pipeline);
            rp.set_bind_group(0, &self.ubo_bind_group, &[]);
            rp.set_vertex_buffer(0, self.vbuf.slice(..));
            rp.set_index_buffer(self.ibuf.slice(..), wgpu::IndexFormat::Uint32);
            rp.draw_indexed(0..self.nidx, 0, 0..1);
        }
        self.queue.submit(Some(encoder.finish()));

        // Readback → PNG
        let bytes_per_pixel = 4u32;
        let unpadded_bpr = self.width * bytes_per_pixel;
        let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        let padded_bpr = ((unpadded_bpr + align - 1) / align) * align;

        let buf_size = (padded_bpr * self.height) as wgpu::BufferAddress;
        let readback = self.device.create_buffer(&wgpu::BufferDescriptor{
            label: Some("terrain-readback"),
            size: buf_size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor{ label: Some("copy-encoder") });
        encoder.copy_texture_to_buffer(
            wgpu::ImageCopyTexture{
                texture: &self.color,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All
            },
            wgpu::ImageCopyBuffer{
                buffer: &readback,
                layout: wgpu::ImageDataLayout{
                    offset: 0,
                    bytes_per_row: Some(NonZeroU32::new(padded_bpr).unwrap().into()),
                    rows_per_image: Some(NonZeroU32::new(self.height).unwrap().into()),
                }
            },
            wgpu::Extent3d{ width: self.width, height: self.height, depth_or_array_layers: 1 }
        );
        self.queue.submit(Some(encoder.finish()));

        let slice = readback.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_|{});
        self.device.poll(wgpu::Maintain::Wait);
        let data = slice.get_mapped_range();

        let mut pixels = Vec::with_capacity((unpadded_bpr * self.height) as usize);
        for row in 0..self.height {
            let start = (row * padded_bpr) as usize;
            let end   = start + unpadded_bpr as usize;
            pixels.extend_from_slice(&data[start..end]);
        }
        drop(data);
        readback.unmap();

        let img = image::RgbaImage::from_raw(self.width, self.height, pixels)
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("Invalid image buffer"))?;
        img.save(path).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        Ok(())
    }
}

// ---------- Geometry (analytic spike) ----------

fn build_grid_mesh(device: &wgpu::Device, n: u32) -> (wgpu::Buffer, wgpu::Buffer, u32) {
    let n = n as usize;
    let w = n;
    let h = n;

    let scale = 1.5f32;
    let step_x = (2.0 * scale) / (w as f32 - 1.0);
    let step_z = (2.0 * scale) / (h as f32 - 1.0);

    let f = |x: f32, z: f32| -> f32 {
        (x * 1.3).sin() * 0.25 + (z * 1.1).cos() * 0.25
    };

    // positions
    let mut pos = vec![0.0f32; w*h*3];
    for j in 0..h {
        for i in 0..w {
            let x = -scale + i as f32 * step_x;
            let z = -scale + j as f32 * step_z;
            let y = f(x, z);
            let idx = (j*w + i) * 3;
            pos[idx+0] = x; pos[idx+1] = y; pos[idx+2] = z;
        }
    }

    // normals via central differences
    let mut nrm = vec![0.0f32; w*h*3];
    for j in 0..h {
        for i in 0..w {
            let i0 = if i>0   { i-1 } else { i };
            let i1 = if i+1<w { i+1 } else { i };
            let j0 = if j>0   { j-1 } else { j };
            let j1 = if j+1<h { j+1 } else { j };

            let p = |ii, jj| {
                let k = (jj*w + ii)*3;
                glam::Vec3::new(pos[k], pos[k+1], pos[k+2])
            };
            let dx = p(i1,j) - p(i0,j);
            let dz = p(i,j1) - p(i,j0);
            let n  = dz.cross(dx).normalize_or_zero();

            let k = (j*w + i)*3;
            nrm[k]=n.x; nrm[k+1]=n.y; nrm[k+2]=n.z;
        }
    }

    // interleave pos + nrm
    let mut verts: Vec<f32> = Vec::with_capacity(w*h*6);
    for k in 0..(w*h) {
        verts.extend_from_slice(&pos[k*3..k*3+3]);
        verts.extend_from_slice(&nrm[k*3..k*3+3]);
    }

    // indices
    let mut idx = Vec::<u32>::with_capacity((w-1)*(h-1)*6);
    for j in 0..h-1 {
        for i in 0..w-1 {
            let a = (j*w + i) as u32;
            let b = (j*w + i + 1) as u32;
            let c = ((j+1)*w + i) as u32;
            let d = ((j+1)*w + i + 1) as u32;
            idx.extend_from_slice(&[a,c,b, b,c,d]);
        }
    }

    let vbuf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor{
        label: Some("terrain-vbuf"),
        contents: bytemuck::cast_slice(&verts),
        usage: wgpu::BufferUsages::VERTEX,
    });
    let ibuf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor{
        label: Some("terrain-ibuf"),
        contents: bytemuck::cast_slice(&idx),
        usage: wgpu::BufferUsages::INDEX,
    });
    (vbuf, ibuf, idx.len() as u32)
}

// MVP + light
fn build_view_matrices(width: u32, height: u32) -> (glam::Mat4, glam::Mat4, glam::Vec3) {
    let aspect = width as f32 / height as f32;
    let proj = glam::Mat4::perspective_rh_gl(45f32.to_radians(), aspect, 0.1, 100.0);
    let view = glam::Mat4::look_at_rh(
        glam::Vec3::new(3.0, 2.0, 3.0),
        glam::Vec3::ZERO,
        glam::Vec3::Y,
    );
    let light = glam::Vec3::new(0.5, 1.0, 0.3).normalize();
    (view, proj, light)
}
// A2-END:terrain-module
