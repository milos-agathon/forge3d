// A2-BEGIN:terrain-module
#![allow(dead_code)]

use std::num::NonZeroU32;
use pyo3::prelude::*;
use wgpu::util::DeviceExt;

// Built-in colormap palettes (256 RGBA8 values each)
const VIRIDIS_PALETTE: &[u8] = include_bytes!("../data/viridis_256.rgba");
const MAGMA_PALETTE: &[u8] = include_bytes!("../data/magma_256.rgba");
const TERRAIN_PALETTE: &[u8] = include_bytes!("../data/terrain_256.rgba");

#[derive(Clone, Copy)]
pub enum ColormapType {
    Viridis,
    Magma,
    Terrain,
}

impl ColormapType {
    pub fn get_palette_data(&self) -> &'static [u8] {
        match self {
            ColormapType::Viridis => VIRIDIS_PALETTE,
            ColormapType::Magma => MAGMA_PALETTE,
            ColormapType::Terrain => TERRAIN_PALETTE,
        }
    }
    
    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "viridis" => Some(ColormapType::Viridis),
            "magma" => Some(ColormapType::Magma),
            "terrain" => Some(ColormapType::Terrain),
            _ => None,
        }
    }
}

// CPU reference for unit testing
pub fn sample_colormap_cpu(colormap: ColormapType, normalized_height: f32) -> [u8; 4] {
    let palette = colormap.get_palette_data();
    let idx = (normalized_height.clamp(0.0, 1.0) * 255.0) as usize;
    let offset = idx * 4;
    [palette[offset], palette[offset + 1], palette[offset + 2], palette[offset + 3]]
}

pub struct ColormapLUT {
    pub texture: wgpu::Texture,
    pub view: wgpu::TextureView,
    pub sampler: wgpu::Sampler,
}

impl ColormapLUT {
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue, colormap: ColormapType) -> Self {
        let palette_data = colormap.get_palette_data();
        
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("colormap_lut"),
            size: wgpu::Extent3d { width: 256, height: 1, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        
        queue.write_texture(
            wgpu::ImageCopyTexture {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            palette_data,
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(256 * 4), // 256 RGBA8 pixels = 1024 bytes
                rows_per_image: Some(1),
            },
            wgpu::Extent3d { width: 256, height: 1, depth_or_array_layers: 1 },
        );
        
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("colormap_sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });
        
        Self { texture, view, sampler }
    }
}

// Terrain uniforms structure with height normalization
#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct TerrainUniforms {
    pub mvp: [[f32; 4]; 4], // 64 bytes - mat4x4
    pub light: [f32; 4],    // 16 bytes - vec4 (xyz + padding)
    pub h_min: f32,         // 4 bytes
    pub h_max: f32,         // 4 bytes  
    pub exaggeration: f32,  // 4 bytes
    pub _padding: f32,      // 4 bytes - total 96 bytes (aligned)
}

impl TerrainUniforms {
    pub fn new(mvp: glam::Mat4, light: glam::Vec3, h_min: f32, h_max: f32, exaggeration: f32) -> Self {
        Self {
            mvp: mvp.to_cols_array_2d(),
            light: [light.x, light.y, light.z, 0.0],
            h_min,
            h_max,
            exaggeration,
            _padding: 0.0,
        }
    }
}

const TEXTURE_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8UnormSrgb;
const DEPTH_FORMAT:   wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

#[pyclass(module = "_vulkan_forge", name = "TerrainSpike")]
pub struct TerrainSpike {
    width: u32,
    height: u32,
    grid: u32,

    device: wgpu::Device,
    queue:  wgpu::Queue,

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
}

#[pymethods]
impl TerrainSpike {
    #[new]
    #[pyo3(text_signature = "(width, height, grid=128, colormap='viridis')")]
    pub fn new(width: u32, height: u32, grid: Option<u32>, colormap: Option<String>) -> PyResult<Self> {
        let grid = grid.unwrap_or(128).max(2);
        let colormap_type = match colormap {
            Some(ref s) => {
                ColormapType::from_str(s).ok_or_else(|| {
                    pyo3::exceptions::PyRuntimeError::new_err(
                        format!("Unknown colormap '{}'. Supported: viridis, magma, terrain", s)
                    )
                })?
            }
            None => ColormapType::Viridis, // Default when not specified
        };

        // Instance + adapter + device
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
            source: wgpu::ShaderSource::Wgsl(include_str!("./shaders/terrain.wgsl").into()),
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
                        min_binding_size: None,
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
                buffers: &[
                    wgpu::VertexBufferLayout {
                        array_stride: (4 * 3 + 4 * 3) as wgpu::BufferAddress, // pos(vec3) + normal(vec3)
                        step_mode: wgpu::VertexStepMode::Vertex,
                        attributes: &[
                            wgpu::VertexAttribute { shader_location: 0, offset: 0,                     format: wgpu::VertexFormat::Float32x3 },
                            wgpu::VertexAttribute { shader_location: 1, offset: (4 * 3) as u64,        format: wgpu::VertexFormat::Float32x3 },
                        ],
                    },
                ],
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
        let (mvp, light) = build_view_matrices(width, height);
        let terrain_uniforms = TerrainUniforms::new(mvp, light, -1.0, 1.0, 1.0); // Default height range
        let ubo = device.create_buffer_init(&wgpu::util::BufferInitDescriptor{
            label: Some("ubo"),
            contents: bytemuck::cast_slice(&[terrain_uniforms]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        
        // Create colormap LUT texture
        let colormap_lut = ColormapLUT::new(&device, &queue, colormap_type);
        
        let ubo_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor{
            label: Some("ubo-bdg"),
            layout: &ubo_layout,
            entries: &[
                wgpu::BindGroupEntry{ binding: 0, resource: ubo.as_entire_binding() },
                wgpu::BindGroupEntry{ binding: 1, resource: wgpu::BindingResource::TextureView(&colormap_lut.view) },
                wgpu::BindGroupEntry{ binding: 2, resource: wgpu::BindingResource::Sampler(&colormap_lut.sampler) },
            ],
        });

        Ok(Self{
            width, height, grid,
            device, queue,
            pipeline,
            vbuf, ibuf, nidx,
            ubo, ubo_bind_group,
            colormap_lut,
            color, color_view,
            depth, depth_view,
        })
    }

    #[pyo3(text_signature = "($self, path)")]
    pub fn render_png(&mut self, path: String) -> PyResult<()> {
        // Draw
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor{ label: Some("terrain-encoder") });
        {
            let mut rp = encoder.begin_render_pass(&wgpu::RenderPassDescriptor{
                label: Some("terrain-rp"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment{
                    view: &self.color_view,
                    resolve_target: None,
                    ops: wgpu::Operations{
                        load:  wgpu::LoadOp::Clear(wgpu::Color{ r: 1.0, g: 1.0, b: 1.0, a: 1.0 }),
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
                    // wgpu 0.19 expects Option<u32> here; convert NonZeroU32 → u32 via .into()
                    bytes_per_row: Some(NonZeroU32::new(padded_bpr).unwrap().into()),
                    rows_per_image: Some(NonZeroU32::new(self.height).unwrap().into()),
                }
            },
            wgpu::Extent3d{ width: self.width, height: self.height, depth_or_array_layers: 1 }
        );
        self.queue.submit(Some(encoder.finish()));

        // Map & unpad
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
        img.save(path).map_err(|e|
            pyo3::exceptions::PyRuntimeError::new_err(e.to_string())
        )?;
        Ok(())
    }
}

// Build a simple XZ grid with analytic height and CPU normals.
fn build_grid_mesh(device: &wgpu::Device, n: u32) -> (wgpu::Buffer, wgpu::Buffer, u32) {
    let n = n as usize;
    let w = n;
    let h = n;

    let scale = 1.5f32;
    let step_x = (2.0 * scale) / (w as f32 - 1.0);
    let step_z = (2.0 * scale) / (h as f32 - 1.0);

    // Height function (removed unnecessary outer parentheses)
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

// MVP + light direction (separated for new uniform structure)
fn build_view_matrices(width: u32, height: u32) -> (glam::Mat4, glam::Vec3) {
    let aspect = width as f32 / height as f32;
    let proj = glam::Mat4::perspective_rh_gl(45f32.to_radians(), aspect, 0.1, 100.0);
    let view = glam::Mat4::look_at_rh(
        glam::Vec3::new(3.0, 2.0, 3.0),
        glam::Vec3::ZERO,
        glam::Vec3::Y,
    );
    let model = glam::Mat4::IDENTITY;
    let mvp = proj * view * model;

    let light = glam::Vec3::new(0.5, 1.0, 0.3).normalize();
    
    (mvp, light)
}
// A2-END:terrain-module
