// src/viewer/mod.rs
// Workstream I1: Interactive windowed viewer for forge3d
// - Creates window with winit 0.29
// - Handles input events (mouse, keyboard)
// - Renders frames at 60 FPS
// - Orbit and FPS camera modes

pub mod camera_controller;

use camera_controller::{CameraController, CameraMode};
use glam::{Mat4, Vec3};
use std::io::BufRead;
use std::sync::Arc;
use std::time::{Duration, Instant};
use wgpu::{Device, Instance, Queue, Surface, SurfaceConfiguration};
use wgpu::util::DeviceExt;
use image;
use std::path::Path;
use crate::core::ibl::{IBLRenderer, IBLQuality};
use winit::{
    dpi::PhysicalSize,
    event::*,
    event_loop::EventLoop,
    keyboard::{KeyCode, PhysicalKey},
    window::{Window, WindowBuilder},
};
// once_cell imported via path for INITIAL_CMDS; no direct use import needed

#[derive(Clone)]
pub struct ViewerConfig {
    pub width: u32,
    pub height: u32,
    pub title: String,
    pub vsync: bool,
    pub fov_deg: f32,
    pub znear: f32,
    pub zfar: f32,
}

// Global initial commands for viewer (set by CLI parser in example)
static INITIAL_CMDS: once_cell::sync::OnceCell<Vec<String>> = once_cell::sync::OnceCell::new();

pub fn set_initial_commands(cmds: Vec<String>) {
    let _ = INITIAL_CMDS.set(cmds);
}

impl Default for ViewerConfig {
    fn default() -> Self {
        Self {
            width: 1024,
            height: 768,
            title: "forge3d Interactive Viewer".to_string(),
            vsync: true,
            fov_deg: 45.0,
            znear: 0.1,
            zfar: 1000.0,
        }
    }
}
pub struct Viewer {
    window: Arc<Window>,
    surface: Surface<'static>,
    device: Device,
    queue: Queue,
    config: SurfaceConfiguration,
    camera: CameraController,
    view_config: ViewerConfig,
    frame_count: u64,
    fps_counter: FpsCounter,
    // Input state
    keys_pressed: std::collections::HashSet<KeyCode>,
    shift_pressed: bool,
    // GI manager and toggles
    gi: Option<crate::core::screen_space_effects::ScreenSpaceEffectsManager>,
    // Snapshot request path (processed on next frame before present)
    snapshot_request: Option<String>,
    // Offscreen color to read back when snapshotting this frame
    pending_snapshot_tex: Option<wgpu::Texture>,
    // GBuffer geometry pipeline and resources
    geom_bind_group_layout: Option<wgpu::BindGroupLayout>,
    geom_pipeline: Option<wgpu::RenderPipeline>,
    geom_camera_buffer: Option<wgpu::Buffer>,
    geom_bind_group: Option<wgpu::BindGroup>,
    geom_vb: Option<wgpu::Buffer>,
    geom_ib: Option<wgpu::Buffer>,
    geom_index_count: u32,
    z_texture: Option<wgpu::Texture>,
    z_view: Option<wgpu::TextureView>,
    // Albedo texture for geometry
    albedo_texture: Option<wgpu::Texture>,
    albedo_view: Option<wgpu::TextureView>,
    albedo_sampler: Option<wgpu::Sampler>,
    // Composite pipeline (debug show material GBuffer on screen)
    comp_bind_group_layout: Option<wgpu::BindGroupLayout>,
    comp_pipeline: Option<wgpu::RenderPipeline>,
    comp_uniform: Option<wgpu::Buffer>,
    // Lit viz compute pipeline (albedo+normal shading)
    lit_bind_group_layout: wgpu::BindGroupLayout,
    lit_pipeline: wgpu::ComputePipeline,
    lit_uniform: wgpu::Buffer,
    lit_output: wgpu::Texture,
    lit_output_view: wgpu::TextureView,
    // Lit params (exposed via :lit-* commands)
    lit_sun_intensity: f32,
    lit_ibl_intensity: f32,
    lit_use_ibl: bool,
    // Fallback pipeline to draw a solid color when GI/geometry path is unavailable
    fallback_pipeline: wgpu::RenderPipeline,
    viz_mode: VizMode,
    // SSAO composite control
    use_ssao_composite: bool,
    // IBL integration
    ibl_renderer: Option<IBLRenderer>,
    ibl_env_view: Option<wgpu::TextureView>,
    ibl_sampler: Option<wgpu::Sampler>,
    // Viz depth override
    viz_depth_max_override: Option<f32>,
    // Auto-snapshot support (one-time)
    auto_snapshot_path: Option<String>,
    auto_snapshot_done: bool,
    // Debug: log render gate and snapshot once
    debug_logged_render_gate: bool,
}

struct FpsCounter {
    frames: u32,
    last_report: Instant,
    current_fps: f32,
}

impl FpsCounter {
    fn new() -> Self {
        Self {
            frames: 0,
            last_report: Instant::now(),
            current_fps: 0.0,
        }
    }

    fn tick(&mut self) -> Option<f32> {
        self.frames += 1;
        let elapsed = self.last_report.elapsed();
        if elapsed >= Duration::from_secs(1) {
            self.current_fps = self.frames as f32 / elapsed.as_secs_f32();
            self.frames = 0;
            self.last_report = Instant::now();
            Some(self.current_fps)
        } else {
            None
        }
    }

    fn fps(&self) -> f32 {
        self.current_fps
    }
}

impl Viewer {
    fn load_ibl(&mut self, path: &str) -> anyhow::Result<()> {
        // Load HDR image from disk
        let hdr_img = crate::formats::hdr::load_hdr(path)
            .map_err(|e| anyhow::anyhow!("failed to load HDR '{}': {}", path, e))?;

        // Build IBL renderer and upload environment
        let mut ibl = IBLRenderer::new(&self.device, IBLQuality::Low);
        ibl.set_base_resolution(IBLQuality::Low.base_environment_size());
        ibl.load_environment_map(
            &self.device,
            &self.queue,
            &hdr_img.data,
            hdr_img.width,
            hdr_img.height,
        )
        .map_err(|e| anyhow::anyhow!("failed to upload environment: {}", e))?;
        ibl.initialize(&self.device, &self.queue)
            .map_err(|e| anyhow::anyhow!("failed to initialize IBL: {}", e))?;

        // Wire SSGI to irradiance and SSR to specular
        let (irr_tex_opt, spec_tex_opt, _) = ibl.textures();
        if let Some(ref mut gi) = self.gi {
            if let Some(irr_tex) = irr_tex_opt {
                gi.set_ssgi_env(&self.device, irr_tex);
            }
            if let Some(spec_tex) = spec_tex_opt {
                gi.set_ssr_env(&self.device, spec_tex);
                // Keep a viewer-side view/sampler for diagnostics
                let cube_view = spec_tex.create_view(&wgpu::TextureViewDescriptor {
                    label: Some("viewer.ibl.specular.cube.view"),
                    format: Some(wgpu::TextureFormat::Rgba16Float),
                    dimension: Some(wgpu::TextureViewDimension::Cube),
                    aspect: wgpu::TextureAspect::All,
                    base_mip_level: 0,
                    mip_level_count: None,
                    base_array_layer: 0,
                    array_layer_count: Some(6),
                });
                let env_sampler = self.device.create_sampler(&wgpu::SamplerDescriptor {
                    label: Some("viewer.ibl.env.sampler"),
                    address_mode_u: wgpu::AddressMode::ClampToEdge,
                    address_mode_v: wgpu::AddressMode::ClampToEdge,
                    address_mode_w: wgpu::AddressMode::ClampToEdge,
                    mag_filter: wgpu::FilterMode::Linear,
                    min_filter: wgpu::FilterMode::Linear,
                    mipmap_filter: wgpu::FilterMode::Linear,
                    ..Default::default()
                });
                self.ibl_env_view = Some(cube_view);
                self.ibl_sampler = Some(env_sampler);
            }
        }

        // Keep IBL resources alive
        self.ibl_renderer = Some(ibl);
        Ok(())
    }
    pub async fn new(
        window: Arc<Window>,
        config: ViewerConfig,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let size = window.inner_size();

        // Create wgpu instance
        let instance = Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        // Create surface - use Arc::clone to satisfy lifetime requirements
        let surface = instance.create_surface(Arc::clone(&window))?;

        // Request adapter
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .ok_or("Failed to find suitable adapter")?;

        // Request device and queue
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("Viewer Device"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                },
                None,
            )
            .await?;

        // Configure surface
        let surface_caps = surface.get_capabilities(&adapter);
        let surface_format = surface_caps
            .formats
            .iter()
            .copied()
            .find(|f| f.is_srgb())
            .unwrap_or(surface_caps.formats[0]);

        let surface_config = SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            format: surface_format,
            width: size.width,
            height: size.height,
            present_mode: if config.vsync {
                wgpu::PresentMode::AutoVsync
            } else {
                wgpu::PresentMode::AutoNoVsync
            },
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };

        surface.configure(&device, &surface_config);

        // Initialize P5 Screen-space effects manager (optional)
        let gi = match crate::core::screen_space_effects::ScreenSpaceEffectsManager::new(
            &device,
            surface_config.width,
            surface_config.height,
        ) {
            Ok(m) => Some(m),
            Err(e) => {
                eprintln!("Failed to create ScreenSpaceEffectsManager: {}", e);
                None
            }
        };

        // Build geometry pipeline only if GI is available (needs GBuffer formats)
        let (
            geom_bind_group_layout,
            geom_pipeline,
            geom_camera_buffer,
            geom_bind_group,
            geom_vb,
            z_texture,
            z_view,
            albedo_texture,
            albedo_view,
            albedo_sampler,
            comp_bind_group_layout,
            comp_pipeline,
        ) = if let Some(ref gi_ref) = gi {
            // Z-buffer for rasterization
            let z_texture = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("viewer.gbuf.z"),
                size: wgpu::Extent3d { width: surface_config.width, height: surface_config.height, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Depth32Float,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                view_formats: &[],
            });
            let z_view = z_texture.create_view(&wgpu::TextureViewDescriptor::default());

            // Camera uniform
            let geom_camera_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("viewer.gbuf.cam"),
                size: (std::mem::size_of::<[[f32;4];4]>() * 2) as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });

            // Bind group layout: camera uniform + albedo texture + sampler
            let geom_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("viewer.gbuf.geom.bgl"),
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

            // Shader for geometry GBuffer write (with texcoords)
            let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("viewer.gbuf.geom.shader"),
                source: wgpu::ShaderSource::Wgsl(r#"
                    struct Camera {
                        view : mat4x4<f32>,
                        proj : mat4x4<f32>,
                    };
                    @group(0) @binding(0) var<uniform> uCam : Camera;
                    @group(0) @binding(1) var tAlbedo : texture_2d<f32>;
                    @group(0) @binding(2) var sAlbedo : sampler;

                    struct VSIn {
                        @location(0) pos : vec3<f32>,
                        @location(1) nrm : vec3<f32>,
                        @location(2) uv  : vec2<f32>,
                    };
                    struct VSOut {
                        @builtin(position) pos : vec4<f32>,
                        @location(0) v_nrm_vs : vec3<f32>,
                        @location(1) v_depth_vs : f32,
                        @location(2) v_uv : vec2<f32>,
                    };

                    @vertex
                    fn vs_main(inp: VSIn) -> VSOut {
                        var out: VSOut;
                        let pos_ws = vec4<f32>(inp.pos, 1.0);
                        let pos_vs = uCam.view * pos_ws;
                        out.pos = uCam.proj * pos_vs;
                        let nrm_vs = (uCam.view * vec4<f32>(inp.nrm, 0.0)).xyz;
                        out.v_nrm_vs = normalize(nrm_vs);
                        out.v_depth_vs = -pos_vs.z; // positive view-space depth
                        out.v_uv = inp.uv;
                        return out;
                    }

                    struct FSOut {
                        @location(0) normal_rgba : vec4<f32>,
                        @location(1) albedo_rgba : vec4<f32>,
                        @location(2) depth_r : f32,
                    };

                    @fragment
                    fn fs_main(inp: VSOut) -> FSOut {
                        var out: FSOut;
                        out.normal_rgba = vec4<f32>(normalize(inp.v_nrm_vs), 1.0);
                        let color = textureSample(tAlbedo, sAlbedo, inp.v_uv);
                        out.albedo_rgba = vec4<f32>(color.rgb, 1.0);
                        out.depth_r = inp.v_depth_vs;
                        return out;
                    }
                "#.into()),
            });

            // Vertex buffer (textured cube, 36 vertices: pos, nrm, uv)
            let verts: [f32; 36 * 8] = [
                // A simple cube centered at origin with normals and UVs per face
                // +Z face
                -1.0, -1.0,  1.0,   0.0, 0.0, 1.0,   0.0, 0.0,
                 1.0, -1.0,  1.0,   0.0, 0.0, 1.0,   1.0, 0.0,
                 1.0,  1.0,  1.0,   0.0, 0.0, 1.0,   1.0, 1.0,
                -1.0, -1.0,  1.0,   0.0, 0.0, 1.0,   0.0, 0.0,
                 1.0,  1.0,  1.0,   0.0, 0.0, 1.0,   1.0, 1.0,
                -1.0,  1.0,  1.0,   0.0, 0.0, 1.0,   0.0, 1.0,
                // -Z face
                -1.0, -1.0, -1.0,   0.0, 0.0,-1.0,   1.0, 0.0,
                 1.0,  1.0, -1.0,   0.0, 0.0,-1.0,   0.0, 1.0,
                 1.0, -1.0, -1.0,   0.0, 0.0,-1.0,   0.0, 0.0,
                -1.0, -1.0, -1.0,   0.0, 0.0,-1.0,   1.0, 0.0,
                -1.0,  1.0, -1.0,   0.0, 0.0,-1.0,   1.0, 1.0,
                 1.0,  1.0, -1.0,   0.0, 0.0,-1.0,   0.0, 1.0,
                // +X face
                 1.0, -1.0, -1.0,   1.0, 0.0, 0.0,   0.0, 0.0,
                 1.0,  1.0,  1.0,   1.0, 0.0, 0.0,   1.0, 1.0,
                 1.0, -1.0,  1.0,   1.0, 0.0, 0.0,   1.0, 0.0,
                 1.0, -1.0, -1.0,   1.0, 0.0, 0.0,   0.0, 0.0,
                 1.0,  1.0, -1.0,   1.0, 0.0, 0.0,   0.0, 1.0,
                 1.0,  1.0,  1.0,   1.0, 0.0, 0.0,   1.0, 1.0,
                // -X face
                -1.0, -1.0, -1.0,  -1.0, 0.0, 0.0,   1.0, 0.0,
                -1.0, -1.0,  1.0,  -1.0, 0.0, 0.0,   0.0, 0.0,
                -1.0,  1.0,  1.0,  -1.0, 0.0, 0.0,   0.0, 1.0,
                -1.0, -1.0, -1.0,  -1.0, 0.0, 0.0,   1.0, 0.0,
                -1.0,  1.0,  1.0,  -1.0, 0.0, 0.0,   0.0, 1.0,
                -1.0,  1.0, -1.0,  -1.0, 0.0, 0.0,   1.0, 1.0,
                // +Y face
                -1.0,  1.0, -1.0,   0.0, 1.0, 0.0,   0.0, 1.0,
                 1.0,  1.0,  1.0,   0.0, 1.0, 0.0,   1.0, 0.0,
                 1.0,  1.0, -1.0,   0.0, 1.0, 0.0,   1.0, 1.0,
                -1.0,  1.0, -1.0,   0.0, 1.0, 0.0,   0.0, 1.0,
                -1.0,  1.0,  1.0,   0.0, 1.0, 0.0,   0.0, 0.0,
                 1.0,  1.0,  1.0,   0.0, 1.0, 0.0,   1.0, 0.0,
                // -Y face
                -1.0, -1.0, -1.0,   0.0,-1.0, 0.0,   0.0, 0.0,
                 1.0, -1.0, -1.0,   0.0,-1.0, 0.0,   1.0, 0.0,
                 1.0, -1.0,  1.0,   0.0,-1.0, 0.0,   1.0, 1.0,
                -1.0, -1.0, -1.0,   0.0,-1.0, 0.0,   0.0, 0.0,
                 1.0, -1.0,  1.0,   0.0,-1.0, 0.0,   1.0, 1.0,
                -1.0, -1.0,  1.0,   0.0,-1.0, 0.0,   0.0, 1.0,
            ];
            let geom_vb = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("viewer.gbuf.geom.vb"),
                contents: bytemuck::cast_slice(&verts),
                usage: wgpu::BufferUsages::VERTEX,
            });

            // Pipeline
            let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("viewer.gbuf.geom.pl"),
                bind_group_layouts: &[&geom_bgl],
                push_constant_ranges: &[],
            });

            let gb = gi_ref.gbuffer();
            let gb_cfg = gb.config();
            let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("viewer.gbuf.geom.pipeline"),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: "vs_main",
                    buffers: &[wgpu::VertexBufferLayout {
                        array_stride: (8 * std::mem::size_of::<f32>()) as u64,
                        step_mode: wgpu::VertexStepMode::Vertex,
                        attributes: &[
                            wgpu::VertexAttribute { shader_location: 0, offset: 0, format: wgpu::VertexFormat::Float32x3 },
                            wgpu::VertexAttribute { shader_location: 1, offset: (3 * 4) as u64, format: wgpu::VertexFormat::Float32x3 },
                            wgpu::VertexAttribute { shader_location: 2, offset: (6 * 4) as u64, format: wgpu::VertexFormat::Float32x2 },
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
                    module: &shader,
                    entry_point: "fs_main",
                    targets: &[
                        Some(wgpu::ColorTargetState { format: gb_cfg.normal_format, blend: None, write_mask: wgpu::ColorWrites::ALL }),
                        Some(wgpu::ColorTargetState { format: gb_cfg.material_format, blend: None, write_mask: wgpu::ColorWrites::ALL }),
                        Some(wgpu::ColorTargetState { format: gb_cfg.depth_format, blend: None, write_mask: wgpu::ColorWrites::ALL }),
                    ],
                }),
                multiview: None,
            });

            // Create an albedo texture (procedural checkerboard)
            let tex_size = 256u32;
            let mut pixels = vec![0u8; (tex_size * tex_size * 4) as usize];
            for y in 0..tex_size {
                for x in 0..tex_size {
                    let idx = ((y * tex_size + x) * 4) as usize;
                    let c = if ((x / 32) + (y / 32)) % 2 == 0 { 230 } else { 50 };
                    pixels[idx + 0] = c; // R
                    pixels[idx + 1] = 180; // G
                    pixels[idx + 2] = 80; // B
                    pixels[idx + 3] = 255; // A
                }
            }
            let albedo_texture = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("viewer.geom.albedo.tex"),
                size: wgpu::Extent3d { width: tex_size, height: tex_size, depth_or_array_layers: 1 },
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
                wgpu::ImageDataLayout { offset: 0, bytes_per_row: Some(tex_size * 4), rows_per_image: Some(tex_size) },
                wgpu::Extent3d { width: tex_size, height: tex_size, depth_or_array_layers: 1 },
            );
            let albedo_sampler = device.create_sampler(&wgpu::SamplerDescriptor::default());

            // Geometry bind group
            let geom_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("viewer.gbuf.geom.bg"),
                layout: &geom_bgl,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: geom_camera_buffer.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&albedo_view) },
                    wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(&albedo_sampler) },
                ],
            });

            // Composite pass: display selected viz (material/normal/depth/GI) onto swapchain
            let comp_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("viewer.comp.bgl"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: false },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });
            let comp_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("viewer.comp.shader"),
                source: wgpu::ShaderSource::Wgsl(r#"
                    struct CompParams { mode: u32, far: f32, _pad: vec2<f32> };
                    @group(0) @binding(2) var<uniform> uComp : CompParams;
                    @vertex
                    fn vs_fullscreen(@builtin(vertex_index) vid: u32) -> @builtin(position) vec4<f32> {
                        let x = f32((vid << 1u) & 2u);
                        let y = f32(vid & 2u);
                        return vec4<f32>(x * 2.0 - 1.0, 1.0 - y * 2.0, 0.0, 1.0);
                    }
                    @group(0) @binding(0) var gbuf_tex : texture_2d<f32>;
                    @group(0) @binding(1) var gbuf_sam : sampler;
                    @fragment
                    fn fs_fullscreen(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
                        let dims = vec2<f32>(textureDimensions(gbuf_tex));
                        let uv = pos.xy / dims;
                        var c = textureSample(gbuf_tex, gbuf_sam, uv);
                        if (uComp.mode == 1u) {
                            // normal: [-1,1] -> [0,1]
                            c = vec4<f32>(0.5 * (c.xyz + vec3<f32>(1.0)), 1.0);
                        } else if (uComp.mode == 2u) {
                            // depth: view-space depth mapped by far
                            let d = clamp(c.r / max(0.0001, uComp.far), 0.0, 1.0);
                            c = vec4<f32>(d, d, d, 1.0);
                        }
                        return c;
                    }
                "#.into()),
            });
            let comp_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("viewer.comp.pl"),
                bind_group_layouts: &[&comp_bgl],
                push_constant_ranges: &[],
            });
            let comp_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("viewer.comp.pipeline"),
                layout: Some(&comp_pl),
                vertex: wgpu::VertexState { module: &comp_shader, entry_point: "vs_fullscreen", buffers: &[] },
                primitive: wgpu::PrimitiveState::default(),
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                fragment: Some(wgpu::FragmentState {
                    module: &comp_shader,
                    entry_point: "fs_fullscreen",
                    targets: &[Some(wgpu::ColorTargetState { format: surface_config.format, blend: None, write_mask: wgpu::ColorWrites::ALL })],
                }),
                multiview: None,
            });

            (Some(geom_bgl), Some(pipeline), Some(geom_camera_buffer), Some(geom_bg), Some(geom_vb), Some(z_texture), Some(z_view), Some(albedo_texture), Some(albedo_view), Some(albedo_sampler), Some(comp_bgl), Some(comp_pipeline))
        } else {
            (None, None, None, None, None, None, None, None, None, None, None, None)
        };

        // Always-available fallback pipeline (solid fullscreen triangle)
        let fb_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("viewer.fallback.shader"),
            source: wgpu::ShaderSource::Wgsl(r#"
                @vertex
                fn vs_fb(@builtin(vertex_index) vid: u32) -> @builtin(position) vec4<f32> {
                    let x = f32((vid << 1u) & 2u);
                    let y = f32(vid & 2u);
                    return vec4<f32>(x * 2.0 - 1.0, 1.0 - y * 2.0, 0.0, 1.0);
                }
                @fragment
                fn fs_fb() -> @location(0) vec4<f32> {
                    return vec4<f32>(0.05, 0.0, 0.15, 1.0);
                }
            "#.into()),
        });
        let fb_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("viewer.fallback.pipeline"),
            layout: None,
            vertex: wgpu::VertexState { module: &fb_shader, entry_point: "vs_fb", buffers: &[] },
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &fb_shader,
                entry_point: "fs_fb",
                targets: &[Some(wgpu::ColorTargetState { format: surface_config.format, blend: None, write_mask: wgpu::ColorWrites::ALL })],
            }),
            multiview: None,
        });

        // Lit viz compute pipeline (albedo+normal shading with optional IBL)
        let lit_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("viewer.lit.bgl"),
            entries: &[
                // normal, material, depth
                wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Texture { sample_type: wgpu::TextureSampleType::Float { filterable: false }, view_dimension: wgpu::TextureViewDimension::D2, multisampled: false }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 1, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Texture { sample_type: wgpu::TextureSampleType::Float { filterable: false }, view_dimension: wgpu::TextureViewDimension::D2, multisampled: false }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 2, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Texture { sample_type: wgpu::TextureSampleType::Float { filterable: false }, view_dimension: wgpu::TextureViewDimension::D2, multisampled: false }, count: None },
                // output
                wgpu::BindGroupLayoutEntry { binding: 3, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::StorageTexture { access: wgpu::StorageTextureAccess::WriteOnly, format: wgpu::TextureFormat::Rgba8Unorm, view_dimension: wgpu::TextureViewDimension::D2 }, count: None },
                // env cube + sampler
                wgpu::BindGroupLayoutEntry { binding: 4, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Texture { sample_type: wgpu::TextureSampleType::Float { filterable: true }, view_dimension: wgpu::TextureViewDimension::Cube, multisampled: false }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 5, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering), count: None },
                // params
                wgpu::BindGroupLayoutEntry { binding: 6, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None },
            ],
        });
        let lit_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("viewer.lit.compute.shader"),
            source: wgpu::ShaderSource::Wgsl(r#"
                struct LitParams { sun_dir_vs: vec3<f32>, sun_intensity: f32, ibl_intensity: f32, use_ibl: u32 };
                @group(0) @binding(0) var normal_tex : texture_2d<f32>;
                @group(0) @binding(1) var albedo_tex : texture_2d<f32>;
                @group(0) @binding(2) var depth_tex  : texture_2d<f32>;
                @group(0) @binding(3) var out_tex    : texture_storage_2d<rgba8unorm, write>;
                @group(0) @binding(4) var env_cube   : texture_cube<f32>;
                @group(0) @binding(5) var env_samp   : sampler;
                @group(0) @binding(6) var<uniform> P : LitParams;

                @compute @workgroup_size(8,8,1)
                fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
                    let dims = textureDimensions(normal_tex);
                    if (gid.x >= dims.x || gid.y >= dims.y) { return; }
                    let coord = vec2<i32>(gid.xy);
                    var n = textureLoad(normal_tex, coord, 0).xyz; // view-space [-1,1]
                    n = normalize(n);
                    let a = textureLoad(albedo_tex, coord, 0).rgb;
                    let l = normalize(P.sun_dir_vs);
                    let ndl = max(dot(n, l), 0.0);
                    var col = a * (0.1 + P.sun_intensity * ndl);
                    if (P.use_ibl != 0u) {
                        let env = textureSampleLevel(env_cube, env_samp, n, 0.0).rgb;
                        col += a * env * P.ibl_intensity;
                    }
                    textureStore(out_tex, coord, vec4<f32>(col, 1.0));
                }
            "#.into()),
        });
        let lit_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some("viewer.lit.pl"), bind_group_layouts: &[&lit_bgl], push_constant_ranges: &[] });
        let lit_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor { label: Some("viewer.lit.pipeline"), layout: Some(&lit_pl), module: &lit_shader, entry_point: "cs_main" });
        let lit_params: [f32; 8] = [
            // sun_dir_vs.xyz, sun_intensity
            0.3, 0.6, -1.0, 1.0,
            // ibl_intensity, use_ibl (as float for alignment), pad, pad
            0.6, 1.0, 0.0, 0.0,
        ];
        let lit_uniform = device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some("viewer.lit.uniform"), contents: bytemuck::cast_slice(&lit_params), usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST });

        // Dummy IBL cube (1x1x6) and sampler as fallback so lit viz always binds
        let dummy_env = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("viewer.lit.dummy.env"),
            size: wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 6 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let dummy_env_view = dummy_env.create_view(&wgpu::TextureViewDescriptor {
            label: Some("viewer.lit.dummy.env.view"),
            format: Some(wgpu::TextureFormat::Rgba8Unorm),
            dimension: Some(wgpu::TextureViewDimension::Cube),
            aspect: wgpu::TextureAspect::All,
            base_mip_level: 0,
            mip_level_count: None,
            base_array_layer: 0,
            array_layer_count: Some(6),
        });
        let dummy_env_sampler = device.create_sampler(&wgpu::SamplerDescriptor { label: Some("viewer.lit.dummy.env.sampler"), mag_filter: wgpu::FilterMode::Linear, min_filter: wgpu::FilterMode::Linear, mipmap_filter: wgpu::FilterMode::Linear, ..Default::default() });

        // Lit output target
        let lit_output = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("viewer.lit.output"),
            size: wgpu::Extent3d { width: surface_config.width, height: surface_config.height, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let lit_output_view = lit_output.create_view(&wgpu::TextureViewDescriptor::default());

        Ok(Self {
            window,
            surface,
            device,
            queue,
            config: surface_config,
            camera: CameraController::new(),
            view_config: config,
            frame_count: 0,
            fps_counter: FpsCounter::new(),
            keys_pressed: std::collections::HashSet::new(),
            shift_pressed: false,
            gi,
            snapshot_request: None,
            pending_snapshot_tex: None,
            geom_bind_group_layout,
            geom_pipeline,
            geom_camera_buffer,
            geom_bind_group,
            geom_vb,
            geom_ib: None,
            geom_index_count: 36,
            z_texture: z_texture,
            z_view: z_view,
            albedo_texture: albedo_texture,
            albedo_view: albedo_view,
            albedo_sampler: albedo_sampler,
            comp_bind_group_layout,
            comp_pipeline,
            comp_uniform: None,
            lit_bind_group_layout: lit_bgl,
            lit_pipeline,
            lit_uniform,
            lit_output,
            lit_output_view,
            // Lit params defaults must match the initial lit_params above
            lit_sun_intensity: 1.0,
            lit_ibl_intensity: 0.6,
            lit_use_ibl: true,
            fallback_pipeline: fb_pipeline,
            viz_mode: VizMode::Material,
            use_ssao_composite: true,
            ibl_renderer: None,
            ibl_env_view: Some(dummy_env_view),
            ibl_sampler: Some(dummy_env_sampler),
            viz_depth_max_override: None,
            auto_snapshot_path: std::env::var("FORGE3D_AUTO_SNAPSHOT_PATH").ok(),
            auto_snapshot_done: false,
            debug_logged_render_gate: false,
        })
    }

    pub fn window(&self) -> &Window {
        &self.window
    }

    fn update_lit_uniform(&mut self) {
        // Keep sun_dir consistent with compute shader default
        let sun_dir = [0.3f32, 0.6, -1.0];
        let params: [f32; 8] = [
            sun_dir[0], sun_dir[1], sun_dir[2], self.lit_sun_intensity,
            self.lit_ibl_intensity, if self.lit_use_ibl { 1.0 } else { 0.0 }, 0.0, 0.0,
        ];
        self.queue
            .write_buffer(&self.lit_uniform, 0, bytemuck::cast_slice(&params));
    }

    pub fn resize(&mut self, new_size: PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            self.config.width = new_size.width;
            self.config.height = new_size.height;
            self.surface.configure(&self.device, &self.config);
            if let Some(ref mut gi) = self.gi {
                gi.gbuffer_mut().resize(&self.device, new_size.width, new_size.height).ok();
            }
            // Recreate lit output
            self.lit_output = self.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("viewer.lit.output"),
                size: wgpu::Extent3d { width: new_size.width, height: new_size.height, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8Unorm,
                usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            });
            self.lit_output_view = self.lit_output.create_view(&wgpu::TextureViewDescriptor::default());
            // Recreate depth buffer for geometry pass
            if self.geom_pipeline.is_some() {
                let z_texture = self.device.create_texture(&wgpu::TextureDescriptor {
                    label: Some("viewer.gbuf.z"),
                    size: wgpu::Extent3d { width: new_size.width, height: new_size.height, depth_or_array_layers: 1 },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: wgpu::TextureFormat::Depth32Float,
                    usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                    view_formats: &[],
                });
                let z_view = z_texture.create_view(&wgpu::TextureViewDescriptor::default());
                self.z_texture = Some(z_texture);
                self.z_view = Some(z_view);
            }
        }
    }

    pub fn handle_input(&mut self, event: &WindowEvent) -> bool {
        match event {
            WindowEvent::KeyboardInput {
                event: key_event, ..
            } => {
                if let PhysicalKey::Code(keycode) = key_event.physical_key {
                    let pressed = key_event.state == ElementState::Pressed;

                    // Track shift
                    if matches!(keycode, KeyCode::ShiftLeft | KeyCode::ShiftRight) {
                        self.shift_pressed = pressed;
                    }

                    // Track WASD, Q, E for FPS mode
                    if pressed {
                        self.keys_pressed.insert(keycode);
                    } else {
                        self.keys_pressed.remove(&keycode);
                    }

                    // Toggle camera mode with Tab
                    if pressed && keycode == KeyCode::Tab {
                        let new_mode = match self.camera.mode() {
                            CameraMode::Orbit => CameraMode::Fps,
                            CameraMode::Fps => CameraMode::Orbit,
                        };
                        self.camera.set_mode(new_mode);
                        println!("Camera mode: {:?}", new_mode);
                        return true;
                    }
                }

                true
            }
            WindowEvent::MouseInput { state, button, .. } => {
                if *button == MouseButton::Left {
                    self.camera.mouse_pressed = *state == ElementState::Pressed;
                }
                true
            }
            WindowEvent::CursorMoved { position, .. } => {
                self.camera
                    .handle_mouse_move(position.x as f32, position.y as f32);
                true
            }
            WindowEvent::MouseWheel { delta, .. } => {
                let scroll = match delta {
                    MouseScrollDelta::LineDelta(_, y) => *y,
                    MouseScrollDelta::PixelDelta(pos) => pos.y as f32 * 0.1,
                };
                self.camera.handle_mouse_scroll(scroll);
                true
            }
            _ => false,
        }
    }

    pub fn update(&mut self, dt: f32) {
        // Update FPS camera movement
        let mut forward = 0.0;
        let mut right = 0.0;
        let mut up = 0.0;

        let speed_mult = if self.shift_pressed { 2.0 } else { 1.0 };

        if self.keys_pressed.contains(&KeyCode::KeyW) {
            forward += speed_mult;
        }
        if self.keys_pressed.contains(&KeyCode::KeyS) {
            forward -= speed_mult;
        }
        if self.keys_pressed.contains(&KeyCode::KeyD) {
            right += speed_mult;
        }
        if self.keys_pressed.contains(&KeyCode::KeyA) {
            right -= speed_mult;
        }
        if self.keys_pressed.contains(&KeyCode::KeyE) {
            up += speed_mult;
        }
        if self.keys_pressed.contains(&KeyCode::KeyQ) {
            up -= speed_mult;
        }

        self.camera.update_fps(dt, forward, right, up);

        // Update GI camera params
        if let Some(ref mut gi) = self.gi {
            let aspect = self.config.width as f32 / self.config.height as f32;
            let fov = self.view_config.fov_deg.to_radians();
            let proj = Mat4::perspective_rh(fov, aspect, self.view_config.znear, self.view_config.zfar);
            let view = self.camera.view_matrix();
            let inv_proj = proj.inverse();

            fn to_arr4(m: Mat4) -> [[f32; 4]; 4] {
                let c = m.to_cols_array();
                [
                    [c[0], c[1], c[2], c[3]],
                    [c[4], c[5], c[6], c[7]],
                    [c[8], c[9], c[10], c[11]],
                    [c[12], c[13], c[14], c[15]],
                ]
            }
            let eye = self.camera.eye();
            let inv_view = view.inverse();
            let cam = crate::core::screen_space_effects::CameraParams {
                view_matrix: to_arr4(view),
                inv_view_matrix: to_arr4(inv_view),
                proj_matrix: to_arr4(proj),
                inv_proj_matrix: to_arr4(inv_proj),
                camera_pos: [eye.x, eye.y, eye.z],
                _pad: 0.0,
            };
            gi.update_camera(&self.queue, &cam);
        }
    }

    pub fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        let output = self.surface.get_current_texture()?;
        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        if self.frame_count == 0 {
            eprintln!("[viewer-debug] entering render loop (first frame)");
        }

        // Ensure auto-snapshot request is registered before encoding so we render to an offscreen texture
        if self.snapshot_request.is_none() && !self.auto_snapshot_done {
            if let Some(ref p) = self.auto_snapshot_path {
                self.snapshot_request = Some(p.clone());
                self.auto_snapshot_done = true;
                eprintln!("[viewer-debug] auto snapshot requested: {}", p);
            }
        }

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Render Encoder"),
            });

        // Composite debug: after GI/geometry, show GBuffer material on swapchain

        // Execute screen-space effects if any are enabled
        let have_gi = self.gi.is_some();
        let have_pipe = self.geom_pipeline.is_some();
        let have_cam = self.geom_camera_buffer.is_some();
        let have_vb = self.geom_vb.is_some();
        let have_z = self.z_view.is_some();
        let have_bgl = self.geom_bind_group_layout.is_some();
        if !(have_gi && have_pipe && have_cam && have_vb && have_z && have_bgl) {
            if !self.debug_logged_render_gate {
                eprintln!(
                    "[viewer-debug] render gate: gi={} pipe={} cam={} vb={} z={} bgl={}",
                    have_gi, have_pipe, have_cam, have_vb, have_z, have_bgl
                );
                self.debug_logged_render_gate = true;
            }
        }
        if let (Some(gi), Some(pipe), Some(cam_buf), Some(vb), Some(zv), Some(_bgl)) = (
            self.gi.as_ref(),
            self.geom_pipeline.as_ref(),
            self.geom_camera_buffer.as_ref(),
            self.geom_vb.as_ref(),
            self.z_view.as_ref(),
            self.geom_bind_group_layout.as_ref(),
        ) {
            // Update camera uniform (view, proj)
            let aspect = self.config.width as f32 / self.config.height as f32;
            let fov = self.view_config.fov_deg.to_radians();
            let proj = Mat4::perspective_rh(fov, aspect, self.view_config.znear, self.view_config.zfar);
            let view_mat = self.camera.view_matrix();
            fn to_arr4(m: Mat4) -> [[f32;4];4] {
                let c = m.to_cols_array();
                [[c[0],c[1],c[2],c[3]],[c[4],c[5],c[6],c[7]],[c[8],c[9],c[10],c[11]],[c[12],c[13],c[14],c[15]]]
            }
            let cam_data = [to_arr4(view_mat), to_arr4(proj)];
            self.queue.write_buffer(cam_buf, 0, bytemuck::cast_slice(&cam_data));

            // Geometry bind group (camera + albedo)
            let bg_ref = match self.geom_bind_group.as_ref() {
                Some(bg) => bg,
                None => {
                    // Create a minimal bind group if missing (shouldn't happen)
                    let sampler = self.albedo_sampler.get_or_insert_with(|| self.device.create_sampler(&wgpu::SamplerDescriptor::default()));
                    let white_tex = self.device.create_texture(&wgpu::TextureDescriptor {
                        label: Some("viewer.geom.albedo.fallback2"),
                        size: wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
                        mip_level_count: 1,
                        sample_count: 1,
                        dimension: wgpu::TextureDimension::D2,
                        format: wgpu::TextureFormat::Rgba8UnormSrgb,
                        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                        view_formats: &[],
                    });
                    self.queue.write_texture(
                        wgpu::ImageCopyTexture { texture: &white_tex, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
                        &[255,255,255,255],
                        wgpu::ImageDataLayout { offset: 0, bytes_per_row: Some(4), rows_per_image: Some(1) },
                        wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
                    );
                    let view = white_tex.create_view(&wgpu::TextureViewDescriptor::default());
                    self.albedo_texture = Some(white_tex);
                    let bgl = self.geom_bind_group_layout.as_ref().unwrap();
                    let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("viewer.gbuf.geom.bg.autogen"),
                        layout: bgl,
                        entries: &[
                            wgpu::BindGroupEntry { binding: 0, resource: cam_buf.as_entire_binding() },
                            wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&view) },
                            wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(sampler) },
                        ],
                    });
                    self.albedo_view = Some(view);
                    self.geom_bind_group = Some(bg);
                    self.geom_bind_group.as_ref().unwrap()
                }
            };

            // Rasterize quad into GBuffer attachments with a depth buffer
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("viewer.gbuf.geom.pass"),
                color_attachments: &[
                    Some(wgpu::RenderPassColorAttachment {
                        view: &gi.gbuffer().normal_view,
                        resolve_target: None,
                        ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::BLACK), store: wgpu::StoreOp::Store },
                    }),
                    Some(wgpu::RenderPassColorAttachment {
                        view: &gi.gbuffer().material_view,
                        resolve_target: None,
                        ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::BLACK), store: wgpu::StoreOp::Store },
                    }),
                    Some(wgpu::RenderPassColorAttachment {
                        view: &gi.gbuffer().depth_view,
                        resolve_target: None,
                        ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::BLACK), store: wgpu::StoreOp::Store },
                    }),
                ],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: zv,
                    depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Clear(1.0), store: wgpu::StoreOp::Store }),
                    stencil_ops: None,
                }),
                occlusion_query_set: None,
                timestamp_writes: None,
            });
            pass.set_pipeline(pipe);
            pass.set_bind_group(0, bg_ref, &[]);
            pass.set_vertex_buffer(0, vb.slice(..));
            if let Some(ib) = self.geom_ib.as_ref() {
                pass.set_index_buffer(ib.slice(..), wgpu::IndexFormat::Uint32);
                pass.draw_indexed(0..self.geom_index_count, 0, 0..1);
            } else {
                pass.draw(0..self.geom_index_count, 0..1);
            }
            drop(pass);

            // Execute effects
            let _ = gi.execute(&self.device, &mut encoder);

            // Composite the material GBuffer to the swapchain
            if let (Some(comp_pl), Some(comp_bgl)) = (self.comp_pipeline.as_ref(), self.comp_bind_group_layout.as_ref()) {
                // Select source texture based on viz_mode
                // If Lit, compute into lit_output first
                if matches!(self.viz_mode, VizMode::Lit) {
                    let env_view = if let Some(ref v) = self.ibl_env_view { v } else { &self.ibl_env_view.as_ref().unwrap() };
                    let env_samp = if let Some(ref s) = self.ibl_sampler { s } else { &self.ibl_sampler.as_ref().unwrap() };
                    let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("viewer.lit.bg"),
                        layout: &self.lit_bind_group_layout,
                        entries: &[
                            wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&gi.gbuffer().normal_view) },
                            wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&gi.gbuffer().material_view) },
                            wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&gi.gbuffer().depth_view) },
                            wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(&self.lit_output_view) },
                            wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::TextureView(env_view) },
                            wgpu::BindGroupEntry { binding: 5, resource: wgpu::BindingResource::Sampler(env_samp) },
                            wgpu::BindGroupEntry { binding: 6, resource: self.lit_uniform.as_entire_binding() },
                        ],
                    });
                    let gx = (self.config.width + 7) / 8; let gy = (self.config.height + 7) / 8;
                    {
                        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("viewer.lit.compute"), timestamp_writes: None });
                        cpass.set_pipeline(&self.lit_pipeline);
                        cpass.set_bind_group(0, &bg, &[]);
                        cpass.dispatch_workgroups(gx, gy, 1);
                    }
                }
                let (mode_u32, src_view) = match self.viz_mode {
                    VizMode::Material => {
                        if self.use_ssao_composite {
                            if let Some(v) = gi.material_with_ao_view() { (0u32, v) } else { (0u32, &gi.gbuffer().material_view) }
                        } else {
                            (0u32, &gi.gbuffer().material_view)
                        }
                    },
                    VizMode::Normal => (1u32, &gi.gbuffer().normal_view),
                    VizMode::Depth => (2u32, &gi.gbuffer().depth_view),
                    VizMode::Gi => {
                        if let Some(v) = gi.gi_debug_view() { (3u32, v) } else { (0u32, &gi.gbuffer().material_view) }
                    },
                    VizMode::Lit => (0u32, &self.lit_output_view),
                };
                // Prepare comp uniform (mode, far)
                let params: [f32; 4] = [
                    mode_u32 as f32,
                    self.viz_depth_max_override.unwrap_or(self.view_config.zfar),
                    0.0,
                    0.0,
                ];
                let buf_ref: &wgpu::Buffer = if let Some(ref ub) = self.comp_uniform {
                    self.queue.write_buffer(ub, 0, bytemuck::cast_slice(&params));
                    ub
                } else {
                    let ub = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("viewer.comp.uniform"),
                        contents: bytemuck::cast_slice(&params),
                        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                    });
                    self.comp_uniform = Some(ub);
                    self.comp_uniform.as_ref().unwrap()
                };
                // Sampler: non-filtering so we can bind depth/non-filterable textures safely
                let comp_sampler = self.device.create_sampler(&wgpu::SamplerDescriptor {
                    label: Some("viewer.comp.sampler"),
                    mag_filter: wgpu::FilterMode::Nearest,
                    min_filter: wgpu::FilterMode::Nearest,
                    mipmap_filter: wgpu::FilterMode::Nearest,
                    ..Default::default()
                });
                let comp_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("viewer.comp.bg"),
                    layout: comp_bgl,
                    entries: &[
                        wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(src_view) },
                        wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&comp_sampler) },
                        wgpu::BindGroupEntry { binding: 2, resource: buf_ref.as_entire_binding() },
                    ],
                });
                // If a snapshot is requested, render the composite to an offscreen texture too
                if self.snapshot_request.is_some() {
                    let snap_tex = self.device.create_texture(&wgpu::TextureDescriptor {
                        label: Some("viewer.snapshot.offscreen"),
                        size: wgpu::Extent3d { width: self.config.width, height: self.config.height, depth_or_array_layers: 1 },
                        mip_level_count: 1,
                        sample_count: 1,
                        dimension: wgpu::TextureDimension::D2,
                        format: self.config.format,
                        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
                        view_formats: &[],
                    });
                    let snap_view = snap_tex.create_view(&wgpu::TextureViewDescriptor::default());
                    let mut pass_snap = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("viewer.comp.pass.snapshot"),
                        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                            view: &snap_view,
                            resolve_target: None,
                            ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color { r: 0.0, g: 0.0, b: 0.0, a: 1.0 }), store: wgpu::StoreOp::Store },
                        })],
                        depth_stencil_attachment: None,
                        timestamp_writes: None,
                        occlusion_query_set: None,
                    });
                    pass_snap.set_pipeline(comp_pl);
                    pass_snap.set_bind_group(0, &comp_bg, &[]);
                    pass_snap.draw(0..3, 0..1);
                    drop(pass_snap);
                    // Store to be read back after submit
                    self.pending_snapshot_tex = Some(snap_tex);
                }
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("viewer.comp.pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &view,
                        resolve_target: None,
                        ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color { r: 0.0, g: 0.0, b: 0.0, a: 1.0 }), store: wgpu::StoreOp::Store },
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                });
                pass.set_pipeline(comp_pl);
                pass.set_bind_group(0, &comp_bg, &[]);
                pass.draw(0..3, 0..1);
                drop(pass);
            }
        }

        // If we didn't composite anything (GI path unavailable), draw fallback to swapchain now
        if !(have_gi && have_pipe && have_cam && have_vb && have_z && have_bgl) {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("viewer.fallback.pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color { r: 0.05, g: 0.0, b: 0.15, a: 1.0 }), store: wgpu::StoreOp::Store },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            pass.set_pipeline(&self.fallback_pipeline);
            pass.draw(0..3, 0..1);
            drop(pass);
        }

        // Submit rendering
        self.queue.submit(std::iter::once(encoder.finish()));

        // Auto-snapshot once if env var is set and we haven't requested yet
        if self.snapshot_request.is_none() && !self.auto_snapshot_done {
            if let Some(ref p) = self.auto_snapshot_path {
                self.snapshot_request = Some(p.clone());
                self.auto_snapshot_done = true;
            }
        }

        // Snapshot if requested (read back the swapchain texture before present)
        if let Some(path) = self.snapshot_request.take() {
            eprintln!("[viewer-debug] snapshot_request taken: {}", path);
            // Prefer offscreen snapshot texture if we rendered one; otherwise fallback to surface texture
            if let Some(tex) = self.pending_snapshot_tex.take() {
                if let Err(e) = self.snapshot_swapchain_to_png(&tex, &path) {
                    eprintln!("Snapshot failed: {}", e);
                } else {
                    println!("Saved snapshot to {}", path);
                }
            } else if let Err(e) = self.snapshot_swapchain_to_png(&output.texture, &path) {
                eprintln!("Snapshot failed: {}", e);
            } else {
                println!("Saved snapshot to {}", path);
            }
        }
        output.present();

        self.frame_count += 1;
        if let Some(fps) = self.fps_counter.tick() {
            let viz = match self.viz_mode { VizMode::Material => "material", VizMode::Normal => "normal", VizMode::Depth => "depth", VizMode::Gi => "gi", VizMode::Lit => "lit" };
            self.window.set_title(&format!(
                "{} | FPS: {:.1} | Mode: {:?} | Viz: {}",
                self.view_config.title,
                fps,
                self.camera.mode(),
                viz
            ));
        }

        Ok(())
    }

    pub fn fps(&self) -> f32 {
        self.fps_counter.fps()
    }
}

// Simple command interface event type carried by the winit EventLoop
#[derive(Debug, Clone)]
enum ViewerCmd {
    GiToggle(&'static str, bool), // ("ssao"|"ssgi"|"ssr", on)
    Snapshot(Option<String>),
    LoadObj(String),
    LoadGltf(String),
    SetViz(String),
    LoadIbl(String),
    SetSsaoRadius(f32),
    SetSsaoIntensity(f32),
    SetSsgiSteps(u32),
    SetSsgiRadius(f32),
    SetSsrMaxSteps(u32),
    SetSsrThickness(f32),
    SetSsgiHalf(bool),
    SetSsgiTemporalAlpha(f32),
    SetSsaoTechnique(u32),
    SetVizDepthMax(f32),
    SetFov(f32),
    SetCamLookAt { eye: [f32;3], target: [f32;3], up: [f32;3] },
    SetSize(u32, u32),
    SetSsaoComposite(bool),
    SetSsaoCompositeMul(f32),
    // SSGI edge-aware upsample controls
    SetSsgiEdges(bool),
    SetSsgiUpsigma(f32),
    SetSsgiNormalExp(f32),
    // Lit viz controls
    SetLitSun(f32),
    SetLitIbl(f32),
    Quit,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum VizMode { Material, Normal, Depth, Gi, Lit }

impl Viewer {
    fn handle_cmd(&mut self, cmd: ViewerCmd) {
        match cmd {
            ViewerCmd::GiToggle(effect, on) => {
                use crate::core::screen_space_effects::ScreenSpaceEffect as SSE;
                let eff = match effect {
                    "ssao" => SSE::SSAO,
                    "ssgi" => SSE::SSGI,
                    "ssr" => SSE::SSR,
                    _ => return,
                };
                if let Some(ref mut gi) = self.gi {
                    if on {
                        if let Err(e) = gi.enable_effect(&self.device, eff) {
                            eprintln!("Failed to enable {:?}: {}", eff, e);
                        } else {
                            println!("Enabled {:?}", eff);
                        }
                    } else {
                        gi.disable_effect(eff);
                        println!("Disabled {:?}", eff);
                    }
                }
            }
            ViewerCmd::Snapshot(path) => {
                let p = path.unwrap_or_else(|| "snapshot.png".to_string());
                self.snapshot_request = Some(p);
            }
            ViewerCmd::LoadObj(path) => {
                match crate::io::obj_read::import_obj(&path) {
                    Ok(obj) => {
                        if let Err(e) = self.upload_mesh(&obj.mesh) {
                            eprintln!("Failed to upload OBJ mesh: {}", e);
                        } else {
                            // If material diffuse_texture exists, try to load it
                            if let Some(mat) = obj.materials.get(0) {
                                if let Some(tex_rel) = &mat.diffuse_texture {
                                    if let Some(base) = Path::new(&path).parent() {
                                        let tex_path = base.join(tex_rel);
                                        let _ = self.load_albedo_texture(tex_path.as_path());
                                    }
                                }
                            }
                            println!("Loaded OBJ geometry: {}", path);
                        }
                    }
                    Err(e) => eprintln!("OBJ import failed: {}", e),
                }
            }
            ViewerCmd::LoadGltf(path) => {
                match crate::io::gltf_read::import_gltf_to_mesh(&path) {
                    Ok(mesh) => {
                        if let Err(e) = self.upload_mesh(&mesh) {
                            eprintln!("Failed to upload glTF mesh: {}", e);
                        } else {
                            println!("Loaded glTF geometry: {}", path);
                        }
                    }
                    Err(e) => eprintln!("glTF import failed: {}", e),
                }
            }
            ViewerCmd::SetViz(mode) => {
                let m = match mode.as_str() {
                    "material"|"mat" => VizMode::Material,
                    "normal"|"normals" => VizMode::Normal,
                    "depth" => VizMode::Depth,
                    "gi" => VizMode::Gi,
                    "lit" => VizMode::Lit,
                    _ => { eprintln!("Unknown viz mode: {}", mode); self.viz_mode },
                };
                self.viz_mode = m;
            }
            ViewerCmd::SetLitSun(v) => {
                self.lit_sun_intensity = v.max(0.0);
                self.update_lit_uniform();
            }
            ViewerCmd::SetLitIbl(v) => {
                self.lit_ibl_intensity = v.max(0.0);
                self.lit_use_ibl = self.lit_ibl_intensity > 0.0;
                self.update_lit_uniform();
            }
            ViewerCmd::LoadIbl(path) => {
                match self.load_ibl(&path) {
                    Ok(_) => println!("Loaded IBL: {}", path),
                    Err(e) => eprintln!("IBL load failed: {}", e),
                }
            }
            ViewerCmd::SetSsaoRadius(v) => {
                if let Some(ref mut gi) = self.gi {
                    gi.update_ssao_settings(&self.queue, |s| s.radius = v);
                }
            }
            ViewerCmd::SetSsaoIntensity(v) => {
                if let Some(ref mut gi) = self.gi {
                    gi.update_ssao_settings(&self.queue, |s| s.intensity = v);
                }
            }
            ViewerCmd::SetSsgiSteps(v) => {
                if let Some(ref mut gi) = self.gi {
                    gi.update_ssgi_settings(&self.queue, |s| s.num_steps = v);
                }
            }
            ViewerCmd::SetSsgiRadius(v) => {
                if let Some(ref mut gi) = self.gi {
                    gi.update_ssgi_settings(&self.queue, |s| s.radius = v);
                }
            }
            ViewerCmd::SetSsrMaxSteps(v) => {
                if let Some(ref mut gi) = self.gi {
                    gi.update_ssr_settings(&self.queue, |s| s.max_steps = v);
                }
            }
            ViewerCmd::SetSsrThickness(v) => {
                if let Some(ref mut gi) = self.gi {
                    gi.update_ssr_settings(&self.queue, |s| s.thickness = v);
                }
            }
            ViewerCmd::SetSsgiHalf(on) => {
                if let Some(ref mut gi) = self.gi {
                    gi.set_ssgi_half_res_with_queue(&self.device, &self.queue, on);
                }
            }
            ViewerCmd::SetSsgiTemporalAlpha(alpha) => {
                if let Some(ref mut gi) = self.gi {
                    gi.update_ssgi_settings(&self.queue, |s| s.temporal_alpha = alpha.clamp(0.0, 1.0));
                }
            }
            ViewerCmd::SetSsaoTechnique(tech) => {
                if let Some(ref mut gi) = self.gi {
                    gi.update_ssao_settings(&self.queue, |s| s.technique = if tech != 0 { 1 } else { 0 });
                }
            }
            ViewerCmd::SetVizDepthMax(v) => {
                self.viz_depth_max_override = Some(v.max(0.001));
            }
            ViewerCmd::SetFov(fov) => {
                self.view_config.fov_deg = fov.max(1.0).min(179.0);
            }
            ViewerCmd::SetCamLookAt { eye, target, up } => {
                let e = Vec3::new(eye[0], eye[1], eye[2]);
                let t = Vec3::new(target[0], target[1], target[2]);
                let u = Vec3::new(up[0], up[1], up[2]);
                self.camera.set_look_at(e, t, u);
            }
            ViewerCmd::SetSize(w, h) => {
                let w = w.max(1); let h = h.max(1);
                let _ = self.window.request_inner_size(PhysicalSize::new(w, h));
            }
            ViewerCmd::SetSsaoComposite(on) => {
                self.use_ssao_composite = on;
            }
            ViewerCmd::SetSsaoCompositeMul(mul) => {
                if let Some(ref mut gi) = self.gi {
                    gi.set_ssao_composite_multiplier(&self.queue, mul);
                }
            }
            ViewerCmd::SetSsgiEdges(on) => {
                if let Some(ref mut gi) = self.gi {
                    gi.update_ssgi_settings(&self.queue, |s| s.use_edge_aware = if on {1} else {0});
                }
            }
            ViewerCmd::SetSsgiUpsigma(v) => {
                if let Some(ref mut gi) = self.gi {
                    gi.update_ssgi_settings(&self.queue, |s| s.upsample_depth_sigma = v);
                }
            }
            ViewerCmd::SetSsgiNormalExp(v) => {
                if let Some(ref mut gi) = self.gi {
                    gi.update_ssgi_settings(&self.queue, |s| s.upsample_normal_exp = v);
                }
            }
            ViewerCmd::Quit => {
                // handled by event loop
            }
        }
    }

    fn upload_mesh(&mut self, mesh: &crate::geometry::MeshBuffers) -> anyhow::Result<()> {
        // Build interleaved vertex buffer: pos(3), nrm(3), uv(2)
        let n = mesh.positions.len();
        let mut verts: Vec<f32> = Vec::with_capacity(n * 8);
        for i in 0..n {
            let p = mesh.positions[i];
            let nrm = if i < mesh.normals.len() { mesh.normals[i] } else { [0.0, 0.0, 1.0] };
            let uv = if i < mesh.uvs.len() { mesh.uvs[i] } else { [0.0, 0.0] };
            verts.extend_from_slice(&[p[0], p[1], p[2], nrm[0], nrm[1], nrm[2], uv[0], uv[1]]);
        }
        let vb = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("viewer.geom.mesh.vb"),
            contents: bytemuck::cast_slice(&verts),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let mut ib_opt: Option<wgpu::Buffer> = None;
        let mut idx_count = 0u32;
        if !mesh.indices.is_empty() {
            let ib = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("viewer.geom.mesh.ib"),
                contents: bytemuck::cast_slice(&mesh.indices),
                usage: wgpu::BufferUsages::INDEX,
            });
            idx_count = mesh.indices.len() as u32;
            ib_opt = Some(ib);
        }
        self.geom_vb = Some(vb);
        self.geom_ib = ib_opt;
        self.geom_index_count = idx_count.max(n as u32);

        // Recreate geometry bind group if albedo is available
        if let (Some(bgl), Some(cam_buf)) = (self.geom_bind_group_layout.as_ref(), self.geom_camera_buffer.as_ref()) {
            // Ensure we have an albedo texture/view; if missing, create a 1x1 white texture
            let sampler = self.albedo_sampler.get_or_insert_with(|| self.device.create_sampler(&wgpu::SamplerDescriptor::default()));
            if self.albedo_view.is_none() {
                let tex = self.device.create_texture(&wgpu::TextureDescriptor {
                    label: Some("viewer.geom.albedo.fallback"),
                    size: wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: wgpu::TextureFormat::Rgba8UnormSrgb,
                    usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                    view_formats: &[],
                });
                self.queue.write_texture(
                    wgpu::ImageCopyTexture { texture: &tex, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
                    &[255, 255, 255, 255],
                    wgpu::ImageDataLayout { offset: 0, bytes_per_row: Some(4), rows_per_image: Some(1) },
                    wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
                );
                let v = tex.create_view(&wgpu::TextureViewDescriptor::default());
                self.albedo_texture = Some(tex);
                self.albedo_view = Some(v);
            }
            let albedo_view_ref = self.albedo_view.as_ref().unwrap();
            let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("viewer.gbuf.geom.bg"),
                layout: bgl,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: cam_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(albedo_view_ref) },
                    wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(sampler) },
                ],
            });
            self.geom_bind_group = Some(bg);
        }
        Ok(())
    }

    fn load_albedo_texture(&mut self, path: &Path) -> anyhow::Result<()> {
        match image::open(path) {
            Ok(img) => {
                let rgba = img.to_rgba8();
                let (w, h) = rgba.dimensions();
                let tex = self.device.create_texture(&wgpu::TextureDescriptor {
                    label: Some("viewer.geom.albedo.tex.file"),
                    size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: wgpu::TextureFormat::Rgba8UnormSrgb,
                    usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                    view_formats: &[],
                });
                self.queue.write_texture(
                    wgpu::ImageCopyTexture { texture: &tex, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
                    &rgba,
                    wgpu::ImageDataLayout { offset: 0, bytes_per_row: Some(w * 4), rows_per_image: Some(h) },
                    wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
                );
                let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
                let sampler = self.albedo_sampler.get_or_insert_with(|| self.device.create_sampler(&wgpu::SamplerDescriptor::default()));
                self.albedo_texture = Some(tex);
                self.albedo_view = Some(view);
                if let (Some(bgl), Some(cam_buf)) = (self.geom_bind_group_layout.as_ref(), self.geom_camera_buffer.as_ref()) {
                    let view_ref = self.albedo_view.as_ref().unwrap();
                    let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("viewer.gbuf.geom.bg"),
                        layout: bgl,
                        entries: &[
                            wgpu::BindGroupEntry { binding: 0, resource: cam_buf.as_entire_binding() },
                            wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(view_ref) },
                            wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(sampler) },
                        ],
                    });
                    self.geom_bind_group = Some(bg);
                }
                Ok(())
            }
            Err(e) => Err(anyhow::anyhow!("failed to open texture {:?}: {}", path, e)),
        }
    }

    fn snapshot_swapchain_to_png(
        &self,
        texture: &wgpu::Texture,
        path: &str,
    ) -> anyhow::Result<()> {
        let data = crate::renderer::readback::read_texture_tight(
            &self.device,
            &self.queue,
            texture,
            (self.config.width, self.config.height),
            self.config.format,
        )?;
        match self.config.format {
            wgpu::TextureFormat::Rgba8Unorm | wgpu::TextureFormat::Rgba8UnormSrgb => {
                crate::util::image_write::write_png_rgba8(
                    std::path::Path::new(path),
                    &data,
                    self.config.width,
                    self.config.height,
                )?;
                Ok(())
            }
            wgpu::TextureFormat::Bgra8Unorm | wgpu::TextureFormat::Bgra8UnormSrgb => {
                // Convert BGRA -> RGBA in place to reuse PNG writer
                let mut rgba = data;
                for px in rgba.chunks_exact_mut(4) {
                    let b = px[0];
                    let r = px[2];
                    px[0] = r; // R
                    // G stays px[1]
                    px[2] = b; // B
                    // A stays px[3]
                }
                crate::util::image_write::write_png_rgba8(
                    std::path::Path::new(path),
                    &rgba,
                    self.config.width,
                    self.config.height,
                )?;
                Ok(())
            }
            other => anyhow::bail!("unsupported format {:?} for snapshot", other),
        }
    }
}

pub fn run_viewer(config: ViewerConfig) -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();

    let event_loop: EventLoop<ViewerCmd> = winit::event_loop::EventLoopBuilder::<ViewerCmd>::with_user_event().build()?;
    let window = Arc::new(
        WindowBuilder::new()
            .with_title(&config.title)
            .with_inner_size(PhysicalSize::new(config.width, config.height))
            .build(&event_loop)?,
    );

    println!("forge3d Interactive Viewer");
    println!("Controls:");
    println!("  Tab       - Toggle camera mode (Orbit/FPS)");
    println!("  Orbit mode:");
    println!("    Drag    - Rotate camera");
    println!("    Scroll  - Zoom in/out");
    println!("  FPS mode:");
    println!("    WASD    - Move forward/left/backward/right");
    println!("    Q/E     - Move down/up");
    println!("    Mouse   - Look around (hold left button)");
    println!("    Shift   - Move faster");
    println!("  Esc       - Exit");

    // Spawn input thread for simple command interface
    let proxy = event_loop.create_proxy();
    // Collect any initial commands provided by CLI flags; we'll apply them after viewer is created
    let mut pending_cmds: Vec<ViewerCmd> = Vec::new();
    if let Some(cmds) = INITIAL_CMDS.get() {
        for line in cmds.iter() {
            let l = line.trim().to_lowercase();
            if l.is_empty() { continue; }
            if l.starts_with(":gi") || l.starts_with("gi ") {
                let toks: Vec<&str> = l.trim_start_matches(":").split_whitespace().collect();
                if toks.len() >= 3 {
                    let eff = match toks[1] { "ssao"|"ssgi"|"ssr" => toks[1], _ => { println!("Unknown effect '{}'", toks[1]); continue; } };
                    let on = match toks[2] { "on"|"1"|"true" => true, "off"|"0"|"false" => false, _ => { println!("Unknown state '{}', expected on/off", toks[2]); continue; } };
                    let eff_str = match eff {"ssao"=>"ssao","ssgi"=>"ssgi","ssr"=>"ssr",_=>"ssao"};
                    pending_cmds.push(ViewerCmd::GiToggle(eff_str, on));
                } else { println!("Usage: :gi <ssao|ssgi|ssr> <on|off>"); }
            } else if l.starts_with(":snapshot") || l.starts_with("snapshot") {
                let path = l.split_whitespace().nth(1).map(|s| s.to_string());
                pending_cmds.push(ViewerCmd::Snapshot(path));
            } else if l.starts_with(":viz") || l.starts_with("viz ") {
                if let Some(mode) = l.split_whitespace().nth(1) { pending_cmds.push(ViewerCmd::SetViz(mode.to_string())); }
            } else if l.starts_with(":size") || l.starts_with("size ") {
                if let (Some(ws), Some(hs)) = (l.split_whitespace().nth(1), l.split_whitespace().nth(2)) {
                    if let (Ok(w), Ok(h)) = (ws.parse::<u32>(), hs.parse::<u32>()) { pending_cmds.push(ViewerCmd::SetSize(w, h)); }
                }
            } else if l.starts_with(":fov") || l.starts_with("fov ") {
                if let Some(val) = l.split_whitespace().nth(1).and_then(|s| s.parse::<f32>().ok()) { pending_cmds.push(ViewerCmd::SetFov(val)); }
            } else if l.starts_with(":cam-lookat") || l.starts_with("cam-lookat ") {
                let toks: Vec<&str> = l.split_whitespace().collect();
                if toks.len() == 7 || toks.len() == 10 {
                    let ex = toks[1].parse::<f32>().unwrap_or(0.0);
                    let ey = toks[2].parse::<f32>().unwrap_or(0.0);
                    let ez = toks[3].parse::<f32>().unwrap_or(0.0);
                    let tx = toks[4].parse::<f32>().unwrap_or(0.0);
                    let ty = toks[5].parse::<f32>().unwrap_or(0.0);
                    let tz = toks[6].parse::<f32>().unwrap_or(0.0);
                    let (ux,uy,uz) = if toks.len() == 10 {
                        (
                            toks[7].parse::<f32>().unwrap_or(0.0),
                            toks[8].parse::<f32>().unwrap_or(1.0),
                            toks[9].parse::<f32>().unwrap_or(0.0)
                        )
                    } else { (0.0, 1.0, 0.0) };
                    pending_cmds.push(ViewerCmd::SetCamLookAt { eye: [ex,ey,ez], target: [tx,ty,tz], up: [ux,uy,uz] });
                }
            } else if l.starts_with(":ibl") || l.starts_with("ibl ") {
                if let Some(path) = l.split_whitespace().nth(1) { pending_cmds.push(ViewerCmd::LoadIbl(path.to_string())); }
            } else if l.starts_with(":viz-depth-max") || l.starts_with("viz-depth-max ") {
                if let Some(val) = l.split_whitespace().nth(1).and_then(|s| s.parse::<f32>().ok()) { pending_cmds.push(ViewerCmd::SetVizDepthMax(val)); }
            } else if l.starts_with(":ssao-composite") || l.starts_with("ssao-composite ") {
                if let Some(tok) = l.split_whitespace().nth(1) { pending_cmds.push(ViewerCmd::SetSsaoComposite(matches!(tok, "on"|"1"|"true"))); }
            } else if l.starts_with(":ssao-mul") || l.starts_with("ssao-mul ") {
                if let Some(val) = l.split_whitespace().nth(1).and_then(|s| s.parse::<f32>().ok()) { pending_cmds.push(ViewerCmd::SetSsaoCompositeMul(val)); }
            } else if l.starts_with(":ssgi-edges") || l.starts_with("ssgi-edges ") {
                if let Some(tok) = l.split_whitespace().nth(1) { pending_cmds.push(ViewerCmd::SetSsgiEdges(matches!(tok, "on"|"1"|"true"))); }
            } else if l.starts_with(":ssgi-upsigma") || l.starts_with("ssgi-upsigma ") {
                if let Some(val) = l.split_whitespace().nth(1).and_then(|s| s.parse::<f32>().ok()) { pending_cmds.push(ViewerCmd::SetSsgiUpsigma(val)); }
            } else if l.starts_with(":ssgi-normexp") || l.starts_with("ssgi-normexp ") {
                if let Some(val) = l.split_whitespace().nth(1).and_then(|s| s.parse::<f32>().ok()) { pending_cmds.push(ViewerCmd::SetSsgiNormalExp(val)); }
            } else if l.starts_with(":ssao-radius") || l.starts_with("ssao-radius ") {
                if let Some(val) = l.split_whitespace().nth(1).and_then(|s| s.parse::<f32>().ok()) { pending_cmds.push(ViewerCmd::SetSsaoRadius(val)); }
            } else if l.starts_with(":ssao-intensity") || l.starts_with("ssao-intensity ") {
                if let Some(val) = l.split_whitespace().nth(1).and_then(|s| s.parse::<f32>().ok()) { pending_cmds.push(ViewerCmd::SetSsaoIntensity(val)); }
            } else if l.starts_with(":lit-sun") || l.starts_with("lit-sun ") {
                if let Some(val) = l.split_whitespace().nth(1).and_then(|s| s.parse::<f32>().ok()) { pending_cmds.push(ViewerCmd::SetLitSun(val)); }
            } else if l.starts_with(":lit-ibl") || l.starts_with("lit-ibl ") {
                if let Some(val) = l.split_whitespace().nth(1).and_then(|s| s.parse::<f32>().ok()) { pending_cmds.push(ViewerCmd::SetLitIbl(val)); }
            } else if l.starts_with(":ssao-technique") || l.starts_with("ssao-technique ") {
                if let Some(tok) = l.split_whitespace().nth(1) { let tech = match tok { "gtao"|"1" => 1u32, _ => 0u32 }; pending_cmds.push(ViewerCmd::SetSsaoTechnique(tech)); }
            } else if l.starts_with(":ssgi-steps") || l.starts_with("ssgi-steps ") {
                if let Some(val) = l.split_whitespace().nth(1).and_then(|s| s.parse::<u32>().ok()) { pending_cmds.push(ViewerCmd::SetSsgiSteps(val)); }
            } else if l.starts_with(":ssgi-radius") || l.starts_with("ssgi-radius ") {
                if let Some(val) = l.split_whitespace().nth(1).and_then(|s| s.parse::<f32>().ok()) { pending_cmds.push(ViewerCmd::SetSsgiRadius(val)); }
            } else if l.starts_with(":ssr-max-steps") || l.starts_with("ssr-max-steps ") {
                if let Some(val) = l.split_whitespace().nth(1).and_then(|s| s.parse::<u32>().ok()) { pending_cmds.push(ViewerCmd::SetSsrMaxSteps(val)); }
            } else if l.starts_with(":ssr-thickness") || l.starts_with("ssr-thickness ") {
                if let Some(val) = l.split_whitespace().nth(1).and_then(|s| s.parse::<f32>().ok()) { pending_cmds.push(ViewerCmd::SetSsrThickness(val)); }
            } else if l.starts_with(":ssgi-half") || l.starts_with("ssgi-half ") {
                if let Some(tok) = l.split_whitespace().nth(1) { pending_cmds.push(ViewerCmd::SetSsgiHalf(matches!(tok, "on"|"1"|"true"))); }
            } else if l.starts_with(":ssgi-temporal-alpha") || l.starts_with("ssgi-temporal-alpha ") {
                if let Some(val) = l.split_whitespace().nth(1).and_then(|s| s.parse::<f32>().ok()) { pending_cmds.push(ViewerCmd::SetSsgiTemporalAlpha(val)); }
            }
        }
    }
    std::thread::spawn(move || {
        let stdin = std::io::stdin();
        let mut iter = stdin.lock().lines();
        while let Some(Ok(line)) = iter.next() {
            let l = line.trim().to_lowercase();
            if l.is_empty() { continue; }
            if l.starts_with(":gi") || l.starts_with("gi ") {
                let toks: Vec<&str> = l.trim_start_matches(":").split_whitespace().collect();
                if toks.len() >= 3 {
                    let eff = match toks[1] { "ssao"|"ssgi"|"ssr" => toks[1], _ => { println!("Unknown effect '{}'", toks[1]); continue; } };
                    let on = match toks[2] { "on"|"1"|"true" => true, "off"|"0"|"false" => false, _ => { println!("Unknown state '{}', expected on/off", toks[2]); continue; } };
                    let _ = proxy.send_event(ViewerCmd::GiToggle(match eff {"ssao"=>"ssao","ssgi"=>"ssgi","ssr"=>"ssr",_=>"ssao"}, on));
                } else {
                    println!("Usage: :gi <ssao|ssgi|ssr> <on|off>");
                }
            } else if l.starts_with(":snapshot") || l.starts_with("snapshot") {
                let path = l.split_whitespace().nth(1).map(|s| s.to_string());
                let _ = proxy.send_event(ViewerCmd::Snapshot(path));
            } else if l.starts_with(":obj") || l.starts_with("obj ") {
                if let Some(path) = l.split_whitespace().nth(1) {
                    let _ = proxy.send_event(ViewerCmd::LoadObj(path.to_string()));
                } else {
                    println!("Usage: :obj <path>");
                }
            } else if l.starts_with(":gltf") || l.starts_with("gltf ") {
                if let Some(path) = l.split_whitespace().nth(1) {
                    let _ = proxy.send_event(ViewerCmd::LoadGltf(path.to_string()));
                } else {
                    println!("Usage: :gltf <path>");
                }
            } else if l.starts_with(":viz") || l.starts_with("viz ") {
                if let Some(mode) = l.split_whitespace().nth(1) {
                    let _ = proxy.send_event(ViewerCmd::SetViz(mode.to_string()));
                } else {
                    println!("Usage: :viz <material|normal|depth|gi|lit>");
                }
            } else if l.starts_with(":size") || l.starts_with("size ") {
                if let (Some(ws), Some(hs)) = (l.split_whitespace().nth(1), l.split_whitespace().nth(2)) {
                    if let (Ok(w), Ok(h)) = (ws.parse::<u32>(), hs.parse::<u32>()) {
                        let _ = proxy.send_event(ViewerCmd::SetSize(w, h));
                    } else { println!("Usage: :size <w> <h>"); }
                } else { println!("Usage: :size <w> <h>"); }
            } else if l.starts_with(":fov") || l.starts_with("fov ") {
                if let Some(val) = l.split_whitespace().nth(1).and_then(|s| s.parse::<f32>().ok()) {
                    let _ = proxy.send_event(ViewerCmd::SetFov(val));
                } else { println!("Usage: :fov <degrees>"); }
            } else if l.starts_with(":cam-lookat") || l.starts_with("cam-lookat ") {
                let toks: Vec<&str> = l.split_whitespace().collect();
                if toks.len() == 7 || toks.len() == 10 {
                    let ex = toks[1].parse::<f32>().unwrap_or(0.0);
                    let ey = toks[2].parse::<f32>().unwrap_or(0.0);
                    let ez = toks[3].parse::<f32>().unwrap_or(0.0);
                    let tx = toks[4].parse::<f32>().unwrap_or(0.0);
                    let ty = toks[5].parse::<f32>().unwrap_or(0.0);
                    let tz = toks[6].parse::<f32>().unwrap_or(0.0);
                    let (ux,uy,uz) = if toks.len() == 10 {
                        (
                            toks[7].parse::<f32>().unwrap_or(0.0),
                            toks[8].parse::<f32>().unwrap_or(1.0),
                            toks[9].parse::<f32>().unwrap_or(0.0)
                        )
                    } else { (0.0, 1.0, 0.0) };
                    let _ = proxy.send_event(ViewerCmd::SetCamLookAt { eye: [ex,ey,ez], target: [tx,ty,tz], up: [ux,uy,uz] });
                } else { println!("Usage: :cam-lookat ex ey ez tx ty tz [ux uy uz]"); }
            } else if l.starts_with(":ssao-composite") || l.starts_with("ssao-composite ") {
                if let Some(tok) = l.split_whitespace().nth(1) {
                    let on = matches!(tok, "on"|"1"|"true");
                    let _ = proxy.send_event(ViewerCmd::SetSsaoComposite(on));
                } else { println!("Usage: :ssao-composite <on|off>"); }
            } else if l.starts_with(":ssao-mul") || l.starts_with("ssao-mul ") {
                if let Some(val) = l.split_whitespace().nth(1).and_then(|s| s.parse::<f32>().ok()) {
                    let _ = proxy.send_event(ViewerCmd::SetSsaoCompositeMul(val));
                } else { println!("Usage: :ssao-mul <0..1>"); }
            } else if l.starts_with(":ssgi-edges") || l.starts_with("ssgi-edges ") {
                if let Some(tok) = l.split_whitespace().nth(1) {
                    let on = matches!(tok, "on"|"1"|"true");
                    let _ = proxy.send_event(ViewerCmd::SetSsgiEdges(on));
                } else { println!("Usage: :ssgi-edges <on|off>"); }
            } else if l.starts_with(":ssgi-upsigma") || l.starts_with("ssgi-upsigma ") {
                if let Some(val) = l.split_whitespace().nth(1).and_then(|s| s.parse::<f32>().ok()) {
                    let _ = proxy.send_event(ViewerCmd::SetSsgiUpsigma(val));
                } else { println!("Usage: :ssgi-upsigma <float>"); }
            } else if l.starts_with(":ssgi-normexp") || l.starts_with("ssgi-normexp ") {
                if let Some(val) = l.split_whitespace().nth(1).and_then(|s| s.parse::<f32>().ok()) {
                    let _ = proxy.send_event(ViewerCmd::SetSsgiNormalExp(val));
                } else { println!("Usage: :ssgi-normexp <float>"); }
            } else if l.starts_with(":ibl") || l.starts_with("ibl ") {
                if let Some(path) = l.split_whitespace().nth(1) {
                    let _ = proxy.send_event(ViewerCmd::LoadIbl(path.to_string()));
                } else {
                    println!("Usage: :ibl <path.hdr|path.exr>");
                }
            } else if l.starts_with(":ssao-radius") || l.starts_with("ssao-radius ") {
                if let Some(val) = l.split_whitespace().nth(1).and_then(|s| s.parse::<f32>().ok()) {
                    let _ = proxy.send_event(ViewerCmd::SetSsaoRadius(val));
                } else { println!("Usage: :ssao-radius <float>"); }
            } else if l.starts_with(":ssao-intensity") || l.starts_with("ssao-intensity ") {
                if let Some(val) = l.split_whitespace().nth(1).and_then(|s| s.parse::<f32>().ok()) {
                    let _ = proxy.send_event(ViewerCmd::SetSsaoIntensity(val));
                } else { println!("Usage: :ssao-intensity <float>"); }
            } else if l.starts_with(":viz-depth-max") || l.starts_with("viz-depth-max ") {
                if let Some(val) = l.split_whitespace().nth(1).and_then(|s| s.parse::<f32>().ok()) {
                    let _ = proxy.send_event(ViewerCmd::SetVizDepthMax(val));
                } else { println!("Usage: :viz-depth-max <float>"); }
            } else if l.starts_with(":ssgi-steps") || l.starts_with("ssgi-steps ") {
                if let Some(val) = l.split_whitespace().nth(1).and_then(|s| s.parse::<u32>().ok()) {
                    let _ = proxy.send_event(ViewerCmd::SetSsgiSteps(val));
                } else { println!("Usage: :ssgi-steps <u32>"); }
            } else if l.starts_with(":ssgi-radius") || l.starts_with("ssgi-radius ") {
                if let Some(val) = l.split_whitespace().nth(1).and_then(|s| s.parse::<f32>().ok()) {
                    let _ = proxy.send_event(ViewerCmd::SetSsgiRadius(val));
                } else { println!("Usage: :ssgi-radius <float>"); }
            } else if l.starts_with(":ssr-max-steps") || l.starts_with("ssr-max-steps ") {
                if let Some(val) = l.split_whitespace().nth(1).and_then(|s| s.parse::<u32>().ok()) {
                    let _ = proxy.send_event(ViewerCmd::SetSsrMaxSteps(val));
                } else { println!("Usage: :ssr-max-steps <u32>"); }
            } else if l.starts_with(":ssr-thickness") || l.starts_with("ssr-thickness ") {
                if let Some(val) = l.split_whitespace().nth(1).and_then(|s| s.parse::<f32>().ok()) {
                    let _ = proxy.send_event(ViewerCmd::SetSsrThickness(val));
                } else { println!("Usage: :ssr-thickness <float>"); }
            } else if l.starts_with(":ssgi-half") || l.starts_with("ssgi-half ") {
                if let Some(tok) = l.split_whitespace().nth(1) {
                    let on = matches!(tok, "on"|"1"|"true");
                    let _ = proxy.send_event(ViewerCmd::SetSsgiHalf(on));
                } else { println!("Usage: :ssgi-half <on|off|1|0>"); }
            } else if l.starts_with(":ssgi-temporal-alpha") || l.starts_with("ssgi-temporal-alpha ") {
                if let Some(val) = l.split_whitespace().nth(1).and_then(|s| s.parse::<f32>().ok()) {
                    let _ = proxy.send_event(ViewerCmd::SetSsgiTemporalAlpha(val));
                } else { println!("Usage: :ssgi-temporal-alpha <float 0..1>"); }
            } else if l.starts_with(":lit-sun") || l.starts_with("lit-sun ") {
                if let Some(val) = l.split_whitespace().nth(1).and_then(|s| s.parse::<f32>().ok()) {
                    let _ = proxy.send_event(ViewerCmd::SetLitSun(val));
                } else { println!("Usage: :lit-sun <float>"); }
            } else if l.starts_with(":lit-ibl") || l.starts_with("lit-ibl ") {
                if let Some(val) = l.split_whitespace().nth(1).and_then(|s| s.parse::<f32>().ok()) {
                    let _ = proxy.send_event(ViewerCmd::SetLitIbl(val));
                } else { println!("Usage: :lit-ibl <float>"); }
            } else if l.starts_with(":ssao-technique") || l.starts_with("ssao-technique ") {
                if let Some(tok) = l.split_whitespace().nth(1) {
                    let tech = match tok { "gtao"|"1" => 1u32, _ => 0u32 };
                    let _ = proxy.send_event(ViewerCmd::SetSsaoTechnique(tech));
                } else { println!("Usage: :ssao-technique <ssao|gtao>"); }
            } else if l == ":quit" || l == "quit" || l == ":exit" || l == "exit" {
                let _ = proxy.send_event(ViewerCmd::Quit);
                break;
            } else {
                println!(
                    "Commands: \n  :gi <ssao|ssgi|ssr> <on|off>\n  :viz <material|normal|depth|gi|lit>\n  :viz-depth-max <float>\n  :ibl <path.hdr|path.exr>\n  :snapshot [path]\n  :obj <path> | :gltf <path>\n  Lit:  :lit-sun <float> | :lit-ibl <float>\n  SSAO: :ssao-technique <ssao|gtao> | :ssao-radius <f> | :ssao-intensity <f> | :ssao-composite <on|off> | :ssao-mul <0..1>\n  SSGI: :ssgi-steps <u32> | :ssgi-radius <f> | :ssgi-half <on|off> | :ssgi-temporal-alpha <0..1> | :ssgi-edges <on|off> | :ssgi-upsigma <f> | :ssgi-normexp <f>\n  SSR:  :ssr-max-steps <u32> | :ssr-thickness <f>\n  :quit"
                );
            }
        }
    });

    // Create viewer in blocking manner
    let mut viewer_opt: Option<Viewer> = None;
    let mut last_frame = Instant::now();

    event_loop.run(move |event, elwt| {
        match event {
            Event::Resumed => {
                // Initialize viewer on resume (required for some platforms)
                if viewer_opt.is_none() {
                    let v = pollster::block_on(Viewer::new(Arc::clone(&window), config.clone()));
                    match v {
                        Ok(v) => {
                            viewer_opt = Some(v);
                            last_frame = Instant::now();
                            // Apply any pending commands from CLI now that viewer exists
                            for cmd in pending_cmds.drain(..) {
                                if let Some(viewer) = viewer_opt.as_mut() {
                                    viewer.handle_cmd(cmd);
                                }
                            }
                        }
                        Err(e) => {
                            eprintln!("Failed to create viewer: {}", e);
                            elwt.exit();
                        }
                    }
                }
            }
            Event::WindowEvent {
                ref event,
                window_id,
            } if window_id == window.id() && !matches!(event, WindowEvent::RedrawRequested) => {
                if let Some(viewer) = viewer_opt.as_mut() {
                    if !viewer.handle_input(event) {
                        match event {
                            WindowEvent::CloseRequested => {
                                elwt.exit();
                            }
                            WindowEvent::KeyboardInput {
                                event: key_event, ..
                            } => {
                                if key_event.state == ElementState::Pressed {
                                    if let PhysicalKey::Code(KeyCode::Escape) =
                                        key_event.physical_key
                                    {
                                        elwt.exit();
                                    }
                                }
                            }
                            WindowEvent::Resized(physical_size) => {
                                viewer.resize(*physical_size);
                            }
                            _ => {}
                        }
                    }
                }
            }
            Event::AboutToWait => {
                window.request_redraw();
            }
            Event::WindowEvent { event: WindowEvent::RedrawRequested, window_id } if window_id == window.id() => {
                if let Some(viewer) = viewer_opt.as_mut() {
                    let now = Instant::now();
                    let dt = (now - last_frame).as_secs_f32();
                    last_frame = now;

                    viewer.update(dt);
                    match viewer.render() {
                        Ok(_) => {}
                        Err(wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated) => {
                            viewer.resize(viewer.window().inner_size())
                        }
                        Err(wgpu::SurfaceError::OutOfMemory) => {
                            eprintln!("Out of memory!");
                            elwt.exit();
                        }
                        Err(wgpu::SurfaceError::Timeout) => {
                            eprintln!("Surface timeout!");
                        }
                    }
                }
            }
            Event::UserEvent(cmd) => {
                match cmd {
                    ViewerCmd::Quit => {
                        elwt.exit();
                    }
                    other => {
                        if let Some(viewer) = viewer_opt.as_mut() {
                            viewer.handle_cmd(other);
                        }
                    }
                }
            }
            _ => {}
        }
    })?;

    Ok(())
}
