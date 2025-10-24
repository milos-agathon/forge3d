// src/viewer/mod.rs
// Workstream I1: Interactive windowed viewer for forge3d
// - Creates window with winit 0.29
// - Handles input events (mouse, keyboard)
// - Renders frames at 60 FPS
// - Orbit and FPS camera modes

pub mod camera_controller;
mod image_viewer_stub;
mod textured_mesh;

use camera_controller::{CameraController, CameraMode};
<<<<<<< HEAD
pub use image_viewer_stub::run_image_viewer;
pub use textured_mesh::run_mesh_viewer_with_texture;
use winit::{
    event::*,
    event_loop::EventLoop,
    window::{Window, WindowBuilder},
    dpi::PhysicalSize,
    keyboard::{KeyCode, PhysicalKey},
};
use wgpu::{Instance, Surface, Device, Queue, SurfaceConfiguration, Buffer, RenderPipeline, BindGroupLayout, BindGroup, TextureView, Texture};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use wgpu::util::DeviceExt;
use glam::Mat4;
use std::sync::mpsc::Sender;

#[derive(Debug, Clone)]
pub struct CameraState {
    pub eye: [f32; 3],
    pub target: [f32; 3],
    pub distance: f32,
    pub theta: f32,  // degrees
    pub phi: f32,    // degrees
    pub fov: f32,    // degrees
}

#[derive(Debug, Clone)]
pub enum ViewerCommand {
    Export(String),
    SetCamera { distance: Option<f32>, theta: Option<f32>, phi: Option<f32> },
    Snapshot { path: String, width: Option<u32>, height: Option<u32> },
    GetCamera { response_tx: std::sync::mpsc::Sender<CameraState> },
}

static VIEWER_SENDER: Mutex<Option<Sender<ViewerCommand>>> = Mutex::new(None);

pub fn viewer_send_command(cmd: ViewerCommand) -> Result<(), String> {
    let guard = VIEWER_SENDER.lock().map_err(|e| format!("Lock error: {}", e))?;
    if let Some(tx) = &*guard {
        tx.send(cmd).map_err(|e| format!("Send error: {}", e))?;
        Ok(())
    } else {
        Err("Viewer not running".to_string())
    }
}
=======
use std::sync::Arc;
use std::time::{Duration, Instant};
use wgpu::{Device, Instance, Queue, Surface, SurfaceConfiguration};
use winit::{
    dpi::PhysicalSize,
    event::*,
    event_loop::EventLoop,
    keyboard::{KeyCode, PhysicalKey},
    window::{Window, WindowBuilder},
};
>>>>>>> 136f57c (PBR + IBL terrain with Triplanar + POM pipeline production-ready)

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

// Simple mesh shader with MVP transform
const MESH_SHADER: &str = r#"
struct Camera {
    mvp: mat4x4<f32>,
};
@group(0) @binding(0) var<uniform> u_camera: Camera;

struct VertexInput {
    @location(0) position: vec3<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
};

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.clip_position = u_camera.mvp * vec4<f32>(in.position, 1.0);
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // Output screen-space gradient to verify shader runs
    let ndc = in.clip_position.xy / in.clip_position.w;
    let uv = (ndc + 1.0) * 0.5;  // Map [-1,1] to [0,1]
    return vec4<f32>(uv.x, uv.y, 0.0, 1.0);
}
"#;

// Textured mesh shader with UV coordinates
const TEXTURED_MESH_SHADER: &str = r#"
struct Camera {
    mvp: mat4x4<f32>,
};
@group(0) @binding(0) var<uniform> u_camera: Camera;
@group(1) @binding(0) var t_texture: texture_2d<f32>;
@group(1) @binding(1) var s_texture: sampler;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) uv: vec2<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.clip_position = u_camera.mvp * vec4<f32>(in.position, 1.0);
    out.uv = in.uv;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return textureSample(t_texture, s_texture, in.uv);
}
"#;

// OpenGL to wgpu/Vulkan depth conversion matrix
// OpenGL NDC has Z in [-1, 1], wgpu/Vulkan expects [0, 1]
// This matrix maps Z: [-1, 1] -> [0, 1] via z' = z*0.5 + 0.5
const OPENGL_TO_WGPU_MATRIX: Mat4 = Mat4::from_cols(
    glam::Vec4::new(1.0, 0.0, 0.0, 0.0),
    glam::Vec4::new(0.0, 1.0, 0.0, 0.0),
    glam::Vec4::new(0.0, 0.0, 0.5, 0.0),
    glam::Vec4::new(0.0, 0.0, 0.5, 1.0),
);

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
    // Mesh rendering resources
    mesh_pipeline: Option<RenderPipeline>,
    mesh_vertex_buffer: Option<Buffer>,
    mesh_index_buffer: Option<Buffer>,
    mesh_index_count: u32,
    camera_buffer: Option<Buffer>,
    camera_bind_group_layout: Option<BindGroupLayout>,
    camera_bind_group: Option<BindGroup>,
    depth_texture: Option<Texture>,
    depth_view: Option<TextureView>,
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
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
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
            mesh_pipeline: None,
            mesh_vertex_buffer: None,
            mesh_index_buffer: None,
            mesh_index_count: 0,
            camera_buffer: None,
            camera_bind_group_layout: None,
            camera_bind_group: None,
            depth_texture: None,
            depth_view: None,
        })
    }

    pub fn window(&self) -> &Window {
        &self.window
    }

    pub fn resize(&mut self, new_size: PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            self.config.width = new_size.width;
            self.config.height = new_size.height;
            self.surface.configure(&self.device, &self.config);
            
            // Recreate depth buffer
            if self.depth_texture.is_some() {
                let depth_texture = self.device.create_texture(&wgpu::TextureDescriptor {
                    label: Some("depth_texture"),
                    size: wgpu::Extent3d {
                        width: new_size.width,
                        height: new_size.height,
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: wgpu::TextureFormat::Depth32Float,
                    usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                    view_formats: &[],
                });
                let depth_view = depth_texture.create_view(&wgpu::TextureViewDescriptor::default());
                self.depth_texture = Some(depth_texture);
                self.depth_view = Some(depth_view);
            }
        }
    }
    
    pub fn init_mesh(&mut self, vertices: &[f32], indices: &[u32]) -> Result<(), String> {
        if vertices.len() % 3 != 0 {
            return Err("Vertex data length must be multiple of 3".to_string());
        }
        if indices.is_empty() {
            return Err("Index data cannot be empty".to_string());
        }
        
        eprintln!("[DEBUG] init_mesh: {} vertices, {} indices", vertices.len() / 3, indices.len());
        if vertices.len() >= 3 {
            eprintln!("[DEBUG] First vertex: [{:.3}, {:.3}, {:.3}]", vertices[0], vertices[1], vertices[2]);
        }
        if indices.len() >= 3 {
            eprintln!("[DEBUG] First triangle: [{}, {}, {}]", indices[0], indices[1], indices[2]);
        }
        
        // Create vertex buffer
        let vertex_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("mesh_vertex_buffer"),
            contents: bytemuck::cast_slice(vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });
        
        // Create index buffer
        let index_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("mesh_index_buffer"),
            contents: bytemuck::cast_slice(indices),
            usage: wgpu::BufferUsages::INDEX,
        });
        
        // Create camera uniform buffer
        let camera_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("camera_buffer"),
            size: 64, // mat4x4<f32>
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        
        // Create bind group layout
        let camera_bind_group_layout = self.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("camera_bind_group_layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        
        // Create bind group
        let camera_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("camera_bind_group"),
            layout: &camera_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: camera_buffer.as_entire_binding(),
            }],
        });
        
        // Create depth texture
        let depth_texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("depth_texture"),
            size: wgpu::Extent3d {
                width: self.config.width,
                height: self.config.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let depth_view = depth_texture.create_view(&wgpu::TextureViewDescriptor::default());
        
        // Create shader module
        println!("[INIT] Creating shader module...");
        let shader = self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("mesh_shader"),
            source: wgpu::ShaderSource::Wgsl(MESH_SHADER.into()),
        });
        println!("[INIT] Shader module created successfully");
        
        // Create pipeline layout
        let pipeline_layout = self.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("mesh_pipeline_layout"),
            bind_group_layouts: &[&camera_bind_group_layout],
            push_constant_ranges: &[],
        });
        
        // Create render pipeline
        let mesh_pipeline = self.device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("mesh_pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: 12, // 3 * f32
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &[wgpu::VertexAttribute {
                        format: wgpu::VertexFormat::Float32x3,
                        offset: 0,
                        shader_location: 0,
                    }],
                }],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format: self.config.format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                unclipped_depth: false,
                polygon_mode: wgpu::PolygonMode::Fill,
                conservative: false,
            },
            depth_stencil: None,  // Temporarily disabled for debugging
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
        });
        
        self.mesh_vertex_buffer = Some(vertex_buffer);
        self.mesh_index_buffer = Some(index_buffer);
        self.mesh_index_count = indices.len() as u32;
        self.camera_buffer = Some(camera_buffer);
        self.camera_bind_group_layout = Some(camera_bind_group_layout);
        self.camera_bind_group = Some(camera_bind_group);
        self.mesh_pipeline = Some(mesh_pipeline);
        self.depth_texture = Some(depth_texture);
        self.depth_view = Some(depth_view);
        
        Ok(())
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
        
        // Update camera uniform buffer
        if let Some(camera_buffer) = &self.camera_buffer {
            let aspect = self.config.width as f32 / self.config.height as f32;
            // perspective_rh produces OpenGL-style NDC with Z in [-1, 1]
            // wgpu expects Z in [0, 1], so we need to remap
            let proj = Mat4::perspective_rh(self.view_config.fov_deg.to_radians(), aspect, self.view_config.znear, self.view_config.zfar);
            let view = self.camera.view_matrix();
            let mvp = OPENGL_TO_WGPU_MATRIX * proj * view;
            self.queue.write_buffer(camera_buffer, 0, bytemuck::cast_slice(&mvp.to_cols_array()));
            
            if self.frame_count == 0 {
                let eye = self.camera.eye();
                let target = self.camera.target();
                println!("[DEBUG] Camera eye: {:?}", eye);
                println!("[DEBUG] Camera target: {:?}", target);
                println!("[DEBUG] FOV: {:.1} deg, aspect: {:.2}, znear: {:.2}, zfar: {:.2}",
                    self.view_config.fov_deg, aspect, self.view_config.znear, self.view_config.zfar);
                println!("[DEBUG] View matrix:\n{:?}", view);
                println!("[DEBUG] Projection matrix:\n{:?}", proj);
                println!("[DEBUG] MVP matrix:\n{:?}", mvp);
                
                // Test transform a point
                let test_point = glam::Vec4::new(0.0, -4.0, 0.0, 1.0);
                let clip_pos = mvp * test_point;
                let ndc = clip_pos / clip_pos.w;
                println!("[DEBUG] Test vertex [0, -4, 0] -> clip: {:?}, NDC: {:?}", clip_pos, ndc);
            }
            if self.frame_count % 60 == 0 {
                println!("[UPDATE] Frame {}: Camera buffer updated", self.frame_count);
            }
        }
    }

    pub fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        if self.frame_count % 60 == 0 {
            println!("[RENDER] Frame {}: render() called", self.frame_count);
        }
        let output = self.surface.get_current_texture()?;
        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Render Encoder"),
            });

        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.1,
                            g: 0.2,
                            b: 0.3,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,  // Disabled for debugging
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            // Render mesh if available
            if let (Some(pipeline), Some(vb), Some(ib), Some(bind_group)) = 
                (&self.mesh_pipeline, &self.mesh_vertex_buffer, &self.mesh_index_buffer, &self.camera_bind_group) {
                if self.frame_count % 60 == 0 {
                    println!("[RENDER] Frame {}: Drawing {} indices with viewport {}x{}", 
                        self.frame_count, self.mesh_index_count, self.config.width, self.config.height);
                }
                
                // Set viewport and scissor rect
                render_pass.set_viewport(
                    0.0, 0.0,
                    self.config.width as f32, self.config.height as f32,
                    0.0, 1.0
                );
                render_pass.set_scissor_rect(0, 0, self.config.width, self.config.height);
                
                render_pass.set_pipeline(pipeline);
                render_pass.set_bind_group(0, bind_group, &[]);
                render_pass.set_vertex_buffer(0, vb.slice(..));
                render_pass.set_index_buffer(ib.slice(..), wgpu::IndexFormat::Uint32);
                render_pass.draw_indexed(0..self.mesh_index_count, 0, 0..1);
            } else {
                if self.frame_count % 60 == 0 {
                    println!("[RENDER] Frame {}: NO MESH (pipeline={}, vb={}, ib={}, bind_group={})",
                        self.frame_count, self.mesh_pipeline.is_some(), self.mesh_vertex_buffer.is_some(),
                        self.mesh_index_buffer.is_some(), self.camera_bind_group.is_some());
                }
            }
        }

        self.queue.submit(std::iter::once(encoder.finish()));
        output.present();

        self.frame_count += 1;
        if let Some(fps) = self.fps_counter.tick() {
            self.window.set_title(&format!(
                "{} | FPS: {:.1} | Mode: {:?}",
                self.view_config.title,
                fps,
                self.camera.mode()
            ));
        }

        Ok(())
    }

    pub fn fps(&self) -> f32 {
        self.fps_counter.fps()
    }
}

pub fn run_viewer(config: ViewerConfig) -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();

    let event_loop = EventLoop::new()?;
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
            } if window_id == window.id() => {
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
            Event::WindowEvent {
                event: WindowEvent::RedrawRequested,
                ..
            } => {
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
            _ => {}
        }
    })?;

    Ok(())
}

pub fn run_mesh_viewer(
    config: ViewerConfig,
    vertices: Vec<f32>,
    indices: Vec<u32>,
) -> Result<(), Box<dyn std::error::Error>> {
    run_mesh_viewer_with_camera(config, vertices, indices, None, None)
}

pub fn run_mesh_viewer_with_camera(
    config: ViewerConfig,
    vertices: Vec<f32>,
    indices: Vec<u32>,
    camera_eye: Option<[f32; 3]>,
    camera_target: Option<[f32; 3]>,
) -> Result<(), Box<dyn std::error::Error>> {
    let event_loop = EventLoop::new()?;
    let window = Arc::new(
        WindowBuilder::new()
            .with_title(&config.title)
            .with_inner_size(PhysicalSize::new(config.width, config.height))
            .build(&event_loop)?
    );

    println!("forge3d Mesh Viewer");
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

    let mut viewer_opt: Option<Viewer> = None;
    let mut last_frame = Instant::now();
    let mut pending_mesh: Option<(Vec<f32>, Vec<u32>)> = Some((vertices, indices));
    let pending_camera = camera_eye.zip(camera_target);

    event_loop.run(move |event, elwt| {
        match event {
            Event::Resumed => {
                if viewer_opt.is_none() {
                    let v = pollster::block_on(Viewer::new(Arc::clone(&window), config.clone()));
                    match v {
                        Ok(mut v) => {
                            // Initialize mesh if we have data
                            if let Some((verts, inds)) = pending_mesh.take() {
                                if let Err(e) = v.init_mesh(&verts, &inds) {
                                    eprintln!("Failed to initialize mesh: {}", e);
                                    elwt.exit();
                                    return;
                                }
                            }
                            // Set camera position if provided
                            if let Some((eye, target)) = pending_camera {
                                v.camera.set_position(
                                    glam::Vec3::from_array(eye),
                                    glam::Vec3::from_array(target),
                                );
                            }
                            viewer_opt = Some(v);
                            last_frame = Instant::now();
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
            } if window_id == window.id() => {
                if let Some(viewer) = viewer_opt.as_mut() {
                    match event {
                        WindowEvent::RedrawRequested => {
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
                        WindowEvent::CloseRequested => {
                            elwt.exit();
                        }
                        WindowEvent::KeyboardInput {
                            event: key_event,
                            ..
                        } => {
                            viewer.handle_input(event);
                            if key_event.state == ElementState::Pressed {
                                if let PhysicalKey::Code(KeyCode::Escape) = key_event.physical_key {
                                    elwt.exit();
                                }
                            }
                        }
                        WindowEvent::Resized(physical_size) => {
                            viewer.resize(*physical_size);
                        }
                        _ => {
                            viewer.handle_input(event);
                        }
                    }
                }
            }
            Event::AboutToWait => {
                window.request_redraw();
            }
            _ => {}
        }
    })?;

    Ok(())
}
