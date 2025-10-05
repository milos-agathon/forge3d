// src/viewer/mod.rs
// Workstream I1: Interactive windowed viewer for forge3d
// - Creates window with winit 0.29
// - Handles input events (mouse, keyboard)
// - Renders frames at 60 FPS
// - Orbit and FPS camera modes

pub mod camera_controller;

use camera_controller::{CameraController, CameraMode};
use winit::{
    event::*,
    event_loop::EventLoop,
    window::{Window, WindowBuilder},
    dpi::PhysicalSize,
    keyboard::{KeyCode, PhysicalKey},
};
use wgpu::{Instance, Surface, Device, Queue, SurfaceConfiguration};
use std::sync::Arc;
use std::time::{Duration, Instant};

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

pub struct Viewer {
    window: Arc<Window>,
    surface: Surface<'static>,
    device: Device,
    queue: Queue,
    config: SurfaceConfiguration,
    camera: CameraController,
    view_config: ViewerConfig,
    last_frame: Instant,
    frame_count: u64,
    fps_counter: FpsCounter,
    // Input state
    keys_pressed: std::collections::HashSet<KeyCode>,
    shift_pressed: bool,
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
    pub async fn new(window: Arc<Window>, config: ViewerConfig) -> Result<Self, Box<dyn std::error::Error>> {
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
            last_frame: Instant::now(),
            frame_count: 0,
            fps_counter: FpsCounter::new(),
            keys_pressed: std::collections::HashSet::new(),
            shift_pressed: false,
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
        }
    }

    pub fn handle_input(&mut self, event: &WindowEvent) -> bool {
        match event {
            WindowEvent::KeyboardInput {
                event: key_event,
                ..
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
                self.camera.handle_mouse_move(position.x as f32, position.y as f32);
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
    }

    pub fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
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
            let _render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
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
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            // TODO: Render scene here
            // For now, just clear to blue-gray
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
            .build(&event_loop)?
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
                                event: key_event,
                                ..
                            } => {
                                if key_event.state == ElementState::Pressed {
                                    if let PhysicalKey::Code(KeyCode::Escape) = key_event.physical_key {
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
