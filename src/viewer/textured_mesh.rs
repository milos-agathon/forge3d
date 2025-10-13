// Textured mesh viewer with hillshade/water colors

use super::{ViewerConfig, CameraController, CameraMode, TEXTURED_MESH_SHADER, OPENGL_TO_WGPU_MATRIX, ViewerCommand, CameraState};
use winit::{event::*, event_loop::EventLoop, window::{Window, WindowBuilder}, dpi::PhysicalSize, keyboard::{KeyCode, PhysicalKey}};
use std::sync::Arc;
use std::time::Instant;
use wgpu::util::DeviceExt;
use std::sync::mpsc::{channel, Receiver};

pub fn run_mesh_viewer_with_texture(
    config: ViewerConfig,
    vertices: Vec<f32>,
    indices: Vec<u32>,
    uvs: Vec<f32>,
    texture_rgba: Vec<u8>,
    texture_width: u32,
    texture_height: u32,
    camera_eye: Option<[f32; 3]>,
    camera_target: Option<[f32; 3]>,
) -> Result<(), Box<dyn std::error::Error>> {
    // Set up command receiver for camera control and snapshots
    let (tx, rx) = channel();
    {
        let mut guard = super::VIEWER_SENDER.lock().unwrap();
        *guard = Some(tx);
    }

    let event_loop = EventLoop::new()?;
    let window = Arc::new(WindowBuilder::new()
        .with_title(&config.title)
        .with_inner_size(PhysicalSize::new(config.width, config.height))
        .build(&event_loop)?);

    println!("forge3d Textured Mesh Viewer");
    println!("Rendering terrain with hillshade/water colors");

    let mut viewer_opt: Option<TexturedViewer> = None;
    let mut last_frame = Instant::now();
    let mut pending_data = Some((vertices, indices, uvs, texture_rgba, texture_width, texture_height));
    let pending_camera = camera_eye.zip(camera_target);
    let mut command_rx = Some(rx);

    event_loop.run(move |event, elwt| {
        match event {
            Event::Resumed => {
                if viewer_opt.is_none() {
                    let v = pollster::block_on(TexturedViewer::new(Arc::clone(&window), config.clone()));
                    match v {
                        Ok(mut v) => {
                            if let Some((verts, inds, uvs_data, tex_data, tex_w, tex_h)) = pending_data.take() {
                                if let Err(e) = v.init_textured_mesh(&verts, &inds, &uvs_data, &tex_data, tex_w, tex_h) {
                                    eprintln!("Failed to init textured mesh: {}", e);
                                    elwt.exit();
                                    return;
                                }
                            }
                            if let Some((eye, target)) = pending_camera {
                                v.camera.set_position(glam::Vec3::from_array(eye), glam::Vec3::from_array(target));
                            }
                            if let Some(rx_inner) = command_rx.take() {
                                v.command_rx = Some(rx_inner);
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
            Event::WindowEvent { ref event, window_id } if window_id == window.id() => {
                match event {
                    WindowEvent::RedrawRequested => {
                        if let Some(viewer) = viewer_opt.as_mut() {
                            let now = Instant::now();
                            let dt = (now - last_frame).as_secs_f32();
                            last_frame = now;
                            viewer.process_commands();
                            viewer.update(dt);
                            match viewer.render() {
                                Ok(_) => {}
                                Err(wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated) => viewer.resize(viewer.window.inner_size()),
                                Err(wgpu::SurfaceError::OutOfMemory) => elwt.exit(),
                                Err(wgpu::SurfaceError::Timeout) => {}
                            }
                        }
                    }
                    WindowEvent::CloseRequested => elwt.exit(),
                    WindowEvent::KeyboardInput { event: key_event, .. } => {
                        if let Some(viewer) = viewer_opt.as_mut() {
                            viewer.handle_input(event);
                            if key_event.state == ElementState::Pressed {
                                if let PhysicalKey::Code(KeyCode::Escape) = key_event.physical_key {
                                    elwt.exit();
                                }
                            }
                        }
                    }
                    WindowEvent::Resized(physical_size) => {
                        if let Some(viewer) = viewer_opt.as_mut() {
                            viewer.resize(*physical_size);
                        }
                    }
                    _ => {
                        if let Some(viewer) = viewer_opt.as_mut() {
                            viewer.handle_input(event);
                        }
                    }
                }
            }
            Event::AboutToWait => window.request_redraw(),
            _ => {}
        }
    })?;
    Ok(())
}

struct TexturedViewer {
    window: Arc<Window>,
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    camera: CameraController,
    view_config: ViewerConfig,
    keys_pressed: std::collections::HashSet<KeyCode>,
    shift_pressed: bool,
    pipeline: Option<wgpu::RenderPipeline>,
    vertex_buffer: Option<wgpu::Buffer>,
    index_buffer: Option<wgpu::Buffer>,
    index_count: u32,
    camera_buffer: Option<wgpu::Buffer>,
    camera_bind_group: Option<wgpu::BindGroup>,
    texture_bind_group: Option<wgpu::BindGroup>,
    depth_view: Option<wgpu::TextureView>,
    command_rx: Option<Receiver<ViewerCommand>>,
}

impl TexturedViewer {
    async fn new(window: Arc<Window>, config: ViewerConfig) -> Result<Self, Box<dyn std::error::Error>> {
        let size = window.inner_size();
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor { backends: wgpu::Backends::all(), ..Default::default() });
        let surface = instance.create_surface(Arc::clone(&window))?;
        let adapter = instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: Some(&surface),
            force_fallback_adapter: false,
        }).await.ok_or("Failed to find adapter")?;

        let (device, queue) = adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("Device"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::default(),
        }, None).await?;

        let surface_caps = surface.get_capabilities(&adapter);
        let surface_format = surface_caps.formats.iter().copied().find(|f| f.is_srgb()).unwrap_or(surface_caps.formats[0]);

        let surf_config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: size.width,
            height: size.height,
            present_mode: if config.vsync { wgpu::PresentMode::Fifo } else { wgpu::PresentMode::Immediate },
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &surf_config);

        Ok(Self {
            window,
            surface,
            device,
            queue,
            config: surf_config,
            camera: CameraController::new(),
            view_config: config,
            keys_pressed: std::collections::HashSet::new(),
            shift_pressed: false,
            pipeline: None,
            vertex_buffer: None,
            index_buffer: None,
            index_count: 0,
            camera_buffer: None,
            camera_bind_group: None,
            texture_bind_group: None,
            depth_view: None,
            command_rx: None,
        })
    }

    fn init_textured_mesh(&mut self, vertices: &[f32], indices: &[u32], uvs: &[f32], texture_rgba: &[u8], texture_width: u32, texture_height: u32) -> Result<(), Box<dyn std::error::Error>> {
        let vertex_count = vertices.len() / 3;
        let mut interleaved = Vec::with_capacity(vertex_count * 5);
        for i in 0..vertex_count {
            interleaved.extend_from_slice(&[vertices[i*3], vertices[i*3+1], vertices[i*3+2], uvs[i*2], uvs[i*2+1]]);
        }

        println!("[TexturedViewer] {} vertices, {} indices, texture {}x{}", vertex_count, indices.len(), texture_width, texture_height);

        self.vertex_buffer = Some(self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Vertex Buffer"),
            contents: bytemuck::cast_slice(&interleaved),
            usage: wgpu::BufferUsages::VERTEX,
        }));

        self.index_buffer = Some(self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Index Buffer"),
            contents: bytemuck::cast_slice(indices),
            usage: wgpu::BufferUsages::INDEX,
        }));
        self.index_count = indices.len() as u32;

        let camera_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Camera Buffer"),
            size: 64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let camera_bind_group_layout = self.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Camera Bind Group Layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None },
                count: None,
            }],
        });

        let camera_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Camera Bind Group"),
            layout: &camera_bind_group_layout,
            entries: &[wgpu::BindGroupEntry { binding: 0, resource: camera_buffer.as_entire_binding() }],
        });

        self.camera_buffer = Some(camera_buffer);
        self.camera_bind_group = Some(camera_bind_group);

        let texture_size = wgpu::Extent3d { width: texture_width, height: texture_height, depth_or_array_layers: 1 };
        let texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Terrain Texture"),
            size: texture_size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });

        self.queue.write_texture(
            wgpu::ImageCopyTexture { texture: &texture, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
            texture_rgba,
            wgpu::ImageDataLayout { offset: 0, bytes_per_row: Some(4 * texture_width), rows_per_image: Some(texture_height) },
            texture_size,
        );

        let texture_view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let sampler = self.device.create_sampler(&wgpu::SamplerDescriptor {
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        let texture_bind_group_layout = self.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Texture Bind Group Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture { sample_type: wgpu::TextureSampleType::Float { filterable: true }, view_dimension: wgpu::TextureViewDimension::D2, multisampled: false },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        let texture_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Texture Bind Group"),
            layout: &texture_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&texture_view) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&sampler) },
            ],
        });

        self.texture_bind_group = Some(texture_bind_group);

        let shader = self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Textured Mesh Shader"),
            source: wgpu::ShaderSource::Wgsl(TEXTURED_MESH_SHADER.into()),
        });

        let pipeline_layout = self.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Pipeline Layout"),
            bind_group_layouts: &[&camera_bind_group_layout, &texture_bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = self.device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Render Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: 20,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &[
                        wgpu::VertexAttribute { offset: 0, shader_location: 0, format: wgpu::VertexFormat::Float32x3 },
                        wgpu::VertexAttribute { offset: 12, shader_location: 1, format: wgpu::VertexFormat::Float32x2 },
                    ],
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
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: Some(wgpu::Face::Back),
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
        });

        self.pipeline = Some(pipeline);
        self.create_depth_texture();
        println!("[TexturedViewer] Textured mesh ready");
        Ok(())
    }

    fn create_depth_texture(&mut self) {
        let size = wgpu::Extent3d { width: self.config.width, height: self.config.height, depth_or_array_layers: 1 };
        let depth_texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Depth Texture"),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        self.depth_view = Some(depth_texture.create_view(&wgpu::TextureViewDescriptor::default()));
    }

    fn resize(&mut self, new_size: PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            self.config.width = new_size.width;
            self.config.height = new_size.height;
            self.surface.configure(&self.device, &self.config);
            self.create_depth_texture();
        }
    }

    fn process_commands(&mut self) {
        let mut commands = Vec::new();
        if let Some(rx) = &self.command_rx {
            while let Ok(cmd) = rx.try_recv() {
                commands.push(cmd);
            }
        }
        
        for cmd in commands {
            match cmd {
                ViewerCommand::SetCamera { distance, theta, phi } => {
                    if let Some(d) = distance {
                        self.camera.set_orbit_distance(d);
                    }
                    if let Some(t) = theta {
                        self.camera.set_orbit_yaw(t.to_radians());
                    }
                    if let Some(p) = phi {
                        self.camera.set_orbit_pitch(p.to_radians());
                    }
                    println!("[Viewer] Camera updated: distance={:.1}, theta={:.1}°, phi={:.1}°", 
                             self.camera.orbit_distance(), 
                             self.camera.orbit_yaw().to_degrees(), 
                             self.camera.orbit_pitch().to_degrees());
                }
                ViewerCommand::Snapshot { path, width, height } => {
                    let w = width.unwrap_or(self.config.width);
                    let h = height.unwrap_or(self.config.height);
                    if let Err(e) = self.save_snapshot(&path, w, h) {
                        eprintln!("[Viewer] Failed to save snapshot: {}", e);
                    } else {
                        println!("[Viewer] Snapshot saved to: {}", path);
                    }
                }
                ViewerCommand::Export(path) => {
                    let w = self.config.width;
                    let h = self.config.height;
                    if let Err(e) = self.save_snapshot(&path, w, h) {
                        eprintln!("[Viewer] Failed to export: {}", e);
                    } else {
                        println!("[Viewer] Exported to: {}", path);
                    }
                }
                ViewerCommand::GetCamera { response_tx } => {
                    let eye_pos = self.camera.eye();
                    let target_pos = self.camera.target();
                    let state = CameraState {
                        eye: [eye_pos.x, eye_pos.y, eye_pos.z],
                        target: [target_pos.x, target_pos.y, target_pos.z],
                        distance: self.camera.orbit_distance(),
                        theta: self.camera.orbit_yaw().to_degrees(),
                        phi: self.camera.orbit_pitch().to_degrees(),
                        fov: self.view_config.fov_deg,
                    };
                    let _ = response_tx.send(state);
                }
            }
        }
    }

    fn update(&mut self, dt: f32) {
        let mut forward = 0.0;
        let mut right = 0.0;
        let mut up = 0.0;
        if self.keys_pressed.contains(&KeyCode::KeyW) { forward += 1.0; }
        if self.keys_pressed.contains(&KeyCode::KeyS) { forward -= 1.0; }
        if self.keys_pressed.contains(&KeyCode::KeyD) { right += 1.0; }
        if self.keys_pressed.contains(&KeyCode::KeyA) { right -= 1.0; }
        if self.keys_pressed.contains(&KeyCode::KeyE) { up += 1.0; }
        if self.keys_pressed.contains(&KeyCode::KeyQ) { up -= 1.0; }

        let speed_mult = if self.shift_pressed { 3.0 } else { 1.0 };
        self.camera.update_fps(dt * speed_mult, forward, right, up);

        if let Some(camera_buffer) = &self.camera_buffer {
            let view = self.camera.view_matrix();
            let aspect = self.config.width as f32 / self.config.height as f32;
            let proj = glam::Mat4::perspective_rh(self.view_config.fov_deg.to_radians(), aspect, self.view_config.znear, self.view_config.zfar);
            let mvp = OPENGL_TO_WGPU_MATRIX * proj * view;
            self.queue.write_buffer(camera_buffer, 0, bytemuck::cast_slice(mvp.as_ref()));
        }
    }

    fn save_snapshot(&mut self, path: &str, width: u32, height: u32) -> Result<(), Box<dyn std::error::Error>> {
        // Use the same color format as the active pipeline to keep render pass compatible.
        let color_format = self.config.format;

        // Create offscreen texture
        let texture_desc = wgpu::TextureDescriptor {
            label: Some("Snapshot Texture"),
            size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: color_format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        };
        let texture = self.device.create_texture(&texture_desc);
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());

        // Create depth texture for offscreen render
        let depth_texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Snapshot Depth"),
            size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let depth_view = depth_texture.create_view(&wgpu::TextureViewDescriptor::default());

        // Update camera with snapshot dimensions
        if let Some(camera_buffer) = &self.camera_buffer {
            let view = self.camera.view_matrix();
            let aspect = width as f32 / height as f32;
            let proj = glam::Mat4::perspective_rh(self.view_config.fov_deg.to_radians(), aspect, self.view_config.znear, self.view_config.zfar);
            let mvp = OPENGL_TO_WGPU_MATRIX * proj * view;
            self.queue.write_buffer(camera_buffer, 0, bytemuck::cast_slice(mvp.as_ref()));
        }

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("Snapshot Encoder") });

        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Snapshot Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color { r: 0.1, g: 0.2, b: 0.3, a: 1.0 }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &depth_view,
                    depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Clear(1.0), store: wgpu::StoreOp::Store }),
                    stencil_ops: None,
                }),
                ..Default::default()
            });

            if let (Some(pipeline), Some(vertex_buffer), Some(index_buffer), Some(camera_bind_group), Some(texture_bind_group)) = 
                (&self.pipeline, &self.vertex_buffer, &self.index_buffer, &self.camera_bind_group, &self.texture_bind_group) {
                render_pass.set_pipeline(pipeline);
                render_pass.set_bind_group(0, camera_bind_group, &[]);
                render_pass.set_bind_group(1, texture_bind_group, &[]);
                render_pass.set_vertex_buffer(0, vertex_buffer.slice(..));
                render_pass.set_index_buffer(index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                render_pass.draw_indexed(0..self.index_count, 0, 0..1);
            }
        }

        // Copy texture to buffer for reading. 4 bytes per pixel
        let bytes_per_row = 4 * width;
        let buffer_size = (bytes_per_row * height) as u64;
        let output_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Snapshot Output Buffer"),
            size: buffer_size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        encoder.copy_texture_to_buffer(
            wgpu::ImageCopyTexture {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::ImageCopyBuffer {
                buffer: &output_buffer,
                layout: wgpu::ImageDataLayout {
                    offset: 0,
                    bytes_per_row: Some(bytes_per_row),
                    rows_per_image: Some(height),
                },
            },
            wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
        );

        self.queue.submit(std::iter::once(encoder.finish()));

        // Map buffer and save to PNG
        let buffer_slice = output_buffer.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).unwrap();
        });
        self.device.poll(wgpu::Maintain::Wait);
        rx.recv()??;

        let data = buffer_slice.get_mapped_range();
        let mut rgba_data = vec![0u8; (width * height * 4) as usize];
        // If the surface/pipeline format is BGRA, swizzle to RGBA for saving.
        match color_format {
            wgpu::TextureFormat::Bgra8Unorm | wgpu::TextureFormat::Bgra8UnormSrgb => {
                for (i, chunk) in data.chunks_exact(4).enumerate() {
                    let dst = &mut rgba_data[i * 4..i * 4 + 4];
                    // chunk: [B, G, R, A] -> [R, G, B, A]
                    dst[0] = chunk[2];
                    dst[1] = chunk[1];
                    dst[2] = chunk[0];
                    dst[3] = chunk[3];
                }
            }
            _ => {
                // Already RGBA8
                rgba_data.copy_from_slice(&data);
            }
        }
        drop(data);
        output_buffer.unmap();

        // Save as PNG
        image::save_buffer(path, &rgba_data, width, height, image::ColorType::Rgba8)?;
        Ok(())
    }

    fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        let output = self.surface.get_current_texture()?;
        let view = output.texture.create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("Render Encoder") });

        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color { r: 0.1, g: 0.2, b: 0.3, a: 1.0 }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: self.depth_view.as_ref().map(|depth_view| wgpu::RenderPassDepthStencilAttachment {
                    view: depth_view,
                    depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Clear(1.0), store: wgpu::StoreOp::Store }),
                    stencil_ops: None,
                }),
                ..Default::default()
            });

            if let (Some(pipeline), Some(vertex_buffer), Some(index_buffer), Some(camera_bind_group), Some(texture_bind_group)) = 
                (&self.pipeline, &self.vertex_buffer, &self.index_buffer, &self.camera_bind_group, &self.texture_bind_group) {
                render_pass.set_pipeline(pipeline);
                render_pass.set_bind_group(0, camera_bind_group, &[]);
                render_pass.set_bind_group(1, texture_bind_group, &[]);
                render_pass.set_vertex_buffer(0, vertex_buffer.slice(..));
                render_pass.set_index_buffer(index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                render_pass.draw_indexed(0..self.index_count, 0, 0..1);
            }
        }

        self.queue.submit(std::iter::once(encoder.finish()));
        output.present();
        Ok(())
    }

    fn handle_input(&mut self, event: &WindowEvent) {
        match event {
            WindowEvent::KeyboardInput { event: key_event, .. } => {
                if let PhysicalKey::Code(keycode) = key_event.physical_key {
                    match key_event.state {
                        ElementState::Pressed => {
                            self.keys_pressed.insert(keycode);
                            if keycode == KeyCode::Tab {
                                self.camera.set_mode(if self.camera.mode() == CameraMode::Orbit { CameraMode::Fps } else { CameraMode::Orbit });
                                println!("Camera mode: {:?}", self.camera.mode());
                            }
                            if keycode == KeyCode::ShiftLeft || keycode == KeyCode::ShiftRight {
                                self.shift_pressed = true;
                            }
                        }
                        ElementState::Released => {
                            self.keys_pressed.remove(&keycode);
                            if keycode == KeyCode::ShiftLeft || keycode == KeyCode::ShiftRight {
                                self.shift_pressed = false;
                            }
                        }
                    }
                }
            }
            WindowEvent::MouseInput { state, button: MouseButton::Left, .. } => {
                self.camera.mouse_pressed = *state == ElementState::Pressed;
            }
            WindowEvent::CursorMoved { position, .. } => {
                self.camera.handle_mouse_move(position.x as f32, position.y as f32);
            }
            WindowEvent::MouseWheel { delta, .. } => {
                let scroll_delta = match delta {
                    MouseScrollDelta::LineDelta(_, y) => *y,
                    MouseScrollDelta::PixelDelta(pos) => pos.y as f32 * 0.01,
                };
                self.camera.handle_mouse_scroll(scroll_delta);
            }
            _ => {}
        }
    }
}
