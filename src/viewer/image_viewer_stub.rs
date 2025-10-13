// Image viewer implementation - renders RGBA data as a textured quad
use super::{ViewerConfig, Viewer};
use winit::{
    event::*,
    event_loop::EventLoop,
    window::{Window, WindowBuilder},
    dpi::PhysicalSize,
    keyboard::{KeyCode, PhysicalKey},
};
use std::sync::Arc;
use std::time::Instant;
use wgpu::util::DeviceExt;

const IMAGE_SHADER: &str = r#"
struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    var out: VertexOutput;
    // Fullscreen triangle covering NDC [-1,1]
    // vertex 0: (-1, -1) -> UV (0, 1)
    // vertex 1: (3, -1)  -> UV (2, 1)
    // vertex 2: (-1, 3)  -> UV (0, -1)
    let x = f32((vertex_index & 1u) << 2u);
    let y = f32((vertex_index & 2u) << 1u);
    out.clip_position = vec4<f32>(x - 1.0, y - 1.0, 0.0, 1.0);
    out.uv = vec2<f32>(x * 0.5, 1.0 - y * 0.5);
    return out;
}

@group(0) @binding(0) var t_image: texture_2d<f32>;
@group(0) @binding(1) var s_image: sampler;

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return textureSample(t_image, s_image, in.uv);
}
"#;

pub fn run_image_viewer(
    config: ViewerConfig,
    rgba_data: Vec<u8>,
    img_width: u32,
    img_height: u32,
) -> Result<(), Box<dyn std::error::Error>> {
    let event_loop = EventLoop::new()?;
    let window = Arc::new(
        WindowBuilder::new()
            .with_title(&config.title)
            .with_inner_size(PhysicalSize::new(config.width, config.height))
            .build(&event_loop)?
    );

    println!("forge3d Image Viewer");
    println!("Displaying pre-rendered image: {}x{}", img_width, img_height);
    println!("Controls:");
    println!("  Esc       - Close viewer");

    let mut viewer_opt: Option<ImageViewer> = None;

    event_loop.run(move |event, elwt| {
        match event {
            Event::Resumed => {
                if viewer_opt.is_none() {
                    let v = pollster::block_on(ImageViewer::new(
                        Arc::clone(&window),
                        config.clone(),
                        rgba_data.clone(),
                        img_width,
                        img_height,
                    ));
                    match v {
                        Ok(v) => {
                            println!("[ImageViewer] Viewer created, requesting initial redraw");
                            viewer_opt = Some(v);
                            window.request_redraw();
                        }
                        Err(e) => {
                            eprintln!("Failed to create image viewer: {}", e);
                            elwt.exit();
                        }
                    }
                }
            }
            Event::WindowEvent {
                ref event,
                window_id,
            } if window_id == window.id() => {
                match event {
                    WindowEvent::RedrawRequested => {
                        if let Some(viewer) = viewer_opt.as_mut() {
                            match viewer.render() {
                                Ok(_) => {}
                                Err(wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated) => {
                                    viewer.resize(viewer.window.inner_size())
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
                    WindowEvent::CloseRequested => elwt.exit(),
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
                        if let Some(viewer) = viewer_opt.as_mut() {
                            viewer.resize(*physical_size);
                        }
                    }
                    _ => {}
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

struct ImageViewer {
    window: Arc<Window>,
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    pipeline: wgpu::RenderPipeline,
    bind_group: wgpu::BindGroup,
}

impl ImageViewer {
    async fn new(
        window: Arc<Window>,
        view_config: ViewerConfig,
        rgba_data: Vec<u8>,
        img_width: u32,
        img_height: u32,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let size = window.inner_size();

        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        let surface = instance.create_surface(Arc::clone(&window))?;

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .ok_or("Failed to find suitable adapter")?;

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("Image Viewer Device"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                },
                None,
            )
            .await?;

        let surface_caps = surface.get_capabilities(&adapter);
        let surface_format = surface_caps
            .formats
            .iter()
            .copied()
            .find(|f| f.is_srgb())
            .unwrap_or(surface_caps.formats[0]);

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: size.width,
            height: size.height,
            present_mode: if view_config.vsync {
                wgpu::PresentMode::AutoVsync
            } else {
                wgpu::PresentMode::AutoNoVsync
            },
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };

        surface.configure(&device, &config);

        // Create texture from RGBA data
        let texture_size = wgpu::Extent3d {
            width: img_width,
            height: img_height,
            depth_or_array_layers: 1,
        };

        // Debug: Check texture data
        let expected_size = (img_width * img_height * 4) as usize;
        println!("[ImageViewer] Texture data: {} bytes (expected {})", rgba_data.len(), expected_size);
        if rgba_data.len() >= 4 {
            println!("[ImageViewer] First pixel RGBA: [{}, {}, {}, {}]", 
                rgba_data[0], rgba_data[1], rgba_data[2], rgba_data[3]);
        }
        let non_zero = rgba_data.iter().filter(|&&x| x != 0).count();
        println!("[ImageViewer] Non-zero bytes: {} / {}", non_zero, rgba_data.len());

        // Convert RGBA to BGRA if needed to match common surface formats
        let mut bgra_data = rgba_data.clone();
        for i in (0..bgra_data.len()).step_by(4) {
            bgra_data.swap(i, i + 2); // Swap R and B channels
        }
        println!("[ImageViewer] Converted RGBA -> BGRA");
        if bgra_data.len() >= 4 {
            println!("[ImageViewer] First pixel BGRA: [{}, {}, {}, {}]", 
                bgra_data[0], bgra_data[1], bgra_data[2], bgra_data[3]);
        }

        let texture = device.create_texture_with_data(
            &queue,
            &wgpu::TextureDescriptor {
                label: Some("image_texture"),
                size: texture_size,
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Bgra8UnormSrgb, // Match surface format
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            },
            wgpu::util::TextureDataOrder::LayerMajor,
            &bgra_data,
        );
        println!("[ImageViewer] Texture created successfully with Bgra8UnormSrgb format");

        let texture_view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("image_sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        // Create bind group layout
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("image_bind_group_layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
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

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("image_bind_group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&texture_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
            ],
        });

        // Create shader and pipeline
        println!("[ImageViewer] Creating shader module...");
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("image_shader"),
            source: wgpu::ShaderSource::Wgsl(IMAGE_SHADER.into()),
        });
        println!("[ImageViewer] Shader module created successfully");

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("image_pipeline_layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        println!("[ImageViewer] Creating render pipeline with format: {:?}", config.format);
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("image_pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
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
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
        });
        println!("[ImageViewer] Pipeline created successfully");

        println!("[ImageViewer] Initialized with {}x{} RGBA texture", img_width, img_height);

        Ok(Self {
            window,
            surface,
            device,
            queue,
            config,
            pipeline,
            bind_group,
        })
    }

    fn resize(&mut self, new_size: PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            self.config.width = new_size.width;
            self.config.height = new_size.height;
            self.surface.configure(&self.device, &self.config);
        }
    }

    fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        static mut FRAME_COUNT: u64 = 0;
        unsafe { FRAME_COUNT += 1; }
        
        let fc = unsafe { FRAME_COUNT };
        if fc % 60 == 0 {
            println!("[ImageViewer] Rendering frame {}", fc);
        }
        
        let output = match self.surface.get_current_texture() {
            Ok(output) => output,
            Err(e) => {
                println!("[ImageViewer] ERROR: Failed to get surface texture: {:?}", e);
                return Err(e);
            }
        };
        
        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Image Render Encoder"),
            });

        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Image Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 1.0,
                            g: 0.0,
                            b: 1.0,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            if fc == 1 {
                println!("[ImageViewer] First frame render - viewport: {}x{}", self.config.width, self.config.height);
                println!("[ImageViewer] Drawing fullscreen triangle (3 vertices)");
            }

            render_pass.set_pipeline(&self.pipeline);
            render_pass.set_bind_group(0, &self.bind_group, &[]);
            render_pass.draw(0..3, 0..1); // Fullscreen triangle
            
            if fc == 1 {
                println!("[ImageViewer] Draw call issued");
            }
        }

        if fc == 1 {
            println!("[ImageViewer] Submitting command buffer...");
        }
        self.queue.submit(std::iter::once(encoder.finish()));
        
        if fc == 1 {
            println!("[ImageViewer] Calling present()...");
        }
        output.present();
        
        if fc == 1 {
            println!("[ImageViewer] Frame {} complete", fc);
        }

        Ok(())
    }
}
