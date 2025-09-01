//! I6: Split-buffers performance demo
//!
//! Compares bind group churn (multiple bind groups) vs single bind group performance.
//! Measures frame times and bind group switching overhead.

use std::time::Instant;
use wgpu::*;
use pollster;

/// Configuration for the performance demo
struct PerfConfig {
    pub frames: u32,
    pub objects: u32,
    pub output_path: Option<String>,
}

impl Default for PerfConfig {
    fn default() -> Self {
        Self {
            frames: 600,
            objects: 1000,
            output_path: None,
        }
    }
}

/// Performance metrics for a test run
#[derive(Debug, Clone)]
struct PerfMetrics {
    pub config_name: String,
    pub total_time_ms: f64,
    pub avg_frame_time_ms: f64,
    pub min_frame_time_ms: f64,
    pub max_frame_time_ms: f64,
    pub bind_group_switches: u32,
    pub draw_calls: u32,
}

/// Test context for GPU operations
struct TestContext {
    device: Device,
    queue: Queue,
    config: PerfConfig,
}

impl TestContext {
    async fn new(config: PerfConfig) -> Result<Self, Box<dyn std::error::Error>> {
        let instance = Instance::new(InstanceDescriptor::default());
        let adapter = instance
            .request_adapter(&RequestAdapterOptions::default())
            .await
            .ok_or("Failed to find adapter")?;

        let (device, queue) = adapter
            .request_device(&DeviceDescriptor::default(), None)
            .await?;

        Ok(Self { device, queue, config })
    }
}

/// Split bind groups approach - separate bind groups for each object
struct SplitBindGroupTest {
    render_pipeline: RenderPipeline,
    bind_groups: Vec<BindGroup>,
    vertex_buffer: Buffer,
    index_buffer: Buffer,
    uniform_buffers: Vec<Buffer>,
}

impl SplitBindGroupTest {
    fn new(ctx: &TestContext) -> Result<Self, Box<dyn std::error::Error>> {
        // Create bind group layout for per-object uniforms
        let bind_group_layout = ctx.device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("SplitBG_PerObject"),
            entries: &[BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::VERTEX_FRAGMENT,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        // Create render pipeline
        let pipeline_layout = ctx.device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("SplitBG_Pipeline"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let shader = ctx.device.create_shader_module(ShaderModuleDescriptor {
            label: Some("SplitBG_Shader"),
            source: ShaderSource::Wgsl(include_str!("../../shaders/perf/split_bg.wgsl").into()),
        });

        let render_pipeline = ctx.device.create_render_pipeline(&RenderPipelineDescriptor {
            label: Some("SplitBG_Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: VertexState {
                module: &shader,
                entry_point: "vs_main",
                buffers: &[VertexBufferLayout {
                    array_stride: 12, // 3 floats
                    step_mode: VertexStepMode::Vertex,
                    attributes: &[VertexAttribute {
                        offset: 0,
                        shader_location: 0,
                        format: VertexFormat::Float32x3,
                    }],
                }],
            },
            fragment: Some(FragmentState {
                module: &shader,
                entry_point: "fs_main",
                targets: &[Some(ColorTargetState {
                    format: TextureFormat::Rgba8UnormSrgb,
                    blend: Some(BlendState::REPLACE),
                    write_mask: ColorWrites::ALL,
                })],
            }),
            primitive: PrimitiveState::default(),
            depth_stencil: None,
            multisample: MultisampleState::default(),
            multiview: None,
        });

        // Create vertex and index buffers (simple quad)
        let vertices = [
            [-0.1f32, -0.1, 0.0], [0.1, -0.1, 0.0], [0.1, 0.1, 0.0], [-0.1, 0.1, 0.0]
        ];
        let indices = [0u16, 1, 2, 2, 3, 0];

        let vertex_buffer = ctx.device.create_buffer_init(&util::BufferInitDescriptor {
            label: Some("SplitBG_Vertices"),
            contents: bytemuck::cast_slice(&vertices),
            usage: BufferUsages::VERTEX,
        });

        let index_buffer = ctx.device.create_buffer_init(&util::BufferInitDescriptor {
            label: Some("SplitBG_Indices"),
            contents: bytemuck::cast_slice(&indices),
            usage: BufferUsages::INDEX,
        });

        // Create per-object uniform buffers and bind groups
        let mut uniform_buffers = Vec::new();
        let mut bind_groups = Vec::new();

        for i in 0..ctx.config.objects {
            // Per-object transform matrix (identity + small offset)
            let transform = [
                [1.0f32, 0.0, 0.0, (i as f32 * 0.01) % 2.0 - 1.0],
                [0.0, 1.0, 0.0, ((i * 17) as f32 * 0.01) % 2.0 - 1.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ];

            let uniform_buffer = ctx.device.create_buffer_init(&util::BufferInitDescriptor {
                label: Some(&format!("SplitBG_Uniform_{}", i)),
                contents: bytemuck::cast_slice(&transform),
                usage: BufferUsages::UNIFORM,
            });

            let bind_group = ctx.device.create_bind_group(&BindGroupDescriptor {
                label: Some(&format!("SplitBG_BindGroup_{}", i)),
                layout: &bind_group_layout,
                entries: &[BindGroupEntry {
                    binding: 0,
                    resource: uniform_buffer.as_entire_binding(),
                }],
            });

            uniform_buffers.push(uniform_buffer);
            bind_groups.push(bind_group);
        }

        Ok(Self {
            render_pipeline,
            bind_groups,
            vertex_buffer,
            index_buffer,
            uniform_buffers,
        })
    }

    fn render_frame(&self, ctx: &TestContext, target: &TextureView) -> u32 {
        let mut encoder = ctx.device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("SplitBG_Encoder"),
        });

        let mut bind_group_switches = 0;

        {
            let mut render_pass = encoder.begin_render_pass(&RenderPassDescriptor {
                label: Some("SplitBG_RenderPass"),
                color_attachments: &[Some(RenderPassColorAttachment {
                    view: target,
                    resolve_target: None,
                    ops: Operations {
                        load: LoadOp::Clear(Color::BLACK),
                        store: StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            render_pass.set_pipeline(&self.render_pipeline);
            render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
            render_pass.set_index_buffer(self.index_buffer.slice(..), IndexFormat::Uint16);

            // Render each object with its own bind group (causes churn)
            for (i, bind_group) in self.bind_groups.iter().enumerate() {
                render_pass.set_bind_group(0, bind_group, &[]);
                render_pass.draw_indexed(0..6, 0, i as u32..i as u32 + 1);
                bind_group_switches += 1;
            }
        }

        ctx.queue.submit(Some(encoder.finish()));
        bind_group_switches
    }
}

/// Single bind group approach - one bind group with dynamic offsets or large buffer
struct SingleBindGroupTest {
    render_pipeline: RenderPipeline,
    bind_group: BindGroup,
    vertex_buffer: Buffer,
    index_buffer: Buffer,
    uniform_buffer: Buffer,
}

impl SingleBindGroupTest {
    fn new(ctx: &TestContext) -> Result<Self, Box<dyn std::error::Error>> {
        // Create bind group layout with dynamic offset
        let bind_group_layout = ctx.device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("SingleBG_Layout"),
            entries: &[BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::VERTEX_FRAGMENT,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Uniform,
                    has_dynamic_offset: true,
                    min_binding_size: Some(std::num::NonZeroU64::new(64).unwrap()),
                },
                count: None,
            }],
        });

        // Create render pipeline
        let pipeline_layout = ctx.device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("SingleBG_Pipeline"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let shader = ctx.device.create_shader_module(ShaderModuleDescriptor {
            label: Some("SingleBG_Shader"),
            source: ShaderSource::Wgsl(include_str!("../../shaders/perf/single_bg.wgsl").into()),
        });

        let render_pipeline = ctx.device.create_render_pipeline(&RenderPipelineDescriptor {
            label: Some("SingleBG_Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: VertexState {
                module: &shader,
                entry_point: "vs_main",
                buffers: &[VertexBufferLayout {
                    array_stride: 12,
                    step_mode: VertexStepMode::Vertex,
                    attributes: &[VertexAttribute {
                        offset: 0,
                        shader_location: 0,
                        format: VertexFormat::Float32x3,
                    }],
                }],
            },
            fragment: Some(FragmentState {
                module: &shader,
                entry_point: "fs_main",
                targets: &[Some(ColorTargetState {
                    format: TextureFormat::Rgba8UnormSrgb,
                    blend: Some(BlendState::REPLACE),
                    write_mask: ColorWrites::ALL,
                })],
            }),
            primitive: PrimitiveState::default(),
            depth_stencil: None,
            multisample: MultisampleState::default(),
            multiview: None,
        });

        // Create vertex and index buffers (same as split approach)
        let vertices = [
            [-0.1f32, -0.1, 0.0], [0.1, -0.1, 0.0], [0.1, 0.1, 0.0], [-0.1, 0.1, 0.0]
        ];
        let indices = [0u16, 1, 2, 2, 3, 0];

        let vertex_buffer = ctx.device.create_buffer_init(&util::BufferInitDescriptor {
            label: Some("SingleBG_Vertices"),
            contents: bytemuck::cast_slice(&vertices),
            usage: BufferUsages::VERTEX,
        });

        let index_buffer = ctx.device.create_buffer_init(&util::BufferInitDescriptor {
            label: Some("SingleBG_Indices"),
            contents: bytemuck::cast_slice(&indices),
            usage: BufferUsages::INDEX,
        });

        // Create single large uniform buffer for all objects
        let uniform_size = 64 * ctx.config.objects as u64; // 64 bytes per object (aligned)
        let uniform_buffer = ctx.device.create_buffer(&BufferDescriptor {
            label: Some("SingleBG_LargeUniform"),
            size: uniform_size,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Upload all object transforms to the large buffer
        for i in 0..ctx.config.objects {
            let transform = [
                [1.0f32, 0.0, 0.0, (i as f32 * 0.01) % 2.0 - 1.0],
                [0.0, 1.0, 0.0, ((i * 17) as f32 * 0.01) % 2.0 - 1.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ];
            
            ctx.queue.write_buffer(
                &uniform_buffer,
                i as u64 * 64,
                bytemuck::cast_slice(&transform),
            );
        }

        // Create single bind group
        let bind_group = ctx.device.create_bind_group(&BindGroupDescriptor {
            label: Some("SingleBG_BindGroup"),
            layout: &bind_group_layout,
            entries: &[BindGroupEntry {
                binding: 0,
                resource: BindingResource::Buffer(BufferBinding {
                    buffer: &uniform_buffer,
                    offset: 0,
                    size: Some(std::num::NonZeroU64::new(64).unwrap()),
                }),
            }],
        });

        Ok(Self {
            render_pipeline,
            bind_group,
            vertex_buffer,
            index_buffer,
            uniform_buffer,
        })
    }

    fn render_frame(&self, ctx: &TestContext, target: &TextureView) -> u32 {
        let mut encoder = ctx.device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("SingleBG_Encoder"),
        });

        {
            let mut render_pass = encoder.begin_render_pass(&RenderPassDescriptor {
                label: Some("SingleBG_RenderPass"),
                color_attachments: &[Some(RenderPassColorAttachment {
                    view: target,
                    resolve_target: None,
                    ops: Operations {
                        load: LoadOp::Clear(Color::BLACK),
                        store: StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            render_pass.set_pipeline(&self.render_pipeline);
            render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
            render_pass.set_index_buffer(self.index_buffer.slice(..), IndexFormat::Uint16);

            // Render all objects with dynamic offsets (single bind group, multiple offsets)
            for i in 0..ctx.config.objects {
                let offset = i * 64; // 64-byte aligned offsets
                render_pass.set_bind_group(0, &self.bind_group, &[offset]);
                render_pass.draw_indexed(0..6, 0, i..i + 1);
            }
        }

        ctx.queue.submit(Some(encoder.finish()));
        1 // Only 1 bind group used
    }
}

/// Run performance test and collect metrics
fn run_test<T>(
    test: &T,
    ctx: &TestContext,
    name: &str,
    render_fn: impl Fn(&T, &TestContext, &TextureView) -> u32,
) -> Result<PerfMetrics, Box<dyn std::error::Error>> {
    // Create render target
    let texture = ctx.device.create_texture(&TextureDescriptor {
        label: Some("PerfTest_Target"),
        size: Extent3d { width: 512, height: 512, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: TextureDimension::D2,
        format: TextureFormat::Rgba8UnormSrgb,
        usage: TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    let target_view = texture.create_view(&TextureViewDescriptor::default());

    let mut frame_times = Vec::new();
    let mut total_bind_group_switches = 0;

    println!("Running {} test with {} frames, {} objects...", name, ctx.config.frames, ctx.config.objects);
    
    let test_start = Instant::now();

    for frame in 0..ctx.config.frames {
        let frame_start = Instant::now();
        
        let bind_group_switches = render_fn(test, ctx, &target_view);
        ctx.device.poll(Maintain::Wait); // Ensure completion
        
        let frame_time = frame_start.elapsed().as_secs_f64() * 1000.0;
        frame_times.push(frame_time);
        total_bind_group_switches += bind_group_switches;

        if frame % 100 == 0 {
            println!("  Frame {}/{} - {:.3}ms", frame, ctx.config.frames, frame_time);
        }
    }

    let total_time = test_start.elapsed().as_secs_f64() * 1000.0;
    let avg_frame_time = frame_times.iter().sum::<f64>() / frame_times.len() as f64;
    let min_frame_time = frame_times.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_frame_time = frame_times.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    Ok(PerfMetrics {
        config_name: name.to_string(),
        total_time_ms: total_time,
        avg_frame_time_ms: avg_frame_time,
        min_frame_time_ms: min_frame_time,
        max_frame_time_ms: max_frame_time,
        bind_group_switches: total_bind_group_switches,
        draw_calls: ctx.config.frames * ctx.config.objects,
    })
}

/// Generate CSV output
fn write_csv(metrics: &[PerfMetrics], path: &str) -> Result<(), Box<dyn std::error::Error>> {
    use std::fs::File;
    use std::io::Write;

    let mut file = File::create(path)?;
    
    writeln!(file, "Configuration,TotalTimeMs,AvgFrameTimeMs,MinFrameTimeMs,MaxFrameTimeMs,BindGroupSwitches,DrawCalls")?;
    
    for metric in metrics {
        writeln!(file, "{},{:.3},{:.3},{:.3},{:.3},{},{}", 
                 metric.config_name,
                 metric.total_time_ms,
                 metric.avg_frame_time_ms,
                 metric.min_frame_time_ms,
                 metric.max_frame_time_ms,
                 metric.bind_group_switches,
                 metric.draw_calls)?;
    }

    println!("CSV written to: {}", path);
    Ok(())
}

/// Main entry point
async fn run_demo() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let mut config = PerfConfig::default();

    // Parse command line arguments
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--frames" => {
                i += 1;
                if i < args.len() {
                    config.frames = args[i].parse().unwrap_or(600);
                }
            },
            "--objects" => {
                i += 1;
                if i < args.len() {
                    config.objects = args[i].parse().unwrap_or(1000);
                }
            },
            "--out" => {
                i += 1;
                if i < args.len() {
                    config.output_path = Some(args[i].clone());
                }
            },
            _ => {}
        }
        i += 1;
    }

    println!("I6: Split vs Single Bind Group Performance Demo");
    println!("Configuration: {} frames, {} objects", config.frames, config.objects);

    let ctx = TestContext::new(config).await?;

    // Run tests
    let split_test = SplitBindGroupTest::new(&ctx)?;
    let single_test = SingleBindGroupTest::new(&ctx)?;

    let split_metrics = run_test(
        &split_test,
        &ctx,
        "SplitBindGroups",
        |test, ctx, target| test.render_frame(ctx, target),
    )?;

    let single_metrics = run_test(
        &single_test,
        &ctx,
        "SingleBindGroup",
        |test, ctx, target| test.render_frame(ctx, target),
    )?;

    // Display results
    println!("\n=== PERFORMANCE RESULTS ===");
    println!("Split Bind Groups:");
    println!("  Total time: {:.3}ms", split_metrics.total_time_ms);
    println!("  Avg frame time: {:.3}ms", split_metrics.avg_frame_time_ms);
    println!("  Bind group switches: {}", split_metrics.bind_group_switches);

    println!("\nSingle Bind Group:");
    println!("  Total time: {:.3}ms", single_metrics.total_time_ms);
    println!("  Avg frame time: {:.3}ms", single_metrics.avg_frame_time_ms);
    println!("  Bind group switches: {}", single_metrics.bind_group_switches);

    let improvement = split_metrics.avg_frame_time_ms / single_metrics.avg_frame_time_ms;
    println!("\nImprovement: {:.2}x faster with single bind group", improvement);

    // Write CSV if requested
    if let Some(output_path) = &ctx.config.output_path {
        write_csv(&[split_metrics, single_metrics], output_path)?;
    }

    Ok(())
}

fn main() {
    env_logger::init();
    pollster::block_on(run_demo()).expect("Demo failed");
}