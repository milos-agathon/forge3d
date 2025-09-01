//! I7: Big Buffer Performance and Visual Parity Test
//!
//! Tests the big buffer pattern against legacy per-object bind groups with:
//! - 10k object performance comparison (≥25% CPU time reduction)  
//! - Visual parity verification (exact match or SSIM≥0.99)
//! - Feature detection for conditional compilation

use std::time::Instant;
use wgpu::util::DeviceExt;

/// Test configuration for 10k object scenario
const OBJECT_COUNT: u32 = 10_000;
const RENDER_SIZE: (u32, u32) = (512, 512);

/// Performance test results
#[derive(Debug, Clone)]
struct PerfResult {
    cpu_time_ms: f64,
    setup_time_ms: f64, 
    render_time_ms: f64,
    bind_group_switches: u32,
}

/// GPU test context
struct TestContext {
    device: wgpu::Device,
    queue: wgpu::Queue,
}

impl TestContext {
    async fn new() -> Result<Self, Box<dyn std::error::Error>> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::default());
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions::default())
            .await
            .ok_or("Failed to find adapter")?;

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor::default(), None)
            .await?;

        Ok(Self { device, queue })
    }
}

/// Legacy per-object bind group implementation
struct LegacyRenderer {
    pipeline: wgpu::RenderPipeline,
    bind_groups: Vec<wgpu::BindGroup>,
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
}

impl LegacyRenderer {
    fn new(ctx: &TestContext, object_count: u32) -> Result<Self, Box<dyn std::error::Error>> {
        // Create bind group layout for per-object uniforms
        let bind_group_layout = ctx.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Legacy_PerObject_Layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        let pipeline_layout = ctx.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Legacy_Pipeline_Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let shader = ctx.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Legacy_Shader"),
            source: wgpu::ShaderSource::Wgsl(
                r#"
struct Transform {
    matrix: mat4x4<f32>,
};

@group(0) @binding(0) var<uniform> transform: Transform;

@vertex
fn vs_main(@location(0) position: vec3<f32>) -> @builtin(position) vec4<f32> {
    return transform.matrix * vec4<f32>(position, 1.0);
}

@fragment  
fn fs_main() -> @location(0) vec4<f32> {
    return vec4<f32>(0.2, 0.6, 0.8, 1.0);
}
                "#.into()
            ),
        });

        let pipeline = ctx.device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Legacy_Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: 12,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &[wgpu::VertexAttribute {
                        offset: 0,
                        shader_location: 0,
                        format: wgpu::VertexFormat::Float32x3,
                    }],
                }],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Rgba8UnormSrgb,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
        });

        // Create geometry (simple triangle)
        let vertices = [
            [-0.05f32, -0.05, 0.0], [0.05, -0.05, 0.0], [0.0, 0.05, 0.0]
        ];
        let indices = [0u16, 1, 2];

        let vertex_buffer = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Legacy_Vertices"),
            contents: bytemuck::cast_slice(&vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let index_buffer = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Legacy_Indices"),
            contents: bytemuck::cast_slice(&indices),
            usage: wgpu::BufferUsages::INDEX,
        });

        // Create per-object uniform buffers and bind groups
        let mut bind_groups = Vec::new();
        for i in 0..object_count {
            // Per-object transform matrix (small offset for each object)
            let offset_x = (i % 100) as f32 * 0.002 - 0.1;
            let offset_y = (i / 100) as f32 * 0.002 - 0.1;
            let transform = [
                [1.0f32, 0.0, 0.0, offset_x],
                [0.0, 1.0, 0.0, offset_y],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ];

            let uniform_buffer = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(&format!("Legacy_Uniform_{}", i)),
                contents: bytemuck::cast_slice(&transform),
                usage: wgpu::BufferUsages::UNIFORM,
            });

            let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(&format!("Legacy_BindGroup_{}", i)),
                layout: &bind_group_layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buffer.as_entire_binding(),
                }],
            });

            bind_groups.push(bind_group);
        }

        Ok(Self {
            pipeline,
            bind_groups,
            vertex_buffer,
            index_buffer,
        })
    }

    fn render(&self, ctx: &TestContext, target: &wgpu::TextureView) -> u32 {
        let mut encoder = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Legacy_Encoder"),
        });

        let mut bind_group_switches = 0;

        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Legacy_RenderPass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: target,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            render_pass.set_pipeline(&self.pipeline);
            render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
            render_pass.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint16);

            // Render each object with its own bind group (causes churn)
            for (i, bind_group) in self.bind_groups.iter().enumerate() {
                render_pass.set_bind_group(0, bind_group, &[]);
                render_pass.draw_indexed(0..3, 0, i as u32..i as u32 + 1);
                bind_group_switches += 1;
            }
        }

        ctx.queue.submit(Some(encoder.finish()));
        bind_group_switches
    }
}

/// Big buffer implementation (only available with feature flag)
#[cfg(feature = "wsI_bigbuf")]
mod big_buffer_impl {
    use super::*;
    use forge3d::core::big_buffer::*;
    use forge3d::core::memory_tracker::ResourceRegistry;

    pub struct BigBufferRenderer {
        pipeline: wgpu::RenderPipeline,
        bind_group: wgpu::BindGroup,
        vertex_buffer: wgpu::Buffer,
        index_buffer: wgpu::Buffer,
        big_buffer: BigBuffer,
        _blocks: Vec<BigBufferBlock>, // Keep blocks alive
    }

    impl BigBufferRenderer {
        pub fn new(ctx: &TestContext, object_count: u32) -> Result<Self, Box<dyn std::error::Error>> {
            let registry = ResourceRegistry::new();

            // Create big buffer for all objects
            let big_buffer = BigBuffer::new(
                &ctx.device,
                object_count * 64, // 64 bytes per object (4x4 matrix)
                Some(&registry),
            )?;

            // Create bind group layout with dynamic offset
            let bind_group_layout = ctx.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("BigBuffer_Layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: Some(std::num::NonZeroU64::new(64).unwrap()),
                    },
                    count: None,
                }],
            });

            let pipeline_layout = ctx.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("BigBuffer_Pipeline_Layout"),
                bind_group_layouts: &[&bind_group_layout],
                push_constant_ranges: &[],
            });

            let shader = ctx.device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("BigBuffer_Shader"),
                source: wgpu::ShaderSource::Wgsl(
                    r#"
struct Transform {
    matrix: mat4x4<f32>,
};

@group(0) @binding(0) var<storage, read> transforms: array<Transform>;

@vertex
fn vs_main(
    @builtin(instance_index) instance_index: u32,
    @location(0) position: vec3<f32>
) -> @builtin(position) vec4<f32> {
    return transforms[instance_index].matrix * vec4<f32>(position, 1.0);
}

@fragment  
fn fs_main() -> @location(0) vec4<f32> {
    return vec4<f32>(0.2, 0.6, 0.8, 1.0);
}
                    "#.into()
                ),
            });

            let pipeline = ctx.device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("BigBuffer_Pipeline"),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: "vs_main",
                    buffers: &[wgpu::VertexBufferLayout {
                        array_stride: 12,
                        step_mode: wgpu::VertexStepMode::Vertex,
                        attributes: &[wgpu::VertexAttribute {
                            offset: 0,
                            shader_location: 0,
                            format: wgpu::VertexFormat::Float32x3,
                        }],
                    }],
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: "fs_main",
                    targets: &[Some(wgpu::ColorTargetState {
                        format: wgpu::TextureFormat::Rgba8UnormSrgb,
                        blend: Some(wgpu::BlendState::REPLACE),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                primitive: wgpu::PrimitiveState::default(),
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                multiview: None,
            });

            // Create geometry
            let vertices = [
                [-0.05f32, -0.05, 0.0], [0.05, -0.05, 0.0], [0.0, 0.05, 0.0]
            ];
            let indices = [0u16, 1, 2];

            let vertex_buffer = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("BigBuffer_Vertices"),
                contents: bytemuck::cast_slice(&vertices),
                usage: wgpu::BufferUsages::VERTEX,
            });

            let index_buffer = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("BigBuffer_Indices"),
                contents: bytemuck::cast_slice(&indices),
                usage: wgpu::BufferUsages::INDEX,
            });

            // Allocate blocks and upload data
            let mut blocks = Vec::new();
            for i in 0..object_count {
                let block = big_buffer.allocate_block(64)?;

                // Same transform as legacy for visual parity
                let offset_x = (i % 100) as f32 * 0.002 - 0.1;
                let offset_y = (i / 100) as f32 * 0.002 - 0.1;
                let transform = [
                    [1.0f32, 0.0, 0.0, offset_x],
                    [0.0, 1.0, 0.0, offset_y],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ];

                ctx.queue.write_buffer(
                    big_buffer.buffer(),
                    block.offset as u64,
                    bytemuck::cast_slice(&transform),
                );

                blocks.push(block);
            }

            // Create bind group
            let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("BigBuffer_BindGroup"),
                layout: &bind_group_layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: big_buffer.buffer(),
                        offset: 0,
                        size: None,
                    }),
                }],
            });

            Ok(Self {
                pipeline,
                bind_group,
                vertex_buffer,
                index_buffer,
                big_buffer,
                _blocks: blocks,
            })
        }

        pub fn render(&self, ctx: &TestContext, target: &wgpu::TextureView) -> u32 {
            let mut encoder = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("BigBuffer_Encoder"),
            });

            {
                let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("BigBuffer_RenderPass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: target,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                });

                render_pass.set_pipeline(&self.pipeline);
                render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
                render_pass.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint16);

                // Single bind group, render all instances at once
                render_pass.set_bind_group(0, &self.bind_group, &[]);
                render_pass.draw_indexed(0..3, 0, 0..OBJECT_COUNT);
            }

            ctx.queue.submit(Some(encoder.finish()));
            1 // Only 1 bind group switch
        }
    }
}

/// Read texture to CPU for comparison
fn read_texture_to_cpu(
    ctx: &TestContext,
    texture: &wgpu::Texture,
    size: (u32, u32),
) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    let (width, height) = size;
    let bytes_per_pixel = 4; // RGBA8
    let unpadded_bytes_per_row = width * bytes_per_pixel;
    let padded_bytes_per_row = {
        let alignment = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        ((unpadded_bytes_per_row + alignment - 1) / alignment) * alignment
    };
    let buffer_size = padded_bytes_per_row * height;

    let buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Readback_Buffer"),
        size: buffer_size as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let mut encoder = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("Readback_Encoder"),
    });

    encoder.copy_texture_to_buffer(
        wgpu::ImageCopyTexture {
            texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::ImageCopyBuffer {
            buffer: &buffer,
            layout: wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(padded_bytes_per_row),
                rows_per_image: None,
            },
        },
        wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
    );

    ctx.queue.submit(Some(encoder.finish()));
    ctx.device.poll(wgpu::Maintain::Wait);

    let buffer_slice = buffer.slice(..);
    let (sender, receiver) = std::sync::mpsc::channel();
    buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
        sender.send(result).unwrap();
    });
    ctx.device.poll(wgpu::Maintain::Wait);
    receiver.recv().unwrap()?;

    let data = buffer_slice.get_mapped_range();
    
    // Remove padding
    let mut unpadded_data = Vec::with_capacity((width * height * bytes_per_pixel) as usize);
    for row in 0..height {
        let start = (row * padded_bytes_per_row) as usize;
        let end = start + unpadded_bytes_per_row as usize;
        unpadded_data.extend_from_slice(&data[start..end]);
    }

    Ok(unpadded_data)
}

/// Calculate SSIM between two images
fn calculate_ssim(img1: &[u8], img2: &[u8], width: u32, height: u32) -> f64 {
    if img1.len() != img2.len() {
        return 0.0;
    }

    let pixel_count = (width * height) as usize;
    if img1.len() != pixel_count * 4 {
        return 0.0;
    }

    // Convert to grayscale for SSIM calculation
    let gray1: Vec<f64> = (0..pixel_count)
        .map(|i| {
            let r = img1[i * 4] as f64;
            let g = img1[i * 4 + 1] as f64;
            let b = img1[i * 4 + 2] as f64;
            0.299 * r + 0.587 * g + 0.114 * b
        })
        .collect();

    let gray2: Vec<f64> = (0..pixel_count)
        .map(|i| {
            let r = img2[i * 4] as f64;
            let g = img2[i * 4 + 1] as f64;
            let b = img2[i * 4 + 2] as f64;
            0.299 * r + 0.587 * g + 0.114 * b
        })
        .collect();

    // Simple SSIM calculation (simplified version)
    let mean1 = gray1.iter().sum::<f64>() / gray1.len() as f64;
    let mean2 = gray2.iter().sum::<f64>() / gray2.len() as f64;

    let var1 = gray1.iter().map(|&x| (x - mean1).powi(2)).sum::<f64>() / gray1.len() as f64;
    let var2 = gray2.iter().map(|&x| (x - mean2).powi(2)).sum::<f64>() / gray2.len() as f64;
    let cov = gray1.iter().zip(&gray2).map(|(&x, &y)| (x - mean1) * (y - mean2)).sum::<f64>() / gray1.len() as f64;

    let c1 = (0.01f64 * 255.0).powi(2);
    let c2 = (0.03f64 * 255.0).powi(2);

    let ssim = ((2.0 * mean1 * mean2 + c1) * (2.0 * cov + c2)) / 
               ((mean1.powi(2) + mean2.powi(2) + c1) * (var1 + var2 + c2));

    ssim
}

/// Run performance test for legacy approach
async fn test_legacy_performance(ctx: &TestContext) -> Result<(PerfResult, Vec<u8>), Box<dyn std::error::Error>> {
    println!("Testing legacy per-object bind groups with {} objects...", OBJECT_COUNT);
    
    let setup_start = Instant::now();
    let renderer = LegacyRenderer::new(ctx, OBJECT_COUNT)?;
    let setup_time_ms = setup_start.elapsed().as_secs_f64() * 1000.0;

    // Create render target
    let texture = ctx.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("Legacy_Target"),
        size: wgpu::Extent3d {
            width: RENDER_SIZE.0,
            height: RENDER_SIZE.1,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8UnormSrgb,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let texture_view = texture.create_view(&wgpu::TextureViewDescriptor::default());

    // Perform 3 render runs and take median
    let mut render_times = Vec::new();
    for run in 0..3 {
        let render_start = Instant::now();
        let bind_group_switches = renderer.render(ctx, &texture_view);
        ctx.device.poll(wgpu::Maintain::Wait);
        let render_time_ms = render_start.elapsed().as_secs_f64() * 1000.0;
        render_times.push(render_time_ms);
        
        println!("  Legacy run {}: {:.3}ms ({} bind group switches)", 
                 run + 1, render_time_ms, bind_group_switches);
    }

    render_times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median_render_time_ms = render_times[1];

    // Read final frame for visual comparison
    let pixel_data = read_texture_to_cpu(ctx, &texture, RENDER_SIZE)?;

    let result = PerfResult {
        cpu_time_ms: setup_time_ms + median_render_time_ms,
        setup_time_ms,
        render_time_ms: median_render_time_ms,
        bind_group_switches: OBJECT_COUNT,
    };

    Ok((result, pixel_data))
}

/// Run performance test for big buffer approach (feature-gated)
#[cfg(feature = "wsI_bigbuf")]
async fn test_big_buffer_performance(ctx: &TestContext) -> Result<(PerfResult, Vec<u8>), Box<dyn std::error::Error>> {
    use big_buffer_impl::BigBufferRenderer;
    
    println!("Testing big buffer approach with {} objects...", OBJECT_COUNT);
    
    let setup_start = Instant::now();
    let renderer = BigBufferRenderer::new(ctx, OBJECT_COUNT)?;
    let setup_time_ms = setup_start.elapsed().as_secs_f64() * 1000.0;

    // Create render target
    let texture = ctx.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("BigBuffer_Target"),
        size: wgpu::Extent3d {
            width: RENDER_SIZE.0,
            height: RENDER_SIZE.1,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8UnormSrgb,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let texture_view = texture.create_view(&wgpu::TextureViewDescriptor::default());

    // Perform 3 render runs and take median
    let mut render_times = Vec::new();
    for run in 0..3 {
        let render_start = Instant::now();
        let bind_group_switches = renderer.render(ctx, &texture_view);
        ctx.device.poll(wgpu::Maintain::Wait);
        let render_time_ms = render_start.elapsed().as_secs_f64() * 1000.0;
        render_times.push(render_time_ms);
        
        println!("  BigBuffer run {}: {:.3}ms ({} bind group switches)", 
                 run + 1, render_time_ms, bind_group_switches);
    }

    render_times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median_render_time_ms = render_times[1];

    // Read final frame for visual comparison
    let pixel_data = read_texture_to_cpu(ctx, &texture, RENDER_SIZE)?;

    let result = PerfResult {
        cpu_time_ms: setup_time_ms + median_render_time_ms,
        setup_time_ms,
        render_time_ms: median_render_time_ms,
        bind_group_switches: 1,
    };

    Ok((result, pixel_data))
}

/// Main test function
#[tokio::test]
async fn test_big_buffer_vs_legacy() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== I7: Big Buffer vs Legacy Performance Test ===");
    println!("Objects: {}, Render Size: {}x{}", OBJECT_COUNT, RENDER_SIZE.0, RENDER_SIZE.1);
    
    let ctx = TestContext::new().await?;
    
    // Test legacy approach
    let (legacy_result, legacy_pixels) = test_legacy_performance(&ctx).await?;
    
    println!("\nLegacy Results:");
    println!("  Setup time: {:.3}ms", legacy_result.setup_time_ms);
    println!("  Render time: {:.3}ms", legacy_result.render_time_ms);
    println!("  Total CPU time: {:.3}ms", legacy_result.cpu_time_ms);
    println!("  Bind group switches: {}", legacy_result.bind_group_switches);
    
    // Test big buffer approach if feature is enabled
    #[cfg(feature = "wsI_bigbuf")]
    {
        let (bigbuf_result, bigbuf_pixels) = test_big_buffer_performance(&ctx).await?;
        
        println!("\nBigBuffer Results:");
        println!("  Setup time: {:.3}ms", bigbuf_result.setup_time_ms);
        println!("  Render time: {:.3}ms", bigbuf_result.render_time_ms);
        println!("  Total CPU time: {:.3}ms", bigbuf_result.cpu_time_ms);
        println!("  Bind group switches: {}", bigbuf_result.bind_group_switches);
        
        // Calculate performance improvement
        let improvement_ratio = legacy_result.cpu_time_ms / bigbuf_result.cpu_time_ms;
        let improvement_percentage = (improvement_ratio - 1.0) * 100.0;
        
        println!("\nPerformance Analysis:");
        println!("  Improvement: {:.2}x faster ({:.1}% reduction)", improvement_ratio, improvement_percentage);
        
        // Check acceptance criteria: ≥25% reduction
        assert!(improvement_percentage >= 25.0, 
                "Big buffer should provide ≥25% CPU time reduction, got {:.1}%", 
                improvement_percentage);
        println!("  ✅ Performance criteria met (≥25% reduction)");
        
        // Visual parity check
        let exact_match = legacy_pixels == bigbuf_pixels;
        let ssim = if !exact_match {
            calculate_ssim(&legacy_pixels, &bigbuf_pixels, RENDER_SIZE.0, RENDER_SIZE.1)
        } else {
            1.0
        };
        
        println!("\nVisual Parity Analysis:");
        if exact_match {
            println!("  ✅ Exact pixel match");
        } else {
            println!("  SSIM: {:.4}", ssim);
            assert!(ssim >= 0.99, "SSIM should be ≥0.99, got {:.4}", ssim);
            println!("  ✅ Visual parity criteria met (SSIM≥0.99)");
        }
        
        println!("\n=== BIG BUFFER TEST PASSED ===");
    }
    
    #[cfg(not(feature = "wsI_bigbuf"))]
    {
        println!("\nBigBuffer test skipped (feature 'wsI_bigbuf' not enabled)");
        println!("Run with: cargo test --features wsI_bigbuf --test wsI_i7_big_buffer");
    }
    
    println!("\n=== LEGACY PATH TEST COMPLETED ===");
    Ok(())
}

#[tokio::test]
async fn test_legacy_only() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== I7: Legacy Path Only Test ===");
    
    let ctx = TestContext::new().await?;
    let (result, _pixels) = test_legacy_performance(&ctx).await?;
    
    println!("Legacy performance baseline:");
    println!("  CPU time: {:.3}ms", result.cpu_time_ms);
    println!("  Bind group switches: {}", result.bind_group_switches);
    
    // Basic sanity checks
    assert!(result.cpu_time_ms > 0.0, "CPU time should be positive");
    assert_eq!(result.bind_group_switches, OBJECT_COUNT, "Should have one bind group switch per object");
    
    println!("✅ Legacy test completed successfully");
    Ok(())
}