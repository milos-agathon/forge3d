//! TBN (Tangent-Bitangent-Normal) vertex layout binding validation test
//!
//! This test validates that the TBN vertex buffer layout binds correctly
//! without wgpu validation errors during render pipeline creation.

use wgpu::*;

/// Test that TBN vertex layout with stride 56 binds without validation errors
#[tokio::test]
async fn test_tbn_vertex_layout_binding() {
    // Gate test with enable-tbn feature
    #[cfg(not(feature = "enable-tbn"))]
    {
        println!("Test skipped: enable-tbn feature not enabled");
        return;
    }

    #[cfg(feature = "enable-tbn")]
    {
        // Create headless wgpu instance
        let instance = Instance::new(InstanceDescriptor {
            backends: Backends::all(),
            ..Default::default()
        });

        // Get adapter (headless)
        let adapter = instance
            .request_adapter(&RequestAdapterOptions {
                power_preference: PowerPreference::LowPower,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .expect("Failed to get adapter");

        // Create device
        let (device, queue) = adapter
            .request_device(
                &DeviceDescriptor {
                    label: Some("TBN Test Device"),
                    required_features: Features::empty(),
                    required_limits: Limits::downlevel_defaults(),
                },
                None,
            )
            .await
            .expect("Failed to create device");

        // Push validation error scope before pipeline creation
        device.push_error_scope(ErrorFilter::Validation);

        // Define TBN vertex buffer layout with stride 56
        let vertex_buffer_layout = VertexBufferLayout {
            array_stride: 56, // Total size of TBN vertex
            step_mode: VertexStepMode::Vertex,
            attributes: &[
                // Position: Float32x3 at location 0, offset 0
                VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: VertexFormat::Float32x3,
                },
                // UV: Float32x2 at location 1, offset 12
                VertexAttribute {
                    offset: 12,
                    shader_location: 1,
                    format: VertexFormat::Float32x2,
                },
                // Normal: Float32x3 at location 2, offset 20
                VertexAttribute {
                    offset: 20,
                    shader_location: 2,
                    format: VertexFormat::Float32x3,
                },
                // Tangent: Float32x3 at location 3, offset 32
                VertexAttribute {
                    offset: 32,
                    shader_location: 3,
                    format: VertexFormat::Float32x3,
                },
                // Bitangent: Float32x3 at location 4, offset 44
                VertexAttribute {
                    offset: 44,
                    shader_location: 4,
                    format: VertexFormat::Float32x3,
                },
            ],
        };

        // Minimal WGSL shader consuming locations 0..4
        let shader_source = r#"
            struct VertexInput {
                @location(0) position: vec3<f32>,
                @location(1) uv: vec2<f32>,
                @location(2) normal: vec3<f32>,
                @location(3) tangent: vec3<f32>,
                @location(4) bitangent: vec3<f32>,
            }

            struct VertexOutput {
                @builtin(position) position: vec4<f32>,
                @location(0) color: vec3<f32>,
            }

            @vertex
            fn vs_main(input: VertexInput) -> VertexOutput {
                var output: VertexOutput;
                output.position = vec4<f32>(input.position, 1.0);
                
                // Use TBN vectors to create a simple color
                output.color = normalize(input.normal) * 0.5 + 0.5;
                
                return output;
            }

            @fragment
            fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
                return vec4<f32>(input.color, 1.0);
            }
        "#;

        let shader = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("TBN Test Shader"),
            source: ShaderSource::Wgsl(shader_source.into()),
        });

        // Create render pipeline with sRGB color target
        let render_pipeline = device.create_render_pipeline(&RenderPipelineDescriptor {
            label: Some("TBN Test Pipeline"),
            layout: Some(&device.create_pipeline_layout(&PipelineLayoutDescriptor {
                label: Some("TBN Test Pipeline Layout"),
                bind_group_layouts: &[],
                push_constant_ranges: &[],
            })),
            vertex: VertexState {
                module: &shader,
                entry_point: "vs_main",
                buffers: &[vertex_buffer_layout],
            },
            fragment: Some(FragmentState {
                module: &shader,
                entry_point: "fs_main",
                targets: &[Some(ColorTargetState {
                    format: TextureFormat::Rgba8UnormSrgb, // sRGB color target
                    blend: Some(BlendState::REPLACE),
                    write_mask: ColorWrites::ALL,
                })],
            }),
            primitive: PrimitiveState {
                topology: PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: FrontFace::Ccw,
                cull_mode: Some(Face::Back),
                unclipped_depth: false,
                polygon_mode: PolygonMode::Fill,
                conservative: false,
            },
            depth_stencil: None,
            multisample: MultisampleState::default(),
            multiview: None,
        });

        // Pop error scope and verify no validation errors
        let validation_error = pollster::block_on(device.pop_error_scope());
        
        // Assert no validation errors occurred and print status
        assert!(validation_error.is_none(), "Validation error occurred: {:?}", validation_error);
        println!("wgpu_validation=None");

        // Verify pipeline was created successfully
        drop(render_pipeline);
    }
}