use super::{CsmRenderer, CsmUniforms};

fn execute_msm_front_visibility(source: &str, expression: &str) -> [f32; 2] {
    let context = crate::core::gpu::try_ctx().expect("GPU context");
    let device = &context.device;
    let queue = &context.queue;
    let source = format!(
        "{source}
@vertex
fn test_msm_vertex(@builtin(vertex_index) vertex_index: u32) -> @builtin(position) vec4<f32> {{
    var position = vec2<f32>(-1.0, -1.0);
    if (vertex_index == 1u) {{
        position = vec2<f32>(1.0, -1.0);
    }} else if (vertex_index == 2u) {{
        position = vec2<f32>(0.0, 1.0);
    }}
    return vec4<f32>(position, 0.0, 1.0);
}}

@fragment
fn test_msm_fragment() -> @location(0) vec4<f32> {{
    let visibility = {expression};
    let sampled_mean =
        textureSample(moment_maps, moment_sampler, vec2<f32>(0.5), 0).r;
    return vec4<f32>(visibility, sampled_mean, 0.0, 1.0);
}}"
    );

    device.push_error_scope(wgpu::ErrorFilter::Validation);
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("msm-front-visibility-contract"),
        source: wgpu::ShaderSource::Wgsl(source.into()),
    });
    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("msm-front-visibility-contract"),
        layout: None,
        vertex: wgpu::VertexState {
            module: &module,
            entry_point: "test_msm_vertex",
            buffers: &[],
        },
        fragment: Some(wgpu::FragmentState {
            module: &module,
            entry_point: "test_msm_fragment",
            targets: &[Some(wgpu::ColorTargetState {
                format: wgpu::TextureFormat::Rgba8Unorm,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        multiview: None,
    });
    if let Some(error) = pollster::block_on(device.pop_error_scope()) {
        panic!("MSM visibility harness failed: {error}");
    }
    device.push_error_scope(wgpu::ErrorFilter::Validation);

    let render_target = crate::core::resource_tracker::tracked_create_texture(
        device,
        &wgpu::TextureDescriptor {
            label: Some("msm-front-visibility-output"),
            size: wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        },
    )
    .expect("render target");
    let render_target_view = render_target.create_view(&wgpu::TextureViewDescriptor::default());
    let moment_texture = crate::core::resource_tracker::tracked_create_texture(
        device,
        &wgpu::TextureDescriptor {
            label: Some("msm-front-moments"),
            size: wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        },
    )
    .expect("moment texture");
    let moments = [0.5_f32, 0.26, 0.125, 0.0625]
        .into_iter()
        .flat_map(|value| half::f16::from_f32(value).to_le_bytes())
        .collect::<Vec<_>>();
    queue.write_texture(
        wgpu::ImageCopyTexture {
            texture: &moment_texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        &moments,
        wgpu::ImageDataLayout {
            offset: 0,
            bytes_per_row: Some(8),
            rows_per_image: Some(1),
        },
        wgpu::Extent3d {
            width: 1,
            height: 1,
            depth_or_array_layers: 1,
        },
    );
    let moment_view = moment_texture.create_view(&wgpu::TextureViewDescriptor {
        label: Some("msm-front-moments"),
        dimension: Some(wgpu::TextureViewDimension::D2Array),
        ..Default::default()
    });
    let moment_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("msm-front-moments"),
        mag_filter: wgpu::FilterMode::Nearest,
        min_filter: wgpu::FilterMode::Nearest,
        ..Default::default()
    });
    let mut uniform_values = CsmUniforms::default();
    uniform_values.depth_bias = 0.0;
    uniform_values.slope_bias = 0.0;
    uniform_values.technique_params[2] = 0.0;
    let uniforms = crate::core::resource_tracker::tracked_create_buffer_init(
        device,
        &wgpu::util::BufferInitDescriptor {
            label: Some("msm-front-uniforms"),
            contents: bytemuck::bytes_of(&uniform_values),
            usage: wgpu::BufferUsages::UNIFORM,
        },
    )
    .expect("uniform buffer");

    let resource_entries = [
        wgpu::BindGroupEntry {
            binding: 0,
            resource: uniforms.as_entire_binding(),
        },
        wgpu::BindGroupEntry {
            binding: 3,
            resource: wgpu::BindingResource::TextureView(&moment_view),
        },
        wgpu::BindGroupEntry {
            binding: 4,
            resource: wgpu::BindingResource::Sampler(&moment_sampler),
        },
    ];
    let resource_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("msm-front-resources"),
        layout: &pipeline.get_bind_group_layout(2),
        entries: &resource_entries,
    });
    let empty_bind_groups = (0..2)
        .map(|group| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("msm-front-empty-resources"),
                layout: &pipeline.get_bind_group_layout(group),
                entries: &[],
            })
        })
        .collect::<Vec<_>>();

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("msm-front-visibility-contract"),
    });
    {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("msm-front-visibility-contract"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &render_target_view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            occlusion_query_set: None,
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        for (group, bind_group) in empty_bind_groups.iter().enumerate() {
            pass.set_bind_group(group as u32, bind_group, &[]);
        }
        pass.set_bind_group(2, &resource_bind_group, &[]);
        pass.draw(0..3, 0..1);
    }
    queue.submit(Some(encoder.finish()));
    device.poll(wgpu::Maintain::Wait);
    if let Some(error) = pollster::block_on(device.pop_error_scope()) {
        panic!("MSM visibility draw failed: {error}");
    }

    let pixel = crate::renderer::readback::read_texture_tight(
        device,
        queue,
        &render_target,
        (1, 1),
        wgpu::TextureFormat::Rgba8Unorm,
    )
    .expect("MSM visibility readback");
    [f32::from(pixel[0]) / 255.0, f32::from(pixel[1]) / 255.0]
}

#[test]
fn live_shared_msm_sampler_treats_front_of_mean_as_lit() {
    let [shared, shared_mean] = execute_msm_front_visibility(
        CsmRenderer::shader_source(),
        "sample_shadow_msm(vec4<f32>(0.0, 0.0, -0.5, 1.0), 0u, vec3<f32>(0.0, 1.0, 0.0))",
    );
    assert!((shared_mean - 0.5).abs() < 0.01, "shared mean upload");
    assert_eq!(shared, 1.0, "shared MSM front receiver must be lit");
}
