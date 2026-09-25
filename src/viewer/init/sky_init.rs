// src/viewer/init/sky_init.rs
// Sky pipeline initialization for the Viewer

use std::sync::Arc;
use wgpu::{BindGroupLayout, ComputePipeline, Device, RenderPipeline, TextureView};

use crate::core::error::RenderResult;
use crate::core::resource_tracker::{
    tracked_create_buffer_init, tracked_create_texture, TrackedBuffer, TrackedTexture,
};

use super::super::viewer_types::SkyUniforms;

/// Resources created during sky initialization
pub struct SkyResources {
    pub sky_bind_group_layout0: BindGroupLayout,
    pub sky_bind_group_layout1: BindGroupLayout,
    pub sky_pipeline: ComputePipeline,
    pub celestial_pipeline: RenderPipeline,
    pub sky_params: TrackedBuffer,
    pub sky_camera: TrackedBuffer,
    pub sky_output: TrackedTexture,
    pub sky_output_view: TextureView,
}

/// Create sky compute pipeline and resources
pub fn create_sky_resources(
    device: &Arc<Device>,
    width: u32,
    height: u32,
) -> RenderResult<SkyResources> {
    // Sky BGL0: params (binding 0) + output texture (binding 1)
    let sky_bgl0 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("viewer.sky.bgl0"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::StorageTexture {
                    access: wgpu::StorageTextureAccess::WriteOnly,
                    format: wgpu::TextureFormat::Rgba8Unorm,
                    view_dimension: wgpu::TextureViewDimension::D2,
                },
                count: None,
            },
        ],
    });

    // Sky BGL1: camera uniform (binding 0)
    let sky_bgl1 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("viewer.sky.bgl1"),
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::COMPUTE
                | wgpu::ShaderStages::VERTEX
                | wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }],
    });

    let sky_shader = crate::core::shader_registry::create_labeled_shader_module(
        device,
        "viewer.sky.shader",
        include_str!("../../shaders/sky.wgsl"),
    );

    let sky_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("viewer.sky.pl"),
        bind_group_layouts: &[&sky_bgl0, &sky_bgl1],
        push_constant_ranges: &[],
    });

    let sky_pipeline =
        crate::core::shader_registry::with_error_scope(device, "viewer.sky.pipeline", || {
            crate::core::shader_registry::create_compute_pipeline_scoped(
                device,
                &wgpu::ComputePipelineDescriptor {
                    label: Some("viewer.sky.pipeline"),
                    layout: Some(&sky_pl),
                    module: &sky_shader,
                    entry_point: "cs_render_sky",
                },
            )
        });

    let celestial_shader = crate::core::shader_registry::create_labeled_shader_module(
        device,
        "viewer.celestial.shader",
        include_str!("../../shaders/stars.wgsl"),
    );
    let celestial_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("viewer.celestial.layout"),
        bind_group_layouts: &[&sky_bgl1],
        push_constant_ranges: &[],
    });
    device.push_error_scope(wgpu::ErrorFilter::Validation);
    let celestial_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("viewer.celestial.pipeline"),
        layout: Some(&celestial_layout),
        vertex: wgpu::VertexState {
            module: &celestial_shader,
            entry_point: "vs_celestial",
            buffers: &[wgpu::VertexBufferLayout {
                array_stride: std::mem::size_of::<crate::astro::render::SkyInstance>() as u64,
                step_mode: wgpu::VertexStepMode::Instance,
                attributes: &[
                    wgpu::VertexAttribute {
                        format: wgpu::VertexFormat::Float32x4,
                        offset: 0,
                        shader_location: 0,
                    },
                    wgpu::VertexAttribute {
                        format: wgpu::VertexFormat::Float32x4,
                        offset: 16,
                        shader_location: 1,
                    },
                    wgpu::VertexAttribute {
                        format: wgpu::VertexFormat::Float32x4,
                        offset: 32,
                        shader_location: 2,
                    },
                ],
            }],
        },
        fragment: Some(wgpu::FragmentState {
            module: &celestial_shader,
            entry_point: "fs_celestial",
            targets: &[Some(wgpu::ColorTargetState {
                format: wgpu::TextureFormat::Rgba8Unorm,
                blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        multiview: None,
    });
    if let Some(error) = pollster::block_on(device.pop_error_scope()) {
        return Err(crate::core::error::RenderError::Render(format!(
            "SIDERA celestial pipeline validation failed: {error}"
        )));
    }

    let sky_params_data = SkyUniforms::new([0.3, 0.8, -0.5], 2.0, 0.3, 1.0, 5.0, 1.0, 0);
    let sky_params = tracked_create_buffer_init(
        device,
        &wgpu::util::BufferInitDescriptor {
            label: Some("viewer.sky.params"),
            contents: bytemuck::bytes_of(&sky_params_data),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        },
    )?;

    // Sky camera buffer - matches CameraUniforms struct in sky.wgsl (272 bytes)
    // Layout: view(64) + proj(64) + inv_view(64) + inv_proj(64) + eye_position(12) + viewport_height(4)
    let sky_camera_data: [f32; 68] = [0.0; 68]; // 272 bytes
    let sky_camera = tracked_create_buffer_init(
        device,
        &wgpu::util::BufferInitDescriptor {
            label: Some("viewer.sky.camera"),
            contents: bytemuck::cast_slice(&sky_camera_data),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        },
    )?;

    // Sky output texture
    let sky_output = tracked_create_texture(
        device,
        &wgpu::TextureDescriptor {
            label: Some("viewer.sky.output"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        },
    )?;
    let sky_output_view = sky_output.create_view(&wgpu::TextureViewDescriptor::default());

    Ok(SkyResources {
        sky_bind_group_layout0: sky_bgl0,
        sky_bind_group_layout1: sky_bgl1,
        sky_pipeline,
        celestial_pipeline,
        sky_params,
        sky_camera,
        sky_output,
        sky_output_view,
    })
}

#[cfg(test)]
mod tests {
    use super::create_sky_resources;
    use std::sync::Arc;

    #[test]
    fn creates_sky_pipeline_when_adapter_available() {
        let instance = wgpu::Instance::default();
        let Some(adapter) =
            pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
        else {
            eprintln!("No GPU adapter available, skipping viewer sky pipeline test");
            return;
        };
        let Ok((device, _queue)) =
            pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default(), None))
        else {
            eprintln!("Could not request GPU device, skipping viewer sky pipeline test");
            return;
        };

        let _resources = create_sky_resources(&Arc::new(device), 16, 16).expect("sky resources");
    }

    #[test]
    fn fixed_night_sky_gpu_golden_is_repeatable() {
        use crate::astro::{render, time::UtcDateTime};
        use crate::core::resource_tracker::tracked_create_buffer_init;
        use crate::geo::units::{Angle, Degree};
        use crate::renderer::readback::read_texture_tight;
        use crate::viewer::viewer_types::SkyUniforms;
        use glam::{Mat4, Vec3};
        use sha2::{Digest, Sha256};
        use wgpu::util::BufferInitDescriptor;

        const SIZE: u32 = 1024;
        let instance = wgpu::Instance::default();
        let adapter =
            pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
                .expect("SIDERA night golden requires a real GPU adapter");
        let info = adapter.get_info();
        let (device, queue) =
            pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default(), None))
                .expect("SIDERA night golden requires a GPU device");
        let device = Arc::new(device);
        let resources = create_sky_resources(&device, SIZE, SIZE).expect("night sky pipelines");
        let observation = render::prepare(
            UtcDateTime::parse("2026-09-25T22:00:00Z").unwrap(),
            Angle::<Degree>::new(52.37),
            Angle::<Degree>::new(4.9),
        )
        .unwrap();
        let direction = |azimuth_deg: f64, altitude_deg: f64| {
            let (sa, ca) = azimuth_deg.to_radians().sin_cos();
            let (sh, ch) = altitude_deg.to_radians().sin_cos();
            Vec3::new((ch * sa) as f32, sh as f32, (ch * ca) as f32)
        };
        let sun = direction(
            observation.sun.azimuth.value(),
            observation.sun.altitude.value(),
        );
        let moon = direction(
            observation.moon.azimuth.value(),
            observation.moon.altitude.value(),
        );
        let mut params = SkyUniforms::new(sun.to_array(), 2.5, 0.2, 1.0, 0.0, 1.0, 1);
        params.night_params = [moon.x, moon.y, moon.z, observation.moonlight_relative];
        queue.write_buffer(&resources.sky_params, 0, bytemuck::bytes_of(&params));
        let view = Mat4::look_at_rh(Vec3::ZERO, moon, Vec3::Y);
        let proj = Mat4::perspective_rh(45.0_f32.to_radians(), 1.0, 0.1, 1000.0);
        let matrices = [view, proj, view.inverse(), proj.inverse()];
        let mut camera = [0.0_f32; 68];
        for (index, matrix) in matrices.into_iter().enumerate() {
            camera[index * 16..index * 16 + 16].copy_from_slice(&matrix.to_cols_array());
        }
        camera[67] = SIZE as f32;
        queue.write_buffer(&resources.sky_camera, 0, bytemuck::cast_slice(&camera));
        let bg0 = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("sidera.night.bg0"),
            layout: &resources.sky_bind_group_layout0,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: resources.sky_params.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&resources.sky_output_view),
                },
            ],
        });
        let bg1 = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("sidera.night.bg1"),
            layout: &resources.sky_bind_group_layout1,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: resources.sky_camera.as_entire_binding(),
            }],
        });
        let points = tracked_create_buffer_init(
            &device,
            &BufferInitDescriptor {
                label: Some("sidera.night.points"),
                contents: bytemuck::cast_slice(&observation.instances),
                usage: wgpu::BufferUsages::VERTEX,
            },
        )
        .unwrap();
        let render_once = || {
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("sidera.night.encoder"),
            });
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("sidera.night.sky"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&resources.sky_pipeline);
                pass.set_bind_group(0, &bg0, &[]);
                pass.set_bind_group(1, &bg1, &[]);
                pass.dispatch_workgroups(SIZE / 8, SIZE / 8, 1);
            }
            {
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("sidera.night.celestial"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &resources.sky_output_view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Load,
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                });
                pass.set_pipeline(&resources.celestial_pipeline);
                pass.set_bind_group(0, &bg1, &[]);
                pass.set_vertex_buffer(0, points.slice(..));
                pass.draw(0..6, 0..observation.instances.len() as u32);
            }
            queue.submit(std::iter::once(encoder.finish()));
            read_texture_tight(
                &device,
                &queue,
                &resources.sky_output,
                (SIZE, SIZE),
                wgpu::TextureFormat::Rgba8Unorm,
            )
            .unwrap()
        };
        let first = render_once();
        let second = render_once();
        assert_eq!(first, second, "two actual GPU night renders differ");
        let digest = format!("{:x}", Sha256::digest(&first));
        println!(
            "SIDERA night GPU {} {:?}, 1024x1024 RGBA SHA-256 {}",
            info.name, info.backend, digest
        );
        let golden_dir =
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/golden/sidera");
        let golden_path = golden_dir.join("night.png");
        let certificate_path = golden_dir.join("night_certificate.json");
        if std::env::var_os("SIDERA_WRITE_NIGHT_GOLDEN").is_some() {
            assert!(
                !golden_path.exists(),
                "new SIDERA golden already exists; changing it requires owner approval"
            );
            std::fs::create_dir_all(&golden_dir).unwrap();
            image::save_buffer(&golden_path, &first, SIZE, SIZE, image::ColorType::Rgba8).unwrap();
            let certificate = serde_json::json!({
                "utc": "2026-09-25T22:00:00Z", "latitude_deg": 52.37, "longitude_deg": 4.9,
                "width": SIZE, "height": SIZE, "rgba_sha256": digest,
                "adapter": info.name, "backend": format!("{:?}", info.backend),
                "twilight_model": "civil-to-astronomical solar-depression visual ramp (-6 to -18 degrees)",
                "moonlight_model": "Krisciunas & Schaefer (1991) Eq. 9 phase magnitude, 0.172 V-mag/airmass clear-sky extinction, relative directional display gain 0.12",
                "catalog": "Yale Bright Star Catalogue V/50, J2000, 9096 stars",
                "theories": "VSOP87D and ELP2000-82B; IAU 2006/2000B frames",
            });
            std::fs::write(
                &certificate_path,
                serde_json::to_vec_pretty(&certificate).unwrap(),
            )
            .unwrap();
        } else {
            let certificate: serde_json::Value = serde_json::from_slice(
                &std::fs::read(&certificate_path).expect("committed SIDERA night certificate"),
            )
            .unwrap();
            let qualified_backend = format!("{:?}", info.backend);
            if certificate["adapter"].as_str() == Some(info.name.as_str())
                && certificate["backend"].as_str() == Some(qualified_backend.as_str())
            {
                assert_eq!(certificate["rgba_sha256"].as_str(), Some(digest.as_str()));
                let golden = image::open(&golden_path)
                    .expect("committed SIDERA night golden")
                    .into_rgba8();
                assert_eq!(golden.as_raw(), &first);
            } else {
                println!(
                    "SIDERA night golden ABSENT for adapter {} {:?}; two-run GPU determinism passed",
                    info.name, info.backend
                );
            }
        }
    }
}
