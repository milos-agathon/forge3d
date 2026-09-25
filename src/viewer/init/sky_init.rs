// src/viewer/init/sky_init.rs
// Sky pipeline initialization for the Viewer

use std::sync::Arc;
use wgpu::{BindGroup, BindGroupLayout, ComputePipeline, Device, RenderPipeline, TextureView};

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
    pub moon_albedo: TrackedTexture,
    pub moon_bind_group: BindGroup,
    pub sky_params: TrackedBuffer,
    pub sky_camera: TrackedBuffer,
    pub sky_output: TrackedTexture,
    pub sky_output_view: TextureView,
}

/// Create sky compute pipeline and resources
pub fn create_sky_resources(
    device: &Arc<Device>,
    queue: &wgpu::Queue,
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
    let moon_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("viewer.celestial.moon_albedo.layout"),
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
    let celestial_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("viewer.celestial.layout"),
        bind_group_layouts: &[&sky_bgl1, &moon_bgl],
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
                    wgpu::VertexAttribute {
                        format: wgpu::VertexFormat::Float32x4,
                        offset: 48,
                        shader_location: 3,
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

    const MOON_ALBEDO: &[u8; 256 * 256] = include_bytes!("../../../data/sidera/moon_albedo.bin");
    let moon_albedo = tracked_create_texture(
        device,
        &wgpu::TextureDescriptor {
            label: Some("viewer.celestial.moon_albedo"),
            size: wgpu::Extent3d {
                width: 256,
                height: 256,
                depth_or_array_layers: 1,
            },
            mip_level_count: 9,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        },
    )?;
    let mut mip = MOON_ALBEDO.to_vec();
    for level in 0..9 {
        let side = 256_u32 >> level;
        queue.write_texture(
            wgpu::ImageCopyTexture {
                texture: &moon_albedo,
                mip_level: level,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &mip,
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(side),
                rows_per_image: Some(side),
            },
            wgpu::Extent3d {
                width: side,
                height: side,
                depth_or_array_layers: 1,
            },
        );
        if side > 1 {
            let next_side = (side / 2) as usize;
            let stride = side as usize;
            mip = (0..next_side * next_side)
                .map(|index| {
                    let x = (index % next_side) * 2;
                    let y = (index / next_side) * 2;
                    let sum = u16::from(mip[y * stride + x])
                        + u16::from(mip[y * stride + x + 1])
                        + u16::from(mip[(y + 1) * stride + x])
                        + u16::from(mip[(y + 1) * stride + x + 1]);
                    ((sum + 2) / 4) as u8
                })
                .collect();
        }
    }
    let moon_view = moon_albedo.create_view(&wgpu::TextureViewDescriptor::default());
    let moon_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("viewer.celestial.moon_albedo.sampler"),
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        mipmap_filter: wgpu::FilterMode::Linear,
        ..Default::default()
    });
    let moon_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("viewer.celestial.moon_albedo.bind_group"),
        layout: &moon_bgl,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&moon_view),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Sampler(&moon_sampler),
            },
        ],
    });

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
        moon_albedo,
        moon_bind_group,
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
    fn night_sky_places_saturn_left_of_moon() {
        use crate::astro::{render, time::UtcDateTime};
        use crate::geo::units::{Angle, Degree};
        use glam::{Mat4, Vec3};

        let observation = render::prepare(
            UtcDateTime::parse("2026-09-25T22:00:00Z").unwrap(),
            Angle::<Degree>::new(52.37),
            Angle::<Degree>::new(4.9),
        )
        .unwrap();
        let moon = observation
            .instances
            .iter()
            .find(|instance| instance.sun_kind[3] == 2.0)
            .expect("visible Moon");
        let saturn = observation
            .instances
            .iter()
            .find(|instance| {
                instance.sun_kind[3] == 1.0 && instance.color_flux[..3] == [0.94, 0.84, 0.66]
            })
            .expect("visible Saturn");
        let direction = |instance: &render::SkyInstance| {
            Vec3::new(
                instance.direction_radius[0],
                instance.direction_radius[1],
                instance.direction_radius[2],
            )
        };
        let moon_direction = direction(moon);
        let saturn_direction = direction(saturn);
        let view = Mat4::look_at_rh(Vec3::ZERO, moon_direction, Vec3::Y);
        assert!(
            moon_direction.z > 0.0,
            "Moon in the SSE must face terrain south"
        );
        assert!(
            view.transform_vector3(saturn_direction).x < view.transform_vector3(moon_direction).x,
            "Saturn east of the Moon must appear to the observer's left"
        );
    }

    #[test]
    fn lunar_pole_position_angle_matches_horizons() {
        use crate::astro::{frames, render, time::UtcDateTime};
        use crate::geo::units::{Angle, Degree};
        use glam::DVec3;

        let utc = UtcDateTime::parse("2026-09-25T22:00:00Z").unwrap();
        let latitude = Angle::<Degree>::new(52.37);
        let longitude = Angle::<Degree>::new(4.9);
        let observation = render::prepare(utc, latitude, longitude).unwrap();
        let moon = observation
            .instances
            .iter()
            .find(|instance| instance.sun_kind[3] == 2.0)
            .expect("visible Moon");
        let direction = DVec3::new(
            f64::from(moon.direction_radius[0]),
            f64::from(moon.direction_radius[1]),
            f64::from(moon.direction_radius[2]),
        )
        .normalize();
        let pole = DVec3::new(
            f64::from(moon.moon_pole[0]),
            f64::from(moon.moon_pole[1]),
            f64::from(moon.moon_pole[2]),
        );
        let celestial_north = frames::true_equatorial_direction_to_render_horizon(
            DVec3::Z,
            utc,
            latitude.radians(),
            longitude.radians(),
        )
        .unwrap();
        let sky_north = (celestial_north - direction * celestial_north.dot(direction)).normalize();
        // Horizons quantity 17 measures counter-clockwise from equatorial
        // north; that direction is screen-left in this right-handed view.
        let sky_east = -direction.cross(sky_north).normalize();
        let pole_tangent = (pole - direction * pole.dot(direction)).normalize();
        let position_angle = pole_tangent
            .dot(sky_east)
            .atan2(pole_tangent.dot(sky_north))
            .to_degrees()
            .rem_euclid(360.0);
        // NASA/JPL Horizons observer table: Moon (301), coord@399,
        // SITE_COORD=4.9,52.37,0, 2026-Sep-25 22:00 UT,
        // APPARENT=AIRLESS, quantity 17 (NP.ang) = 338.1496 degrees.
        assert!(
            (position_angle - 338.1496).abs() < 0.001,
            "{position_angle}"
        );
    }

    #[test]
    fn creates_sky_pipeline_when_adapter_available() {
        assert_eq!(std::mem::size_of::<crate::astro::render::SkyInstance>(), 64);
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::util::backend_bits_from_env().unwrap_or(wgpu::Backends::all()),
            ..Default::default()
        });
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

        let _resources =
            create_sky_resources(&Arc::new(device), &_queue, 16, 16).expect("sky resources");
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
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::util::backend_bits_from_env().unwrap_or(wgpu::Backends::all()),
            ..Default::default()
        });
        let Some(adapter) =
            pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
        else {
            eprintln!("SIDERA night golden ABSENT: no GPU adapter available");
            return;
        };
        let info = adapter.get_info();
        let Ok((device, queue)) =
            pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default(), None))
        else {
            eprintln!("SIDERA night golden ABSENT: GPU device unavailable");
            return;
        };
        let device = Arc::new(device);
        let resources =
            create_sky_resources(&device, &queue, SIZE, SIZE).expect("night sky pipelines");
        let observation = render::prepare(
            UtcDateTime::parse("2026-09-25T22:00:00Z").unwrap(),
            Angle::<Degree>::new(52.37),
            Angle::<Degree>::new(4.9),
        )
        .unwrap();
        let direction = |azimuth_deg: f64, altitude_deg: f64| {
            let (sa, ca) = azimuth_deg.to_radians().sin_cos();
            let (sh, ch) = altitude_deg.to_radians().sin_cos();
            Vec3::new((ch * sa) as f32, sh as f32, (-ch * ca) as f32)
        };
        let sun = direction(
            observation.sun.azimuth.value(),
            observation.sun.altitude.value(),
        );
        let moon = direction(
            observation.moon.azimuth.value(),
            observation.moon.refracted_altitude.value(),
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
                pass.set_bind_group(1, &resources.moon_bind_group, &[]);
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
        if let Some(path) = std::env::var_os("SIDERA_NIGHT_PREVIEW") {
            image::save_buffer(path, &first, SIZE, SIZE, image::ColorType::Rgba8).unwrap();
        }
        // A quarter-lit synthetic Moon makes the phase error visible even when
        // the real night-golden Moon is almost full and centred in the camera.
        // Its centre must stay dark when the camera aim moves off the Moon ray.
        let side_sun = moon.cross(Vec3::Y).normalize();
        let quarter_moon = render::SkyInstance {
            direction_radius: [moon.x, moon.y, moon.z, 0.04],
            color_flux: [1.0, 1.0, 1.0, 1.0],
            sun_kind: [side_sun.x, side_sun.y, side_sun.z, 2.0],
            moon_pole: [0.0, 1.0, 0.0, 0.0],
        };
        let quarter_buffer = tracked_create_buffer_init(
            &device,
            &BufferInitDescriptor {
                label: Some("sidera.off_center.quarter_moon"),
                contents: bytemuck::bytes_of(&quarter_moon),
                usage: wgpu::BufferUsages::VERTEX,
            },
        )
        .unwrap();
        let off_view = Mat4::look_at_rh(Vec3::ZERO, (moon - side_sun * 0.4).normalize(), Vec3::Y);
        let off_matrices = [off_view, proj, off_view.inverse(), proj.inverse()];
        for (index, matrix) in off_matrices.into_iter().enumerate() {
            camera[index * 16..index * 16 + 16].copy_from_slice(&matrix.to_cols_array());
        }
        queue.write_buffer(&resources.sky_camera, 0, bytemuck::cast_slice(&camera));
        let mut off_encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("sidera.off_center.encoder"),
        });
        {
            let mut pass = off_encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("sidera.off_center.celestial"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &resources.sky_output_view,
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
            pass.set_pipeline(&resources.celestial_pipeline);
            pass.set_bind_group(0, &bg1, &[]);
            pass.set_bind_group(1, &resources.moon_bind_group, &[]);
            pass.set_vertex_buffer(0, quarter_buffer.slice(..));
            pass.draw(0..6, 0..1);
        }
        queue.submit(std::iter::once(off_encoder.finish()));
        let off_frame = read_texture_tight(
            &device,
            &queue,
            &resources.sky_output,
            (SIZE, SIZE),
            wgpu::TextureFormat::Rgba8Unorm,
        )
        .unwrap();
        let clip = proj * off_view * moon.extend(1.0);
        let x = ((clip.x / clip.w * 0.5 + 0.5) * SIZE as f32) as usize;
        let y = ((0.5 - clip.y / clip.w * 0.5) * SIZE as f32) as usize;
        assert!(x >= 20 && x + 20 < SIZE as usize && y < SIZE as usize);
        let red = |sample_x: usize| off_frame[(y * SIZE as usize + sample_x) * 4];
        let sunward = if off_view.transform_vector3(side_sun).x > 0.0 {
            20_isize
        } else {
            -20_isize
        };
        assert!(red(x) < 10, "quarter Moon centre was lit off centre");
        assert!(
            red((x as isize + sunward) as usize) > red((x as isize - sunward) as usize),
            "Moon bright limb does not face its Sun"
        );
        // With the Sun behind the observer, equal-radius disc pixels have
        // equal phase shading. Their contrast must come from the lunar map.
        let full_moon = render::SkyInstance {
            direction_radius: [moon.x, moon.y, moon.z, 0.04],
            color_flux: [1.0, 1.0, 1.0, 1.0],
            sun_kind: [-moon.x, -moon.y, -moon.z, 2.0],
            moon_pole: [0.0, 1.0, 0.0, 0.0],
        };
        let full_buffer = tracked_create_buffer_init(
            &device,
            &BufferInitDescriptor {
                label: Some("sidera.textured.full_moon"),
                contents: bytemuck::bytes_of(&full_moon),
                usage: wgpu::BufferUsages::VERTEX,
            },
        )
        .unwrap();
        camera[0..16].copy_from_slice(&view.to_cols_array());
        camera[32..48].copy_from_slice(&view.inverse().to_cols_array());
        queue.write_buffer(&resources.sky_camera, 0, bytemuck::cast_slice(&camera));
        let mut texture_encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("sidera.textured.full_moon.encoder"),
        });
        {
            let mut pass = texture_encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("sidera.textured.full_moon.pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &resources.sky_output_view,
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
            pass.set_pipeline(&resources.celestial_pipeline);
            pass.set_bind_group(0, &bg1, &[]);
            pass.set_bind_group(1, &resources.moon_bind_group, &[]);
            pass.set_vertex_buffer(0, full_buffer.slice(..));
            pass.draw(0..6, 0..1);
        }
        queue.submit(std::iter::once(texture_encoder.finish()));
        let full_frame = read_texture_tight(
            &device,
            &queue,
            &resources.sky_output,
            (SIZE, SIZE),
            wgpu::TextureFormat::Rgba8Unorm,
        )
        .unwrap();
        let radius = (0.04 * proj.y_axis.y * SIZE as f32 * 0.5) as usize;
        // A camera roll must rotate lunar landmarks on screen. The old
        // screen-aligned UVs reproduced the unrolled image instead.
        let rolled_up = glam::Quat::from_axis_angle(moon, std::f32::consts::FRAC_PI_2) * Vec3::Y;
        let rolled_view = Mat4::look_at_rh(Vec3::ZERO, moon, rolled_up);
        camera[0..16].copy_from_slice(&rolled_view.to_cols_array());
        camera[32..48].copy_from_slice(&rolled_view.inverse().to_cols_array());
        queue.write_buffer(&resources.sky_camera, 0, bytemuck::cast_slice(&camera));
        let mut roll_encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("sidera.textured.rolled_moon.encoder"),
        });
        {
            let mut pass = roll_encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("sidera.textured.rolled_moon.pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &resources.sky_output_view,
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
            pass.set_pipeline(&resources.celestial_pipeline);
            pass.set_bind_group(0, &bg1, &[]);
            pass.set_bind_group(1, &resources.moon_bind_group, &[]);
            pass.set_vertex_buffer(0, full_buffer.slice(..));
            pass.draw(0..6, 0..1);
        }
        queue.submit(std::iter::once(roll_encoder.finish()));
        let rolled_frame = read_texture_tight(
            &device,
            &queue,
            &resources.sky_output,
            (SIZE, SIZE),
            wgpu::TextureFormat::Rgba8Unorm,
        )
        .unwrap();
        let center = SIZE as isize / 2;
        let mut unchanged_error = 0_u64;
        let mut clockwise_error = 0_u64;
        let mut counterclockwise_error = 0_u64;
        for y in (-25..=25).step_by(5) {
            for x in (-25..=25).step_by(5) {
                let sample = |frame: &[u8], x: isize, y: isize| {
                    frame[(((center + y) * SIZE as isize + center + x) * 4) as usize] as i32
                };
                let reference = sample(&full_frame, x, y);
                unchanged_error += (reference - sample(&rolled_frame, x, y)).unsigned_abs() as u64;
                clockwise_error += (reference - sample(&rolled_frame, y, -x)).unsigned_abs() as u64;
                counterclockwise_error +=
                    (reference - sample(&rolled_frame, -y, x)).unsigned_abs() as u64;
            }
        }
        assert!(
            clockwise_error.min(counterclockwise_error) < unchanged_error,
            "lunar texture stayed locked to the screen during camera roll"
        );
        let sample_y = SIZE as usize / 2 - (radius as f32 * 0.3) as usize;
        let sample_x =
            |sign: isize| (SIZE as isize / 2 + sign * (radius as f32 * 0.4) as isize) as usize;
        let albedo_red = |sample_x: usize| full_frame[(sample_y * SIZE as usize + sample_x) * 4];
        assert!(
            albedo_red(sample_x(-1)) > albedo_red(sample_x(1)) + 10,
            "equal-phase lunar surface pixels lost their source mosaic contrast"
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
                "twilight_model": "civil-to-astronomical solar-depression smoothstep ramp (-6 to -18 degrees)",
                "moonlight_model": "Krisciunas & Schaefer (1991) Eq. 9 phase magnitude and Eq. 3 air mass, 0.172 V-mag/airmass clear-sky extinction, relative directional display gain 0.12",
                "catalog": "Yale Bright Star Catalogue V/50, J2000, 9096 stars",
                "theories": "Complete VSOP87D (25,659 terms) and ELP2000-82B (37,872 terms); IAU 2006/2000B frames; IERS C04 UT1 to 2026-08-25 with forecast thereafter",
                "moon_texture": "NASA SVS Moon Mosaic 5001 (LRO NAC); 256x256 relative grayscale with mip levels, IAU WGCCRE lunar pole orientation, fixed nearside markings without libration",
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
