use super::*;

fn expected_contract(kind: TargetKind) -> (TextureFormat, TextureUsages, bool) {
    use TargetKind::*;
    use TextureUsages as U;
    match kind {
        Lit => (
            TextureFormat::Rgba8Unorm,
            U::STORAGE_BINDING | U::TEXTURE_BINDING,
            false,
        ),
        GiDebug => (
            TextureFormat::Rgba16Float,
            U::STORAGE_BINDING | U::TEXTURE_BINDING,
            false,
        ),
        Depth => (
            TextureFormat::Depth32Float,
            U::RENDER_ATTACHMENT | U::TEXTURE_BINDING,
            false,
        ),
        Sky => (
            TextureFormat::Rgba16Float,
            U::STORAGE_BINDING | U::TEXTURE_BINDING | U::RENDER_ATTACHMENT,
            false,
        ),
        FogHistory | FogHistoryHalf => (
            TextureFormat::Rgba16Float,
            U::TEXTURE_BINDING | U::COPY_DST,
            matches!(kind, FogHistoryHalf),
        ),
        GiBaseline | GiDiffuse | GiSpecular | GiOutput | FogOutput | FogOutputHalf => (
            TextureFormat::Rgba16Float,
            U::STORAGE_BINDING | U::TEXTURE_BINDING | U::COPY_SRC,
            matches!(kind, FogOutputHalf),
        ),
    }
}

#[test]
fn target_descriptors_preserve_formats_usages_and_half_resolution() {
    use TargetKind::*;
    for kind in [
        Lit,
        GiBaseline,
        GiDiffuse,
        GiSpecular,
        GiOutput,
        GiDebug,
        Sky,
        Depth,
        FogOutput,
        FogHistory,
        FogOutputHalf,
        FogHistoryHalf,
    ] {
        let (format, usage, half) = expected_contract(kind);
        for (width, height) in [(101, 67), (1, 1)] {
            let descriptor = kind.descriptor(width, height);
            let expected = if half {
                ((width / 2).max(1), (height / 2).max(1))
            } else {
                (width, height)
            };
            assert_eq!((descriptor.size.width, descriptor.size.height), expected);
            assert_eq!(descriptor.format, format);
            assert_eq!(descriptor.usage, usage);
            assert_eq!(descriptor.sample_count, 1);
            assert_eq!(descriptor.mip_level_count, 1);
        }
    }
}

#[cfg(windows)]
#[test]
fn viewport_resize_is_transactional_and_renders_viewer_frames() {
    // Winit permits one event loop per process. Keep every resize assertion in
    // an isolated test process so the screen-picking fixture can own its loop.
    if std::env::var_os("FORGE3D_VIEWPORT_TEST_CHILD").is_none() {
        let status = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "viewer::render_targets::tests::viewport_resize_is_transactional_and_renders_viewer_frames",
                "--nocapture",
            ])
            .env("FORGE3D_VIEWPORT_TEST_CHILD", "1")
            .status()
            .expect("isolated viewport test process");
        assert!(status.success(), "isolated viewport test failed: {status}");
        return;
    }

    use crate::core::resource_tracker::{
        begin_owner_capture, owner_group_report, AllocationOwner, TextureAllocationFailure,
    };
    use crate::core::screen_space_effects::ScreenSpaceEffect;
    use crate::passes::gi::GiPass;
    use crate::viewer::viewer_config::ViewerConfig;
    use std::sync::Arc;
    use winit::event_loop::EventLoopBuilder;
    use winit::platform::windows::EventLoopBuilderExtWindows;
    use winit::window::WindowBuilder;

    let event_loop = EventLoopBuilder::new()
        .with_any_thread(true)
        .build()
        .expect("Windows event loop");
    let window = Arc::new(
        WindowBuilder::new()
            .with_visible(false)
            .with_inner_size(winit::dpi::PhysicalSize::new(320, 240))
            .build(&event_loop)
            .expect("viewer window"),
    );
    let owner = AllocationOwner::new();
    let _ownership = owner.activate_group("viewport-resize");
    let mut viewer =
        pollster::block_on(Viewer::new(window, ViewerConfig::default())).expect("physical viewer");
    eprintln!(
        "resize adapter: {} / {} / {} / {}",
        viewer.adapter_name,
        viewer.adapter_backend,
        viewer.adapter_device_type,
        viewer.adapter_driver_info
    );
    if std::env::var("FORGE3D_VIEWPORT_REQUIRE_PHYSICAL").as_deref() == Ok("1") {
        assert_eq!(viewer.adapter_device_type, "DiscreteGpu");
        assert!(viewer.adapter_name.to_ascii_lowercase().contains("nvidia"));
    }
    if std::env::var("FORGE3D_VIEWPORT_TEST_FALLBACK").as_deref() == Ok("1") {
        assert_eq!(viewer.adapter_device_type, "Cpu", "explicit WARP fixture");
    }
    viewer
        .upload_mesh(&crate::geometry::generate_unit_box())
        .unwrap();
    // A visible white material, rather than the viewer's uninitialized fallback texel.
    viewer.ensure_geom_bind_group().unwrap();
    viewer.queue.write_texture(
        wgpu::ImageCopyTexture {
            texture: viewer.albedo_texture.as_ref().unwrap(),
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        &[255, 255, 255, 255],
        wgpu::ImageDataLayout {
            offset: 0,
            bytes_per_row: Some(4),
            rows_per_image: Some(1),
        },
        wgpu::Extent3d {
            width: 1,
            height: 1,
            depth_or_array_layers: 1,
        },
    );
    viewer
        .camera
        .set_orbit_pose_target(&viewer.camera_anchor, glam::DVec3::ZERO, 3.0, 0.5, 0.3)
        .unwrap();
    viewer.view_config.znear = 1.0;
    viewer.viz_mode = crate::viewer::viewer_enums::VizMode::Lit;
    viewer.sky_enabled = true;
    viewer.fog_enabled = true;
    viewer.ssr_params.set_enabled(true);
    let gi = viewer.gi.as_mut().expect("screen-space manager");
    for effect in [
        ScreenSpaceEffect::SSAO,
        ScreenSpaceEffect::SSGI,
        ScreenSpaceEffect::SSR,
    ] {
        gi.enable_effect(&viewer.device, effect).unwrap();
    }
    gi.set_ssr_params(&viewer.queue, &viewer.ssr_params);
    gi.set_ssao_blur(false);
    gi.set_ssao_temporal(true);
    gi.set_ssao_temporal_alpha(&viewer.queue, 0.73);
    gi.update_ssao_settings(&viewer.queue, |settings| settings.radius = 0.37);
    gi.update_ssgi_settings(&viewer.queue, |settings| settings.intensity = 0.42);
    gi.set_ssgi_half_res_with_queue(&viewer.device, &viewer.queue, true);
    viewer.gi_pass =
        Some(GiPass::new(&viewer.device, viewer.config.width, viewer.config.height).unwrap());
    viewer.set_taa_enabled(true);
    viewer
        .taa_renderer
        .as_mut()
        .unwrap()
        .set_history_weight(0.83);
    render_checked(&mut viewer);
    // The geometry path composites sky directly. Seed the terrain presenter's
    // real cache as well, to prove its old sky view survives failure and is
    // discarded on successful installation.
    viewer
        .sky_present_bg_cache
        .replace(Some(viewer.device.create_bind_group(
            &wgpu::BindGroupDescriptor {
                label: Some("viewer.resize.test.sky.present"),
                layout: &viewer.sky_present_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&viewer.sky_output_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&viewer.sky_present_sampler),
                    },
                ],
            },
        )));
    assert!(viewer.sky_present_bg_cache.borrow().is_some());

    let original_ids = allocation_ids(&viewer);
    let original_dimensions = (viewer.config.width, viewer.config.height);
    let original_history_count = viewer.history_invalidation_count;
    let original_fog_valid = viewer.fog_history_state.is_valid();
    let original_cache_size = viewer.comp_bind_group_cache.borrow().len();
    let original_bytes = owner_group_report(&owner, "viewport-resize").current_total_bytes();

    // Every allocation boundary must release the prepared prefix and keep live state.
    for label in [
        "viewer.lit.output",
        "viewer.gi.output.hdr",
        "viewer.fog.history.half",
        "gbuffer_depth",
        "p5.hzb.pyramid",
        "ssao_history",
        "ssgi_scene_history_b",
        "p5.ssr.history",
        "taa.history.1",
    ] {
        let fault = TextureAllocationFailure::for_label(label);
        let result = viewer.resize_render_targets(101, 67);
        assert!(fault.attempts() > 0);
        assert!(matches!(result, Err(RenderError::Budget(_))));
        assert_eq!(allocation_ids(&viewer), original_ids);
        assert_eq!(viewer.history_invalidation_count, original_history_count);
        assert_eq!(viewer.fog_history_state.is_valid(), original_fog_valid);
        assert_eq!(
            viewer.comp_bind_group_cache.borrow().len(),
            original_cache_size
        );
        assert!(viewer.sky_present_bg_cache.borrow().is_some());
        assert_eq!(
            owner_group_report(&owner, "viewport-resize").current_total_bytes(),
            original_bytes
        );
    }
    // Real wgpu validation errors are returned, rather than installing invalid handles.
    let invalid = PreparedViewportTargets::prepare_with(&viewer, 101, 67, |device, descriptor| {
        let mut descriptor = descriptor.clone();
        if descriptor.label == Some("viewer.fog.history.half") {
            descriptor.size.width = 0;
        }
        Target::create(device, &descriptor)
    });
    assert!(matches!(invalid, Err(RenderError::Device(_))));
    assert_eq!(allocation_ids(&viewer), original_ids);
    assert_eq!(
        owner_group_report(&owner, "viewport-resize").current_total_bytes(),
        original_bytes
    );
    for size in [(0, 67), (101, 0), (u32::MAX, 67)] {
        assert!(viewer.resize_render_targets(size.0, size.1).is_err());
        viewer.resize(winit::dpi::PhysicalSize::new(size.0, size.1));
        assert_eq!(
            (viewer.config.width, viewer.config.height),
            original_dimensions
        );
        assert_eq!(allocation_ids(&viewer), original_ids);
    }
    // A failed native extent suspends rendering without allocation retry loops.
    let failed_size = winit::dpi::PhysicalSize::new(101, 67);
    let _ = viewer.window.request_inner_size(failed_size);
    assert_eq!(viewer.window.inner_size(), failed_size);
    {
        let fault = TextureAllocationFailure::for_label("taa.history.1");
        viewer.resize(failed_size);
        assert_eq!(viewer.failed_window_resize, Some(failed_size));
        assert!(!viewer.window_can_render());
        let attempts = fault.attempts();
        assert!(!viewer.recover_surface());
        assert!(!viewer.recover_surface());
        assert_eq!(fault.attempts(), attempts);
        assert_eq!(allocation_ids(&viewer), original_ids);
        let original_size =
            winit::dpi::PhysicalSize::new(original_dimensions.0, original_dimensions.1);
        let _ = viewer.window.request_inner_size(original_size);
        assert!(viewer.recover_surface());
        viewer.resize(original_size);
        assert_eq!(fault.attempts(), attempts);
        assert_eq!(allocation_ids(&viewer), original_ids);
        assert_eq!(viewer.fog_history_state.is_valid(), original_fog_valid);
    }
    render_checked(&mut viewer);

    // Measure the inherent old+new allocation peak, including all dependents.
    let before = begin_owner_capture(owner.id()).finish();
    let capture = begin_owner_capture(owner.id());
    let prepared = PreparedViewportTargets::prepare(&viewer, 101, 67).unwrap();
    let during = owner_group_report(&owner, "viewport-resize").current_total_bytes();
    drop(prepared);
    let peak = capture.finish();
    assert_eq!(peak.current_total_bytes(), before.current_total_bytes());
    assert_eq!(peak.peak_total_bytes, during);
    for label in ["ssao_noise", "ssgi_env_cube", "p5.ssr.env.placeholder"] {
        assert!(
            before.by_label.contains_key(label),
            "missing static allocation {label}"
        );
        assert_eq!(peak.by_label[label], before.by_label[label]);
    }
    eprintln!(
        "resize tracked resident bytes: old={} prepared={} peak={} released={}",
        before.current_total_bytes(),
        during - before.current_total_bytes(),
        peak.peak_total_bytes,
        peak.current_total_bytes()
    );

    // Populate real terrain-owned TAA on the same GPU, then attach it to the viewer.
    let mut terrain = crate::viewer::terrain::ViewerTerrainScene::new(
        viewer.device.clone(),
        viewer.queue.clone(),
        viewer.adapter.clone(),
        viewer.config.format,
    )
    .unwrap();
    let mut terrain_taa = crate::core::taa::TaaRenderer::new(
        &viewer.device,
        viewer.config.width,
        viewer.config.height,
    )
    .unwrap();
    terrain_taa.set_enabled(true);
    let mut encoder = viewer
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    let gbuffer = viewer.gi.as_ref().unwrap().gbuffer();
    assert!(terrain_taa.execute(
        &viewer.device,
        &mut encoder,
        &viewer.queue,
        &gbuffer.material_view,
        &gbuffer.velocity_view,
        &gbuffer.depth_view
    ));
    viewer.queue.submit(Some(encoder.finish()));
    viewer.device.poll(wgpu::Maintain::Wait);
    terrain.attach_test_taa(terrain_taa);
    assert!(terrain.taa_history_valid());
    let terrain_ids = terrain.taa_history_allocation_ids();
    viewer.terrain_viewer = Some(terrain);

    viewer.resize_render_targets(101, 67).unwrap();
    assert_eq!(
        (viewer.config.width, viewer.config.height),
        original_dimensions
    );
    assert_dimensions(&viewer, 101, 67);
    let terrain = viewer.terrain_viewer.as_ref().unwrap();
    assert!(!terrain.taa_history_valid());
    assert!(terrain.is_taa_enabled());
    assert_eq!(terrain.taa_history_allocation_ids(), terrain_ids);
    viewer.fog_half_res_enabled = true;
    for (width, height) in [(101, 67), (1, 1), (320, 240)] {
        let size = winit::dpi::PhysicalSize::new(width, height);
        let _ = viewer.window.request_inner_size(size);
        assert_eq!(viewer.window.inner_size(), size);
        viewer.resize(winit::dpi::PhysicalSize::new(width, height));
        assert_dimensions(&viewer, width, height);
        assert!(viewer.sky_present_bg_cache.borrow().is_none());
        assert!(viewer.lit_bind_group_cache.borrow().is_none());
        assert!(viewer.comp_bind_group_cache.borrow().is_empty());
        assert!(!viewer.fog_history_state.is_valid());
        assert_eq!(viewer.fog_frame_index, 0);
        let gi = viewer.gi.as_ref().unwrap();
        assert!(!gi.ssgi_history_valid());
        assert!(!gi.ssr_history_valid());
        assert_eq!(gi.ssao_settings().radius, 0.37);
        assert_eq!(gi.ssao_temporal_alpha(), 0.73);
        assert_eq!(gi.ssgi_settings().unwrap().intensity, 0.42);
        assert_eq!(gi.ssgi_half_res(), Some(true));
        let taa = viewer.taa_renderer.as_ref().unwrap();
        assert!(taa.is_enabled());
        assert_eq!(taa.history_weight(), 0.83);
        assert!(!taa.history_valid());
        render_checked(&mut viewer);
        render_checked(&mut viewer);
        assert!(viewer.gi.as_ref().unwrap().ssgi_history_valid());
        assert!(viewer.gi.as_ref().unwrap().ssr_history_valid());
        assert!(viewer.taa_renderer.as_ref().unwrap().history_valid());
        assert!(viewer.fog_history_state.is_valid());
    }
    assert_eq!(viewer.history_invalidation_count, original_history_count);
    // Same-size window recovery keeps populated histories and identities.
    let before_same = allocation_ids(&viewer);
    let fog_valid = viewer.fog_history_state.is_valid();
    let taa_valid = viewer.taa_renderer.as_ref().unwrap().history_valid();
    {
        let fault = TextureAllocationFailure::for_label("viewer.lit.output");
        viewer.resize(viewer.window.inner_size());
        assert!(viewer.recover_surface());
        assert_eq!(fault.attempts(), 0);
    }
    assert_eq!(allocation_ids(&viewer), before_same);
    assert_eq!(viewer.fog_history_state.is_valid(), fog_valid);
    assert_eq!(
        viewer.taa_renderer.as_ref().unwrap().history_valid(),
        taa_valid
    );

    // Save the actual lit composition and independent sky/fog/GI outputs.
    let evidence = std::env::var_os("FORGE3D_VIEWPORT_EVIDENCE_DIR")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| std::path::PathBuf::from(".task-evidence"));
    std::fs::create_dir_all(&evidence).unwrap();
    let mut captures = Vec::new();
    for (name, half, fog) in [
        ("resize-lit.png", false, false),
        ("resize-lit-fog-full.png", false, true),
        ("resize-lit-fog-half.png", true, true),
    ] {
        viewer.fog_enabled = fog;
        viewer.fog_half_res_enabled = half;
        viewer.invalidate_viewport_histories(viewer.config.width, viewer.config.height);
        viewer.snapshot_request = Some(evidence.join(name).to_string_lossy().into_owned());
        render_checked(&mut viewer);
        let frame = image::open(evidence.join(name))
            .expect("fresh physical lit frame")
            .to_rgba8();
        assert_eq!(frame.dimensions(), (320, 240));
        assert!(
            frame
                .pixels()
                .any(|pixel| pixel.0 != frame.get_pixel(0, 0).0),
            "lit frame contains scene detail"
        );
        captures.push(frame.into_raw());
    }
    assert!(
        captures[0] != captures[1],
        "full-resolution fog changes lit composition"
    );
    assert!(
        captures[0] != captures[2],
        "half-resolution fog changes lit composition"
    );
    viewer
        .device
        .push_error_scope(wgpu::ErrorFilter::Validation);
    viewer
        .reexecute_gi(None)
        .expect("resized GI baseline, split, composition, debug");
    viewer.device.poll(wgpu::Maintain::Wait);
    let error = pollster::block_on(viewer.device.pop_error_scope());
    assert!(
        error.is_none(),
        "GI reexecution validation failed: {error:?}"
    );
    viewer
        .device
        .push_error_scope(wgpu::ErrorFilter::Validation);
    let gi_data = crate::core::hdr::read_hdr_texture(
        &viewer.device,
        &viewer.queue,
        &viewer.gi_output_hdr,
        320,
        240,
        TextureFormat::Rgba16Float,
    )
    .unwrap();
    assert!(gi_data.iter().all(|value| value.is_finite()));
    assert!(gi_data
        .chunks_exact(4)
        .any(|pixel| pixel[..3].iter().any(|value| *value > 0.0)));
    let gi_image = image::RgbaImage::from_raw(
        320,
        240,
        gi_data
            .iter()
            .map(|value| (value.clamp(0.0, 1.0) * 255.0).round() as u8)
            .collect(),
    )
    .unwrap();
    gi_image
        .save(evidence.join("resize-gi-hdr-clamped.png"))
        .unwrap();
    let fog_data = crate::core::hdr::read_hdr_texture(
        &viewer.device,
        &viewer.queue,
        &viewer.fog_output,
        320,
        240,
        TextureFormat::Rgba16Float,
    )
    .unwrap();
    assert!(fog_data.iter().all(|value| value.is_finite()));
    assert!(fog_data
        .chunks_exact(4)
        .any(|pixel| pixel[..3].iter().any(|value| *value > 0.0)));
    let sky = Target::create(
        &viewer.device,
        &wgpu::TextureDescriptor {
            label: Some("viewer.resize.test.sky.capture"),
            size: wgpu::Extent3d {
                width: 320,
                height: 240,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: viewer.config.format,
            usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::COPY_SRC,
            view_formats: &[],
        },
    )
    .unwrap();
    let mut encoder = viewer
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    {
        let _clear = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("viewer.resize.test.sky.clear"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &sky.view,
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
    }
    viewer.present_sky_output(&mut encoder, &sky.view, None);
    viewer.queue.submit(Some(encoder.finish()));
    viewer
        .snapshot_swapchain_to_png(
            &sky.texture,
            &evidence.join("resize-sky.png").to_string_lossy(),
        )
        .unwrap();
    let sky_frame = image::open(evidence.join("resize-sky.png"))
        .unwrap()
        .to_rgba8();
    assert!(sky_frame
        .pixels()
        .any(|pixel| pixel.0 != sky_frame.get_pixel(0, 0).0));
    viewer.device.poll(wgpu::Maintain::Wait);
    let error = pollster::block_on(viewer.device.pop_error_scope());
    assert!(
        error.is_none(),
        "GI/fog/sky GPU validation failed: {error:?}"
    );
    drop(sky);
    drop(viewer);
    assert_eq!(
        owner_group_report(&owner, "viewport-resize").current_total_bytes(),
        0
    );
}

#[cfg(windows)]
fn render_checked(viewer: &mut Viewer) {
    viewer
        .device
        .push_error_scope(wgpu::ErrorFilter::Validation);
    viewer.render().expect("resized viewer frame");
    let gi = viewer.gi.as_ref().unwrap();
    let mut encoder = viewer
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("viewer.resize.test.gi.debug"),
        });
    viewer
        .gi_pass
        .as_mut()
        .unwrap()
        .execute_debug(
            &viewer.device,
            &mut encoder,
            gi.ao_resolved_view().unwrap(),
            gi.ssgi_output_for_display_view().unwrap(),
            gi.ssr_final_view().unwrap(),
            &viewer.gi_debug_view,
        )
        .unwrap();
    viewer.queue.submit(Some(encoder.finish()));
    viewer.device.poll(wgpu::Maintain::Wait);
    let error = pollster::block_on(viewer.device.pop_error_scope());
    assert!(error.is_none(), "frame GPU validation failed: {error:?}");
}

#[cfg(windows)]
fn allocation_ids(viewer: &Viewer) -> Vec<u64> {
    let mut ids = vec![
        viewer.lit_output.ledger_id(),
        viewer.gi_baseline_hdr.ledger_id(),
        viewer.gi_baseline_diffuse_hdr.ledger_id(),
        viewer.gi_baseline_spec_hdr.ledger_id(),
        viewer.gi_output_hdr.ledger_id(),
        viewer.gi_debug.ledger_id(),
        viewer.sky_output.ledger_id(),
        viewer.fog_output.ledger_id(),
        viewer.fog_history.ledger_id(),
        viewer.fog_output_half.ledger_id(),
        viewer.fog_history_half.ledger_id(),
        viewer.z_texture.as_ref().unwrap().ledger_id(),
    ];
    let gi = viewer.gi.as_ref().unwrap();
    ids.extend(
        gi.viewport_textures()
            .iter()
            .map(|texture| texture.ledger_id()),
    );
    ids.extend(
        viewer
            .taa_renderer
            .as_ref()
            .unwrap()
            .history_allocation_ids(),
    );
    ids
}

#[cfg(windows)]
fn assert_dimensions(viewer: &Viewer, width: u32, height: u32) {
    for (kind, texture) in [
        (TargetKind::Lit, &viewer.lit_output),
        (TargetKind::GiBaseline, &viewer.gi_baseline_hdr),
        (TargetKind::GiDiffuse, &viewer.gi_baseline_diffuse_hdr),
        (TargetKind::GiSpecular, &viewer.gi_baseline_spec_hdr),
        (TargetKind::GiOutput, &viewer.gi_output_hdr),
        (TargetKind::GiDebug, &viewer.gi_debug),
        (TargetKind::Sky, &viewer.sky_output),
        (TargetKind::Depth, viewer.z_texture.as_ref().unwrap()),
        (TargetKind::FogOutput, &viewer.fog_output),
        (TargetKind::FogHistory, &viewer.fog_history),
        (TargetKind::FogOutputHalf, &viewer.fog_output_half),
        (TargetKind::FogHistoryHalf, &viewer.fog_history_half),
    ] {
        let (format, usage, half) = expected_contract(kind);
        let expected = if half {
            ((width / 2).max(1), (height / 2).max(1))
        } else {
            (width, height)
        };
        assert_eq!((texture.width(), texture.height()), expected);
        assert_eq!(texture.format(), format);
        assert_eq!(texture.usage(), usage);
    }
    let gi = viewer.gi.as_ref().unwrap();
    assert_eq!(gi.gbuffer().dimensions(), (width, height));
    gi.assert_viewport_dimensions(width, height);
    assert_eq!(
        viewer.gi_pass.as_ref().unwrap().dimensions(),
        (width, height)
    );
    for texture in viewer.taa_renderer.as_ref().unwrap().viewport_textures() {
        assert_eq!((texture.width(), texture.height()), (width, height));
        assert_eq!(texture.format(), TextureFormat::Rgba16Float);
        assert_eq!(
            texture.usage(),
            TextureUsages::TEXTURE_BINDING | TextureUsages::STORAGE_BINDING
        );
    }
    for texture in [
        &gi.gbuffer().depth_texture,
        &gi.gbuffer().normal_texture,
        &gi.gbuffer().material_texture,
        &gi.gbuffer().velocity_texture,
    ] {
        assert_eq!((texture.width(), texture.height()), (width, height));
    }
    assert_eq!(gi.ao_raw_texture().unwrap().width(), width);
    assert_eq!(gi.ssr_output_texture().unwrap().height(), height);
    assert_eq!(gi.ssgi_upscaled_texture().unwrap().size().width, width);
    assert_eq!(gi.ssgi_hit_texture().unwrap().width(), (width / 2).max(1));
    assert_eq!(
        viewer
            .taa_renderer
            .as_ref()
            .unwrap()
            .output_texture()
            .height(),
        height
    );
}
