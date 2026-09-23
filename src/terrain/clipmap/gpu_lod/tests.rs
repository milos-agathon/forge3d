use super::*;

#[cfg(feature = "enable-globe")]
#[test]
fn planetary_altitude_and_up_come_from_globe_frame() {
    use glam::DVec3;

    let frame =
        crate::terrain::clipmap::globe::GlobeFrame::globe(10_000.0, DVec3::X * 12_500.0).unwrap();

    let planet = PlanetLodParams::from_globe(&frame).unwrap();

    assert_eq!(planet.radius, 10_000.0);
    assert_eq!(planet.altitude, 2_500.0);
    assert!((planet.camera_up - Vec3::Z).length() < 1.0e-6);
}

#[cfg(feature = "enable-globe")]
#[test]
fn planetary_tile_builder_uses_full_camera_relative_volume() {
    use glam::DVec3;

    let radius = 6_371_000.0;
    let frame = crate::terrain::clipmap::globe::GlobeFrame::globe(
        radius,
        DVec3::X * (radius + 1_000.0),
    )
    .unwrap();
    let tile = TileInfo::new(0, 0, 0, Vec2::ZERO, Vec2::ZERO)
        .with_globe_bounds(
            &frame,
            Vec3::new(-500.0, -400.0, -1_200.0),
            Vec3::new(700.0, 600.0, -800.0),
        )
        .unwrap();

    assert_eq!(tile.camera_relative_center, [100.0, 100.0, -1_000.0]);
    assert_eq!(tile.bounds_min, [-500.0, -400.0]);
    assert_eq!(tile.bounds_max, [700.0, 600.0]);
    assert_eq!((tile.height_min, tile.height_max), (-1_200.0, -800.0));
    assert!(tile.angular_radius > 0.0);
}

#[test]
fn horizon_keeps_tile_whose_angular_bound_crosses_tangent() {
    let planet = PlanetLodParams {
        radius: 10.0,
        altitude: 10.0,
        camera_up: Vec3::Z,
    };
    let center_angle = 65.0_f32.to_radians();
    let center_from_planet = Vec3::new(center_angle.sin(), 0.0, center_angle.cos()) * planet.radius;
    let center_relative = center_from_planet - planet.camera_up * (planet.radius + planet.altitude);

    assert!(!horizon_visible(center_relative, 0.0, planet));
    assert!(horizon_visible(
        center_relative,
        10.0_f32.to_radians(),
        planet
    ));
}

#[test]
fn horizon_keeps_an_antipodal_tile_with_a_pi_wide_bound() {
    let planet = PlanetLodParams {
        radius: 10.0,
        altitude: 10.0,
        camera_up: Vec3::Z,
    };
    let antipodal_from_planet = -Vec3::Z * planet.radius;
    let center_relative =
        antipodal_from_planet - planet.camera_up * (planet.radius + planet.altitude);

    assert!(horizon_visible(
        center_relative,
        std::f32::consts::PI,
        planet
    ));
}

#[cfg(feature = "enable-globe")]
#[test]
fn planetary_aabb_uses_camera_relative_bounds() {
    use glam::DVec3;

    let radius = 6_371_000.0;
    let frame =
        crate::terrain::clipmap::globe::GlobeFrame::globe(radius, DVec3::X * (radius + 1_000.0))
            .unwrap();
    let tile = TileInfo::new(0, 0, 0, Vec2::new(-100.0, -100.0), Vec2::new(100.0, 100.0))
        .with_height_bounds(-1_100.0, -900.0)
        .with_globe_center(&frame, DVec3::X * radius, 0.001)
        .unwrap();
    let view = Mat4::look_at_rh(Vec3::ZERO, Vec3::NEG_Z, Vec3::Y);
    let projection = Mat4::perspective_rh(45.0_f32.to_radians(), 1.0, 1.0, 5_000.0);

    let result = cpu_lod_select_globe(
        &[tile],
        projection * view,
        &GpuLodConfig::default(),
        (-1_100.0, -900.0),
        &frame,
    )
    .unwrap();

    assert_eq!(result.visible_tiles.len(), 1);
    assert!((result.visible_tiles[0].distance - 1_000.0).abs() < 0.25);
}

#[cfg(feature = "enable-globe")]
#[test]
fn tracked_planetary_encode_propagates_invalid_frame_error() {
    use crate::terrain::clipmap::globe::GlobeFrame;
    use glam::DVec3;

    let context = match crate::core::gpu::try_ctx() {
        Ok(context) => context,
        Err(error) if is_adapter_unavailable(&error) => {
            eprintln!("planetary LOD error propagation skipped: {error}");
            return;
        }
        Err(error) => panic!("planetary LOD adapter setup failed: {error}"),
    };
    let config = GpuLodConfig {
        max_lod: 0,
        ..Default::default()
    };
    let selector = GpuLodSelector::new(&context.device, config);
    let tiles = [TileInfo::new(
        0,
        0,
        0,
        Vec2::splat(-1.0),
        Vec2::splat(1.0),
    )];
    let templates = [IndirectDrawTemplate {
        index_count: 3,
        first_index: 0,
        base_vertex: 0,
        tile_id: tiles[0].tile_id,
    }];
    let resources = selector
        .create_draw_resources(&context.device, &tiles, &templates)
        .unwrap();
    let mut encoder = context
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("planetary_lod_invalid_frame"),
        });
    // A flat zero-anchor frame is valid for the flat path, but cannot define
    // the planetary up vector required by the globe LOD shader.
    let invalid_planet = GlobeFrame::flat(DVec3::ZERO).unwrap();
    let error = selector
        .encode_indirect_globe_tracked(
            &context.queue,
            &mut encoder,
            &resources,
            &tiles,
            Mat4::IDENTITY,
            &invalid_planet,
            false,
            (0.0, 0.0),
            true,
            LodSelectionProvenance(99),
        )
        .unwrap_err();
    assert_eq!(
        error,
        GpuLodEncodeError::GlobeFrame(
            crate::terrain::clipmap::globe::GlobeFrameError::NonFiniteEcef
        )
    );
}

#[cfg(feature = "enable-globe")]
#[test]
fn tracked_planetary_encode_rejects_tile_count_without_consuming_a_ticket() {
    use glam::DVec3;

    let context = match crate::core::gpu::try_ctx() {
        Ok(context) => context,
        Err(error) if is_adapter_unavailable(&error) => {
            eprintln!("planetary LOD mismatch propagation skipped: {error}");
            return;
        }
        Err(error) => panic!("planetary LOD adapter setup failed: {error}"),
    };
    let selector = GpuLodSelector::new(
        &context.device,
        GpuLodConfig {
            max_lod: 0,
            ..Default::default()
        },
    );
    let tiles = [TileInfo::new(
        0,
        0,
        0,
        Vec2::splat(-1.0),
        Vec2::splat(1.0),
    )];
    let templates = [IndirectDrawTemplate {
        index_count: 3,
        first_index: 0,
        base_vertex: 0,
        tile_id: tiles[0].tile_id,
    }];
    let resources = selector
        .create_draw_resources(&context.device, &tiles, &templates)
        .unwrap();
    let frame = crate::terrain::clipmap::globe::GlobeFrame::globe(
        10_000.0,
        DVec3::X * 11_000.0,
    )
    .unwrap();
    let mut encoder = context
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("planetary_lod_tile_count_mismatch"),
        });
    let error = selector
        .encode_indirect_globe_tracked(
            &context.queue,
            &mut encoder,
            &resources,
            &[],
            Mat4::IDENTITY,
            &frame,
            false,
            (0.0, 0.0),
            true,
            LodSelectionProvenance(100),
        )
        .unwrap_err();
    assert_eq!(
        error,
        GpuLodEncodeError::TileCountMismatch {
            expected: 1,
            actual: 0,
        }
    );

    let ticket = selector
        .encode_indirect_globe_tracked(
            &context.queue,
            &mut encoder,
            &resources,
            &tiles,
            Mat4::IDENTITY,
            &frame,
            false,
            (0.0, 0.0),
            true,
            LodSelectionProvenance(101),
        )
        .unwrap()
        .expect("mismatch must not consume or stage a stale readback ticket");
    assert!(resources.cancel_selection(ticket));
}

#[test]
fn flat_selection_preserves_historical_shape_and_distance() {
    let visible = TileInfo::new(0, 7, 9, Vec2::new(-0.5, -0.5), Vec2::new(0.5, 0.5))
        .with_height_bounds(0.0, 0.0);
    let outside = TileInfo::new(0, 8, 9, Vec2::new(2.0, 2.0), Vec2::new(3.0, 3.0))
        .with_height_bounds(0.0, 0.0);
    let config = GpuLodConfig {
        pixel_error_budget: 1.0,
        viewport_height: 2,
        fov_y: std::f32::consts::FRAC_PI_2,
        max_lod: 2,
        tile_size: 1.0,
        ..Default::default()
    };

    let result = cpu_lod_select(
        &[visible, outside],
        Mat4::IDENTITY,
        Vec3::new(3.0, 4.0, 9.0),
        &config,
        (0.0, 0.0),
    );

    assert_eq!(result.visible_tiles.len(), 1);
    assert_eq!(result.visible_tiles[0].tile_id, TileInfo::pack_id(0, 7, 9));
    assert_eq!(result.visible_tiles[0].distance, 5.0);
    assert_eq!(result.visible_tiles[0].selected_lod, 2);
    assert_eq!(result.total_triangles, 2_048);
    assert_eq!(result.culled_count, 1);
}

#[test]
fn readback_ticket_does_not_reuse_submitted_slot() {
    let mut tickets = SelectionReadbackTickets::default();
    let first = tickets.stage(LodSelectionProvenance(1)).unwrap();
    assert!(tickets.mark_submitted(first));
    assert_eq!(
        tickets.state(first),
        Some(SelectionReadbackTicketState::Submitted)
    );

    let second = tickets.stage(LodSelectionProvenance(2)).unwrap();
    assert!(tickets.mark_submitted(second));
    assert_ne!(first.slot, second.slot);
    assert!(tickets.stage(LodSelectionProvenance(3)).is_none());

    assert!(tickets.complete(first));
    assert_eq!(tickets.state(first), None);
    let third = tickets.stage(LodSelectionProvenance(3)).unwrap();
    assert_eq!(third.slot, first.slot);
    assert!(third.id > second.id);
}

#[test]
fn readback_delivers_reused_slots_in_submission_order() {
    let mut tickets = SelectionReadbackTickets::default();
    let first = tickets.stage(LodSelectionProvenance(1)).unwrap();
    assert!(tickets.mark_submitted(first));
    let second = tickets.stage(LodSelectionProvenance(2)).unwrap();
    assert!(tickets.mark_submitted(second));

    assert!(tickets.complete(first));
    let third = tickets.stage(LodSelectionProvenance(3)).unwrap();
    assert!(tickets.mark_submitted(third));
    assert_eq!(third.slot, first.slot);
    assert_eq!(tickets.oldest_submitted(), Some(second));
}

#[test]
fn delayed_selection_is_rejected_for_a_newer_frame_provenance() {
    let delayed = CompletedLodSelection {
        provenance: LodSelectionProvenance(41),
        selection: LodSelectionResult {
            visible_tiles: vec![TileInfo::new(
                0,
                0,
                0,
                Vec2::new(-1.0, -1.0),
                Vec2::new(1.0, 1.0),
            )],
            total_triangles: 2,
            culled_count: 0,
        },
    };

    assert!(
        delayed
            .into_selection_for(LodSelectionProvenance(42))
            .is_none(),
        "frame-A selection must not be rebuilt with frame-B inputs"
    );
}

#[test]
fn exact_submission_leaves_an_abandoned_ticket_cancelable() {
    let mut tickets = SelectionReadbackTickets::default();
    let submitted = tickets.stage(LodSelectionProvenance(1)).unwrap();
    let abandoned = tickets.stage(LodSelectionProvenance(2)).unwrap();

    assert!(tickets.mark_submitted(submitted));
    assert_eq!(
        tickets.state(submitted),
        Some(SelectionReadbackTicketState::Submitted)
    );
    assert_eq!(
        tickets.state(abandoned),
        Some(SelectionReadbackTicketState::Staged)
    );
    assert!(tickets.cancel(abandoned));

    let replacement = tickets.stage(LodSelectionProvenance(3)).unwrap();
    assert_eq!(replacement.slot, abandoned.slot);
    assert_ne!(replacement.id, abandoned.id);
    assert!(!tickets.mark_submitted(abandoned));
}

#[test]
fn staged_ticket_retains_its_originating_provenance() {
    let mut tickets = SelectionReadbackTickets::default();
    let frame_a = tickets.stage(LodSelectionProvenance(100)).unwrap();
    let _frame_b = tickets.stage(LodSelectionProvenance(200)).unwrap();

    assert_eq!(frame_a.provenance, LodSelectionProvenance(100));
}

#[test]
fn lod_uniform_carries_planet_fields_in_the_existing_binding() {
    let params = LodSelectParams::new(
        Mat4::IDENTITY,
        Vec3::new(3.0, 4.0, 5.0),
        &FrustumPlanes::from_view_proj(Mat4::IDENTITY),
        &GpuLodConfig::default(),
        7,
        5,
        false,
        (-2.0, 6.0),
        true,
        Some(PlanetLodParams {
            radius: 10.0,
            altitude: 2.0,
            camera_up: Vec3::Z,
        }),
    );

    assert_eq!(std::mem::size_of::<LodSelectParams>(), 256);
    assert_eq!(params.planet_params, [10.0, 2.0, 1.0, 0.0]);
    assert_eq!(params.camera_up, [0.0, 0.0, 1.0, 0.0]);

    let module =
        naga::front::wgsl::parse_str(include_str!("../../../shaders/clipmap_lod_select.wgsl"))
            .expect("valid terrain LOD WGSL");
    let shader_span = module
        .types
        .iter()
        .find_map(|(_, ty)| match (&ty.name, &ty.inner) {
            (Some(name), naga::TypeInner::Struct { span, .. }) if name == "LodSelectParams" => {
                Some(*span)
            }
            _ => None,
        })
        .expect("LodSelectParams struct");
    assert_eq!(shader_span, std::mem::size_of::<LodSelectParams>() as u32);
}

#[test]
fn every_tile_storage_consumer_matches_the_rust_stride() {
    fn tile_info_span(source: &str) -> u32 {
        let module = naga::front::wgsl::parse_str(source).expect("valid terrain LOD WGSL");
        let span = module
            .types
            .iter()
            .find_map(|(_, ty)| {
                if ty.name.as_deref() != Some("TileInfo") {
                    return None;
                }
                match ty.inner {
                    naga::TypeInner::Struct { span, .. } => Some(span),
                    _ => None,
                }
            })
            .expect("TileInfo struct");
        span
    }

    let rust_stride = std::mem::size_of::<TileInfo>() as u32;
    assert_eq!(rust_stride, 56);
    assert_eq!(
        tile_info_span(include_str!("../../../shaders/clipmap_lod_select.wgsl")),
        rust_stride
    );
    assert_eq!(
        tile_info_span(include_str!("../../../shaders/hzb_cull.wgsl")),
        rust_stride
    );
}

#[test]
fn oracle_selection_prefers_exact_submitted_tiles_without_recomputing() {
    let submitted = [TileInfo {
        tile_id: 97,
        height_min: -1.0,
        bounds_min: [-2.0, -3.0],
        bounds_max: [4.0, 5.0],
        distance: 12.0,
        selected_lod: 2,
        visible: 1,
        height_max: 6.0,
        camera_relative_center: [1.0, 1.0, 0.0],
        angular_radius: 0.0,
    }];
    let selected = prefer_submitted_tiles(Some(&submitted), || {
        panic!("CPU LOD selection must not replace submitted GPU provenance")
    });

    assert_eq!(selected.len(), 1);
    assert_eq!(selected[0].tile_id, 97);
    assert_eq!(selected[0].selected_lod, 2);
}

#[test]
fn test_tile_id_packing() {
    let (lod, x, y) = (3, 100, 200);
    let packed = TileInfo::pack_id(lod, x, y);
    let (l, xx, yy) = TileInfo::unpack_id(packed);
    assert_eq!((l, xx, yy), (lod, x, y));
}

#[test]
fn test_frustum_planes_extraction() {
    let view = Mat4::look_at_rh(Vec3::new(0.0, 100.0, 100.0), Vec3::ZERO, Vec3::Y);
    let proj = Mat4::perspective_rh(45.0_f32.to_radians(), 1.0, 1.0, 1000.0);
    let vp = proj * view;

    let frustum = FrustumPlanes::from_view_proj(vp);

    assert!((frustum.left.xyz().length() - 1.0).abs() < 0.01);
    assert!((frustum.right.xyz().length() - 1.0).abs() < 0.01);
}

#[test]
fn test_cpu_lod_selection() {
    let tiles = vec![
        TileInfo::new(0, 0, 0, Vec2::new(0.0, 0.0), Vec2::new(256.0, 256.0)),
        TileInfo::new(0, 1, 0, Vec2::new(256.0, 0.0), Vec2::new(512.0, 256.0)),
    ];

    let view = Mat4::look_at_rh(
        Vec3::new(128.0, 100.0, 128.0),
        Vec3::new(128.0, 0.0, 128.0),
        Vec3::Y,
    );
    let proj = Mat4::perspective_rh(45.0_f32.to_radians(), 1.0, 1.0, 10000.0);
    let config = GpuLodConfig::default();
    let result = cpu_lod_select(
        &tiles,
        proj * view,
        Vec3::new(128.0, 100.0, 128.0),
        &config,
        (0.0, 1000.0),
    );

    assert!(!result.visible_tiles.is_empty());
    assert!(result.total_triangles > 0);
}

#[test]
fn test_lod_selection_distance_based() {
    let config = GpuLodConfig {
        pixel_error_budget: 2.0,
        viewport_height: 1080,
        fov_y: 45.0_f32.to_radians(),
        max_lod: 4,
        tile_size: 256.0,
        ..Default::default()
    };

    let lod_close = select_lod_cpu(100.0, &config);
    let lod_far = select_lod_cpu(10000.0, &config);

    assert!(lod_far >= lod_close, "Far tiles should use coarser LOD");
}

#[test]
fn cpu_selection_keeps_a_tile_whose_bounds_intersect_the_frustum() {
    let tiles = vec![TileInfo::new(
        0,
        0,
        0,
        Vec2::new(0.9, -0.1),
        Vec2::new(1.5, 0.1),
    )];

    let result = cpu_lod_select(
        &tiles,
        Mat4::IDENTITY,
        Vec3::ZERO,
        &GpuLodConfig::default(),
        (0.0, 1000.0),
    );

    assert_eq!(
        result.visible_tiles.len(),
        1,
        "conservative culling must keep an AABB that crosses the frustum edge"
    );
}

#[test]
fn zero_to_one_near_plane_keeps_points_in_front_of_camera() {
    let eye = Vec3::new(0.0, 0.0, 10.0);
    let view = Mat4::look_at_rh(eye, Vec3::ZERO, Vec3::Y);
    let projection = Mat4::perspective_rh(45.0_f32.to_radians(), 1.0, 0.1, 100.0);
    let planes = FrustumPlanes::from_view_proj(projection * view);
    for plane in planes.to_array() {
        let distance = Vec3::new(plane[0], plane[1], plane[2]).dot(Vec3::ZERO) + plane[3];
        assert!(distance >= 0.0, "origin rejected by plane {plane:?}");
    }
}

fn is_adapter_unavailable(error: &crate::core::error::RenderError) -> bool {
    matches!(
        error,
        crate::core::error::RenderError::Device(message)
            if message.starts_with("No suitable GPU adapter found")
    )
}

#[test]
fn adapter_unavailable_skip_accepts_only_the_actual_typed_error() {
    use crate::core::error::RenderError;

    assert!(is_adapter_unavailable(&RenderError::Device(
        "No suitable GPU adapter found (requested backends: VULKAN)".to_string(),
    )));

    for error in [
        RenderError::Device("No compatible GPU adapter".to_string()),
        RenderError::Device("wrapped: No suitable GPU adapter found".to_string()),
        RenderError::Render("No suitable GPU adapter found".to_string()),
    ] {
        assert!(
            !is_adapter_unavailable(&error),
            "unexpected adapter-unavailable classification for {error:?}"
        );
    }
}

#[cfg(feature = "enable-globe")]
#[test]
fn gpu_and_cpu_select_identical_planetary_tile_ids_for_adapter_sweep() {
    use glam::DVec3;
    use std::time::{Duration, Instant};

    let context = match crate::core::gpu::try_ctx() {
        Ok(context) => context,
        Err(error) if is_adapter_unavailable(&error) => {
            eprintln!("planetary LOD parity skipped: {error}");
            return;
        }
        Err(error) => panic!("planetary LOD adapter setup failed: {error}"),
    };
    let config = GpuLodConfig {
        pixel_error_budget: 128.0,
        terrain_width: 10_000.0,
        tile_size: 50.0,
        max_lod: 4,
        ..Default::default()
    };
    let selector = GpuLodSelector::new(&context.device, config.clone());
    let radius = 5_000.0_f64;
    let world_tiles: Vec<_> = (0..5)
        .flat_map(|lat_index| {
            (0..24).map(move |lon_index| {
                let lon = lon_index as f64 * 15.0;
                let lat = -60.0 + lat_index as f64 * 30.0;
                (lon_index, lat_index, lon, lat)
            })
        })
        .collect();
    let seed_frame =
        crate::terrain::clipmap::globe::GlobeFrame::globe(radius, DVec3::X * (radius + 200.0))
            .unwrap();
    let make_tiles = |frame: &crate::terrain::clipmap::globe::GlobeFrame| {
        let mut tiles = world_tiles
            .iter()
            .map(|&(x, y, lon, lat)| {
                let center_ecef = frame.lonlat_alt_to_ecef(lon, lat, 0.0).unwrap();
                let center_relative = frame.camera_relative(center_ecef).unwrap().position;
                let half_extent = radius as f32 * 8.0_f32.to_radians();
                TileInfo::new(
                    0,
                    x,
                    y,
                    center_relative.truncate() - Vec2::splat(half_extent),
                    center_relative.truncate() + Vec2::splat(half_extent),
                )
                .with_height_bounds(
                    center_relative.z - half_extent,
                    center_relative.z + half_extent,
                )
                .with_globe_center(frame, center_ecef, 8.0_f32.to_radians())
                .unwrap()
            })
            .collect::<Vec<_>>();
        let antipodal_ecef = -frame.camera_anchor().normalize() * radius;
        tiles.push(
            TileInfo::new(0, 31, 31, Vec2::splat(-40_000.0), Vec2::splat(40_000.0))
                .with_height_bounds(-40_000.0, 40_000.0)
                .with_globe_center(frame, antipodal_ecef, std::f32::consts::PI)
                .unwrap(),
        );
        tiles
    };
    let seed_tiles = make_tiles(&seed_frame);
    let templates: Vec<_> = seed_tiles
        .iter()
        .flat_map(|tile| {
            (0..=config.max_lod).map(move |_| IndirectDrawTemplate {
                index_count: 3,
                first_index: 0,
                base_vertex: 0,
                tile_id: tile.tile_id,
            })
        })
        .collect();
    let resources = selector
        .create_draw_resources(&context.device, &seed_tiles, &templates)
        .unwrap();
    let view = Mat4::look_at_rh(Vec3::ZERO, Vec3::NEG_Z, Vec3::Y);
    let projection = Mat4::perspective_rh(config.fov_y, 16.0 / 9.0, 1.0, 40_000.0);
    let view_proj = projection * view;
    let mut visible_sets = std::collections::BTreeSet::new();
    let mut selected_lods = std::collections::BTreeSet::new();
    let mut state = 0xA511_E9B3_u32;

    for camera_index in 0..64 {
        state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        let lon = (state as f64 / u32::MAX as f64) * 360.0;
        state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        let lat = -50.0 + (state as f64 / u32::MAX as f64) * 100.0;
        let altitude = 200.0 + camera_index as f64 * 300.0;
        let anchor = seed_frame.lonlat_alt_to_ecef(lon, lat, altitude).unwrap();
        let frame = crate::terrain::clipmap::globe::GlobeFrame::globe(radius, anchor).unwrap();
        let tiles = make_tiles(&frame);
        let mut encoder = context
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("planetary_lod_parity"),
            });
        let provenance = LodSelectionProvenance(1_000 + camera_index as u64);
        let ticket = selector
            .encode_indirect_globe_tracked(
                &context.queue,
                &mut encoder,
                &resources,
                &tiles,
                view_proj,
                &frame,
                false,
                (-40_000.0, 40_000.0),
                true,
                provenance,
            )
            .expect("valid globe frame must produce planetary LOD parameters")
            .expect("a readback slot must be available after the prior result completes");
        context.queue.submit(Some(encoder.finish()));
        assert!(resources.mark_selection_submitted(ticket));
        let started = Instant::now();
        let completed = loop {
            if let Some(completed) = resources
                .try_read_selection(&context.device)
                .expect("planetary selection readback")
            {
                break completed;
            }
            assert!(
                started.elapsed() < Duration::from_secs(10),
                "camera {camera_index} GPU readback timed out"
            );
            std::thread::yield_now();
        };
        let gpu_result = completed
            .into_selection_for(provenance)
            .expect("completed selection must retain its originating camera provenance");
        let cpu_result =
            cpu_lod_select_globe(&tiles, view_proj, &config, (-40_000.0, 40_000.0), &frame)
                .unwrap();
        let mut cpu_ids: Vec<_> = cpu_result
            .visible_tiles
            .iter()
            .map(|tile| (tile.tile_id, tile.selected_lod))
            .collect();
        let mut gpu_ids: Vec<_> = gpu_result
            .visible_tiles
            .iter()
            .map(|tile| (tile.tile_id, tile.selected_lod))
            .collect();
        cpu_ids.sort_unstable();
        gpu_ids.sort_unstable();
        assert_eq!(gpu_ids, cpu_ids, "camera {camera_index}");
        assert!(
            gpu_ids
                .iter()
                .any(|&(tile_id, _)| tile_id == TileInfo::pack_id(0, 31, 31)),
            "wide antipodal bound must survive the GPU horizon test for camera {camera_index}"
        );
        visible_sets.insert(gpu_ids.iter().map(|&(id, _)| id).collect::<Vec<_>>());
        selected_lods.extend(gpu_ids.iter().map(|&(_, lod)| lod));
    }

    assert!(
        visible_sets.len() > 1,
        "camera sweep must exercise multiple visible tile sets"
    );
    assert!(
        selected_lods.len() > 1,
        "camera sweep must exercise multiple selected LODs"
    );
}

#[cfg(feature = "enable-globe")]
#[test]
fn planetary_frustum_visible_count_survives_a_b_a_reanchor() {
    use glam::DVec3;

    let radius = 6_371_000.0;
    let frame = crate::terrain::clipmap::globe::GlobeFrame::globe(
        radius,
        DVec3::X * (radius + 1_000.0),
    )
    .unwrap();
    // The planar bounds deliberately retain a previous anchor's translation.
    // The current camera-relative center is directly in front of the camera.
    let config = GpuLodConfig::default();
    let view = Mat4::look_at_rh(Vec3::ZERO, Vec3::NEG_Z, Vec3::Y);
    let projection = Mat4::perspective_rh(config.fov_y, 16.0 / 9.0, 0.1, 10_000.0);
    let visible_counts = [
        ([0.0, 0.0, -1_000.0], 50_000.0),
        ([100.0, 0.0, -1_000.0], -50_000.0),
        ([0.0, 0.0, -1_000.0], 50_000.0),
    ]
    .map(|(center, stale_translation)| {
        let mut tile = TileInfo::new(
            0,
            0,
            0,
            Vec2::splat(stale_translation),
            Vec2::splat(stale_translation + 100.0),
        )
        .with_height_bounds(-1_050.0, -950.0);
        tile.camera_relative_center = center;
        tile.angular_radius = 0.01;
        cpu_lod_select_globe(
            &[tile],
            projection * view,
            &config,
            (-2_000.0, 0.0),
            &frame,
        )
        .unwrap()
        .visible_tiles
        .len()
    });

    assert_eq!(visible_counts, [1, 1, 1]);
}
