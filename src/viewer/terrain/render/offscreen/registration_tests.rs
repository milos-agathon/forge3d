use crate::camera::Anchor;
use crate::core::texture_readback::read_texture_sync;
use crate::gis::raster_write::{write_raster, RasterArray, RasterData, WriteRasterOptions};
use crate::gis::types::{AffineTransform, PixelRegistration};
use crate::viewer::terrain::overlay::BlendMode;
use crate::viewer::terrain::scene::ViewerTerrainScene;
use crate::viewer::terrain::vector_overlay::{
    OverlayPrimitive, VectorOverlayLayer, VectorSourceVertex,
};
use crate::viewer::viewer_types::{ActiveCameraKind, FrameCamera};
use glam::{DVec3, Vec3};
use std::sync::Arc;
use wgpu::TextureFormat;

#[test]
fn viewer_registered_checkerboard_coincidence() {
    const DIM: u32 = 9;
    const VIEW: u32 = 63;
    const STEP: f64 = 8.0;
    const WEST: f64 = 500_000.0;
    const NORTH: f64 = 5_200_000.0;
    let root = std::env::var_os("FORGE3D_REGISTRATIO_VIEWER_ARTIFACT_DIR")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| {
            std::env::temp_dir().join(format!("forge3d-registratio-viewer-{}", std::process::id()))
        });
    std::fs::create_dir_all(&root).unwrap();

    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::util::backend_bits_from_env().unwrap_or(wgpu::Backends::PRIMARY),
        ..Default::default()
    });
    let adapter = Arc::new(
        pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
            .expect("viewer registration requires a physical GPU adapter"),
    );
    let info = adapter.get_info();
    println!(
        "viewer registration adapter: {} / {:?}",
        info.name, info.backend
    );
    assert!(matches!(
        info.device_type,
        wgpu::DeviceType::DiscreteGpu | wgpu::DeviceType::IntegratedGpu
    ));
    let (device, queue) = pollster::block_on(adapter.request_device(
        &wgpu::DeviceDescriptor {
            required_features: adapter.features()
                & wgpu::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES,
            required_limits: wgpu::Limits::default().using_resolution(adapter.limits()),
            label: Some("viewer registration"),
        },
        None,
    ))
    .unwrap();
    let (device, queue) = (Arc::new(device), Arc::new(queue));

    let centre = DVec3::new(
        WEST + f64::from(DIM) * STEP * 0.5,
        0.0,
        -NORTH + f64::from(DIM) * STEP * 0.5,
    );
    let mut anchor = Anchor::new();
    anchor.rebase_if_needed(centre);
    let frame = FrameCamera {
        kind: ActiveCameraKind::Terrain,
        anchor,
        eye_world: centre + DVec3::Y * 100.0,
        target_world: centre,
        up: Vec3::NEG_Z,
        fov_deg: 45.0,
        near: 1.0,
        far: 200.0,
        orthographic_half_extents: Some([VIEW as f32 * 0.5; 2]),
    };
    let format = TextureFormat::Rgba8UnormSrgb;
    let heights = (0..DIM)
        .flat_map(|row| {
            (0..DIM).map(move |col| {
                if matches!(col, 2 | 6) || matches!(row, 2 | 5) {
                    2.0_f32
                } else {
                    0.0
                }
            })
        })
        .collect::<Vec<_>>();
    let bands = heights
        .iter()
        .map(|&height| if height > 0.0 { 255 } else { 0 })
        .chain(std::iter::repeat_n(0, 2 * heights.len()))
        .chain(std::iter::repeat_n(255, heights.len()))
        .collect::<Vec<_>>();
    let ridge_x = |column: u32| WEST + (f64::from(column) + 0.5) * STEP;
    let source_vertices = [2, 6]
        .into_iter()
        .flat_map(|column| {
            [1, 7].map(|row| VectorSourceVertex {
                position: DVec3::new(ridge_x(column), 0.0, -NORTH + (f64::from(row) + 0.5) * STEP),
                color: [0.0, 0.0, 1.0, 1.0],
                feature_id: 0,
            })
        })
        .chain([2, 5].into_iter().flat_map(|row| {
            [1, 7].map(|col| VectorSourceVertex {
                position: DVec3::new(ridge_x(col), 0.0, -NORTH + (f64::from(row) + 0.5) * STEP),
                color: [0.0, 0.0, 1.0, 1.0],
                feature_id: 0,
            })
        }))
        .collect::<Vec<_>>();

    for registration in [PixelRegistration::Area, PixelRegistration::Point] {
        let name = registration.name();
        let dem_path = root.join(format!("{name}.tif"));
        let options = WriteRasterOptions {
            crs: Some(crate::gis::raster_write::CrsSpec::from_string("EPSG:32631".into()).unwrap()),
            transform: Some(AffineTransform::new([STEP, 0.0, WEST, 0.0, -STEP, NORTH]).unwrap()),
            nodata: vec![None],
            driver: "GTiff".into(),
            overwrite: true,
            creation_options: Default::default(),
            creation_options_explicit: false,
            like_info: None,
            height_system: crate::gis::types::HEIGHT_SYSTEM_UNSPECIFIED.into(),
            registration,
        };
        write_raster(
            &dem_path,
            RasterArray::new(
                RasterData::F32(heights.clone()),
                &[DIM as usize, DIM as usize],
            )
            .unwrap(),
            options.clone(),
        )
        .unwrap();
        let drape_path = root.join(format!("{name}-drape.tif"));
        write_raster(
            &drape_path,
            RasterArray::new(
                RasterData::U8(bands.clone()),
                &[4, DIM as usize, DIM as usize],
            )
            .unwrap(),
            WriteRasterOptions {
                nodata: vec![None; 4],
                ..options
            },
        )
        .unwrap();

        let mut scene =
            ViewerTerrainScene::new(device.clone(), queue.clone(), adapter.clone(), format)
                .unwrap();
        scene.load_terrain(dem_path.to_str().unwrap()).unwrap();
        scene.pbr_config.enabled = true;
        scene
            .add_overlay_image("drape", &drape_path, None, 1.0, BlendMode::Normal, 0)
            .unwrap();
        if registration == PixelRegistration::Point {
            let bad_drape = root.join("area-drape.tif");
            let err = scene
                .add_overlay_image("mismatch", &bad_drape, None, 1.0, BlendMode::Normal, 0)
                .unwrap_err();
            assert!(err.to_string().contains("raster registration mismatch"));
            assert!(scene
                .load_terrain(root.join("area.tif").to_str().unwrap())
                .is_err());
        }
        scene.set_overlay_preserve_colors(true);

        let snapshot = |scene: &mut ViewerTerrainScene| {
            let mut encoder = device.create_command_encoder(&Default::default());
            let texture = scene
                .render_to_texture(&mut encoder, format, VIEW, VIEW, 0, frame)
                .expect("viewer terrain snapshot must render");
            queue.submit([encoder.finish()]);
            read_texture_sync(
                &device,
                &queue,
                &texture,
                VIEW,
                VIEW,
                format,
                "registration-color",
            )
            .unwrap()
        };
        let base = snapshot(&mut scene);
        let depth = read_texture_sync(
            &device,
            &queue,
            scene.snapshot_depth_texture.as_ref().unwrap(),
            VIEW,
            VIEW,
            TextureFormat::Depth32Float,
            "registration-depth",
        )
        .unwrap();
        scene
            .add_vector_overlay_with_id(
                None,
                VectorOverlayLayer {
                    name: "coincident lines".into(),
                    source_vertices: source_vertices.clone(),
                    indices: vec![0, 1, 2, 3, 4, 5, 6, 7],
                    primitive: OverlayPrimitive::Lines,
                    drape: true,
                    ..Default::default()
                },
                &anchor,
            )
            .unwrap();
        let vectors = snapshot(&mut scene);
        for (suffix, rgba) in [("drape", base.as_slice()), ("vector", vectors.as_slice())] {
            image::save_buffer(
                root.join(format!("{name}-{suffix}.png")),
                rgba,
                VIEW,
                VIEW,
                image::ColorType::Rgba8,
            )
            .unwrap();
        }

        let middle = VIEW as usize / 2;
        for (axis, source) in [("east", 2), ("east", 6), ("south", 2), ("south", 5)] {
            let px = |position: usize| {
                if axis == "east" {
                    4 * (middle * VIEW as usize + position)
                } else {
                    4 * (position * VIEW as usize + middle)
                }
            };
            let expected = (f64::from(source) - 4.0) * STEP + f64::from(VIEW) * 0.5;
            let window = (0..VIEW as usize)
                .filter(|&col| (col as f64 + 0.5 - expected).abs() < 2.0 * STEP)
                .collect::<Vec<_>>();
            let value =
                |col: usize| f32::from_le_bytes(depth[px(col)..px(col) + 4].try_into().unwrap());
            let minimum = window
                .iter()
                .map(|&col| value(col))
                .fold(f32::INFINITY, f32::min);
            assert!(window.iter().any(|&col| value(col) > minimum));
            let positions = [
                average_peak(&window, |col| value(col) == minimum),
                average_peak(&window, |col| {
                    // Drape support remains red under terrain lighting.
                    let [red, green, blue] = base[px(col)..px(col) + 3] else {
                        unreachable!()
                    };
                    red > green && red > blue
                }),
                average_peak(&window, |col| vectors[px(col) + 2] > vectors[px(col)]),
            ];
            let residuals = positions.map(|position| position - expected);
            println!("viewer {name} {axis} ridge {source} residual pixels {residuals:?}, frames in {root:?}");
            assert!(residuals.iter().all(|r| r.abs() <= 1.0));
            for a in positions {
                assert!(positions.iter().all(|b| (a - b).abs() <= 1.0));
            }
        }
    }
}

fn average_peak(window: &[usize], is_peak: impl Fn(usize) -> bool) -> f64 {
    let peaks = window
        .iter()
        .copied()
        .filter(|&col| is_peak(col))
        .collect::<Vec<_>>();
    assert!(!peaks.is_empty(), "missing rendered feature");
    peaks.iter().map(|&col| col as f64 + 0.5).sum::<f64>() / peaks.len() as f64
}
