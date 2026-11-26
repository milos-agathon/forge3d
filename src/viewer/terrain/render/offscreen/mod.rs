mod effects;
mod scene;
mod setup;

use super::*;

pub(super) struct SnapshotRenderState {
    pub(super) use_pbr: bool,
    pub(super) view_mat: glam::Mat4,
    pub(super) proj: glam::Mat4,
    pub(super) view_proj: glam::Mat4,
    pub(super) sun_dir: glam::Vec3,
    pub(super) eye: glam::Vec3,
    pub(super) render_origin_span: [f32; 4],
    pub(super) h_range: f32,
    pub(super) shader_z_scale: f32,
    /// Clip planes of the frame camera (depth linearisation in effects).
    pub(super) near: f32,
    pub(super) far: f32,
    pub(super) vo_view_proj: [[f32; 4]; 4],
    pub(super) vo_sun_dir: [f32; 3],
    pub(super) vo_lighting: [f32; 4],
}

impl ViewerTerrainScene {
    pub fn render_to_texture(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        target_format: wgpu::TextureFormat,
        width: u32,
        height: u32,
        selected_feature_id: u32,
        frame: crate::viewer::viewer_types::FrameCamera,
    ) -> Option<crate::core::resource_tracker::TrackedTexture> {
        self.snapshot_depth_texture = None;
        eprintln!("[DEBUG render_to_texture ENTRY] {}x{}", width, height);
        if self.terrain.is_none() {
            eprintln!("[DEBUG render_to_texture] No terrain, returning None");
            return None;
        }

        self.prepare_snapshot_resources(width, height);
        let scene_format = self.scene_color_format_for(target_format);
        let (color_tex, color_view) = match self.create_snapshot_color_target(
            "terrain_viewer.snapshot_color",
            scene_format,
            width,
            height,
        ) {
            Ok(v) => v,
            Err(e) => {
                eprintln!("[terrain] failed to allocate snapshot color target: {e}");
                return None;
            }
        };
        let (depth_tex, depth_view) = match self.create_snapshot_depth_target(width, height) {
            Ok(v) => v,
            Err(e) => {
                eprintln!("[terrain] failed to allocate snapshot depth target: {e}");
                return None;
            }
        };
        let state =
            match self.build_snapshot_render_state(encoder, scene_format, width, height, frame) {
                Ok(state) => state,
                Err(error) => {
                    self.canonical_media_render_error =
                        Some(format!("media_prepare_failed: {error:#}"));
                    return None;
                }
            };
        let has_vector_overlays = self.prepare_snapshot_overlays();

        self.render_snapshot_scene_pass(
            encoder,
            &color_view,
            &depth_view,
            selected_feature_id,
            &state,
            has_vector_overlays,
        );
        self.render_snapshot_oit_pass(
            encoder,
            &color_view,
            &depth_view,
            width,
            height,
            selected_feature_id,
            &state,
            has_vector_overlays,
        );

        let output = self.apply_snapshot_effects(
            encoder,
            target_format,
            width,
            height,
            &depth_tex,
            &depth_view,
            color_tex,
            color_view,
            &state,
        );
        let output = match output {
            Ok(output) => {
                self.canonical_media_render_error = None;
                output
            }
            Err(error) => {
                self.canonical_media_render_error =
                    Some(format!("media_snapshot_render_failed: {error:#}"));
                return None;
            }
        };
        self.snapshot_depth_texture = Some(depth_tex);
        Some(output)
    }
}

#[cfg(not(target_arch = "wasm32"))]
#[cfg(test)]
mod registration_tests {
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

        let (adapter, device, queue) = test_device();

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

    fn test_device() -> (Arc<wgpu::Adapter>, Arc<wgpu::Device>, Arc<wgpu::Queue>) {
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
        (adapter, Arc::new(device), Arc::new(queue))
    }

    #[test]
    fn vacuum_viewer_masked_geometry_depth() {
        const DIM: usize = 9;
        const VIEW: u32 = 63;
        let root = std::env::var_os("FORGE3D_REGISTRATIO_VIEWER_ARTIFACT_DIR")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|| {
                std::env::temp_dir().join(format!("vacuum-viewer-{}", std::process::id()))
            });
        std::fs::create_dir_all(&root).unwrap();
        let (adapter, device, queue) = test_device();
        let format = TextureFormat::Rgba8UnormSrgb;
        let centre = DVec3::new(36.0, 0.0, 36.0);
        let frame = FrameCamera {
            kind: ActiveCameraKind::Terrain,
            anchor: Anchor::new(),
            eye_world: centre + DVec3::Y * 100.0,
            target_world: centre,
            up: Vec3::NEG_Z,
            fov_deg: 45.0,
            near: 1.0,
            far: 200.0,
            orthographic_half_extents: Some([VIEW as f32 * 0.5; 2]),
        };
        let mut scene =
            ViewerTerrainScene::new(device.clone(), queue.clone(), adapter, format).unwrap();
        scene.pbr_config.enabled = true;
        let mut dense = vec![12.0; DIM * DIM];
        dense[0] = 2.0; // A floor distinctly below the surface around the hole.
        let mut render = |name: &str, data: Vec<f32>, marker, mask: Option<&[u8]>| {
            let path = root.join(format!("{name}.tif"));
            write_raster(
                &path,
                RasterArray::new(RasterData::F32(data), &[DIM, DIM]).unwrap(),
                WriteRasterOptions {
                    crs: Some(
                        crate::gis::raster_write::CrsSpec::from_string("EPSG:32631".into()).unwrap(),
                    ),
                    transform: Some(AffineTransform::new([8., 0., 0., 0., -8., 0.]).unwrap()),
                    nodata: vec![marker],
                    driver: "GTiff".into(),
                    overwrite: true,
                    creation_options: Default::default(),
                    creation_options_explicit: false,
                    like_info: None,
                    height_system: crate::gis::types::HEIGHT_SYSTEM_UNSPECIFIED.into(),
                    registration: PixelRegistration::Area,
                },
            )
            .unwrap();
            if let Some(mask) = mask {
                // RFC 15 dataset mask. Every elevation in the masked hole is finite.
                let file = std::fs::File::create(path.with_extension("tif.msk")).unwrap();
                let mut encoder = tiff::encoder::TiffEncoder::new(file).unwrap();
                let mut image = encoder.new_image::<tiff::encoder::colortype::Gray8>(
                    DIM as u32, DIM as u32,
                ).unwrap();
                image.encoder().write_tag(tiff::tags::Tag::Unknown(42112),
                    r#"<GDALMetadata><Item name="INTERNAL_MASK_FLAGS_1">2</Item></GDALMetadata>"#).unwrap();
                image.write_data(mask).unwrap();
            }
            scene.load_terrain(path.to_str().unwrap()).unwrap();
            let mut encoder = device.create_command_encoder(&Default::default());
            let texture = scene
                .render_to_texture(&mut encoder, format, VIEW, VIEW, 0, frame)
                .unwrap();
            queue.submit([encoder.finish()]);
            let rgba = read_texture_sync(
                &device,
                &queue,
                &texture,
                VIEW,
                VIEW,
                format,
                "vacuum-viewer",
            )
            .unwrap();
            image::save_buffer(
                root.join(format!("{name}.png")),
                &rgba,
                VIEW,
                VIEW,
                image::ColorType::Rgba8,
            )
            .unwrap();
            let raw = read_texture_sync(
                &device,
                &queue,
                scene.snapshot_depth_texture.as_ref().unwrap(),
                VIEW,
                VIEW,
                TextureFormat::Depth32Float,
                "vacuum-depth",
            )
            .unwrap();
            std::fs::write(root.join(format!("{name}.depth-f32")), &raw).unwrap();
            let depth = raw
                .chunks_exact(4)
                .map(|v| f32::from_le_bytes(v.try_into().unwrap()))
                .collect::<Vec<_>>();
            (rgba, depth)
        };
        let reference = render("dense", dense.clone(), None, None);
        let mut low = dense.clone();
        low[0] = -1500.0;
        low[1] = 2.0;
        let mut expected = low.clone();
        expected[0] = 2.0;
        let floor_reference = render("legacy-floor", expected, None, None);
        assert_eq!(render("untagged-low", low.clone(), None, None), floor_reference);
        assert_eq!(render("unused-tag-low", low, Some(-9999.0), None), floor_reference);
        let mut first = None;
        for (i, marker) in [-9999.0, -32768.0, 3.4e38, f32::NAN]
            .into_iter()
            .enumerate()
        {
            let mut values = dense.clone();
            for y in 3..6 {
                values[y * DIM + 3..y * DIM + 6].fill(marker);
            }
            let result = render(&format!("hole-{i}"), values, Some(f64::from(marker)), None);
            let mut holes = 0;
            let mut valid = 0;
            for (&depth, &full) in result.1.iter().zip(&reference.1) {
                if depth == 1.0 {
                    holes += usize::from(full < 1.0);
                } else {
                    assert!(
                        depth.to_bits().abs_diff(full.to_bits()) <= 1,
                        "depth {depth} versus {full}"
                    );
                    valid += 1;
                }
            }
            assert!(holes > 0 && valid > holes);
            let mut bordering = 0;
            for y in 1..VIEW as usize - 1 {
                for x in 1..VIEW as usize - 1 {
                    let pixel = y * VIEW as usize + x;
                    let neighbours = [
                        pixel - 1,
                        pixel + 1,
                        pixel - VIEW as usize,
                        pixel + VIEW as usize,
                    ];
                    if result.1[pixel] < 1.0
                        && neighbours
                            .iter()
                            .any(|&n| result.1[n] == 1.0 && reference.1[n] < 1.0)
                    {
                        let color = 4 * pixel..4 * (pixel + 1);
                        assert_eq!(
                            &result.0[color.clone()],
                            &reference.0[color],
                            "rim at {x},{y}"
                        );
                        bordering += 1;
                    }
                }
            }
            assert!(bordering > 0);
            println!("viewer marker {marker}: {valid} visible, {holes} removed, {bordering} border pixels, zero rim, within one depth ULP");
            if let Some(frame) = &first {
                assert_eq!(frame, &result);
            } else {
                first = Some(result);
            }
        }
        let mut mask = vec![255u8; DIM * DIM];
        for y in 3..6 {
            mask[y * DIM + 3..y * DIM + 6].fill(0);
        }
        let explicit = render("explicit-mask", dense, None, Some(&mask));
        assert_eq!(explicit, first.unwrap(), "file mask must produce the same colour and depth as a tagged hole");
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
}
