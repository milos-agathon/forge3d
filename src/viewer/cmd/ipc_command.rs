use crate::viewer::camera_controller::{prospective_anchor, validate_world_point, CoordRole};
use crate::viewer::event_loop::{
    set_pending_bundle_load, set_pending_bundle_save, update_ipc_transform_stats,
};
use crate::viewer::viewer_enums::ViewerCmd;
use crate::viewer::Viewer;

/// SIDERA azimuths run clockwise from north in the terrain's +z-south frame,
/// while `SetSunDirection` and the terrain sun keep their legacy mapping of
/// azimuth 0 to +z. Reflecting through the east-west axis gives the legacy
/// angle that produces the same world direction.
fn sidera_to_terrain_sun_azimuth_deg(azimuth_deg: f64) -> f64 {
    (180.0 - azimuth_deg).rem_euclid(360.0)
}

pub(crate) fn handle_cmd(viewer: &mut Viewer, cmd: &ViewerCmd) -> bool {
    match cmd {
        ViewerCmd::SetSunDirection {
            azimuth_deg,
            elevation_deg,
        } => {
            let az_rad = azimuth_deg.to_radians();
            let el_rad = elevation_deg.to_radians();
            let dir = glam::Vec3::new(
                el_rad.cos() * az_rad.sin(),
                el_rad.sin(),
                el_rad.cos() * az_rad.cos(),
            );
            if !azimuth_deg.is_finite()
                || !elevation_deg.is_finite()
                || !(-90.0..=90.0).contains(elevation_deg)
            {
                viewer.reject_command(
                    "sun angles must be finite and elevation within [-90, 90]".to_string(),
                );
                return true;
            }
            viewer.observation_sun_direction = Some(dir.to_array());
            viewer.observation_sky_sun_direction = None;
            viewer.observation_night_params = None;
            viewer.celestial_instances = None;
            viewer.celestial_instance_count = 0;
            if let Some(day_intensity) = viewer.observation_day_sun_intensity.take() {
                viewer.lit_sun_intensity = day_intensity;
            }
            viewer.update_lit_uniform();
            if let Some(ref mut terrain_viewer) = viewer.terrain_viewer {
                terrain_viewer.set_sun(*azimuth_deg, *elevation_deg, viewer.lit_sun_intensity);
            }
            true
        }
        ViewerCmd::SetSkyObservation {
            utc,
            latitude_deg,
            longitude_deg,
        } => {
            use crate::geo::units::{Angle, Degree};
            let observation = crate::astro::time::UtcDateTime::parse(utc).and_then(|time| {
                crate::astro::render::prepare(
                    time,
                    Angle::<Degree>::new(*latitude_deg),
                    Angle::<Degree>::new(*longitude_deg),
                )
            });
            let observation = match observation {
                Ok(value) => value,
                Err(error) => {
                    viewer.reject_command(error.to_string());
                    return true;
                }
            };
            let buffer = crate::core::resource_tracker::tracked_create_buffer_init(
                &viewer.device,
                &wgpu::util::BufferInitDescriptor {
                    label: Some("viewer.celestial.instances"),
                    contents: bytemuck::cast_slice(&observation.instances),
                    usage: wgpu::BufferUsages::VERTEX,
                },
            );
            let buffer = match buffer {
                Ok(value) => value,
                Err(error) => {
                    viewer.reject_command(error.to_string());
                    return true;
                }
            };
            let direction = |azimuth_deg: f64, altitude_deg: f64| {
                let (sa, ca) = azimuth_deg.to_radians().sin_cos();
                let (sh, ch) = altitude_deg.to_radians().sin_cos();
                [(ch * sa) as f32, sh as f32, (-ch * ca) as f32]
            };
            let sun = direction(
                observation.sun.azimuth.value(),
                observation.sun.refracted_altitude.value(),
            );
            let sky_sun = direction(
                observation.sun.azimuth.value(),
                observation.sun.altitude.value(),
            );
            let moon = direction(
                observation.moon.azimuth.value(),
                observation.moon.refracted_altitude.value(),
            );
            let day_intensity = *viewer
                .observation_day_sun_intensity
                .get_or_insert(viewer.lit_sun_intensity);
            let moonlit = observation.sun.altitude.value() < -6.0
                && observation.moon.refracted_altitude.value() > 0.0;
            let lit_direction = if moonlit { moon } else { sun };
            let lit_intensity = if moonlit {
                observation.moonlight_relative * 0.12
            } else if observation.sun.refracted_altitude.value() > 0.0 {
                day_intensity
            } else {
                0.0
            };
            viewer.observation_sun_direction = Some(lit_direction);
            viewer.observation_sky_sun_direction = Some(sky_sun);
            viewer.observation_night_params =
                Some([moon[0], moon[1], moon[2], observation.moonlight_relative]);
            viewer.lit_sun_intensity = lit_intensity;
            viewer.celestial_instance_count = observation.instances.len() as u32;
            viewer.celestial_instances = Some(buffer);
            viewer.sky_enabled = true;
            viewer.update_lit_uniform();
            if let Some(ref mut terrain_viewer) = viewer.terrain_viewer {
                let target = if moonlit {
                    &observation.moon
                } else {
                    &observation.sun
                };
                terrain_viewer.set_sun(
                    sidera_to_terrain_sun_azimuth_deg(target.azimuth.value()) as f32,
                    target.refracted_altitude.value() as f32,
                    lit_intensity,
                );
            }
            true
        }
        ViewerCmd::SetIbl { path, intensity } => {
            match viewer.load_ibl(path) {
                Ok(_) => {
                    viewer.lit_ibl_intensity = intensity.max(0.0);
                    viewer.lit_use_ibl = viewer.lit_ibl_intensity > 0.0;
                    viewer.update_lit_uniform();
                    println!("Loaded IBL: {} with intensity {:.2}", path, intensity);
                }
                Err(e) => eprintln!("IBL load failed: {}", e),
            }
            true
        }
        ViewerCmd::SetZScale(value) => {
            #[cfg(feature = "extension-module")]
            {
                if let Some(ref mut _scene) = viewer.terrain_scene {
                    println!(
                        "Terrain z-scale set to {:.2} (terrain scene attached)",
                        value
                    );
                } else {
                    eprintln!("SetZScale error: z-scale only applies to terrain scenes");
                }
            }
            #[cfg(not(feature = "extension-module"))]
            {
                let _ = value;
                eprintln!("SetZScale error: terrain support not compiled in");
            }
            true
        }
        ViewerCmd::SnapshotWithSize {
            path,
            width,
            height,
        } => {
            if let (Some(w), Some(h)) = (width, height) {
                viewer.view_config.snapshot_width = Some(*w);
                viewer.view_config.snapshot_height = Some(*h);
            }
            viewer.snapshot_request = Some(path.clone());
            true
        }
        ViewerCmd::SaveBundle { path, name } => {
            let bundle_name = name.as_deref().unwrap_or("scene");
            println!("SaveBundle requested: {} (name: {})", path, bundle_name);
            viewer.pending_bundle_save = Some((path.clone(), name.clone()));
            set_pending_bundle_save(path.clone(), name.clone());
            true
        }
        ViewerCmd::LoadBundle { path } => {
            println!("LoadBundle requested: {}", path);
            viewer.pending_bundle_load = Some(path.clone());
            set_pending_bundle_load(path.clone());
            true
        }
        ViewerCmd::SetFov(fov) => {
            viewer.view_config.fov_deg = fov.clamp(1.0, 179.0);
            println!("FOV set to {:.1}°", viewer.view_config.fov_deg);
            true
        }
        ViewerCmd::SetCamLookAt { eye, target, up } => {
            let eye = glam::DVec3::from(*eye);
            let target = glam::DVec3::from(*target);
            let up = glam::Vec3::from(*up);
            match viewer
                .camera
                .set_look_at(&viewer.camera_anchor, eye, target, up)
            {
                Ok(()) => println!("Camera: eye={:?} target={:?} up={:?}", eye, target, up),
                Err(err) => {
                    eprintln!("SetCamLookAt rejected: {err}");
                    viewer.reject_command(format!("camera_rejected: {err} unchanged_state=true"));
                }
            }
            true
        }
        ViewerCmd::SetSize(w, h) => {
            println!("Requested size {}x{} (resize via window manager)", w, h);
            true
        }
        ViewerCmd::SetVizDepthMax(_v) => true,
        ViewerCmd::SetTransform {
            translation,
            rotation_quat,
            scale,
        } => {
            let candidate_translation = translation
                .map(glam::DVec3::from)
                .unwrap_or(viewer.object_translation);
            let candidate_rotation = rotation_quat
                .map(glam::Quat::from_array)
                .unwrap_or(viewer.object_rotation);
            let candidate_scale = scale.map(glam::Vec3::from).unwrap_or(viewer.object_scale);
            let validation_anchor =
                if viewer.terrain_viewer.is_some() || viewer.point_cloud_active() {
                    viewer.prospective_frame_camera().anchor
                } else {
                    prospective_anchor(&viewer.camera_anchor, candidate_translation)
                };
            if validate_world_point(CoordRole::Object, candidate_translation, &validation_anchor)
                .is_err()
                || !candidate_rotation.is_finite()
                || candidate_rotation.length_squared() == 0.0
                || !candidate_scale.is_finite()
            {
                eprintln!("[viewer] SetTransform rejected transactionally");
                viewer.reject_command(
                    "object_transform_rejected: invalid finite/residual contract unchanged_state=true",
                );
                return true;
            }
            viewer.object_translation = candidate_translation;
            viewer.object_rotation = candidate_rotation.normalize();
            viewer.object_scale = candidate_scale;
            viewer.transform_version += 1;
            update_ipc_transform_stats(
                viewer.transform_version,
                viewer.object_transform_is_identity(),
            );
            true
        }
        _ => false,
    }
}

#[cfg(test)]
mod sidera_tests {
    use super::*;
    use std::sync::Arc;

    #[test]
    fn sidera_azimuth_reaches_terrain_as_the_same_world_direction() {
        for azimuth_deg in [0.0_f64, 37.5, 90.0, 163.7, 180.0, 271.0, 359.0] {
            let elevation = 21.0_f64.to_radians();
            let (sa, ca) = azimuth_deg.to_radians().sin_cos();
            let sidera = [elevation.cos() * sa, elevation.sin(), -elevation.cos() * ca];
            // Same formula as the viewer terrain's legacy sun setup.
            let legacy = sidera_to_terrain_sun_azimuth_deg(azimuth_deg).to_radians();
            let terrain = [
                elevation.cos() * legacy.sin(),
                elevation.sin(),
                elevation.cos() * legacy.cos(),
            ];
            for axis in 0..3 {
                assert!(
                    (sidera[axis] - terrain[axis]).abs() < 1e-12,
                    "{azimuth_deg}"
                );
            }
        }
    }

    #[test]
    fn sun_and_observation_commands_update_live_viewer() {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::util::backend_bits_from_env().unwrap_or(wgpu::Backends::all()),
            ..Default::default()
        });
        let Some(adapter) =
            pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
        else {
            eprintln!("SIDERA viewer integration ABSENT: no GPU adapter available");
            return;
        };
        let adapter = Arc::new(adapter);
        let Ok((device, queue)) =
            pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default(), None))
        else {
            eprintln!("SIDERA viewer integration ABSENT: GPU device unavailable");
            return;
        };
        let mut viewer = Viewer::new_headless(
            Arc::new(device),
            Arc::new(queue),
            adapter,
            64,
            64,
            crate::viewer::viewer_config::ViewerConfig::default(),
        )
        .unwrap();
        viewer
            .handle_cmd(ViewerCmd::SetSunDirection {
                azimuth_deg: 90.0,
                elevation_deg: 30.0,
            })
            .unwrap();
        let direction = viewer.observation_sun_direction.unwrap();
        assert!((direction[0] - 0.866_025_4).abs() < 1e-5);
        assert!((direction[1] - 0.5).abs() < 1e-5);
        viewer
            .handle_cmd(ViewerCmd::SetSunDirection {
                azimuth_deg: 0.0,
                elevation_deg: 30.0,
            })
            .unwrap();
        // SetSunDirection keeps its legacy convention: azimuth 0 maps to +z.
        let legacy_zero = viewer.observation_sun_direction.unwrap();
        assert!(legacy_zero[0].abs() < 1e-6);
        assert!((legacy_zero[2] - 0.866_025_4).abs() < 1e-5);
        viewer
            .handle_cmd(ViewerCmd::SetSkyObservation {
                utc: "2026-09-25T22:00:00Z".into(),
                latitude_deg: 52.37,
                longitude_deg: 4.9,
            })
            .unwrap();
        let observation = crate::astro::render::prepare(
            crate::astro::time::UtcDateTime::parse("2026-09-25T22:00:00Z").unwrap(),
            crate::geo::units::Angle::new(52.37),
            crate::geo::units::Angle::new(4.9),
        )
        .unwrap();
        let expected_sun_y = observation.sun.altitude.radians().sin() as f32;
        let expected_moon_y = observation.moon.refracted_altitude.radians().sin() as f32;
        assert!(viewer.celestial_instance_count > 4_000);
        assert!((viewer.observation_sky_sun_direction.unwrap()[1] - expected_sun_y).abs() < 1e-6);
        let night = viewer.observation_night_params.unwrap();
        assert!((night[1] - expected_moon_y).abs() < 1e-6);
        assert!(night[3] > 0.0);
        viewer.render_headless_frame(None).unwrap();
        viewer.resize(winit::dpi::PhysicalSize::new(96, 96));
        viewer.render_headless_frame(None).unwrap();
        // At sunset the visible, refracted Sun can still be above the horizon
        // while twilight must use its airless altitude below the horizon.
        viewer
            .handle_cmd(ViewerCmd::SetSkyObservation {
                utc: "2026-09-25T17:30:00Z".into(),
                latitude_deg: 52.37,
                longitude_deg: 4.9,
            })
            .unwrap();
        assert!(viewer.observation_sky_sun_direction.unwrap()[1] < 0.0);
        assert!(viewer.observation_sun_direction.unwrap()[1] > 0.0);
        viewer.render_headless_frame(None).unwrap();
    }
}
