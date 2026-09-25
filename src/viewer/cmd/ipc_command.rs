use crate::viewer::camera_controller::{prospective_anchor, validate_world_point, CoordRole};
use crate::viewer::event_loop::{
    set_pending_bundle_load, set_pending_bundle_save, update_ipc_transform_stats,
};
use crate::viewer::viewer_enums::ViewerCmd;
use crate::viewer::Viewer;

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
                [(ch * sa) as f32, sh as f32, (ch * ca) as f32]
            };
            let sun = direction(
                observation.sun.azimuth.value(),
                observation.sun.altitude.value(),
            );
            let moon = direction(
                observation.moon.azimuth.value(),
                observation.moon.altitude.value(),
            );
            let day_intensity = *viewer
                .observation_day_sun_intensity
                .get_or_insert(viewer.lit_sun_intensity);
            let moonlit =
                observation.sun.altitude.value() < -6.0 && observation.moon.altitude.value() > 0.0;
            let lit_direction = if moonlit { moon } else { sun };
            let lit_intensity = if moonlit {
                observation.moonlight_relative * 0.12
            } else if observation.sun.altitude.value() > 0.0 {
                day_intensity
            } else {
                0.0
            };
            viewer.observation_sun_direction = Some(lit_direction);
            viewer.observation_sky_sun_direction = Some(sun);
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
                    target.azimuth.value() as f32,
                    target.altitude.value() as f32,
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
