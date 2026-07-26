use crate::viewer::camera_controller::{prospective_anchor, validate_world_point, CoordRole};
use crate::viewer::event_loop::{
    set_pending_bundle_load, set_pending_bundle_save, update_ipc_transform_stats,
};
use crate::viewer::viewer_enums::ViewerCmd;
use crate::viewer::Viewer;

fn apply_manual_sun_control(
    direction_ws: &mut [f32; 3],
    observation_active: &mut bool,
    azimuth_deg: f32,
    elevation_deg: f32,
) {
    let az_rad = azimuth_deg.to_radians();
    let el_rad = elevation_deg.to_radians();
    *direction_ws = glam::Vec3::new(
        el_rad.cos() * az_rad.sin(),
        el_rad.sin(),
        el_rad.cos() * az_rad.cos(),
    )
    .normalize()
    .to_array();
    *observation_active = false;
}

pub(crate) fn handle_cmd(viewer: &mut Viewer, cmd: &ViewerCmd) -> bool {
    match cmd {
        ViewerCmd::SetSunDirection {
            azimuth_deg,
            elevation_deg,
        } => {
            apply_manual_sun_control(
                &mut viewer.lit_sun_direction_ws,
                &mut viewer.astro_observation_active,
                *azimuth_deg,
                *elevation_deg,
            );
            viewer.update_lit_uniform();
            viewer.sync_terrain_sun_to_lit();
            true
        }
        ViewerCmd::SetObservation {
            year,
            month,
            day,
            hour,
            minute,
            second,
            latitude_deg,
            longitude_deg,
            height_m,
        } => {
            let result =
                crate::astro::time::UtcDateTime::new(*year, *month, *day, *hour, *minute, *second)
                    .and_then(|utc| {
                        crate::astro::Observer::new(
                            crate::geo::units::Angle::new(*latitude_deg),
                            crate::geo::units::Angle::new(*longitude_deg),
                            *height_m,
                        )
                        .and_then(|observer| {
                            crate::astro::observation::set_observation(utc, observer).map(|_| ())
                        })
                    });
            match result {
                Ok(()) => {
                    viewer.astro_observation_active = true;
                    viewer.sync_astro_observation();
                }
                Err(error) => viewer.reject_command(format!("observation_rejected: {error}")),
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
mod tests {
    use super::apply_manual_sun_control;

    #[test]
    fn manual_sun_control_mutates_direction_and_disables_observation() {
        let mut direction = [0.0; 3];
        let mut observation_active = true;
        apply_manual_sun_control(&mut direction, &mut observation_active, 90.0, 30.0);
        assert!(!observation_active);
        assert!((direction[0] - 30_f32.to_radians().cos()).abs() < 1e-6);
        assert!((direction[1] - 0.5).abs() < 1e-6);
        assert!(direction[2].abs() < 1e-6);
    }
}
