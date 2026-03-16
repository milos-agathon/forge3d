use crate::viewer::event_loop::{
    set_pending_bundle_load, set_pending_bundle_save, update_ipc_transform_stats,
};
use crate::viewer::viewer_enums::ViewerCmd;
use crate::viewer::viewer_types;
use crate::viewer::Viewer;
use wgpu::util::DeviceExt;

pub(crate) fn handle_cmd(viewer: &mut Viewer, cmd: &ViewerCmd) -> bool {
    match cmd {
        ViewerCmd::SetSunDirection {
            azimuth_deg,
            elevation_deg,
        } => {
            let az_rad = azimuth_deg.to_radians();
            let el_rad = elevation_deg.to_radians();
            let _dir = glam::Vec3::new(
                el_rad.cos() * az_rad.sin(),
                el_rad.sin(),
                el_rad.cos() * az_rad.cos(),
            );
            println!(
                "Sun direction: azimuth={:.1}° elevation={:.1}°",
                azimuth_deg, elevation_deg
            );
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
            let eye = glam::Vec3::from(*eye);
            let target = glam::Vec3::from(*target);
            let up = glam::Vec3::from(*up);
            viewer.camera.set_look_at(eye, target, up);
            println!("Camera: eye={:?} target={:?} up={:?}", eye, target, up);
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
            if let Some(t) = translation {
                viewer.object_translation = glam::Vec3::from(*t);
            }
            if let Some(q) = rotation_quat {
                viewer.object_rotation = glam::Quat::from_array(*q).normalize();
            }
            if let Some(s) = scale {
                viewer.object_scale = glam::Vec3::from(*s);
            }
            viewer.object_transform = glam::Mat4::from_scale_rotation_translation(
                viewer.object_scale,
                viewer.object_rotation,
                viewer.object_translation,
            );

            if !viewer.original_mesh_positions.is_empty() {
                use viewer_types::PackedVertex;

                let vertex_count = viewer.original_mesh_positions.len();
                let mut vertices: Vec<PackedVertex> = Vec::with_capacity(vertex_count);

                for i in 0..vertex_count {
                    let orig_pos = glam::Vec3::from(viewer.original_mesh_positions[i]);
                    let transformed = viewer.object_transform.transform_point3(orig_pos);

                    let orig_nrm = if i < viewer.original_mesh_normals.len() {
                        glam::Vec3::from(viewer.original_mesh_normals[i])
                    } else {
                        glam::Vec3::Y
                    };
                    let rot_mat = glam::Mat3::from_quat(viewer.object_rotation);
                    let transformed_nrm = (rot_mat * orig_nrm).normalize();

                    let uv = if i < viewer.original_mesh_uvs.len() {
                        viewer.original_mesh_uvs[i]
                    } else {
                        [0.0, 0.0]
                    };

                    vertices.push(PackedVertex {
                        position: transformed.to_array(),
                        normal: transformed_nrm.to_array(),
                        uv,
                        rough_metal: [0.5, 0.0],
                    });
                }

                let vertex_data = bytemuck::cast_slice(&vertices);
                let new_vb = viewer
                    .device
                    .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("viewer.ipc.mesh.vb.transformed"),
                        contents: vertex_data,
                        usage: wgpu::BufferUsages::VERTEX,
                    });
                viewer.geom_vb = Some(new_vb);

                let msg = format!(
                    "[D1-CPU-TRANSFORM] frame={} vertices={} trans=[{:.3},{:.3},{:.3}] scale=[{:.3},{:.3},{:.3}]\n",
                    viewer.frame_count,
                    vertex_count,
                    viewer.object_translation.x,
                    viewer.object_translation.y,
                    viewer.object_translation.z,
                    viewer.object_scale.x,
                    viewer.object_scale.y,
                    viewer.object_scale.z
                );
                let _ = std::fs::OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open("examples/out/d1_debug.log")
                    .and_then(|mut f| {
                        use std::io::Write;
                        f.write_all(msg.as_bytes())
                    });
            }

            viewer.transform_version += 1;
            let is_identity = viewer.object_translation == glam::Vec3::ZERO
                && viewer.object_rotation == glam::Quat::IDENTITY
                && viewer.object_scale == glam::Vec3::ONE;
            update_ipc_transform_stats(viewer.transform_version, is_identity);
            true
        }
        _ => false,
    }
}
