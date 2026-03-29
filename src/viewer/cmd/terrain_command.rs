use crate::viewer::terrain;
use crate::viewer::viewer_enums::ViewerCmd;
use crate::viewer::Viewer;

pub(crate) fn handle_cmd(viewer: &mut Viewer, cmd: &ViewerCmd) -> bool {
    match cmd {
        ViewerCmd::LoadTerrain(path) => {
            println!("[XYZZY_TERRAIN] LoadTerrain handler entry, path={}", path);
            if viewer.terrain_viewer.is_none() {
                eprintln!("[DEBUG LoadTerrain] Creating new terrain_viewer");
                match terrain::ViewerTerrainScene::new(
                    std::sync::Arc::clone(&viewer.device),
                    std::sync::Arc::clone(&viewer.queue),
                    viewer.config.format,
                ) {
                    Ok(scene) => {
                        viewer.terrain_viewer = Some(scene);
                        eprintln!("[DEBUG LoadTerrain] terrain_viewer created successfully");
                    }
                    Err(e) => {
                        eprintln!("[terrain] Failed to create viewer: {}", e);
                        return true;
                    }
                }
            } else {
                eprintln!("[DEBUG LoadTerrain] terrain_viewer already exists");
            }
            if let Some(ref mut terrain_viewer) = viewer.terrain_viewer {
                match terrain_viewer.load_terrain(path) {
                    Ok(()) => {
                        println!("[terrain] Loaded: {}", path);
                        eprintln!(
                            "[DEBUG LoadTerrain] terrain_viewer has_terrain={}",
                            terrain_viewer.has_terrain()
                        );
                    }
                    Err(e) => eprintln!("[terrain] Failed to load {}: {}", path, e),
                }
            }
            if let Err(err) = viewer.reapply_scene_review_state() {
                eprintln!("[scene_review] failed to reapply after terrain load: {err}");
            }
            true
        }
        ViewerCmd::SetTerrainCamera {
            phi_deg,
            theta_deg,
            radius,
            fov_deg,
        } => {
            if let Some(ref mut terrain_viewer) = viewer.terrain_viewer {
                terrain_viewer.set_camera(*phi_deg, *theta_deg, *radius, *fov_deg);
                println!(
                    "[terrain] Camera: phi={:.1}° theta={:.1}° r={:.1} fov={:.1}°",
                    phi_deg, theta_deg, radius, fov_deg
                );
            }
            true
        }
        ViewerCmd::SetTerrainSun {
            azimuth_deg,
            elevation_deg,
            intensity,
        } => {
            if let Some(ref mut terrain_viewer) = viewer.terrain_viewer {
                terrain_viewer.set_sun(*azimuth_deg, *elevation_deg, *intensity);
                println!(
                    "[terrain] Sun: az={:.1}° el={:.1}° int={:.2}",
                    azimuth_deg, elevation_deg, intensity
                );
            }
            true
        }
        ViewerCmd::SetTerrain {
            phi,
            theta,
            radius,
            fov,
            sun_azimuth,
            sun_elevation,
            sun_intensity,
            ambient,
            zscale,
            shadow,
            background,
            water_level,
            water_color,
        } => {
            if let Some(ref mut terrain_viewer) = viewer.terrain_viewer {
                if let Some(terrain) = terrain_viewer.terrain.as_mut() {
                    if let Some(v) = phi {
                        terrain.cam_phi_deg = *v;
                    }
                    if let Some(v) = theta {
                        terrain.cam_theta_deg = v.clamp(5.0, 85.0);
                    }
                    if let Some(v) = radius {
                        terrain.cam_radius = v.clamp(100.0, 50000.0);
                    }
                    if let Some(v) = fov {
                        terrain.cam_fov_deg = v.clamp(10.0, 120.0);
                    }
                    if let Some(v) = sun_azimuth {
                        terrain.sun_azimuth_deg = *v;
                    }
                    if let Some(v) = sun_elevation {
                        terrain.sun_elevation_deg = v.clamp(-90.0, 90.0);
                    }
                    if let Some(v) = sun_intensity {
                        terrain.sun_intensity = v.max(0.0);
                    }
                    if let Some(v) = ambient {
                        terrain.ambient = v.clamp(0.0, 1.0);
                    }
                    if let Some(v) = zscale {
                        terrain.z_scale = v.max(0.01);
                    }
                    if let Some(v) = shadow {
                        terrain.shadow_intensity = v.clamp(0.0, 1.0);
                    }
                    if let Some(bg) = background {
                        terrain.background_color = *bg;
                    }
                    if let Some(v) = water_level {
                        terrain.water_level = *v;
                    }
                    if let Some(color) = water_color {
                        terrain.water_color = *color;
                    }
                }
                if let Some(params) = terrain_viewer.get_params() {
                    println!("[terrain] {}", params);
                }
            }
            true
        }
        ViewerCmd::GetTerrainParams => {
            if let Some(ref terrain_viewer) = viewer.terrain_viewer {
                if let Some(params) = terrain_viewer.get_params() {
                    println!("[terrain] {}", params);
                }
            }
            true
        }
        ViewerCmd::SetTerrainScatter { batches } => {
            #[cfg(feature = "enable-gpu-instancing")]
            {
                if let Some(ref mut terrain_viewer) = viewer.terrain_viewer {
                    match terrain_viewer.set_scatter_batches_from_configs(batches) {
                        Ok(()) => {
                            println!("[terrain] scatter batches set: {}", batches.len());
                        }
                        Err(e) => eprintln!("[terrain] Failed to set scatter batches: {e:#}"),
                    }
                } else {
                    eprintln!("[terrain] Load terrain before setting scatter batches");
                }
            }
            #[cfg(not(feature = "enable-gpu-instancing"))]
            {
                let _ = batches;
                eprintln!(
                    "[terrain] Scatter batches require Cargo feature 'enable-gpu-instancing'"
                );
            }
            true
        }
        ViewerCmd::ClearTerrainScatter => {
            #[cfg(feature = "enable-gpu-instancing")]
            {
                if let Some(ref mut terrain_viewer) = viewer.terrain_viewer {
                    terrain_viewer.clear_scatter_batches();
                    println!("[terrain] scatter batches cleared");
                }
            }
            #[cfg(not(feature = "enable-gpu-instancing"))]
            {
                eprintln!(
                    "[terrain] Scatter batches require Cargo feature 'enable-gpu-instancing'"
                );
            }
            true
        }
        ViewerCmd::SetTerrainPbr {
            enabled,
            hdr_path,
            ibl_intensity,
            shadow_technique,
            shadow_map_res,
            exposure,
            msaa,
            normal_strength,
            height_ao,
            sun_visibility,
            materials,
            vector_overlay,
            tonemap,
            dof,
            motion_blur,
            lens_effects,
            denoise,
            volumetrics,
            sky,
            debug_mode,
        } => {
            if let Some(ref mut terrain_viewer) = viewer.terrain_viewer {
                terrain_viewer.set_terrain_pbr(
                    *enabled,
                    hdr_path.clone(),
                    *ibl_intensity,
                    shadow_technique.clone(),
                    *shadow_map_res,
                    *exposure,
                    *msaa,
                    *normal_strength,
                    height_ao.as_ref().clone(),
                    sun_visibility.as_ref().clone(),
                    materials.as_ref().clone(),
                    vector_overlay.as_ref().clone(),
                    tonemap.clone(),
                    lens_effects.clone(),
                    dof.as_ref().clone(),
                    motion_blur.clone(),
                    volumetrics.as_ref().clone(),
                    denoise.clone(),
                    *debug_mode,
                );
            }

            if let Some(ref cfg) = sky {
                viewer.sky_enabled = cfg.enabled;
                if cfg.enabled {
                    viewer.sky_turbidity = cfg.turbidity;
                    viewer.sky_ground_albedo = cfg.ground_albedo;
                    viewer.sky_exposure = cfg.sky_exposure;
                    viewer.sky_sun_intensity = cfg.sun_intensity;
                    println!(
                        "[terrain] Sky enabled: turbidity={:.1} ground_albedo={:.2} exposure={:.2}",
                        cfg.turbidity, cfg.ground_albedo, cfg.sky_exposure
                    );
                }
            }
            true
        }
        ViewerCmd::LoadOverlay {
            name,
            path,
            extent,
            opacity,
            z_order,
        } => {
            println!(
                "[overlay] LoadOverlay command received: name='{}' path='{}'",
                name, path
            );
            if let Some(ref mut terrain_viewer) = viewer.terrain_viewer {
                let opacity = opacity.unwrap_or(1.0);
                let z_order = z_order.unwrap_or(0);
                println!("[overlay] terrain_viewer exists, calling add_overlay_image...");
                match terrain_viewer.add_overlay_image(
                    name,
                    std::path::Path::new(path),
                    extent.clone(),
                    opacity,
                    crate::viewer::terrain::BlendMode::Normal,
                    z_order,
                ) {
                    Ok(id) => println!("[overlay] Loaded '{}' from {} (id={})", name, path, id),
                    Err(e) => eprintln!("[overlay] Failed to load '{}': {}", name, e),
                }
            } else {
                eprintln!(
                    "[overlay] No terrain loaded - load terrain first (terrain_viewer is None)"
                );
            }
            true
        }
        ViewerCmd::RemoveOverlay { id } => {
            if let Some(ref mut terrain_viewer) = viewer.terrain_viewer {
                if terrain_viewer.remove_overlay(*id) {
                    println!("[overlay] Removed overlay id={}", id);
                } else {
                    eprintln!("[overlay] Overlay id={} not found", id);
                }
            }
            true
        }
        ViewerCmd::SetOverlayVisible { id, visible } => {
            if let Some(ref mut terrain_viewer) = viewer.terrain_viewer {
                terrain_viewer.set_overlay_visible(*id, *visible);
                println!("[overlay] id={} visible={}", id, visible);
            }
            true
        }
        ViewerCmd::SetOverlayOpacity { id, opacity } => {
            if let Some(ref mut terrain_viewer) = viewer.terrain_viewer {
                terrain_viewer.set_overlay_opacity(*id, *opacity);
                println!("[overlay] id={} opacity={:.2}", id, opacity);
            }
            true
        }
        ViewerCmd::SetGlobalOverlayOpacity { opacity } => {
            if let Some(ref mut terrain_viewer) = viewer.terrain_viewer {
                terrain_viewer.set_global_overlay_opacity(*opacity);
                println!("[overlay] global opacity={:.2}", opacity);
            }
            true
        }
        ViewerCmd::SetOverlaysEnabled { enabled } => {
            if let Some(ref mut terrain_viewer) = viewer.terrain_viewer {
                terrain_viewer.set_overlays_enabled(*enabled);
                println!("[overlay] enabled={}", enabled);
            }
            true
        }
        ViewerCmd::SetOverlaySolid { solid } => {
            if let Some(ref mut terrain_viewer) = viewer.terrain_viewer {
                terrain_viewer.set_overlay_solid(*solid);
                println!("[overlay] solid={}", solid);
            }
            true
        }
        ViewerCmd::ListOverlays => {
            if let Some(ref terrain_viewer) = viewer.terrain_viewer {
                let ids = terrain_viewer.list_overlays();
                if ids.is_empty() {
                    println!("[overlay] No overlays loaded");
                } else {
                    println!("[overlay] Loaded overlays: {:?}", ids);
                }
            } else {
                println!("[overlay] No terrain loaded");
            }
            true
        }
        _ => false,
    }
}
