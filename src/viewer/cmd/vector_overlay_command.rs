use crate::viewer::viewer_enums::ViewerCmd;
use crate::viewer::Viewer;

pub(crate) fn handle_cmd(viewer: &mut Viewer, cmd: &ViewerCmd) -> bool {
    match cmd {
        ViewerCmd::AddVectorOverlay {
            id,
            name,
            vertices,
            indices,
            primitive,
            drape,
            drape_offset,
            opacity,
            depth_bias,
            line_width,
            point_size,
            z_order,
        } => {
            let world_points = vertices
                .iter()
                .map(|vertex| glam::DVec3::from(vertex.position))
                .collect::<Vec<_>>();
            if let Err(err) = viewer.validate_content_points(world_points.iter().copied()) {
                eprintln!("[vector_overlay] rejected '{name}': {err}");
                return true;
            }
            let frame = viewer.prospective_frame_camera();
            if let Some(ref mut terrain_viewer) = viewer.terrain_viewer {
                use crate::viewer::terrain::vector_overlay::{
                    OverlayPrimitive, VectorOverlayLayer, VectorSourceVertex,
                };

                let source_vertices: Vec<VectorSourceVertex> = vertices
                    .iter()
                    .copied()
                    .map(VectorSourceVertex::from)
                    .collect();
                let primitive =
                    OverlayPrimitive::from_str(primitive).unwrap_or(OverlayPrimitive::Triangles);

                let layer = VectorOverlayLayer {
                    name: name.clone(),
                    source_vertices,
                    vertices: Vec::new(),
                    indices: indices.clone(),
                    primitive,
                    drape: *drape,
                    drape_offset: *drape_offset,
                    opacity: *opacity,
                    depth_bias: *depth_bias,
                    line_width: *line_width,
                    point_size: *point_size,
                    visible: true,
                    z_order: *z_order,
                };

                let id = match terrain_viewer.add_vector_overlay_with_id(*id, layer, &frame.anchor)
                {
                    Ok(id) => id,
                    Err(e) => {
                        eprintln!("[vector_overlay] failed to add '{}': {e}", name);
                        return true;
                    }
                };
                println!(
                    "[vector_overlay] Added '{}' with {} vertices (id={})",
                    name,
                    vertices.len(),
                    id
                );

                if let Some((layer_name, render_vertices, render_indices, primitive)) =
                    terrain_viewer.vector_overlay_render_data(id)
                {
                    if let Some(layer_data) =
                        crate::viewer::terrain::vector_overlay::build_layer_bvh(
                            id,
                            &layer_name,
                            &render_vertices,
                            &render_indices,
                            primitive,
                        )
                    {
                        viewer.unified_picking.register_layer_bvh(layer_data);
                    }
                }
            } else {
                eprintln!("[vector_overlay] No terrain loaded - load terrain first");
            }
            true
        }
        ViewerCmd::PollPickEvents => true,
        ViewerCmd::SetLassoMode { enabled } => {
            viewer.unified_picking.set_lasso_enabled(*enabled);
            let state = if *enabled { "active" } else { "inactive" };
            if let Ok(mut lasso_state) = crate::viewer::event_loop::get_lasso_state().lock() {
                *lasso_state = state.to_string();
            }
            println!("[picking] Lasso mode: {}", state);
            true
        }
        ViewerCmd::GetLassoState => true,
        ViewerCmd::ClearSelection => {
            println!("[picking] Clear selection requested");
            true
        }
        ViewerCmd::RemoveVectorOverlay { id } => {
            if let Some(ref mut terrain_viewer) = viewer.terrain_viewer {
                if terrain_viewer.remove_vector_overlay(*id) {
                    println!("[vector_overlay] Removed id={}", id);
                } else {
                    eprintln!("[vector_overlay] id={} not found", id);
                }
            }
            viewer.unified_picking.remove_layer_bvh(*id);
            true
        }
        ViewerCmd::SetVectorOverlayVisible { id, visible } => {
            if let Some(ref mut terrain_viewer) = viewer.terrain_viewer {
                terrain_viewer.set_vector_overlay_visible(*id, *visible);
                println!("[vector_overlay] id={} visible={}", id, visible);
            }
            true
        }
        ViewerCmd::SetVectorOverlayOpacity { id, opacity } => {
            if let Some(ref mut terrain_viewer) = viewer.terrain_viewer {
                terrain_viewer.set_vector_overlay_opacity(*id, *opacity);
                println!("[vector_overlay] id={} opacity={:.2}", id, opacity);
            }
            true
        }
        ViewerCmd::ListVectorOverlays => {
            if let Some(ref terrain_viewer) = viewer.terrain_viewer {
                let ids = terrain_viewer.list_vector_overlays();
                if ids.is_empty() {
                    println!("[vector_overlay] No vector overlays loaded");
                } else {
                    println!("[vector_overlay] Loaded: {:?}", ids);
                }
            } else {
                println!("[vector_overlay] No terrain loaded");
            }
            true
        }
        ViewerCmd::SetVectorOverlaysEnabled { enabled } => {
            if let Some(ref mut terrain_viewer) = viewer.terrain_viewer {
                terrain_viewer.set_vector_overlays_enabled(*enabled);
                println!("[vector_overlay] enabled={}", enabled);
            }
            true
        }
        ViewerCmd::SetGlobalVectorOverlayOpacity { opacity } => {
            if let Some(ref mut terrain_viewer) = viewer.terrain_viewer {
                terrain_viewer.set_global_vector_overlay_opacity(*opacity);
                println!("[vector_overlay] global opacity={:.2}", opacity);
            }
            true
        }
        _ => false,
    }
}
