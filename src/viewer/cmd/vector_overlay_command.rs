use crate::viewer::viewer_enums::ViewerCmd;
use crate::viewer::Viewer;

pub(crate) fn handle_cmd(viewer: &mut Viewer, cmd: &ViewerCmd) -> bool {
    match cmd {
        ViewerCmd::AddVectorOverlay {
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
            if let Some(ref mut terrain_viewer) = viewer.terrain_viewer {
                use crate::viewer::terrain::vector_overlay::{
                    OverlayPrimitive, VectorOverlayLayer, VectorVertex,
                };

                let verts: Vec<VectorVertex> = vertices
                    .iter()
                    .map(|v| {
                        let feature_id = if v.len() > 7 { v[7] as u32 } else { 0 };
                        VectorVertex::with_feature_id(
                            v[0], v[1], v[2], v[3], v[4], v[5], v[6], feature_id,
                        )
                    })
                    .collect();
                let primitive =
                    OverlayPrimitive::from_str(primitive).unwrap_or(OverlayPrimitive::Triangles);

                let layer = VectorOverlayLayer {
                    name: name.clone(),
                    vertices: verts,
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

                let id = terrain_viewer.add_vector_overlay(layer);
                println!(
                    "[vector_overlay] Added '{}' with {} vertices (id={})",
                    name,
                    vertices.len(),
                    id
                );

                if primitive == OverlayPrimitive::Triangles && !indices.is_empty() {
                    use crate::accel::cpu_bvh::{build_bvh_cpu, BuildOptions, MeshCPU};
                    use crate::accel::types::{Aabb, BvhNode, Triangle};
                    use crate::picking::LayerBvhData;

                    let positions: Vec<[f32; 3]> =
                        vertices.iter().map(|v| [v[0], v[1], v[2]]).collect();
                    let triangles: Vec<[u32; 3]> = indices
                        .chunks(3)
                        .filter_map(|chunk| {
                            if chunk.len() == 3 {
                                Some([chunk[0], chunk[1], chunk[2]])
                            } else {
                                None
                            }
                        })
                        .collect();

                    let mesh = MeshCPU::new(positions.clone(), triangles.clone());
                    if let Ok(bvh) = build_bvh_cpu(&mesh, &BuildOptions::default()) {
                        let mut layer_data = LayerBvhData::new(id, name.clone());
                        layer_data.cpu_nodes = bvh
                            .nodes
                            .iter()
                            .map(|node| {
                                let kind = if node.is_leaf() { 1 } else { 0 };
                                BvhNode {
                                    aabb: Aabb::new(node.aabb_min, node.aabb_max),
                                    kind,
                                    left_idx: node.left,
                                    right_idx: node.right,
                                    parent_idx: 0,
                                }
                            })
                            .collect();
                        layer_data.cpu_triangles = bvh
                            .tri_indices
                            .iter()
                            .map(|&tri_idx| {
                                let idx = tri_idx as usize;
                                if idx < triangles.len() {
                                    let tri = triangles[idx];
                                    Triangle::new(
                                        positions[tri[0] as usize],
                                        positions[tri[1] as usize],
                                        positions[tri[2] as usize],
                                    )
                                } else {
                                    Triangle::new([0.0; 3], [0.0; 3], [0.0; 3])
                                }
                            })
                            .collect();
                        layer_data.cpu_feature_ids = bvh
                            .tri_indices
                            .iter()
                            .map(|&tri_idx| {
                                let idx = tri_idx as usize;
                                if idx < triangles.len() {
                                    let tri = triangles[idx];
                                    let vertex_index = tri[0] as usize;
                                    if vertex_index < vertices.len() {
                                        vertices[vertex_index][7] as u32
                                    } else {
                                        id
                                    }
                                } else {
                                    id
                                }
                            })
                            .collect();
                        viewer.unified_picking.register_layer_bvh(layer_data);
                        println!(
                            "[picking] Built BVH for layer {} ({} nodes)",
                            id,
                            bvh.nodes.len()
                        );
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
