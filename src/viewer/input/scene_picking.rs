//! Collect the complete frozen-frame scene hit set before selection runs.
use crate::camera::Anchor;
use crate::labels::LabelData;
use crate::picking::{Ray, RichPickResult};
use crate::viewer::viewer_types::FrameCamera;
use crate::viewer::Viewer;
use glam::DVec3;
use std::collections::HashMap;

fn ray_triangle_distance(
    origin: glam::Vec3,
    direction: glam::Vec3,
    a: glam::Vec3,
    b: glam::Vec3,
    c: glam::Vec3,
) -> Option<f32> {
    let edge1 = b - a;
    let edge2 = c - a;
    let p = direction.cross(edge2);
    let determinant = edge1.dot(p);
    if determinant.abs() <= 1.0e-8 {
        return None;
    }
    let inverse = determinant.recip();
    let offset = origin - a;
    let u = offset.dot(p) * inverse;
    if !(0.0..=1.0).contains(&u) {
        return None;
    }
    let q = offset.cross(edge1);
    let v = direction.dot(q) * inverse;
    if v < 0.0 || u + v > 1.0 {
        return None;
    }
    let distance = edge2.dot(q) * inverse;
    (distance >= 0.0).then_some(distance)
}

impl Viewer {
    fn pick_object_ray(
        &self,
        ray: &crate::picking::Ray,
        frame: crate::viewer::viewer_types::FrameCamera,
    ) -> Option<crate::picking::RichPickResult> {
        let origin = glam::Vec3::from(ray.origin);
        let direction = glam::Vec3::from(ray.direction).normalize_or_zero();
        if direction == glam::Vec3::ZERO {
            return None;
        }
        let model = self.anchored_object_model(frame);
        let mut best: Option<(f32, glam::Vec3)> = None;
        for triangle in self.object_source_indices.chunks_exact(3) {
            let [Some(a), Some(b), Some(c)] = [
                self.object_source_positions.get(triangle[0] as usize),
                self.object_source_positions.get(triangle[1] as usize),
                self.object_source_positions.get(triangle[2] as usize),
            ] else {
                continue;
            };
            let a = model.transform_point3(glam::Vec3::from(*a));
            let b = model.transform_point3(glam::Vec3::from(*b));
            let c = model.transform_point3(glam::Vec3::from(*c));
            let Some(distance) = ray_triangle_distance(origin, direction, a, b, c) else {
                continue;
            };
            if best.is_some_and(|(current, _)| distance >= current) {
                continue;
            }
            best = Some((distance, origin + direction * distance));
        }
        best.map(
            |(hit_distance, render_position)| crate::picking::RichPickResult {
                feature_id: 1,
                layer_name: "Object".to_string(),
                world_pos: render_position.as_dvec3().to_array(),
                attributes: std::collections::HashMap::new(),
                terrain_info: None,
                hit_distance,
            },
        )
    }

    /// Side-effect-free collection: BVH/object hits are render-relative;
    /// points and labels retain their absolute f64 source coordinates.
    pub(super) fn collect_scene_picks(
        &self,
        ray: &Ray,
        frame: FrameCamera,
        screen_pos: (u32, u32),
    ) -> Vec<RichPickResult> {
        let bvh_hits = self.unified_picking.ray_cast(ray);
        let object_hit = self.pick_object_ray(ray, frame);
        let point_hit = self
            .point_cloud
            .as_ref()
            .filter(|_| self.point_cloud_active())
            .and_then(|cloud| cloud.pick_ray(ray));
        let label_hit = self
            .label_manager
            .pick_at(screen_pos.0 as f32, screen_pos.1 as f32)
            .and_then(|id| self.label_manager.get_label(id));
        assemble_scene_picks(&frame.anchor, bvh_hits, object_hit, point_hit, label_hit)
    }
}

/// Label depth is normalized screen depth, not a ray distance. Keep the
/// existing label priority, then order geometry by distance along the ray.
fn assemble_scene_picks(
    anchor: &Anchor,
    mut render_hits: Vec<RichPickResult>,
    object_hit: Option<RichPickResult>,
    point_hit: Option<(usize, f32, DVec3)>,
    label_hit: Option<&LabelData>,
) -> Vec<RichPickResult> {
    render_hits.extend(object_hit);
    for hit in &mut render_hits {
        hit.world_pos = anchor
            .to_world_from_render_f64(DVec3::from(hit.world_pos))
            .to_array();
    }
    if let Some((index, distance, world_pos)) = point_hit {
        let mut attributes = HashMap::new();
        attributes.insert("point_index".to_string(), index.to_string());
        render_hits.push(RichPickResult {
            feature_id: u32::try_from(index).unwrap_or(u32::MAX).saturating_add(1),
            layer_name: "PointCloud".to_string(),
            world_pos: world_pos.to_array(),
            attributes,
            terrain_info: None,
            hit_distance: distance,
        });
    }
    render_hits.sort_by(|left, right| left.hit_distance.total_cmp(&right.hit_distance));
    if let Some(label) = label_hit {
        let mut attributes = HashMap::new();
        attributes.insert("text".to_string(), label.text.clone());
        attributes.insert("type".to_string(), "label".to_string());
        render_hits.insert(
            0,
            RichPickResult {
                feature_id: label.id.0 as u32,
                layer_name: "Labels".to_string(),
                world_pos: label.world_pos.to_array(),
                attributes,
                terrain_info: None,
                hit_distance: label.depth,
            },
        );
    }
    render_hits
}

#[cfg(test)]
mod object_pick_tests {
    use super::*;

    #[test]
    fn transformed_triangle_is_intersected_at_its_visible_model_position_once() {
        let model = glam::Mat4::from_scale_rotation_translation(
            glam::Vec3::new(2.0, 3.0, 4.0),
            glam::Quat::IDENTITY,
            glam::Vec3::new(10.0, 20.0, 30.0),
        );
        let a = model.transform_point3(glam::Vec3::new(-1.0, -1.0, 0.0));
        let b = model.transform_point3(glam::Vec3::new(1.0, -1.0, 0.0));
        let c = model.transform_point3(glam::Vec3::new(0.0, 1.0, 0.0));
        let distance =
            ray_triangle_distance(glam::Vec3::new(10.0, 20.0, 40.0), -glam::Vec3::Z, a, b, c)
                .unwrap();
        assert_eq!(distance, 10.0);
    }
}

#[cfg(test)]
mod scene_pick_tests {
    use super::*;
    use crate::labels::{LabelId, LabelManager, LabelStyle};
    use crate::picking::TerrainHitInfo;

    fn hit(id: u32, layer: &str, distance: f32, position: DVec3) -> RichPickResult {
        RichPickResult {
            feature_id: id,
            layer_name: layer.into(),
            world_pos: position.to_array(),
            attributes: HashMap::from([("source".into(), layer.into())]),
            terrain_info: None,
            hit_distance: distance,
        }
    }

    #[test]
    fn mixed_sources_keep_identity_attributes_world_precision_and_label_priority() {
        let origin = DVec3::new(6_378_137.125, 1_000_000.25, -2_000_000.5);
        let mut anchor = Anchor::new();
        assert!(anchor.rebase_if_needed(origin));
        let mut labels = LabelManager::new(640, 400);
        let label_position = origin + DVec3::new(0.001, 0.002, 0.003);
        let label_id = labels.add_label_with_id(
            Some(LabelId(41)),
            "Label".into(),
            label_position,
            LabelStyle::default(),
        );
        labels.get_label_mut(label_id).unwrap().depth = 0.99;
        let label = labels.get_label(label_id).unwrap();
        let mut vector = hit(17, "Vector", 0.8, DVec3::new(0.0, 0.0, -0.8));
        vector.terrain_info = Some(TerrainHitInfo {
            elevation: 12.0,
            slope: 3.0,
            aspect: 4.0,
            normal: [0.0, 1.0, 0.0],
        });
        let point_position = origin + DVec3::new(0.0001, 0.0002, -0.2);
        let results = assemble_scene_picks(
            &anchor,
            vec![vector],
            Some(hit(1, "Object", 0.4, DVec3::new(0.0, 0.0, -0.4))),
            Some((6, 0.2, point_position)),
            Some(label),
        );
        assert_eq!(
            results
                .iter()
                .map(|h| (h.layer_name.as_str(), h.feature_id))
                .collect::<Vec<_>>(),
            vec![
                ("Labels", 41),
                ("PointCloud", 7),
                ("Object", 1),
                ("Vector", 17)
            ]
        );
        assert_eq!(results[0].world_pos, label_position.to_array());
        assert_eq!(results[0].attributes["text"], "Label");
        assert_eq!(results[0].attributes["type"], "label");
        assert_eq!(results[1].world_pos, point_position.to_array());
        assert_eq!(results[1].attributes["point_index"], "6");
        assert_eq!(
            results[2].world_pos,
            (origin + DVec3::new(0.0, 0.0, -0.4)).to_array()
        );
        assert_eq!(
            results[3].world_pos,
            (origin + DVec3::new(0.0, 0.0, -0.8)).to_array()
        );
        assert_eq!(results[3].attributes["source"], "Vector");
        assert_eq!(results[3].terrain_info.unwrap().elevation, 12.0);
    }

    #[test]
    fn rebasing_changes_render_positions_without_changing_world_hits_or_identity() {
        let origin = DVec3::new(6_378_137.125, 1_000_000.25, -2_000_000.5);
        let world = origin + DVec3::new(0.125, 0.25, -8.0);
        let mut anchor = Anchor::new();
        assert!(anchor.rebase_if_needed(origin));
        let first = assemble_scene_picks(
            &anchor,
            vec![hit(17, "Vector", 8.0, world - anchor.origin())],
            Some(hit(1, "Object", 9.0, world - anchor.origin())),
            Some((6, 10.0, world)),
            None,
        );
        assert!(anchor.rebase_if_needed(origin + DVec3::new(2048.0, 0.0, 0.0)));
        let second = assemble_scene_picks(
            &anchor,
            vec![hit(17, "Vector", 8.0, world - anchor.origin())],
            Some(hit(1, "Object", 9.0, world - anchor.origin())),
            Some((6, 10.0, world)),
            None,
        );
        assert_eq!(
            serde_json::to_value(first).unwrap(),
            serde_json::to_value(second).unwrap()
        );
    }

    #[test]
    fn equal_depth_keeps_source_order_and_empty_scene_has_no_hits() {
        let results = assemble_scene_picks(
            &Anchor::new(),
            vec![hit(17, "Vector", 8.0, DVec3::ZERO)],
            Some(hit(1, "Object", 8.0, DVec3::ZERO)),
            Some((6, 8.0, DVec3::ZERO)),
            None,
        );
        assert_eq!(
            results.iter().map(|h| h.feature_id).collect::<Vec<_>>(),
            vec![17, 1, 7]
        );
        assert!(assemble_scene_picks(&Anchor::new(), vec![], None, None, None).is_empty());
    }

    // The shipped viewer is a Windows app in the required M-06 hardware lane.
    // Exercise its real screen-pick route with GPU-backed source state, without
    // claiming a rendered frame. Other platforms retain the CPU policy checks.
    #[cfg(target_os = "windows")]
    #[test]
    fn shipped_screen_pick_assembles_all_sources_before_selection_and_after_rebase() {
        use crate::accel::types::{BvhNode, Triangle};
        use crate::geo::units::SceneCoord;
        use crate::picking::{LayerBvhData, PickEvent};
        use crate::viewer::event_loop::get_pick_events;
        use crate::viewer::pointcloud::{ColorMode, PointCloudState};
        use crate::viewer::viewer_config::ViewerConfig;
        use crate::viewer::viewer_types::ActiveCameraKind;
        use las::Write;
        use std::sync::Arc;
        use winit::platform::windows::EventLoopBuilderExtWindows;

        fn pick(viewer: &mut Viewer, shift: bool, ctrl: bool) -> PickEvent {
            get_pick_events().lock().unwrap().clear();
            let count = viewer.pick_at_screen(320, 200, shift, ctrl);
            let event = get_pick_events().lock().unwrap().pop().unwrap();
            assert_eq!(event.results.len(), count);
            let first = &event.results[0];
            assert_eq!(viewer.selected_feature_id, first.feature_id);
            assert_eq!(viewer.selected_layer_name, first.layer_name);
            assert_eq!((event.shift_held, event.ctrl_held), (shift, ctrl));
            event
        }

        fn prepare(viewer: &mut Viewer, frame: FrameCamera, vector_world: DVec3) {
            viewer.camera_anchor = frame.anchor;
            viewer.frame_camera = Some(frame);
            let vertices = [
                vector_world + DVec3::new(-1.0, -1.0, 0.0),
                vector_world + DVec3::new(1.0, -1.0, 0.0),
                vector_world + DVec3::new(0.0, 1.0, 0.0),
            ]
            .map(|world| {
                frame
                    .anchor
                    .to_render_f32(SceneCoord::scene(world))
                    .to_array()
            });
            let triangle = Triangle::new(vertices[0], vertices[1], vertices[2]);
            let mut layer = LayerBvhData::new(55, "Vector".into());
            layer.cpu_nodes.push(BvhNode::leaf(triangle.aabb(), 0, 1));
            layer.cpu_triangles.push(triangle);
            layer.cpu_feature_ids.push(17);
            viewer.unified_picking.register_layer_bvh(layer);
            viewer.point_cloud.as_mut().unwrap().prepare_visible(
                &viewer.queue,
                &frame.anchor,
                frame.view_projection(viewer.config.width, viewer.config.height),
            );
            viewer.label_manager.update_with_camera_anchored(
                frame.view_projection(viewer.config.width, viewer.config.height),
                Some(frame.render_eye()),
                None,
                &frame.anchor,
            );
        }

        let event_loop = winit::event_loop::EventLoopBuilder::new()
            .with_any_thread(true)
            .build()
            .unwrap();
        let window = Arc::new(
            winit::window::WindowBuilder::new()
                .with_visible(false)
                .with_inner_size(winit::dpi::PhysicalSize::new(641, 401))
                .build(&event_loop)
                .unwrap(),
        );
        let mut viewer = pollster::block_on(Viewer::new(
            window,
            ViewerConfig {
                // The center raster pixel's center lies on the camera axis.
                width: 641,
                height: 401,
                ..ViewerConfig::default()
            },
        ))
        .unwrap();
        eprintln!(
            "scene pick adapter: {} / {} / {}",
            viewer.adapter_name, viewer.adapter_backend, viewer.adapter_device_type
        );

        let world = DVec3::new(6_378_137.125, 1_000_000.25, -2_000_000.5);
        let mut anchor = Anchor::new();
        assert!(anchor.rebase_if_needed(world));
        let mut frame = FrameCamera {
            kind: ActiveCameraKind::General,
            anchor,
            eye_world: world + DVec3::new(0.0, 0.0, 10.0),
            target_world: world,
            up: glam::Vec3::Y,
            fov_deg: 45.0,
            near: 0.1,
            far: viewer.unified_picking.config().max_ray_distance,
        };
        viewer.object_source_positions = vec![[-1.0, -1.0, 0.0], [1.0, -1.0, 0.0], [0.0, 1.0, 0.0]];
        viewer.object_source_indices = vec![0, 1, 2];
        viewer.object_translation = world + DVec3::new(0.0, 0.0, 2.0);
        viewer.object_scale = glam::Vec3::ONE;
        viewer.object_rotation = glam::Quat::IDENTITY;
        viewer.unified_picking.set_bvh_enabled(true);

        // Load a real LAS fixture through the shipped loader. Only its third
        // point lies on the center ray; retain the loader's exact f64 position.
        let path =
            std::env::temp_dir().join(format!("forge3d-scene-pick-{}.las", std::process::id()));
        {
            let mut builder = las::Builder::default();
            builder.transforms.x.offset = world.x;
            builder.transforms.y.offset = world.z;
            builder.transforms.z.offset = world.y;
            let mut writer = las::Writer::from_path(&path, builder.into_header().unwrap()).unwrap();
            for offset in [
                DVec3::new(-1.0, 0.0, 4.0),
                DVec3::new(1.0, 0.0, 4.0),
                DVec3::new(0.0, 0.0, 4.0),
            ] {
                let position = world + offset;
                writer
                    .write(las::Point {
                        x: position.x,
                        y: position.z,
                        z: position.y,
                        ..las::Point::default()
                    })
                    .unwrap();
            }
        }
        let mut cloud = PointCloudState::new(
            &viewer.device,
            wgpu::TextureFormat::Bgra8UnormSrgb,
            wgpu::TextureFormat::Depth32Float,
        )
        .unwrap();
        cloud
            .load_from_file(
                &viewer.device,
                &viewer.queue,
                &anchor,
                false,
                path.to_str().unwrap(),
                3,
                ColorMode::Elevation,
            )
            .unwrap();
        std::fs::remove_file(path).unwrap();
        let point_world = cloud.source_points[2].position;
        viewer.point_cloud = Some(cloud);
        prepare(&mut viewer, frame, world);

        let geometry = pick(&mut viewer, false, false);
        assert_eq!(
            geometry
                .results
                .iter()
                .map(|h| (h.layer_name.as_str(), h.feature_id))
                .collect::<Vec<_>>(),
            vec![("PointCloud", 3), ("Object", 1), ("Vector", 17)]
        );
        assert_eq!(geometry.results[0].world_pos, point_world.to_array());
        assert_eq!(
            viewer
                .unified_picking
                .selection_manager()
                .primary()
                .unwrap()
                .feature_ids(),
            vec![3]
        );

        viewer.point_cloud.as_mut().unwrap().visible = false;
        assert_eq!(
            pick(&mut viewer, false, true).results[0].layer_name,
            "Object"
        );
        assert_eq!(
            viewer
                .unified_picking
                .selection_manager()
                .primary()
                .unwrap()
                .feature_ids(),
            vec![1]
        );
        viewer.point_cloud.as_mut().unwrap().visible = true;

        let label_world = world + DVec3::new(0.0, 0.0, 1.0);
        viewer.label_manager.add_label_with_id(
            Some(LabelId(41)),
            "Label".into(),
            label_world,
            LabelStyle::default(),
        );
        prepare(&mut viewer, frame, world);
        for multi in [false, true] {
            for shift in [false, true] {
                for ctrl in [false, true] {
                    let selection = viewer.unified_picking.selection_manager_mut();
                    selection.set_multi_select(multi);
                    selection.handle_pick(99, false);
                    let event = pick(&mut viewer, shift, ctrl);
                    assert_eq!(event.results.len(), 4);
                    assert_eq!(event.results[0].layer_name, "Labels");
                    assert_eq!(event.results[0].world_pos, label_world.to_array());
                    let set = viewer
                        .unified_picking
                        .selection_manager()
                        .primary()
                        .unwrap();
                    assert!(set.contains(41));
                    assert_eq!(set.contains(99), multi && shift);
                    assert!(!set.contains(17) && !set.contains(1) && !set.contains(3));
                    pick(&mut viewer, shift, ctrl);
                    assert_eq!(
                        viewer
                            .unified_picking
                            .selection_manager()
                            .primary()
                            .unwrap()
                            .contains(41),
                        !(multi && shift)
                    );
                }
            }
        }
        // Follow the shipped frame publisher: camera motion triggers a rebase
        // onto the eye. Moving along the center ray preserves world hit points.
        frame.eye_world = world + DVec3::new(0.0, 0.0, 2048.0);
        assert!(frame.anchor.rebase_if_needed(frame.eye_world));
        prepare(&mut viewer, frame, world);
        let rebased = pick(&mut viewer, false, false);
        assert_eq!(
            rebased
                .results
                .iter()
                .map(|h| h.feature_id)
                .collect::<Vec<_>>(),
            vec![41, 3, 1, 17]
        );
        assert_eq!(rebased.results[0].world_pos, label_world.to_array());
        assert_eq!(rebased.results[1].world_pos, point_world.to_array());
        // Reuse the required M-06 object-pick rebase tolerance (0.5 mm).
        for (before, after) in geometry.results.iter().zip(&rebased.results[1..]) {
            assert!(
                (DVec3::from(before.world_pos) - DVec3::from(after.world_pos)).length() <= 0.0005,
                "{}: {:?} -> {:?}",
                before.layer_name,
                before.world_pos,
                after.world_pos,
            );
        }
        assert_eq!(
            viewer
                .unified_picking
                .selection_manager()
                .primary()
                .unwrap()
                .feature_ids(),
            vec![41]
        );
        viewer.pick_at_screen(0, 0, true, true);
        assert_eq!(viewer.selected_feature_id, 0);
        assert_eq!(
            viewer
                .unified_picking
                .selection_manager()
                .primary()
                .unwrap()
                .feature_ids(),
            vec![41]
        );
        viewer.pick_at_screen(0, 0, false, true);
        assert!(viewer
            .unified_picking
            .selection_manager()
            .primary()
            .unwrap()
            .is_empty());
    }
}
