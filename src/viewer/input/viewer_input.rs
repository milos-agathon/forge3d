// src/viewer/input/viewer_input.rs
// Viewer input handling methods
// Extracted from mod.rs as part of the viewer refactoring

use winit::event::*;
use winit::keyboard::{KeyCode, PhysicalKey};

use crate::picking::unproject_cursor;
use crate::picking::{PickEvent, PickEventType};
use crate::viewer::camera_controller::CameraMode;
use crate::viewer::event_loop::get_pick_events;
use crate::viewer::Viewer;

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
                world_pos: frame
                    .anchor
                    .to_world_from_render_f64(render_position.as_dvec3())
                    .to_array(),
                attributes: std::collections::HashMap::new(),
                terrain_info: None,
                hit_distance,
            },
        )
    }

    /// Execute the same frozen-frame screen-space pick used by mouse input.
    /// IPC acceptance tests call this route so their absolute f64 results are
    /// produced by the shipped picking implementation, not a test-side ray.
    pub(crate) fn pick_at_screen(
        &mut self,
        x: u32,
        y: u32,
        shift_held: bool,
        ctrl_held: bool,
    ) -> usize {
        let frame = self.current_frame_camera();
        let view_proj = frame.projection(self.config.width, self.config.height) * frame.view();
        let ray = unproject_cursor(
            x,
            y,
            self.config.width,
            self.config.height,
            view_proj.inverse().to_cols_array_2d(),
        );
        let event = PickEvent {
            event_type: PickEventType::Click,
            screen_pos: (x, y),
            shift_held,
            ctrl_held,
            results: Vec::new(),
        };

        let mut results = self.unified_picking.handle_pick_event(&ray, &event);
        for result in &mut results {
            result.world_pos = frame
                .anchor
                .to_world_from_render_f64(glam::DVec3::from(result.world_pos))
                .to_array();
        }
        if let Some(object) = self.pick_object_ray(&ray, frame) {
            results.push(object);
            results.sort_by(|left, right| left.hit_distance.total_cmp(&right.hit_distance));
        }
        if self.point_cloud_active() {
            if let Some((index, distance, world_pos)) = self
                .point_cloud
                .as_ref()
                .and_then(|point_cloud| point_cloud.pick_ray(&ray))
            {
                let mut attributes = std::collections::HashMap::new();
                attributes.insert("point_index".to_string(), index.to_string());
                results.push(crate::picking::RichPickResult {
                    feature_id: u32::try_from(index).unwrap_or(u32::MAX).saturating_add(1),
                    layer_name: "PointCloud".to_string(),
                    world_pos: world_pos.to_array(),
                    attributes,
                    terrain_info: None,
                    hit_distance: distance,
                });
                results.sort_by(|left, right| left.hit_distance.total_cmp(&right.hit_distance));
            }
        }
        if let Some(label_id) = self.label_manager.pick_at(x as f32, y as f32) {
            if let Some(label) = self.label_manager.get_label(label_id) {
                let mut attributes = std::collections::HashMap::new();
                attributes.insert("text".to_string(), label.text.clone());
                attributes.insert("type".to_string(), "label".to_string());
                results.insert(
                    0,
                    crate::picking::RichPickResult {
                        feature_id: label_id.0 as u32,
                        layer_name: "Labels".to_string(),
                        world_pos: label.world_pos.to_array(),
                        attributes,
                        terrain_info: None,
                        hit_distance: label.depth,
                    },
                );
                self.unified_picking
                    .selection_manager_mut()
                    .handle_pick(label_id.0 as u32, shift_held);
            }
        }

        let result_count = results.len();
        if let Some(first_result) = results.first() {
            self.selected_feature_id = first_result.feature_id;
            self.selected_layer_name = first_result.layer_name.clone();
            let mut result_event = event;
            result_event.results = results;
            if let Ok(mut queue) = get_pick_events().lock() {
                queue.push(result_event);
            }
        } else {
            self.selected_feature_id = 0;
            self.selected_layer_name.clear();
            if !shift_held {
                self.unified_picking
                    .selection_manager_mut()
                    .handle_pick(0, false);
            }
        }
        result_count
    }

    pub fn handle_input(&mut self, event: &WindowEvent) -> bool {
        match event {
            WindowEvent::KeyboardInput {
                event: key_event, ..
            } => {
                if let PhysicalKey::Code(keycode) = key_event.physical_key {
                    let pressed = key_event.state == ElementState::Pressed;

                    // Track shift
                    if matches!(keycode, KeyCode::ShiftLeft | KeyCode::ShiftRight) {
                        self.shift_pressed = pressed;
                    }

                    // Track WASD, Q, E for FPS mode
                    if pressed {
                        self.keys_pressed.insert(keycode);
                    } else {
                        self.keys_pressed.remove(&keycode);
                    }

                    // Toggle camera mode with Tab
                    if pressed && keycode == KeyCode::Tab {
                        let new_mode = match self.camera.mode() {
                            CameraMode::Orbit => CameraMode::Fps,
                            CameraMode::Fps => CameraMode::Orbit,
                        };
                        self.camera.set_mode(new_mode);
                        println!("Camera mode: {:?}", new_mode);
                        return true;
                    }
                }

                true
            }
            WindowEvent::MouseInput { state, button, .. } => {
                if *button == MouseButton::Left {
                    let pressed = *state == ElementState::Pressed;
                    self.camera.mouse_pressed = pressed;

                    // On release (click), perform picking if config enables it
                    if !pressed {
                        if let Some((x, y)) = self.camera.last_mouse_pos {
                            let _ =
                                self.pick_at_screen(x as u32, y as u32, self.shift_pressed, false);
                        }
                    }
                }
                true
            }
            WindowEvent::CursorMoved { position, .. } => {
                let new_x = position.x as f32;
                let new_y = position.y as f32;

                // Check what's active
                let pc_active = self.point_cloud_active();
                let terrain_active = self
                    .terrain_viewer
                    .as_ref()
                    .is_some_and(|tv| tv.has_terrain());

                // If mouse is pressed, orbit the appropriate camera
                if self.camera.mouse_pressed {
                    if let Some((last_x, last_y)) = self.camera.last_mouse_pos {
                        let dx = new_x - last_x;
                        let dy = new_y - last_y;

                        if terrain_active {
                            let anchor = self.prospective_frame_camera().anchor;
                            if let Some(ref mut tv) = self.terrain_viewer {
                                if let Err(error) = tv.handle_mouse_drag(&anchor, dx, dy) {
                                    eprintln!("[viewer] terrain orbit rejected: {error}");
                                }
                            }
                        } else if pc_active {
                            let anchor = self.prospective_frame_camera().anchor;
                            if let Some(ref mut pc) = self.point_cloud {
                                if let Err(error) = pc.handle_mouse_drag(&anchor, dx, dy) {
                                    eprintln!("[viewer] point-cloud orbit rejected: {error}");
                                }
                            }
                        }
                    }
                }

                if !terrain_active && !pc_active {
                    let anchor = self.prospective_frame_camera().anchor;
                    if let Err(error) = self.camera.try_handle_mouse_move(&anchor, new_x, new_y) {
                        eprintln!("[viewer] camera orbit rejected: {error}");
                    }
                } else {
                    self.camera.last_mouse_pos = Some((new_x, new_y));
                }
                true
            }
            WindowEvent::MouseWheel { delta, .. } => {
                let scroll = match delta {
                    MouseScrollDelta::LineDelta(_x, y) => *y * 3.0,
                    MouseScrollDelta::PixelDelta(pos) => pos.y as f32 * 0.02,
                };

                // Check what's active
                let terrain_active = self
                    .terrain_viewer
                    .as_ref()
                    .map_or(false, |tv| tv.has_terrain());
                let pc_active = self.point_cloud_active();

                if terrain_active {
                    let anchor = self.prospective_frame_camera().anchor;
                    if let Some(ref mut tv) = self.terrain_viewer {
                        if let Err(error) = tv.handle_scroll(&anchor, scroll) {
                            eprintln!("[viewer] terrain wheel rejected: {error}");
                        }
                    }
                } else if pc_active {
                    let anchor = self.prospective_frame_camera().anchor;
                    if let Some(ref mut pc) = self.point_cloud {
                        if let Err(error) = pc.handle_scroll(&anchor, scroll) {
                            eprintln!("[viewer] point-cloud wheel rejected: {error}");
                        }
                    }
                } else {
                    let anchor = self.prospective_frame_camera().anchor;
                    if let Err(error) = self.camera.try_handle_mouse_scroll(&anchor, scroll) {
                        eprintln!("[viewer] camera wheel rejected: {error}");
                    }
                }
                true
            }
            _ => false,
        }
    }

    pub fn update(&mut self, dt: f32) {
        // Update FPS camera movement
        let mut forward = 0.0;
        let mut right = 0.0;
        let mut up = 0.0;

        let speed_mult = if self.shift_pressed { 2.0 } else { 1.0 };

        if self.keys_pressed.contains(&KeyCode::KeyW)
            || self.keys_pressed.contains(&KeyCode::ArrowUp)
        {
            forward += speed_mult;
        }
        if self.keys_pressed.contains(&KeyCode::KeyS)
            || self.keys_pressed.contains(&KeyCode::ArrowDown)
        {
            forward -= speed_mult;
        }
        if self.keys_pressed.contains(&KeyCode::KeyD)
            || self.keys_pressed.contains(&KeyCode::ArrowRight)
        {
            right += speed_mult;
        }
        if self.keys_pressed.contains(&KeyCode::KeyA)
            || self.keys_pressed.contains(&KeyCode::ArrowLeft)
        {
            right -= speed_mult;
        }
        if self.keys_pressed.contains(&KeyCode::KeyE) {
            up += speed_mult;
        }
        if self.keys_pressed.contains(&KeyCode::KeyQ) {
            up -= speed_mult;
        }

        // If terrain viewer is active, route input to terrain camera
        // Otherwise, if point cloud is active, route to point cloud camera
        let terrain_active = self
            .terrain_viewer
            .as_ref()
            .map_or(false, |tv| tv.has_terrain());
        let pc_active = self.point_cloud_active();

        if terrain_active {
            let anchor = self.prospective_frame_camera().anchor;
            if let Some(ref mut tv) = self.terrain_viewer {
                if let Err(error) = tv.handle_keys(&anchor, forward, right, up) {
                    eprintln!("[viewer] terrain key camera rejected: {error}");
                }
                #[cfg(feature = "enable-gpu-instancing")]
                tv.tick_scatter_time(dt);
            }
        } else if pc_active {
            let anchor = self.prospective_frame_camera().anchor;
            if let Some(ref mut pc) = self.point_cloud {
                if let Err(error) = pc.handle_keys(&anchor, forward, right, up) {
                    eprintln!("[viewer] point-cloud key camera rejected: {error}");
                }
            }
        } else {
            let anchor = self.prospective_frame_camera().anchor;
            if let Err(error) = self.camera.try_update_fps(&anchor, dt, forward, right, up) {
                eprintln!("[viewer] FPS camera rejected: {error}");
            }
        }
    }
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
