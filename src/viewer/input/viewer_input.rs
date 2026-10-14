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

impl Viewer {
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
            results: self.collect_scene_picks(&ray, frame, (x, y)),
        };

        self.unified_picking.apply_assembled_pick_event(&event);
        let result_count = event.results.len();
        if let Some(first_result) = event.results.first() {
            self.selected_feature_id = first_result.feature_id;
            self.selected_layer_name = first_result.layer_name.clone();
            if let Ok(mut queue) = get_pick_events().lock() {
                queue.push(event);
            }
        } else {
            self.selected_feature_id = 0;
            self.selected_layer_name.clear();
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
