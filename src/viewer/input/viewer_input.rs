// src/viewer/input/viewer_input.rs
// Viewer input handling methods
// Extracted from mod.rs as part of the viewer refactoring

use glam::Mat4;
use winit::event::*;
use winit::keyboard::{KeyCode, PhysicalKey};

use crate::viewer::camera_controller::CameraMode;
use crate::viewer::Viewer;

impl Viewer {
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
                    self.camera.mouse_pressed = *state == ElementState::Pressed;
                }
                true
            }
            WindowEvent::CursorMoved { position, .. } => {
                let new_x = position.x as f32;
                let new_y = position.y as f32;
                
                // If terrain viewer is active and mouse is pressed, orbit the terrain camera
                if self.camera.mouse_pressed {
                    if let Some(ref mut tv) = self.terrain_viewer {
                        if tv.has_terrain() {
                            if let Some((last_x, last_y)) = self.camera.last_mouse_pos {
                                let dx = new_x - last_x;
                                let dy = new_y - last_y;
                                tv.handle_mouse_drag(dx, dy);
                            }
                        }
                    }
                }
                
                self.camera
                    .handle_mouse_move(new_x, new_y);
                true
            }
            WindowEvent::MouseWheel { delta, .. } => {
                let scroll = match delta {
                    MouseScrollDelta::LineDelta(x, y) => {
                        println!("[scroll] LineDelta x={} y={}", x, y);
                        *y * 3.0
                    }
                    MouseScrollDelta::PixelDelta(pos) => {
                        println!("[scroll] PixelDelta x={} y={}", pos.x, pos.y);
                        pos.y as f32 * 0.02
                    }
                };
                println!("[scroll] final={}", scroll);
                
                // If terrain viewer is active, zoom the terrain camera
                if let Some(ref mut tv) = self.terrain_viewer {
                    if tv.has_terrain() {
                        tv.handle_scroll(scroll);
                    }
                } else {
                    self.camera.handle_mouse_scroll(scroll);
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

        if self.keys_pressed.contains(&KeyCode::KeyW) || self.keys_pressed.contains(&KeyCode::ArrowUp) {
            forward += speed_mult;
        }
        if self.keys_pressed.contains(&KeyCode::KeyS) || self.keys_pressed.contains(&KeyCode::ArrowDown) {
            forward -= speed_mult;
        }
        if self.keys_pressed.contains(&KeyCode::KeyD) || self.keys_pressed.contains(&KeyCode::ArrowRight) {
            right += speed_mult;
        }
        if self.keys_pressed.contains(&KeyCode::KeyA) || self.keys_pressed.contains(&KeyCode::ArrowLeft) {
            right -= speed_mult;
        }
        if self.keys_pressed.contains(&KeyCode::KeyE) {
            up += speed_mult;
        }
        if self.keys_pressed.contains(&KeyCode::KeyQ) {
            up -= speed_mult;
        }

        // If terrain viewer is active, route input to terrain camera
        let terrain_active = self.terrain_viewer.as_ref().map_or(false, |tv| tv.has_terrain());
        if terrain_active {
            if let Some(ref mut tv) = self.terrain_viewer {
                tv.handle_keys(forward, right, up);
            }
        } else {
            self.camera.update_fps(dt, forward, right, up);
        }

        // Update GI camera params
        if let Some(ref mut gi) = self.gi {
            let aspect = self.config.width as f32 / self.config.height as f32;
            let fov = self.view_config.fov_deg.to_radians();
            let proj =
                Mat4::perspective_rh(fov, aspect, self.view_config.znear, self.view_config.zfar);
            let view = self.camera.view_matrix();
            let inv_proj = proj.inverse();

            fn to_arr4(m: Mat4) -> [[f32; 4]; 4] {
                let c = m.to_cols_array();
                [
                    [c[0], c[1], c[2], c[3]],
                    [c[4], c[5], c[6], c[7]],
                    [c[8], c[9], c[10], c[11]],
                    [c[12], c[13], c[14], c[15]],
                ]
            }
            let eye = self.camera.eye();
            // Apply object transform to view matrix for consistent GI
            let model_view = view * self.object_transform;
            let inv_model_view = model_view.inverse();
            let cam = crate::core::screen_space_effects::CameraParams {
                view_matrix: to_arr4(model_view),
                inv_view_matrix: to_arr4(inv_model_view),
                proj_matrix: to_arr4(proj),
                inv_proj_matrix: to_arr4(inv_proj),
                camera_pos: [eye.x, eye.y, eye.z],
                _pad: 0.0,
            };
            gi.update_camera(&self.queue, &cam);
        }
    }
}
