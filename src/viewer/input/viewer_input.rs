// src/viewer/input/viewer_input.rs
// Viewer input handling methods
// Extracted from mod.rs as part of the viewer refactoring

use glam::Mat4;
use winit::event::*;
use winit::keyboard::{KeyCode, PhysicalKey};

use crate::viewer::camera_controller::CameraMode;
use crate::viewer::Viewer;
use crate::picking::{PickEvent, PickEventType};
use crate::picking::unproject_cursor;
use crate::viewer::event_loop::get_pick_events;

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
                    let pressed = *state == ElementState::Pressed;
                    self.camera.mouse_pressed = pressed;
                    
                    // On release (click), perform picking if config enables it
                    if !pressed {
                        if let Some((x, y)) = self.camera.last_mouse_pos {
                            // Only pick if lasso is not active (or handle picking separately)
                            // Point click only; drag selection is handled by the lasso path.
                            
                            // Calculate matrices for unprojection
                            let aspect = self.config.width as f32 / self.config.height as f32;
                            let fov = self.view_config.fov_deg.to_radians();
                            let proj = Mat4::perspective_rh(fov, aspect, self.view_config.znear, self.view_config.zfar);
                            let view = self.camera.view_matrix();
                            // Apply object transform to view matrix for consistent space
                            let model_view = view * self.object_transform;
                            let view_proj = proj * model_view;
                            let inv_view_proj = view_proj.inverse();
                            
                            // Unproject cursor to ray
                            let ray = unproject_cursor(
                                x as u32, y as u32, 
                                self.config.width, self.config.height, 
                                inv_view_proj.to_cols_array_2d()
                            );
                            
                            // Create event
                            let event = PickEvent {
                                event_type: PickEventType::Click,
                                screen_pos: (x as u32, y as u32),
                                shift_held: self.shift_pressed,
                                ctrl_held: false,
                                results: Vec::new(),
                            };
                            
                            // Perform picking query
                            let mut results = self.unified_picking.handle_pick_event(&ray, &event);
                            
                            // Also check for label hits (screen-space)
                            if let Some(label_id) = self.label_manager.pick_at(x as f32, y as f32) {
                                if let Some(label) = self.label_manager.get_label(label_id) {
                                    // Create a result for the label
                                    let mut attrs = std::collections::HashMap::new();
                                    attrs.insert("text".to_string(), label.text.clone());
                                    attrs.insert("type".to_string(), "label".to_string());
                                    
                                    let label_result = crate::picking::RichPickResult {
                                        feature_id: label_id.0 as u32,
                                        layer_name: "Labels".to_string(),
                                        world_pos: label.world_pos.to_array(),
                                        attributes: attrs,
                                        terrain_info: None,
                                        hit_distance: label.depth, // Use screen depth
                                    };
                                    
                                    // Insert at beginning if closer? 
                                    // Labels are usually on top, so let's put it first
                                    results.insert(0, label_result);

                                    // Update unified selection manager manually for label hit
                                    // since handle_pick_event didn't see it (labels are screen-space)
                                    if event.event_type == PickEventType::Click {
                                         self.unified_picking.selection_manager_mut().handle_pick(label_id.0 as u32, event.shift_held);
                                    }
                                }
                            }
                            
                            // If we hit something, queue the event for IPC and update selection
                            if !results.is_empty() {
                                let mut result_event = event.clone();
                                result_event.results = results.clone();
                                
                                // Update selection state for highlighting
                                if let Some(first_result) = results.first() {
                                    self.selected_feature_id = first_result.feature_id;
                                    self.selected_layer_name = first_result.layer_name.clone();
                                    println!("[picking] Selected feature {} in layer {}", 
                                        first_result.feature_id, &first_result.layer_name);
                                }
                                
                                if let Ok(mut q) = get_pick_events().lock() {
                                    q.push(result_event.clone());
                                }
                                println!("[picking] Picked {} features", result_event.results.len());
                            } else {
                                // Clear selection on miss
                                self.selected_feature_id = 0;
                                self.selected_layer_name.clear();
                                
                                // Also clear unified selection if not holding shift (add mode)
                                if event.event_type == PickEventType::Click && !event.shift_held {
                                    // handle_pick(0, false) clears the primary set
                                    self.unified_picking.selection_manager_mut().handle_pick(0, false);
                                }
                            }
                        }
                    }
                }
                true
            }
            WindowEvent::CursorMoved { position, .. } => {
                let new_x = position.x as f32;
                let new_y = position.y as f32;
                
                // Check what's active
                let pc_count = self.point_cloud.as_ref().map_or(0, |pc| pc.point_count);
                
                // If mouse is pressed, orbit the appropriate camera
                if self.camera.mouse_pressed {
                    if let Some((last_x, last_y)) = self.camera.last_mouse_pos {
                        let dx = new_x - last_x;
                        let dy = new_y - last_y;
                        
                        // Check if terrain viewer is active
                        let terrain_active = self.terrain_viewer.as_ref()
                            .map_or(false, |tv| tv.has_terrain());
                        
                        if terrain_active {
                            if let Some(ref mut tv) = self.terrain_viewer {
                                tv.handle_mouse_drag(dx, dy);
                            }
                        } else if pc_count > 0 {
                            if let Some(ref mut pc) = self.point_cloud {
                                pc.handle_mouse_drag(dx, dy);
                            }
                        }
                    }
                }
                
                self.camera.handle_mouse_move(new_x, new_y);
                true
            }
            WindowEvent::MouseWheel { delta, .. } => {
                let scroll = match delta {
                    MouseScrollDelta::LineDelta(_x, y) => *y * 3.0,
                    MouseScrollDelta::PixelDelta(pos) => pos.y as f32 * 0.02,
                };
                
                // Check what's active
                let terrain_active = self.terrain_viewer.as_ref()
                    .map_or(false, |tv| tv.has_terrain());
                let pc_count = self.point_cloud.as_ref().map_or(0, |pc| pc.point_count);
                
                if terrain_active {
                    if let Some(ref mut tv) = self.terrain_viewer {
                        tv.handle_scroll(scroll);
                    }
                } else if pc_count > 0 {
                    if let Some(ref mut pc) = self.point_cloud {
                        pc.handle_scroll(scroll);
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
        // Otherwise, if point cloud is active, route to point cloud camera
        let terrain_active = self.terrain_viewer.as_ref().map_or(false, |tv| tv.has_terrain());
        let pc_active = self.point_cloud.as_ref().map_or(false, |pc| pc.point_count > 0);
        
        if terrain_active {
            if let Some(ref mut tv) = self.terrain_viewer {
                tv.handle_keys(forward, right, up);
            }
        } else if pc_active {
            if let Some(ref mut pc) = self.point_cloud {
                pc.handle_keys(forward, right, up);
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
            let view_proj = proj * model_view;
            let cam = crate::core::screen_space_effects::CameraParams {
                view_matrix: to_arr4(model_view),
                inv_view_matrix: to_arr4(inv_model_view),
                proj_matrix: to_arr4(proj),
                inv_proj_matrix: to_arr4(inv_proj),
                prev_view_proj_matrix: to_arr4(self.prev_view_proj),
                camera_pos: [eye.x, eye.y, eye.z],
                frame_index: self.frame_count as u32,
                // P1.2: Pass jitter offset to shaders for TAA unjitter
                jitter_offset: self.taa_jitter.offset_array(),
                _pad_jitter: [0.0, 0.0],
            };
            gi.update_camera(&self.queue, &cam);
            // P1.1: Store current view_proj for next frame's motion vectors
            self.prev_view_proj = view_proj;
        }
    }
}
