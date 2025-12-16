// src/viewer/input/mod.rs
// Input handling for the interactive viewer
// Extracted from mod.rs as part of the viewer refactoring

mod keyboard;
mod mouse;
mod viewer_input;

pub use keyboard::handle_keyboard_input;
pub use mouse::{handle_cursor_move, handle_mouse_input, handle_scroll};
// viewer_input exports Viewer::handle_input() and Viewer::update() impl directly

use super::camera_controller::CameraController;
use super::terrain::ViewerTerrainScene;
use std::collections::HashSet;
use winit::event::WindowEvent;
use winit::keyboard::KeyCode;

/// Calculate movement from pressed keys - alias for get_movement_from_keys
pub fn calculate_movement(keys_pressed: &HashSet<KeyCode>, shift_pressed: bool) -> (f32, f32, f32) {
    keyboard::get_movement_from_keys(keys_pressed, shift_pressed)
}

/// Process all input events, delegating to keyboard/mouse handlers
pub fn handle_input(
    event: &WindowEvent,
    camera: &mut CameraController,
    terrain_viewer: &mut Option<ViewerTerrainScene>,
    keys_pressed: &mut HashSet<KeyCode>,
    shift_pressed: &mut bool,
) -> bool {
    match event {
        WindowEvent::KeyboardInput {
            event: key_event, ..
        } => handle_keyboard_input(key_event, camera, keys_pressed, shift_pressed),
        WindowEvent::MouseInput { state, button, .. } => {
            handle_mouse_input(*state, *button, camera)
        }
        WindowEvent::CursorMoved { position, .. } => {
            handle_cursor_move(position.x as f32, position.y as f32, camera, terrain_viewer)
        }
        WindowEvent::MouseWheel { delta, .. } => handle_scroll(delta, camera, terrain_viewer),
        _ => false,
    }
}
