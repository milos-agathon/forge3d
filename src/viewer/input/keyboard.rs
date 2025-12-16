// src/viewer/input/keyboard.rs
// Keyboard input handling for the interactive viewer

use super::super::camera_controller::{CameraController, CameraMode};
use std::collections::HashSet;
use winit::event::{ElementState, KeyEvent};
use winit::keyboard::{KeyCode, PhysicalKey};

/// Handle keyboard input events
/// Returns true if the event was consumed
pub fn handle_keyboard_input(
    key_event: &KeyEvent,
    camera: &mut CameraController,
    keys_pressed: &mut HashSet<KeyCode>,
    shift_pressed: &mut bool,
) -> bool {
    if let PhysicalKey::Code(keycode) = key_event.physical_key {
        let pressed = key_event.state == ElementState::Pressed;

        // Track shift
        if matches!(keycode, KeyCode::ShiftLeft | KeyCode::ShiftRight) {
            *shift_pressed = pressed;
        }

        // Track WASD, Q, E for FPS mode
        if pressed {
            keys_pressed.insert(keycode);
        } else {
            keys_pressed.remove(&keycode);
        }

        // Toggle camera mode with Tab
        if pressed && keycode == KeyCode::Tab {
            let new_mode = match camera.mode() {
                CameraMode::Orbit => CameraMode::Fps,
                CameraMode::Fps => CameraMode::Orbit,
            };
            camera.set_mode(new_mode);
            println!("Camera mode: {:?}", new_mode);
            return true;
        }
    }

    true
}

/// Calculate movement input from pressed keys
/// Returns (forward, right, up) movement values
pub fn get_movement_from_keys(
    keys_pressed: &HashSet<KeyCode>,
    shift_pressed: bool,
) -> (f32, f32, f32) {
    let mut forward = 0.0;
    let mut right = 0.0;
    let mut up = 0.0;

    let speed_mult = if shift_pressed { 2.0 } else { 1.0 };

    if keys_pressed.contains(&KeyCode::KeyW) || keys_pressed.contains(&KeyCode::ArrowUp) {
        forward += speed_mult;
    }
    if keys_pressed.contains(&KeyCode::KeyS) || keys_pressed.contains(&KeyCode::ArrowDown) {
        forward -= speed_mult;
    }
    if keys_pressed.contains(&KeyCode::KeyD) || keys_pressed.contains(&KeyCode::ArrowRight) {
        right += speed_mult;
    }
    if keys_pressed.contains(&KeyCode::KeyA) || keys_pressed.contains(&KeyCode::ArrowLeft) {
        right -= speed_mult;
    }
    if keys_pressed.contains(&KeyCode::KeyE) {
        up += speed_mult;
    }
    if keys_pressed.contains(&KeyCode::KeyQ) {
        up -= speed_mult;
    }

    (forward, right, up)
}
