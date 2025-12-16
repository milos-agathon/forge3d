// src/viewer/input/mouse.rs
// Mouse input handling for the interactive viewer

use super::super::camera_controller::CameraController;
use super::super::terrain::ViewerTerrainScene;
use winit::event::{ElementState, MouseButton, MouseScrollDelta};

/// Handle mouse button input
/// Returns true if the event was consumed
pub fn handle_mouse_input(
    state: ElementState,
    button: MouseButton,
    camera: &mut CameraController,
) -> bool {
    if button == MouseButton::Left {
        camera.mouse_pressed = state == ElementState::Pressed;
    }
    true
}

/// Handle cursor movement
/// Returns true if the event was consumed
pub fn handle_cursor_move(
    new_x: f32,
    new_y: f32,
    camera: &mut CameraController,
    terrain_viewer: &mut Option<ViewerTerrainScene>,
) -> bool {
    // If terrain viewer is active and mouse is pressed, orbit the terrain camera
    if camera.mouse_pressed {
        if let Some(ref mut tv) = terrain_viewer {
            if tv.has_terrain() {
                if let Some((last_x, last_y)) = camera.last_mouse_pos {
                    let dx = new_x - last_x;
                    let dy = new_y - last_y;
                    tv.handle_mouse_drag(dx, dy);
                }
            }
        }
    }

    camera.handle_mouse_move(new_x, new_y);
    true
}

/// Handle mouse scroll/wheel input
/// Returns true if the event was consumed
pub fn handle_scroll(
    delta: &MouseScrollDelta,
    camera: &mut CameraController,
    terrain_viewer: &mut Option<ViewerTerrainScene>,
) -> bool {
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
    if let Some(ref mut tv) = terrain_viewer {
        if tv.has_terrain() {
            tv.handle_scroll(scroll);
        }
    } else {
        camera.handle_mouse_scroll(scroll);
    }
    true
}
