// src/viewer/viewer_config.rs
// Configuration and utility types for the interactive viewer
// RELEVANT FILES: src/viewer/mod.rs

use std::time::{Duration, Instant};

/// Global initial commands for viewer (set by CLI parser in example)
pub static INITIAL_CMDS: once_cell::sync::OnceCell<Vec<String>> = once_cell::sync::OnceCell::new();

/// Set initial commands to be executed when viewer starts
pub fn set_initial_commands(cmds: Vec<String>) {
    let _ = INITIAL_CMDS.set(cmds);
}

/// Optional initial terrain configuration (set by open_terrain_viewer via lib.rs).
#[cfg(feature = "extension-module")]
pub static INITIAL_TERRAIN_CONFIG: once_cell::sync::OnceCell<crate::render::params::RendererConfig> =
    once_cell::sync::OnceCell::new();

/// Set initial terrain configuration for viewer
#[cfg(feature = "extension-module")]
pub fn set_initial_terrain_config(cfg: crate::render::params::RendererConfig) {
    let _ = INITIAL_TERRAIN_CONFIG.set(cfg);
}

/// Viewer window and rendering configuration
#[derive(Clone)]
pub struct ViewerConfig {
    pub width: u32,
    pub height: u32,
    pub title: String,
    pub vsync: bool,
    pub fov_deg: f32,
    pub znear: f32,
    pub zfar: f32,
    pub snapshot_width: Option<u32>,
    pub snapshot_height: Option<u32>,
}

impl Default for ViewerConfig {
    fn default() -> Self {
        Self {
            width: 1024,
            height: 768,
            title: "forge3d Interactive Viewer".to_string(),
            vsync: true,
            fov_deg: 45.0,
            znear: 0.1,
            zfar: 1000.0,
            snapshot_width: None,
            snapshot_height: None,
        }
    }
}

/// FPS counter for performance monitoring
pub struct FpsCounter {
    frames: u32,
    last_report: Instant,
    current_fps: f32,
}

impl FpsCounter {
    pub fn new() -> Self {
        Self {
            frames: 0,
            last_report: Instant::now(),
            current_fps: 0.0,
        }
    }

    pub fn tick(&mut self) -> Option<f32> {
        self.frames += 1;
        let elapsed = self.last_report.elapsed();
        if elapsed >= Duration::from_secs(1) {
            self.current_fps = self.frames as f32 / elapsed.as_secs_f32();
            self.frames = 0;
            self.last_report = Instant::now();
            Some(self.current_fps)
        } else {
            None
        }
    }

    pub fn fps(&self) -> f32 {
        self.current_fps
    }
}

impl Default for FpsCounter {
    fn default() -> Self {
        Self::new()
    }
}
