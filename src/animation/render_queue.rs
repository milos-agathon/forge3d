//! Offline render queue for frame-by-frame animation export
//!
//! Renders animation frames to PNG files with deterministic output.

use super::{CameraAnimation, CameraState};
use std::path::{Path, PathBuf};

/// Progress information for render callbacks
#[derive(Debug, Clone)]
pub struct RenderProgress {
    pub frame: u32,
    pub total_frames: u32,
    pub time: f32,
    pub output_path: PathBuf,
}

impl RenderProgress {
    /// Get progress as percentage (0.0 to 1.0)
    pub fn percent(&self) -> f32 {
        if self.total_frames == 0 {
            0.0
        } else {
            self.frame as f32 / self.total_frames as f32
        }
    }
}

/// Configuration for offline animation rendering
#[derive(Debug, Clone)]
pub struct RenderConfig {
    /// Output directory for frames
    pub output_dir: PathBuf,
    /// Frames per second
    pub fps: u32,
    /// Output width in pixels
    pub width: u32,
    /// Output height in pixels
    pub height: u32,
    /// Frame filename prefix (default: "frame")
    pub filename_prefix: String,
    /// Number of digits in frame number (default: 4, e.g., frame_0001.png)
    pub frame_digits: usize,
}

impl Default for RenderConfig {
    fn default() -> Self {
        Self {
            output_dir: PathBuf::from("./frames"),
            fps: 30,
            width: 1920,
            height: 1080,
            filename_prefix: "frame".to_string(),
            frame_digits: 4,
        }
    }
}

impl RenderConfig {
    pub fn new(output_dir: impl AsRef<Path>, fps: u32, width: u32, height: u32) -> Self {
        Self {
            output_dir: output_dir.as_ref().to_path_buf(),
            fps,
            width,
            height,
            ..Default::default()
        }
    }

    /// Generate frame path for given frame number
    pub fn frame_path(&self, frame: u32) -> PathBuf {
        let filename = format!(
            "{}_{:0width$}.png",
            self.filename_prefix,
            frame,
            width = self.frame_digits
        );
        self.output_dir.join(filename)
    }
}

/// Result of a render operation
#[derive(Debug)]
pub struct RenderResult {
    /// Number of frames successfully rendered
    pub frames_rendered: u32,
    /// Total expected frames
    pub total_frames: u32,
    /// Output directory
    pub output_dir: PathBuf,
    /// Any error that occurred
    pub error: Option<String>,
}

impl RenderResult {
    pub fn success(frames: u32, output_dir: PathBuf) -> Self {
        Self {
            frames_rendered: frames,
            total_frames: frames,
            output_dir,
            error: None,
        }
    }

    pub fn partial(rendered: u32, total: u32, output_dir: PathBuf, error: String) -> Self {
        Self {
            frames_rendered: rendered,
            total_frames: total,
            output_dir,
            error: Some(error),
        }
    }

    pub fn is_complete(&self) -> bool {
        self.error.is_none() && self.frames_rendered == self.total_frames
    }
}

/// Offline render queue for animation frame export
pub struct OfflineRenderQueue {
    animation: CameraAnimation,
    config: RenderConfig,
}

impl OfflineRenderQueue {
    pub fn new(animation: CameraAnimation, config: RenderConfig) -> Self {
        Self { animation, config }
    }

    /// Get animation reference
    pub fn animation(&self) -> &CameraAnimation {
        &self.animation
    }

    /// Get config reference
    pub fn config(&self) -> &RenderConfig {
        &self.config
    }

    /// Calculate time for given frame number
    pub fn frame_time(&self, frame: u32) -> f32 {
        frame as f32 / self.config.fps as f32
    }

    /// Get camera state for given frame
    pub fn camera_state_at_frame(&self, frame: u32) -> Option<CameraState> {
        let time = self.frame_time(frame);
        self.animation.evaluate(time)
    }

    /// Get total number of frames
    pub fn total_frames(&self) -> u32 {
        self.animation.frame_count(self.config.fps)
    }

    /// Iterate over all frame times
    pub fn frame_times(&self) -> impl Iterator<Item = (u32, f32)> + '_ {
        let total = self.total_frames();
        let fps = self.config.fps;
        (0..total).map(move |frame| (frame, frame as f32 / fps as f32))
    }

    /// Create output directory if it doesn't exist
    pub fn ensure_output_dir(&self) -> std::io::Result<()> {
        std::fs::create_dir_all(&self.config.output_dir)
    }
}
