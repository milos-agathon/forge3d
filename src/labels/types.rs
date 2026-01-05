//! Label types and data structures.

use glam::Vec3;

/// Unique identifier for a label.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct LabelId(pub u64);

/// Style configuration for a label.
#[derive(Debug, Clone)]
pub struct LabelStyle {
    /// Font size in pixels.
    pub size: f32,
    /// Text color as RGBA in linear space (0..1).
    pub color: [f32; 4],
    /// Halo (outline) color as RGBA.
    pub halo_color: [f32; 4],
    /// Halo width in pixels (0 = no halo).
    pub halo_width: f32,
    /// Priority for collision resolution (higher = more important).
    pub priority: i32,
    /// Minimum depth (near plane) for visibility.
    pub min_depth: f32,
    /// Maximum depth (far plane) for visibility.
    pub max_depth: f32,
    /// Alpha fade based on depth (0 = no fade, 1 = full fade at max_depth).
    pub depth_fade: f32,
}

impl Default for LabelStyle {
    fn default() -> Self {
        Self {
            size: 14.0,
            color: [0.1, 0.1, 0.1, 1.0], // Dark gray
            halo_color: [1.0, 1.0, 1.0, 0.8], // White with slight transparency
            halo_width: 1.5,
            priority: 0,
            min_depth: 0.0,
            max_depth: 1.0,
            depth_fade: 0.0,
        }
    }
}

impl LabelStyle {
    /// Create a style with custom color.
    pub fn with_color(mut self, r: f32, g: f32, b: f32, a: f32) -> Self {
        self.color = [r, g, b, a];
        self
    }

    /// Create a style with custom size.
    pub fn with_size(mut self, size: f32) -> Self {
        self.size = size;
        self
    }

    /// Create a style with custom priority.
    pub fn with_priority(mut self, priority: i32) -> Self {
        self.priority = priority;
        self
    }

    /// Create a style with custom halo.
    pub fn with_halo(mut self, color: [f32; 4], width: f32) -> Self {
        self.halo_color = color;
        self.halo_width = width;
        self
    }

    /// Create a style with no halo.
    pub fn no_halo(mut self) -> Self {
        self.halo_width = 0.0;
        self
    }
}

/// Data for a single label.
#[derive(Debug, Clone)]
pub struct LabelData {
    /// Unique identifier.
    pub id: LabelId,
    /// Text content.
    pub text: String,
    /// World position (x, y, z).
    pub world_pos: Vec3,
    /// Style configuration.
    pub style: LabelStyle,
    /// Computed screen position (x, y) or None if off-screen.
    pub screen_pos: Option<[f32; 2]>,
    /// Whether the label is currently visible (not occluded/collided).
    pub visible: bool,
    /// Depth value for sorting/occlusion (0 = near, 1 = far).
    pub depth: f32,
}
