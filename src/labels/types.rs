//! Label types and data structures.

use glam::Vec3;

/// Unique identifier for a label.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct LabelId(pub u64);

/// Placement type for line labels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum LineLabelPlacement {
    /// Place at center of line
    #[default]
    Center,
    /// Place along the line (curved/angled)
    Along,
}

/// Style flags for label text.
#[derive(Debug, Clone, Copy, Default)]
pub struct LabelFlags {
    /// Render text with underline
    pub underline: bool,
    /// Render text in small caps
    pub small_caps: bool,
    /// Show leader line to anchor position
    pub leader: bool,
}

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
    /// Minimum zoom level for visibility (0.0 = always visible at min zoom).
    pub min_zoom: f32,
    /// Maximum zoom level for visibility (f32::MAX = always visible at max zoom).
    pub max_zoom: f32,
    /// Rotation angle in radians (0 = horizontal).
    pub rotation: f32,
    /// Offset from anchor position in screen pixels (x, y).
    pub offset: [f32; 2],
    /// Style flags (underline, small-caps, leader).
    pub flags: LabelFlags,
    /// Horizon fade angle in degrees (labels near horizon fade).
    pub horizon_fade_angle: f32,
}

impl Default for LabelStyle {
    fn default() -> Self {
        Self {
            size: 14.0,
            color: [0.1, 0.1, 0.1, 1.0],
            halo_color: [1.0, 1.0, 1.0, 0.8],
            halo_width: 1.5,
            priority: 0,
            min_depth: 0.0,
            max_depth: 1.0,
            depth_fade: 0.0,
            min_zoom: 0.0,
            max_zoom: f32::MAX,
            rotation: 0.0,
            offset: [0.0, 0.0],
            flags: LabelFlags::default(),
            horizon_fade_angle: 5.0,
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

    /// Set min/max zoom range for scale-dependent visibility.
    pub fn with_zoom_range(mut self, min_zoom: f32, max_zoom: f32) -> Self {
        self.min_zoom = min_zoom;
        self.max_zoom = max_zoom;
        self
    }

    /// Set rotation angle in radians.
    pub fn with_rotation(mut self, radians: f32) -> Self {
        self.rotation = radians;
        self
    }

    /// Set offset from anchor position in screen pixels.
    pub fn with_offset(mut self, x: f32, y: f32) -> Self {
        self.offset = [x, y];
        self
    }

    /// Enable underline style.
    pub fn with_underline(mut self) -> Self {
        self.flags.underline = true;
        self
    }

    /// Enable small-caps style.
    pub fn with_small_caps(mut self) -> Self {
        self.flags.small_caps = true;
        self
    }

    /// Enable leader line to anchor.
    pub fn with_leader(mut self) -> Self {
        self.flags.leader = true;
        self
    }

    /// Set horizon fade angle in degrees.
    pub fn with_horizon_fade(mut self, degrees: f32) -> Self {
        self.horizon_fade_angle = degrees;
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
    /// Computed horizon angle for fade (degrees from horizontal).
    pub horizon_angle: f32,
    /// Computed alpha after horizon fade applied.
    pub computed_alpha: f32,
}

/// Data for a line label (text along a polyline).
#[derive(Debug, Clone)]
pub struct LineLabelData {
    /// Unique identifier.
    pub id: LabelId,
    /// Text content.
    pub text: String,
    /// Polyline vertices in world space [(x, y, z), ...].
    pub polyline: Vec<Vec3>,
    /// Style configuration.
    pub style: LabelStyle,
    /// Placement mode (center or along).
    pub placement: LineLabelPlacement,
    /// Repeat distance in screen pixels (0 = no repeat).
    pub repeat_distance: f32,
    /// Computed screen positions for each glyph.
    pub glyph_positions: Vec<GlyphPlacement>,
    /// Whether the line label is currently visible.
    pub visible: bool,
}

/// Placement info for a single glyph in a line label.
#[derive(Debug, Clone, Copy)]
pub struct GlyphPlacement {
    /// Screen position (x, y).
    pub screen_pos: [f32; 2],
    /// Rotation angle in radians (tangent to line).
    pub rotation: f32,
    /// Scale factor.
    pub scale: f32,
}

/// Leader line data for offset labels.
#[derive(Debug, Clone)]
pub struct LeaderLine {
    /// Start point in screen space (anchor).
    pub start: [f32; 2],
    /// End point in screen space (label position).
    pub end: [f32; 2],
    /// Line color.
    pub color: [f32; 4],
    /// Line width in pixels.
    pub width: f32,
}
