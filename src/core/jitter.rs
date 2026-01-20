//! P1.2: Jitter sequence for Temporal Anti-Aliasing (TAA)
//!
//! Provides Halton 2,3 low-discrepancy sequence for sub-pixel projection jitter.
//! The sequence covers the pixel uniformly over time, reducing aliasing artifacts
//! when combined with temporal accumulation.

/// Default TAA jitter sequence length (8 samples is common, 16 for higher quality)
pub const DEFAULT_JITTER_SEQUENCE_LENGTH: u32 = 8;

/// Compute Halton sequence value for a given index and base.
///
/// The Halton sequence is a low-discrepancy sequence that provides
/// well-distributed samples in [0, 1).
///
/// # Arguments
/// * `index` - Sample index (0-based)
/// * `base` - Prime number base (2 or 3 for TAA)
#[inline]
pub fn halton(index: u32, base: u32) -> f32 {
    let mut result = 0.0f32;
    let mut f = 1.0f32;
    let mut i = index;

    while i > 0 {
        f /= base as f32;
        result += f * (i % base) as f32;
        i /= base;
    }

    result
}

/// Compute 2D Halton jitter offset for TAA.
///
/// Uses bases 2 and 3 which provide good 2D coverage.
/// Returns offset in [-0.5, 0.5] range (pixel-centered).
///
/// # Arguments
/// * `frame_index` - Current frame number
/// * `sequence_length` - Number of samples before repeating (default 8)
#[inline]
pub fn halton_2_3(frame_index: u32, sequence_length: u32) -> (f32, f32) {
    // Use 1-based index for Halton (index 0 returns 0,0 which we want to avoid)
    let index = (frame_index % sequence_length) + 1;
    let x = halton(index, 2) - 0.5; // Center around 0
    let y = halton(index, 3) - 0.5;
    (x, y)
}

/// Apply sub-pixel jitter to a projection matrix.
///
/// Modifies the projection matrix to offset the rendered image by a sub-pixel
/// amount. This is used for TAA to sample different sub-pixel locations each frame.
///
/// # Arguments
/// * `proj` - Original projection matrix (will be modified in place)
/// * `jitter_x` - Horizontal jitter in [-0.5, 0.5] pixel units
/// * `jitter_y` - Vertical jitter in [-0.5, 0.5] pixel units
/// * `width` - Render target width in pixels
/// * `height` - Render target height in pixels
///
/// # Returns
/// Modified projection matrix with jitter applied
#[inline]
pub fn apply_jitter(
    proj: glam::Mat4,
    jitter_x: f32,
    jitter_y: f32,
    width: u32,
    height: u32,
) -> glam::Mat4 {
    // Convert pixel jitter to NDC space [-1, 1]
    // jitter_x/y are in [-0.5, 0.5] pixel units
    // NDC offset = 2 * jitter / resolution
    let offset_x = 2.0 * jitter_x / width as f32;
    let offset_y = 2.0 * jitter_y / height as f32;

    // Create translation matrix that shifts in clip space
    // This is equivalent to modifying proj[2][0] and proj[2][1] (the translation terms)
    let jitter_matrix = glam::Mat4::from_cols(
        glam::Vec4::new(1.0, 0.0, 0.0, 0.0),
        glam::Vec4::new(0.0, 1.0, 0.0, 0.0),
        glam::Vec4::new(offset_x, offset_y, 1.0, 0.0),
        glam::Vec4::new(0.0, 0.0, 0.0, 1.0),
    );

    jitter_matrix * proj
}

/// TAA jitter state for a frame.
#[derive(Debug, Clone, Copy)]
pub struct JitterState {
    /// Whether TAA jitter is enabled
    pub enabled: bool,
    /// Current frame index in the jitter sequence
    pub index: u32,
    /// Length of the jitter sequence before repeating
    pub sequence_length: u32,
    /// Scale factor for the jitter offset (1.0 = default amplitude)
    pub scale: f32,
    /// Current jitter offset in pixel units [-0.5, 0.5]
    pub offset: (f32, f32),
}

impl Default for JitterState {
    fn default() -> Self {
        Self::new()
    }
}

impl JitterState {
    /// Create a new jitter state with default settings.
    pub fn new() -> Self {
        Self {
            enabled: false,
            index: 0,
            sequence_length: DEFAULT_JITTER_SEQUENCE_LENGTH,
            scale: 1.0,
            offset: (0.0, 0.0),
        }
    }

    /// Create jitter state with TAA enabled.
    pub fn enabled() -> Self {
        let mut state = Self {
            enabled: true,
            index: 0,
            sequence_length: DEFAULT_JITTER_SEQUENCE_LENGTH,
            scale: 1.0,
            offset: (0.0, 0.0),
        };
        state.refresh_offset();
        state
    }

    /// Advance to the next frame and compute new jitter offset.
    pub fn advance(&mut self) {
        if self.enabled {
            self.index = (self.index + 1) % self.sequence_length;
        }
        self.refresh_offset();
    }

    /// Update jitter scale and recompute the current offset.
    pub fn set_scale(&mut self, scale: f32) {
        self.scale = scale.max(0.0);
        self.refresh_offset();
    }

    /// Enable or disable jitter without resetting the sequence index.
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
        self.refresh_offset();
    }

    /// Get current jitter offset as an array for GPU upload.
    #[inline]
    pub fn offset_array(&self) -> [f32; 2] {
        [self.offset.0, self.offset.1]
    }

    fn refresh_offset(&mut self) {
        if self.enabled {
            let (x, y) = halton_2_3(self.index, self.sequence_length);
            self.offset = (x * self.scale, y * self.scale);
        } else {
            self.offset = (0.0, 0.0);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_halton_base_2() {
        // Halton base 2: 1/2, 1/4, 3/4, 1/8, 5/8, 3/8, 7/8, ...
        assert!((halton(1, 2) - 0.5).abs() < 1e-6);
        assert!((halton(2, 2) - 0.25).abs() < 1e-6);
        assert!((halton(3, 2) - 0.75).abs() < 1e-6);
        assert!((halton(4, 2) - 0.125).abs() < 1e-6);
    }

    #[test]
    fn test_halton_base_3() {
        // Halton base 3: 1/3, 2/3, 1/9, 4/9, 7/9, 2/9, ...
        assert!((halton(1, 3) - 1.0 / 3.0).abs() < 1e-6);
        assert!((halton(2, 3) - 2.0 / 3.0).abs() < 1e-6);
        assert!((halton(3, 3) - 1.0 / 9.0).abs() < 1e-6);
    }

    #[test]
    fn test_halton_2_3_range() {
        // All values should be in [-0.5, 0.5]
        for i in 0..16 {
            let (x, y) = halton_2_3(i, 8);
            assert!(x >= -0.5 && x <= 0.5, "x={} out of range for i={}", x, i);
            assert!(y >= -0.5 && y <= 0.5, "y={} out of range for i={}", y, i);
        }
    }

    #[test]
    fn test_jitter_state_advance() {
        let mut state = JitterState::enabled();
        let initial_offset = state.offset;

        state.advance();
        assert_ne!(state.offset, initial_offset, "Offset should change on advance");
        assert_eq!(state.index, 1);
    }

    #[test]
    fn test_jitter_state_disabled() {
        let mut state = JitterState::new();
        assert!(!state.enabled);
        assert_eq!(state.offset, (0.0, 0.0));

        state.advance();
        assert_eq!(state.offset, (0.0, 0.0), "Disabled jitter should stay at 0");
    }
}
