//! M1: Accumulation AA infrastructure for offline terrain rendering
//!
//! Provides HDR accumulation buffer and deterministic jitter sequences
//! for high-quality antialiasing in offline renders.
//!
//! RELEVANT FILES: src/terrain/renderer.rs, src/terrain/camera.rs

use wgpu::{
    Buffer, CommandEncoder, Device, Extent3d, Texture, TextureDescriptor, TextureDimension,
    TextureFormat, TextureUsages, TextureView, TextureViewDescriptor,
};

/// Accumulation AA configuration
#[derive(Debug, Clone, Copy)]
pub struct AccumulationConfig {
    /// Number of samples to accumulate (1 = no AA, 16/64/256 typical)
    pub samples: u32,
    /// Optional seed for deterministic jitter (None = use default sequence)
    pub seed: Option<u64>,
}

impl Default for AccumulationConfig {
    fn default() -> Self {
        Self {
            samples: 1,
            seed: None,
        }
    }
}

/// HDR accumulation buffer for multi-sample averaging
pub struct AccumulationBuffer {
    /// HDR accumulation texture (Rgba32Float for precision)
    pub texture: Texture,
    /// View for accumulation texture
    pub view: TextureView,
    /// Sample count buffer for averaging
    pub sample_count: u32,
    /// Buffer dimensions
    pub width: u32,
    pub height: u32,
    /// Staging buffer for GPU readback of accumulated result
    staging_buffer: Option<Buffer>,
}

impl AccumulationBuffer {
    /// Create a new accumulation buffer
    pub fn new(device: &Device, width: u32, height: u32) -> Self {
        let texture = device.create_texture(&TextureDescriptor {
            label: Some("terrain.accumulation.texture"),
            size: Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            // Use Rgba32Float for maximum precision during accumulation
            format: TextureFormat::Rgba32Float,
            usage: TextureUsages::RENDER_ATTACHMENT
                | TextureUsages::TEXTURE_BINDING
                | TextureUsages::COPY_SRC
                | TextureUsages::COPY_DST
                | TextureUsages::STORAGE_BINDING,
            view_formats: &[],
        });

        let view = texture.create_view(&TextureViewDescriptor {
            label: Some("terrain.accumulation.view"),
            ..Default::default()
        });

        Self {
            texture,
            view,
            sample_count: 0,
            width,
            height,
            staging_buffer: None,
        }
    }

    /// Reset accumulation for a new render
    pub fn reset(&mut self) {
        self.sample_count = 0;
    }

    /// Check if buffer needs resize
    pub fn needs_resize(&self, width: u32, height: u32) -> bool {
        self.width != width || self.height != height
    }

    /// Resize the accumulation buffer
    pub fn resize(&mut self, device: &Device, width: u32, height: u32) {
        if !self.needs_resize(width, height) {
            return;
        }
        *self = Self::new(device, width, height);
    }

    /// Clear the accumulation buffer
    pub fn clear(&mut self, _encoder: &mut CommandEncoder) {
        // Clear by writing zeros - we'll use a compute pass or copy from a zero buffer
        self.sample_count = 0;
    }

    /// Increment sample count after accumulation
    pub fn increment_sample(&mut self) {
        self.sample_count += 1;
    }

    /// Get current sample count
    pub fn current_samples(&self) -> u32 {
        self.sample_count
    }
}

/// Jitter sequence generator for subpixel camera offsets
///
/// Uses R2 quasi-random sequence (generalized golden ratio) for
/// optimal low-discrepancy sampling, with optional Halton fallback.
#[derive(Debug, Clone)]
pub struct JitterSequence {
    /// Precomputed jitter offsets (x, y) in [-0.5, 0.5] pixel space
    offsets: Vec<(f32, f32)>,
    /// Current index in sequence
    current_index: usize,
}

impl JitterSequence {
    /// Create a new jitter sequence with the given sample count
    ///
    /// Uses R2 sequence (generalized golden ratio) for optimal coverage
    pub fn new(sample_count: u32, seed: Option<u64>) -> Self {
        let offsets = Self::generate_r2_sequence(sample_count, seed);
        Self {
            offsets,
            current_index: 0,
        }
    }

    /// Generate R2 quasi-random sequence
    ///
    /// R2 sequence uses generalized golden ratio for 2D:
    /// φ₂ = 1.32471795724... (plastic constant)
    /// α₁ = 1/φ₂, α₂ = 1/φ₂²
    fn generate_r2_sequence(count: u32, seed: Option<u64>) -> Vec<(f32, f32)> {
        if count <= 1 {
            return vec![(0.0, 0.0)];
        }

        // R2 sequence constants (generalized golden ratio for 2D)
        // φ₂ ≈ 1.32471795724474602596 (plastic constant)
        const PHI2: f64 = 1.32471795724474602596;
        let alpha1 = 1.0 / PHI2;
        let alpha2 = 1.0 / (PHI2 * PHI2);

        // Starting point based on seed
        let start = seed.unwrap_or(0) as f64 * 0.5;

        let mut offsets = Vec::with_capacity(count as usize);
        for i in 0..count {
            let n = (i as f64) + start;
            // R2 sequence formula: x_n = frac(n * α)
            let x = ((n * alpha1) % 1.0) as f32;
            let y = ((n * alpha2) % 1.0) as f32;
            // Map from [0,1] to [-0.5, 0.5] for pixel-center offset
            offsets.push((x - 0.5, y - 0.5));
        }

        offsets
    }

    /// Generate Halton sequence (alternative low-discrepancy sequence)
    #[allow(dead_code)]
    fn generate_halton_sequence(count: u32, seed: Option<u64>) -> Vec<(f32, f32)> {
        if count <= 1 {
            return vec![(0.0, 0.0)];
        }

        let start = seed.unwrap_or(0) as u32;
        let mut offsets = Vec::with_capacity(count as usize);

        for i in 0..count {
            let idx = start + i + 1; // 1-indexed for Halton
            let x = Self::halton(idx, 2);
            let y = Self::halton(idx, 3);
            // Map from [0,1] to [-0.5, 0.5]
            offsets.push((x - 0.5, y - 0.5));
        }

        offsets
    }

    /// Compute Halton sequence value for given index and base
    fn halton(mut index: u32, base: u32) -> f32 {
        let mut result = 0.0f32;
        let mut f = 1.0f32 / base as f32;

        while index > 0 {
            result += f * (index % base) as f32;
            index /= base;
            f /= base as f32;
        }

        result
    }

    /// Reset sequence to beginning
    pub fn reset(&mut self) {
        self.current_index = 0;
    }

    /// Get next jitter offset in sequence
    pub fn next(&mut self) -> (f32, f32) {
        let offset = self.offsets[self.current_index];
        self.current_index = (self.current_index + 1) % self.offsets.len();
        offset
    }

    /// Get jitter offset at specific index
    pub fn get(&self, index: usize) -> (f32, f32) {
        self.offsets[index % self.offsets.len()]
    }

    /// Get total number of samples in sequence
    pub fn len(&self) -> usize {
        self.offsets.len()
    }

    /// Check if sequence is empty
    pub fn is_empty(&self) -> bool {
        self.offsets.is_empty()
    }
}

/// Apply subpixel jitter to projection matrix
///
/// Shifts the projection by a subpixel offset for accumulation AA.
/// The jitter is in pixel space [-0.5, 0.5] and is converted to NDC.
pub fn apply_jitter_to_projection(
    proj: glam::Mat4,
    jitter_x: f32,
    jitter_y: f32,
    width: u32,
    height: u32,
) -> glam::Mat4 {
    // Convert pixel jitter to NDC space
    // NDC range is [-1, 1], so pixel jitter needs to be scaled
    let ndc_jitter_x = (2.0 * jitter_x) / width as f32;
    let ndc_jitter_y = (2.0 * jitter_y) / height as f32;

    // Create translation matrix for jitter
    // Apply to projection matrix columns [2][0] and [2][1] (NDC offset)
    let mut jittered = proj;
    jittered.col_mut(2).x += ndc_jitter_x * proj.col(3).w;
    jittered.col_mut(2).y += ndc_jitter_y * proj.col(3).w;

    jittered
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_jitter_sequence_r2() {
        let seq = JitterSequence::new(16, None);
        assert_eq!(seq.len(), 16);

        // All offsets should be in [-0.5, 0.5]
        for (x, y) in &seq.offsets {
            assert!(*x >= -0.5 && *x <= 0.5, "x={} out of range", x);
            assert!(*y >= -0.5 && *y <= 0.5, "y={} out of range", y);
        }
    }

    #[test]
    fn test_jitter_sequence_deterministic() {
        let seq1 = JitterSequence::new(64, Some(42));
        let seq2 = JitterSequence::new(64, Some(42));

        for i in 0..64 {
            assert_eq!(seq1.get(i), seq2.get(i), "Sequence not deterministic at {}", i);
        }
    }

    #[test]
    fn test_jitter_sequence_single_sample() {
        let seq = JitterSequence::new(1, None);
        assert_eq!(seq.len(), 1);
        assert_eq!(seq.get(0), (0.0, 0.0));
    }

    #[test]
    fn test_halton_sequence() {
        // Halton(1, 2) = 0.5, Halton(1, 3) = 0.333...
        let h2 = JitterSequence::halton(1, 2);
        let h3 = JitterSequence::halton(1, 3);
        assert!((h2 - 0.5).abs() < 0.001);
        assert!((h3 - 0.333).abs() < 0.01);
    }
}
