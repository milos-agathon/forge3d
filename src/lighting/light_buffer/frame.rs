use super::types::{LightBuffer, MAX_LIGHTS};
use crate::lighting::types::Light;
use wgpu::Buffer;
use super::r2::r2_sample;

impl LightBuffer {
    /// Advance to next frame (call once per frame)
    ///
    /// Updates the frame index for triple-buffering and generates new R2 sequence
    /// seeds for TAA-friendly light sampling on the GPU.
    ///
    /// # R2 Sequence Seeds (P1-03)
    ///
    /// Each frame generates a unique 2D seed using the R2 (Roberts) low-discrepancy
    /// sequence. This ensures:
    /// - **Temporal stability**: Deterministic seeds avoid flickering in TAA
    /// - **Spatial uniformity**: Well-distributed samples avoid clustering
    /// - **Wraparound safety**: Frame counter wraps at u64::MAX without issues
    ///
    /// ## WGSL Usage
    ///
    /// Seeds are uploaded to GPU as bit-encoded u32 values in `LightMetadata`:
    /// ```wgsl
    /// struct LightMetadata {
    ///     count: u32,
    ///     frame_index: u32,
    ///     seed_bits_x: u32,  // f32::to_bits() of seed[0]
    ///     seed_bits_y: u32,  // f32::to_bits() of seed[1]
    /// };
    /// ```
    ///
    /// Shaders decode seeds using `bitcast<f32>()` in `light_sequence_seed()`:
    /// ```wgsl
    /// fn light_sequence_seed() -> vec2<f32> {
    ///     return vec2<f32>(
    ///         bitcast<f32>(lightMeta.seed_bits_x),
    ///         bitcast<f32>(lightMeta.seed_bits_y)
    ///     );
    /// }
    /// ```
    ///
    /// Use these seeds as base offsets for stochastic light sampling:
    /// ```wgsl
    /// let base_seed = light_sequence_seed();
    /// let jittered = fract(base_seed + vec2<f32>(pixel_coords));
    /// let light_dir = sample_light(light_index, jittered);
    /// ```
    pub fn next_frame(&mut self) {
        self.frame_index = (self.frame_index + 1) % 3;
        self.frame_counter = self.frame_counter.wrapping_add(1);
        self.sequence_seed = r2_sample(self.frame_counter);
    }

    /// Get bind group for current frame
    ///
    /// Always returns a bind group. If no lights have been uploaded via `update()`,
    /// returns a default bind group with zero lights (neutral state).
    pub fn bind_group(&self) -> Option<&wgpu::BindGroup> {
        self.bind_group.as_ref()
    }

    /// Get bind group layout
    pub fn bind_group_layout(&self) -> &wgpu::BindGroupLayout {
        &self.bind_group_layout
    }

    /// Get current light count
    pub fn light_count(&self) -> u32 {
        self.light_count
    }

    /// Calculate memory usage in bytes
    pub fn memory_bytes(&self) -> u64 {
        let light_buffer_size = (MAX_LIGHTS * std::mem::size_of::<Light>()) as u64;
        let count_buffer_size = 16u64;

        // 3 buffers × (light_buffer + count_buffer)
        3 * (light_buffer_size + count_buffer_size) + 16
    }

    /// Calculate memory usage in megabytes
    pub fn memory_mb(&self) -> f64 {
        self.memory_bytes() as f64 / (1024.0 * 1024.0)
    }

    /// Return the current frame's R2 sequence seed for TAA-friendly light sampling
    pub fn sequence_seed(&self) -> [f32; 2] {
        self.sequence_seed
    }

    /// Expose the monotonic frame counter (useful for debugging)
    pub fn frame_counter(&self) -> u64 {
        self.frame_counter
    }

    /// Get the current frame's light buffer for bind group creation
    pub fn current_light_buffer(&self) -> &Buffer {
        &self.buffers[self.frame_index]
    }

    /// Get the current frame's count buffer for bind group creation
    pub fn current_count_buffer(&self) -> &Buffer {
        &self.count_buffers[self.frame_index]
    }

    /// Get the environment buffer (stub for P1-05, full IBL in P4)
    pub fn environment_buffer(&self) -> &Buffer {
        &self.environment_stub
    }

    // P1-07: Debug inspection API

    /// Get reference to last uploaded lights (P1-07)
    ///
    /// Returns a slice of lights uploaded via the most recent `update()` call.
    /// Useful for debug inspection, validation, and acceptance testing without
    /// GPU readback.
    ///
    /// # Example
    /// ```rust,ignore
    /// light_buffer.update(&device, &queue, &lights)?;
    /// let uploaded = light_buffer.last_uploaded_lights();
    /// assert_eq!(uploaded.len(), lights.len());
    /// ```
    pub fn last_uploaded_lights(&self) -> &[Light] {
        &self.last_uploaded_lights
    }

    /// Format debug information for light buffer state (P1-07)
    ///
    /// Returns a human-readable string describing:
    /// - Light count and frame counter
    /// - Current R2 seed values
    /// - Summary of each uploaded light (type, intensity, key fields)
    ///
    /// Intended for debug output, logging, and acceptance validation.
    ///
    /// # Example Output
    /// ```text
    /// LightBuffer Debug Info:
    ///   Count: 2 lights
    ///   Frame: 42 (seed: [0.234, 0.567])
    ///   
    ///   Light 0: Directional
    ///     Intensity: 3.00, Color: [1.00, 0.90, 0.80]
    ///     Direction: [-0.71, -0.50, 0.50]
    ///   
    ///   Light 1: Point
    ///     Intensity: 10.00, Color: [1.00, 1.00, 1.00]
    ///     Position: [0.00, 5.00, 0.00], Range: 20.00
    /// ```
    pub fn debug_info(&self) -> String {
        use std::fmt::Write;
        let mut output = String::new();

        writeln!(output, "LightBuffer Debug Info:").unwrap();
        writeln!(output, "  Count: {} lights", self.light_count).unwrap();
        writeln!(
            output,
            "  Frame: {} (seed: [{:.3}, {:.3}])",
            self.frame_counter, self.sequence_seed[0], self.sequence_seed[1]
        )
        .unwrap();
        writeln!(output).unwrap();

        for (i, light) in self.last_uploaded_lights.iter().enumerate() {
            writeln!(output, "  Light {}: {}", i, light_type_name(light.kind)).unwrap();
            writeln!(
                output,
                "    Intensity: {:.2}, Color: [{:.2}, {:.2}, {:.2}]",
                light.intensity, light.color[0], light.color[1], light.color[2]
            )
            .unwrap();

            // Type-specific fields
            match light.kind {
                0 => {
                    // Directional
                    writeln!(
                        output,
                        "    Direction: [{:.2}, {:.2}, {:.2}]",
                        light.dir_ws[0], light.dir_ws[1], light.dir_ws[2]
                    )
                    .unwrap();
                }
                1 => {
                    // Point
                    writeln!(
                        output,
                        "    Position: [{:.2}, {:.2}, {:.2}], Range: {:.2}",
                        light.pos_ws[0], light.pos_ws[1], light.pos_ws[2], light.range
                    )
                    .unwrap();
                }
                2 => {
                    // Spot
                    writeln!(
                        output,
                        "    Position: [{:.2}, {:.2}, {:.2}], Direction: [{:.2}, {:.2}, {:.2}]",
                        light.pos_ws[0],
                        light.pos_ws[1],
                        light.pos_ws[2],
                        light.dir_ws[0],
                        light.dir_ws[1],
                        light.dir_ws[2]
                    )
                    .unwrap();
                    writeln!(
                        output,
                        "    Cone: inner_cos={:.2}, outer_cos={:.2}, Range: {:.2}",
                        light.cone_cos[0], light.cone_cos[1], light.range
                    )
                    .unwrap();
                }
                3 => {
                    // Environment
                    writeln!(output, "    Texture Index: {}", light.env_texture_index).unwrap();
                }
                4 => {
                    // AreaRect
                    writeln!(
                        output,
                        "    Position: [{:.2}, {:.2}, {:.2}], Normal: [{:.2}, {:.2}, {:.2}]",
                        light.pos_ws[0],
                        light.pos_ws[1],
                        light.pos_ws[2],
                        light.dir_ws[0],
                        light.dir_ws[1],
                        light.dir_ws[2]
                    )
                    .unwrap();
                    writeln!(
                        output,
                        "    Half-extents: width={:.2}, height={:.2}",
                        light.area_half[0], light.area_half[1]
                    )
                    .unwrap();
                }
                5 => {
                    // AreaDisk
                    writeln!(
                        output,
                        "    Position: [{:.2}, {:.2}, {:.2}], Normal: [{:.2}, {:.2}, {:.2}]",
                        light.pos_ws[0],
                        light.pos_ws[1],
                        light.pos_ws[2],
                        light.dir_ws[0],
                        light.dir_ws[1],
                        light.dir_ws[2]
                    )
                    .unwrap();
                    writeln!(output, "    Radius: {:.2}", light.area_half[0]).unwrap();
                }
                6 => {
                    // AreaSphere
                    writeln!(
                        output,
                        "    Position: [{:.2}, {:.2}, {:.2}]",
                        light.pos_ws[0], light.pos_ws[1], light.pos_ws[2]
                    )
                    .unwrap();
                    writeln!(output, "    Radius: {:.2}", light.area_half[0]).unwrap();
                }
                _ => {
                    writeln!(output, "    (Unknown light type: {})", light.kind).unwrap();
                }
            }
            writeln!(output).unwrap();
        }

        output
    }
}

// Helper function for light type names
pub(crate) fn light_type_name(kind: u32) -> &'static str {
    match kind {
        0 => "Directional",
        1 => "Point",
        2 => "Spot",
        3 => "Environment",
        4 => "AreaRect",
        5 => "AreaDisk",
        6 => "AreaSphere",
        _ => "Unknown",
    }
}
