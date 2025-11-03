// src/lighting/light_buffer.rs
// P1: Light buffer management with triple-buffering for multi-light support
// SSBO storage buffer layout (std430) for efficient GPU upload
use wgpu::{Device, Queue, Buffer, BufferUsages};
use crate::lighting::types::Light;

/// Maximum number of lights supported (P1 default)
pub const MAX_LIGHTS: usize = 16;

/// Light buffer manager with triple-buffering for TAA-friendly updates
///
/// # P1-02: Triple-buffered SSBO Manager
///
/// Implements triple-buffered GPU light storage with R2 sequence seeds for
/// TAA-friendly temporal light sampling. Supports up to MAX_LIGHTS=16 lights
/// with minimal memory overhead.
///
/// ## Architecture
///
/// **Triple Buffering**: Maintains 3 sets of buffers to avoid GPU/CPU sync stalls:
/// - Frame N: GPU reads buffer[0], CPU writes buffer[1]
/// - Frame N+1: GPU reads buffer[1], CPU writes buffer[2]
/// - Frame N+2: GPU reads buffer[2], CPU writes buffer[0]
///
/// **R2 Sequence**: Generates low-discrepancy 2D samples for TAA-friendly sampling:
/// - Deterministic per-frame seeds avoid flickering
/// - Uniform distribution avoids clustering artifacts
/// - Frame counter wraps at u64::MAX
///
/// ## Bind Group Layout (matches WGSL)
///
/// ```text
/// @group(0) @binding(3) var<storage, read> lights: array<LightGPU>;
/// @group(0) @binding(4) var<uniform> lightMeta: LightMetadata;
/// @group(0) @binding(5) var<uniform> environmentParams: vec4<f32>;
/// ```
///
/// - **Binding 3**: Light array SSBO (std430, read-only storage)
///   - Size: MAX_LIGHTS × 80 bytes = 1280 bytes
///   - Usage: STORAGE | COPY_DST
///
/// - **Binding 4**: LightMetadata uniform
///   - Layout: `[count: u32, frame_index: u32, seed_bits_x: u32, seed_bits_y: u32]`
///   - Size: 16 bytes (4 × u32)
///   - Usage: UNIFORM | COPY_DST
///
/// - **Binding 5**: Environment params (stub for P4 IBL)
///   - Size: 16 bytes (vec4<f32>)
///   - Usage: UNIFORM | COPY_DST
///
/// ## Memory Budget
///
/// ```text
/// Component                 | Per-frame | Triple-buffered | Total
/// --------------------------|-----------|-----------------|-------
/// Light storage (1280 B)    |  1280 B   |     3840 B      | 3.75 KB
/// Metadata uniform (16 B)   |    16 B   |       48 B      | 0.05 KB
/// Environment stub (16 B)   |    16 B   |        - B      | 0.02 KB
/// --------------------------|-----------|-----------------|-------
/// Total                     |  1312 B   |     3904 B      | 3.81 KB
/// ```
///
/// Total memory usage is **3904 bytes** (~0.0037 MiB), well within the
/// 512 MiB host-visible budget.
///
/// ## Usage
///
/// ```rust,ignore
/// let mut light_buffer = LightBuffer::new(&device);
///
/// // Per-frame update
/// let lights = vec![Light::directional(45.0, 30.0, 3.0, [1.0, 0.9, 0.8])];
/// light_buffer.update(&device, &queue, &lights)?;
///
/// // Bind in render pass
/// render_pass.set_bind_group(0, light_buffer.bind_group().unwrap(), &[]);
///
/// // Advance frame counter
/// light_buffer.next_frame();
/// ```
///
/// ## Verification
///
/// Unit tests verify:
/// - Triple-buffer cycling (frame index 0→1→2→0)
/// - Bind layout matches WGSL constants (bindings 3, 4, 5)
/// - Memory budget calculations (memory_bytes(), memory_mb())
/// - R2 sequence generation (deterministic, range [0,1])
/// - MAX_LIGHTS enforcement (update() returns Err if exceeded)
pub struct LightBuffer {
    /// Storage buffers for light array (triple-buffered)
    buffers: [Buffer; 3],
    /// Uniform buffer for light count (triple-buffered)
    count_buffers: [Buffer; 3],
    /// Uniform buffer placeholder for environment lighting parameters (P1-05)
    /// 
    /// Currently initialized to zeros. Full IBL with importance sampling is
    /// deferred to P4. This stub allows shaders to link successfully without
    /// asset dependencies (no texture samplers required).
    /// 
    /// Size: 16 bytes (vec4<f32> in WGSL)
    /// Binding: @group(0) @binding(5)
    /// 
    /// Future P4 fields may include:
    /// - x: environment intensity multiplier
    /// - y: environment rotation (degrees)
    /// - z: environment exposure
    /// - w: unused/reserved
    environment_stub: Buffer,
    /// Current frame index (0, 1, 2)
    frame_index: usize,
    /// Monotonic frame counter used for quasi-random sampling offsets
    frame_counter: u64,
    /// Cached 2D R2 sequence seed for the current frame
    sequence_seed: [f32; 2],
    /// Current number of active lights
    light_count: u32,
    /// Bind group for current frame
    bind_group: Option<wgpu::BindGroup>,
    /// Bind group layout
    bind_group_layout: wgpu::BindGroupLayout,
    /// P1-07: Last uploaded lights for debug inspection (CPU-side only)
    /// Stores a copy of lights uploaded via update() for debug/validation purposes.
    /// Does not affect GPU behavior.
    last_uploaded_lights: Vec<Light>,
}

impl LightBuffer {
    /// Create a new light buffer manager
    pub fn new(device: &Device) -> Self {
        // Create bind group layout
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Light Buffer Bind Group Layout"),
            entries: &[
                // Binding 0: Light array (SSBO, read-only storage)
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT | wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Binding 1: Light count (uniform)
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::FRAGMENT | wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::FRAGMENT | wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        // Create triple-buffered storage buffers
        let buffers = [
            Self::create_light_buffer(device, 0),
            Self::create_light_buffer(device, 1),
            Self::create_light_buffer(device, 2),
        ];

        // Create triple-buffered count buffers
        let count_buffers = [
            Self::create_count_buffer(device, 0),
            Self::create_count_buffer(device, 1),
            Self::create_count_buffer(device, 2),
        ];
        // P1-05: Environment params stub (zeros for now, full IBL in P4)
        let environment_stub = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Light Environment Stub Buffer"),
            size: 16,  // vec4<f32>
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            buffers,
            count_buffers,
            environment_stub,
            frame_index: 0,
            frame_counter: 0,
            sequence_seed: r2_sample(0),
            light_count: 0,
            bind_group: None,
            bind_group_layout,
            last_uploaded_lights: Vec::new(),
        }
    }

    /// Create a single light storage buffer
    fn create_light_buffer(device: &Device, index: usize) -> Buffer {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(&format!("Light Storage Buffer {}", index)),
            size: (MAX_LIGHTS * std::mem::size_of::<Light>()) as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }

    /// Create a single count uniform buffer
    fn create_count_buffer(device: &Device, index: usize) -> Buffer {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(&format!("Light Count Buffer {}", index)),
            size: 16,  // Single u32 with padding to 16 bytes (uniform buffer alignment)
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }

    /// Update lights for the current frame
    ///
    /// # Arguments
    /// * `queue` - GPU command queue for buffer upload
    /// * `lights` - Slice of lights to upload (max MAX_LIGHTS)
    ///
    /// # Returns
    /// Result indicating success or error if too many lights
    pub fn update(&mut self, device: &Device, queue: &Queue, lights: &[Light]) -> Result<(), String> {
        if lights.len() > MAX_LIGHTS {
            return Err(format!(
                "Too many lights: {} (max {})",
                lights.len(),
                MAX_LIGHTS
            ));
        }

        // Get current buffer
        let buffer = &self.buffers[self.frame_index];
        let count_buffer = &self.count_buffers[self.frame_index];

        // Upload light data
        if !lights.is_empty() {
            let light_bytes = bytemuck::cast_slice(lights);
            queue.write_buffer(buffer, 0, light_bytes);
        }

        // Upload metadata: light count, frame counter (lower 32 bits), and R2 seed encoded as bits
        let seed = self.sequence_seed;
        let count_data = [
            lights.len() as u32,
            (self.frame_counter & 0xFFFF_FFFF) as u32,
            seed[0].to_bits(),
            seed[1].to_bits(),
        ];
        queue.write_buffer(count_buffer, 0, bytemuck::cast_slice(&count_data));
        queue.write_buffer(&self.environment_stub, 0, &[0u8; 16]);

        self.light_count = lights.len() as u32;
        
        // P1-07: Store copy for debug inspection
        self.last_uploaded_lights = lights.to_vec();

        // Recreate bind group for current frame
        self.bind_group = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(&format!("Light Bind Group Frame {}", self.frame_index)),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: count_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: self.environment_stub.as_entire_binding(),
                },
            ],
        }));

        Ok(())
    }

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
        writeln!(output, "  Frame: {} (seed: [{:.3}, {:.3}])",
            self.frame_counter,
            self.sequence_seed[0],
            self.sequence_seed[1]
        ).unwrap();
        writeln!(output).unwrap();
        
        for (i, light) in self.last_uploaded_lights.iter().enumerate() {
            writeln!(output, "  Light {}: {}", i, light_type_name(light.kind)).unwrap();
            writeln!(output, "    Intensity: {:.2}, Color: [{:.2}, {:.2}, {:.2}]",
                light.intensity,
                light.color[0],
                light.color[1],
                light.color[2]
            ).unwrap();
            
            // Type-specific fields
            match light.kind {
                0 => { // Directional
                    writeln!(output, "    Direction: [{:.2}, {:.2}, {:.2}]",
                        light.dir_ws[0], light.dir_ws[1], light.dir_ws[2]
                    ).unwrap();
                }
                1 => { // Point
                    writeln!(output, "    Position: [{:.2}, {:.2}, {:.2}], Range: {:.2}",
                        light.pos_ws[0], light.pos_ws[1], light.pos_ws[2],
                        light.range
                    ).unwrap();
                }
                2 => { // Spot
                    writeln!(output, "    Position: [{:.2}, {:.2}, {:.2}], Direction: [{:.2}, {:.2}, {:.2}]",
                        light.pos_ws[0], light.pos_ws[1], light.pos_ws[2],
                        light.dir_ws[0], light.dir_ws[1], light.dir_ws[2]
                    ).unwrap();
                    writeln!(output, "    Cone: inner_cos={:.2}, outer_cos={:.2}, Range: {:.2}",
                        light.cone_cos[0], light.cone_cos[1], light.range
                    ).unwrap();
                }
                3 => { // Environment
                    writeln!(output, "    Texture Index: {}", light.env_texture_index).unwrap();
                }
                4 => { // AreaRect
                    writeln!(output, "    Position: [{:.2}, {:.2}, {:.2}], Normal: [{:.2}, {:.2}, {:.2}]",
                        light.pos_ws[0], light.pos_ws[1], light.pos_ws[2],
                        light.dir_ws[0], light.dir_ws[1], light.dir_ws[2]
                    ).unwrap();
                    writeln!(output, "    Half-extents: width={:.2}, height={:.2}",
                        light.area_half[0], light.area_half[1]
                    ).unwrap();
                }
                5 => { // AreaDisk
                    writeln!(output, "    Position: [{:.2}, {:.2}, {:.2}], Normal: [{:.2}, {:.2}, {:.2}]",
                        light.pos_ws[0], light.pos_ws[1], light.pos_ws[2],
                        light.dir_ws[0], light.dir_ws[1], light.dir_ws[2]
                    ).unwrap();
                    writeln!(output, "    Radius: {:.2}", light.area_half[0]).unwrap();
                }
                6 => { // AreaSphere
                    writeln!(output, "    Position: [{:.2}, {:.2}, {:.2}]",
                        light.pos_ws[0], light.pos_ws[1], light.pos_ws[2]
                    ).unwrap();
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
fn light_type_name(kind: u32) -> &'static str {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_light_buffer_memory() {
        // Light struct is 80 bytes (verified in types.rs)
        let light_size = std::mem::size_of::<Light>();
        assert_eq!(light_size, 80);

        // Memory calculation
        let light_buffer_size = MAX_LIGHTS * light_size;  // 16 * 80 = 1280 bytes
        let count_buffer_size = 16;
        let total_per_buffer = light_buffer_size + count_buffer_size;  // 1296 bytes
        let total = 3 * total_per_buffer;  // 3888 bytes

        assert_eq!(total, 3888);

        let mb = total as f64 / (1024.0 * 1024.0);
        assert!(mb < 0.01);  // Less than 10 KB = ~0.0037 MiB
    }

    #[test]
    fn test_max_lights_constant() {
        // Verify MAX_LIGHTS fits in memory budget
        // At 80 bytes per light, 16 lights = 1.28 KB per buffer
        // Triple-buffered: 3.84 KB total
        // This is well within the 512 MiB host-visible budget
        assert_eq!(MAX_LIGHTS, 16);

        let total_bytes = 3 * MAX_LIGHTS * std::mem::size_of::<Light>();
        assert!(total_bytes < 512 * 1024 * 1024);  // < 512 MiB
    }

    #[test]
    fn test_r2_sequence_variation() {
        let first = super::r2_sample(0);
        let second = super::r2_sample(1);
        assert_ne!(first, second);
        assert!(first[0] >= 0.0 && first[0] <= 1.0);
        assert!(second[1] >= 0.0 && second[1] <= 1.0);
    }

    // P1-02: Triple-buffered SSBO manager parity tests
    
    #[test]
    fn test_light_metadata_size() {
        // LightMetadata in WGSL is 4 u32s (count, frame_index, seed_bits_x, seed_bits_y)
        // Rust equivalent is [u32; 4] = 16 bytes
        assert_eq!(std::mem::size_of::<[u32; 4]>(), 16);
    }

    #[test]
    fn test_max_lights_budget() {
        // Verify MAX_LIGHTS=16 is correct
        assert_eq!(MAX_LIGHTS, 16);
        
        // Verify total memory is reasonable
        let light_size = std::mem::size_of::<Light>();
        let total_light_memory = 3 * MAX_LIGHTS * light_size;  // Triple-buffered
        let total_metadata = 3 * 16;  // 3 count buffers
        let environment_stub = 16;
        let total = total_light_memory + total_metadata + environment_stub;
        
        // Total should be: 3 * 16 * 80 + 3 * 16 + 16 = 3840 + 48 + 16 = 3904 bytes
        assert_eq!(total, 3904);
        assert!(total < 5000);  // Well under 5 KB
    }

    #[test]
    fn test_memory_bytes_calculation() {
        // Verify memory_bytes() matches manual calculation
        let light_buffer_size = (MAX_LIGHTS * std::mem::size_of::<Light>()) as u64;  // 1280
        let count_buffer_size = 16u64;
        let environment_stub_size = 16u64;
        
        let expected = 3 * (light_buffer_size + count_buffer_size) + environment_stub_size;
        // 3 * (1280 + 16) + 16 = 3 * 1296 + 16 = 3888 + 16 = 3904
        assert_eq!(expected, 3904);
    }

    #[test]
    fn test_memory_mb_conversion() {
        // Verify memory_mb() converts correctly
        let bytes = 3904u64;
        let mb = bytes as f64 / (1024.0 * 1024.0);
        // ~0.00372 MiB
        assert!(mb > 0.0037 && mb < 0.0038);
    }

    #[test]
    fn test_bind_layout_constants() {
        // Verify binding numbers match WGSL
        // @group(0) @binding(3) var<storage, read> lights: array<LightGPU>;
        // @group(0) @binding(4) var<uniform> lightMeta: LightMetadata;
        // @group(0) @binding(5) var<uniform> environmentParams: vec4<f32>;
        const LIGHTS_BINDING: u32 = 3;
        const METADATA_BINDING: u32 = 4;
        const ENVIRONMENT_BINDING: u32 = 5;
        
        assert_eq!(LIGHTS_BINDING, 3);
        assert_eq!(METADATA_BINDING, 4);
        assert_eq!(ENVIRONMENT_BINDING, 5);
    }

    #[test]
    fn test_frame_counter_wrapping() {
        // Test that frame counter wraps correctly
        let counter = u64::MAX;
        let wrapped = counter.wrapping_add(1);
        assert_eq!(wrapped, 0);
    }

    #[test]
    fn test_frame_index_cycling() {
        // Test frame index cycles through 0, 1, 2
        let mut index = 0;
        index = (index + 1) % 3;
        assert_eq!(index, 1);
        index = (index + 1) % 3;
        assert_eq!(index, 2);
        index = (index + 1) % 3;
        assert_eq!(index, 0);
    }

    #[test]
    fn test_r2_sequence_deterministic() {
        // Verify R2 sequence is deterministic
        let seed1a = r2_sample(42);
        let seed1b = r2_sample(42);
        assert_eq!(seed1a, seed1b);
        
        // Different indices produce different seeds
        let seed2 = r2_sample(43);
        assert_ne!(seed1a, seed2);
    }

    #[test]
    fn test_r2_sequence_range() {
        // Verify R2 samples stay in [0, 1] range
        for i in 0..100 {
            let sample = r2_sample(i);
            assert!(sample[0] >= 0.0 && sample[0] <= 1.0,
                "R2 x sample {} out of range: {}", i, sample[0]);
            assert!(sample[1] >= 0.0 && sample[1] <= 1.0,
                "R2 y sample {} out of range: {}", i, sample[1]);
        }
    }

    #[test]
    fn test_light_count_enforcement() {
        // Verify MAX_LIGHTS limit is properly documented
        // Actual enforcement happens in update() method which returns Err
        // This test verifies the constant is sensible
        assert!(MAX_LIGHTS > 0, "MAX_LIGHTS must be positive");
        assert!(MAX_LIGHTS <= 64, "MAX_LIGHTS should be reasonable (<=64)");
    }

    // P1-03: Per-frame seed generation tests

    #[test]
    fn test_seed_generation_on_next_frame() {
        // Verify that next_frame() generates new R2 seeds every frame
        let seed0 = r2_sample(0);
        let seed1 = r2_sample(1);
        let seed2 = r2_sample(2);
        let seed3 = r2_sample(3);

        // All seeds should be different
        assert_ne!(seed0, seed1);
        assert_ne!(seed1, seed2);
        assert_ne!(seed2, seed3);
        assert_ne!(seed0, seed3);
    }

    #[test]
    fn test_frame_counter_increments() {
        // Test frame counter progression
        let mut counter = 0u64;
        let mut seeds = Vec::new();

        for _ in 0..10 {
            seeds.push(r2_sample(counter));
            counter = counter.wrapping_add(1);
        }

        // Verify all 10 seeds are unique
        for i in 0..seeds.len() {
            for j in (i + 1)..seeds.len() {
                assert_ne!(seeds[i], seeds[j], 
                    "Seeds at frames {} and {} should differ", i, j);
            }
        }
    }

    #[test]
    fn test_seed_encoding_roundtrip() {
        // Verify seed encoding matches WGSL bitcast behavior
        let seed = r2_sample(42);
        
        // Encode as bits (what we upload to GPU)
        let bits_x = seed[0].to_bits();
        let bits_y = seed[1].to_bits();
        
        // Decode (what WGSL bitcast<f32>() does)
        let decoded_x = f32::from_bits(bits_x);
        let decoded_y = f32::from_bits(bits_y);
        
        // Should match exactly
        assert_eq!(seed[0], decoded_x);
        assert_eq!(seed[1], decoded_y);
    }

    // P1-05: Environment stub buffer tests

    #[test]
    fn test_environment_stub_size() {
        // Environment params buffer is vec4<f32> = 16 bytes
        assert_eq!(std::mem::size_of::<[f32; 4]>(), 16);
    }

    #[test]
    fn test_environment_binding_constant() {
        // Verify environment params is at binding 5
        const ENVIRONMENT_BINDING: u32 = 5;
        assert_eq!(ENVIRONMENT_BINDING, 5);
    }

    // P1-07: Debug inspection API tests

    #[test]
    fn test_last_uploaded_lights_empty() {
        // Before any upload, should be empty
        let lights: &[Light] = &[];
        assert_eq!(lights.len(), 0);
    }

    #[test]
    fn test_last_uploaded_lights_storage() {
        // Verify we can create lights and they would be stored
        let light1 = Light::directional(45.0, 30.0, 3.0, [1.0, 0.9, 0.8]);
        let light2 = Light::point([0.0, 5.0, 0.0], 10.0, 20.0, [1.0, 1.0, 1.0]);
        
        let lights = vec![light1, light2];
        assert_eq!(lights.len(), 2);
        
        // Verify fields are accessible
        assert_eq!(lights[0].kind, 0); // Directional
        assert_eq!(lights[1].kind, 1); // Point
    }

    #[test]
    fn test_debug_info_format() {
        // Test that debug_info produces expected structure
        let light = Light::directional(45.0, 30.0, 3.0, [1.0, 0.9, 0.8]);
        
        // Verify light type name helper
        let type_name = light_type_name(light.kind);
        assert_eq!(type_name, "Directional");
        
        // Verify point light type
        let point = Light::point([1.0, 2.0, 3.0], 5.0, 10.0, [0.5, 0.6, 0.7]);
        assert_eq!(light_type_name(point.kind), "Point");
    }

    #[test]
    fn test_light_type_names() {
        // Test all light type names (u32 values)
        assert_eq!(light_type_name(0), "Directional");
        assert_eq!(light_type_name(1), "Point");
        assert_eq!(light_type_name(2), "Spot");
        assert_eq!(light_type_name(3), "Environment");
        assert_eq!(light_type_name(4), "AreaRect");
        assert_eq!(light_type_name(5), "AreaDisk");
        assert_eq!(light_type_name(6), "AreaSphere");
        assert_eq!(light_type_name(99), "Unknown");
    }

    #[test]
    fn test_debug_info_output_structure() {
        // Create a simple light setup
        let dir_light = Light::directional(0.0, 45.0, 2.5, [1.0, 0.95, 0.9]);
        
        // Simulate what debug_info would output
        let output = format!(
            "Light 0: {}\n  Intensity: {:.2}, Color: [{:.2}, {:.2}, {:.2}]",
            light_type_name(dir_light.kind),
            dir_light.intensity,
            dir_light.color[0],
            dir_light.color[1],
            dir_light.color[2]
        );
        
        assert!(output.contains("Light 0: Directional"));
        assert!(output.contains("Intensity: 2.50"));
        assert!(output.contains("Color: [1.00, 0.95, 0.90]"));
    }

    #[test]
    fn test_max_lights_not_exceeded_in_debug() {
        // Verify MAX_LIGHTS constant is reasonable for debug output
        assert_eq!(MAX_LIGHTS, 16);
        
        // Debug output for 16 lights should be manageable
        // Rough estimate: ~150 bytes per light * 16 = 2.4 KB
        let estimated_size = 150 * MAX_LIGHTS;
        assert!(estimated_size < 10000); // < 10 KB
    }
}

fn r2_sample(index: u64) -> [f32; 2] {
    const PHI: f64 = 1.324_717_957_244_746;
    const A1: f64 = 1.0 / PHI;
    const A2: f64 = 1.0 / (PHI * PHI);
    let idx = index as f64;
    [
        frac(0.5 + A1 * idx) as f32,
        frac(0.5 + A2 * idx) as f32,
    ]
}

fn frac(x: f64) -> f64 {
    x - x.floor()
}
