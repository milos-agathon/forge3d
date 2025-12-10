use bytemuck::{Pod, Zeroable};

/// Light types for point and spot lights
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LightType {
    Point = 0,
    Spot = 1,
}

impl Default for LightType {
    fn default() -> Self {
        Self::Point
    }
}

/// Shadow quality levels
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShadowQuality {
    Off = 0,
    Low = 1,
    Medium = 2,
    High = 3,
}

impl Default for ShadowQuality {
    fn default() -> Self {
        Self::Medium
    }
}

/// Debug visualization modes
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DebugMode {
    Normal = 0,
    ShowLightBounds = 1,
    ShowShadows = 2,
}

impl Default for DebugMode {
    fn default() -> Self {
        Self::Normal
    }
}

/// Individual light data (matches WGSL layout exactly - 64 bytes)
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct Light {
    // Position and type (16 bytes)
    pub position: [f32; 3],
    pub light_type: u32, // LightType as u32

    // Direction and range (16 bytes)
    pub direction: [f32; 3], // For spot lights (normalized)
    pub range: f32,          // Maximum light distance

    // Color and intensity (16 bytes)
    pub color: [f32; 3],
    pub intensity: f32,

    // Spot light parameters (16 bytes)
    pub inner_cone_angle: f32,  // Inner cone angle (radians)
    pub outer_cone_angle: f32,  // Outer cone angle (radians)
    pub penumbra_softness: f32, // Penumbra transition softness (0.1-2.0)
    pub shadow_enabled: f32,    // 0.0 = disabled, 1.0 = enabled
}

impl Default for Light {
    fn default() -> Self {
        Self {
            position: [0.0, 5.0, 0.0],
            light_type: LightType::Point as u32,
            direction: [0.0, -1.0, 0.0], // Pointing down
            range: 20.0,
            color: [1.0, 1.0, 1.0],
            intensity: 1.0,
            inner_cone_angle: 30.0f32.to_radians(),
            outer_cone_angle: 45.0f32.to_radians(),
            penumbra_softness: 1.0,
            shadow_enabled: 1.0,
        }
    }
}

impl Light {
    /// Create a new point light
    pub fn point(position: [f32; 3], color: [f32; 3], intensity: f32, range: f32) -> Self {
        Self {
            position,
            light_type: LightType::Point as u32,
            direction: [0.0, -1.0, 0.0], // Direction not used for point lights
            range,
            color,
            intensity,
            inner_cone_angle: 0.0,
            outer_cone_angle: 0.0,
            penumbra_softness: 1.0,
            shadow_enabled: 1.0,
        }
    }

    /// Create a new spot light
    pub fn spot(
        position: [f32; 3],
        direction: [f32; 3],
        color: [f32; 3],
        intensity: f32,
        range: f32,
        inner_cone_deg: f32,
        outer_cone_deg: f32,
        penumbra_softness: f32,
    ) -> Self {
        // Normalize direction
        let dir_len = (direction[0] * direction[0]
            + direction[1] * direction[1]
            + direction[2] * direction[2])
            .sqrt();
        let normalized_dir = if dir_len > 0.0 {
            [
                direction[0] / dir_len,
                direction[1] / dir_len,
                direction[2] / dir_len,
            ]
        } else {
            [0.0, -1.0, 0.0]
        };

        Self {
            position,
            light_type: LightType::Spot as u32,
            direction: normalized_dir,
            range,
            color,
            intensity,
            inner_cone_angle: inner_cone_deg.to_radians(),
            outer_cone_angle: outer_cone_deg.to_radians(),
            penumbra_softness: penumbra_softness.clamp(0.1, 5.0),
            shadow_enabled: 1.0,
        }
    }

    /// Set light position
    pub fn set_position(&mut self, position: [f32; 3]) {
        self.position = position;
    }

    /// Set light direction (for spot lights)
    pub fn set_direction(&mut self, direction: [f32; 3]) {
        let dir_len = (direction[0] * direction[0]
            + direction[1] * direction[1]
            + direction[2] * direction[2])
            .sqrt();
        if dir_len > 0.0 {
            self.direction = [
                direction[0] / dir_len,
                direction[1] / dir_len,
                direction[2] / dir_len,
            ];
        }
    }

    /// Set light color
    pub fn set_color(&mut self, color: [f32; 3]) {
        self.color = color;
    }

    /// Set light intensity
    pub fn set_intensity(&mut self, intensity: f32) {
        self.intensity = intensity.max(0.0);
    }

    /// Set light range
    pub fn set_range(&mut self, range: f32) {
        self.range = range.max(0.1);
    }

    /// Set spot light cone angles
    pub fn set_cone_angles(&mut self, inner_deg: f32, outer_deg: f32) {
        let inner_rad = inner_deg.to_radians().clamp(0.0, std::f32::consts::PI);
        let outer_rad = outer_deg
            .to_radians()
            .clamp(inner_rad, std::f32::consts::PI);
        self.inner_cone_angle = inner_rad;
        self.outer_cone_angle = outer_rad;
    }

    /// Set penumbra softness
    pub fn set_penumbra_softness(&mut self, softness: f32) {
        self.penumbra_softness = softness.clamp(0.1, 5.0);
    }

    /// Enable or disable shadows for this light
    pub fn set_shadow_enabled(&mut self, enabled: bool) {
        self.shadow_enabled = if enabled { 1.0 } else { 0.0 };
    }

    /// Check if point is within light range
    pub fn affects_point(&self, point: [f32; 3]) -> bool {
        let dx = point[0] - self.position[0];
        let dy = point[1] - self.position[1];
        let dz = point[2] - self.position[2];
        let distance_sq = dx * dx + dy * dy + dz * dz;
        distance_sq <= self.range * self.range
    }
}

/// Per-frame uniforms (matches WGSL layout - 128 bytes total)
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct PointSpotLightUniforms {
    // Camera and view parameters (64 bytes)
    pub view_matrix: [[f32; 4]; 4],
    pub proj_matrix: [[f32; 4]; 4],

    // Global lighting (16 bytes)
    pub ambient_color: [f32; 3],
    pub ambient_intensity: f32,

    // Light count and control (16 bytes)
    pub active_light_count: u32,
    pub max_lights: u32,
    pub shadow_quality: u32, // ShadowQuality as u32
    pub debug_mode: u32,     // DebugMode as u32

    // Global shadow parameters (16 bytes)
    pub shadow_bias: f32,
    pub shadow_normal_bias: f32,
    pub shadow_softness: f32,
    pub _pad0: f32,

    // Additional padding to reach 128 bytes (16 bytes)
    pub _pad1: [f32; 4],
}

impl Default for PointSpotLightUniforms {
    fn default() -> Self {
        Self {
            view_matrix: glam::Mat4::IDENTITY.to_cols_array_2d(),
            proj_matrix: glam::Mat4::IDENTITY.to_cols_array_2d(),
            ambient_color: [0.2, 0.2, 0.3],
            ambient_intensity: 0.3,
            active_light_count: 0,
            max_lights: 32,
            shadow_quality: ShadowQuality::Medium as u32,
            debug_mode: DebugMode::Normal as u32,
            shadow_bias: 0.001,
            shadow_normal_bias: 0.01,
            shadow_softness: 0.5,
            _pad0: 0.0,
            _pad1: [0.0; 4],
        }
    }
}
