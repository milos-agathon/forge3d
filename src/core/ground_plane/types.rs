use glam::{Mat4, Vec3};

/// Ground plane rendering modes
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum GroundPlaneMode {
    Disabled,     // No ground plane
    Solid,        // Solid color ground plane
    Grid,         // Grid pattern with major/minor lines
    CheckerBoard, // Checkerboard pattern (future extension)
}

/// Ground plane configuration parameters
#[derive(Debug, Clone)]
pub struct GroundPlaneParams {
    pub mode: GroundPlaneMode,
    pub size: f32,   // Size of the ground plane
    pub height: f32, // Y position of the ground plane
    pub z_bias: f32, // Z-bias to prevent z-fighting

    // Grid parameters
    pub major_spacing: f32, // Spacing between major grid lines
    pub minor_spacing: f32, // Spacing between minor grid lines
    pub major_width: f32,   // Width of major grid lines
    pub minor_width: f32,   // Width of minor grid lines

    // Colors
    pub albedo: Vec3,           // Base ground color (RGB)
    pub alpha: f32,             // Base ground alpha
    pub major_grid_color: Vec3, // Major grid line color
    pub major_grid_alpha: f32,  // Major grid line alpha
    pub minor_grid_color: Vec3, // Minor grid line color
    pub minor_grid_alpha: f32,  // Minor grid line alpha

    // Fading
    pub fade_distance: f32, // Distance at which ground plane starts fading
    pub fade_power: f32,    // Power curve for ground plane fading
    pub grid_fade_distance: f32, // Distance at which grid lines start fading
    pub grid_fade_power: f32, // Power curve for grid line fading
}

impl Default for GroundPlaneParams {
    fn default() -> Self {
        Self {
            mode: GroundPlaneMode::Grid,
            size: 1000.0,   // Large ground plane
            height: 0.0,    // At ground level
            z_bias: 0.0001, // Small bias to prevent z-fighting

            // Grid settings - metric-style grid
            major_spacing: 10.0, // 10 unit major grid
            minor_spacing: 1.0,  // 1 unit minor grid
            major_width: 2.0,    // Moderate major line width
            minor_width: 1.0,    // Thin minor lines

            // Colors - neutral grid
            albedo: Vec3::new(0.2, 0.2, 0.2), // Dark gray ground
            alpha: 0.8,
            major_grid_color: Vec3::new(0.6, 0.6, 0.6), // Light gray major lines
            major_grid_alpha: 0.8,
            minor_grid_color: Vec3::new(0.4, 0.4, 0.4), // Medium gray minor lines
            minor_grid_alpha: 0.4,

            // Fading to prevent grid noise at distance
            fade_distance: 500.0,      // Start fading ground at 500 units
            fade_power: 2.0,           // Quadratic falloff
            grid_fade_distance: 200.0, // Fade grid lines earlier
            grid_fade_power: 1.5,      // Slightly less aggressive fade
        }
    }
}

impl GroundPlaneParams {
    pub fn engineering_grid() -> Self {
        Self {
            mode: GroundPlaneMode::Grid,
            major_spacing: 10.0,
            minor_spacing: 1.0,
            major_grid_color: Vec3::new(0.0, 0.8, 0.0), // Green major lines
            minor_grid_color: Vec3::new(0.0, 0.4, 0.0), // Dark green minor lines
            albedo: Vec3::new(0.05, 0.05, 0.05),        // Nearly black background
            ..Default::default()
        }
    }

    pub fn architectural_grid() -> Self {
        Self {
            mode: GroundPlaneMode::Grid,
            major_spacing: 5.0, // 5 meter grid for architecture
            minor_spacing: 1.0, // 1 meter subdivisions
            major_grid_color: Vec3::new(0.6, 0.6, 0.8), // Light blue major lines
            minor_grid_color: Vec3::new(0.4, 0.4, 0.6), // Darker blue minor lines
            albedo: Vec3::new(0.9, 0.9, 0.95), // Light background
            major_width: 1.5,
            minor_width: 0.8,
            ..Default::default()
        }
    }

    pub fn simple_ground() -> Self {
        Self {
            mode: GroundPlaneMode::Solid,
            albedo: Vec3::new(0.3, 0.25, 0.2), // Brown earth color
            alpha: 1.0,
            ..Default::default()
        }
    }
}

/// Ground plane uniforms structure (must match WGSL exactly)
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GroundPlaneUniforms {
    pub view_proj: [[f32; 4]; 4],       // 64 bytes - View-projection matrix
    pub world_transform: [[f32; 4]; 4], // 64 bytes - World transformation matrix
    pub plane_params: [f32; 4], // 16 bytes - size (x), height (y), grid_enabled (z), z_bias (w)
    pub grid_params: [f32; 4], // 16 bytes - major_spacing (x), minor_spacing (y), major_width (z), minor_width (w)
    pub color_params: [f32; 4], // 16 bytes - albedo (rgb) + alpha (w)
    pub grid_color_params: [f32; 4], // 16 bytes - major_grid_color (rgb) + major_alpha (w)
    pub minor_grid_color_params: [f32; 4], // 16 bytes - minor_grid_color (rgb) + minor_alpha (w)
    pub fade_params: [f32; 4], // 16 bytes - fade_distance (x), fade_power (y), grid_fade_distance (z), grid_fade_power (w)
}

impl Default for GroundPlaneUniforms {
    fn default() -> Self {
        let params = GroundPlaneParams::default();
        let mut uniforms = Self {
            view_proj: Mat4::IDENTITY.to_cols_array_2d(),
            world_transform: Mat4::IDENTITY.to_cols_array_2d(),
            plane_params: [params.size, params.height, 1.0, params.z_bias], // grid enabled
            grid_params: [
                params.major_spacing,
                params.minor_spacing,
                params.major_width,
                params.minor_width,
            ],
            color_params: [
                params.albedo.x,
                params.albedo.y,
                params.albedo.z,
                params.alpha,
            ],
            grid_color_params: [
                params.major_grid_color.x,
                params.major_grid_color.y,
                params.major_grid_color.z,
                params.major_grid_alpha,
            ],
            minor_grid_color_params: [
                params.minor_grid_color.x,
                params.minor_grid_color.y,
                params.minor_grid_color.z,
                params.minor_grid_alpha,
            ],
            fade_params: [
                params.fade_distance,
                params.fade_power,
                params.grid_fade_distance,
                params.grid_fade_power,
            ],
        };

        // Set world transform to position the ground plane
        uniforms.world_transform =
            Mat4::from_translation(Vec3::new(0.0, params.height, 0.0)).to_cols_array_2d();

        uniforms
    }
}

impl GroundPlaneUniforms {
    pub fn update_from_params(&mut self, params: &GroundPlaneParams) {
        // Update plane parameters
        self.plane_params = [
            params.size,
            params.height,
            if params.mode == GroundPlaneMode::Grid {
                1.0
            } else {
                0.0
            }, // grid_enabled
            params.z_bias,
        ];

        // Update grid parameters
        self.grid_params = [
            params.major_spacing,
            params.minor_spacing,
            params.major_width,
            params.minor_width,
        ];

        // Update color parameters
        self.color_params = [
            params.albedo.x,
            params.albedo.y,
            params.albedo.z,
            params.alpha,
        ];

        self.grid_color_params = [
            params.major_grid_color.x,
            params.major_grid_color.y,
            params.major_grid_color.z,
            params.major_grid_alpha,
        ];

        self.minor_grid_color_params = [
            params.minor_grid_color.x,
            params.minor_grid_color.y,
            params.minor_grid_color.z,
            params.minor_grid_alpha,
        ];

        // Update fade parameters
        self.fade_params = [
            params.fade_distance,
            params.fade_power,
            params.grid_fade_distance,
            params.grid_fade_power,
        ];

        // Update world transform
        self.world_transform =
            Mat4::from_translation(Vec3::new(0.0, params.height, 0.0)).to_cols_array_2d();
    }
}
