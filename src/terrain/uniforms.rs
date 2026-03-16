// ---------- Uniforms (std140-compatible, 176 bytes to match WGSL) ----------

// Updated TerrainUniforms to match WGSL shader layout (176 bytes)
#[repr(C, align(16))]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct TerrainUniforms {
    pub view: [[f32; 4]; 4],          // 64 B
    pub proj: [[f32; 4]; 4],          // 64 B
    pub sun_exposure: [f32; 4],       // (sun_dir.x, sun_dir.y, sun_dir.z, exposure) (16 B)
    pub spacing_h_exag_pad: [f32; 4], // (dx, dy, h_exag, palette_index) (16 B)
    pub _pad_tail: [f32; 4],          // keep total 176 bytes (16 B)
}

// Compile-time size and alignment checks
const _: () = assert!(::std::mem::size_of::<TerrainUniforms>() == 176);
const _: () = assert!(::std::mem::align_of::<TerrainUniforms>() == 16);

impl TerrainUniforms {
    /// Note: `h_range = max - min` (pass a single range, not min/max separately)
    pub fn new(
        view: glam::Mat4,
        proj: glam::Mat4,
        sun_dir: glam::Vec3,
        exposure: f32,
        spacing: f32,
        h_range: f32,
        exaggeration: f32,
    ) -> Self {
        Self {
            view: view.to_cols_array_2d(),
            proj: proj.to_cols_array_2d(),
            sun_exposure: [sun_dir.x, sun_dir.y, sun_dir.z, exposure],
            spacing_h_exag_pad: [spacing, h_range, exaggeration, 0.0], // Default palette_index = 0
            _pad_tail: [0.0; 4], // Initialize tail padding to zero
        }
    }

    /// Create TerrainUniforms with explicit palette selection
    pub fn new_with_palette(
        view: glam::Mat4,
        proj: glam::Mat4,
        sun_dir: glam::Vec3,
        exposure: f32,
        spacing: f32,
        h_range: f32,
        exaggeration: f32,
        palette_index: u32,
    ) -> Self {
        Self {
            view: view.to_cols_array_2d(),
            proj: proj.to_cols_array_2d(),
            sun_exposure: [sun_dir.x, sun_dir.y, sun_dir.z, exposure],
            spacing_h_exag_pad: [spacing, h_range, exaggeration, palette_index as f32],
            _pad_tail: [0.0; 4], // Initialize tail padding to zero
        }
    }

    pub fn from_mvp_legacy(
        mvp: glam::Mat4,
        light: glam::Vec3,
        h_min: f32,
        h_max: f32,
        exaggeration: f32,
    ) -> Self {
        let view = glam::Mat4::IDENTITY;
        let h_range = h_max - h_min;
        Self::new(view, mvp, light, 1.0, 1.0, h_range, exaggeration)
    }

    pub fn for_rendering(
        view_matrix: glam::Mat4,
        proj_matrix: glam::Mat4,
        sun_direction: glam::Vec3,
        exposure: f32,
        terrain_spacing: f32,
        height_range: f32,
        height_exaggeration: f32,
    ) -> Self {
        Self::new(
            view_matrix,
            proj_matrix,
            sun_direction,
            exposure,
            terrain_spacing,
            height_range,
            height_exaggeration,
        )
    }

    /// T31: Pack the first 44 floats in the expected T31 lanes order for debug tests
    /// Layout: [0..15]=view, [16..31]=proj, [32..35]=sun_exposure, [36..39]=spacing/h_range/exag/0, [40..43]=pad
    /// Matrices are stored in column-major format (compatible with WGSL/GPU layout)
    pub fn to_debug_lanes_44(&self) -> [f32; 44] {
        let mut lanes = [0.0f32; 44];

        // [0..15] = view matrix (16 floats, keep in column-major format as stored)
        for col in 0..4 {
            for row in 0..4 {
                lanes[col * 4 + row] = self.view[col][row];
            }
        }

        // [16..31] = proj matrix (16 floats, keep in column-major format as stored)
        for col in 0..4 {
            for row in 0..4 {
                lanes[16 + col * 4 + row] = self.proj[col][row];
            }
        }

        // [32..35] = sun_exposure (4 floats: sun_dir.xyz, exposure)
        lanes[32..36].copy_from_slice(&self.sun_exposure);

        // [36..39] = spacing_h_exag_pad (4 floats: spacing, h_range, exaggeration, 0)
        lanes[36..40].copy_from_slice(&self.spacing_h_exag_pad);

        // [40..43] = _pad_tail (4 floats, zeroed)
        lanes[40..44].copy_from_slice(&self._pad_tail);

        lanes
    }
}
