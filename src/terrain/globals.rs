use super::*;

// T2.1 Global state
#[derive(Debug, Clone)]
pub struct Globals {
    pub sun_dir: glam::Vec3,
    pub exposure: f32,
    pub spacing: f32,
    pub h_min: f32,
    pub h_max: f32,
    pub exaggeration: f32,
    pub view_world_position: glam::Vec3,
    pub palette_index: u32, // L2: Palette selection index
    // E2: LOD morphing & skirts
    pub lod_morph: f32,     // [0..1], 1=full detail, 0=coarse
    pub coarse_factor: f32, // >=1, quantization factor for coarse sampling
    pub skirt_depth: f32,   // >=0, units to pull skirt vertices down
    pub skirt_mask: u32,    // bitmask: 1=left,2=right,4=bottom,8=top
}

impl Default for Globals {
    fn default() -> Self {
        Self {
            sun_dir: glam::Vec3::new(0.5, 0.8, 0.6).normalize(),
            exposure: 1.0,
            spacing: 1.0,
            // choose a sane range matching our analytic spike heights (~±0.5)
            h_min: -0.5,
            h_max: 0.5,
            exaggeration: 1.0,
            view_world_position: glam::Vec3::new(0.0, 0.0, 5.0), // Default camera position
            palette_index: 0,                                    // Default to first palette
            // E2 defaults: no morphing, no skirts
            lod_morph: 1.0,
            coarse_factor: 1.0,
            skirt_depth: 0.0,
            skirt_mask: 0xF, // default: enable skirts on all edges
        }
    }
}

impl Globals {
    pub fn to_uniforms(&self, view: glam::Mat4, proj: glam::Mat4) -> TerrainUniforms {
        let h_range = self.h_max - self.h_min;
        let mut u = TerrainUniforms::new_with_palette(
            view,
            proj,
            self.sun_dir,
            self.exposure,
            self.spacing,
            h_range,
            self.exaggeration,
            self.palette_index,
        );
        // E2: write morphing/skirts into tail padding [X=morph, Y=coarse_factor, Z=skirt_depth, W=0]
        u._pad_tail = [
            self.lod_morph,
            self.coarse_factor.max(1.0),
            self.skirt_depth.max(0.0),
            self.skirt_mask as f32,
        ];
        u
    }
}
