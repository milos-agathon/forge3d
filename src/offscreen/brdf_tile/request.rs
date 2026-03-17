use anyhow::{ensure, Result};

use super::params::{normalize_light_dir, BrdfTileOverrides, DEFAULT_LIGHT_DIR};

#[derive(Clone, Copy, Debug)]
pub(super) struct BrdfTileRenderRequest {
    pub(super) model_u32: u32,
    pub(super) roughness: f32,
    pub(super) width: u32,
    pub(super) height: u32,
    pub(super) ndf_only: bool,
    pub(super) g_only: bool,
    pub(super) dfg_only: bool,
    pub(super) spec_only: bool,
    pub(super) roughness_visualize: bool,
    pub(super) exposure: f32,
    pub(super) light_intensity: f32,
    pub(super) base_color: [f32; 3],
    pub(super) clearcoat: f32,
    pub(super) clearcoat_roughness: f32,
    pub(super) sheen: f32,
    pub(super) sheen_tint: f32,
    pub(super) specular_tint: f32,
    pub(super) debug_dot_products: bool,
    pub(super) debug_lambert_only: bool,
    pub(super) debug_diffuse_only: bool,
    pub(super) debug_d: bool,
    pub(super) debug_spec_no_nl: bool,
    pub(super) debug_energy: bool,
    pub(super) debug_angle_sweep: bool,
    pub(super) debug_angle_component: u32,
    pub(super) debug_no_srgb: bool,
    pub(super) output_mode: u32,
    pub(super) metallic_override: f32,
    pub(super) wi3_debug_mode: u32,
    pub(super) wi3_debug_roughness: f32,
    pub(super) sphere_sectors: u32,
    pub(super) sphere_stacks: u32,
    pub(super) overrides: BrdfTileOverrides,
}

#[derive(Clone, Copy, Debug)]
pub(super) struct PreparedBrdfTileRequest {
    pub(super) model_u32: u32,
    pub(super) roughness: f32,
    pub(super) width: u32,
    pub(super) height: u32,
    pub(super) ndf_only: bool,
    pub(super) g_only: bool,
    pub(super) dfg_only: bool,
    pub(super) spec_only: bool,
    pub(super) roughness_visualize: bool,
    pub(super) exposure: f32,
    pub(super) light_intensity: f32,
    pub(super) base_color: [f32; 3],
    pub(super) clearcoat: f32,
    pub(super) clearcoat_roughness: f32,
    pub(super) sheen: f32,
    pub(super) sheen_tint: f32,
    pub(super) specular_tint: f32,
    pub(super) debug_dot_products: bool,
    pub(super) debug_lambert_only: bool,
    pub(super) debug_diffuse_only: bool,
    pub(super) debug_d: bool,
    pub(super) debug_spec_no_nl: bool,
    pub(super) debug_energy: bool,
    pub(super) debug_angle_sweep: bool,
    pub(super) debug_angle_component: u32,
    pub(super) debug_no_srgb: bool,
    pub(super) output_mode: u32,
    pub(super) metallic: f32,
    pub(super) wi3_mode: u32,
    pub(super) wi3_roughness: f32,
    pub(super) sphere_sectors: u32,
    pub(super) sphere_stacks: u32,
    pub(super) light_dir: [f32; 3],
    pub(super) debug_kind: u32,
}

impl BrdfTileRenderRequest {
    pub(super) fn prepare(self) -> Result<PreparedBrdfTileRequest> {
        ensure!(
            self.width > 0 && self.height > 0,
            "tile dimensions must be positive"
        );
        ensure!(
            self.width <= 4096 && self.height <= 4096,
            "tile dimensions must be <= 4096 to avoid GPU timeouts"
        );
        ensure!(
            self.sphere_sectors >= 8 && self.sphere_sectors <= 1024,
            "sphere sectors must be in [8, 1024]"
        );
        ensure!(
            self.sphere_stacks >= 4 && self.sphere_stacks <= 512,
            "sphere stacks must be in [4, 512]"
        );
        ensure!(
            matches!(self.model_u32, 0 | 1 | 4 | 6),
            "invalid BRDF model index: {}. Allowed: 0(Lambert),1(Phong),4(GGX),6(Disney)",
            self.model_u32
        );

        let roughness = self.roughness.clamp(0.0, 1.0);
        let exposure = self.exposure.max(1e-6);
        let light_intensity = self.light_intensity.max(1e-6);
        let metallic = self.metallic_override.clamp(0.0, 1.0);
        let wi3_mode = self.wi3_debug_mode;
        let wi3_roughness = if wi3_mode != 0 {
            self.wi3_debug_roughness.clamp(0.0, 1.0)
        } else {
            roughness
        };
        let light_dir = self
            .overrides
            .light_dir
            .and_then(normalize_light_dir)
            .unwrap_or(DEFAULT_LIGHT_DIR);
        let debug_kind = match self.overrides.debug_kind.unwrap_or(0) {
            0 | 1 | 2 | 3 => self.overrides.debug_kind.unwrap_or(0),
            _ => 0,
        };

        Ok(PreparedBrdfTileRequest {
            model_u32: self.model_u32,
            roughness,
            width: self.width,
            height: self.height,
            ndf_only: self.ndf_only,
            g_only: self.g_only,
            dfg_only: self.dfg_only,
            spec_only: self.spec_only,
            roughness_visualize: self.roughness_visualize,
            exposure,
            light_intensity,
            base_color: self.base_color,
            clearcoat: self.clearcoat,
            clearcoat_roughness: self.clearcoat_roughness,
            sheen: self.sheen,
            sheen_tint: self.sheen_tint,
            specular_tint: self.specular_tint,
            debug_dot_products: self.debug_dot_products,
            debug_lambert_only: self.debug_lambert_only,
            debug_diffuse_only: self.debug_diffuse_only,
            debug_d: self.debug_d,
            debug_spec_no_nl: self.debug_spec_no_nl,
            debug_energy: self.debug_energy,
            debug_angle_sweep: self.debug_angle_sweep,
            debug_angle_component: self.debug_angle_component,
            debug_no_srgb: self.debug_no_srgb,
            output_mode: self.output_mode,
            metallic,
            wi3_mode,
            wi3_roughness,
            sphere_sectors: self.sphere_sectors,
            sphere_stacks: self.sphere_stacks,
            light_dir,
            debug_kind,
        })
    }
}

impl PreparedBrdfTileRequest {
    pub(super) fn expected_buffer_size(&self) -> usize {
        (self.height * self.width * 4) as usize
    }
}
