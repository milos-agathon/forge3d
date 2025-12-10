// src/lighting/types.rs
// P0 lighting type definitions - coordinator module
// Re-exports types from focused submodules

// Re-export all types from submodules
pub use crate::lighting::light::{Light, LightType};
pub use crate::lighting::material::{BrdfModel, MaterialShading, ShadingParamsGPU};
pub use crate::lighting::shadow::{Atmosphere, GiSettings, GiTechnique, ShadowSettings, ShadowTechnique};
pub use crate::lighting::screen_space::{
    SSAOSettings, SSGISettings, SSRSettings, ScreenSpaceEffect, ScreenSpaceSettings,
};
pub use crate::lighting::atmospherics::{
    AtmosphericsSettings, SkyModel, SkySettings, VolumetricPhase, VolumetricSettings,
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_light_sizes() {
        assert_eq!(std::mem::size_of::<Light>() % 16, 0);
        assert_eq!(std::mem::size_of::<MaterialShading>() % 16, 0);
        assert_eq!(std::mem::size_of::<ShadowSettings>() % 16, 0);
        assert_eq!(std::mem::size_of::<GiSettings>() % 16, 0);
        assert_eq!(std::mem::size_of::<Atmosphere>() % 16, 0);
        assert_eq!(std::mem::size_of::<SSAOSettings>() % 16, 0);
        assert_eq!(std::mem::size_of::<SSGISettings>() % 16, 0);
        assert_eq!(std::mem::size_of::<SSRSettings>() % 16, 0);
        assert_eq!(std::mem::size_of::<SSAOSettings>(), 32);
        assert_eq!(std::mem::size_of::<SSGISettings>(), 32);
        assert_eq!(std::mem::size_of::<SSRSettings>(), 32);
        assert_eq!(std::mem::size_of::<SkySettings>() % 16, 0);
        assert_eq!(std::mem::size_of::<VolumetricSettings>() % 16, 0);
        assert_eq!(std::mem::size_of::<SkySettings>(), 48);
        assert_eq!(std::mem::size_of::<VolumetricSettings>(), 80);
    }

    #[test]
    fn test_light_struct_size_and_alignment() {
        assert_eq!(std::mem::size_of::<Light>(), 80);
        assert_eq!(std::mem::align_of::<Light>(), 16);
    }

    #[test]
    fn test_light_field_offsets() {
        use std::mem::offset_of;
        assert_eq!(offset_of!(Light, kind), 0);
        assert_eq!(offset_of!(Light, intensity), 4);
        assert_eq!(offset_of!(Light, range), 8);
        assert_eq!(offset_of!(Light, env_texture_index), 12);
        assert_eq!(offset_of!(Light, color), 16);
        assert_eq!(offset_of!(Light, _pad1), 28);
        assert_eq!(offset_of!(Light, pos_ws), 32);
        assert_eq!(offset_of!(Light, _pad2), 44);
        assert_eq!(offset_of!(Light, dir_ws), 48);
        assert_eq!(offset_of!(Light, _pad3), 60);
        assert_eq!(offset_of!(Light, cone_cos), 64);
        assert_eq!(offset_of!(Light, area_half), 72);
    }

    #[test]
    fn test_light_type_enum_values() {
        assert_eq!(LightType::Directional.as_u32(), 0);
        assert_eq!(LightType::Point.as_u32(), 1);
        assert_eq!(LightType::Spot.as_u32(), 2);
        assert_eq!(LightType::Environment.as_u32(), 3);
        assert_eq!(LightType::AreaRect.as_u32(), 4);
        assert_eq!(LightType::AreaDisk.as_u32(), 5);
        assert_eq!(LightType::AreaSphere.as_u32(), 6);
    }

    #[test]
    fn test_light_pod_safety() {
        let light = Light::default();
        let _bytes: &[u8] = bytemuck::bytes_of(&light);
        assert_eq!(_bytes.len(), 80);
    }

    #[test]
    fn test_light_directional_unused_fields() {
        let light = Light::directional(45.0, 30.0, 2.0, [1.0, 0.9, 0.8]);
        assert_eq!(light.kind, LightType::Directional.as_u32());
        assert_eq!(light.pos_ws, [0.0; 3]);
        assert_eq!(light.range, 0.0);
    }

    #[test]
    fn test_light_environment_fields() {
        let light = Light::environment(1.5, 42);
        assert_eq!(light.kind, LightType::Environment.as_u32());
        assert_eq!(light.env_texture_index, 42);
        assert_eq!(light.intensity, 1.5);
    }

    #[test]
    fn test_light_point_fields() {
        let pos = [10.0, 20.0, 30.0];
        let light = Light::point(pos, 50.0, 3.0, [1.0, 0.5, 0.2]);
        assert_eq!(light.kind, LightType::Point.as_u32());
        assert_eq!(light.pos_ws, pos);
        assert_eq!(light.range, 50.0);
        assert_eq!(light.dir_ws, [0.0; 3]);
    }

    #[test]
    fn test_light_spot_cone_precompute() {
        let light = Light::spot([0.0, 5.0, 0.0], [0.0, -1.0, 0.0], 100.0, 20.0, 35.0, 5.0, [1.0, 1.0, 1.0]);
        assert_eq!(light.kind, LightType::Spot.as_u32());
        assert!(light.cone_cos[0] > 0.9 && light.cone_cos[0] < 1.0);
        assert!(light.cone_cos[1] > 0.8 && light.cone_cos[1] < 0.9);
    }

    #[test]
    fn test_light_area_rect_fields() {
        let light = Light::area_rect([0.0, 10.0, 0.0], [0.0, -1.0, 0.0], 2.5, 1.5, 10.0, [1.0, 1.0, 1.0]);
        assert_eq!(light.kind, LightType::AreaRect.as_u32());
        assert_eq!(light.area_half, [2.5, 1.5]);
    }

    #[test]
    fn test_light_area_disk_fields() {
        let light = Light::area_disk([5.0, 5.0, 5.0], [1.0, 0.0, 0.0], 3.0, 8.0, [1.0, 1.0, 0.8]);
        assert_eq!(light.kind, LightType::AreaDisk.as_u32());
        assert_eq!(light.area_half[0], 3.0);
        assert_eq!(light.area_half[1], 0.0);
    }

    #[test]
    fn test_light_area_sphere_fields() {
        let light = Light::area_sphere([0.0, 15.0, 0.0], 2.0, 12.0, [1.0, 0.9, 0.7]);
        assert_eq!(light.kind, LightType::AreaSphere.as_u32());
        assert_eq!(light.area_half[0], 2.0);
        assert_eq!(light.dir_ws, [0.0; 3]);
    }

    #[test]
    fn test_material_validation() {
        let mut mat = MaterialShading::default();
        assert!(mat.validate().is_ok());
        mat.roughness = 1.5;
        assert!(mat.validate().is_err());
        mat.roughness = 0.5;
        mat.metallic = -0.1;
        assert!(mat.validate().is_err());
    }

    #[test]
    fn test_shadow_validation() {
        let mut shadow = ShadowSettings::default();
        assert!(shadow.validate().is_ok());
        shadow.map_res = 5000;
        assert!(shadow.validate().is_err());
        shadow.map_res = 2047;
        assert!(shadow.validate().is_err());
    }

    #[test]
    fn test_shadow_memory_budget() {
        let shadow = ShadowSettings { map_res: 2048, ..Default::default() };
        assert_eq!(shadow.memory_budget(), 16 * 1024 * 1024);
    }
}
