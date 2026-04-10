use super::*;

#[test]
fn lighting_params_has_correct_defaults() {
    let params = LightingParams::default();
    assert_eq!(params.exposure, 1.0);
    assert_eq!(params.lights.len(), 1);
    assert_eq!(params.lights[0].light_type, LightType::Directional);
}

#[test]
fn shading_params_has_correct_defaults() {
    let params = ShadingParams::default();
    assert_eq!(params.brdf, BrdfModel::CookTorranceGGX);
    assert!(params.normal_maps);
    assert!(!params.clearcoat);
}

#[test]
fn shadow_params_has_correct_defaults() {
    let params = ShadowParams::default();
    assert!(params.enabled);
    assert_eq!(params.technique, ShadowTechnique::Pcf);
    assert_eq!(params.map_size, 2048);
    assert_eq!(params.cascades, 4);
    assert!(params.is_power_of_two_map());
}

#[test]
fn gi_params_has_correct_defaults() {
    let params = GiParams::default();
    assert_eq!(params.modes, vec![GiMode::None]);
    assert_eq!(params.ambient_occlusion_strength, 1.0);
}

#[test]
fn atmosphere_params_has_correct_defaults() {
    let params = AtmosphereParams::default();
    assert!(params.enabled);
    assert_eq!(params.sky, SkyModel::HosekWilkie);
    assert!(params.hdr_path.is_none());
    assert!(params.volumetric.is_none());
}

#[test]
fn renderer_config_has_correct_defaults() {
    let config = RendererConfig::default();
    assert!(!config.lighting.lights.is_empty());
    assert_eq!(config.shading.brdf, BrdfModel::CookTorranceGGX);
    assert!(config.shadows.enabled);
    assert_eq!(config.gi.modes, vec![GiMode::None]);
    assert!(config.atmosphere.enabled);
}
