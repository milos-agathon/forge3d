use super::*;

#[test]
fn gi_ibl_requires_environment_light_or_atmosphere_hdr() {
    let mut cfg = RendererConfig::default();
    cfg.gi.modes = vec![GiMode::Ibl];
    cfg.lighting.lights = vec![];
    cfg.atmosphere.hdr_path = None;
    let err = cfg.validate().unwrap_err();
    assert!(err.to_string().contains("ibl"));
}

#[test]
fn gi_ibl_accepts_environment_light() {
    let mut cfg = RendererConfig::default();
    cfg.gi.modes = vec![GiMode::Ibl];
    cfg.lighting.lights = vec![LightConfig {
        light_type: LightType::Environment,
        intensity: 1.0,
        color: [1.0, 1.0, 1.0],
        direction: None,
        position: None,
        cone_angle: None,
        area_extent: None,
        hdr_path: Some("assets/sky.hdr".to_string()),
    }];
    assert!(cfg.validate().is_ok());
}

#[test]
fn gi_ibl_accepts_atmosphere_hdr() {
    let mut cfg = RendererConfig::default();
    cfg.gi.modes = vec![GiMode::Ibl];
    cfg.atmosphere.hdr_path = Some("assets/sky.hdr".to_string());
    cfg.lighting.lights = vec![LightConfig::default()];
    assert!(cfg.validate().is_ok());
}
