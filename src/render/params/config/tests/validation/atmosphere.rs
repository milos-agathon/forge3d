use super::*;

#[test]
fn atmosphere_hdri_sky_requires_hdr_path() {
    let mut cfg = RendererConfig::default();
    cfg.atmosphere.sky = SkyModel::Hdri;
    cfg.atmosphere.hdr_path = None;
    cfg.lighting.lights = vec![];
    let err = cfg.validate().unwrap_err();
    assert!(err.to_string().contains("sky=hdri"));
}

#[test]
fn atmosphere_hdri_sky_accepts_environment_light() {
    let mut cfg = RendererConfig::default();
    cfg.atmosphere.sky = SkyModel::Hdri;
    cfg.atmosphere.hdr_path = None;
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
fn volumetric_density_must_be_non_negative() {
    let mut cfg = RendererConfig::default();
    cfg.atmosphere.volumetric = Some(VolumetricParams {
        density: -0.1,
        ..VolumetricParams::default()
    });
    let err = cfg.validate().unwrap_err();
    assert!(err.to_string().contains("density"));
}

#[test]
fn volumetric_hg_anisotropy_must_be_in_range() {
    let mut cfg = RendererConfig::default();
    cfg.atmosphere.volumetric = Some(VolumetricParams {
        phase: VolumetricPhase::HenyeyGreenstein,
        anisotropy: 1.5,
        ..VolumetricParams::default()
    });
    let err = cfg.validate().unwrap_err();
    assert!(err.to_string().contains("anisotropy"));

    cfg.atmosphere.volumetric = Some(VolumetricParams {
        phase: VolumetricPhase::HenyeyGreenstein,
        anisotropy: -1.5,
        ..VolumetricParams::default()
    });
    let err = cfg.validate().unwrap_err();
    assert!(err.to_string().contains("anisotropy"));
}

#[test]
fn volumetric_hg_anisotropy_boundary_values() {
    let mut cfg = RendererConfig::default();
    cfg.atmosphere.volumetric = Some(VolumetricParams {
        phase: VolumetricPhase::HenyeyGreenstein,
        anisotropy: 0.999,
        ..VolumetricParams::default()
    });
    assert!(cfg.validate().is_ok());

    cfg.atmosphere.volumetric = Some(VolumetricParams {
        phase: VolumetricPhase::HenyeyGreenstein,
        anisotropy: -0.999,
        ..VolumetricParams::default()
    });
    assert!(cfg.validate().is_ok());
}
