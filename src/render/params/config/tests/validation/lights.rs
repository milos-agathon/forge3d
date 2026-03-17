use super::*;

#[test]
fn validation_catches_missing_direction() {
    let mut cfg = RendererConfig::default();
    if let Some(light) = cfg.lighting.lights.get_mut(0) {
        light.direction = None;
    }
    let err = cfg.validate().unwrap_err();
    assert!(err.to_string().contains("lights[0].direction required"));
}

#[test]
fn validation_point_light_requires_position() {
    let mut cfg = RendererConfig::default();
    cfg.lighting.lights = vec![LightConfig {
        light_type: LightType::Point,
        intensity: 5.0,
        color: [1.0, 1.0, 1.0],
        direction: None,
        position: None,
        cone_angle: None,
        area_extent: None,
        hdr_path: None,
    }];
    let err = cfg.validate().unwrap_err();
    assert!(err.to_string().contains("position required"));
}

#[test]
fn validation_spot_light_requires_position() {
    let mut cfg = RendererConfig::default();
    cfg.lighting.lights = vec![LightConfig {
        light_type: LightType::Spot,
        intensity: 5.0,
        color: [1.0, 1.0, 1.0],
        direction: None,
        position: None,
        cone_angle: Some(45.0),
        area_extent: None,
        hdr_path: None,
    }];
    let err = cfg.validate().unwrap_err();
    assert!(err.to_string().contains("position required"));
}

#[test]
fn validation_area_lights_require_position() {
    for light_type in [
        LightType::AreaRect,
        LightType::AreaDisk,
        LightType::AreaSphere,
    ] {
        let mut cfg = RendererConfig::default();
        cfg.lighting.lights = vec![LightConfig {
            light_type,
            intensity: 5.0,
            color: [1.0, 1.0, 1.0],
            direction: None,
            position: None,
            cone_angle: None,
            area_extent: Some([1.0, 1.0]),
            hdr_path: None,
        }];
        let err = cfg.validate().unwrap_err();
        assert!(err.to_string().contains("position required"));
    }
}

#[test]
fn validation_environment_light_requires_hdr_path() {
    let mut cfg = RendererConfig::default();
    cfg.lighting.lights = vec![LightConfig {
        light_type: LightType::Environment,
        intensity: 1.0,
        color: [1.0, 1.0, 1.0],
        direction: None,
        position: None,
        cone_angle: None,
        area_extent: None,
        hdr_path: None,
    }];
    cfg.atmosphere.hdr_path = None;
    let err = cfg.validate().unwrap_err();
    assert!(err.to_string().contains("hdr_path required"));
}

#[test]
fn validation_environment_light_accepts_atmosphere_hdr() {
    let mut cfg = RendererConfig::default();
    cfg.lighting.lights = vec![LightConfig {
        light_type: LightType::Environment,
        intensity: 1.0,
        color: [1.0, 1.0, 1.0],
        direction: None,
        position: None,
        cone_angle: None,
        area_extent: None,
        hdr_path: None,
    }];
    cfg.atmosphere.hdr_path = Some("assets/sky.hdr".to_string());
    assert!(cfg.validate().is_ok());
}

#[test]
fn validation_cone_angle_must_be_valid() {
    let mut cfg = RendererConfig::default();
    cfg.lighting.lights = vec![LightConfig {
        light_type: LightType::Spot,
        intensity: 5.0,
        color: [1.0, 1.0, 1.0],
        direction: None,
        position: Some([0.0, 0.0, 0.0]),
        cone_angle: Some(200.0),
        area_extent: None,
        hdr_path: None,
    }];
    let err = cfg.validate().unwrap_err();
    assert!(err.to_string().contains("cone_angle"));
}

#[test]
fn validation_area_extent_must_be_positive() {
    let mut cfg = RendererConfig::default();
    cfg.lighting.lights = vec![LightConfig {
        light_type: LightType::AreaRect,
        intensity: 5.0,
        color: [1.0, 1.0, 1.0],
        direction: None,
        position: Some([0.0, 0.0, 0.0]),
        cone_angle: None,
        area_extent: Some([1.0, -1.0]),
        hdr_path: None,
    }];
    let err = cfg.validate().unwrap_err();
    assert!(err.to_string().contains("area_extent"));
}
