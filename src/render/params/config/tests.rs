use super::*;
use super::*;

#[test]
fn default_config_serializes_and_validates() {
    let cfg = RendererConfig::default();
    let json = serde_json::to_string(&cfg).expect("serialize default config");
    let de: RendererConfig = serde_json::from_str(&json).expect("deserialize default config");
    de.validate().expect("default config should validate");
}

#[test]
fn parse_enums_from_strings() {
    assert_eq!(
        "directional".parse::<LightType>().unwrap(),
        LightType::Directional
    );
    assert_eq!(
        "cooktorrance-ggx".parse::<BrdfModel>().unwrap(),
        BrdfModel::CookTorranceGGX
    );
    assert_eq!(
        "pcf".parse::<ShadowTechnique>().unwrap(),
        ShadowTechnique::Pcf
    );
    assert_eq!("ssao".parse::<GiMode>().unwrap(), GiMode::Ssao);
    assert_eq!(
        "hosek-wilkie".parse::<SkyModel>().unwrap(),
        SkyModel::HosekWilkie
    );
    assert_eq!(
        "hg".parse::<VolumetricPhase>().unwrap(),
        VolumetricPhase::HenyeyGreenstein
    );
}

// P0-01: Comprehensive enum coverage and round-trip tests
#[test]
fn light_type_all_variants_parse_and_round_trip() {
    let cases = vec![
        (LightType::Directional, "directional"),
        (LightType::Point, "point"),
        (LightType::Spot, "spot"),
        (LightType::AreaRect, "area-rect"),
        (LightType::AreaDisk, "area-disk"),
        (LightType::AreaSphere, "area-sphere"),
        (LightType::Environment, "environment"),
    ];
    for (variant, canonical) in cases {
        // Test canonical name matches
        assert_eq!(variant.canonical(), canonical);
        // Test parsing canonical returns correct variant
        assert_eq!(canonical.parse::<LightType>().unwrap(), variant);
        // Test round-trip: variant -> canonical -> parse -> variant
        let round_trip = variant.canonical().parse::<LightType>().unwrap();
        assert_eq!(round_trip, variant);
    }
}

#[test]
fn light_type_aliases_parse_correctly() {
    assert_eq!("dir".parse::<LightType>().unwrap(), LightType::Directional);
    assert_eq!("sun".parse::<LightType>().unwrap(), LightType::Directional);
    assert_eq!("pointlight".parse::<LightType>().unwrap(), LightType::Point);
    assert_eq!("spotlight".parse::<LightType>().unwrap(), LightType::Spot);
    assert_eq!("rect".parse::<LightType>().unwrap(), LightType::AreaRect);
    assert_eq!("disk".parse::<LightType>().unwrap(), LightType::AreaDisk);
    assert_eq!(
        "sphere".parse::<LightType>().unwrap(),
        LightType::AreaSphere
    );
    assert_eq!("env".parse::<LightType>().unwrap(), LightType::Environment);
    assert_eq!("hdri".parse::<LightType>().unwrap(), LightType::Environment);
}

#[test]
fn brdf_model_all_variants_parse_and_round_trip() {
    let cases = vec![
        (BrdfModel::Lambert, "lambert"),
        (BrdfModel::Phong, "phong"),
        (BrdfModel::BlinnPhong, "blinn-phong"),
        (BrdfModel::OrenNayar, "oren-nayar"),
        (BrdfModel::CookTorranceGGX, "cooktorrance-ggx"),
        (BrdfModel::CookTorranceBeckmann, "cooktorrance-beckmann"),
        (BrdfModel::DisneyPrincipled, "disney-principled"),
        (BrdfModel::AshikhminShirley, "ashikhmin-shirley"),
        (BrdfModel::Ward, "ward"),
        (BrdfModel::Toon, "toon"),
        (BrdfModel::Minnaert, "minnaert"),
        (BrdfModel::Subsurface, "subsurface"),
        (BrdfModel::Hair, "hair"),
    ];
    for (variant, canonical) in cases {
        assert_eq!(variant.canonical(), canonical);
        assert_eq!(canonical.parse::<BrdfModel>().unwrap(), variant);
        let round_trip = variant.canonical().parse::<BrdfModel>().unwrap();
        assert_eq!(round_trip, variant);
    }
}

#[test]
fn brdf_model_aliases_parse_correctly() {
    assert_eq!(
        "blinnphong".parse::<BrdfModel>().unwrap(),
        BrdfModel::BlinnPhong
    );
    assert_eq!(
        "orennayar".parse::<BrdfModel>().unwrap(),
        BrdfModel::OrenNayar
    );
    assert_eq!(
        "ggx".parse::<BrdfModel>().unwrap(),
        BrdfModel::CookTorranceGGX
    );
    assert_eq!(
        "beckmann".parse::<BrdfModel>().unwrap(),
        BrdfModel::CookTorranceBeckmann
    );
    assert_eq!(
        "disney".parse::<BrdfModel>().unwrap(),
        BrdfModel::DisneyPrincipled
    );
    assert_eq!(
        "ashikhminshirley".parse::<BrdfModel>().unwrap(),
        BrdfModel::AshikhminShirley
    );
    assert_eq!("sss".parse::<BrdfModel>().unwrap(), BrdfModel::Subsurface);
    assert_eq!("kajiyakay".parse::<BrdfModel>().unwrap(), BrdfModel::Hair);
    assert_eq!("kajiya-kay".parse::<BrdfModel>().unwrap(), BrdfModel::Hair);
}

#[test]
fn shadow_technique_all_variants_parse_and_round_trip() {
    let cases = vec![
        (ShadowTechnique::Hard, "hard"),
        (ShadowTechnique::Pcf, "pcf"),
        (ShadowTechnique::Pcss, "pcss"),
        (ShadowTechnique::Vsm, "vsm"),
        (ShadowTechnique::Evsm, "evsm"),
        (ShadowTechnique::Msm, "msm"),
        (ShadowTechnique::Csm, "csm"),
    ];
    for (variant, canonical) in cases {
        assert_eq!(variant.canonical(), canonical);
        assert_eq!(canonical.parse::<ShadowTechnique>().unwrap(), variant);
        let round_trip = variant.canonical().parse::<ShadowTechnique>().unwrap();
        assert_eq!(round_trip, variant);
    }
}

#[test]
fn gi_mode_all_variants_parse_and_round_trip() {
    let cases = vec![
        (GiMode::None, "none"),
        (GiMode::Ibl, "ibl"),
        (GiMode::IrradianceProbes, "irradiance-probes"),
        (GiMode::Ddgi, "ddgi"),
        (GiMode::VoxelConeTracing, "voxel-cone-tracing"),
        (GiMode::Ssao, "ssao"),
        (GiMode::Gtao, "gtao"),
        (GiMode::Ssgi, "ssgi"),
        (GiMode::Ssr, "ssr"),
    ];
    for (variant, canonical) in cases {
        assert_eq!(variant.canonical(), canonical);
        assert_eq!(canonical.parse::<GiMode>().unwrap(), variant);
        let round_trip = variant.canonical().parse::<GiMode>().unwrap();
        assert_eq!(round_trip, variant);
    }
}

#[test]
fn gi_mode_aliases_parse_correctly() {
    assert_eq!(
        "irradianceprobes".parse::<GiMode>().unwrap(),
        GiMode::IrradianceProbes
    );
    assert_eq!(
        "probes".parse::<GiMode>().unwrap(),
        GiMode::IrradianceProbes
    );
    assert_eq!(
        "voxelconetracing".parse::<GiMode>().unwrap(),
        GiMode::VoxelConeTracing
    );
    assert_eq!("vct".parse::<GiMode>().unwrap(), GiMode::VoxelConeTracing);
}

#[test]
fn sky_model_all_variants_parse_and_round_trip() {
    let cases = vec![
        (SkyModel::HosekWilkie, "hosek-wilkie"),
        (SkyModel::Preetham, "preetham"),
        (SkyModel::Hdri, "hdri"),
    ];
    for (variant, canonical) in cases {
        assert_eq!(variant.canonical(), canonical);
        assert_eq!(canonical.parse::<SkyModel>().unwrap(), variant);
        let round_trip = variant.canonical().parse::<SkyModel>().unwrap();
        assert_eq!(round_trip, variant);
    }
}

#[test]
fn sky_model_aliases_parse_correctly() {
    assert_eq!(
        "hosekwilkie".parse::<SkyModel>().unwrap(),
        SkyModel::HosekWilkie
    );
    assert_eq!("environment".parse::<SkyModel>().unwrap(), SkyModel::Hdri);
    assert_eq!("envmap".parse::<SkyModel>().unwrap(), SkyModel::Hdri);
}

#[test]
fn volumetric_phase_all_variants_parse_and_round_trip() {
    let cases = vec![
        (VolumetricPhase::Isotropic, "isotropic"),
        (VolumetricPhase::HenyeyGreenstein, "henyey-greenstein"),
    ];
    for (variant, canonical) in cases {
        assert_eq!(variant.canonical(), canonical);
        assert_eq!(canonical.parse::<VolumetricPhase>().unwrap(), variant);
        let round_trip = variant.canonical().parse::<VolumetricPhase>().unwrap();
        assert_eq!(round_trip, variant);
    }
}

#[test]
fn volumetric_phase_aliases_parse_correctly() {
    assert_eq!(
        "henyeygreenstein".parse::<VolumetricPhase>().unwrap(),
        VolumetricPhase::HenyeyGreenstein
    );
    assert_eq!(
        "hg".parse::<VolumetricPhase>().unwrap(),
        VolumetricPhase::HenyeyGreenstein
    );
}

#[test]
fn volumetric_mode_all_variants_parse() {
    let cases = vec![
        (VolumetricMode::Raymarch, "raymarch"),
        (VolumetricMode::Froxels, "froxels"),
    ];
    for (variant, canonical) in cases {
        assert_eq!(canonical.parse::<VolumetricMode>().unwrap(), variant);
    }
}

#[test]
fn volumetric_mode_aliases_parse_correctly() {
    assert_eq!(
        "rm".parse::<VolumetricMode>().unwrap(),
        VolumetricMode::Raymarch
    );
    assert_eq!(
        "0".parse::<VolumetricMode>().unwrap(),
        VolumetricMode::Raymarch
    );
    assert_eq!(
        "fx".parse::<VolumetricMode>().unwrap(),
        VolumetricMode::Froxels
    );
    assert_eq!(
        "1".parse::<VolumetricMode>().unwrap(),
        VolumetricMode::Froxels
    );
}

#[test]
fn normalize_key_handles_variations() {
    assert_eq!(normalize_key("Cook-Torrance_GGX"), "cooktorranceggx");
    assert_eq!(normalize_key(" Blinn Phong "), "blinnphong");
    assert_eq!(normalize_key("OREN.NAYAR"), "orennayar");
    assert_eq!(normalize_key("henyey-greenstein"), "henyeygreenstein");
}

// P0-02: Comprehensive validation tests for all invariants

// Light validation tests
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

// Shadow validation tests
#[test]
fn shadows_require_power_of_two_map() {
    let mut cfg = RendererConfig::default();
    cfg.shadows.map_size = 300;
    let err = cfg.validate().unwrap_err();
    assert!(err.to_string().contains("power of two"));
}

#[test]
fn shadows_map_size_must_be_nonzero() {
    let mut cfg = RendererConfig::default();
    cfg.shadows.map_size = 0;
    let err = cfg.validate().unwrap_err();
    assert!(err.to_string().contains("greater than zero"));
}

#[test]
fn shadows_cascades_must_be_in_range() {
    let mut cfg = RendererConfig::default();
    cfg.shadows.cascades = 0;
    let err = cfg.validate().unwrap_err();
    assert!(err.to_string().contains("[1, 4]"));

    cfg.shadows.cascades = 5;
    let err = cfg.validate().unwrap_err();
    assert!(err.to_string().contains("[1, 4]"));
}

#[test]
fn shadows_csm_requires_multiple_cascades() {
    let mut cfg = RendererConfig::default();
    cfg.shadows.technique = ShadowTechnique::Csm;
    cfg.shadows.cascades = 1;
    let err = cfg.validate().unwrap_err();
    assert!(err.to_string().contains(">= 2"));
}

#[test]
fn shadows_pcss_requires_positive_blocker_radius() {
    let mut cfg = RendererConfig::default();
    cfg.shadows.technique = ShadowTechnique::Pcss;
    cfg.shadows.pcss_blocker_radius = -0.1;
    let err = cfg.validate().unwrap_err();
    assert!(err.to_string().contains("pcss_blocker_radius"));
}

#[test]
fn shadows_pcss_requires_positive_filter_radius() {
    let mut cfg = RendererConfig::default();
    cfg.shadows.technique = ShadowTechnique::Pcss;
    cfg.shadows.pcss_filter_radius = -0.1;
    let err = cfg.validate().unwrap_err();
    assert!(err.to_string().contains("pcss_filter_radius"));
}

#[test]
fn shadows_pcss_requires_positive_light_size() {
    let mut cfg = RendererConfig::default();
    cfg.shadows.technique = ShadowTechnique::Pcss;
    cfg.shadows.light_size = 0.0;
    let err = cfg.validate().unwrap_err();
    assert!(err.to_string().contains("light_size"));
}

#[test]
fn shadows_moment_techniques_require_positive_bias() {
    for technique in [
        ShadowTechnique::Vsm,
        ShadowTechnique::Evsm,
        ShadowTechnique::Msm,
    ] {
        let mut cfg = RendererConfig::default();
        cfg.shadows.technique = technique;
        cfg.shadows.moment_bias = 0.0;
        let err = cfg.validate().unwrap_err();
        assert!(err.to_string().contains("moment_bias"));
    }
}

#[test]
fn shadow_memory_budget_is_enforced() {
    let mut cfg = RendererConfig::default();
    cfg.shadows.technique = ShadowTechnique::Evsm;
    cfg.shadows.map_size = 8192;
    cfg.shadows.cascades = 4;
    let err = cfg.validate().unwrap_err();
    assert!(err.to_string().contains("256 MiB"));
}

#[test]
fn shadows_filtered_techniques_recommend_min_resolution() {
    let mut cfg = RendererConfig::default();
    cfg.shadows.technique = ShadowTechnique::Pcf;
    cfg.shadows.map_size = 128;
    let err = cfg.validate().unwrap_err();
    assert!(err.to_string().contains("at least 256"));
}

// Atmosphere validation tests
#[test]
fn atmosphere_hdri_sky_requires_hdr_path() {
    let mut cfg = RendererConfig::default();
    cfg.atmosphere.sky = SkyModel::Hdri;
    cfg.atmosphere.hdr_path = None;
    cfg.lighting.lights = vec![]; // No environment light
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

// GI validation tests
#[test]
fn gi_ibl_requires_environment_light_or_atmosphere_hdr() {
    let mut cfg = RendererConfig::default();
    cfg.gi.modes = vec![GiMode::Ibl];
    cfg.lighting.lights = vec![]; // No environment light
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

// Struct defaults verification
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
    assert!(config.lighting.lights.len() > 0);
    assert_eq!(config.shading.brdf, BrdfModel::CookTorranceGGX);
    assert!(config.shadows.enabled);
    assert_eq!(config.gi.modes, vec![GiMode::None]);
    assert!(config.atmosphere.enabled);
}
