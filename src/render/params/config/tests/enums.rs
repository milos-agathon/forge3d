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
        assert_eq!(variant.canonical(), canonical);
        assert_eq!(canonical.parse::<LightType>().unwrap(), variant);
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
