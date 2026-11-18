//! The committed sample projects must load, validate, and survive a round trip.
//!
//! This is the Phase 7 "repeatable project round trips" guard: every shipped
//! sample is a real file with real references, and CI opens it the same way
//! the app does (`persist::load` resolves the relative asset paths).
//!
//! The GeoTIFF references point at Git LFS objects, so their *content* is not
//! asserted here — existence and resolution are. The HDRI ships as a normal
//! file and is checked byte-for-byte.

use std::path::{Path, PathBuf};

use studio_core::persist;
use studio_core::project::{AssetKind, Project};
use studio_core::timeline::{self, Channel};
use studio_core::SCHEMA_VERSION;

fn samples_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../samples")
}

fn sample(name: &str) -> Project {
    let path = samples_dir().join(name);
    let project = persist::load(&path)
        .unwrap_or_else(|error| panic!("{} must load: {error}", path.display()));
    assert_eq!(project.schema_version, SCHEMA_VERSION);
    project
        .validate()
        .unwrap_or_else(|reason| panic!("{name} must validate: {reason}"));
    project
}

#[test]
fn every_sample_file_loads_and_validates() {
    let mut seen = 0;
    for entry in std::fs::read_dir(samples_dir()).expect("samples directory") {
        let path = entry.expect("entry").path();
        if path.extension().and_then(|ext| ext.to_str()) != Some("studio3d") {
            continue;
        }
        let project = persist::load(&path)
            .unwrap_or_else(|error| panic!("{} must load: {error}", path.display()));
        project
            .validate()
            .unwrap_or_else(|reason| panic!("{} must validate: {reason}", path.display()));
        assert!(
            !project.scene.objects.is_empty(),
            "{} has no objects",
            path.display()
        );
        seen += 1;
    }
    assert!(seen >= 3, "expected the committed samples, found {seen}");
}

#[test]
fn sample_assets_resolve_against_the_sample_directory() {
    for name in [
        "bryce-terrain.studio3d",
        "cube-animation.studio3d",
        "hdri-studio.studio3d",
    ] {
        let project = sample(name);
        for asset in &project.assets {
            assert!(
                Path::new(&asset.path).is_absolute(),
                "{name}: {} was not resolved by load",
                asset.path
            );
            assert!(
                Path::new(&asset.path).is_file(),
                "{name}: referenced file is missing: {}",
                asset.path
            );
        }
    }

    // The HDRI is committed with the samples (not LFS), so its size is real.
    let project = sample("hdri-studio.studio3d");
    let environment = project
        .render
        .environment
        .expect("the HDRI sample must set an environment");
    let asset = project.asset(environment).expect("environment asset");
    assert_eq!(asset.kind, AssetKind::Hdr);
    assert_eq!(
        std::fs::metadata(&asset.path).expect("HDRI on disk").len(),
        asset.byte_size,
        "the HDRI asset size must match the file"
    );
}

#[test]
fn the_bryce_sample_is_the_terrain_scene_the_readme_describes() {
    let project = sample("bryce-terrain.studio3d");
    let terrain = project.terrain_object().expect("terrain scene");
    let kind = terrain.terrain().expect("terrain kind");
    assert_eq!(kind.dimensions, Some([1024, 1024]));
    assert_eq!(kind.world_span_xz, Some([4000.0, 3800.0]));
    assert!(
        project.cartography.iter().any(|layer| matches!(
            layer.kind,
            studio_core::CartographyLayerKind::Imagery { .. }
        )),
        "the Bryce sample carries its NAIP drape"
    );
    // Lit by the committed HDRI, which is what the Environment control sets.
    assert!(project.render.environment.is_some());
    let kinds: Vec<AssetKind> = project.assets.iter().map(|asset| asset.kind).collect();
    assert!(kinds.contains(&AssetKind::Dem));
    assert!(kinds.contains(&AssetKind::Hdr));
}

#[test]
fn the_cube_sample_animates_the_frame_the_readme_shows() {
    let mut project = sample("cube-animation.studio3d");
    assert_eq!(project.timeline.tracks.len(), 1);
    assert_eq!(project.timeline.tracks[0].channels.len(), 2);
    let object_id = project.timeline.tracks[0].object_id;

    timeline::evaluate(&mut project, 24.0);
    let cube = project.scene.object(object_id).expect("animated cube");
    assert_eq!(
        timeline::channel_value(cube, Channel::TranslationX),
        Some(3.0),
        "frame 24 is the cube's rightmost key"
    );
    timeline::evaluate(&mut project, 0.0);
    let cube = project.scene.object(object_id).unwrap();
    assert_eq!(timeline::channel_value(cube, Channel::RotationZ), Some(0.0));
}

#[test]
fn every_sample_round_trips_through_save_and_load() {
    let dir = tempfile::tempdir().unwrap();
    for name in [
        "bryce-terrain.studio3d",
        "cube-animation.studio3d",
        "hdri-studio.studio3d",
    ] {
        let project = sample(name);
        let path = dir.path().join(name);
        persist::save_atomic(&path, &project).expect("save sample");
        let reloaded = persist::load(&path).expect("reload sample");
        assert_eq!(reloaded.scene, project.scene, "{name}: scene drifted");
        assert_eq!(
            reloaded.timeline, project.timeline,
            "{name}: timeline drifted"
        );
        assert_eq!(reloaded.render, project.render, "{name}: render drifted");
        // Asset references must survive the round trip as a set (paths may
        // become absolute, ids must not).
        let ids = |value: &Project| {
            let mut ids: Vec<_> = value.assets.iter().map(|asset| asset.id).collect();
            ids.sort();
            ids
        };
        assert_eq!(ids(&reloaded), ids(&project), "{name}: asset ids drifted");
    }
}
