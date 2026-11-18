//! The per-DEM vertical-scale default.
//!
//! `zScale` was a fixed `1.0`, which is right for the bundled Bryce sample
//! (11.3% relief) and wrong for a country-size DEM: the Swiss Alps are 4,031 m
//! of relief across 352 km — 1.1% — and rendered as a printed sheet. The
//! default now follows the DEM's own relief, and only ever exaggerates, so a
//! DEM that already reads as terrain keeps its true scale.

use studio_core::command::Command;
use studio_core::editor::Editor;
use studio_core::project::{Asset, AssetKind, Project, TerrainSeed};
use studio_core::scene::{dem_default_z_scale, MAX_Z_SCALE, MIN_Z_SCALE};
use studio_core::{ids, TerrainObject};

const BRYCE_RANGE: [f32; 2] = [2099.23, 2553.58];
const BRYCE_SPAN: [f64; 2] = [4_000.0, 3_800.0];
const SWISS_RANGE: [f32; 2] = [189.0, 4_220.0];
const SWISS_SPAN: [f64; 2] = [352_427.0, 225_654.0];

fn dem_asset(name: &str) -> Asset {
    Asset {
        id: ids::new_id(),
        name: name.to_owned(),
        path: format!("/data/{name}"),
        kind: AssetKind::Dem,
        byte_size: 1,
    }
}

fn seed(asset: &Asset, range: [f32; 2], span: [f64; 2]) -> TerrainSeed {
    TerrainSeed {
        title: None,
        name: asset.name.clone(),
        path: asset.path.clone(),
        byte_size: asset.byte_size,
        dimensions: Some([1024, 1024]),
        elevation_range: Some(range),
        world_origin_xz: Some([0.0, 0.0]),
        world_span_xz: Some(span),
        overlay: None,
    }
}

#[test]
fn a_steep_dem_keeps_its_true_scale() {
    // Bryce Canyon is 454 m of relief across 4 km (11.3%), so the 6% target
    // needs no exaggeration and the bundled sample must not move.
    assert_eq!(
        dem_default_z_scale(Some(BRYCE_RANGE), Some(BRYCE_SPAN)),
        1.0
    );
}

#[test]
fn a_country_size_dem_is_exaggerated_to_six_percent_relief() {
    // 0.06 * 352_427 / 4_031 = 5.2459...
    let scale = dem_default_z_scale(Some(SWISS_RANGE), Some(SWISS_SPAN));
    assert!((scale - 5.246).abs() < 0.001, "got {scale}");
}

#[test]
fn missing_or_degenerate_measurements_stay_at_true_scale() {
    // No georeferenced footprint (a local DEM) is not a reason to guess.
    assert_eq!(dem_default_z_scale(Some(SWISS_RANGE), None), 1.0);
    assert_eq!(dem_default_z_scale(None, Some(SWISS_SPAN)), 1.0);
    // A flat DEM has no relief to exaggerate.
    assert_eq!(
        dem_default_z_scale(Some([100.0, 100.0]), Some(SWISS_SPAN)),
        1.0
    );
    assert_eq!(
        dem_default_z_scale(Some([0.0, -5.0]), Some(SWISS_SPAN)),
        1.0
    );
}

#[test]
fn the_default_stays_inside_the_engines_range() {
    // A continent of almost flat ground would ask for ~150x; the engine's
    // ceiling is 50.0, so the derived value is capped there, not refused.
    let scale = dem_default_z_scale(Some([0.0, 1.0]), Some([3_000_000.0, 2_000_000.0]));
    assert_eq!(scale, MAX_Z_SCALE);
    assert!((MIN_Z_SCALE..=MAX_Z_SCALE).contains(&scale));
}

#[test]
fn attaching_a_dem_derives_the_scale() {
    let asset = dem_asset("switzerland_dem.tif");
    let terrain = TerrainObject::new(asset.id).with_dem_info(
        [1024, 1024],
        SWISS_RANGE,
        Some([0.0, 0.0]),
        Some(SWISS_SPAN),
    );
    assert!(
        (terrain.z_scale - 5.246).abs() < 0.001,
        "got {}",
        terrain.z_scale
    );
}

#[test]
fn replacing_the_dem_re_derives_the_scale_and_undo_restores_it() {
    let alps = dem_asset("switzerland_dem.tif");
    let canyon = dem_asset("Bryce_Canyon_1024.tif");
    let mut project = Project::starter("Alps", Some(seed(&alps, SWISS_RANGE, SWISS_SPAN)));
    project.assets.push(canyon.clone());
    let terrain_id = project.terrain_object().expect("terrain").id;
    let mut editor = Editor::new(project);

    let alp_scale = editor
        .project()
        .scene
        .object(terrain_id)
        .and_then(|object| object.terrain())
        .expect("terrain")
        .z_scale;
    assert!((alp_scale - 5.246).abs() < 0.001, "got {alp_scale}");

    editor
        .apply(Command::SetTerrainDem {
            id: terrain_id,
            asset: canyon.id,
            dimensions: Some([1024, 1024]),
            elevation_range: Some(BRYCE_RANGE),
            world_origin_xz: Some([0.0, 0.0]),
            world_span_xz: Some(BRYCE_SPAN),
            z_scale: None,
        })
        .expect("a DEM replacement is valid");
    let canyon_scale = editor
        .project()
        .scene
        .object(terrain_id)
        .and_then(|object| object.terrain())
        .expect("terrain")
        .z_scale;
    assert!(
        (canyon_scale - 1.0).abs() < f32::EPSILON,
        "a steep DEM keeps its true scale, got {canyon_scale}"
    );

    editor.undo().expect("undo is available");
    let restored = editor
        .project()
        .scene
        .object(terrain_id)
        .and_then(|object| object.terrain())
        .expect("terrain")
        .z_scale;
    assert!(
        (restored - alp_scale).abs() < f32::EPSILON,
        "undo must restore the previous scale, got {restored}"
    );
}

#[test]
fn the_model_refuses_a_vertical_scale_the_engine_refuses() {
    // The engine raises "z_scale must be 0.1-50.0" instead of clamping, so a
    // value outside the range must never reach it.
    let asset = dem_asset("switzerland_dem.tif");
    let project = Project::starter("Alps", Some(seed(&asset, SWISS_RANGE, SWISS_SPAN)));
    let terrain_id = project.terrain_object().expect("terrain").id;
    let mut editor = Editor::new(project);

    for bad in [0.05_f32, 51.0, f32::NAN] {
        let error = editor
            .apply(Command::SetTerrainZScale {
                id: terrain_id,
                z_scale: bad,
            })
            .expect_err("out-of-range scale must be rejected")
            .to_string();
        assert!(error.contains("vertical scale"), "{error}");
    }
}

#[test]
fn fit_relief_adopts_the_dems_default_and_undo_restores_true_scale() {
    // A project saved before the per-DEM default carries zScale 1.0, which
    // renders the Alps as a plate. One undoable command adopts the rule the
    // core owns — no second copy of 0.06 in the shell.
    let asset = dem_asset("switzerland_dem.tif");
    let mut project = Project::starter("Alps", Some(seed(&asset, SWISS_RANGE, SWISS_SPAN)));
    let terrain_id = project.terrain_object().expect("terrain").id;
    {
        let object = project.scene.object_mut(terrain_id).expect("terrain");
        if let studio_core::ObjectKind::Terrain(terrain) = &mut object.kind {
            terrain.z_scale = 1.0;
        }
    }
    let mut editor = Editor::new(project);

    editor
        .apply(Command::FitTerrainRelief { id: terrain_id })
        .expect("fit relief");
    let fitted = editor
        .project()
        .scene
        .object(terrain_id)
        .and_then(|object| object.terrain())
        .expect("terrain")
        .z_scale;
    assert!((fitted - 5.246).abs() < 0.001, "got {fitted}");

    editor.undo().expect("undo is available");
    let restored = editor
        .project()
        .scene
        .object(terrain_id)
        .and_then(|object| object.terrain())
        .expect("terrain")
        .z_scale;
    assert!((restored - 1.0).abs() < f32::EPSILON, "got {restored}");
}

#[test]
fn the_map_look_sets_relief_camera_sun_and_hillshade_and_undo_restores_them() {
    // One command, one undo: the whole cartography setup.
    let asset = dem_asset("switzerland_dem.tif");
    let project = Project::starter("Alps", Some(seed(&asset, SWISS_RANGE, SWISS_SPAN)));
    let terrain_id = project.terrain_object().expect("terrain").id;
    let camera_id = project
        .scene
        .objects
        .iter()
        .find(|object| matches!(object.kind, studio_core::ObjectKind::Camera(_)))
        .expect("a camera")
        .id;
    let sun_id = project
        .scene
        .objects
        .iter()
        .find(|object| matches!(object.kind, studio_core::ObjectKind::Light(_)))
        .expect("a sun")
        .id;
    let camera_before = project.scene.object(camera_id).expect("camera").transform;
    let sun_before = project.scene.object(sun_id).expect("sun").transform;
    let mut editor = Editor::new(project);

    editor
        .apply(Command::ApplyMapLook { id: terrain_id })
        .expect("the map look applies");

    let terrain = editor
        .project()
        .scene
        .object(terrain_id)
        .and_then(|object| object.terrain())
        .expect("terrain")
        .z_scale;
    assert!((terrain - 5.246).abs() < 0.001, "relief: {terrain}");
    assert!(
        editor.project().render.sun_visibility,
        "the hillshade is on"
    );
    assert_eq!(editor.project().render.sun_visibility_steps, 48);

    // The camera moved and widened: the still's fixed orbit radius has to see
    // the whole country, so the field of view grows.
    let camera = editor
        .project()
        .scene
        .object(camera_id)
        .expect("camera")
        .clone();
    assert_ne!(camera.transform, camera_before, "the camera moved");
    let fov = match &camera.kind {
        studio_core::ObjectKind::Camera(kind) => kind.fov_deg,
        _ => panic!("a camera"),
    };
    assert!(fov > 45.0, "the still's field of view widens, got {fov}");

    // The sun points north-west now: the rotation is not the starter's.
    let sun = editor.project().scene.object(sun_id).expect("sun").clone();
    assert_ne!(sun.transform, sun_before, "the sun moved");

    editor.undo().expect("undo is available");
    let restored = editor.project().scene.object(camera_id).expect("camera");
    assert_eq!(
        restored.transform, camera_before,
        "undo restores the camera"
    );
    let restored_sun = editor.project().scene.object(sun_id).expect("sun");
    assert_eq!(restored_sun.transform, sun_before, "undo restores the sun");
    assert!(
        !editor.project().render.sun_visibility,
        "undo restores the hillshade"
    );
}
