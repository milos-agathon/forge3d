//! Phase 4 acceptance: terrain-first scenes and their handoff.
//!
//! The Studio default project is a DEM, not a cube, so these tests pin the
//! starter scene and the orbit camera the renderer needs.

use studio_core::command::Command;
use studio_core::editor::Editor;
use studio_core::persist::{from_json, save_atomic};
use studio_core::project::{AssetKind, Project, TerrainSeed, SCHEMA_VERSION, UNTITLED};
use studio_core::scene::ObjectKind;
use studio_core::workspace::{handoff, terrain_camera};
use studio_core::{Transform, Workspace};

fn seed() -> TerrainSeed {
    TerrainSeed {
        title: Some("Bryce Canyon".to_owned()),
        name: "Bryce_Canyon_1024.tif".to_owned(),
        path: "/data/Bryce_Canyon_1024.tif".to_owned(),
        byte_size: 3_321_356,
        // Measured from the bundled sample: 1024² pixels, 2099–2554 m, and a
        // georeferenced footprint of 4000 × 3800 m (EPSG:26912).
        dimensions: Some([1024, 1024]),
        elevation_range: Some([2099.23, 2553.58]),
        world_origin_xz: Some([395_998.5, -4_165_574.5]),
        world_span_xz: Some([4000.0, 3800.0]),
        overlay: None,
    }
}

fn naip_seed() -> TerrainSeed {
    TerrainSeed {
        overlay: Some(studio_core::TerrainSeedOverlay {
            name: "Bryce_Canyon_naip.tif".to_owned(),
            path: "/data/Bryce_Canyon_naip.tif".to_owned(),
            byte_size: 28_000_000,
            opacity: 0.82,
        }),
        ..seed()
    }
}

#[test]
fn a_project_without_a_dem_falls_back_to_the_primitive_starter() {
    let project = Project::starter(UNTITLED, None);
    assert_eq!(project.scene.objects.len(), 3);
    assert!(project.terrain_object().is_none());
    assert!(project.assets.is_empty());
}

#[test]
fn the_starter_scene_is_the_terrain_when_a_dem_is_available() {
    let project = Project::starter("Bryce", Some(seed()));

    assert_eq!(project.assets.len(), 1);
    assert_eq!(project.assets[0].kind, AssetKind::Dem);
    assert_eq!(project.assets[0].byte_size, 3_321_356);

    let terrain = project.terrain_object().expect("terrain object");
    assert_eq!(terrain.name, "Bryce Canyon");
    let terrain_object = terrain.terrain().expect("terrain payload");
    assert_eq!(terrain_object.asset, project.assets[0].id);
    assert!((terrain_object.z_scale - 1.0).abs() < f32::EPSILON);

    // The cube is gone; camera and sun remain so the scene is framed.
    assert!(project.scene.objects.iter().all(|object| !object.is_mesh()));
    assert_eq!(project.scene.objects.len(), 3);
    assert!(project.scene.camera().is_some());
    project.validate().expect("terrain starter is valid");
}

#[test]
fn the_bryce_starter_ships_with_the_naip_albedo() {
    let project = Project::starter("Bryce", Some(naip_seed()));

    assert_eq!(project.assets.len(), 2);
    let albedo = project
        .assets
        .iter()
        .find(|asset| asset.kind == AssetKind::Orthophoto)
        .expect("the NAIP asset is registered");
    assert_eq!(albedo.name, "Bryce_Canyon_naip.tif");

    // The starter albedo is the map's imagery layer now.
    let imagery = project
        .cartography
        .iter()
        .find(|layer| {
            matches!(
                layer.kind,
                studio_core::CartographyLayerKind::Imagery { .. }
            )
        })
        .expect("the starter albedo is draped on the terrain");
    let studio_core::CartographyLayerKind::Imagery {
        asset,
        opacity,
        preserve_colors,
        despeckle,
    } = imagery.kind
    else {
        unreachable!("the layer is an imagery layer");
    };
    assert_eq!(asset, albedo.id);
    assert!((opacity - 0.82).abs() < f32::EPSILON);
    assert!(preserve_colors);
    // An orthophoto is not a class raster: despeckle starts off.
    assert!(!despeckle);
    project
        .validate()
        .expect("terrain starter with albedo is valid");
}

#[test]
fn a_scene_holds_exactly_one_terrain() {
    let mut editor = Editor::new(Project::starter("Bryce", Some(seed())));
    let terrain_id = editor
        .project()
        .terrain_object()
        .expect("starter terrain")
        .id;

    let error = editor
        .apply(Command::DuplicateObject {
            id: terrain_id,
            name: None,
        })
        .expect_err("duplicating the terrain must be rejected")
        .to_string();
    assert!(error.contains("one terrain"), "{error}");
    assert_eq!(
        editor
            .project()
            .scene
            .objects
            .iter()
            .filter(|object| object.is_terrain())
            .count(),
        1,
        "the rejected duplicate must not leave a second terrain behind"
    );
}

#[test]
fn terrain_transform_edits_are_rejected() {
    // The renderer always draws the DEM centred at the origin; moving the
    // Solid-view copy would desync the two views instead of moving the render.
    let mut editor = Editor::new(Project::starter("Bryce", Some(seed())));
    let terrain_id = editor
        .project()
        .terrain_object()
        .expect("starter terrain")
        .id;
    let mut transform = Transform::default();
    transform.translation.x = 12.0;

    let error = editor
        .apply(Command::SetTransform {
            id: terrain_id,
            transform,
        })
        .expect_err("moving terrain must be rejected")
        .to_string();
    assert!(error.contains("georeferenced"), "{error}");
    let terrain = editor.project().scene.object(terrain_id).expect("terrain");
    assert_eq!(
        terrain.transform,
        Transform::default(),
        "the rejected edit must leave the transform untouched"
    );
}

#[test]
fn the_handoff_carries_the_dem_and_an_orbit_camera() {
    let project = Project::starter("Bryce", Some(seed()));
    let plan = handoff(&project, 1, Workspace::Forge3d);

    let terrain = plan.terrain.expect("DEM in the handoff");
    assert_eq!(terrain.name, "Bryce_Canyon_1024.tif");
    assert_eq!(terrain.path, "/data/Bryce_Canyon_1024.tif");
    assert!((terrain.z_scale - 1.0).abs() < f32::EPSILON);

    // The terrain starter frames the scene from (11, 7.5, 13): 66.2° down from
    // vertical; east 11 / south 13 → azimuth 310.2° in the engine's
    // geographic orbit frame (east → north). The old atan2(z, x) reported
    // 49.8° — mirrored by ~100°, and disagreeing with the viewport's live
    // view, which already maps south = -z.
    let orbit = plan.terrain_camera.expect("terrain orbit camera");
    assert!((orbit.theta_deg - 66.2).abs() < 0.5, "{orbit:?}");
    assert!((orbit.phi_deg - 310.2).abs() < 0.5, "{orbit:?}");
    assert!((orbit.fov_deg - 45.0).abs() < 1e-4);

    // Framing reproduced from the viewer's own formula: a georeferenced DEM
    // lives in metres, far from the origin, so a pixel-space guess would aim at
    // empty space (which is exactly what happened before this was fixed).
    let radius = orbit.radius.expect("radius from the DEM footprint");
    assert!((radius - 6000.0).abs() < 0.5, "{radius}");
    let target = orbit.target.expect("target from the DEM footprint");
    assert!((target[0] - 397_998.5).abs() < 0.01, "{target:?}");
    assert!((target[2] + 4_163_674.5).abs() < 0.01, "{target:?}");
    assert!(
        (target[1] - f64::from(2553.58f32 - 2099.23f32) * 0.5).abs() < 0.1,
        "{target:?}"
    );

    // The look-at camera is still reported for mesh scenes.
}

#[test]
fn the_orbit_camera_matches_the_engine_north_mode_convention() {
    // The engine's north-mode orbit: eye = target + r·(sinθ·cosφ, sinθ·sinφ,
    // cosθ) in the geographic frame (+X east, +Y north, +Z up). Reconstruct
    // that direction from the reported angles and compare with the Solid
    // camera transposed into the same frame (the Solid +Z axis is south, so
    // north = -z). The old test rebuilt the dropped viewer's Y-up convention
    // and pinned the mirrored azimuth.
    let project = Project::starter("Bryce", Some(seed()));
    let orbit = terrain_camera(&project).unwrap();
    let theta: f32 = orbit.theta_deg.to_radians();
    let phi: f32 = orbit.phi_deg.to_radians();
    let direction = [
        theta.sin() * phi.cos(), // east
        theta.sin() * phi.sin(), // north
        theta.cos(),             // up
    ];

    let camera = project.scene.camera().unwrap();
    let eye = camera.transform.translation;
    let length = (eye.x * eye.x + eye.y * eye.y + eye.z * eye.z).sqrt();
    let expected = [eye.x / length, -eye.z / length, eye.y / length];
    for (index, (component, want)) in direction.iter().zip(expected).enumerate() {
        assert!(
            (component - want).abs() < 1e-3,
            "axis {index}: engine {component} vs solid-mapped {want}"
        );
    }
}

#[test]
fn a_terrain_pointing_at_a_mesh_asset_is_rejected() {
    let mut project = Project::starter("Bryce", Some(seed()));
    project.assets[0].kind = AssetKind::Glb;
    let error = project.validate().unwrap_err();
    assert!(error.contains("as terrain"), "{error}");
}

#[test]
fn v2_projects_migrate_to_v3_and_stay_renderable() {
    let project = Project::starter("Bryce", Some(seed()));
    let mut value = serde_json::to_value(&project).unwrap();
    value["schemaVersion"] = serde_json::json!(2);

    let migrated = from_json(&value.to_string(), "v2.studio3d").expect("v2 migrates");
    assert_eq!(migrated.schema_version, SCHEMA_VERSION);
    assert_eq!(migrated, project, "migration must not touch scene content");
}

#[test]
fn terrain_and_dem_survive_a_save_and_reload() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("bryce.studio3d");
    let project = Project::starter("Bryce", Some(seed()));
    save_atomic(&path, &project).unwrap();

    let reloaded = studio_core::persist::load(&path).unwrap();
    assert_eq!(reloaded, project);
    assert!(matches!(
        reloaded.scene.objects[0].kind,
        ObjectKind::Terrain(_)
    ));
}
