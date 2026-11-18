//! The wire format is a contract with the TypeScript UI.
//!
//! `src/bridge/types.ts` mirrors these shapes by hand (Phase 1 does not
//! generate types). This test fails the moment the Rust side renames a field,
//! changes a tag, or drops a key, so the mirror cannot drift silently.

use std::collections::BTreeSet;

use serde_json::Value;
use studio_core::command::Command;
use studio_core::math::{Transform, Vec3};
use studio_core::project::{Project, UNTITLED};

fn keys(value: &Value) -> BTreeSet<String> {
    value
        .as_object()
        .expect("expected an object")
        .keys()
        .cloned()
        .collect()
}

fn expect_keys(value: &Value, expected: &[&str]) {
    let expected: BTreeSet<String> = expected.iter().map(|key| (*key).to_owned()).collect();
    assert_eq!(keys(value), expected, "unexpected key set in {value}");
}

#[test]
fn project_document_key_shape_is_stable() {
    let value = serde_json::to_value(Project::new(UNTITLED)).unwrap();
    expect_keys(
        &value,
        &[
            "format",
            "schemaVersion",
            "id",
            "name",
            "createdUnixMs",
            "updatedUnixMs",
            "scene",
            "cartography",
            "assets",
            "selection",
            "workspace",
            "viewport",
            "render",
            "timeline",
        ],
    );
    assert_eq!(value["format"], "studio3d");
    assert_eq!(value["schemaVersion"], studio_core::SCHEMA_VERSION);
    assert_eq!(value["workspace"], "studio");

    expect_keys(&value["viewport"], &["mode", "gridVisible", "hillshade"]);
    assert_eq!(value["viewport"]["mode"], "solid");
    expect_keys(
        &value["render"],
        &[
            "width",
            "height",
            "exposure",
            "shadows",
            "sky",
            "iblIntensity",
            "environment",
            "environmentBackgroundIntensity",
            "stillAov",
            "stillDenoise",
            "shadowTechnique",
            "shadowResolution",
            "sunVisibility",
            "sunVisibilityDistance",
            "sunVisibilitySteps",
            "skyModel",
            "skyTurbidity",
            "skyGroundAlbedo",
            "skyExposure",
            "water",
            "fog",
            "clouds",
            "bloom",
            "vignette",
        ],
    );
    assert_eq!(value["render"]["environmentBackgroundIntensity"], 1.0);
    // Terrain self-shadowing is off by default and its rays reach 2 km.
    // The map's own decisions are layers now, not render settings.
    assert_eq!(value["cartography"][0]["type"], "relief");
    assert_eq!(value["cartography"][0]["visible"], true);
    assert_eq!(value["render"]["sunVisibility"], false);
    assert_eq!(value["render"]["sunVisibilityDistance"], 2000.0);
    assert_eq!(value["render"]["sunVisibilitySteps"], 32);
    // Water is off until the project asks for it; the engine derives the mask.
    expect_keys(
        &value["render"]["water"],
        &["enabled", "level", "slopeThreshold", "roughness"],
    );
    assert_eq!(value["render"]["water"]["enabled"], false);
    assert_eq!(value["render"]["water"]["level"], 0.0);
    expect_keys(
        &value["render"]["fog"],
        &["enabled", "density", "reach", "inscatter"],
    );
    assert_eq!(value["render"]["fog"]["enabled"], false);
    expect_keys(
        &value["render"]["bloom"],
        &["enabled", "threshold", "softness", "intensity"],
    );
    assert_eq!(value["render"]["bloom"]["enabled"], false);
    assert_eq!(value["render"]["vignette"], 0.0);
    // The default project uses the built-in studio sky.
    assert_eq!(value["render"]["environment"], serde_json::Value::Null);

    let timeline = &value["timeline"];
    expect_keys(timeline, &["frameRate", "length", "currentFrame", "tracks"]);
    assert_eq!(timeline["frameRate"], studio_core::DEFAULT_FRAME_RATE);
    assert_eq!(timeline["length"], studio_core::DEFAULT_LENGTH);
    assert_eq!(timeline["currentFrame"], 0.0);
    assert!(timeline["tracks"].as_array().unwrap().is_empty());
}

#[test]
fn scene_and_object_shapes_are_stable() {
    let value = serde_json::to_value(Project::new(UNTITLED)).unwrap();
    let scene = &value["scene"];
    expect_keys(scene, &["id", "name", "objects", "activeCamera"]);

    let objects = scene["objects"].as_array().unwrap();
    assert_eq!(objects.len(), 3);
    for object in objects {
        expect_keys(
            object,
            &[
                "id",
                "name",
                "parent",
                "transform",
                "visible",
                "locked",
                "kind",
            ],
        );
        expect_keys(
            &object["transform"],
            &["translation", "rotationDeg", "scale"],
        );
        expect_keys(&object["transform"]["translation"], &["x", "y", "z"]);
    }

    let cube = objects
        .iter()
        .find(|object| object["kind"]["type"] == "mesh")
        .expect("starter scene has a cube");
    expect_keys(&cube["kind"], &["type", "source", "material"]);
    expect_keys(
        &cube["kind"]["material"],
        &["baseColor", "roughness", "metallic", "texture"],
    );
    expect_keys(&cube["kind"]["source"], &["type", "primitive"]);
    assert_eq!(cube["kind"]["source"]["type"], "primitive");
    assert_eq!(cube["kind"]["source"]["primitive"]["type"], "cube");
    expect_keys(&cube["kind"]["source"]["primitive"], &["type", "size"]);
    assert_eq!(cube["kind"]["source"]["primitive"]["size"], 2.0);

    let camera = objects
        .iter()
        .find(|object| object["kind"]["type"] == "camera")
        .expect("starter scene has a camera");
    expect_keys(&camera["kind"], &["type", "fovDeg", "near", "far"]);

    let light = objects
        .iter()
        .find(|object| object["kind"]["type"] == "light")
        .expect("starter scene has a light");
    expect_keys(&light["kind"], &["type", "kind", "color", "intensity"]);
    assert_eq!(light["kind"]["kind"], "directional");
}

#[test]
fn primitives_and_light_kinds_use_the_documented_tags() {
    let mesh = serde_json::to_value(studio_core::Primitive::Sphere { radius: 1.5 }).unwrap();
    assert_eq!(mesh, serde_json::json!({"type": "sphere", "radius": 1.5}));
    let plane = serde_json::to_value(studio_core::Primitive::Plane { size: 4.0 }).unwrap();
    assert_eq!(plane, serde_json::json!({"type": "plane", "size": 4.0}));
    let light = serde_json::to_value(studio_core::LightKind::Point).unwrap();
    assert_eq!(light, "point");
}

#[test]
fn commands_use_camel_case_type_tags() {
    let transform = Transform {
        translation: Vec3::new(1.0, 2.0, 3.0),
        rotation_deg: Vec3::ZERO,
        scale: Vec3::ONE,
    };
    let value = serde_json::to_value(Command::SetTransform {
        id: studio_core::ids::new_id(),
        transform,
    })
    .unwrap();
    expect_keys(&value, &["type", "id", "transform"]);
    assert_eq!(value["type"], "setTransform");

    let visibility = serde_json::to_value(Command::SetVisibility {
        id: studio_core::ids::new_id(),
        visible: false,
    })
    .unwrap();
    assert_eq!(visibility["type"], "setVisibility");

    let active = serde_json::to_value(Command::SetActiveCamera { id: None }).unwrap();
    assert_eq!(
        active,
        serde_json::json!({"type": "setActiveCamera", "id": null})
    );

    let batch = serde_json::to_value(Command::Batch { commands: vec![] }).unwrap();
    assert_eq!(batch, serde_json::json!({"type": "batch", "commands": []}));

    let duplicate = serde_json::to_value(Command::DuplicateObject {
        id: studio_core::ids::new_id(),
        name: None,
    })
    .unwrap();
    assert_eq!(duplicate["type"], "duplicateObject");
    assert!(duplicate.get("name").is_some());
}

#[test]
fn session_state_shape_matches_the_ui_mirror() {
    let editor = studio_core::Editor::new(Project::new(UNTITLED));
    let value = serde_json::to_value(editor.state()).unwrap();
    expect_keys(
        &value,
        &[
            "project",
            "revision",
            "dirty",
            "path",
            "canUndo",
            "canRedo",
            "undoLabel",
            "redoLabel",
            "historyLen",
        ],
    );
}
