//! Phase 2 acceptance: imported meshes, duplication, and asset references.
//!
//! The plan's Phase 2 exit criterion is that editing changes the canonical
//! model (not a viewport-only copy) and reloading preserves the edits. These
//! tests cover the model half; the viewport half is exercised in the app.

use std::fs;

use serde_json::json;
use studio_core::command::Command;
use studio_core::editor::Editor;
use studio_core::ids;
use studio_core::persist::{from_json, load, save_atomic};
use studio_core::project::{Asset, AssetKind, Project, SCHEMA_VERSION, UNTITLED};
use studio_core::scene::{
    MeshObject, MeshSource, ObjectKind, Primitive, SceneObject, TerrainObject,
};
use studio_core::Transform;

fn cube_id(editor: &Editor) -> uuid::Uuid {
    editor
        .project()
        .scene
        .objects
        .iter()
        .find(|object| object.is_mesh())
        .unwrap()
        .id
}

/// A project exactly as a v1 build wrote it: meshes carry `kind.primitive`.
fn v1_document() -> serde_json::Value {
    json!({
        "format": "studio3d",
        "schemaVersion": 1,
        "id": ids::new_id().to_string(),
        "name": "Legacy project",
        "createdUnixMs": 1,
        "updatedUnixMs": 2,
        "scene": {
            "id": ids::new_id().to_string(),
            "name": "Scene",
            "activeCamera": null,
            "objects": [
                {
                    "id": ids::new_id().to_string(),
                    "name": "Cube",
                    "parent": null,
                    "transform": {
                        "translation": {"x": 1.0, "y": 2.0, "z": 3.0},
                        "rotationDeg": {"x": 0.0, "y": 45.0, "z": 0.0},
                        "scale": {"x": 1.0, "y": 1.0, "z": 1.0}
                    },
                    "visible": true,
                    "locked": false,
                    "kind": {"type": "mesh", "primitive": {"type": "cube", "size": 2.0}}
                }
            ]
        },
        "selection": [],
        "workspace": "studio",
        "viewport": {"mode": "solid", "gridVisible": true, "hillshade": true},
        "render": {"width": 1920, "height": 1080}
    })
}

#[test]
fn v1_projects_are_migrated_on_load() {
    let legacys = v1_document().to_string();
    let project = from_json(&legacys, "legacy.studio3d").expect("v1 must migrate");
    assert_eq!(project.schema_version, SCHEMA_VERSION);
    assert!(project.assets.is_empty(), "v2 adds an empty registry");

    let mesh = project.scene.objects[0].mesh().expect("cube survived");
    assert_eq!(
        mesh.source,
        MeshSource::Primitive {
            primitive: Primitive::Cube { size: 2.0 }
        }
    );
    // Transforms and identities are untouched by the migration.
    assert_eq!(project.scene.objects[0].transform.translation.x, 1.0);
    assert_eq!(project.scene.objects[0].transform.rotation_deg.y, 45.0);
}

#[test]
fn migrated_projects_are_written_back_at_the_current_version() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("migrated.studio3d");
    let project = from_json(&v1_document().to_string(), "legacy").unwrap();
    save_atomic(&path, &project).unwrap();

    let text = fs::read_to_string(&path).unwrap();
    let value: serde_json::Value = serde_json::from_str(&text).unwrap();
    assert_eq!(value["schemaVersion"], SCHEMA_VERSION);
    assert_eq!(
        value["scene"]["objects"][0]["kind"]["source"]["type"],
        "primitive"
    );
    assert!(value["scene"]["objects"][0]["kind"]
        .get("primitive")
        .is_none());
}

#[test]
fn duplicate_creates_a_free_name_and_one_undo_step() {
    let mut editor = Editor::new(Project::new(UNTITLED));
    let id = cube_id(&editor);
    let original = editor.project().scene.object(id).unwrap().clone();

    let applied = editor
        .apply(Command::DuplicateObject { id, name: None })
        .unwrap();
    assert_eq!(applied.affected.len(), 1);
    let duplicate_id = applied.affected[0];

    let duplicate = editor.project().scene.object(duplicate_id).unwrap();
    assert_eq!(duplicate.name, "Cube.001");
    assert_ne!(duplicate.id, original.id);
    assert_eq!(
        duplicate.transform, original.transform,
        "duplicate is in place"
    );
    assert_eq!(editor.project().scene.objects.len(), 4);

    editor.undo().unwrap();
    assert!(editor.project().scene.object(duplicate_id).is_none());
    assert_eq!(editor.project().scene.objects.len(), 3);

    editor.redo().unwrap();
    assert!(editor.project().scene.object(duplicate_id).is_some());
}

#[test]
fn explicit_duplicate_names_are_validated() {
    let mut editor = Editor::new(Project::new(UNTITLED));
    let id = cube_id(&editor);
    let error = editor
        .apply(Command::DuplicateObject {
            id,
            name: Some("  ".to_owned()),
        })
        .unwrap_err()
        .to_string();
    assert!(error.contains("object name"), "{error}");
    assert_eq!(editor.project().scene.objects.len(), 3);
}

#[test]
fn imported_meshes_need_their_asset_and_refuse_primitive_edits() {
    let mut editor = Editor::new(Project::new(UNTITLED));
    let asset = Asset {
        id: ids::new_id(),
        name: "duck.glb".to_owned(),
        path: "/models/duck.glb".to_owned(),
        kind: AssetKind::Glb,
        byte_size: 1024,
    };
    let imported = SceneObject {
        id: ids::new_id(),
        name: "duck".to_owned(),
        parent: None,
        transform: Default::default(),
        visible: true,
        locked: false,
        kind: ObjectKind::Mesh(MeshObject::gltf(asset.id, None)),
    };

    // Creating the object before the asset is registered is refused.
    let error = editor
        .apply(Command::CreateObject {
            object: imported.clone(),
            index: None,
        })
        .unwrap_err()
        .to_string();
    assert!(error.contains("asset"), "{error}");

    // Register + create as one undoable step.
    editor
        .apply(Command::Batch {
            commands: vec![
                Command::AddAsset {
                    asset: asset.clone(),
                },
                Command::CreateObject {
                    object: imported.clone(),
                    index: None,
                },
            ],
        })
        .unwrap();
    assert_eq!(editor.project().assets.len(), 1);
    assert!(editor.project().scene.object(imported.id).is_some());

    // An imported mesh is not a primitive.
    let error = editor
        .apply(Command::SetPrimitive {
            id: imported.id,
            primitive: Primitive::Sphere { radius: 1.0 },
        })
        .unwrap_err()
        .to_string();
    assert!(error.contains("not a primitive"), "{error}");

    // The asset cannot be dropped while the object uses it.
    let error = editor
        .apply(Command::RemoveAsset { id: asset.id })
        .unwrap_err()
        .to_string();
    assert!(error.contains("still used"), "{error}");

    editor.undo().unwrap(); // undoes add + create together
    assert!(editor.project().assets.is_empty());
    assert!(editor.project().scene.object(imported.id).is_none());
}

#[test]
fn unused_assets_can_be_removed_and_restored() {
    let mut editor = Editor::new(Project::new(UNTITLED));
    let asset = Asset {
        id: ids::new_id(),
        name: "unused.gltf".to_owned(),
        path: "/models/unused.gltf".to_owned(),
        kind: AssetKind::Gltf,
        byte_size: 10,
    };
    editor
        .apply(Command::AddAsset {
            asset: asset.clone(),
        })
        .unwrap();
    editor.apply(Command::RemoveAsset { id: asset.id }).unwrap();
    assert!(editor.project().assets.is_empty());

    editor.undo().unwrap();
    assert_eq!(editor.project().assets, vec![asset]);
}

#[test]
fn relinking_an_asset_revalidates_the_file_size() {
    // AddAsset rejects empty files; relinking must hold the same invariant or
    // autosave (which validates the whole project) fails long afterwards.
    let mut editor = Editor::new(Project::new(UNTITLED));
    let asset = Asset {
        id: ids::new_id(),
        name: "albedo.png".to_owned(),
        path: "/a/albedo.png".to_owned(),
        kind: AssetKind::Orthophoto,
        byte_size: 12,
    };
    editor
        .apply(Command::AddAsset {
            asset: asset.clone(),
        })
        .unwrap();

    let error = editor
        .apply(Command::SetAssetPath {
            id: asset.id,
            path: "/b/albedo.png".to_owned(),
            byte_size: 0,
        })
        .expect_err("a size of 0 reads as an empty file")
        .to_string();
    assert!(error.contains("asset file is empty"), "{error}");
    assert_eq!(
        editor.project().assets[0].path,
        "/a/albedo.png",
        "the rejected relink must leave the asset untouched"
    );

    editor
        .apply(Command::SetAssetPath {
            id: asset.id,
            path: "/b/albedo.png".to_owned(),
            byte_size: 99,
        })
        .unwrap();
    assert_eq!(editor.project().assets[0].byte_size, 99);
}

#[test]
fn creating_a_terrain_requires_its_dem_asset() {
    let mut editor = Editor::new(Project::new(UNTITLED));
    let object = SceneObject {
        id: ids::new_id(),
        name: "Terrain".to_owned(),
        parent: None,
        transform: Transform::default(),
        visible: true,
        locked: false,
        kind: ObjectKind::Terrain(TerrainObject::new(ids::new_id())),
    };

    let error = editor
        .apply(Command::CreateObject {
            object,
            index: None,
        })
        .expect_err("a terrain referencing a missing asset must be rejected")
        .to_string();
    assert!(error.contains("does not exist"), "{error}");
    assert!(editor.project().terrain_object().is_none());
}

#[test]
fn asset_paths_are_saved_relative_and_reloaded_absolute() {
    let dir = tempfile::tempdir().unwrap();
    let assets_dir = dir.path().join("assets");
    fs::create_dir(&assets_dir).unwrap();
    let model = assets_dir.join("duck.glb");
    fs::write(&model, b"glTF-ish bytes").unwrap();

    let project_path = dir.path().join("scene.studio3d");
    let mut project = Project::new("Assets");
    let asset = Asset {
        id: ids::new_id(),
        name: "duck.glb".to_owned(),
        path: model.to_string_lossy().into_owned(),
        kind: AssetKind::Glb,
        byte_size: 16,
    };
    project.assets.push(asset.clone());
    project.scene.objects[0].kind = ObjectKind::Mesh(MeshObject::gltf(asset.id, None));

    save_atomic(&project_path, &project).unwrap();
    let text = fs::read_to_string(&project_path).unwrap();
    assert!(
        text.contains("\"path\": \"assets/duck.glb\""),
        "asset path was not stored relatively:\n{text}"
    );

    let reloaded = load(&project_path).unwrap();
    assert_eq!(reloaded.assets.len(), 1);
    assert_eq!(
        reloaded.assets[0].path,
        model.to_string_lossy(),
        "relative path was not resolved against the project directory"
    );
    assert_eq!(reloaded, project);
}

#[test]
fn assets_outside_the_project_directory_stay_absolute() {
    let dir = tempfile::tempdir().unwrap();
    let outside = dir.path().join("elsewhere").join("model.glb");
    let project_path = dir.path().join("project").join("scene.studio3d");
    fs::create_dir_all(outside.parent().unwrap()).unwrap();
    fs::write(&outside, b"bytes").unwrap();

    let mut project = Project::new("External");
    project.assets.push(Asset {
        id: ids::new_id(),
        name: "model.glb".to_owned(),
        path: outside.to_string_lossy().into_owned(),
        kind: AssetKind::Glb,
        byte_size: 5,
    });
    save_atomic(&project_path, &project).unwrap();

    let reloaded = load(&project_path).unwrap();
    assert_eq!(reloaded.assets[0].path, outside.to_string_lossy());
}

#[test]
fn created_primitives_keep_their_id_across_undo_and_redo() {
    let mut editor = Editor::new(Project::new(UNTITLED));
    let applied = editor
        .apply(Command::CreatePrimitive {
            primitive: Primitive::Sphere { radius: 1.5 },
            name: None,
        })
        .unwrap();
    let id = applied.affected[0];
    assert_eq!(editor.project().scene.object(id).unwrap().name, "Sphere");

    editor.undo().unwrap();
    assert!(editor.project().scene.object(id).is_none());

    editor.redo().unwrap();
    assert!(
        editor.project().scene.object(id).is_some(),
        "redo minted a different id, so undo/redo no longer describe the same object"
    );

    // A second sphere gets a free name rather than colliding.
    let second = editor
        .apply(Command::CreatePrimitive {
            primitive: Primitive::Sphere { radius: 1.0 },
            name: None,
        })
        .unwrap();
    let second_id = second.affected[0];
    assert_eq!(
        editor.project().scene.object(second_id).unwrap().name,
        "Sphere.001"
    );
}

#[test]
fn importing_a_gltf_registers_the_asset_and_creates_the_object_atomically() {
    let mut editor = Editor::new(Project::new(UNTITLED));
    let applied = editor
        .apply(Command::ImportGltf {
            name: "duck.glb".to_owned(),
            path: "/models/duck.glb".to_owned(),
            kind: AssetKind::Glb,
            byte_size: 4096,
            node: None,
        })
        .unwrap();

    let object_id = applied.affected[0];
    assert_eq!(applied.label, "Import duck.glb");
    let project = editor.project();
    assert_eq!(project.assets.len(), 1);
    let object = project.scene.object(object_id).expect("object created");
    assert_eq!(object.name, "duck");
    let mesh = object.mesh().expect("mesh object");
    assert_eq!(mesh.asset_id(), Some(project.assets[0].id));
    assert_eq!(project.used_assets(), vec![project.assets[0].id]);

    // Undo removes both, in dependency order.
    editor.undo().unwrap();
    assert!(editor.project().assets.is_empty());
    assert!(editor.project().scene.object(object_id).is_none());

    // Redo restores the asset *before* the object that references it.
    editor.redo().unwrap();
    let project = editor.project();
    assert_eq!(project.assets.len(), 1);
    assert_eq!(project.scene.object(object_id).unwrap().name, "duck");
    project.validate().expect("project stays valid after redo");
}

#[test]
fn importing_the_same_file_twice_names_the_second_object_uniquely() {
    let mut editor = Editor::new(Project::new(UNTITLED));
    for _ in 0..2 {
        editor
            .apply(Command::ImportGltf {
                name: "duck.glb".to_owned(),
                path: "/models/duck.glb".to_owned(),
                kind: AssetKind::Glb,
                byte_size: 4096,
                node: None,
            })
            .unwrap();
    }
    let names: Vec<&str> = editor
        .project()
        .scene
        .objects
        .iter()
        .map(|object| object.name.as_str())
        .collect();
    assert!(names.contains(&"duck"), "{names:?}");
    assert!(names.contains(&"duck.001"), "{names:?}");
    assert_eq!(editor.project().assets.len(), 2);
}

#[test]
fn empty_imports_are_refused() {
    let mut editor = Editor::new(Project::new(UNTITLED));
    let error = editor
        .apply(Command::ImportGltf {
            name: "empty.glb".to_owned(),
            path: "/models/empty.glb".to_owned(),
            kind: AssetKind::Glb,
            byte_size: 0,
            node: None,
        })
        .unwrap_err()
        .to_string();
    assert!(error.contains("empty"), "{error}");
    assert!(editor.project().assets.is_empty());
    assert_eq!(editor.project().scene.objects.len(), 3);
}

#[test]
fn selecting_a_gltf_node_updates_the_source_and_is_reversible() {
    let mut editor = Editor::new(Project::new(UNTITLED));
    let applied = editor
        .apply(Command::ImportGltf {
            name: "duck.glb".to_owned(),
            path: "/models/duck.glb".to_owned(),
            kind: AssetKind::Glb,
            byte_size: 4096,
            node: None,
        })
        .unwrap();
    let id = applied.affected[0];

    let source_node = |editor: &Editor| -> Option<String> {
        let object = editor.project().scene.object(id).expect("imported object");
        let mesh = object.mesh().expect("mesh object");
        match &mesh.source {
            MeshSource::Gltf { node, .. } => node.clone(),
            other => panic!("expected gltf source, got {other:?}"),
        }
    };
    assert_eq!(
        source_node(&editor),
        None,
        "import starts with the whole file"
    );

    editor
        .apply(Command::SetGltfNode {
            id,
            node: Some("Body".to_owned()),
        })
        .unwrap();
    assert_eq!(source_node(&editor).as_deref(), Some("Body"));

    editor.undo().unwrap();
    assert_eq!(
        source_node(&editor),
        None,
        "undo restores the whole-file choice"
    );
    editor.redo().unwrap();
    assert_eq!(source_node(&editor).as_deref(), Some("Body"));

    editor
        .apply(Command::SetGltfNode { id, node: None })
        .unwrap();
    assert_eq!(
        source_node(&editor),
        None,
        "clearing returns to the whole file"
    );

    // Guards: a primitive has no nodes, and empty names are rejected.
    let cube = cube_id(&editor);
    let error = editor
        .apply(Command::SetGltfNode {
            id: cube,
            node: Some("Body".to_owned()),
        })
        .expect_err("a primitive has no nodes")
        .to_string();
    assert!(error.contains("not an imported glTF mesh"), "{error}");
    let error = editor
        .apply(Command::SetGltfNode {
            id,
            node: Some("  ".to_owned()),
        })
        .expect_err("empty node names are rejected")
        .to_string();
    assert!(error.contains("empty"), "{error}");
    assert_eq!(
        source_node(&editor),
        None,
        "a rejected set leaves the source alone"
    );
}
