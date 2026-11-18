//! Phase 1 acceptance: a project opens, saves, closes, and reopens with stable
//! IDs and transforms, and every edit is reversible.
//!
//! These are the guarantees the desktop shell depends on, exercised without
//! Tauri, a window, or a GPU.

use std::fs;
use std::path::Path;

use studio_core::autosave::AutosaveStore;
use studio_core::command::Command;
use studio_core::editor::Editor;
use studio_core::math::{Transform, Vec3};
use studio_core::persist::{load, save_atomic};
use studio_core::project::{Project, UNTITLED};
use studio_core::StudioError;

fn ids_of_kind(project: &Project, kind: &str) -> Vec<uuid::Uuid> {
    project
        .scene
        .objects
        .iter()
        .filter(|object| object.kind_label() == kind)
        .map(|object| object.id)
        .collect()
}

#[test]
fn save_close_reopen_preserves_ids_and_transforms() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("round-trip.studio3d");

    let mut editor = Editor::new(Project::new("Round trip"));
    let cube = ids_of_kind(editor.project(), "mesh")[0];
    let edited = Transform {
        translation: Vec3::new(2.0, 1.0, -3.5),
        rotation_deg: Vec3::new(10.0, 20.0, 30.0),
        scale: Vec3::new(1.0, 0.5, 2.0),
    };
    editor
        .apply(Command::SetTransform {
            id: cube,
            transform: edited,
        })
        .unwrap();
    editor.set_selection(vec![cube]).unwrap();
    let before_save = editor.project().clone();
    save_atomic(&path, editor.project()).unwrap();
    editor.mark_saved();
    assert!(!editor.is_dirty());

    // "Close" is dropping the editor; reopen from disk in a fresh session.
    drop(editor);
    let reopened = load(&path).unwrap();

    assert_eq!(reopened, before_save, "reopened project differs from saved");
    assert_eq!(reopened.id, before_save.id);
    assert_eq!(reopened.scene.id, before_save.scene.id);
    let cube_after = reopened.scene.object(cube).expect("stable object id");
    assert_eq!(cube_after.transform, edited);
    assert_eq!(reopened.selection, vec![cube]);

    let opened = Editor::opened(reopened, &path);
    assert!(!opened.is_dirty());
    assert!(!opened.can_undo());
}

#[test]
fn edits_are_fully_reversible_including_deletion_of_the_active_camera() {
    let mut editor = Editor::new(Project::new(UNTITLED));
    // Compare the serialized scene: it is the part undo must restore exactly.
    // (Timestamps are metadata and legitimately move.)
    let before = serde_json::to_string(&editor.project().scene).unwrap();

    let camera = ids_of_kind(editor.project(), "camera")[0];
    editor.apply(Command::DeleteObject { id: camera }).unwrap();
    assert!(editor.project().scene.active_camera.is_none());
    assert!(editor.project().scene.camera().is_none());

    editor.undo().unwrap();
    assert_eq!(editor.project().scene.active_camera, Some(camera));
    assert!(editor.project().scene.camera().is_some());
    assert_eq!(
        serde_json::to_string(&editor.project().scene).unwrap(),
        before,
        "undo did not restore the scene byte-for-byte"
    );
}

#[test]
fn a_batch_undoes_as_one_step() {
    let mut editor = Editor::new(Project::new(UNTITLED));
    let cube = ids_of_kind(editor.project(), "mesh")[0];
    let original = editor.project().scene.object(cube).unwrap().clone();

    let batch = Command::Batch {
        commands: vec![
            Command::RenameObject {
                id: cube,
                name: "Crate".to_owned(),
            },
            Command::SetVisibility {
                id: cube,
                visible: false,
            },
            Command::SetTransform {
                id: cube,
                transform: Transform {
                    translation: Vec3::new(0.0, 3.0, 0.0),
                    ..Transform::default()
                },
            },
        ],
    };
    editor.apply(batch).unwrap();
    let edited = editor.project().scene.object(cube).unwrap().clone();
    assert_eq!(edited.name, "Crate");
    assert!(!edited.visible);

    editor.undo().unwrap();
    let restored = editor.project().scene.object(cube).unwrap();
    assert_eq!(restored, &original);
    assert!(!editor.can_undo());
}

#[test]
fn failed_commands_leave_the_project_untouched() {
    let mut editor = Editor::new(Project::new(UNTITLED));
    let before = editor.project().clone();
    let cube = ids_of_kind(&before, "mesh")[0];

    // A camera payload on a mesh is rejected before any mutation.
    let error = editor
        .apply(Command::SetCamera {
            id: cube,
            camera: studio_core::CameraObject {
                fov_deg: 45.0,
                near: 0.1,
                far: 100.0,
            },
        })
        .unwrap_err();
    assert!(matches!(
        error,
        StudioError::Command(studio_core::CommandError::NotACamera { .. })
    ));

    // Out-of-range values are rejected with a message that names the field.
    let error = editor
        .apply(Command::SetCamera {
            id: ids_of_kind(&before, "camera")[0],
            camera: studio_core::CameraObject {
                fov_deg: 500.0,
                near: 0.1,
                far: 100.0,
            },
        })
        .unwrap_err()
        .to_string();
    assert!(error.contains("field of view"), "{error}");

    assert_eq!(editor.project().scene.objects, before.scene.objects);
    assert!(!editor.can_undo());
}

#[test]
fn deleting_a_parent_is_refused_until_children_are_gone() {
    let mut editor = Editor::new(Project::new(UNTITLED));
    let cube = ids_of_kind(editor.project(), "mesh")[0];
    let child = studio_core::SceneObject {
        id: studio_core::ids::new_id(),
        name: "Child".to_owned(),
        parent: Some(cube),
        transform: Transform::default(),
        visible: true,
        locked: false,
        kind: studio_core::ObjectKind::Mesh(studio_core::MeshObject::primitive(
            studio_core::Primitive::Cube { size: 1.0 },
        )),
    };
    editor
        .apply(Command::CreateObject {
            object: child.clone(),
            index: None,
        })
        .unwrap();

    let error = editor
        .apply(Command::DeleteObject { id: cube })
        .unwrap_err();
    assert!(error.to_string().contains("has children"), "{error}");

    editor
        .apply(Command::DeleteObject { id: child.id })
        .unwrap();
    editor.apply(Command::DeleteObject { id: cube }).unwrap();
    assert_eq!(editor.project().scene.objects.len(), 2);
}

#[test]
fn save_is_atomic_and_leaves_no_temporaries() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("atomic.studio3d");
    let project = Project::new(UNTITLED);

    save_atomic(&path, &project).unwrap();
    save_atomic(&path, &project).unwrap();

    let names: Vec<String> = fs::read_dir(dir.path())
        .unwrap()
        .flatten()
        .map(|entry| entry.file_name().to_string_lossy().into_owned())
        .collect();
    assert_eq!(names, vec!["atomic.studio3d".to_owned()]);
}

#[test]
fn a_failed_save_keeps_the_previous_file_intact() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("project.studio3d");
    let project = Project::new(UNTITLED);
    save_atomic(&path, &project).unwrap();
    let original = fs::read_to_string(&path).unwrap();

    // Make the target unusable: a directory cannot be replaced by rename.
    let blocked = dir.path().join("blocked.studio3d");
    fs::create_dir(&blocked).unwrap();
    let error = save_atomic(&blocked, &project).unwrap_err();
    assert!(!error.to_string().is_empty());
    assert!(blocked.is_dir());
    assert_eq!(fs::read_to_string(&path).unwrap(), original);

    // And no temporary from the failed write survived.
    let leftovers: Vec<String> = fs::read_dir(dir.path())
        .unwrap()
        .flatten()
        .map(|entry| entry.file_name().to_string_lossy().into_owned())
        .filter(|name| name.ends_with(".tmp"))
        .collect();
    assert!(leftovers.is_empty(), "left behind: {leftovers:?}");
}

#[test]
fn newer_projects_do_not_load_and_are_not_rewritten() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("future.studio3d");
    let mut value = serde_json::to_value(Project::new(UNTITLED)).unwrap();
    value["schemaVersion"] = serde_json::json!(studio_core::SCHEMA_VERSION + 1);
    let text = serde_json::to_string_pretty(&value).unwrap();
    fs::write(&path, &text).unwrap();

    let error = load(&path).unwrap_err();
    assert!(
        error.to_string().contains(&format!(
            "schema version {}",
            studio_core::SCHEMA_VERSION + 1
        )),
        "{error}"
    );
    assert_eq!(fs::read_to_string(&path).unwrap(), text);
}

#[test]
fn autosave_snapshots_recover_the_unsaved_state() {
    let dir = tempfile::tempdir().unwrap();
    let store = AutosaveStore::new(dir.path().join("recovery"));
    let mut editor = Editor::new(Project::new(UNTITLED));
    let cube = ids_of_kind(editor.project(), "mesh")[0];
    editor
        .apply(Command::RenameObject {
            id: cube,
            name: "Unsaved Crate".to_owned(),
        })
        .unwrap();

    let entry = store.write(None, editor.project()).unwrap();
    let listed = store.list().unwrap();
    assert_eq!(listed.len(), 1);
    assert_eq!(listed[0].id, entry.id);

    let recovered = store.load(&entry.id).unwrap();
    assert_eq!(recovered.scene.object(cube).unwrap().name, "Unsaved Crate");

    store.discard(&entry.id).unwrap();
    assert!(store.list().unwrap().is_empty());
}

#[test]
fn project_file_has_no_absolute_paths_or_process_handles() {
    let json = serde_json::to_string(&Project::new(UNTITLED)).unwrap();
    assert!(!json.contains("C:\\\\"), "{json}");
    assert!(!json.contains("\"/home/"), "{json}");
    assert!(!json.contains("pid"), "{json}");
}

#[test]
fn extension_constant_matches_saved_file_names() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir
        .path()
        .join(format!("named.{}", studio_core::persist::PROJECT_EXTENSION));
    save_atomic(&path, &Project::new(UNTITLED)).unwrap();
    assert!(Path::new(&path).exists());
}

#[test]
fn materials_and_environment_settings_survive_save_and_reopen() {
    // Phase 5 exit criterion: materials and render configuration persist and
    // reload predictably — one mesh material edit plus the environment knobs.
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("phase5.studio3d");
    let mut editor = Editor::new(Project::new(UNTITLED));
    let cube = ids_of_kind(editor.project(), "mesh")[0];

    editor
        .apply(Command::SetMeshMaterial {
            id: cube,
            material: studio_core::Material {
                base_color: [0.9, 0.1, 0.1],
                roughness: 0.33,
                metallic: 0.5,
                texture: None,
            },
        })
        .unwrap();
    let mut render = editor.project().render.clone();
    render.sky = false;
    render.ibl_intensity = 0.4;
    editor.apply(Command::SetRenderSettings { render }).unwrap();

    save_atomic(&path, editor.project()).unwrap();
    let reloaded = load(&path).unwrap();

    let mesh = reloaded
        .scene
        .objects
        .iter()
        .find(|object| object.id == cube)
        .expect("cube survives");
    let material = mesh.mesh().expect("mesh kind").material;
    assert_eq!(material.base_color, [0.9, 0.1, 0.1]);
    assert!((material.roughness - 0.33).abs() < 1e-6);
    assert!(!reloaded.render.sky, "sky setting persists");
    assert!((reloaded.render.ibl_intensity - 0.4).abs() < 1e-6);
}
