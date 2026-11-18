//! Phase 6 acceptance: timeline, keyframes, interpolation, and the guarantee
//! that playback and offline rendering evaluate the same state.
//!
//! The plan's exit criteria, tested without a window or a GPU:
//! - edits change the canonical model and survive save/reload;
//! - the playhead is view state (not undoable, not autosave-worthy);
//! - one evaluation function drives both the session and the export path;
//! - camera animation reaches the handoff Forge3D renders from.

use studio_core::command::Command;
use studio_core::editor::Editor;
use studio_core::ids;
use studio_core::persist::{from_json, load, save_atomic};
use studio_core::project::{Project, UNTITLED};
use studio_core::scene::ObjectKind;
use studio_core::timeline::{self, Channel, ChannelKeys, Interpolation, Keyframe, Timeline};
use studio_core::workspace;
use studio_core::Workspace;

fn key(frame: u32, value: f32, interpolation: Interpolation) -> Keyframe {
    Keyframe {
        frame,
        value,
        interpolation,
    }
}

fn channel(channel: Channel, keys: Vec<Keyframe>) -> ChannelKeys {
    ChannelKeys { channel, keys }
}

/// A project with one animated mesh: translation X runs 0 → 100 over 10 frames.
fn animated_project() -> (Editor, ids::ObjectId) {
    let mut editor = Editor::new(Project::new(UNTITLED));
    let object_id = editor.project().scene.objects[0].id;
    editor
        .apply(Command::SetChannelKeys {
            id: object_id,
            channels: vec![channel(
                Channel::TranslationX,
                vec![
                    key(0, 0.0, Interpolation::Linear),
                    key(10, 100.0, Interpolation::Linear),
                ],
            )],
        })
        .unwrap();
    (editor, object_id)
}

#[test]
fn keying_channels_is_reversible_and_survives_save() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("phase6.studio3d");
    let (mut editor, object_id) = animated_project();

    // The edit is on the undo stack with a readable label.
    assert_eq!(editor.state().undo_label.as_deref(), Some("Key Cube"));
    let track = editor
        .project()
        .timeline
        .track(object_id)
        .expect("track exists");
    assert_eq!(track.channels.len(), 1);
    assert_eq!(track.channels[0].channel, Channel::TranslationX);

    // Undo removes the track; redo restores it.
    editor.undo().unwrap();
    assert!(editor.project().timeline.track(object_id).is_none());
    editor.redo().unwrap();
    assert!(editor.project().timeline.track(object_id).is_some());

    // Save → reload keeps the keyframes (the Phase 6 exit criterion: edits
    // change the canonical model and reload preserves them).
    save_atomic(&path, editor.project()).unwrap();
    let reloaded = load(&path).unwrap();
    let track = reloaded
        .timeline
        .track(object_id)
        .expect("track survives reload");
    assert_eq!(track.channels[0].keys.len(), 2);
    assert_eq!(track.channels[0].keys[1].value, 100.0);
    assert_eq!(reloaded.schema_version, studio_core::SCHEMA_VERSION);
}

#[test]
fn clearing_channels_removes_the_track_and_undo_brings_it_back() {
    let (mut editor, object_id) = animated_project();
    editor
        .apply(Command::SetChannelKeys {
            id: object_id,
            channels: Vec::new(),
        })
        .unwrap();
    assert!(editor.project().timeline.track(object_id).is_none());
    assert_eq!(
        editor.state().undo_label.as_deref(),
        Some("Clear animation of Cube")
    );

    editor.undo().unwrap();
    assert!(editor.project().timeline.track(object_id).is_some());
}

#[test]
fn channel_keys_reject_terrain_cameras_only_fov_and_bad_order() {
    let mut editor = Editor::new(Project::new(UNTITLED));
    let mesh_id = editor.project().scene.objects[0].id;
    let camera_id = editor
        .project()
        .scene
        .objects
        .iter()
        .find(|object| matches!(object.kind, ObjectKind::Camera(_)))
        .unwrap()
        .id;

    // Terrain cannot be animated (georeferenced, locked transform).
    let terrain_asset = studio_core::Asset {
        id: ids::new_id(),
        name: "dem.tif".to_owned(),
        path: "/d/dem.tif".to_owned(),
        kind: studio_core::AssetKind::Dem,
        byte_size: 10,
    };
    editor
        .apply(Command::AddAsset {
            asset: terrain_asset.clone(),
        })
        .unwrap();
    let terrain = studio_core::scene::TerrainObject::new(terrain_asset.id);
    let terrain_id = ids::new_id();
    editor
        .apply(Command::CreateObject {
            object: studio_core::scene::SceneObject {
                id: terrain_id,
                name: "Terrain".to_owned(),
                parent: None,
                transform: studio_core::Transform::default(),
                visible: true,
                locked: false,
                kind: ObjectKind::Terrain(terrain),
            },
            index: None,
        })
        .unwrap();
    let error = editor
        .apply(Command::SetChannelKeys {
            id: terrain_id,
            channels: vec![channel(
                Channel::TranslationX,
                vec![key(0, 0.0, Interpolation::Linear)],
            )],
        })
        .expect_err("terrain cannot be keyed")
        .to_string();
    assert!(error.contains("georeferenced"), "{error}");

    // FOV only exists on a camera.
    let error = editor
        .apply(Command::SetChannelKeys {
            id: mesh_id,
            channels: vec![channel(
                Channel::FovDeg,
                vec![key(0, 45.0, Interpolation::Linear)],
            )],
        })
        .expect_err("meshes have no FOV")
        .to_string();
    assert!(error.contains("camera"), "{error}");

    // A camera accepts FOV keys.
    editor
        .apply(Command::SetChannelKeys {
            id: camera_id,
            channels: vec![channel(
                Channel::FovDeg,
                vec![key(0, 45.0, Interpolation::Linear)],
            )],
        })
        .unwrap();

    // Out-of-range FOV and unsorted keys are rejected by validation.
    let error = editor
        .apply(Command::SetChannelKeys {
            id: camera_id,
            channels: vec![channel(
                Channel::FovDeg,
                vec![key(0, 200.0, Interpolation::Linear)],
            )],
        })
        .expect_err("200 degrees is not a usable FOV")
        .to_string();
    assert!(error.contains("valid range"), "{error}");
    let error = editor
        .apply(Command::SetChannelKeys {
            id: mesh_id,
            channels: vec![channel(
                Channel::TranslationX,
                vec![
                    key(5, 0.0, Interpolation::Linear),
                    key(5, 1.0, Interpolation::Linear),
                ],
            )],
        })
        .expect_err("duplicate frames are ambiguous")
        .to_string();
    assert!(error.contains("strictly increasing"), "{error}");

    // A missing object is a distinct, specific error.
    let error = editor
        .apply(Command::SetChannelKeys {
            id: ids::new_id(),
            channels: vec![channel(
                Channel::TranslationX,
                vec![key(0, 0.0, Interpolation::Linear)],
            )],
        })
        .expect_err("no such object")
        .to_string();
    assert!(error.contains("does not exist"), "{error}");
}

#[test]
fn playhead_is_view_state_not_an_edit() {
    let (mut editor, object_id) = animated_project();
    editor.mark_saved(); // isolate the playhead's own effect on dirty state
    let revision_before = editor.revision();
    let history_before = editor.state().history_len;

    // Scrub to the midpoint: the object moves, the model evaluates.
    let applied = editor.preview_playhead(5.0);
    assert_eq!(applied, 5.0);
    let object = editor.project().scene.object(object_id).unwrap();
    assert_eq!(object.transform.translation.x, 50.0);
    assert_eq!(editor.project().timeline.current_frame, 5.0);

    // No revision bump, no dirty flag, no undo entry: scrubbing is not an
    // edit, so autosave stays quiet and the undo stack stays honest.
    assert_eq!(
        editor.revision(),
        revision_before,
        "playhead must not bump revision"
    );
    assert_eq!(
        editor.state().history_len,
        history_before,
        "playhead must not push undo"
    );
    assert!(
        !editor.is_dirty(),
        "playhead must not mark the project dirty"
    );

    // Out-of-range frames clamp instead of failing.
    assert_eq!(
        editor.preview_playhead(10_000.0),
        editor.project().timeline.length as f32
    );
    assert_eq!(editor.preview_playhead(-1.0), 0.0);
    assert_eq!(editor.preview_playhead(f32::NAN), 0.0);

    // The last real command is still what undo reverses.
    editor.undo().unwrap();
    assert!(editor.project().timeline.track(object_id).is_none());
}

#[test]
fn playback_and_export_share_one_evaluation_function() {
    // The Phase 6 exit criterion: what you scrub is what renders. Both the
    // session path (preview_playhead) and the export path (a clone + evaluate,
    // exactly what render_animation_frame does) call timeline::evaluate, so
    // this asserts they cannot drift: same frame in → same model out.
    let (mut editor, _) = animated_project();
    let mut exported = editor.project().clone();

    for frame in [0.0, 2.5, 5.0, 9.99, 10.0, 42.0] {
        editor.preview_playhead(frame);
        timeline::evaluate(&mut exported, frame);
        assert_eq!(
            editor.project().scene,
            exported.scene,
            "playback and export diverged at frame {frame}"
        );
        assert_eq!(
            editor.project().timeline.current_frame,
            exported.timeline.current_frame
        );
    }
}

fn bryce_seed() -> studio_core::TerrainSeed {
    studio_core::TerrainSeed {
        title: Some("Bryce Canyon".to_owned()),
        name: "Bryce_Canyon_1024.tif".to_owned(),
        path: "/data/Bryce_Canyon_1024.tif".to_owned(),
        byte_size: 3_321_356,
        dimensions: Some([1024, 1024]),
        elevation_range: Some([2099.23, 2553.58]),
        world_origin_xz: Some([395_998.5, -4_165_574.5]),
        world_span_xz: Some([4000.0, 3800.0]),
        overlay: None,
    }
}

#[test]
fn camera_animation_reaches_the_render_handoff() {
    // Camera animation handed to Forge3D: key the camera's position, evaluate,
    // and confirm the handoff's terrain camera (phi/theta/fov) moves with it.
    // The terrain camera only exists on a terrain scene, so seed one.
    let mut editor = Editor::new(Project::starter(UNTITLED, Some(bryce_seed())));
    let camera_id = editor
        .project()
        .scene
        .objects
        .iter()
        .find(|object| matches!(object.kind, ObjectKind::Camera(_)))
        .unwrap()
        .id;

    let start = workspace::handoff(editor.project(), 0, Workspace::Forge3d)
        .terrain_camera
        .expect("terrain scene has a terrain camera");

    // Key the camera's CURRENT position at frame 0 (so returning to frame 0
    // must reproduce the start exactly), then orbit it over 24 frames and
    // zoom the FOV in. The first key comes from the object itself — the same
    // "key current values" payload the UI's Key button builds.
    let camera = editor.project().scene.object(camera_id).unwrap();
    let (x0, z0, fov0) = (
        camera.transform.translation.x,
        camera.transform.translation.z,
        timeline::channel_value(camera, Channel::FovDeg).expect("camera has an FOV"),
    );
    editor
        .apply(Command::SetChannelKeys {
            id: camera_id,
            channels: vec![
                channel(
                    Channel::TranslationX,
                    vec![
                        key(0, x0, Interpolation::Smooth),
                        key(24, x0 + 14.0, Interpolation::Smooth),
                    ],
                ),
                channel(
                    Channel::TranslationZ,
                    vec![
                        key(0, z0, Interpolation::Smooth),
                        key(24, z0 - 16.0, Interpolation::Smooth),
                    ],
                ),
                channel(
                    Channel::FovDeg,
                    vec![
                        key(0, fov0, Interpolation::Linear),
                        key(24, 20.0, Interpolation::Linear),
                    ],
                ),
            ],
        })
        .unwrap();

    editor.preview_playhead(24.0);
    let end = workspace::handoff(editor.project(), 1, Workspace::Forge3d)
        .terrain_camera
        .expect("terrain camera after evaluation");

    // The orbit angle moved and the FOV followed its keys.
    assert!(
        (end.phi_deg - start.phi_deg).abs() > 1.0,
        "camera orbit must move: {} -> {}",
        start.phi_deg,
        end.phi_deg
    );
    assert!(
        (end.fov_deg - 20.0).abs() < 1e-3,
        "FOV keys must reach the handoff"
    );

    // The evaluation is deterministic: scrubbing back reproduces the start.
    editor.preview_playhead(0.0);
    let back = workspace::handoff(editor.project(), 2, Workspace::Forge3d)
        .terrain_camera
        .unwrap();
    assert!(
        (back.phi_deg - start.phi_deg).abs() < 1e-3,
        "returning to frame 0 must reproduce the original framing"
    );
}

#[test]
fn camera_rotation_animation_reaches_the_exported_render_view() {
    // The export used to read `terrain_camera`, whose angles come from the
    // camera's POSITION relative to the terrain: it ignores the object's
    // rotation and distance, so a rotation-only animation rendered a static
    // frame. `render_animation_frame` now hands `camera_render_view` to
    // Forge3D; this is the regression that keeps it honest.
    let mut editor = Editor::new(Project::starter(UNTITLED, Some(bryce_seed())));
    let camera_id = editor
        .project()
        .scene
        .objects
        .iter()
        .find(|object| matches!(object.kind, ObjectKind::Camera(_)))
        .unwrap()
        .id;
    let rotation_y = editor
        .project()
        .scene
        .object(camera_id)
        .unwrap()
        .transform
        .rotation_deg
        .y;
    let start = workspace::camera_render_view(editor.project()).expect("terrain + camera");

    editor
        .apply(Command::SetChannelKeys {
            id: camera_id,
            channels: vec![channel(
                Channel::RotationY,
                vec![
                    key(0, rotation_y, Interpolation::Linear),
                    key(12, rotation_y + 90.0, Interpolation::Linear),
                ],
            )],
        })
        .unwrap();

    editor.preview_playhead(12.0);
    let end = workspace::camera_render_view(editor.project()).expect("view at frame 12");
    assert!(
        (end.phi_deg - start.phi_deg).abs() > 1.0,
        "camera rotation must move the exported orbit: {} -> {}",
        start.phi_deg,
        end.phi_deg
    );
    // Rotation alone leaves the eye fixed and the aim ten units ahead, so the
    // radius is unchanged — it is the camera's distance, not the DEM framing.
    assert!(
        (end.radius - start.radius).abs() < 1e-2,
        "rotation must not change the orbit radius: {} -> {}",
        start.radius,
        end.radius
    );

    // Deterministic: scrubbing back reproduces the original view.
    editor.preview_playhead(0.0);
    let back = workspace::camera_render_view(editor.project()).unwrap();
    assert!((back.phi_deg - start.phi_deg).abs() < 1e-3);
    assert!((back.theta_deg - start.theta_deg).abs() < 1e-3);
    assert_eq!(back.target, start.target);
}

#[test]
fn timeline_config_edits_are_validated_and_reversible() {
    let mut editor = Editor::new(Project::new(UNTITLED));

    editor
        .apply(Command::SetTimelineConfig {
            frame_rate: 30,
            length: 150,
        })
        .unwrap();
    assert_eq!(editor.project().timeline.frame_rate, 30);
    assert_eq!(editor.project().timeline.length, 150);
    assert_eq!(
        editor.state().undo_label.as_deref(),
        Some("Timeline settings")
    );

    // Invalid configurations are rejected before they touch the model.
    for (rate, length, expected) in [
        (0u32, 96u32, "frame rate"),
        (24, 0, "length"),
        (1000, 96, "frame rate"),
        (24, 1_000_000, "length"),
    ] {
        let error = editor
            .apply(Command::SetTimelineConfig {
                frame_rate: rate,
                length,
            })
            .expect_err("invalid timeline config")
            .to_string();
        assert!(error.contains(expected), "wanted {expected:?} in: {error}");
    }
    assert_eq!(
        editor.project().timeline.frame_rate,
        30,
        "rejects leave no trace"
    );

    // Shrinking below the playhead parks it inside the new range.
    editor.preview_playhead(140.0);
    editor
        .apply(Command::SetTimelineConfig {
            frame_rate: 24,
            length: 50,
        })
        .unwrap();
    assert_eq!(editor.project().timeline.current_frame, 50.0);

    editor.undo().unwrap();
    assert_eq!(editor.project().timeline.length, 150);
}

#[test]
fn v3_documents_migrate_with_a_default_timeline() {
    // A v3 document (no timeline key) loads into this build with the default
    // timeline and the bumped version; a v4 document is rejected by older
    // builds (the version gate) instead of silently dropping keyframes.
    let mut value = serde_json::to_value(Project::new(UNTITLED)).unwrap();
    value["schemaVersion"] = serde_json::json!(3);
    value.as_object_mut().unwrap().remove("timeline");
    let text = serde_json::to_string_pretty(&value).unwrap();

    let migrated = from_json(&text, "v3 fixture").unwrap();
    assert_eq!(migrated.schema_version, studio_core::SCHEMA_VERSION);
    assert_eq!(migrated.timeline, Timeline::default());

    // And a document from the future still fails loudly.
    let mut future = serde_json::to_value(Project::new(UNTITLED)).unwrap();
    future["schemaVersion"] = serde_json::json!(studio_core::SCHEMA_VERSION + 1);
    let error = from_json(&serde_json::to_string(&future).unwrap(), "future fixture")
        .expect_err("newer schema must be rejected")
        .to_string();
    assert!(error.contains("schema version"), "{error}");
}

#[test]
fn playhead_state_round_trips_through_the_session() {
    // The shell's set_playhead path end to end: key channels through the
    // session, scrub through `set_playhead`, save, reload.
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("session.studio3d");

    let mut session = studio_core::Session::new();
    session.new_project(None);
    let object_id = session.state().unwrap().project.scene.objects[0].id;
    session
        .apply(Command::SetChannelKeys {
            id: object_id,
            channels: vec![channel(
                Channel::TranslationX,
                vec![
                    key(0, 0.0, Interpolation::Linear),
                    key(10, 100.0, Interpolation::Linear),
                ],
            )],
        })
        .unwrap();

    let revision_before = session.state().unwrap().revision;
    let (frame, state) = session.set_playhead(7.5).unwrap();
    assert_eq!(frame, 7.5);
    assert_eq!(state.project.timeline.current_frame, 7.5);
    assert_eq!(
        state
            .project
            .scene
            .object(object_id)
            .unwrap()
            .transform
            .translation
            .x,
        75.0
    );
    // View state: scrubbing adds no undo entries and does not itself dirty
    // the project (the keying command already did, and stays the only undo).
    assert_eq!(state.history_len, 1, "scrubbing must not add undo entries");
    assert_eq!(
        state.revision, revision_before,
        "scrubbing must not bump revision"
    );

    save_atomic(&path, &state.project).unwrap();
    let reloaded = load(&path).unwrap();
    assert_eq!(reloaded.timeline.current_frame, 7.5);
    assert_eq!(
        reloaded
            .scene
            .object(object_id)
            .unwrap()
            .transform
            .translation
            .x,
        75.0
    );
}
