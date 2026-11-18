//! Phase 7 performance guards for the canonical model.
//!
//! These are regression budgets, not micro-benchmarks: the numbers are loose
//! enough for a busy CI runner and only fail if a change makes the model
//! orders of magnitude slower. The measured value is printed so a human (or a
//! future test) can see the trend.
//!
//! `cargo test -p studio-core --test performance -- --nocapture` prints them.
//! `cargo test -p studio-core --test performance -- --ignored` runs the
//! 10k-object stress case that is deliberately not part of CI.

use std::time::Instant;

use studio_core::project::{Project, UNTITLED};
use studio_core::scene::{MeshObject, ObjectKind, Primitive, SceneObject};
use studio_core::timeline::{self, Channel, ChannelKeys, Interpolation, Keyframe, ObjectTrack};
use studio_core::{ids, persist, Transform};

/// Milliseconds a change is allowed to spend on the interaction-path work.
/// Generous by two orders of magnitude: this catches an accidental O(n²), not
/// a slow machine.
const EVALUATE_BUDGET_MS: u128 = 5_000;
const SAVE_LOAD_BUDGET_MS: u128 = 10_000;

fn object(index: usize) -> SceneObject {
    SceneObject {
        id: ids::new_id(),
        name: format!("Object {index}"),
        parent: None,
        transform: Transform::default(),
        visible: true,
        locked: false,
        kind: ObjectKind::Mesh(MeshObject::primitive(Primitive::Cube { size: 1.0 })),
    }
}

fn channel(channel: Channel, keys: usize) -> ChannelKeys {
    ChannelKeys {
        channel,
        // Strictly increasing frames, finite values: a valid track.
        keys: (0..keys)
            .map(|index| Keyframe {
                frame: (index * 2) as u32,
                value: index as f32,
                interpolation: Interpolation::Linear,
            })
            .collect(),
    }
}

/// A scene with `objects` objects, `keyed` of them animated on three channels
/// with `keys` keyframes each, over a 480-frame timeline.
fn large_project(objects: usize, keyed: usize, keys: usize) -> Project {
    let mut project = Project::new(UNTITLED);
    project.scene.objects.clear();
    project.scene.active_camera = None;
    for index in 0..objects {
        project.scene.objects.push(object(index));
    }
    project.timeline.length = 480;
    for scene_object in project.scene.objects.iter().take(keyed) {
        project.timeline.tracks.push(ObjectTrack {
            object_id: scene_object.id,
            channels: vec![
                channel(Channel::TranslationX, keys),
                channel(Channel::TranslationY, keys),
                channel(Channel::RotationZ, keys),
            ],
        });
    }
    project
}

#[test]
fn evaluating_a_large_keyed_scene_stays_interactive() {
    let mut project = large_project(2_000, 500, 40);
    // 500 objects × 3 channels × 240 frames = 360k channel samples, each a
    // scan over ≤ 40 keys.
    project
        .validate()
        .expect("the generated scene must be valid");

    let started = Instant::now();
    for frame in 0..240u32 {
        timeline::evaluate(&mut project, frame as f32);
    }
    let elapsed = started.elapsed().as_millis();
    eprintln!(
        "evaluate: 2 000 objects, 500 animated × 3 channels × 40 keys, 240 frames in {elapsed} ms"
    );

    // Correctness rides along with the timing: 40 keys at frames 0..78 hold
    // their last value afterwards, so frame 239 must be 39 — not a stale or
    // uninitialised transform.
    let first = &project.timeline.tracks[0];
    let animated = project.scene.object(first.object_id).unwrap();
    assert_eq!(animated.transform.translation.x, 39.0);
    assert!(project.timeline.current_frame == 239.0);

    assert!(
        elapsed < EVALUATE_BUDGET_MS,
        "evaluating 240 frames took {elapsed} ms (budget {EVALUATE_BUDGET_MS} ms)"
    );
}

#[test]
fn saving_and_loading_a_large_scene_round_trips() {
    let project = large_project(2_000, 500, 40);
    project.validate().unwrap();
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("large.studio3d");

    let started = Instant::now();
    persist::save_atomic(&path, &project).unwrap();
    let loaded = persist::load(&path).unwrap();
    let elapsed = started.elapsed().as_millis();
    let bytes = std::fs::metadata(&path).unwrap().len();
    eprintln!("save+load: 2 000 objects, 500 tracks in {elapsed} ms ({bytes} bytes)");

    assert_eq!(loaded.scene, project.scene);
    assert_eq!(loaded.timeline, project.timeline);
    assert!(
        elapsed < SAVE_LOAD_BUDGET_MS,
        "save+load took {elapsed} ms (budget {SAVE_LOAD_BUDGET_MS} ms)"
    );
}

#[test]
#[ignore = "stress case (10k objects, 2000 tracks); run with --ignored"]
fn a_ten_thousand_object_scene_still_evaluates() {
    let mut project = large_project(10_000, 2_000, 200);
    project.validate().unwrap();

    let started = Instant::now();
    for frame in (0..1_000).step_by(10) {
        timeline::evaluate(&mut project, frame as f32);
    }
    let elapsed = started.elapsed().as_millis();
    eprintln!("stress: 10 000 objects, 2 000 animated × 200 keys, 100 frames in {elapsed} ms");
    assert!(elapsed < 60_000, "stress evaluation took {elapsed} ms");
}
