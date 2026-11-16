//! Timeline: keyframes, interpolation, and evaluation in the canonical model.
//!
//! Phase 6's core guarantee is that **playback and offline frame rendering
//! evaluate the same state**: there is exactly one [`evaluate`] function, the
//! session's playhead calls it, and the shell's frame-sequence export calls it
//! on a project clone. Nothing here lives in the viewport.
//!
//! A timeline is a list of per-object tracks. Each track lists the channels it
//! animates (the nine transform scalars, plus a camera's field of view), and
//! each channel holds its keyframes. Channels a track does not mention keep the
//! object's static value; objects with no track are untouched by evaluation.

use serde::{Deserialize, Serialize};

use crate::ids::ObjectId;
use crate::project::Project;
use crate::scene::{ObjectKind, SceneObject};

/// Frames per second of the default timeline.
pub const DEFAULT_FRAME_RATE: u32 = 24;
/// Length in frames of the default timeline (4 s at 24 fps).
pub const DEFAULT_LENGTH: u32 = 96;
/// Upper bound on the frame rate, so a typo cannot make playback a busy loop.
pub const MAX_FRAME_RATE: u32 = 240;
/// Upper bound on timeline length (a little over an hour at 24 fps).
pub const MAX_LENGTH: u32 = 100_000;

/// How the value moves between two keyframes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum Interpolation {
    /// Straight-line between the keys.
    #[default]
    Linear,
    /// Hold the left key's value until the next keyframe.
    Constant,
    /// Ease in and out with a smoothstep curve.
    Smooth,
}

/// One animatable scalar. Nine transform components plus the camera FOV —
/// everything the current renderers actually read off an object.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum Channel {
    TranslationX,
    TranslationY,
    TranslationZ,
    RotationX,
    RotationY,
    RotationZ,
    ScaleX,
    ScaleY,
    ScaleZ,
    /// Camera field of view, degrees. Rejected on non-camera objects.
    FovDeg,
}

impl Channel {
    /// Every channel in a stable, UI-friendly order.
    pub const ALL: [Channel; 10] = [
        Channel::TranslationX,
        Channel::TranslationY,
        Channel::TranslationZ,
        Channel::RotationX,
        Channel::RotationY,
        Channel::RotationZ,
        Channel::ScaleX,
        Channel::ScaleY,
        Channel::ScaleZ,
        Channel::FovDeg,
    ];

    /// Short label for the timeline UI and error messages.
    pub fn label(self) -> &'static str {
        match self {
            Channel::TranslationX => "position X",
            Channel::TranslationY => "position Y",
            Channel::TranslationZ => "position Z",
            Channel::RotationX => "rotation X",
            Channel::RotationY => "rotation Y",
            Channel::RotationZ => "rotation Z",
            Channel::ScaleX => "scale X",
            Channel::ScaleY => "scale Y",
            Channel::ScaleZ => "scale Z",
            Channel::FovDeg => "field of view",
        }
    }
}

/// One keyframe: the value of a channel at an integer frame.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Keyframe {
    /// Integer frame the key sits on.
    pub frame: u32,
    /// Value at that frame. Finite only (validated on the way in).
    pub value: f32,
    /// How the value moves from this key to the next.
    pub interpolation: Interpolation,
}

/// All keyframes of one channel on one object's track.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ChannelKeys {
    pub channel: Channel,
    /// Sorted by `frame`, strictly increasing (no duplicate frames).
    pub keys: Vec<Keyframe>,
}

/// One object's animation: which channels are keyed, and how.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ObjectTrack {
    pub object_id: ObjectId,
    pub channels: Vec<ChannelKeys>,
}

/// The project's timeline. Persisted with the document (schema v4).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Timeline {
    /// Frames per second for playback and export timing.
    pub frame_rate: u32,
    /// Total frames in the timeline. Playback and export run `0..=length`.
    pub length: u32,
    /// Where the playhead sits. View state: moving it is not an edit.
    pub current_frame: f32,
    /// One entry per animated object. Objects may be deleted and re-created;
    /// a track whose object is missing is kept (undo restores it) and simply
    /// skipped during evaluation.
    pub tracks: Vec<ObjectTrack>,
}

impl Default for Timeline {
    fn default() -> Self {
        Self {
            frame_rate: DEFAULT_FRAME_RATE,
            length: DEFAULT_LENGTH,
            current_frame: 0.0,
            tracks: Vec::new(),
        }
    }
}

impl Timeline {
    /// Validate structure: rates, lengths, key ordering, and value ranges.
    /// Called from `Project::validate`, so a corrupt document fails to load
    /// with a specific message instead of rendering garbage.
    pub fn validate(&self) -> Result<(), String> {
        if self.frame_rate == 0 || self.frame_rate > MAX_FRAME_RATE {
            return Err(format!(
                "timeline frame rate must be between 1 and {MAX_FRAME_RATE}"
            ));
        }
        if self.length == 0 || self.length > MAX_LENGTH {
            return Err(format!(
                "timeline length must be between 1 and {MAX_LENGTH} frames"
            ));
        }
        if !self.current_frame.is_finite() || self.current_frame < 0.0 {
            return Err("timeline current frame must be a finite, non-negative number".to_owned());
        }
        if self.current_frame > self.length as f32 {
            return Err(format!(
                "timeline current frame {} is past the timeline length {}",
                self.current_frame, self.length
            ));
        }
        for track in &self.tracks {
            for keys in &track.channels {
                if keys.keys.is_empty() {
                    return Err(format!(
                        "channel {} on track {} has no keyframes",
                        keys.channel.label(),
                        crate::ids::short_id(&track.object_id)
                    ));
                }
                let mut previous: Option<u32> = None;
                for key in &keys.keys {
                    if !key.value.is_finite() {
                        return Err(format!(
                            "keyframe at frame {} on {} must be a finite value",
                            key.frame,
                            keys.channel.label()
                        ));
                    }
                    if key.channel_out_of_range(keys.channel) {
                        return Err(format!(
                            "keyframe at frame {} on {} is outside the valid range",
                            key.frame,
                            keys.channel.label()
                        ));
                    }
                    if let Some(previous) = previous {
                        if key.frame <= previous {
                            return Err(format!(
                                "keyframes on {} must be strictly increasing (frame {} follows {previous})",
                                keys.channel.label(),
                                key.frame
                            ));
                        }
                    }
                    previous = Some(key.frame);
                }
            }
        }
        Ok(())
    }

    /// The track for an object, if it has one.
    pub fn track(&self, object_id: ObjectId) -> Option<&ObjectTrack> {
        self.tracks
            .iter()
            .find(|track| track.object_id == object_id)
    }

    /// The track for an object, if it has one (mutable).
    pub fn track_mut(&mut self, object_id: ObjectId) -> Option<&mut ObjectTrack> {
        self.tracks
            .iter_mut()
            .find(|track| track.object_id == object_id)
    }

    /// Remove an object's track entirely (used when its channels are cleared).
    pub fn remove_track(&mut self, object_id: ObjectId) {
        self.tracks.retain(|track| track.object_id != object_id);
    }

    /// Clamp a requested frame into the timeline's valid range.
    pub fn clamp_frame(&self, frame: f32) -> f32 {
        if !frame.is_finite() {
            return 0.0;
        }
        frame.clamp(0.0, self.length as f32)
    }
}

impl Keyframe {
    /// Channel-specific range check: the camera FOV must stay a usable angle,
    /// everything else is bounded only by finiteness (validated separately).
    fn channel_out_of_range(self, channel: Channel) -> bool {
        channel == Channel::FovDeg && !(1.0..179.0).contains(&self.value)
    }
}

/// Set an object's track channels wholesale: an empty list removes the track,
/// otherwise the track is created or its channels replaced. This is the
/// mutation `Command::SetChannelKeys` applies (after validating a probe copy).
pub fn set_track_channels(
    timeline: &mut Timeline,
    object_id: ObjectId,
    channels: Vec<ChannelKeys>,
) {
    if channels.is_empty() {
        timeline.remove_track(object_id);
        return;
    }
    if let Some(track) = timeline.track_mut(object_id) {
        track.channels = channels;
        return;
    }
    timeline.tracks.push(ObjectTrack {
        object_id,
        channels,
    });
}

/// Sample a channel's keyframes at `frame`.
///
/// Before the first key and after the last key the endpoint value is held
/// (a channel is fully defined everywhere, not just between its keys).
/// Returns `None` only when there are no keys at all — `Timeline::validate`
/// rejects that state, but evaluation is defensive about it.
pub fn sample_keys(keys: &[Keyframe], frame: f32) -> Option<f32> {
    let first = keys.first()?;
    if frame <= first.frame as f32 {
        return Some(first.value);
    }
    let last = keys.last()?;
    if frame >= last.frame as f32 {
        return Some(last.value);
    }
    // Find the segment [left, right] containing `frame`. Keys are sorted and
    // strictly increasing, so a linear scan is fine for realistic key counts.
    let mut left = first;
    for key in keys.iter().skip(1) {
        if (key.frame as f32) > frame {
            let span = (key.frame - left.frame) as f32;
            let t = if span <= 0.0 {
                0.0
            } else {
                (frame - left.frame as f32) / span
            };
            let t = match left.interpolation {
                Interpolation::Linear => t,
                Interpolation::Constant => 0.0,
                Interpolation::Smooth => t * t * (3.0 - 2.0 * t),
            };
            return Some(left.value + (key.value - left.value) * t);
        }
        left = key;
    }
    Some(last.value)
}

/// Evaluate the timeline at `frame` **into the project's object state**.
///
/// This is the single evaluation path: the session's playhead and the shell's
/// frame-sequence export both call it, so what you scrub is what renders.
/// Channels without keyframes and objects without tracks are left alone.
/// Returns the clamped frame that was actually applied.
pub fn evaluate(project: &mut Project, frame: f32) -> f32 {
    let frame = project.timeline.clamp_frame(frame);
    // Collect the work first: tracks borrow the timeline, objects mutably
    // borrow the scene, and both hang off `project`.
    let sampled: Vec<(ObjectId, Vec<(Channel, f32)>)> = project
        .timeline
        .tracks
        .iter()
        .map(|track| {
            let values = track
                .channels
                .iter()
                .filter_map(|keys| {
                    sample_keys(&keys.keys, frame).map(|value| (keys.channel, value))
                })
                .collect();
            (track.object_id, values)
        })
        .collect();

    for (object_id, values) in sampled {
        let Some(object) = project.scene.object_mut(object_id) else {
            continue; // dangling track: kept for undo, skipped here
        };
        for (channel, value) in values {
            apply_channel(object, channel, value);
        }
    }
    project.timeline.current_frame = frame;
    frame
}

/// Write one sampled channel value onto an object.
///
/// Terrain is never animated (its transform is georeferenced and locked by
/// `Command::SetTransform`), so a stray track on a terrain is ignored here
/// the same way the command layer refuses to create one.
fn apply_channel(object: &mut SceneObject, channel: Channel, value: f32) {
    if object.is_terrain() {
        return;
    }
    match channel {
        Channel::TranslationX => object.transform.translation.x = value,
        Channel::TranslationY => object.transform.translation.y = value,
        Channel::TranslationZ => object.transform.translation.z = value,
        Channel::RotationX => object.transform.rotation_deg.x = value,
        Channel::RotationY => object.transform.rotation_deg.y = value,
        Channel::RotationZ => object.transform.rotation_deg.z = value,
        Channel::ScaleX => object.transform.scale.x = value,
        Channel::ScaleY => object.transform.scale.y = value,
        Channel::ScaleZ => object.transform.scale.z = value,
        Channel::FovDeg => {
            if let ObjectKind::Camera(camera) = &mut object.kind {
                camera.fov_deg = value;
            }
        }
    }
}

/// Convenience: read a channel's current value from an object (used by the
/// shell when it builds "key the current value" payloads, and by tests).
pub fn channel_value(object: &SceneObject, channel: Channel) -> Option<f32> {
    let transform = &object.transform;
    let value = match channel {
        Channel::TranslationX => transform.translation.x,
        Channel::TranslationY => transform.translation.y,
        Channel::TranslationZ => transform.translation.z,
        Channel::RotationX => transform.rotation_deg.x,
        Channel::RotationY => transform.rotation_deg.y,
        Channel::RotationZ => transform.rotation_deg.z,
        Channel::ScaleX => transform.scale.x,
        Channel::ScaleY => transform.scale.y,
        Channel::ScaleZ => transform.scale.z,
        Channel::FovDeg => match &object.kind {
            ObjectKind::Camera(camera) => camera.fov_deg,
            _ => return None,
        },
    };
    Some(value)
}

/// Build a one-key channel from an object's current value (the "Key" button).
pub fn key_channel(object: &SceneObject, channel: Channel, frame: u32) -> Option<ChannelKeys> {
    let value = channel_value(object, channel)?;
    Some(ChannelKeys {
        channel,
        keys: vec![Keyframe {
            frame,
            value,
            interpolation: Interpolation::default(),
        }],
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::Transform;
    use crate::project::UNTITLED;
    use crate::scene::{CameraObject, SceneObject};

    fn key(frame: u32, value: f32, interpolation: Interpolation) -> Keyframe {
        Keyframe {
            frame,
            value,
            interpolation,
        }
    }

    #[test]
    fn sample_holds_endpoints_outside_the_keyed_range() {
        let keys = [
            key(10, 2.0, Interpolation::Linear),
            key(20, 8.0, Interpolation::Linear),
        ];
        assert_eq!(sample_keys(&keys, 0.0), Some(2.0));
        assert_eq!(sample_keys(&keys, 10.0), Some(2.0));
        assert_eq!(sample_keys(&keys, 999.0), Some(8.0));
        assert_eq!(sample_keys(&[], 5.0), None);
    }

    #[test]
    fn sample_interpolates_each_mode() {
        let linear = [
            key(0, 0.0, Interpolation::Linear),
            key(10, 10.0, Interpolation::Linear),
        ];
        assert_eq!(sample_keys(&linear, 5.0), Some(5.0));

        // Constant holds the left key until the right one is reached.
        let constant = [
            key(0, 0.0, Interpolation::Constant),
            key(10, 10.0, Interpolation::Linear),
        ];
        assert_eq!(sample_keys(&constant, 9.99), Some(0.0));
        assert_eq!(sample_keys(&constant, 10.0), Some(10.0));

        // Smooth is a smoothstep: 0.5 at the midpoint (same as linear there),
        // but slower than linear near the ends (t=0.25 → 0.15625, not 0.25).
        let smooth = [
            key(0, 0.0, Interpolation::Smooth),
            key(10, 10.0, Interpolation::Linear),
        ];
        assert_eq!(sample_keys(&smooth, 5.0), Some(5.0));
        let at_quarter = sample_keys(&smooth, 2.5).unwrap();
        assert!((at_quarter - 1.5625).abs() < 1e-6, "got {at_quarter}");
    }

    #[test]
    fn evaluate_writes_keyed_channels_and_leaves_the_rest_alone() {
        let mut project = Project::new(UNTITLED);
        let object_id = project.scene.objects[0].id;
        let original = project.scene.objects[0].transform;

        project.timeline.tracks.push(ObjectTrack {
            object_id,
            channels: vec![ChannelKeys {
                channel: Channel::TranslationX,
                keys: vec![
                    key(0, 0.0, Interpolation::Linear),
                    key(10, 100.0, Interpolation::Linear),
                ],
            }],
        });

        let applied = evaluate(&mut project, 5.0);
        assert_eq!(applied, 5.0);
        let object = project.scene.object(object_id).unwrap();
        assert_eq!(object.transform.translation.x, 50.0);
        // Unkeyed channels keep their static values.
        assert_eq!(object.transform.translation.y, original.translation.y);
        assert_eq!(object.transform.scale, original.scale);
        assert_eq!(project.timeline.current_frame, 5.0);
    }

    #[test]
    fn evaluate_skips_dangling_tracks_and_clamps_frames() {
        let mut project = Project::new(UNTITLED);
        project.timeline.tracks.push(ObjectTrack {
            object_id: crate::ids::new_id(), // no such object
            channels: vec![ChannelKeys {
                channel: Channel::TranslationX,
                keys: vec![key(0, 7.0, Interpolation::Linear)],
            }],
        });
        // Must not panic, and must clamp into range.
        assert_eq!(
            evaluate(&mut project, 10_000.0),
            project.timeline.length as f32
        );
        assert_eq!(evaluate(&mut project, -3.0), 0.0);
        assert_eq!(evaluate(&mut project, f32::NAN), 0.0);
    }

    #[test]
    fn fov_keys_only_apply_to_cameras() {
        let mut project = Project::new(UNTITLED);
        let camera_id = project
            .scene
            .objects
            .iter()
            .find(|object| matches!(object.kind, ObjectKind::Camera(_)))
            .unwrap()
            .id;
        project.timeline.tracks.push(ObjectTrack {
            object_id: camera_id,
            channels: vec![ChannelKeys {
                channel: Channel::FovDeg,
                keys: vec![
                    key(0, 30.0, Interpolation::Linear),
                    key(10, 90.0, Interpolation::Linear),
                ],
            }],
        });
        evaluate(&mut project, 10.0);
        let camera = project.scene.object(camera_id).unwrap();
        assert_eq!(channel_value(camera, Channel::FovDeg), Some(90.0));
    }

    #[test]
    fn timeline_validation_catches_corrupt_documents() {
        let mut timeline = Timeline::default();
        assert!(timeline.validate().is_ok());

        timeline.frame_rate = 0;
        assert!(timeline.validate().unwrap_err().contains("frame rate"));
        timeline = Timeline::default();

        timeline.length = 0;
        assert!(timeline.validate().unwrap_err().contains("length"));
        timeline = Timeline::default();

        timeline.tracks.push(ObjectTrack {
            object_id: crate::ids::new_id(),
            channels: vec![ChannelKeys {
                channel: Channel::TranslationX,
                keys: vec![
                    key(5, 1.0, Interpolation::Linear),
                    key(5, 2.0, Interpolation::Linear),
                ],
            }],
        });
        assert!(timeline
            .validate()
            .unwrap_err()
            .contains("strictly increasing"));
        timeline.tracks[0].channels[0].keys.pop();
        timeline.tracks[0].channels[0].keys[0].value = f32::NAN;
        assert!(timeline.validate().unwrap_err().contains("finite"));

        // FOV keys must stay a usable camera angle.
        timeline.tracks[0].channels[0].keys[0].value = 200.0;
        timeline.tracks[0].channels[0].channel = Channel::FovDeg;
        assert!(timeline.validate().unwrap_err().contains("valid range"));
    }

    #[test]
    fn key_channel_captures_the_current_value() {
        let object = SceneObject {
            id: crate::ids::new_id(),
            name: "Cam".to_owned(),
            parent: None,
            transform: Transform::default(),
            visible: true,
            locked: false,
            kind: ObjectKind::Camera(CameraObject {
                fov_deg: 55.0,
                near: 0.1,
                far: 1000.0,
            }),
        };
        let keys = key_channel(&object, Channel::FovDeg, 12).unwrap();
        assert_eq!(keys.keys[0].frame, 12);
        assert_eq!(keys.keys[0].value, 55.0);
        assert!(key_channel(&object, Channel::TranslationX, 0).is_some());
    }
}
