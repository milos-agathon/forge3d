import { useCallback, useEffect, useRef, useState } from "react";
import { open as openDialog } from "@tauri-apps/plugin-dialog";

import type {
  Channel,
  ChannelKeys,
  Interpolation,
  Keyframe,
  Project,
  SceneObject,
  Transform,
} from "../bridge/types";
import { useSession } from "../state/session-context";
import { viewStore } from "../viewport/viewStore";
import { NumberField } from "./fields";
import { frameStepMs, nextPlayheadFrame, tickClock } from "../state/playback";
import { reachHint } from "../state/animationReach";

/** Channels the Key button writes for the selected object. */
function keyableChannels(object: SceneObject): Channel[] {
  const transform: Channel[] = [
    "translationX",
    "translationY",
    "translationZ",
    "rotationX",
    "rotationY",
    "rotationZ",
    "scaleX",
    "scaleY",
    "scaleZ",
  ];
  return object.kind.type === "camera" ? [...transform, "fovDeg"] : transform;
}

/** The object's current value for a channel (what "Key" captures). */
function currentValue(object: SceneObject, channel: Channel): number | null {
  const t = object.transform;
  switch (channel) {
    case "translationX":
      return t.translation.x;
    case "translationY":
      return t.translation.y;
    case "translationZ":
      return t.translation.z;
    case "rotationX":
      return t.rotationDeg.x;
    case "rotationY":
      return t.rotationDeg.y;
    case "rotationZ":
      return t.rotationDeg.z;
    case "scaleX":
      return t.scale.x;
    case "scaleY":
      return t.scale.y;
    case "scaleZ":
      return t.scale.z;
    case "fovDeg":
      return object.kind.type === "camera" ? object.kind.fovDeg : null;
  }
}

/** The camera a path is recorded onto: the selection if it is one, else the
 *  active camera, else the first camera in the scene. */
function recordTarget(project: Project): SceneObject | null {
  const selected =
    project.selection.length === 1
      ? project.scene.objects.find((object) => object.id === project.selection[0])
      : undefined;
  if (selected?.kind.type === "camera") return selected;
  const active = project.scene.objects.find(
    (object) => object.id === project.scene.activeCamera,
  );
  if (active?.kind.type === "camera") return active;
  return project.scene.objects.find((object) => object.kind.type === "camera") ?? null;
}

/** Frames between sampled path keys. Four frames is 6 keys/s at 24 fps. */
const RECORD_KEY_INTERVAL = 4;

/**
 * Write one sampled pose into a take's key lists.
 *
 * A path is six channels — the camera's position and its orientation. The
 * orientation is the camera's own quaternion (the studio transform's forward is
 * -Z, the axis a three.js camera looks down), so a replayed frame faces where
 * the take faced.
 */
function putPose(
  keys: Map<Channel, Keyframe[]>,
  frame: number,
  transform: Transform,
  interpolation: Interpolation,
): void {
  const put = (channel: Channel, value: number) => {
    const list = keys.get(channel) ?? [];
    list.push({ frame, value, interpolation });
    keys.set(channel, list);
  };
  put("translationX", transform.translation.x);
  put("translationY", transform.translation.y);
  put("translationZ", transform.translation.z);
  put("rotationX", transform.rotationDeg.x);
  put("rotationY", transform.rotationDeg.y);
  put("rotationZ", transform.rotationDeg.z);
}

/**
 * The animation timeline: playback, scrubbing, keyframing, camera-path
 * recording and frame-sequence export.
 *
 * Playback and export share one evaluation function in studio-core — this
 * component only ever *asks* for a frame (`setPlayhead`) or asks the shell to
 * render one (`exportFrames`), so what you scrub is what renders.
 *
 * Recording reads the viewport's live camera (`viewStore.cameraTransform`) and
 * commits the whole path as one `setChannelKeys` command, so a take is a single
 * undo step instead of one per key.
 */
export function Timeline() {
  const { state, actions, exporting, flyMode, setFlyMode } = useSession();
  const [playing, setPlaying] = useState(false);
  const [recording, setRecording] = useState(false);
  const [interpolation, setInterpolation] = useState<Interpolation>("linear");

  const project = state?.project;
  const timeline = project?.timeline;
  const selection = project?.selection ?? [];
  const selected =
    project && selection.length === 1
      ? project.scene.objects.find((object) => object.id === selection[0])
      : undefined;

  const frame = timeline?.currentFrame ?? 0;
  const length = timeline?.length ?? 96;
  const frameRate = timeline?.frameRate ?? 24;

  const playheadRef = useRef(frame);
  playheadRef.current = frame;
  const actionsRef = useRef(actions);
  actionsRef.current = actions;
  const stateRef = useRef(state);
  stateRef.current = state;
  const settingsRef = useRef({ frameRate, length });
  settingsRef.current = { frameRate, length };
  /** Sampled keys for the take in progress, keyed by channel. */
  const recordedRef = useRef(new Map<Channel, ChannelKeys["keys"]>());
  const recordCameraRef = useRef<string | null>(null);
  /** A scrub during playback restarts the loop's counter from the new value. */
  const resync = useRef<number | null>(null);

  const stopRecording = useCallback(() => {
    setRecording(false);
    setFlyMode(false);
    const cameraId = recordCameraRef.current;
    const recorded = recordedRef.current;
    recordCameraRef.current = null;
    if (!cameraId || recorded.size === 0) return;
    const current = stateRef.current?.project;
    if (!current) return;
    const existing =
      current.timeline.tracks.find((track) => track.objectId === cameraId)?.channels ?? [];
    const channels: ChannelKeys[] = [...recorded.entries()].map(([channel, keys]) => ({
      channel,
      keys: [...keys].sort((a, b) => a.frame - b.frame),
    }));
    // Channels the take did not touch keep their keys.
    for (const entry of existing) {
      if (!recorded.has(entry.channel)) channels.push(entry);
    }
    void actionsRef.current.apply({ type: "setChannelKeys", id: cameraId, channels });
  }, [setFlyMode]);

  const startRecording = useCallback(
    (camera: SceneObject | null) => {
    const current = stateRef.current?.project;
    if (!current || !camera || camera.kind.type !== "camera") {
      return;
    }
    recordedRef.current = new Map();
    recordCameraRef.current = camera.id;
    setPlaying(false);
    setFlyMode(true);
    setRecording(true);
    // Select the camera: the timeline draws the keys of the selected object,
    // so a take that nobody can see is a take nobody can fix.
    void actionsRef.current.select([camera.id]);
    // The pose the take starts from is a key too.
    const transform = viewStore.cameraTransform;
    if (transform) {
      putPose(
        recordedRef.current,
        Math.round(current.timeline.currentFrame),
        transform,
        interpolation,
      );
    }
    },
    [setFlyMode, interpolation],
  );

  // Playback: advance one frame per 1/frameRate seconds, wrapping at the end.
  // Each step is a setPlayhead IPC — the same evaluation the export uses.
  //
  // The effect depends on `playing`/`recording` ONLY. `timeline` and `actions`
  // get a new identity on every snapshot, so depending on them restarted the
  // loop (and reset its clock) on every frame the playhead produced: the
  // accumulator never reached one frame's worth of time and playback froze. The
  // clock arithmetic lives in `state/playback.ts`, pinned by its own test.
  useEffect(() => {
    if (!playing && !recording) return;
    let handle = 0;
    let last = performance.now();
    let rest = 0;
    let inFlight = false;
    let next = resync.current ?? Math.round(playheadRef.current);
    resync.current = null;
    const tick = (now: number) => {
      handle = requestAnimationFrame(tick);
      const clock = tickClock(rest, now - last, frameStepMs(settingsRef.current.frameRate));
      last = now;
      rest = clock.rest;
      // One setPlayhead in flight at a time: the shell answers out of order
      // otherwise, and a fast machine could stack frames on a slow one.
      if (!clock.due || inFlight) return;
      if (resync.current !== null) {
        next = resync.current;
        resync.current = null;
      }
      next = nextPlayheadFrame(next, settingsRef.current.length);
      inFlight = true;
      void actionsRef.current.setPlayhead(next).finally(() => {
        inFlight = false;
        if (!recording) return;
        // Sample the viewport's live camera, not the evaluated object: flying
        // is what the take is made of.
        const transform = viewStore.cameraTransform;
        const cameraId = recordCameraRef.current;
        if (transform && cameraId && next % RECORD_KEY_INTERVAL === 0) {
          putPose(recordedRef.current, next, transform, interpolation);
        }
        // The take fills the timeline: stop instead of wrapping and re-recording.
        if (next >= settingsRef.current.length) stopRecording();
      });
    };
    handle = requestAnimationFrame(tick);
    return () => {
      cancelAnimationFrame(handle);
      resync.current = null;
    };
  }, [playing, recording, stopRecording]);

  // Stop playback and recording when the project closes.
  useEffect(() => {
    if (!state) {
      setPlaying(false);
      stopRecording();
    }
  }, [state, stopRecording]);

  const scrub = useCallback(
    (next: number) => {
      resync.current = next;
      void actions.setPlayhead(next);
    },
    [actions],
  );

  const commitConfig = useCallback(
    (patch: { frameRate?: number; length?: number }) => {
      void actions.apply({
        type: "setTimelineConfig",
        frameRate: patch.frameRate ?? frameRate,
        length: patch.length ?? length,
      });
    },
    [actions, frameRate, length],
  );

  /** Insert keys at the playhead from the object's current values. */
  const keySelected = useCallback(() => {
    if (!selected || !timeline) return;
    const frameIndex = Math.round(frame);
    const existing =
      timeline.tracks.find((track) => track.objectId === selected.id)?.channels ?? [];
    const merged: ChannelKeys[] = keyableChannels(selected).flatMap((channel) => {
      const value = currentValue(selected, channel);
      if (value === null) return [];
      const keys = (existing.find((entry) => entry.channel === channel)?.keys ?? []).filter(
        (key) => key.frame !== frameIndex,
      );
      keys.push({ frame: frameIndex, value, interpolation });
      keys.sort((a, b) => a.frame - b.frame);
      return [{ channel, keys }];
    });
    void actions.apply({ type: "setChannelKeys", id: selected.id, channels: merged });
  }, [selected, timeline, frame, interpolation, actions]);

  const clearSelected = useCallback(() => {
    if (!selected) return;
    void actions.apply({ type: "setChannelKeys", id: selected.id, channels: [] });
  }, [selected, actions]);

  /** Pick a folder and hand it to the session's export loop. */
  const pickAndExport = useCallback(async () => {
    const directory = await openDialog({ directory: true, multiple: false });
    if (typeof directory !== "string") return;
    await actions.exportFrames(directory);
  }, [actions]);

  if (!state || !timeline) return null;

  const track = selected
    ? timeline.tracks.find((entry) => entry.objectId === selected.id)
    : undefined;
  // Union of key frames across the selected object's channels, for the dots.
  const keyFrames = [
    ...new Set((track?.channels ?? []).flatMap((entry) => entry.keys.map((key) => key.frame))),
  ].sort((a, b) => a - b);
  const busy = exporting !== null || recording;
  const camera = recordTarget(state.project);
  const recordingTitle = camera
    ? `Record the viewport camera onto "${camera.name}" while you fly (WASD, drag to look)`
    : "Recording a path needs a camera object";

  return (
    <div className="timeline">
      <div className="timeline__transport">
        <button
          type="button"
          className="timeline__play"
          aria-label={playing ? "Stop playback" : "Start playback"}
          disabled={busy}
          onClick={() => setPlaying((value) => !value)}
        >
          {playing ? "■" : "▶"}
        </button>
        <span className="timeline__frame mono">
          {Math.round(frame)} / {length}
        </span>
        <button
          type="button"
          className={`button button--record${recording ? " is-recording" : ""}`}
          disabled={exporting !== null || !camera}
          title={recordingTitle}
          onClick={() => (recording ? stopRecording() : startRecording(camera))}
        >
          {recording ? "Stop take" : "Record path"}
        </button>
      </div>

      <input
        className="timeline__scrub"
        type="range"
        min={0}
        max={length}
        step={1}
        value={Math.round(frame)}
        disabled={busy || recording}
        onChange={(event) => scrub(Number(event.target.value))}
        aria-label="Playhead"
      />

      {keyFrames.length > 0 && (
        <div className="timeline__keys" aria-hidden="true">
          {keyFrames.map((keyFrame) => (
            <span
              key={keyFrame}
              className="timeline__key-dot"
              style={{ left: `${length === 0 ? 0 : (keyFrame / length) * 100}%` }}
              title={`key @ ${keyFrame}`}
            />
          ))}
        </div>
      )}

      <div className="timeline__config">
        <NumberField
          label="FPS"
          value={timeline.frameRate}
          step={1}
          min={1}
          max={240}
          onCommit={(frameRate) => commitConfig({ frameRate })}
        />
        <NumberField
          label="Frames"
          value={timeline.length}
          step={1}
          min={1}
          max={100_000}
          onCommit={(length) => commitConfig({ length })}
        />
        <label className="field">
          <span className="field-label">Interpolation</span>
          <select
            className="input input--select"
            value={interpolation}
            onChange={(event) => setInterpolation(event.target.value as Interpolation)}
          >
            <option value="linear">Linear</option>
            <option value="constant">Constant</option>
            <option value="smooth">Smooth</option>
          </select>
        </label>
      </div>

      <div className="timeline__actions">
        <button
          type="button"
          className="button"
          disabled={!selected || selected.kind.type === "terrain" || busy}
          title={
            selected
              ? `Key ${selected.name} at frame ${Math.round(frame)} with its current values`
              : "Select one object to key"
          }
          onClick={keySelected}
        >
          Key selected
        </button>
        <button
          type="button"
          className="button"
          disabled={!track || busy}
          onClick={clearSelected}
        >
          Clear keys
        </button>
        <button
          type="button"
          className="button button--primary"
          disabled={busy}
          title={`Render frames 0..${length} at ${state.project.render.width}×${state.project.render.height} into a chosen folder`}
          onClick={() => void pickAndExport()}
        >
          {exporting ? `Exporting ${exporting.done}/${exporting.total}…` : "Export frames…"}
        </button>
        {exporting !== null && (
          <button type="button" className="button" onClick={actions.cancelExport}>
            Stop
          </button>
        )}
      </div>
      {selected && selected.kind.type !== "terrain" && (
        <p className="hint hint--warn">{reachHint(selected)}</p>
      )}

      {!camera && (
        <p className="hint hint--warn">
          Recording a path needs a camera object — add one from the viewport Add menu.
        </p>
      )}
      <p className="hint">
        {recording
          ? `Recording onto "${camera?.name}" — fly with W A S D and Q E, drag to look. The take fills the timeline.`
          : flyMode
            ? "Fly mode is on: W A S D to move, Q E for down and up, drag to look."
            : "Playback and Export evaluate the same timeline in studio-core, so the sequence matches what you scrub. Keying captures the selected object's current values at the playhead; terrain cannot be animated (its transform is georeferenced)."}
      </p>
    </div>
  );
}
