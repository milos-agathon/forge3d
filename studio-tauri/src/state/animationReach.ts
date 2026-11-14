/**
 * Which timeline channels can change a rendered frame (the "still"), and
 * which only move the live preview.
 *
 * The export evaluates the timeline in studio-core and then builds the
 * renderer's request from the evaluated project — nothing else reaches the
 * engine. Verified against the handoff path (`workspace::handoff`):
 *
 *  - The camera is consumed by `workspace::camera_render_view`, which reads
 *    `transform.translation` (the eye), `transform.forward()` (the rotation)
 *    and `camera.fov_deg` — never the camera's scale.
 *  - The sun is consumed by `workspace::sun_angles`, which reads the
 *    directional light's `forward()` for the azimuth/elevation and its
 *    intensity/colour — the light's position and scale never reach it; other
 *    light kinds are invisible to the renderer entirely.
 *  - Meshes are not in the recipe at all.
 *
 * Keep this in sync with `crates/studio-core/src/workspace.rs`; the shell
 * test in `timeline evaluate → handoff sun` pins the Rust half.
 */
import type { Channel, SceneObject } from "../bridge/types";

export type ChannelReach = "still" | "preview";

const CAMERA_REACH: ReadonlySet<Channel> = new Set([
  "translationX",
  "translationY",
  "translationZ",
  "rotationX",
  "rotationY",
  "rotationZ",
  "fovDeg",
]);

const SUN_REACH: ReadonlySet<Channel> = new Set([
  "rotationX",
  "rotationY",
  "rotationZ",
]);

/** Does keying `channel` on `object` change the exported frames? */
export function channelReach(object: SceneObject, channel: Channel): ChannelReach {
  switch (object.kind.type) {
    case "camera":
      // Scale rides the transform for ortho helpers; the renderer ignores it.
      return CAMERA_REACH.has(channel) ? "still" : "preview";
    case "light":
      if (object.kind.kind === "directional" && SUN_REACH.has(channel)) return "still";
      return "preview";
    default:
      return "preview";
  }
}

/** The channels on `object` that change an exported frame. */
export function reachedChannels(object: SceneObject): Channel[] {
  const all: Channel[] = [
    "translationX",
    "translationY",
    "translationZ",
    "rotationX",
    "rotationY",
    "rotationZ",
    "scaleX",
    "scaleY",
    "scaleZ",
    "fovDeg",
  ];
  return all.filter((channel) => channelReach(object, channel) === "still");
}

/** The hint the timeline shows next to the keying controls. */
export function reachHint(object: SceneObject): string {
  switch (object.kind.type) {
    case "camera":
      return "Camera keys reach the export — translation, rotation, FOV. Scale keys are preview-only.";
    case "light":
      if (object.kind.kind === "directional") {
        return "Sun rotation keys reach the export as the sun's azimuth and elevation; translation and scale keys are preview-only (the renderer reads the light's direction only).";
      }
      return "Only the directional sun reaches the export; this light's keys are preview-only.";
    case "mesh":
      return "A mesh does not reach the exported frames yet — its keys are preview-only.";
    default:
      return "This object's keys are preview-only; no frame channel consumes them.";
  }
}
