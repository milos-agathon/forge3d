/**
 * Pins the preview-only / still-reaching channel matrix the timeline shows.
 *
 * The matrix is derived from studio-core's handoff (see animationReach.ts);
 * the Rust test in studio-core pins the evaluate→handoff path for the sun.
 */
import {
  channelReach,
  reachedChannels,
  reachHint,
  type ChannelReach,
} from "./animationReach.ts";
import type { Channel, SceneObject } from "../bridge/types";

let failures = 0;
function ok(name: string, condition: boolean): void {
  if (!condition) {
    failures += 1;
    console.error(`FAIL ${name}`);
  }
}

const ALL_CHANNELS: Channel[] = [
  "translationX", "translationY", "translationZ",
  "rotationX", "rotationY", "rotationZ",
  "scaleX", "scaleY", "scaleZ", "fovDeg",
];

const base = {
  id: "o1",
  parent: null,
  name: "Object",
  visible: true,
  locked: false,
  transform: {
    translation: { x: 0, y: 0, z: 0 },
    rotationDeg: { x: 0, y: 0, z: 0 },
    scale: { x: 1, y: 1, z: 1 },
  },
};

const camera: SceneObject = {
  ...base,
  kind: { type: "camera", fovDeg: 45, near: 0.1, far: 500 },
};

const sun: SceneObject = {
  ...base,
  kind: { type: "light", kind: "directional", color: [1, 1, 1], intensity: 3 },
};

const pointLight: SceneObject = {
  ...base,
  kind: { type: "light", kind: "point", color: [1, 1, 1], intensity: 1 },
};

const mesh: SceneObject = {
  ...base,
  kind: {
    type: "mesh",
    source: { type: "primitive", primitive: { type: "cube", size: 1 } },
    material: {
      baseColor: [1, 1, 1],
      roughness: 0.6,
      metallic: 0,
      texture: null,
    },
  },
};

// --- camera: translation + rotation + fov reach; scale does not --------
const expectStill = (object: SceneObject, still: Channel[], preview: Channel[]) => {
  for (const channel of ALL_CHANNELS) {
    const got: ChannelReach = channelReach(object, channel);
    const want: ChannelReach = still.includes(channel) ? "still" : "preview";
    ok(`${object.kind.type}:${channel}`, got === want);
  }
  ok(`${object.kind.type} reachedChannels length`, reachedChannels(object).length === still.length);
  void preview;
};

expectStill(
  camera,
  ["translationX", "translationY", "translationZ", "rotationX", "rotationY", "rotationZ", "fovDeg"],
  ["scaleX", "scaleY", "scaleZ"],
);

// --- directional light: rotation reaches (azimuth/elevation) -----------
expectStill(
  sun,
  ["rotationX", "rotationY", "rotationZ"],
  ["translationX", "translationY", "translationZ", "scaleX", "scaleY", "scaleZ", "fovDeg"],
);

// --- point light: nothing reaches (the renderer consumes the sun) ------
expectStill(pointLight, [], ALL_CHANNELS);

// --- mesh: nothing reaches ----------------------------------------------
expectStill(mesh, [], ALL_CHANNELS);

// --- the hint text names the right things ------------------------------
ok("camera hint mentions FOV", reachHint(camera).includes("FOV"));
ok("sun hint mentions azimuth", reachHint(sun).includes("azimuth"));
ok("mesh hint mentions preview-only", reachHint(mesh).includes("preview-only"));

if (failures > 0) {
  throw new Error(`animationReach.test: ${failures} assertion(s) failed`);
}
console.log("animationReach: all assertions passed");
