/**
 * Regression tests for the Solid-view → renderer camera mapping.
 *
 * Runs with zero dependencies:
 *   npm test   (node --experimental-strip-types)
 *
 * The golden vector below is the starter scene: camera (11, 7.5, 13) aiming
 * at (0, 0.8, 0) over the Bryce DEM (4000 × 3800 m, relief 454.35 m, zScale
 * 1). Its values were cross-checked against the engine when the camera-parity
 * fix landed — if they drift, Rendered stops matching what Solid shows.
 */
import { TERRAIN_VIEW_SIZE, computeRenderView } from "./viewMath.ts";

let failures = 0;

function near(name: string, actual: number, expected: number, tolerance = 0.05): void {
  if (!(Math.abs(actual - expected) <= tolerance)) {
    failures += 1;
    console.error(`FAIL ${name}: got ${actual}, want ${expected} ±${tolerance}`);
  }
}

function isNull(name: string, value: unknown): void {
  if (value !== null) {
    failures += 1;
    console.error(`FAIL ${name}: expected null, got ${JSON.stringify(value)}`);
  }
}

// --- golden vector: the starter scene -------------------------------
const starter = computeRenderView({
  camera: [11, 7.5, 13],
  aim: [0, 0.8, 0],
  fovDeg: 45,
  spanXz: [4000, 3800],
  elevationRange: [2099.23, 2553.58],
  zScale: 1,
});
if (!starter) {
  throw new Error("FAIL starter: view must exist");
}
near("starter phi (azimuth east→north)", starter.phiDeg, 310.21);
near("starter theta (polar from up)", starter.thetaDeg, 68.52);
near("starter radius in metres", starter.radius, 6100.0, 0.5);
near("starter target east", starter.target[0], 0);
near("starter target north", starter.target[1], 0);
// aim at 0.8 viewport units above the base, minus half the (centred) relief.
near("starter target up", starter.target[2], 39.49);
near("starter fov passthrough", starter.fovDeg, 45, 1e-6);

// --- axis conventions on a 1200 m square DEM (scale: 100 units → metres) ---
const square = {
  fovDeg: 40,
  spanXz: [1200, 1200] as [number, number],
  elevationRange: [0, 100] as [number, number],
  zScale: 1,
};

const east = computeRenderView({ ...square, camera: [10, 5, 0], aim: [0, 5, 0] });
if (!east) throw new Error("FAIL east: view must exist");
near("camera east of aim → phi 0", east.phiDeg, 0, 1e-6);
near("level camera → theta 90", east.thetaDeg, 90, 1e-6);
near("radius = 10 viewport units in metres", east.radius, 1000, 1e-6);
// the renderer centres relief on zero: target up = aim − halfRelief
near("centred relief datum", east.target[2], 500 - 50, 1e-6);

const north = computeRenderView({ ...square, camera: [0, 5, -10], aim: [0, 5, 0] });
if (!north) throw new Error("FAIL north: view must exist");
// the Solid +Z axis is south: a camera at −Z stands NORTH of the aim
near("camera north of aim → phi 90", north.phiDeg, 90, 1e-6);

const top = computeRenderView({ ...square, camera: [0, 20, 0], aim: [0, 5, 0] });
if (!top) throw new Error("FAIL top: view must exist");
near("camera above aim → theta 0", top.thetaDeg, 0, 1e-6);

const stretched = computeRenderView({ ...square, camera: [10, 5, 0], aim: [0, 5, 0], zScale: 2 });
if (!stretched) throw new Error("FAIL stretched: view must exist");
near("zScale doubles the datum shift", stretched.target[2], 500 - 100, 1e-6);

const noRange = computeRenderView({
  ...square,
  camera: [10, 5, 0],
  aim: [0, 5, 0],
  elevationRange: null,
});
if (!noRange) throw new Error("FAIL noRange: view must exist");
near("missing elevation range → no datum shift", noRange.target[2], 500, 1e-6);

// --- guards --------------------------------------------------------
const localFallback = computeRenderView({
  ...square,
  spanXz: null,
  dimensions: [1024, 1024],
  camera: [1, 0, 0],
  aim: [0, 0, 0],
});
if (!localFallback) throw new Error("FAIL localFallback: view must exist");
near(
  "un-georeferenced DEM falls back to pixel dimensions",
  localFallback.radius,
  1024 / TERRAIN_VIEW_SIZE,
  1e-6,
);

isNull("camera on the aim → no view", computeRenderView({ ...square, camera: [0, 5, 0], aim: [0, 5, 0] }));
isNull("no DEM footprint → no view", computeRenderView({ ...square, spanXz: null, camera: [10, 5, 0], aim: [0, 5, 0] }));
isNull("zero-span DEM → no view", computeRenderView({ ...square, spanXz: [0, 0], camera: [10, 5, 0], aim: [0, 5, 0] }));

if (failures > 0) {
  throw new Error(`${failures} viewMath assertion(s) failed`);
}
console.log(`viewMath: all assertions passed (viewport unit = ${TERRAIN_VIEW_SIZE})`);
