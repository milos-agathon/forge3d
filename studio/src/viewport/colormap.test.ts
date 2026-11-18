/**
 * The elevation ramp the Material preview paints on the terrain.
 *
 * The stops are the engine's own `Colormap.rgba` samples (linear RGB), so this
 * check pins the interpolation the preview does on them: ends at the stop
 * colours, teal in the middle of viridis, clamping outside the domain, and
 * strength 0 returning the material colour so the Material/Solid switch has a
 * real off.
 *
 * Runs with zero dependencies: `npm test` (node --experimental-strip-types).
 */
import { CLAY_LINEAR, colormapTitle, linearToCss, rampColor } from "./colormap.ts";

function close(actual: number, expected: number, what: string): void {
  if (Math.abs(actual - expected) > 1e-6) {
    throw new Error(`${what}: expected ${expected}, got ${actual}`);
  }
}

function equal(actual: unknown, expected: unknown, what: string): void {
  if (JSON.stringify(actual) !== JSON.stringify(expected)) {
    throw new Error(`${what}: expected ${JSON.stringify(expected)}, got ${JSON.stringify(actual)}`);
  }
}

// The engine's linear viridis samples (t = 0, 0.5, 1), as `get_colormap`
// returns them: #440154, #21918c, #fde725 decoded to linear.
const stops: [number, number, number][] = [
  [0.0578, 0.0003, 0.0887],
  [0.0154, 0.2856, 0.2614],
  [0.9823, 0.7991, 0.0185],
];

const low = rampColor(stops, 0);
const high = rampColor(stops, 1);
const mid = rampColor(stops, 0.5);

close(low[0], stops[0][0], "t=0 red");
close(low[1], stops[0][1], "t=0 green");
close(low[2], stops[0][2], "t=0 blue");
close(high[0], stops[2][0], "t=1 red");
close(high[1], stops[2][1], "t=1 green");
close(high[2], stops[2][2], "t=1 blue");
close(mid[1], stops[1][1], "t=0.5 is the middle stop");

// The middle of viridis is teal — green above red and blue.
if (!(mid[1] > mid[0] && mid[1] > mid[2])) {
  throw new Error(`mid ramp is not green-dominant: ${mid.join(",")}`);
}

// Out-of-range elevations clamp to the ends instead of wrapping.
equal(rampColor(stops, -2), low, "clamp below");
equal(rampColor(stops, 3), high, "clamp above");

// Strength 0 is the material colour: the preview's own "off".
equal(rampColor(stops, 0.3, 0), CLAY_LINEAR, "strength 0 is the material");
const half = rampColor(stops, 0.3, 0.5);
const full = rampColor(stops, 0.3, 1);
for (let channel = 0; channel < 3; channel += 1) {
  close(
    half[channel],
    (full[channel] + CLAY_LINEAR[channel]) / 2,
    `strength mix channel ${channel}`,
  );
}

// The panel's strip is the display encoding of the same linear colour.
if (linearToCss([0, 0, 0]) !== "rgb(0 0 0)") {
  throw new Error(`black strip: ${linearToCss([0, 0, 0])}`);
}
if (linearToCss([1, 1, 1]) !== "rgb(255 255 255)") {
  throw new Error(`white strip: ${linearToCss([1, 1, 1])}`);
}
const yellow = linearToCss([0.9823, 0.7991, 0.0185]);
if (yellow !== "rgb(253 231 37)") {
  throw new Error(`viridis high stop should encode #fde725, got ${yellow}`);
}

if (colormapTitle("viridis") !== "forge3d:viridis") {
  throw new Error("colormapTitle must name the engine's catalogue entry");
}

console.log("colormap: all assertions passed (engine stops, clamp, strength, display encoding)");
