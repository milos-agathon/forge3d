/**
 * The visibility bake is the still's ray march in grid space; pin its rules
 * on synthetic geometry where the answer is known.
 */
import { bakeSunVisibility } from "./sunVisibility.ts";

let failures = 0;

function ok(name: string, condition: boolean): void {
  if (!condition) {
    failures += 1;
    console.error(`FAIL ${name}`);
  }
}

const VERTICAL_SUN: [number, number, number] = [0, 1, 0];
const LOW_SUN_SOUTH: [number, number, number] = [0, 0.35, 0.94];
const HIGH_SUN_SOUTH: [number, number, number] = [0, 0.85, 0.53];

// A flat 8x8 plate: everything is lit by every sun above the horizon.
{
  const heights = new Array(64).fill(100);
  const out = bakeSunVisibility({
    heights, resolution: 8, minElevation: 100, spanXz: [1000, 1000], zScale: 1, toward: LOW_SUN_SOUTH, steps: 32,
  });
  ok("flat plane: all lit", out.every((v) => v === 255));
}

// An east-west wall (row 3 high), sun low from the south (+z): shadow falls north —
// rows 0..2 get shadowed, rows 3..7 stay lit.
{
  const heights = [
    0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,
    400,400,400,400,400,400,400,400,
    0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,
  ];
  const out = bakeSunVisibility({
    heights, resolution: 8, minElevation: 0, spanXz: [800, 800], zScale: 1, toward: LOW_SUN_SOUTH, steps: 64,
  });
  ok("wall: the north side is in shadow", out[1 * 8] === 0 && out[2 * 8 + 7] === 0);
  ok("wall: the wall itself and the south face stay lit", out[3 * 8] === 255 && out[7 * 8] === 255);
}

// Same wall, high sun: the shadow shrinks — row 1 clears, low sun keeps it dark.
{
  const heights = [
    0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,
    400,400,400,400,400,400,400,400,
    0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,
  ];
  const out = bakeSunVisibility({
    heights, resolution: 8, minElevation: 0, spanXz: [800, 800], zScale: 1, toward: HIGH_SUN_SOUTH, steps: 64,
  });
  ok("wall: high sun clears row 0", out[0 * 8] === 255);
}

// A narrow spike: its own top lit, its far side dark, the far side of everything else lit.
{
  const heights = [
    0,0,0,0,0,0,0,0,
    0,0,0,800,0,0,0,0,
    0,0,600,600,0,0,0,0,
    0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,
  ];
  const out = bakeSunVisibility({
    heights, resolution: 8, minElevation: 0, spanXz: [800, 800], zScale: 1, toward: LOW_SUN_SOUTH, steps: 64,
  });
  ok("spike: its south face is lit", out[3 * 8 + 2] === 255 || out[4 * 8 + 1] === 255);
  ok("spike: due-north is blocked by the spike", out[0 * 8 + 3] === 0);
  ok("spike: its own south face is lit", out[1 * 8 + 3] === 255);
  ok("spike: rows south of it stay lit", out[7 * 8] === 255);
}

// Zenith sun: nothing occludes anything, even the wall case.
{
  const heights = [
    0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,
    400,400,400,400,400,400,400,400,
    0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,
  ];
  const out = bakeSunVisibility({
    heights, resolution: 8, minElevation: 0, spanXz: [800, 800], zScale: 1, toward: VERTICAL_SUN, steps: 64,
  });
  ok("zenith: all lit", out.every((v) => v === 255));
}

if (failures > 0) {
  throw new Error(`sunVisibility: ${failures} assertion(s) failed`);
}
console.log("sunVisibility: all assertions passed");
