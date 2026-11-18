/**
 * Regression tests for the cartography overlay math.
 *
 * Runs with zero dependencies:
 *   npm test   (node --experimental-strip-types)
 */
import {
  anchorWorld,
  clipToLattice,
  drapeLift,
  drapePolylines,
  sampleHeight,
} from "./cartography.ts";

let failures = 0;

function near(
  name: string,
  actual: number,
  expected: number,
  tolerance = 1e-6,
): void {
  if (!(Math.abs(actual - expected) <= tolerance)) {
    failures += 1;
    console.error(`FAIL ${name}: got ${actual}, want ${expected} ±${tolerance}`);
  }
}

function ok(name: string, condition: boolean): void {
  if (!condition) {
    failures += 1;
    console.error(`FAIL ${name}`);
  }
}

// A 4×4 preview, row 0 at the north edge, ramping southward 100 m per row.
const preview = {
  resolution: 4,
  heights: [
    900, 800, 700, 600,
    900, 800, 700, 600,
    900, 800, 700, 600,
    900, 800, 700, 600,
  ],
  minElevation: 600,
  maxElevation: 900,
};

// The grid's four corners, read through the sampler the way the mesh does.
near("north-west corner", sampleHeight(preview, 0, 0), 900);
near("north-east corner", sampleHeight(preview, 1, 0), 600);
near("south-west corner", sampleHeight(preview, 0, 1), 900);
near("south-east corner", sampleHeight(preview, 1, 1), 600);
near("centre", sampleHeight(preview, 0.5, 0.5), 750);
// Clamping keeps a stray coordinate from reading outside the grid.
near("clamped u", sampleHeight(preview, 1.4, 0), 600);
near("clamped v", sampleHeight(preview, 0, -2), 900);

// --- drapeLift scales with the relief in viewer units -------------------
// A 4 km footprint on the 12-unit frame at zScale 1, 300 m relief:
// horizontal = 12/4000, relief_world = 300 * 1 * 12/4000 = 0.9.
// lift = 0.9 * 0.008 = 0.0072 -> the floor wins.
near("lift floor", drapeLift([4000, 4000], [100, 400], 1), 0.02);
// 5000 m relief -> relief_world = 15, lift = 0.12.
near("lift scales", drapeLift([4000, 4000], [0, 5000], 1), 0.12);

// --- drapePolylines ------------------------------------------------------
// One diagonal line, two endpoints, from (u,v)=(0,0) to (1,1).
// horizontal = 12 / 100 = 0.12; the corners are exactly on the grid.
const { positions, pairs } = drapePolylines(
  [[[0, 0], [1, 1]]],
  preview,
  [100, 100],
  1,
  0.02,
);
ok("one line -> two vertices", positions.length === 6);
ok("one line -> one segment", pairs.length === 2 && pairs[0] === 0 && pairs[1] === 1);
// u=0,v=0: elevation 300 above min, x = (0-0.5)*12 = -6, z = (0-0.5)*12 = -6.
near("west vertex x", positions[0], -6);
near("vertex y = relief + lift", positions[1], 300 * 1 * 0.12 + 0.02);
near("vertex z", positions[2], -6);
// u=1,v=1: x = +6, z = +6, elevation 0.
near("east vertex x", positions[3], 6);
near("east vertex y", positions[4], 0.02);
near("east vertex z", positions[5], 6);
// Disjoint lines get distinct index ranges — no stray connective segments.
const two = drapePolylines(
  [[[0, 0], [1, 1]], [[0.5, 0], [0.5, 1]]],
  preview,
  [100, 100],
  1,
  0.02,
);
ok("two lines -> four vertices", two.positions.length === 12);
ok(
  "two lines -> two disjoint segments",
  two.pairs.length === 4 && two.pairs[0] === 0 && two.pairs[1] === 1 && two.pairs[2] === 2 && two.pairs[3] === 3,
);

// --- anchorWorld ---------------------------------------------------------
const anchor = anchorWorld(preview, [100, 100], 1, 0, 0);
near("anchor x", anchor[0], -6);
near("anchor z", anchor[2], -6);

if (failures > 0) {
  throw new Error(`cartography.test: ${failures} assertion(s) failed`);
}
console.log("cartography.test: all checks passed");

import { expandPairs } from "./cartography.ts";
{
  const preview = {
    resolution: 2,
    heights: [100, 200, 300, 400],
    minElevation: 100,
    maxElevation: 400,
  };
  const { positions, pairs } = drapePolylines([[[0, 0], [1, 1]]], preview, [100, 100], 1, 0.02);
  const flat = expandPairs(positions, pairs);
  ok("expandPairs: six floats per segment", flat.length === 6);
  near("expandPairs: segment starts at vertex 0", flat[0], positions[0]);
  near("expandPairs: segment ends at vertex 1", flat[3], positions[3]);
}

// --- clipToLattice -------------------------------------------------------
ok("clip: fully inside is untouched", JSON.stringify(clipToLattice([0.1, 0.1], [0.9, 0.9])) === JSON.stringify([[0.1, 0.1], [0.9, 0.9]]));
ok("clip: fully outside (same side) is dropped", clipToLattice([1.2, 1.2], [2, 2]) === null);
ok("clip: fully outside (same x-side) is dropped", clipToLattice([1.2, 0.5], [2, 0.6]) === null);
{
  const clipped = clipToLattice([-0.5, 0.5], [0.5, 0.5]);
  ok("clip: crossing the west edge lands on u=0", clipped !== null && Math.abs(clipped[0][0]) < 1e-12 && Math.abs(clipped[0][1] - 0.5) < 1e-12);
  ok("clip: the inside end survives", clipped !== null && Math.abs(clipped[1][0] - 0.5) < 1e-12);
}
{
  const clipped = clipToLattice([0.5, -0.4], [0.5, 0.8]);
  ok("clip: crossing the bottom edge lands on v=0", clipped !== null && Math.abs(clipped[0][1]) < 1e-12);
}
// A line crossing the far side still clips to the opposite edge.
{
  const clipped = clipToLattice([0.5, 0.2], [0.5, 1.6]);
  ok("clip: crossing the top edge lands on v=1", clipped !== null && Math.abs(clipped[1][1] - 1) < 1e-12);
}
// Drape honours the clip: one segment crossing the west edge contributes one
// segment; a segment fully outside contributes none.
{
  const { pairs } = drapePolylines([[[1.2, 0.5], [0.5, 0.5]]], preview, [100, 100], 1, 0.02);
  ok("drape: crossing segment is clipped, not dropped", pairs.length === 2);
  const out = drapePolylines([[[1.2, 1.2], [2, 2]]], preview, [100, 100], 1, 0.02);
  ok("drape: segment outside the footprint vanishes", out.pairs.length === 0 && out.positions.length === 0);
}
