/**
 * Cartography overlays for the Solid/Material viewport: the geometry the
 * still draws, in the same 0..1 terrain lattice the terrain preview uses,
 * lifted onto the mesh so the lines sit on the ground.
 *
 * Pure math, no three.js — the meshes are assembled by the caller, the tests
 * pin the numbers.
 */
import { TERRAIN_VIEW_SIZE } from "./viewMath.ts";

export interface HeightPreview {
  resolution: number;
  /** Row-major heights; row 0 is the north edge (the raster's top). */
  heights: number[];
  minElevation: number;
  maxElevation: number;
}

/**
 * Bilinear sample of the preview's height grid at a terrain UV.
 * u runs west→east, v runs north→south — the same lattice the vector
 * preview normalises onto.
 */
export function sampleHeight(preview: HeightPreview, u: number, v: number): number {
  const res = preview.resolution;
  const gx = Math.min(Math.max(u, 0), 1) * (res - 1);
  const gy = Math.min(Math.max(v, 0), 1) * (res - 1);
  const x0 = Math.floor(gx);
  const y0 = Math.floor(gy);
  const x1 = Math.min(x0 + 1, res - 1);
  const y1 = Math.min(y0 + 1, res - 1);
  const tx = gx - x0;
  const ty = gy - y0;
  const at = (col: number, row: number): number => {
    const value = preview.heights[row * res + col];
    return Number.isFinite(value) ? value : preview.minElevation;
  };
  const south = at(x0, y0) * (1 - tx) + at(x1, y0) * tx;
  const north = at(x0, y1) * (1 - tx) + at(x1, y1) * tx;
  return south * (1 - ty) + north * ty;
}

/**
 * The world-space lift a drape needs to clear the mesh: one percent of the
 * relief in viewer units, with a floor so a nearly flat DEM still clears
 * z-fighting on the surface.
 */
export function drapeLift(
  spanXz: [number, number],
  elevationRange: [number, number],
  zScale: number,
): number {
  const horizontal = TERRAIN_VIEW_SIZE / Math.max(spanXz[0], spanXz[1]);
  const relief = (elevationRange[1] - elevationRange[0]) * zScale * horizontal;
  return Math.max(0.02, relief * 0.008);
}

/**
 * The world position of a terrain UV, on the mesh surface plus `lift`.
 * Mirrors `buildTerrainMesh`: the grid spans the unit plane, scaled to the
 * footprint, with elevation relative to the preview's minimum.
 */
export function worldPosition(
  preview: HeightPreview,
  spanXz: [number, number],
  zScale: number,
  u: number,
  v: number,
  lift: number,
): [number, number, number] {
  const horizontal = TERRAIN_VIEW_SIZE / Math.max(spanXz[0], spanXz[1]);
  const elevation = sampleHeight(preview, u, v) - preview.minElevation;
  return [
    (u - 0.5) * spanXz[0] * horizontal,
    elevation * zScale * horizontal + lift,
    (v - 0.5) * spanXz[1] * horizontal,
  ];
}

/** Cohen–Sutherland outcodes for the 0..1 terrain lattice. */
const LEFT = 1;
const RIGHT = 2;
const BOTTOM = 4;
const TOP = 8;

function outcode(x: number, y: number): number {
  let code = 0;
  if (x < 0) code |= LEFT;
  else if (x > 1) code |= RIGHT;
  if (y < 0) code |= BOTTOM;
  else if (y > 1) code |= TOP;
  return code;
}

/**
 * Clip one segment to the unit square (the DEM's footprint). A segment that
 * ends outside the terrain — a rail clipped by a national framing, say — is
 * cut at the frame, exactly like the still's off-frame segments vanish.
 * Returns the clipped endpoints, or null when fully outside.
 */
export function clipToLattice(
  a: [number, number],
  b: [number, number],
): [[number, number], [number, number]] | null {
  let [x0, y0] = a;
  let [x1, y1] = b;
  let code0 = outcode(x0, y0);
  let code1 = outcode(x1, y1);
  for (;;) {
    if ((code0 | code1) === 0) return [[x0, y0], [x1, y1]];
    if ((code0 & code1) !== 0) return null;
    const code = code0 !== 0 ? code0 : code1;
    let x = 0;
    let y = 0;
    if (code & TOP) {
      x = x0 + ((x1 - x0) * (1 - y0)) / (y1 - y0);
      y = 1;
    } else if (code & BOTTOM) {
      x = x0 + ((x1 - x0) * (0 - y0)) / (y1 - y0);
      y = 0;
    } else if (code & RIGHT) {
      y = y0 + ((y1 - y0) * (1 - x0)) / (x1 - x0);
      x = 1;
    } else {
      y = y0 + ((y1 - y0) * (0 - x0)) / (x1 - x0);
      x = 0;
    }
    if (code === code0) {
      x0 = x;
      y0 = y;
      code0 = outcode(x0, y0);
    } else {
      x1 = x;
      y1 = y;
      code1 = outcode(x1, y1);
    }
  }
}

/**
 * Drape a batch of polylines on the terrain and flatten them into a segment
 * soup (one `LineSegments2` draw call per layer).
 *
 * Segments that leave the DEM's footprint are clipped to it (the still's
 * depth test does the same); a polyline with no visible stretch emits
 * nothing. Returns flat vertex positions (xyz per vertex) and `pairs`, the
 * indices of each segment.
 */
export function drapePolylines(
  lines: number[][][],
  preview: HeightPreview,
  spanXz: [number, number],
  zScale: number,
  lift: number,
): { positions: Float32Array; pairs: Uint32Array } {
  const vertices: number[] = [];
  const segments: number[] = [];
  for (const line of lines) {
    for (let index = 0; index + 1 < line.length; index += 1) {
      const a = line[index];
      const b = line[index + 1];
      if (a.length < 2 || b.length < 2) continue;
      const clipped = clipToLattice([a[0], a[1]], [b[0], b[1]]);
      if (!clipped) continue;
      const base = vertices.length / 3;
      for (const [u, v] of clipped) {
        const [x, y, z] = worldPosition(preview, spanXz, zScale, u, v, lift);
        vertices.push(x, y, z);
      }
      segments.push(base, base + 1);
    }
  }
  return {
    positions: new Float32Array(vertices),
    pairs: new Uint32Array(segments),
  };
}

/** A grid annotation anchored on the ground: its world position, ready to project. */
export function anchorWorld(
  preview: HeightPreview,
  spanXz: [number, number],
  zScale: number,
  u: number,
  v: number,
): [number, number, number] {
  // A small lift so the label's ray does not clip the mesh at the anchor.
  return worldPosition(preview, spanXz, zScale, u, v, 0.02);
}

/**
 * Expand the `pairs` index list to the flat endpoint-position layout
 * `LineSegmentsGeometry.setPositions` expects (six floats per segment).
 */
export function expandPairs(positions: Float32Array, pairs: Uint32Array): Float32Array {
  const out = new Float32Array(pairs.length * 3);
  for (let index = 0; index < pairs.length; index += 1) {
    const vertex = pairs[index] * 3;
    out[index * 3] = positions[vertex];
    out[index * 3 + 1] = positions[vertex + 1];
    out[index * 3 + 2] = positions[vertex + 2];
  }
  return out;
}
