import type { RenderView } from "../bridge/types";

/**
 * How many scene units the Solid viewport gives a DEM's longest span.
 *
 * The viewport draws a DEM across this many units regardless of its real
 * size, and the renderer works in metres; every conversion between the two
 * uses it (mirrors `SCENE_TERRAIN_SIZE` in studio-core).
 */
export const TERRAIN_VIEW_SIZE = 12;

export interface TerrainViewInput {
  /** Solid-camera position, viewport units (Y up, north at -Z). */
  camera: [number, number, number];
  /** A point on the view axis: the orbit target, or ahead of the camera in
   *  camera view (any point on the axis frames the same picture — the mapping
   *  is a similarity transform). */
  aim: [number, number, number];
  /** Vertical field of view, degrees. */
  fovDeg: number;
  /** DEM footprint in metres, or null when the DEM is not georeferenced. */
  spanXz: [number, number] | null;
  /** DEM pixel dimensions — the span fallback for DEMs without a footprint
   *  (mirrors `TerrainObject::viewer_framing`: pixels as world units). */
  dimensions?: [number, number] | null;
  /** DEM elevation range `[min, max]` in metres. */
  elevationRange: [number, number] | null;
  /** Terrain vertical exaggeration. */
  zScale: number;
}

/**
 * Map the Solid viewport camera onto the Forge3D renderer's frame, so a
 * render reproduces the view on screen instead of a canned framing.
 *
 * Axes: viewport east = +X, north = -Z, up = +Y; the renderer's
 * `mesh:zup:north` frame is +X east, +Y north, +Z up with the terrain
 * centred on the origin. Horizontal and vertical deltas both scale by
 * `maxSpan / TERRAIN_VIEW_SIZE` (the viewport world is a uniform shrink of
 * the metre frame); the vertical datum shifts because the viewport terrain
 * sits on its base while the renderer centres its relief on zero.
 *
 * Returns null when there is no terrain to render against (the caller clears
 * the view store and the shell falls back to its default framing).
 */
export function computeRenderView(input: TerrainViewInput): RenderView | null {
  const { camera, aim, fovDeg, spanXz, dimensions, elevationRange, zScale } = input;
  const span = spanXz ?? dimensions;
  if (!span || !(Math.max(span[0], span[1]) > 0)) return null;
  const scale = Math.max(span[0], span[1]) / TERRAIN_VIEW_SIZE;
  const [low, high] = elevationRange ?? [0, 0];
  const halfRelief = ((high - low) * zScale) / 2;

  const dE = (camera[0] - aim[0]) * scale;
  const dN = -(camera[2] - aim[2]) * scale;
  const dU = (camera[1] - aim[1]) * scale;
  const radius = Math.hypot(dE, dN, dU);
  if (!(radius > 1e-6)) return null;

  const phiDeg = (Math.atan2(dN, dE) * (180 / Math.PI) + 360) % 360;
  const thetaDeg = (Math.acos(Math.min(1, Math.max(-1, dU / radius))) * 180) / Math.PI;
  return {
    phiDeg,
    thetaDeg,
    radius,
    target: [aim[0] * scale, -aim[2] * scale, aim[1] * scale - halfRelief],
    fovDeg,
  };
}
