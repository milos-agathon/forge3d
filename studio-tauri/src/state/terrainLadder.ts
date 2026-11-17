/**
 * The preview ladder's rungs for a remote DEM.
 *
 * The engine's COG reader picks the coarsest overview that still carries the
 * requested resolution (`remote_cog_preview`). The viewport only has to choose
 * a resolution from how close the camera is: a far orbit sees the whole DEM
 * and needs the overview; a close orbit sees a fraction of it and needs the
 * finer level. The bytes fetched grow with the rung, so crossing a boundary
 * is a deliberate, testable decision rather than a hidden read.
 */

export const TERRAIN_RUNGS = [96, 160, 256] as const;

/**
 * Rung for an orbit distance in metres over a DEM of `span` metres.
 *
 * `radius >= span` sees the whole footprint -> 96.
 * `span/2 <= radius < span` -> 160.
 * Closer than half the span -> 256.
 * A missing/invalid span or distance falls back to the coarse rung, so a
 * broken measurement cannot silently download the finest level.
 */
export function previewResolutionForDistance(
  radiusMetres: number | null | undefined,
  spanMetres: number | null | undefined,
): number {
  if (
    !Number.isFinite(radiusMetres) ||
    !Number.isFinite(spanMetres) ||
    (radiusMetres as number) <= 0 ||
    (spanMetres as number) <= 0
  ) {
    return TERRAIN_RUNGS[0];
  }
  const radius = radiusMetres as number;
  const span = spanMetres as number;
  if (radius >= span) return TERRAIN_RUNGS[0];
  if (radius >= span / 2) return TERRAIN_RUNGS[1];
  return TERRAIN_RUNGS[2];
}

/** True when an asset path is a remote URL rather than a local DEM file. */
export function isRemoteDemPath(path: string | null | undefined): boolean {
  return typeof path === "string" && /^(https?:|file:)\/\//i.test(path);
}
