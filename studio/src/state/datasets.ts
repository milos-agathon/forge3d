import type { DatasetInfo } from "../bridge/types";

/**
 * The engine's dataset registry, as the import sites see it.
 *
 * The registry is the engine's own (`forge3d.datasets`): a name, a kind and
 * a file. Resolving a name to a local path is the shell's `dataset_path`
 * command — the engine fetches it (hash-verified) and caches it under
 * `~/.forge3d/datasets`; nothing is downloaded until the user picks one.
 */
export type DatasetKind = "dem" | "vector" | "overlay";

/** The registry entries of one kind, in the registry's own order. */
export function datasetEntries(
  datasets: DatasetInfo[] | null | undefined,
  kind: DatasetKind,
): DatasetInfo[] {
  return (datasets ?? []).filter((entry) => entry.kind === kind);
}

/**
 * The kinds the import sites can open, and what each one is for.
 *
 * `overlay` is the engine's name for a raster drape (the Swiss land cover);
 * `vector` covers the two GeoPackage vector datasets; `dem` the heightmaps.
 */
export const DATASET_KINDS: Record<DatasetKind, string> = {
  dem: "Terrain DEM",
  vector: "Vector layer",
  overlay: "Raster overlay",
};
