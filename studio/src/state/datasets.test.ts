/**
 * The registry picker filters the engine's dataset list by kind.
 *
 * A wrong filter would put a point cloud where a DEM belongs (or offer a
 * raster as a vector layer), so the mapping is pinned here: the registry's
 * kinds are the engine's, and each import site takes exactly one of them.
 */
import { datasetEntries, DATASET_KINDS } from "./datasets.ts";
import type { DatasetInfo } from "../bridge/types";

const entry = (name: string, kind: string): DatasetInfo => ({
  name,
  kind,
  filename: `${name}.tif`,
  description: null,
  path: undefined,
});

const registry: DatasetInfo[] = [
  entry("fuji", "dem"),
  entry("swiss", "dem"),
  entry("luxembourg-rail", "vector"),
  entry("mount-fuji-places", "vector"),
  entry("swiss-land-cover", "overlay"),
  entry("mt-st-helens", "copc"),
  entry("mini_dem", "bundled"),
];

const names = (kind: "dem" | "vector" | "overlay") =>
  datasetEntries(registry, kind).map((item) => item.name);

if (names("dem").join(",") !== "fuji,swiss") {
  throw new Error(`FAIL dem entries: got ${names("dem").join(",")}`);
}
if (names("vector").join(",") !== "luxembourg-rail,mount-fuji-places") {
  throw new Error(`FAIL vector entries: got ${names("vector").join(",")}`);
}
if (names("overlay").join(",") !== "swiss-land-cover") {
  throw new Error(`FAIL overlay entries: got ${names("overlay").join(",")}`);
}
// A point cloud is not a DEM: the point-cloud kind is the engine's own and
// has no import site yet, so it must not leak into one.
if (datasetEntries(registry, "dem").some((item) => item.kind === "copc")) {
  throw new Error("FAIL a point cloud leaked into the DEM list");
}
// An empty registry yields an empty list, not a crash.
if (datasetEntries(null, "dem").length !== 0) {
  throw new Error("FAIL a missing registry must yield no entries");
}
if (datasetEntries(undefined, "vector").length !== 0) {
  throw new Error("FAIL an undefined registry must yield no entries");
}
// Every import site has a label.
for (const kind of ["dem", "vector", "overlay"] as const) {
  if (!DATASET_KINDS[kind]) throw new Error(`FAIL ${kind} has no label`);
}

console.log("datasets: all assertions passed");
