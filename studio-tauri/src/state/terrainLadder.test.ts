/** The preview ladder must be driven by the camera, not by a hidden full fetch. */
import { isRemoteDemPath, previewResolutionForDistance, TERRAIN_RUNGS } from "./terrainLadder.ts";

const span = 24_000;
const rung = (radius: number) => previewResolutionForDistance(radius, span);

if (rung(30_000) !== 96 || rung(24_000) !== 96) throw new Error("far orbit must use the overview");
if (rung(18_000) !== 160) throw new Error("mid orbit must refine one rung");
if (rung(12_000) !== 160) throw new Error("the boundary at span/2 is inclusive");
if (rung(8_000) !== 256) throw new Error("close orbit must use the finest rung");
if (TERRAIN_RUNGS.join(",") !== "96,160,256") throw new Error("rungs changed without a reason");

// A broken measurement must never silently trigger the most expensive read.
if (previewResolutionForDistance(null, span) !== 96) throw new Error("null radius");
if (previewResolutionForDistance(-1, span) !== 96) throw new Error("negative radius");
if (previewResolutionForDistance(1_000, 0) !== 96) throw new Error("zero span");
if (previewResolutionForDistance(Number.NaN, span) !== 96) throw new Error("NaN radius");

if (!isRemoteDemPath("https://example.com/a.tif")) throw new Error("https URL");
if (!isRemoteDemPath("http://example.com/a.tif")) throw new Error("http URL");
if (!isRemoteDemPath("file:///data/a.tif")) throw new Error("file URL");
if (isRemoteDemPath("/data/a.tif")) throw new Error("a local path is not remote");
if (isRemoteDemPath(null)) throw new Error("null is not remote");

console.log("terrainLadder: all assertions passed");
